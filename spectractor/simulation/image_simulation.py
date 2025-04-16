from spectractor import parameters
from spectractor.config import set_logger
from spectractor.tools import (rebin, pixel_rotation, set_wcs_file_name, set_sources_file_name,
                               set_gaia_catalog_file_name, load_wcs_from_file, ensure_dir,
                               plot_image_simple, iraf_source_detection)
from spectractor.extractor.images import Image, find_target
from spectractor.astrometry import get_gaia_coords_after_proper_motion
from spectractor.extractor.background import remove_image_background_sextractor
from spectractor.simulation.simulator import SpectrogramModel
from spectractor.simulation.atmosphere import Atmosphere
from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.psf import PSF

from astropy.io import fits, ascii
import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import copy
import os


class StarModel:
    """Class to model a star in the image simulation process.

    Attributes
    ----------
    x0: float
        X position of the star centroid in pixels.
    y0: float
        Y position of the star centroid in pixels.
    amplitude: amplitude
        The amplitude of the star in image units.
    """

    def __init__(self, centroid_coords, psf, amplitude):
        """Create a StarModel instance.

        The model is based on an Astropy Fittable2DModel. The centroid and amplitude
        parameters of the given model are updated by the dedicated arguments.

        Parameters
        ----------
        centroid_coords: array_like
            Tuple of (x,y) coordinates of the desired star centroid in pixels.
        psf: PSF
            PSF model
        amplitude: float
            The desired amplitude of the star in image units.

        Examples
        --------
        >>> from spectractor.extractor.psf import Moffat
        >>> p = (100, 50, 50, 5, 2, 200)
        >>> psf = Moffat(p)
        >>> s = StarModel((20, 10), psf, 200)
        >>> s.plot_model()
        >>> s.x0
        20
        >>> s.y0
        10
        >>> s.amplitude
        200
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.x0 = centroid_coords[0]
        self.y0 = centroid_coords[1]
        self.amplitude = amplitude
        self.psf = copy.deepcopy(psf)
        self.psf.params.values[1] = self.x0
        self.psf.params.values[2] = self.y0
        self.psf.params.values[0] = self.amplitude
        # to be realistic, usually fitted fwhm is too big, divide gamma by 2
        self.fwhm = self.psf.params.values[3]
        # self.sigma = self.model.stddev / 2

    def plot_model(self):
        """
        Plot the star model.
        """
        x = np.arange(self.x0 - 5 * self.fwhm, self.x0 + 5 * self.fwhm)
        y = np.arange(self.y0 - 5 * self.fwhm, self.y0 + 5 * self.fwhm)
        xx, yy = np.meshgrid(x, y)
        star = self.psf.evaluate(np.array([xx, yy]))
        fig, ax = plt.subplots(1, 1)
        plot_image_simple(ax, star, title=f'Star model: A={self.amplitude:.2f}, fwhm={self.fwhm:.2f}',
                          units='Arbitrary units')
        if parameters.DISPLAY:
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()


class StarFieldModel:

    def __init__(self, base_image, flux_factor=1):
        """
        Examples
        --------

        >>> from spectractor.extractor.images import Image, find_target
        >>> im = Image('tests/data/reduc_20170530_134.fits', target_label="HD111980")
        >>> x0, y0 = find_target(im, guess=(740, 680))
        >>> s = StarFieldModel(im)
        >>> s.plot_model()

        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.image = base_image
        self.target = base_image.target
        self.field = None
        self.stars = []
        self.pixcoords = []
        self.fwhm = base_image.target_star2D.params.values[3]
        self.flux_factor = flux_factor
        self.set_star_list()

    # noinspection PyUnresolvedReferences
    def set_star_list(self):
        x0, y0 = self.image.target_pixcoords
        sources_file_name = set_sources_file_name(self.image.file_name)
        wcs_file_name = set_wcs_file_name(self.image.file_name)
        gaia_catalog_file_name = set_gaia_catalog_file_name(self.image.file_name)
        if os.path.isfile(wcs_file_name) and os.path.isfile(gaia_catalog_file_name):
            # load gaia catalog
            gaia_catalog = ascii.read(gaia_catalog_file_name, format="ecsv")
            gaia_coord_after_motion = get_gaia_coords_after_proper_motion(gaia_catalog, self.image.date_obs)
            # load WCS
            wcs = load_wcs_from_file(wcs_file_name)
            # catalog matching to set star positions using Gaia
            target_coord = wcs.all_pix2world([x0 * parameters.CCD_REBIN], [y0 * parameters.CCD_REBIN], 0)
            target_coord = SkyCoord(ra=target_coord[0] * units.deg, dec=target_coord[1] * units.deg,
                                    frame="icrs", obstime=self.image.date_obs, equinox="J2000")
            gaia_target_index, dist_2d, dist_3d = target_coord.match_to_catalog_sky(gaia_coord_after_motion)
            dx, dy = 0, 0
            for gaia_i in range(len(gaia_catalog)):
                x, y = np.array(wcs.all_world2pix(gaia_coord_after_motion[gaia_i].ra,
                                                  gaia_coord_after_motion[gaia_i].dec, 0)) / parameters.CCD_REBIN
                if gaia_i == gaia_target_index[0]:
                    dx = x0 - x
                    dy = y0 - y
                A = 10 ** (-gaia_catalog['phot_g_mean_mag'][gaia_i] / 2.5)
                self.stars.append(StarModel([x, y], self.image.target_star2D, A))
                self.pixcoords.append([x, y])
            # rescale using target fitted amplitude
            amplitudes = np.array([star.amplitude for star in self.stars])
            target_flux = self.image.target_star2D.params.values[0]
            amplitudes *= target_flux / self.stars[gaia_target_index[0]].amplitude * self.flux_factor
            for k, star in enumerate(self.stars):
                star.amplitude = amplitudes[k]
                # shift x,y star positions according to target position
                star.x0 += dx
                star.y0 += dy
                star.psf.params.values[1] += dx
                star.psf.params.values[2] += dy
                star.psf.params.values[0] = amplitudes[k]
        elif os.path.isfile(sources_file_name):
            # load sources positions and flux
            sources = Table.read(sources_file_name)
            sources['X'].name = "xcentroid"
            sources['Y'].name = "ycentroid"
            sources['FLUX'].name = "flux"
            for k, source in enumerate(sources):
                x, y = np.array([sources['xcentroid'][k], sources['ycentroid'][k]]) / parameters.CCD_REBIN
                A = sources['flux'][k] * self.flux_factor
                self.stars.append(StarModel([x, y], self.image.target_star2D, A))
                self.pixcoords.append([x, y])
        else:
            # try extraction using iraf source detection
            # mask background, faint stars, and saturated pixels
            data = np.copy(self.image.data)
            # mask order0 and spectrum
            margin = 30
            mask = np.zeros(data.shape, dtype=bool)
            for y in range(int(y0) - 100, int(y0) + 100):
                for x in range(self.image.data.shape[1]):
                    u, v = pixel_rotation(x, y, self.image.disperser.theta([x0, y0]) * np.pi / 180., x0, y0)
                    if margin > v > -margin:
                        mask[y, x] = True
            # remove background and detect sources
            data_wo_bkg = remove_image_background_sextractor(data)
            sources = iraf_source_detection(data_wo_bkg, mask=mask)
            for k, source in enumerate(sources):
                x, y = sources['xcentroid'][k], sources['ycentroid'][k]
                A = sources['flux'][k] * self.flux_factor
                self.stars.append(StarModel([x, y], self.image.target_star2D, A))
                self.pixcoords.append([x, y])
        self.pixcoords = np.array(self.pixcoords).T

    def model(self, x, y):
        if self.field is None:
            window = int(20 * self.fwhm)
            self.field = self.stars[0].psf.evaluate(np.array([x, y]))
            for k in range(1, len(self.stars)):
                left = max(0, int(self.pixcoords[0][k]) - window)
                right = min(np.max(x), int(self.pixcoords[0][k]) + window)
                low = max(0, int(self.pixcoords[1][k]) - window)
                up = min(np.max(y), int(self.pixcoords[1][k]) + window)
                if up < low or left > right:
                    continue
                yy, xx = np.mgrid[low:up, left:right]
                self.field[low:up, left:right] += self.stars[k].psf.evaluate(np.array([xx, yy]))
        return self.field

    def plot_model(self):
        fig, ax = plt.subplots(1, 1)
        plot_image_simple(ax, self.field, scale="log10", target_pixcoords=self.pixcoords)
        if parameters.DISPLAY:
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()


class BackgroundModel:
    """Class to model the background of the simulated image.

    Attributes
    ----------
    Nx: int
        Size of the background along X axis in pixels.
    Ny: int
        Size of the background along Y axis in pixels.
    level: float
        The mean level of the background in image units.
    frame: array_like
        (x, y, smooth) right and upper limits in pixels of a vignetting frame,
        and the smoothing gaussian width (default: None).
    """

    def __init__(self, Nx, Ny, level, frame=None):
        """Create a BackgroundModel instance.

        Parameters
        ----------
        Nx: int
            Size of the background along X axis in pixels.
        Ny: int
            Size of the background along Y axis in pixels.
        level: float
            The mean level of the background in image units.
        frame: array_like, None
            (x, y, smooth) right and upper limits in pixels of a vignetting frame,
            and the smoothing gaussian width (default: None).

        Examples
        --------
        >>> from spectractor import parameters
        >>> Nx, Ny = 200, 300
        >>> bgd = BackgroundModel(Nx, Ny, 10)
        >>> model = bgd.model()
        >>> np.all(model==10)
        True
        >>> model.shape
        (200, 200)
        >>> bgd = BackgroundModel(Nx, Ny, 10, frame=(160, 180, 3))
        >>> bgd.plot_model()
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.Nx = Nx
        self.Ny = Ny
        self.level = level
        if self.level <= 0:
            self.my_logger.warning('\n\tBackground level must be strictly positive.')
        else:
            self.my_logger.info(f'\n\tBackground set to {level:.3f} ADU/s.')
        self.frame = frame

    def model(self):
        """Compute the background model for the image simulation in image units.

        A shadowing vignetting frame is roughly simulated if self.frame is set.

        Returns
        -------
        bkgd: array_like
            The array of the background model.

        """
        xx, yy = np.mgrid[0:self.Ny:1, 0:self.Nx:1]
        bkgd = self.level * np.ones_like(xx)
        if self.frame is None:
            return bkgd
        else:
            xlim, ylim, width = self.frame
            bkgd[ylim:, :] = self.level / 100
            bkgd[:, xlim:] = self.level / 100
            kernel = np.outer(gaussian(self.Nx, width), gaussian(self.Ny, width))
            bkgd = fftconvolve(bkgd, kernel, mode='same')
            bkgd *= self.level / bkgd[self.Ny // 2, self.Nx // 2]
            return bkgd

    def plot_model(self):
        """Plot the background model.

        """
        bkgd = self.model()
        fig, ax = plt.subplots(1, 1)
        im = plt.imshow(bkgd, origin='lower', cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Background model')
        cb = plt.colorbar(im, ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7, prune=None)
        cb.update_ticks()
        cb.set_label('Arbitrary units')  # ,fontsize=16)
        if parameters.DISPLAY:
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()


class FlatModel:
    """Class to model the pixel flat of the simulated image. Flat is dimensionless and its average must be one.

    Attributes
    ----------
    Nx: int
        Size of the background along X axis in pixels.
    Ny: int
        Size of the background along Y axis in pixels.
    gains: array_like
        The list of gains to apply. The average must be one.
    randomness_level: float
        Level of random quantum efficiency to apply to pixels (default: 0.).
    """

    def __init__(self, Nx, Ny, gains, randomness_level=0.):
        """Create a FlatModel instance. Flat is dimensionless and its average must be one.

        Parameters
        ----------
        Nx: int
            Size of the background along X axis in pixels.
        Ny: int
            Size of the background along Y axis in pixels.
        gains: array_like
            The list of gains to apply. The average must be one.
        randomness_level: float
            Level of random quantum efficiency to apply to pixels (default: 0.).

        Examples
        --------
        >>> from spectractor import parameters
        >>> Nx, Ny = 200, 300
        >>> flat = FlatModel(Nx, Ny, gains=[[1, 2, 3, 4], [4, 3, 2, 1]])
        >>> model = flat.model()
        >>> print(f"{np.mean(model):.4f}")
        1.0000
        >>> model.shape
        (200, 200)
        >>> flat.plot_model()
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.Nx = Nx
        self.Ny = Ny
        self.gains = np.atleast_2d(gains).astype(float)
        if len(self.gains) <= 0:
            raise ValueError(f"Gains list is empty")
        if np.any(self.gains <= 0):
            raise ValueError(f"One the gain values is negative. Got {self.gains}.")
        if np.mean(self.gains) != 1.:
            self.my_logger.warning(f"\n\tGains list average is not one but {np.mean(self.gains)}. "
                                   "I scaled them to have an average of one.")
            self.gains /= np.mean(self.gains)
        self.my_logger.warning(f'\n\tRelative gains are set to {self.gains}.')
        self.randomness_level = randomness_level

    def model(self):
        """Compute the flat model for the image simulation (no units).

        Returns
        -------
        flat: array_like
            The array of the flat model.
        """
        yy, xx = np.mgrid[0:self.Nx:1, 0:self.Ny:1]
        flat = np.ones_like(xx, dtype=float)
        hflats = np.array_split(flat, self.gains.shape[0])
        for h in range(self.gains.shape[0]):
            vflats = np.array_split(hflats[h].T, self.gains.shape[1])
            for v in range(self.gains.shape[1]):
                vflats[v] *= self.gains[h,v]
            hflats[h] = np.concatenate(vflats).T
        flat = np.concatenate(hflats).T
        if self.randomness_level != 0:
            flat += np.random.uniform(-self.randomness_level, self.randomness_level, size=flat.shape)

        return flat

    def plot_model(self):
        """Plot the flat model.

        """
        flat = self.model()
        fig, ax = plt.subplots(1, 1)
        im = plt.imshow(flat, origin='lower', cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Flat model')
        cb = plt.colorbar(im, ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7, prune=None)
        cb.update_ticks()
        cb.set_label('Dimensionless')  # ,fontsize=16)
        if parameters.DISPLAY:
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()



class ImageModel(Image):

    def __init__(self, filename, target_label=None):
        self.my_logger = set_logger(self.__class__.__name__)
        Image.__init__(self, filename, target_label=target_label)
        self.true_lambdas = None
        self.true_spectrum = None

    def compute(self, star, background, spectrogram, starfield=None, flat=None):
        yy, xx = np.mgrid[0:parameters.CCD_IMSIZE:1, 0:parameters.CCD_IMSIZE:1]
        if starfield is not None:
            starfield_mod = starfield.model(xx, yy)
            self.data = starfield_mod
            self.starfield = np.copy(starfield_mod)
            if parameters.DEBUG:
                self.plot_image(scale='symlog', target_pixcoords=starfield.pixcoords)
                starfield.plot_model()
        else:
            self.data = star.psf.evaluate(np.array([xx, yy]))
        self.data += background.model()
        if spectrogram.full_image:
            self.data[spectrogram.spectrogram_ymin:spectrogram.spectrogram_ymax, :] += spectrogram.spectrogram_data
        else:
            self.data[spectrogram.spectrogram_ymin:spectrogram.spectrogram_ymax,
                      spectrogram.spectrogram_xmin:spectrogram.spectrogram_xmax] += spectrogram.spectrogram_data
        if flat is not None:
            flat_mod = flat.model()
            self.data *= flat_mod
            self.flat = flat_mod

    def add_poisson_and_read_out_noise(self):  # pragma: no cover
        if self.units != 'ADU':
            raise AttributeError('Poisson noise procedure has to be applied on map in ADU units')
        d = np.copy(self.data).astype(float)
        # convert to electron counts
        d *= self.gain
        # Poisson noise
        noisy = np.random.poisson(d).astype(float)
        # Add read-out noise is available
        if self.read_out_noise is not None:
            noisy += np.random.normal(scale=self.read_out_noise)
        # reconvert to ADU
        self.data = noisy / self.gain
        # removes zeros
        min_noz = np.min(self.data[self.data > 0])
        self.data[self.data <= 0] = min_noz

    def save_image(self, output_filename, overwrite=False):
        hdu0 = fits.PrimaryHDU()
        hdu0.data = self.data
        hdu0.header = self.header
        hdu1 = fits.ImageHDU()
        # hdu1.data = [self.true_lambdas, self.true_spectrum]
        hdulist = fits.HDUList([hdu0, hdu1])
        hdulist.writeto(output_filename, overwrite=overwrite)
        self.my_logger.info('\n\tImage saved in %s' % output_filename)

    def load_image(self, filename):
        super(ImageModel, self).load_image(filename)
        # hdu_list = fits.open(filename)
        # self.true_lambdas, self.true_spectrum = hdu_list[1].data


def ImageSim(image_filename, spectrum_filename, outputdir, pwv=5, ozone=300, aerosols=0.03, A1=1, A2=1, A3=1, angstrom_exponent=None,
             psf_poly_params=None, psf_type=None, diffraction_orders=None, with_rotation=True, with_starfield=True, with_adr=True, with_noise=True, with_flat=True):
    """ The basic use of the extractor consists first to define:
    - the path to the fits image from which to extract the image,
    - the path of the output directory to save the extracted spectrum (created automatically if does not exist yet),
    - the rough position of the object in the image,
    - the name of the target (to search for the extra-atmospheric spectrum if available).
    Then different parameters and systematics can be set:
    - pwv: the pressure water vapor (in mm)
    - ozone: the ozone quantity (in XX)
    - aerosols: the vertical aerosol optical depth
    - A1: a global grey absorption parameter for the spectrum
    - A2: the relative amplitude of second order compared with first order
    - with_rotation: rotate the spectrum according to the disperser characteristics (True by default)
    - with_stars: include stars in the image field (True by default)
    - with_adr: include ADR effect (True by default)
    - with_flat: include flat (True by default)
    """
    my_logger = set_logger(__name__)
    my_logger.info(f'\n\tStart IMAGE SIMULATOR')
    # Load reduced image
    spectrum = Spectrum(spectrum_filename)
    parameters.CALLING_CODE = ""
    if diffraction_orders is None:
        diffraction_orders = np.arange(spectrum.order, spectrum.order + 3 * np.sign(spectrum.order), np.sign(spectrum.order))
    image = ImageModel(image_filename, target_label=spectrum.target.label)
    guess = np.array([spectrum.header['TARGETX'], spectrum.header['TARGETY']])
    if parameters.CCD_REBIN != 1:
        # these lines allow to simulate images using rebinned spectrum files
        guess *= parameters.CCD_REBIN
        new_shape = np.asarray((parameters.CCD_IMSIZE, parameters.CCD_IMSIZE))
        old_edge = parameters.CCD_IMSIZE * parameters.CCD_REBIN
        image.gain = rebin(image.gain[:old_edge, :old_edge], new_shape, FLAG_MAKESUM=False)
        image.read_out_noise = rebin(image.read_out_noise[:old_edge, :old_edge], new_shape, FLAG_MAKESUM=False)

    if parameters.DEBUG:
        image.plot_image(scale='symlog', target_pixcoords=guess)
    # Fit the star 2D profile
    my_logger.info('\n\tSearch for the target in the image...')
    target_pixcoords = find_target(image, guess)
    # Background model
    my_logger.info('\n\tBackground model...')
    bgd_level = float(np.mean(spectrum.spectrogram_bgd))
    background = BackgroundModel(parameters.CCD_IMSIZE, parameters.CCD_IMSIZE, level=bgd_level, frame=None)
    if parameters.DEBUG:
        background.plot_model()

    # Target model
    my_logger.info('\n\tStar model...')
    # Spectrogram is simulated with spectrum.x0 target position: must be this position to simulate the target.
    star = StarModel(np.array(image.target_pixcoords) / parameters.CCD_REBIN, image.target_star2D, image.target_star2D.params.values[0])
    # reso = star.fwhm
    if parameters.DEBUG:
        star.plot_model()

    # Star field model
    starfield = None
    if with_starfield:
        my_logger.info('\n\tStar field model...')
        starfield = StarFieldModel(image, flux_factor=1)

    # Flat model
    flat = None
    if with_flat:
        my_logger.info('\n\tFlat model...')
        flat = FlatModel(parameters.CCD_IMSIZE, parameters.CCD_IMSIZE, gains=[[1, 2, 3, 4], [4, 3, 2, 1]], randomness_level=1e-2)
        if parameters.DEBUG:
            flat.plot_model()

    # Spectrum model
    my_logger.info('\n\tSpectrum model...')
    airmass = image.header['AIRMASS']
    pressure = image.header['OUTPRESS']
    temperature = image.header['OUTTEMP']

    # Rotation: useful only to fill the Dy_disp_axis column in PSF table
    if not with_rotation:
        rotation_angle = 0
    else:
        rotation_angle = spectrum.rotation_angle

    # Load PSF
    if psf_type is not None:
        from spectractor.extractor.psf import load_PSF
        parameters.PSF_TYPE = psf_type
        psf = load_PSF(psf_type=psf_type)
        spectrum.psf = psf
        spectrum.chromatic_psf.psf = psf
    if psf_poly_params is None:
        my_logger.info('\n\tUse PSF parameters from _table.csv file.')
        psf_poly_params = spectrum.chromatic_psf.from_table_to_poly_params()
    else:
        spectrum.chromatic_psf.deg = ((len(psf_poly_params) - 1) // (len(spectrum.chromatic_psf.psf.params.labels) - 2) - 1) // len(diffraction_orders)
        spectrum.chromatic_psf.set_polynomial_degrees(spectrum.chromatic_psf.deg)
        if spectrum.chromatic_psf.deg == 0:  # x_c must have deg >= 1
            psf_poly_params.insert(1, 0)
        my_logger.info(f'\n\tUse PSF parameters {psf_poly_params} as polynoms of '
                       f'degree {spectrum.chromatic_psf.degrees}')
    if psf_type is not None and psf_poly_params is not None:
        spectrum.chromatic_psf.init_from_table()

    # Simulate spectrogram
    atmosphere = Atmosphere(airmass, pressure, temperature)
    spectrogram = SpectrogramModel(spectrum, atmosphere=atmosphere, fast_sim=False, full_image=True,
                                   with_adr=with_adr, diffraction_orders=diffraction_orders)
    spectrogram.simulate(A1, A2, A3, aerosols, angstrom_exponent, ozone, pwv,
                         parameters.DISTANCE2CCD, 0, 0, rotation_angle, psf_poly_params)

    # Image model
    my_logger.info('\n\tImage model...')
    image.compute(star, background, spectrogram, starfield=starfield, flat=flat)

    # Recover true spectrum
    spectrogram.set_true_spectrum(spectrogram.lambdas, aerosols, ozone, pwv, shift_t=0)
    true_lambdas = np.copy(spectrogram.true_lambdas)
    true_spectrum = np.copy(spectrogram.true_spectrum)

    # Saturation effects
    saturated_pixels = np.where(spectrogram.spectrogram_data > image.saturation)[0]
    if len(saturated_pixels) > 0:
        my_logger.warning(f"\n\t{len(saturated_pixels)} saturated pixels detected above saturation "
                          f"level at {image.saturation} ADU/s in the spectrogram."
                          f"\n\tSpectrogram maximum is at {np.max(spectrogram.spectrogram_data)} ADU/s.")
    image.data[image.data > image.saturation] = image.saturation

    # Convert data from ADU/s in ADU
    image.convert_to_ADU_units()

    # Add Poisson and read-out noise
    if with_noise:
        image.add_poisson_and_read_out_noise()

    # Round float ADU into closest integers
    # image.data = np.around(image.data)
    if parameters.OBS_NAME == "AUXTEL":
        image.data = image.data.T[::-1, ::-1]

    # Plot
    if parameters.VERBOSE and parameters.DISPLAY:  # pragma: no cover
        image.convert_to_ADU_rate_units()
        image.plot_image(scale="symlog", title="Image simulation", target_pixcoords=target_pixcoords, units=image.units)
        image.convert_to_ADU_units()

    # Set output path
    ensure_dir(outputdir)
    output_filename = image_filename.split('/')[-1]
    output_filename = (output_filename.replace('reduc', 'sim')).replace('trim', 'sim').replace('exposure', 'sim')
    output_filename = os.path.join(outputdir, output_filename)

    # Save images and parameters
    image.header['A1_T'] = A1
    image.header['A2_T'] = A2
    image.header['A3_T'] = A3
    image.header['X0_T'] = spectrum.x0[0]
    image.header['Y0_T'] = spectrum.x0[1]
    image.header['D2CCD_T'] = float(parameters.DISTANCE2CCD)
    image.header['OZONE_T'] = ozone
    image.header['PWV_T'] = pwv
    image.header['VAOD_T'] = aerosols
    image.header['ROT_T'] = rotation_angle
    image.header['ROTATION'] = int(with_rotation)
    image.header['STARS'] = int(with_starfield)
    image.header['BKGD_LEV'] = background.level
    image.header['PSF_DEG'] = spectrogram.chromatic_psf.deg
    image.header['PSF_TYPE'] = parameters.PSF_TYPE
    psf_poly_params_truth = np.array(psf_poly_params)
    if psf_poly_params_truth.size > spectrum.spectrogram_Nx:
        psf_poly_params_truth = psf_poly_params_truth[spectrum.spectrogram_Nx:]
    image.header['LBDAS_T'] = str(np.round(true_lambdas, decimals=2).tolist())
    image.header['AMPLIS_T'] = str(true_spectrum.tolist())
    image.header['PSF_P_T'] = str(psf_poly_params_truth.tolist())
    image.save_image(output_filename, overwrite=True)
    return image


if __name__ == "__main__":
    import doctest

    doctest.testmod()

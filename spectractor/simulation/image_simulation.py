from spectractor import parameters
from spectractor.config import set_logger
from spectractor.tools import (pixel_rotation, detect_peaks, set_wcs_file_name, set_sources_file_name,
                               set_gaia_catalog_file_name, load_wcs_from_file, ensure_dir,
                               plot_image_simple)
from spectractor.extractor.images import Image, find_target
from spectractor.astrometry import get_gaia_coords_after_proper_motion, source_detection
from spectractor.extractor.background import  remove_image_background_sextractor
from spectractor.simulation.throughput import TelescopeTransmission
from spectractor.simulation.simulator import SpectrogramSimulatorCore, SimulatorInit

from astropy.io import fits, ascii
import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.signal import fftconvolve, gaussian
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
    target: Target
        The associated Target instance (default: None).
    """

    def __init__(self, pixcoords, model, amplitude, target=None):
        """Create a StarModel instance.

        The model is based on an Astropy Fittable2DModel. The centroid and amplitude
        parameters of the given model are updated by the dedicated arguments.

        Parameters
        ----------
        pixcoords: array_like
            Tuple of (x,y) coordinates of the desired star centroid in pixels.
        model: Fittable2DModel
            Astropy fittable 2D model
        amplitude: float
            The desired amplitude of the star in image units.
        target: Target
            The associated Target instance (default: None).

        Examples
        --------
        >>> from spectractor.extractor.psf import PSF2D
        >>> from spectractor.extractor.images import fit_PSF2D_minuit
        >>> p = (100, 50, 50, 3, 2, -0.1, 1, 200)
        >>> psf = PSF2D(*p)
        >>> yy, xx = np.mgrid[:100,:50]
        >>> data = psf.evaluate(xx, yy, *p)
        >>> model = fit_PSF2D_minuit(xx, yy, data, guess=p)
        >>> s = StarModel((20, 10), model, 200, target=None)
        >>> s.plot_model()
        >>> s.x0
        20
        >>> s.y0
        10
        >>> s.model.amplitude
        200
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.x0 = pixcoords[0]
        self.y0 = pixcoords[1]
        self.amplitude = amplitude
        self.target = target
        self.model = copy.deepcopy(model)
        self.model.x_mean = self.x0
        self.model.y_mean = self.y0
        self.model.amplitude_moffat = amplitude
        self.my_logger.warning(f"{self.model}")
        # to be realistic, usually fitted fwhm is too big, divide by 2
        self.fwhm = self.model.gamma / 2
        self.sigma = self.model.stddev / 2

    def plot_model(self):
        """
        Plot the star model.
        """
        x = np.linspace(self.x0 - 10 * self.fwhm, self.x0 + 10 * self.fwhm, 50)
        y = np.linspace(self.y0 - 10 * self.fwhm, self.y0 + 10 * self.fwhm, 50)
        xx, yy = np.meshgrid(x, y)
        star = self.model(xx, yy)
        fig, ax = plt.subplots(1, 1)
        im = plt.pcolor(x, y, star, cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Star model: A=%.2f, fwhm=%.2f' % (self.amplitude, self.fwhm))
        cb = plt.colorbar(im, ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7, prune=None)
        cb.update_ticks()
        cb.set_label('Arbitrary units')  # ,fontsize=16)
        if parameters.DISPLAY:
            plt.show()


class StarFieldModel:

    def __init__(self, base_image, flux_factor=1):
        """
        Examples
        --------

        >>> from spectractor.extractor.images import Image, find_target
        >>> im = Image('tests/data/reduc_20170530_134.fits', target_label="HD111980")
        >>> x0, y0 = find_target(im, guess=(740, 680), use_wcs=False)
        >>> s = StarFieldModel(im)
        >>> s.plot_model()

        """
        self.image = base_image
        self.target = base_image.target
        self.field = None
        self.stars = []
        self.pixcoords = []
        self.fwhm = base_image.target_star2D.gamma
        self.flux_factor = flux_factor
        self.set_star_list()

    def set_star_list(self):
        x0, y0 = self.image.target_pixcoords
        sources_file_name = set_sources_file_name(self.image.file_name)
        if os.path.isfile(sources_file_name):
            # load sources positions and flux
            sources = Table.read(sources_file_name)
            sources['X'].name = "xcentroid"
            sources['Y'].name = "ycentroid"
            sources['FLUX'].name = "flux"
            # test presence of WCS and gaia catalog files
            wcs_file_name = set_wcs_file_name(self.image.file_name)
            gaia_catalog_file_name = set_gaia_catalog_file_name(self.image.file_name)
            if os.path.isfile(wcs_file_name) and os.path.isfile(gaia_catalog_file_name):
                # load gaia catalog
                gaia_catalog = ascii.read(gaia_catalog_file_name, format="ecsv")
                gaia_coord_after_motion = get_gaia_coords_after_proper_motion(gaia_catalog, self.image.date_obs)
                # load WCS
                wcs = load_wcs_from_file(wcs_file_name)
                # catalog matching to set star positions using Gaia
                sources_coord = wcs.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)
                sources_coord = SkyCoord(ra=sources_coord[0] * units.deg, dec=sources_coord[1] * units.deg,
                                         frame="icrs", obstime=self.image.date_obs, equinox="J2000")
                gaia_index, dist_2d, dist_3d = sources_coord.match_to_catalog_sky(gaia_coord_after_motion)
                for k, gaia_i in enumerate(gaia_index):
                    x, y = wcs.all_world2pix(gaia_coord_after_motion[gaia_i].ra, gaia_coord_after_motion[gaia_i].dec, 0)
                    A = sources['flux'][k] * self.flux_factor
                    self.image.my_logger.warning(f"\n\t{x} {y} {A}")
                    self.stars.append(StarModel([x, y], self.image.target_star2D, A))
                    self.pixcoords.append([x, y])
            else:
                for k, source in enumerate(sources):
                    x, y = sources['xcentroid'][k], sources['ycentroid'][k]
                    A = sources['flux'][k] * self.flux_factor
                    self.stars.append(StarModel([x, y], self.image.target_star2D, A))
                    self.pixcoords.append([x, y])
        else:
            # mask background, faint stars, and saturated pixels
            data = np.copy(self.image.data)
            # self.saturation = 0.99 * parameters.CCD_MAXADU / base_image.expo
            # self.saturated_pixels = np.where(image_thresholded > self.saturation)
            # image_thresholded[self.saturated_pixels] = 0.
            # image_thresholded -= threshold
            # image_thresholded[np.where(image_thresholded < 0)] = 0.
            # mask order0 and spectrum
            margin = 30
            mask = np.zeros(data.shape, dtype=bool)
            for y in range(int(y0) - 100, int(y0) + 100):
                for x in range(parameters.CCD_IMSIZE):
                    u, v = pixel_rotation(x, y, self.image.disperser.theta([x0, y0]) * np.pi / 180., x0, y0)
                    if margin > v > -margin:
                        mask[y, x] = True
            # remove background and detect sources
            data_wo_bkg = remove_image_background_sextractor(data)
            sources = source_detection(data_wo_bkg, mask=mask)
            for k, source in enumerate(sources):
                x, y = sources['xcentroid'][k], sources['ycentroid'][k]
                A = sources['flux'][k] * self.flux_factor
                self.stars.append(StarModel([x, y], self.image.target_star2D, A))
                self.pixcoords.append([x, y])
        self.pixcoords = np.array(self.pixcoords).T

    def model(self, x, y):
        if self.field is None:
            window = int(20 * self.fwhm)
            self.field = self.stars[0].model(x, y)
            for k in range(1, len(self.stars)):
                left = max(0, int(self.pixcoords[0][k]) - window)
                right = min(parameters.CCD_IMSIZE, int(self.pixcoords[0][k]) + window)
                low = max(0, int(self.pixcoords[1][k]) - window)
                up = min(parameters.CCD_IMSIZE, int(self.pixcoords[1][k]) + window)
                yy, xx = np.mgrid[low:up, left:right]
                self.field[low:up, left:right] += self.stars[k].model(xx, yy)
        return self.field

    def plot_model(self):
        yy, xx = np.mgrid[0:parameters.CCD_IMSIZE:1, 0:parameters.CCD_IMSIZE:1]
        starfield = self.model(xx, yy)
        fig, ax = plt.subplots(1, 1)
        plot_image_simple(ax, starfield, scale="log10", target_pixcoords=self.pixcoords)
        # im = plt.imshow(starfield, origin='lower', cmap='jet')
        # ax.grid(color='white', ls='solid')
        # ax.grid(True)
        # ax.set_xlabel('X [pixels]')
        # ax.set_ylabel('Y [pixels]')
        # ax.set_title(f'Star field model: fwhm={self.fwhm.value:.2f}')
        # cb = plt.colorbar(im, ax=ax)
        # cb.formatter.set_powerlimits((0, 0))
        # cb.locator = MaxNLocator(7, prune=None)
        # cb.update_ticks()
        # cb.set_label('Arbitrary units')  # ,fontsize=16)
        if parameters.DISPLAY:
            plt.show()


class BackgroundModel:
    """Class to model the background of the simulated image.

    The background model size is set with the parameters.CCD_IMSIZE global keyword.

    Attributes
    ----------
    level: float
        The mean level of the background in image units.
    frame: array_like
        (x, y, smooth) right and upper limits in pixels of a vignetting frame,
        and the smoothing gaussian width (default: None).
    """

    def __init__(self, level, frame=None):
        """Create a BackgroundModel instance.

        The background model size is set with the parameters.CCD_IMSIZE global keyword.

        Parameters
        ----------
        level: float
            The mean level of the background in image units.
        frame: array_like, None
            (x, y, smooth) right and upper limits in pixels of a vignetting frame,
            and the smoothing gaussian width (default: None).

        Examples
        --------
        >>> from spectractor import parameters
        >>> parameters.CCD_IMSIZE = 200
        >>> bgd = BackgroundModel(10)
        >>> model = bgd.model()
        >>> np.all(model==10)
        True
        >>> model.shape
        (200, 200)
        >>> bgd = BackgroundModel(10, frame=(160, 180, 3))
        >>> bgd.plot_model()
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.level = level
        if self.level <= 0:
            self.my_logger.warning('\n\tBackground level must be strictly positive.')
        self.frame = frame

    def model(self):
        """Compute the background model for the image simulation in image units.

        A shadowing vignetting frame is roughly simulated if self.frame is set.
        The background model size is set with the parameters.CCD_IMSIZE global keyword.

        Returns
        -------
        bkgd: array_like
            The array of the background model.

        """
        yy, xx = np.mgrid[0:parameters.CCD_IMSIZE:1, 0:parameters.CCD_IMSIZE:1]
        bkgd = self.level * np.ones_like(xx)
        if self.frame is None:
            return bkgd
        else:
            xlim, ylim, width = self.frame
            bkgd[ylim:, :] = self.level / 100
            bkgd[:, xlim:] = self.level / 100
            kernel = np.outer(gaussian(parameters.CCD_IMSIZE, width), gaussian(parameters.CCD_IMSIZE, width))
            bkgd = fftconvolve(bkgd, kernel, mode='same')
            bkgd *= self.level / bkgd[parameters.CCD_IMSIZE // 2, parameters.CCD_IMSIZE // 2]
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


class ImageModel(Image):

    def __init__(self, filename, target_label=None):
        self.my_logger = set_logger(self.__class__.__name__)
        Image.__init__(self, filename, target_label=target_label.label)
        self.true_lambdas = None
        self.true_spectrum = None

    def compute(self, star, background, spectrogram, starfield=None):
        yy, xx = np.mgrid[0:parameters.CCD_IMSIZE:1, 0:parameters.CCD_IMSIZE:1]
        self.data = star.model(xx, yy) + background.model()
        self.data[spectrogram.spectrogram_ymin:spectrogram.spectrogram_ymax,
        spectrogram.spectrogram_xmin:spectrogram.spectrogram_xmax] += spectrogram.data  # - spectrogram.spectrogram_bgd)
        self.true_lambdas = spectrogram.lambdas
        self.true_spectrum = spectrogram.true_spectrum
        if starfield is not None:
            self.data += starfield.model(xx, yy)

    def add_poisson_and_read_out_noise(self):
        if self.units != 'ADU':
            self.my_logger.error('\n\tPoisson noise procedure has to be applied on map in ADU units')
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
        hdu1.data = [self.true_lambdas, self.true_spectrum]
        hdulist = fits.HDUList([hdu0, hdu1])
        hdulist.writeto(output_filename, overwrite=overwrite)
        self.my_logger.info('\n\tImage saved in %s' % output_filename)

    def load_image(self, filename):
        super(ImageModel, self).load_image(filename)
        # hdu_list = fits.open(filename)
        # self.true_lambdas, self.true_spectrum = hdu_list[1].data


def ImageSim(image_filename, spectrum_filename, outputdir, pwv=5, ozone=300, aerosols=0.03, A1=1, A2=0.05,
             psf_poly_params=None,
             with_rotation=True,
             with_stars=True):
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
    """
    my_logger = set_logger(__name__)
    my_logger.info(f'\n\tStart IMAGE SIMULATOR')
    # Load reduced image
    spectrum, telescope, disperser, target = SimulatorInit(spectrum_filename)
    image = ImageModel(image_filename, target_label=target)
    guess = [spectrum.header['TARGETX'], spectrum.header['TARGETY']]
    if parameters.DEBUG:
        image.plot_image(scale='log10', target_pixcoords=guess)
    # Fit the star 2D profile
    my_logger.info('\n\tSearch for the target in the image...')
    target_pixcoords = find_target(image, guess, use_wcs=False)
    # Find the exact target position using WCS if available
    wcs_file_name = set_wcs_file_name(image.file_name)
    if os.path.isfile(wcs_file_name):
        target_pixcoords = find_target(image, guess, use_wcs=True)
    # Background model
    my_logger.info('\n\tBackground model...')
    yy, xx = np.mgrid[:parameters.XWINDOW, :parameters.YWINDOW]
    bgd_level = float(np.mean(image.target_bkgd2D(xx, yy)))
    background = BackgroundModel(level=bgd_level, frame=None)  # frame=(1600, 1650))
    if parameters.DEBUG:
        background.plot_model()

    # Target model
    my_logger.info('\n\tStar model...')
    # Spectrogram is simulated with spectrum.x0 target position: must be this position to simualte the target.
    star = StarModel(image.target_pixcoords, image.target_star2D, image.target_star2D.amplitude_moffat.value,
                     target=image.target)
    reso = star.sigma
    if parameters.DEBUG:
        star.plot_model()
    # Star field model
    starfield = None
    if with_stars:
        my_logger.info('\n\tStar field model...')
        starfield = StarFieldModel(image)
        if parameters.VERBOSE:
            image.plot_image(scale='log10', target_pixcoords=starfield.pixcoords)
            starfield.plot_model()

    # Spectrum model
    my_logger.info('\n\tSpectum model...')
    airmass = image.header['AIRMASS']
    pressure = image.header['OUTPRESS']
    temperature = image.header['OUTTEMP']
    telescope = TelescopeTransmission(image.filter)

    # Load PSF
    if psf_poly_params is None:
        my_logger.info('\n\tUse PSF parameters from _table.csv file.')
        psf_poly_params = spectrum.chromatic_psf.from_table_to_poly_params()
        # TODO: solve this Gaussian PSF part issue
        psf_poly_params[-7:-1] = 0.

    # Increase
    spectrogram = SpectrogramSimulatorCore(spectrum, telescope, disperser, airmass, pressure,
                                           temperature, pwv=pwv, ozone=ozone, aerosols=aerosols, A1=A1, A2=A2,
                                           D=spectrum.disperser.D, shift_x=0., shift_y=0., shift_t=0.,
                                           angle=spectrum.rotation_angle,
                                           psf_poly_params=psf_poly_params, with_background=False, fast_sim=False)

    # now we include effects related to the wrong extraction of the spectrum:
    # wrong estimation of the order 0 position and wrong DISTANCE2CCD
    # distance = spectrum.chromatic_psf.get_distance_along_dispersion_axis()
    # spectrum.disperser.D = parameters.DISTANCE2CCD
    # spectrum.lambdas = spectrum.disperser.grating_pixel_to_lambda(distance, spectrum.x0, order=1)

    # Image model
    my_logger.info('\n\tImage model...')
    image.compute(star, background, spectrogram, starfield=starfield)

    # Convert data from ADU/s in ADU
    image.convert_to_ADU_units()

    # Saturation effects
    image.data[image.data > image.saturation] = image.saturation

    # Add Poisson and read-out noise
    image.add_poisson_and_read_out_noise()

    # Round float ADU into closest integers
    image.data = np.around(image.data)

    # Plot
    if parameters.VERBOSE and parameters.DISPLAY:  # pragma: no cover
        image.plot_image(scale="log", title="Image simulation", target_pixcoords=target_pixcoords, units=image.units)

    # Set output path
    ensure_dir(outputdir)
    output_filename = image_filename.split('/')[-1]
    output_filename = (output_filename.replace('reduc', 'sim')).replace('trim', 'sim')
    output_filename = os.path.join(outputdir, output_filename)

    # # Save truth file
    # txt = f"pwv {pwv}\nozone {ozone}\naerosols{aerosols}\nA1 {A1}\nA2 {A2}\n" \
    #       f"D {spectrum.disperser.D}\nshift_x 0\nshift_y 0\nshift_t 0\nangle {spectrum.rotation_angle}\n"
    # for ip, p in enumerate(spectrum.chromatic_psf.poly_params_labels[spectrogram.spectrogram_Nx:]):
    #     txt += f"{p} {psf_poly_params[ip]}\n"
    # f = open(output_filename.replace('.fits','_truth.txt'), 'w')
    # f.write(txt)
    # f.close()
    # if parameters.VERBOSE:
    #     my_logger.info(f"\t\nWrite truth parameters in {output_filename.replace('.fits','_truth.txt')}.")

    # Save images and parameters
    image.header['A1'] = A1
    image.header['A2'] = A2
    image.header['X0_T'] = spectrum.x0[0]
    image.header['Y0_T'] = spectrum.x0[1]
    image.header['D2CCD_T'] = spectrum.disperser.D
    image.header['OZONE'] = ozone
    image.header['PWV'] = pwv
    image.header['VAOD'] = aerosols
    image.header['PSF_DEG'] = spectrum.spectrogram_deg
    psf_poly_params_truth = np.array(psf_poly_params)
    if psf_poly_params_truth.size > spectrum.spectrogram_Nx:
        psf_poly_params_truth = psf_poly_params_truth[spectrum.spectrogram_Nx:]
    image.header['PSF_POLY'] = np.array_str(psf_poly_params_truth, max_line_width=1000, precision=4)
    image.header['RESO'] = reso
    image.header['ROTATION'] = int(with_rotation)
    image.header['ROTANGLE'] = spectrum.rotation_angle
    image.header['STARS'] = int(with_stars)
    image.header['BKGD_LEV'] = background.level
    image.save_image(output_filename, overwrite=True)
    return image


if __name__ == "__main__":
    from spectractor.logbook import LogBook
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                        help="Write results in given output directory (default: ./outputs/).")
    parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = ['tests/data/reduc_20170530_134.fits']
    spectrum_file_name = 'outputs/reduc_20170530_134_spectrum.fits'
    # guess = [720, 670]
    # hologramme HoloAmAg
    psf_poly_params = [0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955,
                       -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901,
                       0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673,
                       528.3594585697788, 628.4966480821147, 12.438043546369354, 499.99999999999835]
    # psf_poly_params = [0.11298966008548948, -0.396825836448203, 10.60387678061209, 2.0649268678546955,
    #                    -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901,
    #                    0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673,
    #                    528.3594585697788, 628.4966480821147, 12.438043546369354, 499.99999999999835]
    # file_name="../CTIOAnaJun2017/ana_31may17/OverScanRemove/trim_images/trim_20170531_150.fits"
    # guess = [840, 530]
    # target = "HD205905"
    # x = np.linspace(-1, 1, 100)
    # plt.plot(x, np.polynomial.legendre.legval(x, psf_poly_params[0:3]))
    # plt.show()
    logbook = LogBook(logbook=args.logbook)
    for file_name in file_names:
        tag = file_name.split('/')[-1]
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue

        image = ImageSim(file_name, spectrum_file_name, args.output_directory, A1=1, A2=0.05,
                         psf_poly_params=psf_poly_params, with_stars=False)

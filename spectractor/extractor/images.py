from astropy.coordinates import Angle, SkyCoord
from astropy.modeling import fitting
from astropy.io import fits
import astropy.units as units
from scipy import ndimage
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os

from spectractor import parameters
from spectractor.config import set_logger, load_config
from spectractor.extractor.targets import load_target
from spectractor.extractor.dispersers import Hologram
from spectractor.extractor.psf import fit_PSF2D_minuit, PSF, MoffatGauss, Moffat, PSFFitWorkspace
from spectractor.tools import (plot_image_simple, save_fits, load_fits, fit_poly1d,
                               fit_poly1d_outlier_removal, weighted_avg_and_std,
                               fit_poly2d_outlier_removal, hessian_and_theta,
                               set_wcs_file_name, load_wcs_from_file, imgslice)


class Image(object):

    def __init__(self, file_name, target_label="", disperser_label="", config=""):
        """
        The image class contains all the features necessary to load an image and extract a spectrum.

        Parameters
        ----------
        file_name: str
            The file name where the image is.
        target_label: str, optional
            The target name, to be found in data bases.
        disperser_label: str, optional
            The disperser label to load its properties
        config: str, optional
            A config file name to load some parameter values for a given instrument (default: "").

        Examples
        --------

        .. doctest::

            >>> im = Image('tests/data/reduc_20170605_028.fits')

        .. doctest::
            :hide:

            >>> assert im.file_name == 'tests/data/reduc_20170605_028.fits'
            >>> assert im.data is not None and np.mean(im.data) > 0
            >>> assert im.stat_errors is not None and np.mean(im.stat_errors) > 0
            >>> assert im.header is not None
            >>> assert im.gain is not None and np.mean(im.gain) > 0

        """
        self.my_logger = set_logger(self.__class__.__name__)
        if config != "":
            load_config(config)
        self.file_name = file_name
        self.units = 'ADU'
        self.expo = -1
        self.airmass = -1
        self.date_obs = None
        self.disperser = None
        self.disperser_label = disperser_label
        self.target_label = target_label
        self.filter = None
        self.filters = None
        self.header = None
        self.data = None
        self.data_rotated = None
        self.gain = None  # in e-/ADU
        self.read_out_noise = None
        self.stat_errors = None
        self.stat_errors_rotated = None
        self.rotation_angle = 0
        self.parallactic_angle = None
        self.saturation = None
        if parameters.CALLING_CODE != 'LSST_DM':
            self.load_image(file_name)
        else:
            # data provided by the LSST shim, just instantiate objects
            # necessary for the following code not to fail
            self.header = fits.header.Header()
        # Load the target if given
        self.target = None
        self.target_pixcoords = None
        self.target_pixcoords_rotated = None
        self.target_bkgd2D = None
        self.target_star2D = None
        self.header['TARGET'] = self.target_label
        self.header.comments['TARGET'] = 'name of the target in the image'
        self.header['REDSHIFT'] = 0
        self.header.comments['REDSHIFT'] = 'redshift of the target'
        self.header["GRATING"] = self.disperser_label
        self.header.comments["GRATING"] = "name of the disperser"
        self.header['ROTANGLE'] = self.rotation_angle
        self.header.comments["ROTANGLE"] = "[deg] angle of the dispersion axis"
        self.header['D2CCD'] = parameters.DISTANCE2CCD
        self.header.comments["D2CCD"] = "[mm] distance between disperser and CCD"
        if self.target_label != "":
            self.target = load_target(self.target_label, verbose=parameters.VERBOSE)
            self.header['REDSHIFT'] = str(self.target.redshift)

    def load_image(self, file_name):
        """
        Load the image and store some information from header in class attributes.
        Then load the target and disperser properties. Called when an Image instance is created.

        Parameters
        ----------
        file_name: str
            The fits file name.

        """
        if parameters.OBS_NAME == 'CTIO':
            load_CTIO_image(self)
        elif parameters.OBS_NAME == 'LPNHE':
            load_LPNHE_image(self)
        if parameters.OBS_NAME == "AUXTEL":
            load_AUXTEL_image(self)
        # Load the disperser
        self.my_logger.info(f'\n\tLoading disperser {self.disperser_label}...')
        self.header["GRATING"] = self.disperser_label
        self.header["AIRMASS"] = self.airmass
        self.header["DATE-OBS"] = self.date_obs
        self.header["EXPTIME"] = self.expo
        self.disperser = Hologram(self.disperser_label, D=parameters.DISTANCE2CCD,
                                  data_dir=parameters.DISPERSER_DIR, verbose=parameters.VERBOSE)
        self.compute_statistical_error()
        self.convert_to_ADU_rate_units()

    def save_image(self, output_file_name, overwrite=False):
        """Save the image in a fits file.

        Parameters
        ----------
        output_file_name: str
            The output file name.
        overwrite: bool
            If True, overwrite the file if it exists previously (default: False).

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> im.save_image('tests/data/reduc_20170605_028_copy.fits', overwrite=True)
        """
        save_fits(output_file_name, self.header, self.data, overwrite=overwrite)
        self.my_logger.info(f'\n\tImage saved in {output_file_name}')

    def convert_to_ADU_rate_units(self):
        """Convert Image data from ADU to ADU/s units.

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> print(im.expo)
        600.0
        >>> data_before = np.copy(im.data)
        >>> im.convert_to_ADU_rate_units()

        .. doctest::
            :hide:

            >>> assert np.all(np.isclose(data_before, im.data * im.expo))
        """
        self.data = self.data.astype(np.float64) / self.expo
        self.stat_errors /= self.expo
        if self.stat_errors_rotated is not None:
            self.stat_errors_rotated /= self.expo
        self.units = 'ADU/s'

    def convert_to_ADU_units(self):
        """Convert Image data from ADU/s to ADU units.

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> print(im.expo)
        600.0
        >>> data_before = np.copy(im.data)
        >>> im.convert_to_ADU_rate_units()
        >>> data_after = np.copy(im.data)
        >>> im.convert_to_ADU_units()

        .. doctest::
            :hide:

            >>> assert np.all(np.isclose(data_before, im.data))
        """
        self.data *= self.expo
        self.stat_errors *= self.expo
        if self.stat_errors_rotated is not None:
            self.stat_errors_rotated *= self.expo
        self.units = 'ADU'

    def compute_statistical_error(self):
        """Compute the image noise map from Image.data. The latter must be in ADU.
        The function first converts the image in electron counts units, evaluate the Poisson noise,
        add in quadrature the read-out noise, takes the square root and returns a map in ADU units.

        Examples
        --------
        .. doctest::

            >>> im = Image('tests/data/reduc_20170530_134.fits')
            >>> im.compute_statistical_error()
            >>> im.convert_to_ADU_units()
            >>> im.compute_statistical_error()
            >>> im.plot_statistical_error()

        .. plot::

            from spectractor.extractor.images import Image
            im = Image('tests/data/reduc_20170530_134.fits')
            im.convert_to_ADU_units()
            im.plot_statistical_error()

        """
        if self.units != 'ADU':
            self.my_logger.error(f'\n\tNoise must be estimated on an image in ADU units. '
                                 f'Currently self.units={self.units}.')
        # removes the zeros and negative pixels first
        # set to minimum positive value
        data = np.copy(self.data)
        # OLD: compute poisson noise in ADU/s without read-out noise
        # self.stat_errors = np.sqrt(data) / np.sqrt(self.gain * self.expo)
        # convert in e- counts
        err2 = data * self.gain
        if self.read_out_noise is not None:
            err2 += self.read_out_noise * self.read_out_noise
        # remove negative values (due to dead columns for instance
        min_noz = np.min(err2[err2 > 0])
        err2[err2 <= 0] = min_noz
        self.stat_errors = np.sqrt(err2)
        # convert in ADU
        self.stat_errors /= self.gain
        # check uncertainty model
        self.check_statistical_error()

    def check_statistical_error(self):
        """Check that statistical uncertainty map follows the input uncertainty model
        in terms of gain and read-out noise.

        A linear model is fitted to the squared pixel uncertainty values with respect to the pixel data values.
        The slop gives the gain value and the intercept gives the read-out noise value.

        Returns
        -------
        fit: tuple
            The best fit parameter of the linear model.
        x: array_like
            The x data used for the fit (data).
        y: array_like
            The y data used for the fit (squared uncertainties).
        model: array_like
            The linear model values.

        Examples
        --------

        .. doctest::

            >>> im = Image('tests/data/reduc_20170530_134.fits')
            >>> im.convert_to_ADU_units()
            >>> fit, x, y, model = im.check_statistical_error()

        .. doctest::
            :hide:

            >>> assert fit is not None
            >>> assert len(fit) == 2
            >>> assert x.shape == y.shape
            >>> assert y.shape == model.shape

        """
        if self.units != "ADU":
            self.my_logger.error(f"\n\tNoise map must be in ADU units to be plotted and analyzed. "
                                 f"Currently self.units={self.units}.")
        data = np.copy(self.data)
        min_noz = np.min(data[data > 0])
        data[data <= 0] = min_noz
        y = self.stat_errors.flatten() ** 2
        x = data.flatten()
        fit, cov, model = fit_poly1d(x, y, order=1)
        gain = 1 / fit[0]
        read_out = np.sqrt(fit[1]) * gain
        if not np.isclose(gain, np.mean(self.gain), rtol=1e-2):
            self.my_logger.warning(f"\n\tFitted gain seems to be different than input gain. "
                                   f"Fit={gain} but average of self.gain is {np.mean(self.gain)}.")
        if not np.isclose(read_out, np.mean(self.read_out_noise), rtol=1e-2):
            self.my_logger.warning(f"\n\tFitted read out noise seems to be different than input readout noise. "
                                   f"Fit={read_out} but average of self.read_out_noise is "
                                   f"{np.mean(self.read_out_noise)}.")
        return fit, x, y, model

    def plot_statistical_error(self):
        """Plot the statistical uncertainty map and check it is a Poisson noise.

        The image units must be ADU.

        Examples
        --------

        .. doctest::

            >>> im = Image('tests/data/reduc_20170530_134.fits')
            >>> im.convert_to_ADU_units()
            >>> im.plot_statistical_error()

        .. plot::

            from spectractor.extractor.images import Image
            im = Image('tests/data/reduc_20170530_134.fits')
            im.convert_to_ADU_units()
            im.plot_statistical_error()

        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fit, x, y, model = self.check_statistical_error()
        gain = 1 / fit[0]
        read_out = np.sqrt(fit[1]) * gain
        ax[0].text(0.05, 0.95, f"fitted gain={gain:.3g} [e-/ADU]\nintercept={fit[1]:.3g} [ADU$^2$]"
                               f"\nfitted read-out={read_out:.3g} [ADU]",
                   horizontalalignment='left', verticalalignment='top', transform=ax[0].transAxes)
        ax[0].scatter(x, y)
        ax[0].plot(x, model, "k-")
        ax[0].grid()
        ax[0].set_ylabel(r"$\sigma_{\mathrm{ADU}}^2$ [ADU$^2$]")
        ax[0].set_xlabel(r"Data pixel values [ADU]")
        plot_image_simple(ax[1], data=self.stat_errors, scale="log10", title="Statistical uncertainty map",
                          units=self.units, target_pixcoords=None, aspect="auto", cmap=None)
        fig.tight_layout()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'uncertainty_map.png'))
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()

    def compute_parallactic_angle(self):
        """Compute the parallactic angle.

        Script from A. Guyonnet.
        """
        latitude = parameters.OBS_LATITUDE.split()
        latitude = float(latitude[0]) - float(latitude[1]) / 60. - float(latitude[2]) / 3600.
        latitude = Angle(latitude, units.deg).radian
        ha = Angle(self.header['HA'], unit='hourangle').radian
        dec = Angle(self.header['DEC'], unit=units.deg).radian
        parallactic_angle = np.arctan(np.sin(ha) / (np.cos(dec) * np.tan(latitude) - np.sin(dec) * np.cos(ha)))
        self.parallactic_angle = parallactic_angle * 180 / np.pi
        self.header['PARANGLE'] = self.parallactic_angle
        self.header.comments['PARANGLE'] = 'parallactic angle in degree'
        return self.parallactic_angle

    def plot_image(self, ax=None, scale="lin", title="", units="", plot_stats=False,
                   target_pixcoords=None, figsize=[7.3, 6], aspect=None, vmin=None, vmax=None,
                   cmap=None, cax=None):
        """Plot image.

        Parameters
        ----------
        ax: Axes, optional
            Axes instance (default: None).
        scale: str
            Scaling of the image (choose between: lin, log or log10, symlog) (default: lin)
        title: str
            Title of the image (default: "")
        units: str
            Units of the image to be written in the color bar label (default: "")
        cmap: colormap
            Color map label (default: None)
        target_pixcoords: array_like, optional
            2D array  giving the (x,y) coordinates of the targets on the image: add a scatter plot (default: None)
        vmin: float
            Minimum value of the image (default: None)
        vmax: float
            Maximum value of the image (default: None)
        aspect: str
            Aspect keyword to be passed to imshow (default: None)
        cax: Axes, optional
            Color bar axes if necessary (default: None).
        figsize: tuple
            Figure size (default: [9.3, 8]).
        plot_stats: bool
            If True, plot the uncertainty map instead of the image (default: False).

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> im.plot_image(target_pixcoords=[820, 580])
        >>> if parameters.DISPLAY: plt.show()
        """
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        data = np.copy(self.data)
        if plot_stats:
            data = np.copy(self.stat_errors)
        if units == "":
            units = self.units
        plot_image_simple(ax, data=data, scale=scale, title=title, units=units, cax=cax,
                          target_pixcoords=target_pixcoords, aspect=aspect, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.legend()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()


def load_CTIO_image(image):
    """Specific routine to load CTIO fits files and load their data and properties for Spectractor.

    Parameters
    ----------
    image: Image
        The Image instance to fill with file data and header.
    """
    image.my_logger.info(f'\n\tLoading CTIO image {image.file_name}...')
    image.header, image.data = load_fits(image.file_name)

    image.date_obs = image.header['DATE-OBS']
    image.airmass = float(image.header['AIRMASS'])
    image.expo = float(image.header['EXPTIME'])
    image.filters = image.header['FILTERS']
    image.filter = image.header['FILTER1']
    image.disperser_label = image.header['FILTER2']

    parameters.CCD_IMSIZE = int(image.header['XLENGTH'])
    parameters.CCD_PIXEL2ARCSEC = float(image.header['XPIXSIZE'])
    if image.header['YLENGTH'] != parameters.CCD_IMSIZE:
        image.my_logger.warning(
            f'\n\tImage rectangular: X={parameters.CCD_IMSIZE:d} pix, Y={image.header["YLENGTH"]:d} pix')
    if image.header['YPIXSIZE'] != parameters.CCD_PIXEL2ARCSEC:
        image.my_logger.warning('\n\tPixel size rectangular: X=%d arcsec, Y=%d arcsec' % (
            parameters.CCD_PIXEL2ARCSEC, image.header['YPIXSIZE']))
    image.coord = SkyCoord(image.header['RA'] + ' ' + image.header['DEC'], unit=(units.hourangle, units.deg),
                           obstime=image.header['DATE-OBS'])
    image.my_logger.info(f'\n\tImage {image.file_name} loaded.')
    # compute CCD gain map
    build_CTIO_gain_map(image)
    build_CTIO_read_out_noise_map(image)
    image.compute_parallactic_angle()


def build_CTIO_gain_map(image):
    """Compute the CTIO gain map according to header GAIN values.

    Parameters
    ----------
    image: Image
        The Image instance to fill with file data and header.
    """
    size = parameters.CCD_IMSIZE
    image.gain = np.zeros_like(image.data)
    # ampli 11
    image.gain[0:size // 2, 0:size // 2] = image.header['GTGAIN11']
    # ampli 12
    image.gain[0:size // 2, size // 2:size] = image.header['GTGAIN12']
    # ampli 21
    image.gain[size // 2:size, 0:size] = image.header['GTGAIN21']
    # ampli 22
    image.gain[size // 2:size, size // 2:size] = image.header['GTGAIN22']


def build_CTIO_read_out_noise_map(image):
    """Compute the CTIO gain map according to header GAIN values.

    Parameters
    ----------
    image: Image
        The Image instance to fill with file data and header.
    """
    size = parameters.CCD_IMSIZE
    image.read_out_noise = np.zeros_like(image.data)
    # ampli 11
    image.read_out_noise[0:size // 2, 0:size // 2] = image.header['GTRON11']
    # ampli 12
    image.read_out_noise[0:size // 2, size // 2:size] = image.header['GTRON12']
    # ampli 21
    image.read_out_noise[size // 2:size, 0:size] = image.header['GTRON21']
    # ampli 22
    image.read_out_noise[size // 2:size, size // 2:size] = image.header['GTRON22']


def load_LPNHE_image(image):  # pragma: no cover
    """Specific routine to load LPNHE fits files and load their data and properties for Spectractor.

    Parameters
    ----------
    image: Image
        The Image instance to fill with file data and header.
    """
    image.my_logger.info(f'\n\tLoading LPNHE image {image.file_name}...')
    hdus = fits.open(image.file_name)
    image.header = hdus[0].header
    hdu1 = hdus["CHAN_14"]
    hdu2 = hdus["CHAN_06"]
    data1 = hdu1.data[imgslice(hdu1.header['DATASEC'])].astype(np.float64)
    bias1 = np.median(hdu1.data[imgslice(hdu1.header['BIASSEC'])].astype(np.float64))
    data1 -= bias1
    detsecy, detsecx = imgslice(hdu1.header['DETSEC'])
    if detsecy.start > detsecy.stop:
        data1 = data1[:, ::-1]
    if detsecx.start > detsecx.stop:
        data1 = data1[::-1, :]
    data2 = hdu2.data[imgslice(hdu2.header['DATASEC'])].astype(np.float64)
    bias2 = np.median(hdu2.data[imgslice(hdu2.header['BIASSEC'])].astype(np.float64))
    data2 -= bias2
    detsecy, detsecx = imgslice(hdu2.header['DETSEC'])
    if detsecy.start > detsecy.stop:
        data2 = data2[:, ::-1]
    if detsecx.start > detsecx.stop:
        data2 = data2[::-1, :]
    data = np.concatenate([data2, data1])
    image.data = data[::-1, :].T
    image.date_obs = image.header['DATE-OBS']
    image.expo = float(image.header['EXPTIME'])
    image.airmass = -1
    parameters.DISTANCE2CCD -= float(hdus["XYZ"].header["ZPOS"])
    if "mm" not in hdus["XYZ"].header.comments["ZPOS"]:
        image.my_logger.error(f'\n\tmm is absent from ZPOS key in XYZ header. Had {hdus["XYZ"].header.comments["ZPOS"]}'
                              f'Distances along Z axis must be in mm.')
    image.my_logger.info(f'\n\tDistance to CCD adjusted to {parameters.DISTANCE2CCD} mm '
                         f'considering XYZ platform is set at ZPOS={float(hdus["XYZ"].header["ZPOS"])} mm.')
    # compute CCD gain map
    image.gain = float(image.header['CCDGAIN']) * np.ones_like(image.data)
    image.read_out_noise = float(image.header['CCDNOISE']) * np.ones_like(image.data)
    parameters.CCD_IMSIZE = image.data.shape[1]
    # save xys platform position into main header
    image.header["XPOS"] = hdus["XYZ"].header["XPOS"]
    image.header.comments["XPOS"] = hdus["XYZ"].header.comments["XPOS"]
    image.header["YPOS"] = hdus["XYZ"].header["YPOS"]
    image.header.comments["YPOS"] = hdus["XYZ"].header.comments["YPOS"]
    image.header["ZPOS"] = hdus["XYZ"].header["ZPOS"]
    image.header.comments["ZPOS"] = hdus["XYZ"].header.comments["ZPOS"]
    image.my_logger.info('\n\tImage loaded')


def load_AUXTEL_image(image):  # pragma: no cover
    """Specific routine to load AUXTEL fits files and load their data and properties for Spectractor.

    Parameters
    ----------
    image: Image
        The Image instance to fill with file data and header.
    """
    image.my_logger.info(f'\n\tLoading AUXTEL image {image.file_name}...')
    image.my_logger.warning(image.header)
    hdu_list = fits.open(image.file_name)
    image.header = hdu_list[0].header
    image.data = hdu_list[1].data.astype(np.float64)
    hdu_list.close()  # need to free allocation for file descripto
    # image.data = np.concatenate((data[10:-10, 10:-10], data2[10:-10, 10:-10]))
    image.date_obs = image.header['DATE-OBS']
    image.expo = float(image.header['EXPTIME'])
    image.data = image.data.T[:, ::-1]
    if image.header["AMSTART"] is not None:
        image.airmass = 0.5 * (float(image.header["AMSTART"]) + float(image.header["AMEND"]))
    else:
        image.airmass = -1
    image.my_logger.info('\n\tImage loaded')
    # compute CCD gain map
    image.gain = float(parameters.CCD_GAIN) * np.ones_like(image.data)
    parameters.CCD_IMSIZE = image.data.shape[1]
    image.disperser_label = image.header['GRATING']
    image.read_out_noise = np.zeros_like(image.data)


def find_target(image, guess=None, rotated=False, use_wcs=True):
    """Find the target in the Image instance.

    The object is search in a windows of size defined by the XWINDOW and YWINDOW parameters,
    using two iterative fits of a PSF model.
    User must give a guess array in the raw image.

    Parameters
    ----------
    image: Image
        The Image instance.
    guess: array_like
        Two parameter array giving the estimated position of the target in the image, optional if WCS is used.
    rotated: bool
        If True, the target is searched in the rotated image.
    use_wcs: bool
        If True, the WCS file (if found) is used to set the target position in pixels,
        guess parameter is then unnecessary.

    Returns
    -------
    x0: float
        The x position of the target.
    y0: float
        The y position of the target.

    Examples
    --------
    >>> im = Image('tests/data/reduc_20170605_028.fits')
    >>> im.plot_image()
    >>> guess = [820, 580]
    >>> parameters.VERBOSE = True
    >>> parameters.DEBUG = True
    >>> x1, y1 = find_target(im, guess)
    """
    my_logger = set_logger(__name__)
    target_pixcoords = [-1, -1]
    theX = -1
    theY = -1
    if use_wcs:
        wcs_file_name = set_wcs_file_name(image.file_name)
        if os.path.isfile(wcs_file_name):
            my_logger.info(f"\n\tUse WCS {wcs_file_name} to find target pixel position.")
            if rotated:
                target_pixcoords = find_target_after_rotation(image)
                theX, theY = target_pixcoords
            else:
                wcs = load_wcs_from_file(wcs_file_name)
                target_coord_after_motion = image.target.get_radec_position_after_pm(image.date_obs)
                # noinspection PyUnresolvedReferences
                target_pixcoords = np.array(wcs.all_world2pix(target_coord_after_motion.ra,
                                                              target_coord_after_motion.dec, 0))
                theX, theY = target_pixcoords
            if parameters.DEBUG:
                plt.figure(figsize=(5, 5))
                sub_image, x0, y0, Dx, Dy, sub_errors = find_target_init(image=image, guess=[theX, theY],
                                                                         rotated=rotated, widths=(20, 20))
                plot_image_simple(plt.gca(), data=sub_image, scale="lin", title="", units=image.units,
                                  target_pixcoords=[theX - x0 + Dx, theY - y0 + Dy])
                plt.show()
        else:
            my_logger.info(f"\n\tNo WCS {wcs_file_name} available, use 2D fit to find target pixel position.")
    if target_pixcoords[0] == -1 and target_pixcoords[1] == -1:
        if guess is None:
            my_logger.error(f"\n\tguess parameter must be set if WCS solution is not found.")
        Dx = parameters.XWINDOW
        Dy = parameters.YWINDOW
        theX, theY = guess
        if rotated:
            guess2 = find_target_after_rotation(image)
            x0 = int(guess2[0])
            y0 = int(guess2[1])
            guess = [x0, y0]
            Dx = parameters.XWINDOW_ROT
            Dy = parameters.YWINDOW_ROT
        niter = 2
        sub_image_subtracted, x0, y0, Dx, Dy, sub_errors = find_target_init(image=image, guess=guess, rotated=rotated,
                                                                            widths=[Dx, Dy])
        for i in range(niter):
            # find the target
            try:
                avX, avY = find_target_2Dprofile(image, sub_image_subtracted, sub_errors=sub_errors)
            except (Exception, ValueError):
                image.target_star2D = None
                avX, avY = find_target_2DprofileASTROPY(image, sub_image_subtracted, sub_errors=sub_errors)
            # compute target position
            theX = x0 - Dx + avX
            theY = y0 - Dy + avY
            # crop for next iteration
            Dx = Dx // (i + 2)
            Dy = Dy // (i + 2)
            x0 = int(theX)
            y0 = int(theY)
            NY, NX = sub_image_subtracted.shape
            sub_image_subtracted = sub_image_subtracted[max(0, int(avY) - Dy):min(NY, int(avY) + Dy),
                                                        max(0, int(avX) - Dx):min(NX, int(avX) + Dx)]
            sub_errors = sub_errors[max(0, int(avY) - Dy):min(NY, int(avY) + Dy),
                                    max(0, int(avX) - Dx):min(NX, int(avX) + Dx)]
            if int(avX) - Dx < 0:
                Dx = int(avX)
            if int(avY) - Dy < 0:
                Dy = int(avY)
    image.my_logger.info(f'\n\tX,Y target position in pixels: {theX:.3f},{theY:.3f}')
    if rotated:
        image.target_pixcoords_rotated = [theX, theY]
    else:
        image.target_pixcoords = [theX, theY]
        image.header['TARGETX'] = theX
        image.header.comments['TARGETX'] = 'target position on X axis'
        image.header['TARGETY'] = theY
        image.header.comments['TARGETY'] = 'target position on Y axis'
    return [theX, theY]


def find_target_after_rotation(image):
    angle = image.rotation_angle * np.pi / 180.
    rotmat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    vec = np.array(image.target_pixcoords) - 0.5 * np.array(image.data.shape[::-1])
    target_pixcoords_after_rotation = rotmat @ vec + 0.5 * np.array(image.data_rotated.shape[::-1])
    return target_pixcoords_after_rotation


def find_target_init(image, guess, rotated=False, widths=[parameters.XWINDOW, parameters.YWINDOW]):
    """Initialize the search of the target: crop the image, set the saturation level,
    estimate and subtract a 2D polynomial background.

    Parameters
    ----------
    image: Image
        The Image instance.
    guess: array_like
        Two parameter array giving the estimated position of the target in the image.
    rotated: bool
        If True, the target is searched in the rotated image.
    widths: array_like
        Two parameter array to define the width of the cropped image (default: [parameters.XWINDOW, parameters.YWINDOW])

    Returns
    -------
    sub_image: array_like
        The cropped image data array where the fit has to be performed.
    x0: float
        The x position of the target.
    y0: float
        The y position of the target.
    Dx: int
        The width of the cropped image.
    Dy: int
        The height of the cropped image.
    sub_errors: array_like
        The cropped image uncertainty array where the fit has to be performed.
    """
    x0 = int(guess[0])
    y0 = int(guess[1])
    Dx, Dy = widths
    if rotated:
        sub_image = np.copy(image.data_rotated[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
        sub_errors = np.copy(image.stat_errors[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
    else:
        sub_image = np.copy(image.data[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
        sub_errors = np.copy(image.stat_errors[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
    image.saturation = parameters.CCD_MAXADU / image.expo
    NY, NX = sub_image.shape
    Y, X = np.mgrid[:NY, :NX]
    bkgd_2D = fit_poly2d_outlier_removal(X, Y, sub_image, order=1, sigma=3)
    image.target_bkgd2D = bkgd_2D
    sub_image_subtracted = sub_image - bkgd_2D(X, Y)
    saturated_pixels = np.where(sub_image >= image.saturation)
    if len(saturated_pixels[0]) > 0:
        image.my_logger.debug(
            f'\n\t{len(saturated_pixels[0])} saturated pixels: set saturation level '
            f'to {image.saturation} {image.units}.')
        sub_errors[sub_image >= 0.99 * image.saturation] = 10 * image.saturation  # np.min(np.abs(sub_errors))
    # sub_image = clean_target_spikes(sub_image, image.saturation)
    return sub_image_subtracted, x0, y0, Dx, Dy, sub_errors


def find_target_1Dprofile(image, sub_image, guess):
    """
    Find precisely the position of the targeted object fitting a PSF model
    on each projection of the image along x and y, using outlier removal.

    Parameters
    ----------
    image: Image
        The Image instance.
    sub_image: array_like
        The cropped image data array where the fit is performed.
    guess: array_like
        Two parameter array giving the estimated position of the target in the image.

    Examples
    --------
    >>> im = Image('tests/data/reduc_20170605_028.fits')
    >>> guess = [820, 580]
    >>> parameters.DEBUG = True
    >>> sub_image, x0, y0, Dx, Dy, sub_errors = find_target_init(im, guess, rotated=False)
    >>> x1, y1 = find_target_1Dprofile(im, sub_image, guess)

    .. doctest::
        :hide:

        >>> assert np.isclose(x1, np.argmax(np.nansum(sub_image, axis=0)), rtol=1e-2)
        >>> assert np.isclose(y1, np.argmax(np.nansum(sub_image, axis=1)), rtol=1e-2)

    """
    NY, NX = sub_image.shape
    X = np.arange(NX)
    Y = np.arange(NY)
    # Mask aigrette
    saturated_pixels = np.where(sub_image >= image.saturation)
    if parameters.DEBUG:
        image.my_logger.info('\n\t%d saturated pixels: set saturation level to %d %s.' % (
            len(saturated_pixels[0]), image.saturation, image.units))
        # sub_image[sub_image >= 0.5*image.saturation] = np.nan
    # compute profiles
    profile_X_raw = np.sum(sub_image, axis=0)
    profile_Y_raw = np.sum(sub_image, axis=1)
    # fit and subtract smooth polynomial background
    # with 3sigma rejection of outliers (star peaks)
    bkgd_X, outliers = fit_poly1d_outlier_removal(X, profile_X_raw, order=2)
    bkgd_Y, outliers = fit_poly1d_outlier_removal(Y, profile_Y_raw, order=2)
    profile_X = profile_X_raw - bkgd_X(X)  # np.min(profile_X)
    profile_Y = profile_Y_raw - bkgd_Y(Y)  # np.min(profile_Y)
    avX, sigX = weighted_avg_and_std(X, profile_X ** 4)
    avY, sigY = weighted_avg_and_std(Y, profile_Y ** 4)
    if profile_X[int(avX)] < 0.8 * np.max(profile_X):
        image.my_logger.warning('\n\tX position determination of the target probably wrong')
    if profile_Y[int(avY)] < 0.8 * np.max(profile_Y):
        image.my_logger.warning('\n\tY position determination of the target probably wrong')
    if parameters.DEBUG:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        plot_image_simple(ax1, data=sub_image, scale="log", title="", units=image.units, target_pixcoords=[avX, avY])
        ax1.legend(loc=1)
        ax2.plot(X, profile_X_raw, 'r-', lw=2)
        ax2.plot(X, bkgd_X(X), 'g--', lw=2, label='bkgd')
        ax2.axvline(avX, color='b', linestyle='-', label='new', lw=2)
        ax2.grid(True)
        ax2.set_xlabel('X [pixels]')
        ax2.legend(loc=1)
        ax3.plot(Y, profile_Y_raw, 'r-', lw=2)
        ax3.plot(Y, bkgd_Y(Y), 'g--', lw=2, label='bkgd')
        ax3.axvline(avY, color='b', linestyle='-', label='new', lw=2)
        ax3.grid(True)
        ax3.set_xlabel('Y [pixels]')
        ax3.legend(loc=1)
        f.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            f.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'namethisplot1.pdf'))
    return avX, avY


def find_target_2Dprofile(image, sub_image_subtracted, sub_errors=None):
    """
    Find precisely the position of the targeted object fitting a PSF model.
    A polynomial 2D background is subtracted first. Saturated pixels are masked with np.nan values.

    Parameters
    ----------
    image: Image
        The Image instance.
    sub_image_subtracted: array_like
        The cropped image data array where the fit is performed, background has been subtracted.
    sub_errors: array_like
        The image data uncertainties.

    Examples
    --------
    >>> im = Image('tests/data/reduc_20170605_028.fits')
    >>> guess = [820, 580]
    >>> parameters.VERBOSE = True
    >>> parameters.DEBUG = True
    >>> sub_image_subtracted, x0, y0, Dx, Dy, sub_errors = find_target_init(im, guess, rotated=False) #, widths=[30,30])
    >>> xmax = np.argmax(np.sum(sub_image_subtracted, axis=0))
    >>> ymax = np.argmax(np.sum(sub_image_subtracted, axis=1))
    >>> x1, y1 = find_target_2Dprofile(im, sub_image_subtracted, sub_errors=sub_errors)

    .. doctest::
        :hide:

        >>> assert np.isclose(x1, xmax, rtol=1e-2)
        >>> assert np.isclose(y1, ymax, rtol=1e-2)

    """
    # fit and subtract smooth polynomial background
    # with 3sigma rejection of outliers (star peaks)
    NY, NX = sub_image_subtracted.shape
    XX = np.arange(NX)
    YY = np.arange(NY)
    # find a first guess of the target position
    avX, sigX = weighted_avg_and_std(XX, np.sum(sub_image_subtracted, axis=0) ** 4)
    avY, sigY = weighted_avg_and_std(YY, np.sum(sub_image_subtracted, axis=1) ** 4)
    # fit a 2D star profile close to this position
    # guess = [np.max(sub_image_subtracted),avX,avY,1,1] #for Moffat2Ds
    # guess = [np.max(sub_image_subtracted),avX-2,avY-2,2,2,0] #for Gauss2D
    psf = Moffat()
    total_flux = np.sum(sub_image_subtracted)
    psf.p[:3] = [total_flux, avX, avY]
    psf.p[-1] = image.saturation
    if image.target_star2D is not None:
        psf.p = image.target_star2D.p
        psf.p[1] = avX
        psf.p[2] = avY
    mean_prior = 10  # in pixels
    # bounds = [ [0.5*np.max(sub_image_subtracted),avX-mean_prior,avY-mean_prior,0,-np.inf],
    # [2*np.max(sub_image_subtracted),avX+mean_prior,avY+mean_prior,np.inf,np.inf] ] #for Moffat2D
    # bounds = [ [0.5*np.max(sub_image_subtracted),avX-mean_prior,avY-mean_prior,0,0,0],
    # [100*image.saturation,avX+mean_prior,avY+mean_prior,10,10,np.pi] ] #for Gauss2D
    # bounds = [[0.5 * np.max(sub_image_subtracted), avX - mean_prior, avY - mean_prior, 2, 0.9 * image.saturation],
    # [10 * image.saturation, avX + mean_prior, avY + mean_prior, 15, 1.1 * image.saturation]]
    psf.bounds[:3] = [[0.1 * total_flux, 4 * total_flux],
                      [avX - mean_prior, avX + mean_prior],
                      [avY - mean_prior, avY + mean_prior]]
    # fit
    # star2D = fit_PSF2D(X, Y, sub_image_subtracted, guess=guess, bounds=bounds)
    # star2D = fit_PSF2D_minuit(X, Y, sub_image_subtracted, guess=guess, bounds=bounds)
    psf.fit_psf(sub_image_subtracted, data_errors=sub_errors, bgd_model_func=image.target_bkgd2D)
    new_avX = psf.p[1]
    new_avY = psf.p[2]
    image.target_star2D = psf
    # check target positions
    dist = np.sqrt((new_avY - avY) ** 2 + (new_avX - avX) ** 2)
    if dist > mean_prior / 2:
        image.my_logger.warning(
            f'\n\tX={new_avX:.2f}, Y={new_avY:.2f} target position determination probably wrong: '
            f'{dist:.1f} pixels from profile detection ({avX:.2f}, {avY:.2f})')
    # debugging plots
    if parameters.DEBUG:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        vmin = 0
        vmax = float(np.nanmax(sub_image_subtracted))
        X, Y = np.mgrid[:NX, :NY]
        star2D = psf.evaluate(pixels=np.array([X, Y]))
        plot_image_simple(ax1, data=sub_image_subtracted, scale="lin", title="", units=image.units,
                          target_pixcoords=[new_avX, new_avY], vmin=vmin, vmax=vmax)
        ax1.legend(loc=1)

        plot_image_simple(ax2, data=star2D, scale="lin", title="",
                          units=f'Background+Star2D ({image.units})', vmin=vmin, vmax=vmax)
        plot_image_simple(ax3, data=sub_image_subtracted - star2D, scale="lin", title="",
                          units=f'Background+Star2D subtracted image\n({image.units})',
                          target_pixcoords=[new_avX, new_avY], vmin=vmin, vmax=vmax)
        ax3.legend(loc=1)

        f.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            f.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'namethisplot2.pdf'))
    return new_avX, new_avY


def find_target_2DprofileASTROPY(image, sub_image_subtracted, sub_errors=None):
    """
    Find precisely the position of the targeted object fitting a PSF model.
    A polynomial 2D background was subtracted before. Saturated pixels are masked with np.nan values.

    THE ERROR ARRAY IS NOT USED FOR THE MOMENT.

    Parameters
    ----------
    image: Image
        The Image instance.
    sub_image_subtracted: array_like
        The cropped image data array where the fit is performed.
    sub_errors: array_like
        The image data uncertainties.

    Examples
    --------
    >>> im = Image('tests/data/reduc_20170605_028.fits')
    >>> guess = [820, 580]
    >>> parameters.DEBUG = True
    >>> sub_image_subtracted, x0, y0, Dx, Dy, sub_errors = find_target_init(im, guess, rotated=False)
    >>> xmax = np.argmax(np.sum(sub_image_subtracted, axis=0))
    >>> ymax = np.argmax(np.sum(sub_image_subtracted, axis=1))
    >>> x1, y1 = find_target_2DprofileASTROPY(im, sub_image_subtracted)

    .. doctest::
        :hide:

        >>> assert np.isclose(x1, xmax, rtol=1e-2)
        >>> assert np.isclose(y1, ymax, rtol=1e-2)

    """
    # TODO: replace with minuit and test on image _133.fits or decrease mean_prior
    # fit and subtract smooth polynomial background
    # with 3sigma rejection of outliers (star peaks)
    NY, NX = sub_image_subtracted.shape
    XX = np.arange(NX)
    YY = np.arange(NY)
    Y, X = np.mgrid[:NY, :NX]
    # find a first guess of the target position
    avX, sigX = weighted_avg_and_std(XX, np.sum(sub_image_subtracted, axis=0) ** 4)
    avY, sigY = weighted_avg_and_std(YY, np.sum(sub_image_subtracted, axis=1) ** 4)
    # fit a 2D star profile close to this position
    # guess = [np.max(sub_image_subtracted),avX,avY,1,1] #for Moffat2D
    # guess = [np.max(sub_image_subtracted),avX-2,avY-2,2,2,0] #for Gauss2D
    guess = [np.max(sub_image_subtracted), avX, avY, 1, 1, 0.1, 2, image.saturation]
    if image.target_star2D is not None:
        guess = fitting._model_to_fit_params(image.target_star2D)[0]
        guess[1] = avX
        guess[2] = avY
    mean_prior = 10  # in pixels
    # bounds = [ [0.5*np.max(sub_image_subtracted),avX-mean_prior,avY-mean_prior,0,-np.inf],
    # [2*np.max(sub_image_subtracted),avX+mean_prior,avY+mean_prior,np.inf,np.inf] ] #for Moffat2D
    # bounds = [ [0.5*np.max(sub_image_subtracted),avX-mean_prior,avY-mean_prior,0,0,0],
    # [100*image.saturation,avX+mean_prior,avY+mean_prior,10,10,np.pi] ] #for Gauss2D
    # bounds = [[0.5 * np.max(sub_image_subtracted), avX - mean_prior, avY - mean_prior, 2, 0.9 * image.saturation],
    # [10 * image.saturation, avX + mean_prior, avY + mean_prior, 15, 1.1 * image.saturation]]
    bounds = [[0.1 * np.max(sub_image_subtracted), avX - mean_prior, avY - mean_prior, 1, 0, -100, 1,
               0.9 * image.saturation],
              [10 * image.saturation, avX + mean_prior, avY + mean_prior, 30, 10, 200, 15, 1.1 * image.saturation]]
    sub_image = sub_image_subtracted + image.target_bkgd2D(X, Y)
    saturated_pixels = np.where(sub_image >= image.saturation)
    if len(saturated_pixels[0]) > 0:
        if parameters.DEBUG:
            image.my_logger.info(
                f'\n\t{len(saturated_pixels[0])} saturated pixels: set saturation level '
                f'to {image.saturation} {image.units}.')
        sub_image_subtracted[sub_image >= 0.9 * image.saturation] = np.nan
        sub_image_subtracted[sub_image >= 0.9 * image.saturation] = np.nan
    # fit
    bounds = list(np.array(bounds).T)
    # star2D = fit_PSF2D(X, Y, sub_image_subtracted, guess=guess, bounds=bounds)
    star2D = fit_PSF2D_minuit(X, Y, sub_image_subtracted, guess=guess, bounds=bounds)
    new_avX = star2D.x_mean.value
    new_avY = star2D.y_mean.value
    image.target_star2D = star2D
    # check target positions
    dist = np.sqrt((new_avY - avY) ** 2 + (new_avX - avX) ** 2)
    if dist > mean_prior / 2:
        image.my_logger.warning(
            f'\n\tX={new_avX:.2f}, Y={new_avY:.2f} target position determination probably wrong: '
            f'{dist:.1f} pixels from profile detection ({avX:.2f}, {avY:.2f})')
    # debugging plots
    if parameters.DEBUG:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        vmin = 0
        vmax = float(np.nanmax(sub_image_subtracted))
        plot_image_simple(ax1, data=sub_image_subtracted, scale="lin", title="", units=image.units,
                          target_pixcoords=[new_avX, new_avY], vmin=vmin, vmax=vmax)
        ax1.legend(loc=1)

        plot_image_simple(ax2, data=star2D(X, Y), scale="lin", title="",
                          units=f'Background+Star2D ({image.units})', vmin=vmin, vmax=vmax)
        plot_image_simple(ax3, data=sub_image_subtracted - star2D(X, Y), scale="lin", title="",
                          units=f'Background+Star2D subtracted image\n({image.units})',
                          target_pixcoords=[new_avX, new_avY], vmin=vmin, vmax=vmax)
        ax3.legend(loc=1)

        f.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            f.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'namethisplot2.pdf'))
    return new_avX, new_avY


def compute_rotation_angle_hessian(image, angle_range=(-10, 10), width_cut=parameters.YWINDOW,
                                   right_edge=parameters.CCD_IMSIZE - 200,
                                   margin_cut=12):
    """Compute the rotation angle in degree of a spectrogram with the Hessian of the image.
    Use the disperser rotation angle map as a prior and the target_pixcoords values to crop the image
    around the spectrogram.

    Parameters
    ----------
    image: Image
        The Image instance.
    angle_range: (float, float)
        Don't consider pixel with Hessian angle outside this range (default: (-10,10)).
    width_cut: int
        Half with of the image to consider in height (default: parameters.YWINDOW).
    right_edge: int
        Maximum pixel on the right edge (default: parameters.CCD_IMSIZE - 200).
    margin_cut: int
        After computing the Hessian, to avoid bad values on the edges the function cut on the
        edge of image margin_cut pixels.

    Returns
    -------
    theta: float
        The median value of the histogram of angles deduced with the Hessian of the pixels (in degree).

    Examples
    --------
    >>> im=Image('tests/data/reduc_20170605_028.fits', disperser_label='HoloPhAg')

    Create a mock spectrogram:

    >>> N = parameters.CCD_IMSIZE
    >>> im.data = np.ones((N, N))
    >>> slope = -0.1
    >>> y = lambda x: slope * (x - 0.5*N) + 0.5*N
    >>> for x in np.arange(N):
    ...     im.data[int(y(x)), x] = 10
    ...     im.data[int(y(x))+1, x] = 10
    >>> im.target_pixcoords=(N//2, N//2)
    >>> parameters.DEBUG = True
    >>> theta = compute_rotation_angle_hessian(im)
    >>> print(f'{theta:.2f}, {np.arctan(slope)*180/np.pi:.2f}')
    -5.72, -5.71

    .. doctest::
        :hide:

        >>> assert np.isclose(theta, np.arctan(slope)*180/np.pi, rtol=1e-2)

    """
    x0, y0 = np.array(image.target_pixcoords).astype(int)
    # extract a region
    data = np.copy(image.data[y0 - width_cut:y0 + width_cut, 0:right_edge])
    lambda_plus, lambda_minus, theta = hessian_and_theta(data, margin_cut)
    # thresholds
    lambda_threshold = np.min(lambda_minus)
    mask = np.where(lambda_minus > lambda_threshold)
    theta_mask = np.copy(theta)
    theta_mask[mask] = np.nan
    minimum_pixels = 0.01 * 2 * width_cut * right_edge
    while len(theta_mask[~np.isnan(theta_mask)]) < minimum_pixels:
        lambda_threshold /= 2
        mask = np.where(lambda_minus > lambda_threshold)
        theta_mask = np.copy(theta)
        theta_mask[mask] = np.nan
        # print len(theta_mask[~np.isnan(theta_mask)]), lambda_threshold
    theta_guess = image.disperser.theta(image.target_pixcoords)
    mask2 = np.logical_or(angle_range[0] > theta - theta_guess, theta - theta_guess > angle_range[1])
    theta_mask[mask2] = np.nan
    theta_mask = theta_mask[2:-2, 2:-2]
    theta_hist = theta_mask[~np.isnan(theta_mask)].flatten()
    if parameters.OBS_OBJECT_TYPE != 'STAR':
        pixels = np.where(~np.isnan(theta_mask))
        p = np.polyfit(pixels[1], pixels[0], deg=1)
        theta_median = np.arctan(p[0]) * 180 / np.pi
    else:
        theta_median = float(np.median(theta_hist))
    # theta_critical = 180. * np.arctan(20. / parameters.CCD_IMSIZE) / np.pi
    image.header['THETAFIT'] = theta_median
    image.header.comments['THETAFIT'] = '[deg] [USED] rotation angle from the Hessian analysis'
    image.header['THETAINT'] = theta_guess
    image.header.comments['THETAINT'] = '[deg] rotation angle interp from disperser scan'
    # if abs(theta_median - theta_guess) > theta_critical:
    #     image.my_logger.warning(
    #         f'\n\tInterpolated angle and fitted angle disagrees with more than 20 pixels '
    #         f'over {parameters.CCD_IMSIZE:d} pixels: {theta_median:.2f} vs {theta_guess:.2f}')
    if parameters.DEBUG:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        xindex = np.arange(data.shape[1])
        x_new = np.linspace(xindex.min(), xindex.max(), 50)
        y_new = width_cut + (x_new - x0) * np.tan(theta_median * np.pi / 180.)
        ax1.imshow(theta_mask, origin='lower', cmap=cm.brg, aspect='auto', vmin=angle_range[0], vmax=angle_range[1])
        ax1.plot(x_new, y_new, 'b-')
        ax1.set_ylim(0, 2 * width_cut)
        ax1.set_xlabel('X [pixels]')
        ax1.set_xlabel('Y [pixels]')
        ax1.grid(True)
        n, bins, patches = ax2.hist(theta_hist, bins=int(np.sqrt(len(theta_hist))))
        ax2.plot([theta_median, theta_median], [0, np.max(n)])
        ax2.set_xlabel("Rotation angles [degrees]")
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            f.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'rotation_hessian.pdf'))
    return theta_median


def turn_image(image):
    """Compute the rotation angle using the Hessian algorithm and turn the image.

    The results are stored in Image.data_rotated and Image.stat_errors_rotated.

    Parameters
    ----------
    image: Image
        The Image instance.

    Examples
    --------

    Create of False spectrogram:

    >>> im=Image('tests/data/reduc_20170605_028.fits', disperser_label='HoloPhAg')
    >>> N = parameters.CCD_IMSIZE
    >>> im.data = np.ones((N, N))
    >>> slope = -0.1
    >>> y = lambda x: slope * (x - 0.5*N) + 0.5*N
    >>> for x in np.arange(N):
    ...     im.data[int(y(x)), x] = 10
    ...     im.data[int(y(x))+1, x] = 10

    .. plot::

        from spectractor.extractor.images import Image
        import spectractor.parameters as parameters
        im=Image('tests/data/reduc_20170605_028.fits', disperser_label='HoloPhAg')
        N = parameters.CCD_IMSIZE
        im.data = np.ones((N, N))
        slope = -0.1
        y = lambda x: slope * (x - 0.5*N) + 0.5*N
        for x in np.arange(N):
            im.data[int(y(x)), x] = 10
            im.data[int(y(x))+1, x] = 10
        plt.imshow(im.data, origin='lower')
        plt.show()

    >>> im.target_pixcoords=(N//2, N//2)
    >>> parameters.DEBUG = True
    >>> turn_image(im)

    .. doctest::
        :hide:

        >>> assert im.data_rotated is not None
        >>> assert np.isclose(im.rotation_angle, np.arctan(slope)*180/np.pi, rtol=1e-2)

    .. plot::

        from spectractor.extractor.images import Image, turn_image
        import spectractor.parameters as parameters
        im=Image('tests/data/reduc_20170605_028.fits', disperser_label='HoloPhAg')
        N = parameters.CCD_IMSIZE
        im.data = np.ones((N, N))
        slope = -0.1
        y = lambda x: slope * (x - 0.5*N) + 0.5*N
        for x in np.arange(N):
            im.data[int(y(x)), x] = 10
            im.data[int(y(x))+1, x] = 10

        im.target_pixcoords=(N//2, N//2)
        turn_image(im)
        plt.imshow(im.data_rotated, origin='lower')
        plt.show()
    """
    image.rotation_angle = compute_rotation_angle_hessian(image, width_cut=parameters.YWINDOW,
                                                          angle_range=(parameters.ROT_ANGLE_MIN,
                                                                       parameters.ROT_ANGLE_MAX),
                                                          right_edge=parameters.CCD_IMSIZE - 200)
    image.header['ROTANGLE'] = image.rotation_angle
    image.my_logger.info(f'\n\tRotate the image with angle theta={image.rotation_angle:.2f} degree')
    image.data_rotated = np.copy(image.data)
    if not np.isnan(image.rotation_angle):
        image.data_rotated = ndimage.interpolation.rotate(image.data, image.rotation_angle,
                                                          prefilter=parameters.ROT_PREFILTER,
                                                          order=parameters.ROT_ORDER)
        image.stat_errors_rotated = np.sqrt(
            np.abs(ndimage.interpolation.rotate(image.stat_errors ** 2, image.rotation_angle,
                                                prefilter=parameters.ROT_PREFILTER,
                                                order=parameters.ROT_ORDER)))
        min_noz = np.min(image.stat_errors_rotated[image.stat_errors_rotated > 0])
        image.stat_errors_rotated[image.stat_errors_rotated <= 0] = min_noz
    if parameters.DEBUG:
        margin = 100
        y0 = int(image.target_pixcoords[1])
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=[8, 8])
        plot_image_simple(ax1, data=image.data[max(0, y0 - 2 * parameters.YWINDOW):
                                               min(y0 + 2 * parameters.YWINDOW, image.data.shape[0]),
                                    margin:-margin],
                          scale="symlog", title='Raw image (log10 scale)', units=image.units,
                          target_pixcoords=(image.target_pixcoords[0] - margin, 2 * parameters.YWINDOW), aspect='auto')
        ax1.plot([0, image.data.shape[0] - 2 * margin], [parameters.YWINDOW, parameters.YWINDOW], 'k-')
        plot_image_simple(ax2, data=image.data_rotated[max(0, y0 - 2 * parameters.YWINDOW):
                                                       min(y0 + 2 * parameters.YWINDOW, image.data.shape[0]),
                                    margin:-margin],
                          scale="symlog", title='Turned image (log10 scale)',
                          units=image.units, target_pixcoords=image.target_pixcoords_rotated, aspect='auto')
        ax2.plot([0, image.data_rotated.shape[0] - 2 * margin], [2 * parameters.YWINDOW, 2 * parameters.YWINDOW], 'k-')
        f.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            f.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'rotated_image.pdf'))


if __name__ == "__main__":
    import doctest

    doctest.testmod()

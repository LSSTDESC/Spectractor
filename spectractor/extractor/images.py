from astropy.coordinates import Angle, SkyCoord, Latitude
from astropy.io import fits
import astropy.units as units
from scipy import ndimage
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import copy

from spectractor import parameters
from spectractor.config import set_logger, load_config
from spectractor.extractor.targets import load_target
from spectractor.extractor.dispersers import Hologram
from spectractor.extractor.psf import Moffat
from spectractor.simulation.adr import hadec2zdpar
from spectractor.simulation.throughput import TelescopeTransmission
from spectractor.tools import (plot_image_simple, save_fits, load_fits, fit_poly1d, plot_compass_simple,
                               fit_poly1d_outlier_removal, weighted_avg_and_std,
                               fit_poly2d_outlier_removal, hessian_and_theta,
                               set_wcs_file_name, load_wcs_from_file, imgslice, rebin)


class Image(object):
    """ The image class contains all the features necessary to load an image and extract a spectrum.

    Attributes
    ----------
    my_logger: logging
        Logging object
    file_name: str
        The file name of the exposure.
    units: str
        Units of the image.
    data: array
        Image 2D array in self.units units.
    stat_errors: array
        Image 2D uncertainty array in self.units units.
    target_pixcoords: array
        Target position [x,y] in the image in pixels.
    data_rotated: array
        Rotated image 2D array in self.units units.
    stat_errors_rotated: array
        Rotated image 2D uncertainty array in self.units units.
    target_pixcoords_rotated: array
        Target position [x,y] in the rotated image in pixels.
    date_obs: str
        Date of the observation.
    airmass: float
        Airmass of the current target.
    expo: float
        Exposure time in seconds.
    disperser_label: str
        Label of the disperser.
    filter_label: str
        Label of the filter.
    target_label: str:
        Label of the current target.
    rotation_angle: float
        Dispersion axis angle in the image in degrees, positive if anticlockwise.
    parallactic_angle: float
        Parallactic angle in degrees.
    header: Fits.Header
        FITS file header.
    disperser: Disperser
        Disperser instance that describes the disperser.
    target: Target
        Target instance that describes the current target.
    ra: float
        Right ascension coordinate of the current exposure.
    dec: float
        Declination coordinate of the current exposure.
    hour_angle: float
        Hour angle coordinate of the current exposure.
    temperature: float
        Outside temperature in Celsius degrees.
    pressure: float
        Outside pressure in hPa.
    humidity: float
        Outside relative humidity in fraction of one.
    saturation: float
        Level of saturation in the image in image units.
    target_star2D: PSF
        PSF instance fitted on the current target.
    target_bkgd2D: callable
        Function that models the background behind the current target.

    """

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
        if not os.path.isfile(file_name) and parameters.CALLING_CODE != 'LSST_DM':
            raise FileNotFoundError(f"File {file_name} does not exist.")
        self.file_name = file_name
        self.units = 'ADU'
        self.expo = -1
        self.airmass = -1
        self.date_obs = None
        self.disperser = None
        self.disperser_label = disperser_label
        self.target_label = target_label
        self.target_guess = None
        self.filter_label = ""
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

        self.ra = None
        self.dec = None
        self.hour_angle = None
        self.temperature = 0
        self.pressure = 0
        self.humidity = 0

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

        if self.filter_label != "" and "empty" not in self.filter_label.lower():
            t = TelescopeTransmission(filter_label=self.filter_label)
            t.reset_lambda_range(transmission_threshold=1e-4)

        if self.target_label != "":
            self.target = load_target(self.target_label, verbose=parameters.VERBOSE)
            self.header['REDSHIFT'] = str(self.target.redshift)

    def rebin(self):
        """Rebin the image and reset some related parameters.

        Examples
        --------
        >>> parameters.CCD_REBIN = 2
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> im.target_guess = [810, 590]
        >>> im.data.shape
        (2048, 2048)
        >>> im.rebin()
        >>> im.data.shape
        (1024, 1024)
        >>> im.stat_errors.shape
        (1024, 1024)
        >>> im.target_guess
        array([405., 295.])
        """
        new_shape = np.asarray(self.data.shape) // parameters.CCD_REBIN
        self.data = rebin(self.data, new_shape)
        self.stat_errors = np.sqrt(rebin(self.stat_errors ** 2, new_shape))
        if self.target_guess is not None:
            self.target_guess = np.asarray(self.target_guess) / parameters.CCD_REBIN

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
        self.header['OUTTEMP'] = self.temperature
        self.header['OUTPRESS'] = self.pressure
        self.header['OUTHUM'] = self.humidity
        self.header['CCDREBIN'] = parameters.CCD_REBIN

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
            raise AttributeError(f'Noise must be estimated on an image in ADU units. '
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
            raise AttributeError(f"Noise map must be in ADU units to be plotted and analyzed. "
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
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'uncertainty_map.pdf'))
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()

    def compute_parallactic_angle(self):
        """Compute the parallactic angle.

        Script from A. Guyonnet.
        """
        latitude = Latitude(parameters.OBS_LATITUDE, unit=units.deg)
        ha = self.hour_angle
        dec = self.dec
        # parallactic_angle = Angle(np.arctan2(np.sin(ha), (np.cos(dec) * np.tan(latitude) - np.sin(dec) * np.cos(ha))))
        zenithal_distance, parallactic_angle = hadec2zdpar(ha, dec, latitude, deg=False)
        self.parallactic_angle = parallactic_angle.value * 180 / np.pi
        self.header['PARANGLE'] = self.parallactic_angle
        self.header.comments['PARANGLE'] = 'parallactic angle in degree'
        return self.parallactic_angle

    def plot_image(self, ax=None, scale="lin", title="", units="", plot_stats=False,
                   target_pixcoords=None, figsize=(7.3, 6), aspect=None, vmin=None, vmax=None,
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
        >>> im = Image('tests/data/reduc_20170605_028.fits', config="./config/ctio.ini")
        >>> im.plot_image(target_pixcoords=[820, 580], scale="symlog")
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
        if parameters.OBS_OBJECT_TYPE == "STAR":
            plot_compass_simple(ax, self.parallactic_angle, arrow_size=0.1, origin=[0.15, 0.15])
        plt.legend()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            plt.gcf().savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'image.pdf'))
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()

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
    if "dia" not in image.header['FILTER1'].lower():
        image.filter_label = image.header['FILTER1']
    image.disperser_label = image.header['FILTER2']
    image.ra = Angle(image.header['RA'], unit="hourangle")
    image.dec = Angle(image.header['DEC'], unit="deg")
    image.hour_angle = Angle(image.header['HA'], unit="hourangle")
    image.temperature = image.header['OUTTEMP']
    image.pressure = image.header['OUTPRESS']
    image.humidity = image.header['OUTHUM']

    parameters.CCD_IMSIZE = int(image.header['XLENGTH'])
    parameters.CCD_PIXEL2ARCSEC = float(image.header['XPIXSIZE'])
    if image.header['YLENGTH'] != parameters.CCD_IMSIZE:
        image.my_logger.warning(
            f'\n\tImage rectangular: X={parameters.CCD_IMSIZE:d} pix, Y={image.header["YLENGTH"]:d} pix')
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
        raise KeyError(f'mm is absent from ZPOS key in XYZ header. Had {hdus["XYZ"].header.comments["ZPOS"]}'
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
    hdu_list = fits.open(image.file_name)
    image.header = hdu_list[0].header
    image.data = hdu_list[1].data.astype(np.float64)
    hdu_list.close()  # need to free allocation for file descripto
    image.date_obs = image.header['DATE']
    image.expo = float(image.header['EXPTIME'])
    if "empty" not in image.header['FILTER'].lower():
        image.filter_label = image.header['FILTER']
    # transformations so that stars are like in Stellarium up to a rotation
    # with spectrogram nearly horizontal and on the right of central star
    image.data = image.data.T[::-1, ::-1]
    if image.header["AMSTART"] is not None:
        image.airmass = 0.5 * (float(image.header["AMSTART"]) + float(image.header["AMEND"]))
    else:
        image.airmass = float(image.header['AIRMASS'])
    image.my_logger.info('\n\tImage loaded')
    # compute CCD gain map
    image.gain = float(parameters.CCD_GAIN) * np.ones_like(image.data)
    parameters.CCD_IMSIZE = image.data.shape[1]
    image.disperser_label = image.header['GRATING']
    image.ra = Angle(image.header['RA'], unit="deg")
    image.dec = Angle(image.header['DEC'], unit="deg")
    if 'HASTART' in image.header and image.header['HASTART'] is not None:
        image.hour_angle = Angle(image.header['HASTART'], unit="hourangle")
    else:
        image.hour_angle = Angle(image.header['HA'], unit="deg")
    if 'AIRTEMP' in image.header:
        image.temperature = image.header['AIRTEMP']
    else:
        image.temperature = 10
    if 'PRESSURE' in image.header:
        image.pressure = image.header['PRESSURE']
    else:
        image.pressure = 743
    if 'HUMIDITY' in image.header:
        image.humidity = image.header['HUMIDITY']
    else:
        image.humidity = 40
    if 'adu' in image.header['BUNIT']:
        image.units = 'ADU'
    parameters.OBS_CAMERA_ROTATION = 90 - float(image.header["ROTPA"])
    if parameters.OBS_CAMERA_ROTATION > 360:
        parameters.OBS_CAMERA_ROTATION -= 360
    if parameters.OBS_CAMERA_ROTATION < -360:
        parameters.OBS_CAMERA_ROTATION += 360
    if "CD2_1" in hdu_list[1].header:
        rotation_wcs = 180 / np.pi * np.arctan2(hdu_list[1].header["CD2_1"], hdu_list[1].header["CD1_1"]) + 90
        if not np.isclose(rotation_wcs % 360, parameters.OBS_CAMERA_ROTATION % 360, atol=2):
            image.my_logger.warning(f"\n\tWCS rotation angle is {rotation_wcs} degree while "
                                    f"parameters.OBS_CAMERA_ROTATION={parameters.OBS_CAMERA_ROTATION} degree. "
                                    f"\nBoth differs by more than 2 degree... bug ?")
    parameters.OBS_ALTITUDE = float(image.header['OBS-ELEV']) / 1000
    parameters.OBS_LATITUDE = image.header['OBS-LAT']
    image.read_out_noise = 8.5 * np.ones_like(image.data)
    image.target_label = image.header["OBJECT"].replace(" ", "")
    if "OBJECTX" in image.header:
        image.target_guess = [parameters.CCD_IMSIZE - float(image.header["OBJECTY"]),
                              parameters.CCD_IMSIZE - float(image.header["OBJECTX"])]
    image.disperser_label = image.header["GRATING"]
    parameters.DISTANCE2CCD = 115 + float(image.header["LINSPOS"])  # mm
    image.compute_parallactic_angle()


def find_target(image, guess=None, rotated=False, widths=[parameters.XWINDOW, parameters.YWINDOW]):
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
    widths: (int, int)
        Two parameter array to define the width of the cropped image (default: [parameters.XWINDOW, parameters.YWINDOW])

    Returns
    -------
    x0: float
        The x position of the target.
    y0: float
        The y position of the target.

    Examples
    --------
    >>> im = Image('tests/data/reduc_20170605_028.fits', target_label="PNG321.0+3.9")
    >>> im.plot_image()
    >>> guess = [820, 580]
    >>> parameters.VERBOSE = True
    >>> parameters.DEBUG = True
    >>> parameters.SPECTRACTOR_FIT_TARGET_CENTROID = "fit"
    >>> find_target(im, guess)  #doctest: +ELLIPSIS
    [816.8... 587.3...]
    >>> parameters.SPECTRACTOR_FIT_TARGET_CENTROID = "WCS"
    >>> find_target(im, guess)  #doctest: +ELLIPSIS
    [816.9... 587.1...]
    >>> parameters.SPECTRACTOR_FIT_TARGET_CENTROID = "guess"
    >>> find_target(im, guess)
    [820, 580]
    """
    my_logger = set_logger(__name__)
    target_pixcoords = [-1, -1]
    theX = -1
    theY = -1
    if parameters.SPECTRACTOR_FIT_TARGET_CENTROID == "WCS" and not rotated:
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
                theX, theY = target_pixcoords / parameters.CCD_REBIN
            sub_image_subtracted, x0, y0, Dx, Dy, sub_errors = find_target_init(image=image, guess=[theX, theY],
                                                                                rotated=rotated, widths=widths)
            sub_image_x0 = theX - x0 + Dx
            sub_image_y0 = theX - y0 + Dy
            if parameters.DEBUG:
                plt.figure(figsize=(5, 5))
                plot_image_simple(plt.gca(), data=sub_image_subtracted, scale="lin", title="", units=image.units,
                                  target_pixcoords=[theX - x0 + Dx, theX - x0 + Dx])
                plt.show()
            if parameters.PdfPages:
                parameters.PdfPages.savefig()
        else:
            my_logger.info(f"\n\tNo WCS {wcs_file_name} available, use 2D fit to find target pixel position.")
    if parameters.SPECTRACTOR_FIT_TARGET_CENTROID == "fit" or rotated:
        if target_pixcoords[0] == -1 and target_pixcoords[1] == -1:
            if guess is None:
                raise ValueError(f"Guess parameter must be set if WCS solution is not found.")
            Dx, Dy = widths
            theX, theY = guess
            if rotated:
                guess2 = find_target_after_rotation(image)
                x0 = int(guess2[0])
                y0 = int(guess2[1])
                guess = [x0, y0]
            niter = 2
            sub_image_subtracted, x0, y0, Dx, Dy, sub_errors = find_target_init(image=image, guess=guess, rotated=rotated,
                                                                                widths=(Dx, Dy))
            sub_image_x0, sub_image_y0 = x0, y0
            for i in range(niter):
                # find the target
                # try:
                sub_image_x0, sub_image_y0 = find_target_Moffat2D(image, sub_image_subtracted, sub_errors=sub_errors)
                # except (Exception, ValueError):
                #     image.target_star2D = None
                #     avX, avY = find_target_2DprofileASTROPY(image, sub_image_subtracted, sub_errors=sub_errors)
                # compute target position
                theX = x0 - Dx + sub_image_x0
                theY = y0 - Dy + sub_image_y0
                # crop for next iteration
                if i < niter - 1:
                    Dx = Dx // (i + 2)
                    Dy = Dy // (i + 2)
                    x0 = int(theX)
                    y0 = int(theY)
                    NY, NX = sub_image_subtracted.shape
                    sub_image_subtracted = sub_image_subtracted[
                                           max(0, int(sub_image_y0) - Dy):min(NY, int(sub_image_y0) + Dy),
                                           max(0, int(sub_image_x0) - Dx):min(NX, int(sub_image_x0) + Dx)]
                    sub_errors = sub_errors[max(0, int(sub_image_y0) - Dy):min(NY, int(sub_image_y0) + Dy),
                                 max(0, int(sub_image_x0) - Dx):min(NX, int(sub_image_x0) + Dx)]
                    if int(sub_image_x0) - Dx < 0:
                        Dx = int(sub_image_x0)
                    if int(sub_image_y0) - Dy < 0:
                        Dy = int(sub_image_y0)
        else:
            Dx, Dy = widths
            sub_image_subtracted, x0, y0, Dx, Dy, sub_errors = find_target_init(image=image, guess=target_pixcoords,
                                                                                rotated=rotated, widths=(Dx, Dy))
            theX, theY = target_pixcoords
            sub_image_x0 = target_pixcoords[0] - x0 + Dx
            sub_image_y0 = target_pixcoords[1] - y0 + Dy
    elif parameters.SPECTRACTOR_FIT_TARGET_CENTROID == "guess":
        Dx, Dy = widths
        sub_image_subtracted, x0, y0, Dx, Dy, sub_errors = find_target_init(image=image, guess=target_pixcoords,
                                                                            rotated=rotated, widths=(Dx, Dy))
        theX, theY = guess
        sub_image_x0 = theX - x0 + Dx
        sub_image_y0 = theY - y0 + Dy
    elif parameters.SPECTRACTOR_FIT_TARGET_CENTROID == "WCS" and not rotated:
        pass
    else:
        raise ValueError(f"For unrotated images, parameters.SPECTRACTOR_FIT_TARGET_CENTROID muste be either: "
                         f"guess, fit or WCS. Got {parameters.SPECTRACTOR_FIT_TARGET_CENTROID}.")
    image.my_logger.info(f'\n\tX,Y target position in pixels: {theX:.3f},{theY:.3f}')
    if rotated:
        image.target_pixcoords_rotated = [theX, theY]
    else:
        image.target.image = sub_image_subtracted
        image.target.image_x0 = sub_image_x0
        image.target.image_y0 = sub_image_y0
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
    widths: (int, int)
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
        ax2.set_xlabel(parameters.PLOT_XLABEL)
        ax2.legend(loc=1)
        ax3.plot(Y, profile_Y_raw, 'r-', lw=2)
        ax3.plot(Y, bkgd_Y(Y), 'g--', lw=2, label='bkgd')
        ax3.axvline(avY, color='b', linestyle='-', label='new', lw=2)
        ax3.grid(True)
        ax3.set_xlabel(parameters.PLOT_YLABEL)
        ax3.legend(loc=1)
        f.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            f.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'namethisplot1.pdf'))
        if parameters.PdfPages:
            parameters.PdfPages.savefig()
    return avX, avY


def find_target_Moffat2D(image, sub_image_subtracted, sub_errors=None):
    """
    Find precisely the position of the targeted object fitting a Moffat PSF model.
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
    >>> x1, y1 = find_target_Moffat2D(im, sub_image_subtracted, sub_errors=sub_errors)

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
    psf = Moffat(clip=True)
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
        Y, X = np.mgrid[:NY, :NX]
        star2D = psf.evaluate(pixels=np.array([X, Y]))
        plot_image_simple(ax1, data=sub_image_subtracted, scale="lin", title="", units=image.units,
                          target_pixcoords=[new_avX, new_avY], vmin=vmin, vmax=vmax)
        ax1.legend(loc=1)

        ax1.text(0.05, 0.05, f'Data', color="white",
                 horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes)
        ax2.text(0.05, 0.05, f'Background+Moffat2D model', color="white",
                 horizontalalignment='left', verticalalignment='bottom', transform=ax2.transAxes)
        ax3.text(0.05, 0.05, f'Residuals', color="white",
                 horizontalalignment='left', verticalalignment='bottom', transform=ax3.transAxes)
        plot_image_simple(ax2, data=star2D, scale="lin", title="",
                          units=f'{image.units}', vmin=vmin, vmax=vmax)
        plot_image_simple(ax3, data=sub_image_subtracted - star2D, scale="lin", title="",
                          units=f'{image.units}', target_pixcoords=[new_avX, new_avY], vmin=vmin, vmax=vmax)
        ax3.legend(loc=1)

        f.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            f.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'order0_centroid_fit.pdf'))
    return new_avX, new_avY


def compute_rotation_angle_hessian(image, angle_range=(-10, 10), width_cut=parameters.YWINDOW,
                                   edges=(0, parameters.CCD_IMSIZE),
                                   margin_cut=12, pixel_fraction=0.01):
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
    edges: (int, int)
        Minimum and maximum pixel on the right edge (default: (0, parameters.CCD_IMSIZE)).
    margin_cut: int
        After computing the Hessian, to avoid bad values on the edges the function cut on the
        edge of image margin_cut pixels (default: 12).
    pixel_fraction: float
        Minimum pixel fraction to keep after thresholding the lambda minus map (default: 0.01).

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
    ...     im.data[int(y(x)), x] = 100
    ...     im.data[int(y(x))+1, x] = 100
    >>> im.target_pixcoords=(N//2, N//2)
    >>> parameters.DEBUG = True
    >>> theta = compute_rotation_angle_hessian(im)

    .. doctest::
        :hide:

        >>> assert np.isclose(theta, np.arctan(slope)*180/np.pi, rtol=1e-1)

    """
    x0, y0 = np.array(image.target_pixcoords).astype(int)
    # extract a region
    left_edge, right_edge = edges
    data = np.copy(image.data[y0 - width_cut:y0 + width_cut, left_edge:right_edge])
    lambda_plus, lambda_minus, theta = hessian_and_theta(data, margin_cut)
    # thresholds
    lambda_threshold = np.min(lambda_minus)
    mask = np.where(lambda_minus > lambda_threshold)
    theta_mask = np.copy(theta)
    theta_mask[mask] = np.nan
    minimum_pixels = pixel_fraction * 2 * width_cut * right_edge
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
        gs_kw = dict(width_ratios=[3, 1], height_ratios=[1])
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 3), gridspec_kw=gs_kw)
        xindex = np.arange(data.shape[1])
        x_new = np.linspace(xindex.min(), xindex.max(), 50)
        y_new = width_cut - margin_cut - 3 + (x_new - x0) * np.tan(theta_median * np.pi / 180.)
        cmap = copy.copy(cm.get_cmap('bwr'))
        cmap.set_bad(color='lightgrey')
        im = ax1.imshow(theta_mask, origin='lower', cmap=cmap, aspect='auto', vmin=angle_range[0], vmax=angle_range[1])
        cb = plt.colorbar(im, ax=ax1)
        cb.set_label(parameters.PLOT_ROT_LABEL, labelpad=-10)
        ax1.plot(x_new, y_new, 'k-', label=rf"Mean dispersion axis: $\varphi_d$={theta_median:.2f}°")
        ax1.set_ylim(0, theta_mask.shape[0])
        ax1.set_xlim(0, theta_mask.shape[1])
        ax1.legend()
        ax1.set_xlabel(parameters.PLOT_XLABEL)
        ax1.set_ylabel(parameters.PLOT_YLABEL)
        ax1.grid(True)
        ax2.hist(theta_hist, bins=int(np.sqrt(len(theta_hist))))
        ax2.axvline(theta_median, color='k')
        ax2.set_xlabel(parameters.PLOT_ROT_LABEL)
        ax2.grid()
        f.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            f.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'rotation_hessian.pdf'))
        if parameters.PdfPages:
            parameters.PdfPages.savefig()
    return theta_median


def turn_image(image):
    """Compute the rotation angle using the Hessian algorithm and turn the image.

    The results are stored in Image.data_rotated and Image.stat_errors_rotated.

    Parameters
    ----------
    image: Image
        The Image instance.

    Returns
    -------
    rotation_angle: float
        Rotation angle in degree.

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
    >>> parameters.SPECTRACTOR_COMPUTE_ROTATION_ANGLE = False
    >>> turn_image(im)
    0
    >>> parameters.SPECTRACTOR_COMPUTE_ROTATION_ANGLE = "disperser"
    >>> turn_image(im)
    -1.915
    >>> parameters.SPECTRACTOR_COMPUTE_ROTATION_ANGLE = "hessian"
    >>> turn_image(im)  #doctest: +ELLIPSIS
    -5.90...

    .. doctest::
        :hide:

        >>> assert im.data_rotated is not None
        >>> assert np.isclose(im.rotation_angle, np.arctan(slope)*180/np.pi, rtol=5e-2)

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
    if parameters.SPECTRACTOR_COMPUTE_ROTATION_ANGLE == "hessian":
        image.rotation_angle = compute_rotation_angle_hessian(image, width_cut=parameters.YWINDOW,
                                                              angle_range=(parameters.ROT_ANGLE_MIN,
                                                                           parameters.ROT_ANGLE_MAX),
                                                              edges=(0, parameters.CCD_IMSIZE))
    elif parameters.SPECTRACTOR_COMPUTE_ROTATION_ANGLE == "disperser":
        image.rotation_angle = image.disperser.theta_tilt
    elif parameters.SPECTRACTOR_COMPUTE_ROTATION_ANGLE is False:
        image.rotation_angle = 0
    else:
        raise ValueError(f"Unknown method for rotation angle computation: choose among False, disperser, hessian. "
                         f"Got {parameters.SPECTRACTOR_COMPUTE_ROTATION_ANGLE}")
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
        margin = 100 // parameters.CCD_REBIN
        y0 = int(image.target_pixcoords[1])
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=[8, 8])
        plot_image_simple(ax1, data=image.data[max(0, y0 - 2 * parameters.YWINDOW):
                                               min(y0 + 2 * parameters.YWINDOW, image.data.shape[0]),
                                    margin:-margin],
                          scale="symlog", title='Raw image (log10 scale)', units=image.units,
                          target_pixcoords=(image.target_pixcoords[0] - margin, 2 * parameters.YWINDOW), aspect='auto')
        if parameters.OBS_OBJECT_TYPE == "STAR":
            plot_compass_simple(ax1, image.parallactic_angle, arrow_size=0.1, origin=[0.15, 0.15])
        ax2.axhline(parameters.YWINDOW, color='k')
        plot_image_simple(ax2, data=image.data_rotated[max(0, y0 - 2 * parameters.YWINDOW):
                                                       min(y0 + 2 * parameters.YWINDOW, image.data.shape[0]),
                                    margin:-margin],
                          scale="symlog", title='Turned image (log10 scale)',
                          units=image.units, target_pixcoords=image.target_pixcoords_rotated, aspect='auto')
        ax2.axhline(2 * parameters.YWINDOW, color='k')
        f.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            f.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'rotated_image.pdf'))
    return image.rotation_angle


if __name__ == "__main__":
    import doctest

    doctest.testmod()

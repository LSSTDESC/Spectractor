from scipy.signal import argrelextrema, savgol_filter
from scipy.interpolate import interp1d
from astropy.io import fits
from scipy import integrate
from iminuit import Minuit
import matplotlib.pyplot as plt
import numpy as np
import os

from spectractor import parameters
from spectractor.config import set_logger, load_config, apply_rebinning_to_parameters
from spectractor.extractor.dispersers import Hologram
from spectractor.extractor.targets import load_target
from spectractor.tools import (ensure_dir, load_fits, plot_image_simple,
                               find_nearest, plot_spectrum_simple, fit_poly1d_legendre, gauss,
                               rescale_x_for_legendre, fit_multigauss_and_bgd, multigauss_and_bgd)
from spectractor.extractor.psf import load_PSF
from spectractor.extractor.chromaticpsf import ChromaticPSF
from spectractor.simulation.adr import adr_calib, flip_and_rotate_adr_to_image_xy_coordinates
from spectractor.simulation.throughput import TelescopeTransmission


class Spectrum:
    """ Class used to store information and methods relative to spectra and their extraction.

    Attributes
    ----------
    my_logger: logging
        Logging object
    fast_load: bool
        If True, only load the spectrum but not the spectrogram.
    units: str
        Units of the spectrum.
    lambdas: array
        Spectrum wavelengths in nm.
    data: array
        Spectrum amplitude array in self.units units.
    err: array
        Spectrum amplitude uncertainties in self.units units.
    cov_matrix: array
        Spectrum amplitude covariance matrix between wavelengths in self.units units.
    lambdas_binwidths: array
        Bin widths of the wavelength array in nm.
    lambdas_order2: array
        Spectrum wavelengths for order 2 contamination in nm.
    data_order2: array
        Spectrum amplitude array  for order 2 contamination in self.units units.
    err_order2: array
        Spectrum amplitude uncertainties  for order 2 contamination in self.units units.
    lambda_ref: float
        Reference wavelength for ADR computations in nm.
    order: int
        Index of the diffraction order.
    x0: array
        Target position [x,y] in the image in pixels.
    psf: PSF
        PSF instance to model the spectrum PSF.
    chromatic_psf: ChromaticPSF
        ChromaticPSF object that contains data on the PSF shape and evolution in wavelength.
    date_obs: str
        Date of the observation.
    airmass: float
        Airmass of the current target.
    expo: float
        Exposure time in seconds.
    disperser_label: str
        Label of the disperser.
    filter_label: str:
        Label of the filter.
    rotation_angle: float
        Dispersion axis angle in the image in degrees, positive if anticlockwise.
    parallactic_angle: float
        Parallactic angle in degrees.
    lines: Lines
        Lines instance that contains data on the emission or absorption lines to be searched and fitted in the spectrum.
    header: Fits.Header
        FITS file header.
    disperser: Disperser
        Disperser instance that describes the disperser.
    target: Target
        Target instance that describes the current exposure.
    dec: float
        Declination coordinate of the current exposure.
    hour_angle float
        Hour angle coordinate of the current exposure.
    temperature: float
        Outside temperature in Celsius degrees.
    pressure: float
        Outside pressure in hPa.
    humidity: float
        Outside relative humidity in fraction of one.
    spectrogram: array
        Spectrogram 2D image in image units.
    spectrogram_bgd: array
        Estimated 2D background fitted below the spectrogram in image units.
    spectrogram_bgd_rms: array
        Estimated 2D background RMS fitted below the spectrogram in image units.
    spectrogram_err: array
        Estimated 2D background uncertainty fitted below the spectrogram in image units.
    spectrogram_fit: array
        Best fitting model of the spectrogram in image units.
    spectrogram_residuals: array
        Residuals between the spectrogram data and the best fitting model of the spectrogram in image units.
    spectrogram_x0: float
        Relative position of the target in the spectrogram array along the x axis.
    spectrogram_y0: float
        Relative position of the target in the spectrogram array along the y axis.
    spectrogram_xmin: int
        Left index of the spectrogram crop in the image.
    spectrogram_xmax: int
        Right index of the spectrogram crop in the image.
    spectrogram_ymin: int
        Bottom index of the spectrogram crop in the image.
    spectrogram_ymax: int
        Top index of the spectrogram crop in the image.
    spectrogram_deg: int
        Degree of the polynomial functions to model wavelength evolutions of the PSF parameters.
    spectrogram_saturation: float
        Level of saturation in the spectrogram in image units.
    spectrogram_Nx: int
        Size of the spectrogram along the x axis.
    spectrogram_Ny: int
        Size of the spectrogram along the y axis.
    """

    def __init__(self, file_name="", image=None, order=1, target=None, config="", fast_load=False):
        """ Class used to store information and methods relative to spectra and their extraction.

        Parameters
        ----------
        file_name: str, optional
            Path to the spectrum file (default: "").
        image: Image, optional
            Image object from which to create the Spectrum object:
            copy the information from the Image header (default: None).
        order: int
            Order of the spectrum (default: 1)
        target: Target, optional
            Target object if provided (default: None)
        config: str, optional
            A config file name to load some parameter values for a given instrument (default: "").
        fast_load: bool, optional
            If True, only the spectrum is loaded (not the PSF nor the spectrogram data) (default: False).

        Examples
        --------
        Load a spectrum from a fits file
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
        >>> print(s.order)
        1
        >>> print(s.target.label)
        PNG321.0+3.9
        >>> print(s.disperser_label)
        HoloPhAg

        Load a spectrum from a fits image file
        >>> from spectractor.extractor.images import Image
        >>> image = Image('tests/data/reduc_20170605_028.fits', target_label='PNG321.0+3.9')
        >>> s = Spectrum(image=image)
        >>> print(s.target.label)
        PNG321.0+3.9
        """
        self.fast_load = fast_load
        self.my_logger = set_logger(self.__class__.__name__)
        if config != "":
            load_config(config)
        self.target = target
        self.data = None
        self.err = None
        self.cov_matrix = None
        self.x0 = None
        self.pixels = None
        self.lambdas = None
        self.lambdas_binwidths = None
        self.lambdas_indices = None
        self.lambda_ref = None
        self.order = order
        self.chromatic_psf = None
        self.filter_label = ""
        self.filters = None
        self.units = 'ADU/s'
        self.gain = parameters.CCD_GAIN
        self.psf = load_PSF(psf_type="Moffat", target=self.target)
        self.chromatic_psf = ChromaticPSF(self.psf, Nx=1, Ny=1, deg=1, saturation=1)
        self.rotation_angle = 0
        self.parallactic_angle = None
        self.spectrogram = None
        self.spectrogram_bgd = None
        self.spectrogram_bgd_rms = None
        self.spectrogram_err = None
        self.spectrogram_residuals = None
        self.spectrogram_fit = None
        self.spectrogram_x0 = None
        self.spectrogram_y0 = None
        self.spectrogram_xmin = None
        self.spectrogram_xmax = None
        self.spectrogram_ymin = None
        self.spectrogram_ymax = None
        self.spectrogram_deg = None
        self.spectrogram_saturation = None
        self.spectrogram_Nx = None
        self.spectrogram_Ny = None
        self.lambdas_order2 = None
        self.data_order2 = None
        self.err_order2 = None
        self.filename = file_name
        if file_name != "":
            self.load_spectrum(file_name)
        if image is not None:
            self.header = image.header
            self.date_obs = image.date_obs
            self.airmass = image.airmass
            self.expo = image.expo
            self.filters = image.filters
            self.filter_label = image.filter_label
            self.disperser_label = image.disperser_label
            self.disperser = image.disperser
            self.target = image.target
            self.lines = self.target.lines
            self.x0 = image.target_pixcoords
            self.target_pixcoords = image.target_pixcoords
            self.target_pixcoords_rotated = image.target_pixcoords_rotated
            self.units = image.units
            self.gain = image.gain
            self.rotation_angle = image.rotation_angle
            self.my_logger.info('\n\tSpectrum info copied from image')
            self.dec = image.dec
            self.hour_angle = image.hour_angle
            self.temperature = image.temperature
            self.pressure = image.pressure
            self.humidity = image.humidity
            self.parallactic_angle = image.parallactic_angle
            self.adr_params = [self.dec, self.hour_angle, self.temperature, self.pressure,
                               self.humidity, self.airmass]
        self.load_filter()

    def convert_from_ADUrate_to_flam(self):
        """Convert units from ADU/s to erg/s/cm^2/nm.
        The SED is supposed to be in flam units ie erg/s/cm^2/nm

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
        >>> s.convert_from_ADUrate_to_flam()

        .. doctest::
            :hide:

            >>> assert np.max(s.data) < 1e-2
            >>> assert np.max(s.err) < 1e-2

        """
        if self.units == 'erg/s/cm$^2$/nm' or self.units == "flam":
            self.my_logger.warning(f"You ask to convert spectrum already in {self.units}"
                                   f" in erg/s/cm^2/nm... check your code ! Skip the instruction.")
            return
        ldl = parameters.FLAM_TO_ADURATE * self.lambdas * np.abs(self.lambdas_binwidths)
        self.data /= ldl
        if self.err is not None:
            self.err /= ldl
        if self.cov_matrix is not None:
            ldl_mat = np.outer(ldl, ldl)
            self.cov_matrix /= ldl_mat
        if self.data_order2 is not None:
            ldl_2 = parameters.FLAM_TO_ADURATE * self.lambdas_order2 * np.abs(np.gradient(self.lambdas_order2))
            self.data_order2 /= ldl_2
            self.err_order2 /= ldl_2
        self.units = 'erg/s/cm$^2$/nm'

    def convert_from_flam_to_ADUrate(self):
        """Convert units from erg/s/cm^2/nm to ADU/s.
        The SED is supposed to be in flam units ie erg/s/cm^2/nm

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
        >>> s.convert_from_flam_to_ADUrate()

        .. doctest::
            :hide:

            >>> assert np.max(s.data) > 1e-2
            >>> assert np.max(s.err) > 1e-2

        """
        if self.units == "ADU/s":
            self.my_logger.warning(f"You ask to convert spectrum already in {self.units} in ADU/s... check your code ! "
                                   f"Skip the instruction")
            return
        ldl = parameters.FLAM_TO_ADURATE * self.lambdas * np.abs(self.lambdas_binwidths)
        self.data *= ldl
        if self.err is not None:
            self.err *= ldl
        if self.cov_matrix is not None:
            ldl_mat = np.outer(ldl, ldl)
            self.cov_matrix *= ldl_mat
        if self.data_order2 is not None:
            ldl_2 = parameters.FLAM_TO_ADURATE * self.lambdas_order2 * np.abs(np.gradient(self.lambdas_order2))
            self.data_order2 *= ldl_2
            self.err_order2 *= ldl_2
        self.units = 'ADU/s'

    def load_filter(self):
        """Load filter properties and set relevant LAMBDA_MIN and LAMBDA_MAX values.

        Examples
        --------
        >>> s = Spectrum()
        >>> s.filter_label = 'FGB37'
        >>> s.load_filter()

        .. doctest::
            :hide:

            >>> assert np.isclose(parameters.LAMBDA_MIN, 300)
            >>> assert np.isclose(parameters.LAMBDA_MAX, 760)

        """
        if self.filter_label != "" and "empty" not in self.filter_label.lower():
            t = TelescopeTransmission(filter_label=self.filter_label)
            t.reset_lambda_range(transmission_threshold=1e-4)

    def plot_spectrum(self, ax=None, xlim=None, live_fit=False, label='', force_lines=False):
        """Plot spectrum with emission and absorption lines.

        Parameters
        ----------
        ax: Axes, optional
            Axes instance (default: None).
        label: str
            Label for the legend (default: '').
        xlim: list, optional
            List of minimum and maximum abscisses (default: None)
        live_fit: bool, optional
            If True the spectrum is plotted in live during the fitting procedures
            (default: False).
        force_lines: bool
            Force the over plot of vertical lines for atomic lines if set to True (default: False).

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170530_134_spectrum.fits')
        >>> s.plot_spectrum(xlim=[500,900], live_fit=False, force_lines=True)
        """
        if ax is None:
            plt.figure(figsize=[12, 6])
            ax = plt.gca()
        if label == '':
            label = f'Order {self.order:d} spectrum\n' \
                    r'$D_{\mathrm{CCD}}=' \
                    rf'{self.disperser.D:.2f}\,$mm'
        if self.x0 is not None:
            label += rf', $x_0={self.x0[0]:.2f}\,$pix'
        title = self.target.label
        if self.lambdas_order2 is not None:
            distance = self.disperser.grating_lambda_to_pixel(self.lambdas_order2, self.x0, order=2)
            lambdas_order2_contamination = self.disperser.grating_pixel_to_lambda(distance, self.x0, order=1)
            data_order2_contamination = self.data_order2 * (self.lambdas_order2 * np.gradient(self.lambdas_order2)) \
                                        / (lambdas_order2_contamination * np.gradient(lambdas_order2_contamination))
            if np.sum(data_order2_contamination) / np.sum(self.data) > 0.01:
                data_interp = interp1d(self.lambdas, self.data, kind="linear", fill_value="0", bounds_error=False)
                plot_spectrum_simple(ax, lambdas_order2_contamination,
                                     data_interp(lambdas_order2_contamination) + data_order2_contamination,
                                     data_err=None, xlim=xlim, label='Order 2 contamination', linestyle="--", lw=1)
        plot_spectrum_simple(ax, self.lambdas, self.data, data_err=self.err, xlim=xlim, label=label,
                             title=title, units=self.units)
        if len(self.target.spectra) > 0:
            for k in range(len(self.target.spectra)):
                plot_indices = np.logical_and(self.target.wavelengths[k] > np.min(self.lambdas),
                                              self.target.wavelengths[k] < np.max(self.lambdas))
                s = self.target.spectra[k] / np.max(self.target.spectra[k][plot_indices]) * np.max(self.data)
                ax.plot(self.target.wavelengths[k], s, lw=2, label='Tabulated spectra #%d' % k)
        if self.lambdas is not None:
            self.lines.plot_detected_lines(ax, print_table=parameters.VERBOSE)
        if self.lambdas is not None and self.lines is not None:
            self.lines.plot_atomic_lines(ax, fontsize=12, force=force_lines)
        ax.legend(loc='best')
        if self.filters is not None:
            ax.get_legend().set_title(self.filters)
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            plt.gcf().savefig(os.path.join(parameters.LSST_SAVEFIGPATH, f'{self.target.label}_spectrum.pdf'))
        if parameters.DISPLAY:
            if live_fit:
                plt.draw()
                plt.pause(1e-8)
                plt.close()
            else:
                plt.show()

    def plot_spectrogram(self, ax=None, scale="lin", title="", units="Image units", plot_stats=False,
                         target_pixcoords=None, vmin=None, vmax=None, figsize=[9.3, 8], aspect=None,
                         cmap=None, cax=None):
        """Plot spectrogram.

        Parameters
        ----------
        ax: Axes, optional
            Axes instance (default: None).
        scale: str
            Scaling of the image (choose between: lin, log or log10) (default: lin)
        title: str
            Title of the image (default: "")
        units: str
            Units of the image to be written in the color bar label (default: "Image units")
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
            If True, plot the uncertainty map instead of the spectrogram (default: False).

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
        >>> s.plot_spectrogram()
        >>> if parameters.DISPLAY: plt.show()

        .. plot::

            from spectractor.extractor.spectrum import Spectrum
            s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
            s.plot_spectrogram()

        """
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        data = np.copy(self.spectrogram)
        if plot_stats:
            data = np.copy(self.spectrogram_err)
        plot_image_simple(ax, data=data, scale=scale, title=title, units=units, cax=cax,
                          target_pixcoords=target_pixcoords, aspect=aspect, vmin=vmin, vmax=vmax, cmap=cmap)
        if parameters.DISPLAY:
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()

    def save_spectrum(self, output_file_name, overwrite=False):
        """Save the spectrum into a fits file (data, error and wavelengths).

        Parameters
        ----------
        output_file_name: str
            Path of the output fits file.
        overwrite: bool
            If True overwrite the output file if needed (default: False).

        Examples
        --------
        >>> import os
        >>> s = Spectrum(file_name='tests/data/reduc_20170530_134_spectrum.fits')
        >>> s.save_spectrum('./tests/test.fits')

        .. doctest::
            :hide:

            >>> assert os.path.isfile('./tests/test.fits')

        Overwrite previous file:
        >>> s.save_spectrum('./tests/test.fits', overwrite=True)

        .. doctest::
            :hide:

            >>> assert os.path.isfile('./tests/test.fits')
            >>> os.remove('./tests/test.fits')
        """
        self.header["REBIN"] = parameters.CCD_REBIN
        self.header.comments['REBIN'] = 'original image rebinning factor to get spectrum.'
        self.header['UNIT1'] = "nanometer"
        self.header['UNIT2'] = self.units
        self.header['COMMENTS'] = 'First column gives the wavelength in unit UNIT1, ' \
                                  'second column gives the spectrum in unit UNIT2, ' \
                                  'third column the corresponding errors.'
        hdu1 = fits.PrimaryHDU()
        hdu1.header = self.header
        hdu1.header["EXTNAME"] = "SPECTRUM"
        hdu2 = fits.ImageHDU()
        hdu2.header["EXTNAME"] = "SPEC_COV"
        hdu3 = fits.ImageHDU()
        hdu3.header["EXTNAME"] = "ORDER2"
        hdu4 = fits.ImageHDU()
        hdu4.header["EXTNAME"] = "ORDER0"
        hdu1.data = [self.lambdas, self.data, self.err]
        hdu2.data = self.cov_matrix
        hdu3.data = [self.lambdas_order2, self.data_order2, self.err_order2]
        hdu4.data = self.target.image
        hdu4.header["IM_X0"] = self.target.image_x0
        hdu4.header["IM_Y0"] = self.target.image_y0
        hdu = fits.HDUList([hdu1, hdu2, hdu3, hdu4])
        output_directory = '/'.join(output_file_name.split('/')[:-1])
        ensure_dir(output_directory)
        hdu.writeto(output_file_name, overwrite=overwrite)
        self.my_logger.info(f'\n\tSpectrum saved in {output_file_name}')

    def save_spectrogram(self, output_file_name, overwrite=False):
        """Save the spectrogram into a fits file (data, error and background).

        Parameters
        ----------
        output_file_name: str
            Path of the output fits file.
        overwrite: bool,  optional
            If True overwrite the output file if needed (default: False).

        Examples
        --------
        """
        self.header['UNIT1'] = self.units
        self.header['COMMENTS'] = 'First HDU gives the data in UNIT1 units, ' \
                                  'second HDU gives the uncertainties, ' \
                                  'third HDU the  fitted background.'
        self.header['S_X0'] = self.spectrogram_x0
        self.header['S_Y0'] = self.spectrogram_y0
        self.header['S_XMIN'] = self.spectrogram_xmin
        self.header['S_XMAX'] = self.spectrogram_xmax
        self.header['S_YMIN'] = self.spectrogram_ymin
        self.header['S_YMAX'] = self.spectrogram_ymax
        self.header['S_DEG'] = self.spectrogram_deg
        self.header['S_SAT'] = self.spectrogram_saturation
        hdu1 = fits.PrimaryHDU()
        hdu1.header["EXTNAME"] = "S_DATA"
        hdu2 = fits.ImageHDU()
        hdu2.header["EXTNAME"] = "S_ERR"
        hdu3 = fits.ImageHDU()
        hdu3.header["EXTNAME"] = "S_BGD"
        hdu4 = fits.ImageHDU()
        hdu4.header["EXTNAME"] = "S_BGD_ER"
        hdu5 = fits.ImageHDU()
        hdu5.header["EXTNAME"] = "S_FIT"
        hdu6 = fits.ImageHDU()
        hdu6.header["EXTNAME"] = "S_RES"
        hdu1.header = self.header
        hdu1.data = self.spectrogram
        hdu2.data = self.spectrogram_err
        hdu3.data = self.spectrogram_bgd
        hdu4.data = self.spectrogram_bgd_rms
        hdu5.data = self.spectrogram_fit
        hdu6.data = self.spectrogram_residuals
        hdu = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6])
        output_directory = '/'.join(output_file_name.split('/')[:-1])
        ensure_dir(output_directory)
        hdu.writeto(output_file_name, overwrite=overwrite)
        self.my_logger.info('\n\tSpectrogram saved in %s' % output_file_name)

    def load_spectrum(self, input_file_name):
        """Load the spectrum from a fits file (data, error and wavelengths).

        Parameters
        ----------
        input_file_name: str
            Path to the input fits file

        Examples
        --------
        >>> s = Spectrum()
        >>> s.load_spectrum('tests/data/reduc_20170605_028_spectrum.fits')
        >>> print(s.units)
        erg/s/cm$^2$/nm
        """
        if os.path.isfile(input_file_name):
            self.header, raw_data = load_fits(input_file_name)
            self.lambdas = raw_data[0]
            self.lambdas_binwidths = np.gradient(self.lambdas)
            self.data = raw_data[1]
            if len(raw_data) > 2:
                self.err = raw_data[2]
            if self.header['DATE-OBS'] != "":
                self.date_obs = self.header['DATE-OBS']
            if self.header['EXPTIME'] != "":
                self.expo = self.header['EXPTIME']
            if self.header['AIRMASS'] != "":
                self.airmass = self.header['AIRMASS']
            if self.header['GRATING'] != "":
                self.disperser_label = self.header['GRATING']
            if self.header['TARGET'] != "":
                self.target = load_target(self.header['TARGET'], verbose=parameters.VERBOSE)
                self.lines = self.target.lines
            if self.header['UNIT2'] != "":
                self.units = self.header['UNIT2']
            if self.header['ROTANGLE'] != "":
                self.rotation_angle = self.header['ROTANGLE']
            if self.header['TARGETX'] != "" and self.header['TARGETY'] != "":
                self.x0 = [self.header['TARGETX'], self.header['TARGETY']]
            if self.header['D2CCD'] != "":
                parameters.DISTANCE2CCD = float(self.header["D2CCD"])
            if 'DEC' in self.header and self.header['DEC'] != "":
                self.dec = self.header['DEC']
            if 'RA' in self.header and self.header['HA'] != "":
                self.hour_angle = self.header['HA']
            if 'OUTTEMP' in self.header and self.header['OUTTEMP'] != "":
                self.temperature = self.header['OUTTEMP']
            if 'OUTPRESS' in self.header and self.header['OUTPRESS'] != "":
                self.pressure = self.header['OUTPRESS']
            if 'OUTHUM' in self.header and self.header['OUTHUM'] != "":
                self.humidity = self.header['OUTHUM']
            if self.header['LBDA_REF'] != "":
                self.lambda_ref = self.header['LBDA_REF']
            if 'PARANGLE' in self.header and self.header['PARANGLE'] != "":
                self.parallactic_angle = self.header['PARANGLE']
            if 'CCDREBIN' in self.header and self.header['CCDREBIN'] != "":
                if parameters.CCD_REBIN != self.header['CCDREBIN']:
                    raise ValueError("Different values of rebinning parameters between config file and header. Choose.")

            self.my_logger.info('\n\tLoading disperser %s...' % self.disperser_label)
            self.disperser = Hologram(self.disperser_label, D=parameters.DISTANCE2CCD,
                                      data_dir=parameters.DISPERSER_DIR, verbose=parameters.VERBOSE)
            self.my_logger.info('\n\tSpectrum loaded from %s' % input_file_name)
            spectrogram_file_name = input_file_name.replace('spectrum', 'spectrogram')
            if parameters.OBS_OBJECT_TYPE == "STAR":
                self.adr_params = [self.dec, self.hour_angle, self.temperature,
                                   self.pressure, self.humidity, self.airmass]

            hdu_list = fits.open(input_file_name)
            if len(hdu_list) > 1:
                self.cov_matrix = hdu_list["SPEC_COV"].data
                if len(hdu_list) > 2:
                    self.lambdas_order2, self.data_order2, self.err_order2 = hdu_list["ORDER2"].data
                    if len(hdu_list) > 3:
                        self.target.image = hdu_list["ORDER0"].data
                        self.target.image_x0 = float(hdu_list["ORDER0"].header["IM_X0"])
                        self.target.image_y0 = float(hdu_list["ORDER0"].header["IM_Y0"])
            else:
                self.cov_matrix = np.diag(self.err ** 2)
            if not self.fast_load:
                self.my_logger.info(f'\n\tLoading spectrogram from {spectrogram_file_name}...')
                if os.path.isfile(spectrogram_file_name):
                    self.load_spectrogram(spectrogram_file_name)
                else:
                    raise FileNotFoundError(f"Spectrogram file {spectrogram_file_name} does not exist.")
                psf_file_name = input_file_name.replace('spectrum.fits', 'table.csv')
                self.my_logger.info(f'\n\tLoading PSF from {psf_file_name}...')
                if os.path.isfile(psf_file_name):
                    self.load_chromatic_psf(psf_file_name)
                else:
                    raise FileNotFoundError(f"PSF file {psf_file_name} does not exist.")
        else:
            raise FileNotFoundError(f'\n\tSpectrum file {input_file_name} not found')

    def load_spectrogram(self, input_file_name):
        """Load the spectrum from a fits file (data, error and wavelengths).

        Parameters
        ----------
        input_file_name: str
            Path to the input fits file

        Examples
        --------
        >>> s = Spectrum()
        >>> s.load_spectrum('tests/data/reduc_20170605_028_spectrum.fits')
        """
        if os.path.isfile(input_file_name):
            hdu_list = fits.open(input_file_name)
            header = hdu_list[0].header
            self.spectrogram = hdu_list[0].data
            self.spectrogram_err = hdu_list[1].data
            self.spectrogram_bgd = hdu_list[2].data
            if len(hdu_list) > 3:
                self.spectrogram_bgd_rms = hdu_list[3].data
                self.spectrogram_fit = hdu_list[4].data
                self.spectrogram_residuals = hdu_list[5].data
            self.spectrogram_x0 = float(header['S_X0'])
            self.spectrogram_y0 = float(header['S_Y0'])
            self.spectrogram_xmin = int(header['S_XMIN'])
            self.spectrogram_xmax = int(header['S_XMAX'])
            self.spectrogram_ymin = int(header['S_YMIN'])
            self.spectrogram_ymax = int(header['S_YMAX'])
            self.spectrogram_deg = int(header['S_DEG'])
            self.spectrogram_saturation = float(header['S_SAT'])
            self.spectrogram_Nx = self.spectrogram_xmax - self.spectrogram_xmin
            self.spectrogram_Ny = self.spectrogram_ymax - self.spectrogram_ymin
            hdu_list.close()  # need to free allocation for file descripto
            self.my_logger.info('\n\tSpectrogram loaded from %s' % input_file_name)
        else:
            self.my_logger.warning('\n\tSpectrogram file %s not found' % input_file_name)

    def load_chromatic_psf(self, input_file_name):
        """Load the spectrum from a fits file (data, error and wavelengths).

        Parameters
        ----------
        input_file_name: str
            Path to the input fits file

        Examples
        --------
        >>> s = Spectrum()
        >>> s.load_spectrum('./tests/data/reduc_20170530_134_spectrum.fits')
        >>> print(s.chromatic_psf.table)  #doctest: +ELLIPSIS
             lambdas               Dx        ...
        """
        if os.path.isfile(input_file_name):
            self.psf = load_PSF(psf_type=parameters.PSF_TYPE, target=self.target)
            self.chromatic_psf = ChromaticPSF(self.psf, self.spectrogram_Nx, self.spectrogram_Ny,
                                              x0=self.spectrogram_x0, y0=self.spectrogram_y0,
                                              deg=self.spectrogram_deg, saturation=self.spectrogram_saturation,
                                              file_name=input_file_name)
            if 'PSF_REG' in self.header and float(self.header["PSF_REG"]) > 0:
                self.chromatic_psf.opt_reg = float(self.header["PSF_REG"])
            self.my_logger.info(f'\n\tSpectrogram loaded from {input_file_name}')
        else:
            self.my_logger.warning(f'\n\tSpectrogram file {input_file_name} not found')


def detect_lines(lines, lambdas, spec, spec_err=None, cov_matrix=None, fwhm_func=None, snr_minlevel=3, ax=None,
                 calibration_lines_only=False,
                 xlim=(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX)):
    """Detect and fit the lines in a spectrum. The method is to look at maxima or minima
    around emission or absorption tabulated lines, and to select surrounding pixels
    to fit a (positive or negative) gaussian and a polynomial background. If several regions
    overlap, a multi-gaussian fit is performed above a common polynomial background.
    The mean global shift (in nm) between the detected and tabulated lines is returned, considering
    only the lines with a signal-to-noise ratio above a threshold.
    The order of the polynomial background is set in parameters.py with CALIB_BGD_ORDER.

    Parameters
    ----------
    lines: Lines
        The Lines object containing the line characteristics
    lambdas: float array
        The wavelength array (in nm)
    spec: float array
        The spectrum amplitude array
    spec_err: float array, optional
        The spectrum amplitude uncertainty array (default: None)
    cov_matrix: float array, optional
        The spectrum amplitude 2D covariance matrix array (default: None)
    fwhm_func: callable, optional
        The fwhm of the cross spectrum to reset CALIB_PEAK_WIDTH parameter as a function of lambda (default: None)
    snr_minlevel: float
        The minimum signal over noise ratio to consider using a fitted line in the computation of the mean
        shift output and to print it in the outpur table (default: 3)
    ax: Axes, optional
        An Axes instance to over plot the result of the fit (default: None).
    calibration_lines_only: bool, optional
        If True, try to detect only the lines with use_for_calibration attributes set True.
    xlim: array, optional
        (min, max) list limiting the wavelength interval where to detect spectral lines (default:
        (parameters.LAMBDA_MIN, parameters.LAMBDA_MAX))

    Returns
    -------
    shift: float
        The mean shift (in nm) between the detected and tabulated lines

    Examples
    --------

    Creation of a mock spectrum with emission and absorption lines:

    >>> import numpy as np
    >>> from spectractor.extractor.spectroscopy import Lines, HALPHA, HBETA, O2_1
    >>> lambdas = np.arange(300,1000,1)
    >>> spectrum = 1e4*np.exp(-((lambdas-600)/200)**2)
    >>> spectrum += HALPHA.gaussian_model(lambdas, A=5000, sigma=3)
    >>> spectrum += HBETA.gaussian_model(lambdas, A=3000, sigma=2)
    >>> spectrum += O2_1.gaussian_model(lambdas, A=-3000, sigma=7)
    >>> spectrum_err = np.sqrt(spectrum)
    >>> cov = np.diag(spectrum_err)
    >>> spectrum = np.random.poisson(spectrum)
    >>> spec = Spectrum()
    >>> spec.lambdas = lambdas
    >>> spec.data = spectrum
    >>> spec.err = spectrum_err
    >>> fwhm_func = interp1d(lambdas, 0.01 * lambdas)

    Detect the lines:

    >>> lines = Lines([HALPHA, HBETA, O2_1], hydrogen_only=True,
    ... atmospheric_lines=True, redshift=0, emission_spectrum=True)
    >>> global_chisq = detect_lines(lines, lambdas, spectrum, spectrum_err, cov, fwhm_func=fwhm_func)

    .. doctest::
        :hide:

        >>> assert(global_chisq < 2)

    Plot the result:

    >>> import matplotlib.pyplot as plt
    >>> spec.lines = lines
    >>> fig = plt.figure()
    >>> plot_spectrum_simple(plt.gca(), lambdas, spec.data, data_err=spec.err)
    >>> lines.plot_detected_lines(plt.gca())
    >>> if parameters.DISPLAY: plt.show()
    """

    # main settings
    peak_width = parameters.CALIB_PEAK_WIDTH
    bgd_width = parameters.CALIB_BGD_WIDTH
    # if lines.hydrogen_only:
    #     peak_width = 7
    #     bgd_width = 15
    fwhm_to_peak_width_factor = 1.5
    len_index_to_bgd_npar_factor = 0 * 0.12 / 0.024 * parameters.CCD_PIXEL2MM
    baseline_prior = 3  # *sigma gaussian prior on base line fit
    # filter the noise
    # plt.errorbar(lambdas,spec,yerr=spec_err)
    spec = np.copy(spec)
    spec_smooth = savgol_filter(spec, parameters.CALIB_SAVGOL_WINDOW, parameters.CALIB_SAVGOL_ORDER)
    # plt.plot(lambdas,spec)
    # plt.show()
    # initialisation
    lambda_shifts = []
    snrs = []
    index_list = []
    bgd_npar_list = []
    peak_index_list = []
    guess_list = []
    bounds_list = []
    lines_list = []
    for line in lines.lines:
        # reset line fit attributes
        line.fitted = False
        line.fit_popt = None
        line.high_snr = False
        if not line.use_for_calibration and calibration_lines_only:
            continue
        # wavelength of the line: find the nearest pixel index
        line_wavelength = line.wavelength
        if fwhm_func is not None:
            peak_width = max(fwhm_to_peak_width_factor * fwhm_func(line_wavelength), parameters.CALIB_PEAK_WIDTH)
        if line_wavelength < xlim[0] or line_wavelength > xlim[1]:
            continue
        l_index, l_lambdas = find_nearest(lambdas, line_wavelength)
        # reject if pixel index is too close to image bounds
        if l_index < peak_width or l_index > len(lambdas) - peak_width:
            continue
        # search for local extrema to detect emission or absorption line
        # around pixel index +/- peak_width
        line_strategy = np.greater  # look for emission line
        bgd_strategy = np.less
        if not lines.emission_spectrum or line.atmospheric:
            line_strategy = np.less  # look for absorption line
            bgd_strategy = np.greater
        index = np.arange(l_index - peak_width, l_index + peak_width, 1).astype(int)
        # skip if data is masked with NaN
        if np.any(np.isnan(spec_smooth[index])):
            continue
        extrema = argrelextrema(spec_smooth[index], line_strategy)
        if len(extrema[0]) == 0:
            continue
        peak_index = index[0] + extrema[0][0]
        # if several extrema, look for the greatest
        if len(extrema[0]) > 1:
            if line_strategy == np.greater:
                test = -1e20
                for m in extrema[0]:
                    idx = index[0] + m
                    if spec_smooth[idx] > test:
                        peak_index = idx
                        test = spec_smooth[idx]
            elif line_strategy == np.less:
                test = 1e20
                for m in extrema[0]:
                    idx = index[0] + m
                    if spec_smooth[idx] < test:
                        peak_index = idx
                        test = spec_smooth[idx]
        # search for first local minima around the local maximum
        # or for first local maxima around the local minimum
        # around +/- 3*peak_width
        index_inf = peak_index - 1  # extrema on the left
        while index_inf > max(0, peak_index - 3 * peak_width):
            test_index = np.arange(index_inf, peak_index, 1).astype(int)
            minm = argrelextrema(spec_smooth[test_index], bgd_strategy)
            if len(minm[0]) > 0:
                index_inf = index_inf + minm[0][0]
                break
            else:
                index_inf -= 1
        index_sup = peak_index + 1  # extrema on the right
        while index_sup < min(len(spec_smooth) - 1, peak_index + 3 * peak_width):
            test_index = np.arange(peak_index, index_sup, 1).astype(int)
            minm = argrelextrema(spec_smooth[test_index], bgd_strategy)
            if len(minm[0]) > 0:
                index_sup = peak_index + minm[0][0]
                break
            else:
                index_sup += 1
        index_sup = max(index_sup, peak_index + peak_width)
        index_inf = min(index_inf, peak_index - peak_width)
        # pixel range to consider around the peak, adding bgd_width pixels
        # to fit for background around the peak
        index = list(np.arange(max(0, index_inf - bgd_width),
                               min(len(lambdas), index_sup + bgd_width), 1).astype(int))
        # skip if data is masked with NaN
        if np.any(np.isnan(spec_smooth[index])):
            continue
        # first guess and bounds to fit the line properties and
        # the background with CALIB_BGD_ORDER order polynom
        # guess = [0] * bgd_npar + [0.5 * np.max(spec_smooth[index]), lambdas[peak_index],
        #                          0.5 * (line.width_bounds[0] + line.width_bounds[1])]
        bgd_npar = max(parameters.CALIB_BGD_NPARAMS, int(len_index_to_bgd_npar_factor * (index[-1] - index[0])))
        bgd_npar_list.append(bgd_npar)
        guess = [0] * bgd_npar + [0.5 * np.max(spec_smooth[index]), line_wavelength,
                                  0.5 * (line.width_bounds[0] + line.width_bounds[1])]
        if line_strategy == np.less:
            # noinspection PyTypeChecker
            guess[bgd_npar] = -0.5 * np.max(spec_smooth[index])  # look for abosrption under bgd
        # bounds = [[-np.inf] * bgd_npar + [-abs(np.max(spec[index])), lambdas[index_inf], line.width_bounds[0]],
        #          [np.inf] * bgd_npar + [abs(np.max(spec[index])), lambdas[index_sup], line.width_bounds[1]]]
        bounds = [[-np.inf] * bgd_npar + [-abs(np.max(spec[index])), line_wavelength - peak_width / 2,
                                          line.width_bounds[0]],
                  [np.inf] * bgd_npar + [abs(np.max(spec[index])), line_wavelength + peak_width / 2,
                                         line.width_bounds[1]]]
        # gaussian amplitude bounds depend if line is emission/absorption
        if line_strategy == np.less:
            bounds[1][bgd_npar] = 0  # look for absorption under bgd
        else:
            bounds[0][bgd_npar] = 0  # look for emission above bgd
        peak_index_list.append(peak_index)
        index_list.append(index)
        lines_list.append(line)
        guess_list.append(guess)
        bounds_list.append(bounds)
    # now gather lines together if pixel index ranges overlap
    idx = 0
    merges = [[0]]
    while idx < len(index_list) - 1:
        idx = merges[-1][-1]
        if idx == len(index_list) - 1:
            break
        if index_list[idx + 1][0] > index_list[idx][0]:  # increasing order
            if index_list[idx][-1] > index_list[idx + 1][0]:
                merges[-1].append(idx + 1)
            else:
                merges.append([idx + 1])
                idx += 1
        else:  # decreasing order
            if index_list[idx][0] < index_list[idx + 1][-1]:
                merges[-1].append(idx + 1)
            else:
                merges.append([idx + 1])
                idx += 1
    # reorder merge list with respect to lambdas in guess list
    new_merges = []
    for merge in merges:
        if len(guess_list) == 0:
            continue
        tmp_guess = [guess_list[i][-2] for i in merge]
        new_merges.append([x for _, x in sorted(zip(tmp_guess, merge))])
    # reorder lists with merges
    new_peak_index_list = []
    new_index_list = []
    new_guess_list = []
    new_bounds_list = []
    new_lines_list = []
    for merge in new_merges:
        new_peak_index_list.append([])
        new_index_list.append([])
        new_guess_list.append([])
        new_bounds_list.append([[], []])
        new_lines_list.append([])
        for i in merge:
            # add the bgd parameters
            bgd_npar = bgd_npar_list[i]
            # if i == merge[0]:
            #     new_guess_list[-1] += guess_list[i][:bgd_npar]
            #     new_bounds_list[-1][0] += bounds_list[i][0][:bgd_npar]
            #     new_bounds_list[-1][1] += bounds_list[i][1][:bgd_npar]
            # add the gauss parameters
            new_peak_index_list[-1].append(peak_index_list[i])
            new_index_list[-1] += index_list[i]
            new_guess_list[-1] += guess_list[i][bgd_npar:]
            new_bounds_list[-1][0] += bounds_list[i][0][bgd_npar:]
            new_bounds_list[-1][1] += bounds_list[i][1][bgd_npar:]
            new_lines_list[-1].append(lines_list[i])
        # set central peak bounds exactly between two close lines
        for k in range(len(merge) - 1):
            new_bounds_list[-1][0][3 * (k + 1) + 1] = 0.5 * (
                    new_guess_list[-1][3 * k + 1] + new_guess_list[-1][3 * (k + 1) + 1])
            new_bounds_list[-1][1][3 * k + 1] = 0.5 * (
                    new_guess_list[-1][3 * k + 1] + new_guess_list[-1][3 * (k + 1) + 1]) + 1e-3
            # last term is to avoid equalities
            # between bounds in some pathological case
        # sort pixel indices and remove doublons
        new_index_list[-1] = sorted(list(set(new_index_list[-1])))
    # fit the line subsets and background
    global_chisq = 0
    for k in range(len(new_index_list)):
        # first guess for the base line with the lateral bands
        peak_index = new_peak_index_list[k]
        index = new_index_list[k]
        guess = new_guess_list[k]
        bounds = new_bounds_list[k]
        bgd_index = []
        if fwhm_func is not None:
            peak_width = fwhm_to_peak_width_factor * np.mean(fwhm_func(lambdas[index]))
        for i in index:
            is_close_to_peak = False
            for j in peak_index:
                if abs(i - j) < peak_width:
                    is_close_to_peak = True
                    break
            if not is_close_to_peak:
                bgd_index.append(i)
        # add background guess and bounds
        bgd_npar = max(parameters.CALIB_BGD_ORDER + 1, int(len_index_to_bgd_npar_factor * len(bgd_index)))
        parameters.CALIB_BGD_NPARAMS = bgd_npar
        guess = [0] * bgd_npar + guess
        bounds[0] = [-np.inf] * bgd_npar + bounds[0]
        bounds[1] = [np.inf] * bgd_npar + bounds[1]
        if len(bgd_index) > 0:
            try:
                if spec_err is not None:
                    w = 1. / spec_err[bgd_index]
                else:
                    w = np.ones_like(lambdas[bgd_index])
                fit, cov, model = fit_poly1d_legendre(lambdas[bgd_index], spec[bgd_index], order=bgd_npar - 1, w=w)
            except:
                if spec_err is not None:
                    w = 1. / spec_err[index]
                else:
                    w = np.ones_like(lambdas[index])
                fit, cov, model = fit_poly1d_legendre(lambdas[index], spec[index], order=bgd_npar - 1, w=w)
        else:
            if spec_err is not None:
                w = 1. / spec_err[index]
            else:
                w = np.ones_like(lambdas[index])
            fit, cov, model = fit_poly1d_legendre(lambdas[index], spec[index], order=bgd_npar - 1, w=w)
        # lines.my_logger.warning(f'{bgd_npar} {fit}')
        # fig = plt.figure()
        # plt.plot(lambdas[index], spec[index])
        # plt.plot(lambdas[bgd_index], spec[bgd_index], 'ro')
        # x_norm = rescale_x_for_legendre(lambdas[index])
        # lines.my_logger.warning(f'tototot {x_norm}')
        # plt.plot(lambdas[index], np.polynomial.legendre.legval(x_norm, fit), 'b-')
        # plt.plot(lambdas[bgd_index], model, 'b--')
        # plt.title(f"{fit}")
        # plt.show()
        for n in range(bgd_npar):
            # guess[n] = getattr(bgd, bgd.param_names[parameters.CALIB_BGD_ORDER - n]).value
            guess[n] = fit[n]
            b = abs(baseline_prior * guess[n])
            if np.isclose(b, 0, rtol=1e-2 * float(np.mean(spec_smooth[bgd_index]))):
                b = baseline_prior * np.std(spec_smooth[bgd_index])
                if np.isclose(b, 0, rtol=1e-2 * float(np.mean(spec_smooth[bgd_index]))):
                    b = np.inf
            bounds[0][n] = guess[n] - b
            bounds[1][n] = guess[n] + b
        for j in range(len(new_lines_list[k])):
            idx = new_peak_index_list[k][j]
            x_norm = rescale_x_for_legendre(lambdas[idx])
            guess[bgd_npar + 3 * j] = np.sign(guess[bgd_npar + 3 * j]) * abs(
                spec_smooth[idx] - np.polynomial.legendre.legval(x_norm, guess[:bgd_npar]))
            if np.sign(guess[bgd_npar + 3 * j]) < 0:  # absorption
                bounds[0][bgd_npar + 3 * j] = 2 * guess[bgd_npar + 3 * j]
            else:  # emission
                bounds[1][bgd_npar + 3 * j] = 2 * guess[bgd_npar + 3 * j]
        # fit local extrema with a multigaussian + CALIB_BGD_ORDER polynom
        # account for the spectrum uncertainties if provided
        sigma = None
        if spec_err is not None:
            sigma = spec_err[index]
        if cov_matrix is not None:
            sigma = cov_matrix[index, index]
        # my_logger.warning(f'\n{guess} {np.mean(spec[bgd_index])} {np.std(spec[bgd_index])}')
        popt, pcov = fit_multigauss_and_bgd(lambdas[index], spec[index], guess=guess, bounds=bounds, sigma=sigma)
        # noise level defined as the std of the residuals if no error
        noise_level = np.std(spec[index] - multigauss_and_bgd(lambdas[index], *popt))
        # otherwise mean of error bars of bgd lateral bands
        if sigma is not None:
            chisq = np.sum((multigauss_and_bgd(lambdas[index], *popt) - spec[index]) ** 2 / (sigma * sigma))
        else:
            chisq = np.sum((multigauss_and_bgd(lambdas[index], *popt) - spec[index]) ** 2)
        chisq /= len(index)
        global_chisq += chisq
        if spec_err is not None:
            noise_level = np.sqrt(np.mean(spec_err[index] ** 2))

        for j in range(len(new_lines_list[k])):
            line = new_lines_list[k][j]
            peak_pos = popt[bgd_npar + 3 * j + 1]
            # FWHM
            FWHM = np.abs(popt[bgd_npar + 3 * j + 2]) * 2.355
            # SNR computation
            # signal_level = popt[bgd_npar+3*j]
            signal_level = popt[
                bgd_npar + 3 * j]  # multigauss_and_bgd(peak_pos, *popt) - np.polyval(popt[:bgd_npar], peak_pos)
            snr = np.abs(signal_level / noise_level)
            # save fit results
            line.fitted = True
            line.fit_index = index
            line.fit_lambdas = lambdas[index]

            x_norm = rescale_x_for_legendre(lambdas[index])

            x_step = 0.1  # nm
            x_int = np.arange(max(np.min(lambdas), peak_pos - 5 * np.abs(popt[bgd_npar + 3 * j + 2])),
                              min(np.max(lambdas), peak_pos + 5 * np.abs(popt[bgd_npar + 3 * j + 2])), x_step)
            middle = 0.5 * (np.max(lambdas[index]) + np.min(lambdas[index]))
            x_int_norm = x_int - middle
            if np.max(lambdas[index] - middle) != 0:
                x_int_norm = x_int_norm / np.max(lambdas[index] - middle)

            # jmin and jmax a bit larger than x_int to avoid extrapolation
            jmin = max(0, int(np.argmin(np.abs(lambdas - (x_int[0] - x_step))) - 2))
            jmax = min(len(lambdas), int(np.argmin(np.abs(lambdas - (x_int[-1] + x_step))) + 2))
            if jmax-2 < jmin+2:  # decreasing order
                jmin, jmax = max(0, jmax-4), min(len(lambdas), jmin+4)
            spectr_data = interp1d(lambdas[jmin:jmax], spec[jmin:jmax],
                                   bounds_error=False, fill_value="extrapolate")(x_int)

            Continuum = np.polynomial.legendre.legval(x_int_norm, popt[:bgd_npar])
            Gauss = gauss(x_int, *popt[bgd_npar + 3 * j:bgd_npar + 3 * j + 3])

            Y = -Gauss / Continuum
            Ydata = 1 - spectr_data / Continuum

            line.fit_eqwidth_mod = integrate.simps(Y, x_int)  # sol1
            line.fit_eqwidth_data = integrate.simps(Ydata, x_int)  # sol2

            line.fit_popt = popt
            line.fit_popt_gaussian = popt[bgd_npar + 3 * j:bgd_npar + 3 * j + 3]
            line.fit_gauss = gauss(lambdas[index], *popt[bgd_npar + 3 * j:bgd_npar + 3 * j + 3])

            line.fit_bgd = np.polynomial.legendre.legval(x_norm, popt[:bgd_npar])
            line.fit_snr = snr
            line.fit_chisq = chisq
            line.fit_fwhm = FWHM
            line.fit_bgd_npar = bgd_npar
            if snr < snr_minlevel:
                continue
            line.high_snr = True
            if line.use_for_calibration:
                # wavelength shift between tabulate and observed lines
                lambda_shifts.append(peak_pos - line.wavelength)
                snrs.append(snr)
    if ax is not None:
        lines.plot_detected_lines(ax, print_table=parameters.DEBUG)
    if len(lambda_shifts) > 0:
        global_chisq /= len(lambda_shifts)
        shift = np.average(np.abs(lambda_shifts) ** 2, weights=np.array(snrs) ** 2)
        # if guess values on tabulated lines have not moved: penalize the chisq
        global_chisq += shift
        # lines.my_logger.debug(f'\n\tNumber of calibration lines detected {len(lambda_shifts):d}'
        #                      f'\n\tTotal chisq: {global_chisq:.3f} with shift {shift:.3f}pix')
    else:
        global_chisq = 2 * len(parameters.LAMBDAS)
        # lines.my_logger.debug(
        #    f'\n\tNumber of calibration lines detected {len(lambda_shifts):d}\n\tTotal chisq: {global_chisq:.3f}')
    return global_chisq


def calibrate_spectrum(spectrum, with_adr=False):
    """Convert pixels into wavelengths given the position of the order 0,
    the data for the spectrum, the properties of the disperser. Fit the absorption
    (and eventually the emission) lines to perform a second calibration of the
    distance between the CCD and the disperser. The number of fitting steps is
    limited to 30.

    Finally convert the spectrum amplitude from ADU rate to erg/s/cm2/nm.

    Parameters
    ----------
    spectrum: Spectrum
        Spectrum object to calibrate
    with_adr: bool, optional
        If True, the ADR longitudinal shift is subtracted to distances.
        Must be False if the spectrum has already been decontaminated from ADR (default: False).

    Returns
    -------
    lambdas: array_like
        The new wavelength array in nm.

    Examples
    --------
    >>> spectrum = Spectrum('tests/data/reduc_20170605_028_spectrum.fits')
    >>> parameters.LAMBDA_MIN = 550
    >>> parameters.LAMBDA_MAX = 800
    >>> lambdas = calibrate_spectrum(spectrum, with_adr=False)
    >>> spectrum.plot_spectrum()
    """
    with_adr = int(with_adr)
    distance = spectrum.chromatic_psf.get_algebraic_distance_along_dispersion_axis()
    spectrum.lambdas = spectrum.disperser.grating_pixel_to_lambda(distance, spectrum.x0, order=spectrum.order)
    if spectrum.lambda_ref is None:
        lambda_ref = np.sum(spectrum.lambdas * spectrum.data) / np.sum(spectrum.data)
        spectrum.lambda_ref = lambda_ref
        spectrum.header['LBDA_REF'] = lambda_ref
    # ADR is x>0 westward and y>0 northward while CTIO images are x>0 westward and y>0 southward
    # Must project ADR along dispersion axis
    if with_adr > 0:
        adr_ra, adr_dec = adr_calib(spectrum.lambdas, spectrum.adr_params, parameters.OBS_LATITUDE,
                                    lambda_ref=spectrum.lambda_ref)
        adr_u, _ = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec,
                                                               dispersion_axis_angle=spectrum.rotation_angle)
    else:
        adr_u = np.zeros_like(distance)
    x0 = spectrum.x0
    if x0 is None:
        x0 = spectrum.target_pixcoords
        spectrum.x0 = x0

    # Detect emission/absorption lines and calibrate pixel/lambda
    fwhm_func = interp1d(spectrum.chromatic_psf.table['lambdas'],
                         spectrum.chromatic_psf.table['fwhm'],
                         fill_value=(parameters.CALIB_PEAK_WIDTH, parameters.CALIB_PEAK_WIDTH), bounds_error=False)

    def shift_minimizer(params):
        spectrum.disperser.D, shift = params
        if np.isnan(spectrum.disperser.D):  # reset the value in case of bad gradient descent
            spectrum.disperser.D = parameters.DISTANCE2CCD
        if np.isnan(shift):  # reset the value in case of bad gradient descent
            shift = 0
        dist = spectrum.chromatic_psf.get_algebraic_distance_along_dispersion_axis(shift_x=shift)
        spectrum.lambdas = spectrum.disperser.grating_pixel_to_lambda(dist - with_adr * adr_u,
                                                                      x0=[x0[0] + shift, x0[1]], order=spectrum.order)
        spectrum.lambdas_binwidths = np.gradient(spectrum.lambdas)
        spectrum.convert_from_ADUrate_to_flam()
        chisq = detect_lines(spectrum.lines, spectrum.lambdas, spectrum.data, spec_err=spectrum.err,
                             fwhm_func=fwhm_func, ax=None, calibration_lines_only=True)
        chisq += ((shift) / parameters.PIXSHIFT_PRIOR) ** 2
        if parameters.DEBUG and parameters.DISPLAY:
            if parameters.LIVE_FIT:
                spectrum.plot_spectrum(live_fit=True, label=f'Order {spectrum.order:d} spectrum\n'
                                                            r'$D_\mathrm{CCD}'
                                                            rf'={D:.2f}\,$mm, $\delta u_0={shift:.2f}\,$pix')
        spectrum.convert_from_flam_to_ADUrate()
        return chisq

    # grid exploration of the parameters
    # necessary because of the the line detection algo
    D = parameters.DISTANCE2CCD
    if spectrum.header['D2CCD'] != '':
        D = spectrum.header['D2CCD']
    D_err = parameters.DISTANCE2CCD_ERR
    D_step = D_err / 2
    pixel_shift_step = parameters.PIXSHIFT_PRIOR / 5
    pixel_shift_prior = parameters.PIXSHIFT_PRIOR
    Ds = np.arange(D - 5 * D_err, D + 6 * D_err, D_step)
    pixel_shifts = np.arange(-pixel_shift_prior, pixel_shift_prior + pixel_shift_step, pixel_shift_step)
    # pixel_shifts = np.array([0])
    chisq_grid = np.zeros((len(Ds), len(pixel_shifts)))
    for i, D in enumerate(Ds):
        for j, pixel_shift in enumerate(pixel_shifts):
            chisq_grid[i, j] = shift_minimizer([D, pixel_shift])
    imin, jmin = np.unravel_index(chisq_grid.argmin(), chisq_grid.shape)
    D = Ds[imin]
    pixel_shift = pixel_shifts[jmin]
    start = np.array([D, pixel_shift])
    if imin == 0 or imin == Ds.size or jmin == 0 or jmin == pixel_shifts.size:
        spectrum.my_logger.warning('\n\tMinimum chisq is on the edge of the exploration grid.')
    if parameters.DEBUG:
        fig = plt.figure(figsize=(7, 4))
        im = plt.imshow(np.log10(chisq_grid), origin='lower', aspect='auto',
                        extent=(
                            np.min(pixel_shifts) - pixel_shift_step / 2, np.max(pixel_shifts) + pixel_shift_step / 2,
                            np.min(Ds) - D_step / 2, np.max(Ds) + D_step / 2))
        plt.gca().scatter(pixel_shift, D, marker='o', s=100, edgecolors='k', facecolors='none',
                          label='Minimum', linewidth=2)
        c = plt.colorbar(im)
        c.set_label('Log10(chisq)')
        plt.xlabel(r'Pixel shift $\delta u_0$ [pix]')
        plt.ylabel(r'$D_\mathrm{CCD}$ [mm]')
        plt.legend()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'D2CCD_x0_fit.pdf'))
    # now minimize around the global minimum found previously
    # res = opt.minimize(shift_minimizer, start, args=(), method='L-BFGS-B',
    #                    options={'maxiter': 200, 'ftol': 1e-3},
    #                    bounds=((D - 5 * parameters.DISTANCE2CCD_ERR, D + 5 * parameters.DISTANCE2CCD_ERR), (-2, 2)))
    error = [parameters.DISTANCE2CCD_ERR, pixel_shift_step]
    fix = [False, False]
    m = Minuit(shift_minimizer, start)
    m.errors = error
    m.errordef = 1
    m.fixed = fix
    m.print_level = 0
    m.limits = ((D - 5 * parameters.DISTANCE2CCD_ERR, D + 5 * parameters.DISTANCE2CCD_ERR), (-2, 2))
    m.migrad()
    # if parameters.DEBUG:
    #     print(m.prin)
    # if not res.success:
    #     spectrum.my_logger.warning('\n\tMinimizer failed.')
    #     print(res)
    D, pixel_shift = np.array(m.values[:])
    spectrum.disperser.D = D
    x0 = [x0[0] + pixel_shift, x0[1]]
    spectrum.x0 = x0
    # check success, xO or D on the edge of their priors
    distance = spectrum.chromatic_psf.get_algebraic_distance_along_dispersion_axis(shift_x=pixel_shift)
    lambdas = spectrum.disperser.grating_pixel_to_lambda(distance - with_adr * adr_u, x0=x0, order=spectrum.order)
    spectrum.lambdas = lambdas
    spectrum.lambdas_order2 = spectrum.disperser.grating_pixel_to_lambda(distance - with_adr * adr_u, x0=x0,
                                                                         order=spectrum.order + 1)
    spectrum.lambdas_binwidths = np.gradient(lambdas)
    spectrum.convert_from_ADUrate_to_flam()
    spectrum.chromatic_psf.table['Dx'] -= pixel_shift
    spectrum.chromatic_psf.table['Dy_disp_axis'] = distance * np.sin(spectrum.rotation_angle * np.pi / 180)
    spectrum.pixels = np.copy(spectrum.chromatic_psf.table['Dx'])
    detect_lines(spectrum.lines, spectrum.lambdas, spectrum.data, spec_err=spectrum.err,
                 fwhm_func=fwhm_func, ax=None, calibration_lines_only=False)
    # Convert back to flam units
    # spectrum.convert_from_ADUrate_to_flam()
    spectrum.my_logger.info(
        f"\n\tOrder0 total shift: {pixel_shift:.3f}pix"
        f"\n\tD = {D:.3f} mm (default: DISTANCE2CCD = {parameters.DISTANCE2CCD:.2f} "
        f"+/- {parameters.DISTANCE2CCD_ERR:.2f} mm, "
        f"{(D - parameters.DISTANCE2CCD) / parameters.DISTANCE2CCD_ERR:.1f} sigma shift)")
    spectrum.header['PIXSHIFT'] = pixel_shift
    spectrum.header['D2CCD'] = D
    return lambdas


if __name__ == "__main__":
    import doctest

    doctest.testmod()

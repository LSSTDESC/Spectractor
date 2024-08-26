from scipy.signal import argrelextrema, savgol_filter
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import string
import astropy
import warnings
import itertools
warnings.filterwarnings('ignore', category=astropy.io.fits.card.VerifyWarning, append=True)

from spectractor import parameters
from spectractor.config import set_logger, load_config, update_derived_parameters
from spectractor.extractor.dispersers import Hologram
from spectractor.extractor.targets import load_target
from spectractor.tools import (ensure_dir, load_fits, plot_image_simple, plot_table_in_axis,
                               find_nearest, plot_spectrum_simple, fit_poly1d_legendre, gauss,
                               rescale_x_to_legendre, fit_multigauss_and_bgd, multigauss_and_bgd, multigauss_and_bgd_jacobian)
from spectractor.extractor.psf import load_PSF
from spectractor.extractor.chromaticpsf import ChromaticPSF
from spectractor.simulation.adr import adr_calib, flip_and_rotate_adr_to_image_xy_coordinates
from spectractor.simulation.throughput import TelescopeTransmission
from spectractor.fit.fitter import FitWorkspace, FitParameters, run_minimisation

from lsst.utils.threads import disable_implicit_threading
disable_implicit_threading()


fits_mappings = {'config': 'CONFIG',
                 'date_obs': 'DATE-OBS',
                 'expo': 'EXPTIME',
                 'airmass': 'AIRMASS',
                 'disperser_label': 'GRATING',
                 'units': 'UNIT2',
                 'rotation_angle': 'ROTANGLE',
                 'dec': 'DEC',
                 'hour_angle': 'HA',
                 'temperature': 'OUTTEMP',
                 'pressure': 'OUTPRESS',
                 'humidity': 'OUTHUM',
                 'lambda_ref': 'LBDA_REF',
                 'parallactic_angle': 'PARANGLE',
                 'filter_label': 'FILTER',
                 'camera_angle': 'CAM_ROT',
                 'spectrogram_x0': 'S_X0',
                 'spectrogram_y0': 'S_Y0',
                 'spectrogram_xmin': 'S_XMIN',
                 'spectrogram_xmax': 'S_XMAX',
                 'spectrogram_ymin': 'S_YMIN',
                 'spectrogram_ymax': 'S_YMAX',
                 'spectrogram_Nx': 'S_NX',
                 'spectrogram_Ny': 'S_NY',
                 'spectrogram_deg': 'S_DEG',
                 'spectrogram_saturation': 'S_SAT',
                 'order': 'S_ORDER'
                 }


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
    data_next_order: array
        Spectrum amplitude array for next diffraction order in self.units units.
    err_next_order: array
        Spectrum amplitude uncertainties for next diffraction order in self.units units.
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
    camera_angle: float
        The North-West axe angle with respect to the camera horizontal axis in degrees.
    lines: Lines
        Lines instance that contains data on the emission or absorption lines to be searched and fitted in the spectrum.
    header: Fits.Header
        FITS file header.
    disperser: Disperser
        Disperser instance that describes the disperser.
    target: Target
        Target instance that describes the current exposure.
    dec: float
        Declination coordinate of the current exposure in degrees.
    hour_angle float
        Hour angle coordinate of the current exposure in degrees.
    temperature: float
        Outside temperature in Celsius degrees.
    pressure: float
        Outside pressure in hPa.
    humidity: float
        Outside relative humidity in fraction of one.
    throughput: callable
        Instrumental throughput of the telescope.
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

    def __init__(self, file_name="", image=None, order=1, target=None, config="", fast_load=False,
                 spectrogram_file_name_override=None,
                 psf_file_name_override=None,):
        """ Class used to store information and methods relative to spectra and their extraction.
        If a file name is provided, for Spectractor software version strictly below 2.4 one must provide
        a config file also, otherwise do not set a config file (default).
        Config parameters are loaded from file header since version 2.4.

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
        config: str, optional
            If empty, load the config from the spectrum file if it exists, otherwise load the config from the given config file (deftault: '').

        Examples
        --------
        Load a spectrum from a fits file
        >>> s = Spectrum(file_name='./tests/data/reduc_20170530_134_spectrum.fits', config="")
        >>> print(s.order)
        1
        >>> print(s.target.label)
        HD111980
        >>> print(s.disperser_label)
        HoloAmAg

        Load a spectrum from a fits image file
        >>> from spectractor.extractor.images import Image
        >>> image = Image('tests/data/reduc_20170605_028.fits', target_label='PNG321.0+3.9', config="./config/ctio.ini")
        >>> s = Spectrum(image=image)
        >>> print(s.target.label)
        PNG321.0+3.9
        """
        self.fast_load = fast_load
        self.my_logger = set_logger(self.__class__.__name__)
        self.config = config
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
        self.lambda_ref = 550
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
        self.camera_angle = 0
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
        self.data_next_order = None
        self.err_next_order = None
        self.dec = None
        self.hour_angle = None
        self.temperature = None
        self.pressure = None
        self.humidity = None
        self.parallactic_angle = None
        self.filename = file_name
        if file_name != "":
            self.load_spectrum(file_name,
                               spectrogram_file_name_override=spectrogram_file_name_override,
                               psf_file_name_override=psf_file_name_override, fast_load=fast_load)
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
            self.camera_angle = parameters.OBS_CAMERA_ROTATION
            self.my_logger.info('\n\tSpectrum info copied from image')
            self.dec = image.dec
            self.hour_angle = image.hour_angle
            self.temperature = image.temperature
            self.pressure = image.pressure
            self.humidity = image.humidity
            self.parallactic_angle = image.parallactic_angle
            self.adr_params = [self.dec, self.hour_angle, self.temperature, self.pressure,
                               self.humidity, self.airmass]

        self.throughput = self.load_filter()
        if self.target is not None and len(self.target.spectra) > 0:
            spec = self.target.spectra[0] * self.throughput.transmission(self.target.wavelengths[0])
            lambda_ref = np.sum(self.target.wavelengths[0] * spec) / np.sum(spec)
            self.lambda_ref = lambda_ref
            self.header['LBDA_REF'] = lambda_ref

    def convert_from_ADUrate_to_flam(self):
        """Convert units from ADU/s to erg/s/cm^2/nm.
        The SED is supposed to be in flam units ie erg/s/cm^2/nm

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits', config="./config/ctio.ini")
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
        if self.data_next_order is not None:
            self.data_next_order /= ldl
            self.err_next_order /= ldl
        self.units = 'erg/s/cm$^2$/nm'

    def convert_from_flam_to_ADUrate(self):
        """Convert units from erg/s/cm^2/nm to ADU/s.
        The SED is supposed to be in flam units ie erg/s/cm^2/nm

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits', config="./config/ctio.ini")
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
        if self.data_next_order is not None:
            self.data_next_order *= ldl
            self.err_next_order *= ldl
        self.units = 'ADU/s'

    def load_filter(self):
        """Load filter properties and set relevant LAMBDA_MIN and LAMBDA_MAX values.

        Examples
        --------
        >>> s = Spectrum(config="./config/ctio.ini")
        >>> s.filter_label = 'FGB37'
        >>> _ = s.load_filter()

        .. doctest::
            :hide:

            >>> assert np.isclose(parameters.LAMBDA_MIN, 358, atol=1)
            >>> assert np.isclose(parameters.LAMBDA_MAX, 760, atol=1)

        """
        t = TelescopeTransmission(filter_label=self.filter_label)
        if self.filter_label != "" and "empty" not in self.filter_label.lower():
            t.reset_lambda_range(transmission_threshold=1e-4)
        return t

    def plot_spectrum(self, ax=None, xlim=None, live_fit=False, label='', force_lines=False, calibration_only=False):
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
        calibration_only: bool
            Plot only the lines used for calibration if True (default: False).

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170530_134_spectrum.fits')
        >>> s.plot_spectrum(xlim=[500,900], live_fit=False, force_lines=True)
        """
        if ax is None:
            doplot = True
            plt.figure(figsize=[12, 6])
            ax = plt.gca()
        else:
            doplot = False
        if label == '':
            label = f'Order {self.order:d} spectrum\n' \
                    r'$D_{\mathrm{CCD}}=' \
                    rf'{self.disperser.D:.2f}\,$mm'
        if self.x0 is not None:
            label += rf', $x_0={self.x0[0]:.2f}\,$pix'
        title = self.target.label
        if self.data_next_order is not None and np.sum(np.abs(self.data_next_order)) > 0.05 * np.sum(np.abs(self.data)):
            distance = self.disperser.grating_lambda_to_pixel(self.lambdas, self.x0, order=parameters.SPEC_ORDER+1)
            max_index = np.argmin(np.abs(distance + self.x0[0] - parameters.CCD_IMSIZE))
            plot_spectrum_simple(ax, self.lambdas[:max_index], self.data_next_order[:max_index], data_err=self.err_next_order[:max_index],
                                 xlim=xlim, label=f'Order {parameters.SPEC_ORDER+1} spectrum', linestyle="--", lw=1, color="firebrick")
        plot_spectrum_simple(ax, self.lambdas, self.data, data_err=self.err, xlim=xlim, label=label,
                             title=title, units=self.units, lw=1, linestyle="-")
        if len(self.target.spectra) > 0:
            for k in range(len(self.target.spectra)):
                plot_indices = np.logical_and(self.target.wavelengths[k] > np.min(self.lambdas),
                                              self.target.wavelengths[k] < np.max(self.lambdas))
                s = self.target.spectra[k] / np.max(self.target.spectra[k][plot_indices]) * np.max(self.data)
                ax.plot(self.target.wavelengths[k], s, lw=2, label=f'Tabulated spectra #{k}')
        if self.lambdas is not None:
            self.lines.plot_detected_lines(ax)
        if self.lines is not None and len(self.lines.table) > 0:
            self.my_logger.info(f"\n{self.lines.table}")
        if self.lambdas is not None and self.lines is not None:
            self.lines.plot_atomic_lines(ax, fontsize=12, force=force_lines, calibration_only=calibration_only)
        ax.legend(loc='best')
        if self.filters is not None:
            ax.get_legend().set_title(self.filters)
        plt.gcf().tight_layout()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            plt.gcf().savefig(os.path.join(parameters.LSST_SAVEFIGPATH, f'{self.target.label}_spectrum.pdf'))
        if parameters.DISPLAY and doplot:  # pragma: no cover
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
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits', config="ctio.ini")
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
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.PdfPages:  # pragma: no cover
            parameters.PdfPages.savefig()

    def plot_spectrum_summary(self, xlim=None, figsize=(12, 12), save_as=''):
        """Plot spectrum with emission and absorption lines.

        Parameters
        ----------
        xlim: list, optional
            List of minimum and maximum abscisses (default: None).
        figsize: tuple
            Figure size (default: (12, 12)).
        save_as : str, optional
            Path to save the figure to, if specified.

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170530_134_spectrum.fits')
        >>> s.plot_spectrum_summary()
        """
        if not np.any([line.fitted for line in self.lines.lines]):
            fwhm_func = interp1d(self.chromatic_psf.table['lambdas'],
                                 self.chromatic_psf.table['fwhm'],
                                 fill_value=(parameters.CALIB_PEAK_WIDTH, parameters.CALIB_PEAK_WIDTH),
                                 bounds_error=False)
            detect_lines(self.lines, self.lambdas, self.data, self.err, fwhm_func=fwhm_func,
                         calibration_lines_only=True)

        def generate_axes(fig):
            tableShrink = 3
            tableGap = 1
            gridspec = fig.add_gridspec(nrows=23, ncols=20)
            axes = {}
            axes['A'] = fig.add_subplot(gridspec[0:3, 0:19])
            axes['C'] = fig.add_subplot(gridspec[3:6, 0:19], sharex=axes['A'])
            axes['B'] = fig.add_subplot(gridspec[6:14, 0:19])
            axes['CA'] = fig.add_subplot(gridspec[0:3, 19:20])
            axes['CC'] = fig.add_subplot(gridspec[3:6, 19:20])
            axes['D'] = fig.add_subplot(gridspec[14:16, 0:19], sharex=axes['B'])
            axes['E'] = fig.add_subplot(gridspec[16+tableGap:23, tableShrink:19-tableShrink])
            return axes

        fig = plt.figure(figsize=figsize)
        axes = generate_axes(fig)
        plt.suptitle(f"{self.target.label} {self.date_obs}", y=0.91, fontsize=16)
        mainPlot = axes['B']
        spectrogramPlot = axes['A']
        spectrogramPlotCb = axes['CA']
        residualsPlot = axes['C']
        residualsPlotCb = axes['CC']
        widthPlot = axes['D']
        tablePlot = axes['E']

        label = f'Order {self.order:d} spectrum\n' \
                r'$D_{\mathrm{CCD}}=' \
                rf'{self.disperser.D:.2f}\,$mm'
        plot_spectrum_simple(mainPlot, self.lambdas, self.data, data_err=self.err, xlim=xlim, label=label,
                             title='', units=self.units, lw=1, linestyle="-")
        if len(self.target.spectra) > 0:
            plot_indices = np.logical_and(self.target.wavelengths[0] > np.min(self.lambdas),
                                          self.target.wavelengths[0] < np.max(self.lambdas))
            s = self.target.spectra[0] / np.max(self.target.spectra[0][plot_indices]) * np.max(self.data)
            mainPlot.plot(self.target.wavelengths[0], s, lw=2, label='Normalized\nCALSPEC spectrum')
        self.lines.plot_atomic_lines(mainPlot, fontsize=12, force=False, calibration_only=True)
        self.lines.plot_detected_lines(mainPlot, calibration_only=True)

        table = self.lines.build_detected_line_table(calibration_only=True)
        plot_table_in_axis(tablePlot, table)

        mainPlot.legend()

        widthPlot.plot(self.lambdas, np.array(self.chromatic_psf.table['fwhm']), "r-", lw=2)
        widthPlot.set_ylabel("FWHM [pix]")
        widthPlot.set_xlabel(r'$\lambda$ [nm]')
        widthPlot.grid()

        spectrogram = np.copy(self.spectrogram)
        res = self.spectrogram_residuals.reshape((-1, self.spectrogram_Nx))
        std = np.std(res)
        if spectrogram.shape[0] != res.shape[0]:
            margin = (spectrogram.shape[0] - res.shape[0]) // 2
            spectrogram = spectrogram[margin:-margin]
        plot_image_simple(spectrogramPlot, data=spectrogram, title='Data',
                          aspect='auto', cax=spectrogramPlotCb, units='ADU/s', cmap='viridis')
        spectrogramPlot.set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
        spectrogramPlot.grid(False)
        plot_image_simple(residualsPlot, data=res, vmin=-5 * std, vmax=5 * std, title='(Data-Model)/Err',
                          aspect='auto', cax=residualsPlotCb, units=r'$\sigma$', cmap='bwr')
        residualsPlot.set_title('(Data-Model)/Err', fontsize=10, loc='center', color='black', y=0.8)
        residualsPlot.grid(False)

        # hide the tick labels in the plots which share an x axis
        for label in itertools.chain(mainPlot.get_xticklabels(), residualsPlot.get_xticklabels(), spectrogramPlot.get_xticklabels()):
            label.set_visible(False)

        # align y labels
        for ax in [spectrogramPlot, residualsPlot, mainPlot, widthPlot]:
            ax.yaxis.set_label_coords(-0.05, 0.5)

        fig.subplots_adjust(hspace=0)
        if save_as:
            plt.savefig(save_as)
        plt.show()

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
        from spectractor._version import __version__
        self.header["VERSION"] = str(__version__)
        self.header["REBIN"] = parameters.CCD_REBIN
        self.header.comments['REBIN'] = 'original image rebinning factor to get spectrum.'
        self.header['UNIT1'] = "nanometer"
        self.header['UNIT2'] = self.units
        self.header['COMMENTS'] = 'First column gives the wavelength in unit UNIT1, ' \
                                  'second column gives the spectrum in unit UNIT2, ' \
                                  'third column the corresponding errors.'
        hdu1 = fits.PrimaryHDU()
        hdu1.header = self.header

        for attribute, header_key in fits_mappings.items():
            try:
                value = getattr(self, attribute)
            except AttributeError:
                self.my_logger.warning(f"Failed to get {attribute}")
                continue
            if isinstance(value, astropy.coordinates.angles.Angle):
                value = value.degree
            hdu1.header[header_key] = value
            # print(f"Set header key {header_key} to {value} from attr {attribute}")

        extnames = ["SPECTRUM", "SPEC_COV", "ORDER2", "ORDER0"]  # spectrum data
        extnames += ["S_DATA", "S_ERR", "S_BGD", "S_BGD_ER", "S_FIT", "S_RES"]  # spectrogram data
        extnames += ["PSF_TAB"]  # PSF parameter table
        extnames += ["LINES"]  # spectroscopic line table
        extnames += ["CONFIG"]  # config parameters
        hdus = {"SPECTRUM": hdu1}
        for k, extname in enumerate(extnames):
            if extname == "SPECTRUM":
                hdus[extname].data = [self.lambdas, self.data, self.err]
                continue
            hdus[extname] = fits.ImageHDU()
            if extname == "SPEC_COV":
                hdus[extname].data = self.cov_matrix
            elif extname == "ORDER2":
                hdus[extname].data = [self.lambdas, self.data_next_order, self.err_next_order]
            elif extname == "ORDER0":
                hdus[extname].data = self.target.image
                hdus[extname].header["IM_X0"] = self.target.image_x0
                hdus[extname].header["IM_Y0"] = self.target.image_y0
            elif extname == "S_DATA":
                hdus[extname].data = self.spectrogram
                hdus[extname].header['UNIT1'] = self.units
            elif extname == "S_ERR":
                hdus[extname].data = self.spectrogram_err
            elif extname == "S_BGD":
                hdus[extname].data = self.spectrogram_bgd
            elif extname == "S_BGD_ER":
                hdus[extname].data = self.spectrogram_bgd_rms
            elif extname == "S_FIT":
                hdus[extname].data = self.spectrogram_fit
            elif extname == "S_RES":
                hdus[extname].data = self.spectrogram_residuals
            elif extname == "PSF_TAB":
                hdus[extname] = fits.table_to_hdu(self.chromatic_psf.table)
            elif extname == "LINES":
                u.set_enabled_aliases({'flam': u.erg / u.s / u.cm**2 / u.nm,
                                       'reduced': u.dimensionless_unscaled})
                tab = self.lines.build_detected_line_table(amplitude_units=self.units.replace("erg/s/cm$^2$/nm", "flam"))
                hdus[extname] = fits.table_to_hdu(tab)
            elif extname == "CONFIG":
                # HIERARCH and CONTINUE not compatible together in FITS headers
                # We must use short keys built by parametersToShortKeyedDict and use CONTINUE
                # waiting for cfitsio upgrade
                # Store the parameter translation <-> shortkeys
                for item in dir(parameters):
                    if item.startswith("__") or item[0].islower():  # ignore the special stuff
                        continue
                    if item in parameters.STYLE_PARAMETERS:  # don't save plot or verbosity parameters
                        continue
                    try:
                        value = getattr(parameters, item)
                        if isinstance(value, astropy.coordinates.angles.Angle):
                            value = value.degree
                        if isinstance(value, astropy.units.quantity.Quantity):
                            value = value.value
                        if isinstance(value, (np.ndarray, list)):
                            continue
                        if not isinstance(value, (float, int, str, np.ndarray, list)):
                            raise ValueError(f"Can't handle {parameters.item} type {type(parameters.item)}.")
                    except AttributeError:
                        raise KeyError(f"Failed to get parameters.{item}.")
                    if len(item) > 8:
                        fits_longkey = "HIERARCH " + item
                        char_set = string.ascii_uppercase + string.digits
                        while (shortkey := "X_" + ''.join(random.sample(char_set * 6, 6))) in hdus[extname].header.values():
                            pass
                        hdus[extname].header[fits_longkey] = shortkey
                        hdus[extname].header[shortkey] = value
                    else:
                        hdus[extname].header[item] = value
            else:
                raise ValueError(f"Unknown EXTNAME extension: {extname}.")
            hdus[extname].header["EXTNAME"] = extname
        hdu = fits.HDUList([hdus[extname] for extname in extnames])
        ensure_dir(os.path.dirname(output_file_name))
        hdu.writeto(output_file_name, overwrite=overwrite)
        self.my_logger.info(f'\n\tSpectrum saved in {output_file_name}')

    def save_spectrogram(self, output_file_name, overwrite=False):  # pragma: no cover
        """OBOSOLETE: save the spectrogram into a fits file (data, error and background).

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

    def load_spectrum(self, input_file_name, spectrogram_file_name_override=None,
                      psf_file_name_override=None, fast_load=False):
        """Load the spectrum from a fits file (data, error and wavelengths).

        Parameters
        ----------
        input_file_name: str
            Path to the input fits file
        spectrogram_file_name_override : str
            Manually specify a path to the spectrogram file.
        psf_file_name_override : str
            Manually specify a path to the psf file.
        fast_load: bool, optional
            If True, only the spectrum is loaded (not the PSF nor the spectrogram data) (default: False).

        Examples
        --------

        Latest Spectractor output format: do not provide a config file (parameters are loaded from file header)

        >>> from spectractor import parameters
        >>> s = Spectrum(config="")
        >>> s.load_spectrum('tests/data/reduc_20170530_134_spectrum.fits')

        .. doctest::
            :hide:

            >>> assert parameters.OBS_CAMERA_ROTATION == s.header["CAM_ROT"]
            >>> assert parameters.CCD_REBIN == s.header["REBIN"]
            >>> assert s.parallactic_angle == s.header["PARANGLE"]

        Spectractor output format older than version <=2.3: must give the config file

        >>> parameters.VERBOSE = False
        >>> s = Spectrum(config="./config/ctio.ini")
        >>> s.load_spectrum('tests/data/reduc_20170605_028_spectrum.fits')
        >>> print(s.units)
        erg/s/cm$^2$/nm

        .. doctest::
            :hide:

            >>> assert parameters.OBS_CAMERA_ROTATION == s.header["CAM_ROT"]
            >>> assert parameters.CCD_REBIN == s.header["REBIN"]
            >>> assert s.parallactic_angle == s.header["PARANGLE"]

        """
        self.fast_load = fast_load
        if not os.path.isfile(input_file_name):
            raise FileNotFoundError(f'\n\tSpectrum file {input_file_name} not found')

        self.header, raw_data = load_fits(input_file_name)
        # check the version of the file
        if "VERSION" in self.header:
            from spectractor._version import __version__
            from packaging import version
            if self.config != "":
                raise AttributeError(f"With Spectractor above 2.4 do not provide a config file in Spectrum(config=...)."
                                     f"Now config parameters are loaded from the file header. Got {self.config=}.")
            if self.header["VERSION"] != str(__version__):
                self.my_logger.debug(f"\n\tSpectrum file spectractor version {self.header['VERSION']} is "
                                     f"different from current Spectractor software {__version__}.")
            if version.parse(self.header["VERSION"]) < version.parse("3.0"):
                self.my_logger.warning(f"\n\tSpectrum file spectractor version {self.header['VERSION']} is "
                                       f"below Spectractor software 3.0. It may be deprecated.")
            self.load_spectrum_latest(input_file_name)
        else:
            self.my_logger.warning("\n\tNo information about Spectractor software version is given in the header. "
                                   "Use old load function.")
            if self.config == "":
                raise AttributeError("With old Spectrum files you must provide a config file in Spectrum(config=...).")
            self.load_spectrum_older_24(input_file_name, spectrogram_file_name_override=spectrogram_file_name_override,
                                        psf_file_name_override=psf_file_name_override)

    def load_spectrum_older_24(self, input_file_name, spectrogram_file_name_override=None,
                               psf_file_name_override=None, fast_load=False):
        """Load the spectrum from a FITS file (data, error and wavelengths) from Spectrum files generated
        with Spectractor software strictly older than 2.4 version. The parameters must be loaded via the config files.

        Parameters
        ----------
        input_file_name: str
            Path to the input fits file
        spectrogram_file_name_override : str
            Manually specify a path to the spectrogram file.
        psf_file_name_override : str
            Manually specify a path to the psf file.
        fast_load: bool, optional
            If True, only the spectrum is loaded (not the PSF nor the spectrogram data) (default: False).

        Examples
        --------
        >>> s = Spectrum(config="./config/ctio.ini")
        >>> s.load_spectrum('tests/data/reduc_20170605_028_spectrum.fits')
        >>> print(s.units)
        erg/s/cm$^2$/nm
        """
        self.fast_load = fast_load
        if not os.path.isfile(input_file_name):
            raise FileNotFoundError(f'\n\tSpectrum file {input_file_name} not found')

        self.header, raw_data = load_fits(input_file_name)
        self.lambdas = raw_data[0]
        self.lambdas_binwidths = np.gradient(self.lambdas)
        self.data = raw_data[1]
        if len(raw_data) > 2:
            self.err = raw_data[2]
            self.cov_matrix = np.diag(self.err ** 2)

        # set the config parameters first
        if "CAM_ROT" in self.header:
            parameters.OBS_CAMERA_ROTATION = float(self.header["CAM_ROT"])
        else:
            self.my_logger.warning("\n\tNo information about camera rotation in Spectrum header.")
        if self.header.get('CCDREBIN'):
            if parameters.CCD_REBIN != self.header.get('CCDREBIN'):
                raise ValueError("Different values of rebinning parameters between config file and header. Choose.")
            parameters.CCD_REBIN = self.header.get('CCDREBIN')
        if self.header.get('D2CCD'):
            parameters.DISTANCE2CCD = float(self.header.get('D2CCD'))

        # set the simple items from the mappings. More complex items, i.e.
        # those needing function calls, follow
        for attribute, header_key in fits_mappings.items():
            if self.header.get(header_key) is not None:
                setattr(self, attribute, self.header.get(header_key))
            else:
                self.my_logger.warning(f'\n\tFailed to set spectrum attribute {attribute} using header {header_key}')

        # set the more complex items by hand here
        if self.header.get('TARGET'):
            self.target = load_target(self.header.get('TARGET'), verbose=parameters.VERBOSE)
            self.lines = self.target.lines
        if self.header.get('TARGETX') and self.header.get('TARGETY'):
            self.x0 = [self.header.get('TARGETX'), self.header.get('TARGETY')]  # should be a tuple not a list
        self.my_logger.info(f'\n\tLoading disperser {self.disperser_label}...')
        self.disperser = Hologram(self.disperser_label, D=parameters.DISTANCE2CCD,
                                  data_dir=parameters.DISPERSER_DIR, verbose=parameters.VERBOSE)
        self.my_logger.info(f'\n\tSpectrum loaded from {input_file_name}')
        if parameters.OBS_OBJECT_TYPE == "STAR":
            self.adr_params = [self.dec, self.hour_angle, self.temperature,
                               self.pressure, self.humidity, self.airmass]

        self.psf = load_PSF(psf_type=parameters.PSF_TYPE, target=self.target)
        self.chromatic_psf = ChromaticPSF(self.psf, self.spectrogram_Nx, self.spectrogram_Ny,
                                          x0=self.spectrogram_x0, y0=self.spectrogram_y0,
                                          deg=self.spectrogram_deg, saturation=self.spectrogram_saturation)
        if 'PSF_REG' in self.header and float(self.header["PSF_REG"]) > 0:
            self.chromatic_psf.opt_reg = float(self.header["PSF_REG"])

        # original, hard-coded spectrogram/table relative paths
        spectrogram_file_name = input_file_name.replace('spectrum', 'spectrogram')
        psf_file_name = input_file_name.replace('spectrum.fits', 'table.csv')
        if spectrogram_file_name_override and psf_file_name_override:
            self.fast_load = False
            spectrogram_file_name = spectrogram_file_name_override
            psf_file_name = psf_file_name_override

        if not self.fast_load:
            with fits.open(input_file_name) as hdu_list:
                # load other spectrum info
                if len(hdu_list) > 1:
                    self.cov_matrix = hdu_list["SPEC_COV"].data
                    if len(hdu_list) > 2:
                        _, self.data_next_order, self.err_next_order = hdu_list["ORDER2"].data
                        if len(hdu_list) > 3:
                            self.target.image = hdu_list["ORDER0"].data
                            self.target.image_x0 = float(hdu_list["ORDER0"].header["IM_X0"])
                            self.target.image_y0 = float(hdu_list["ORDER0"].header["IM_Y0"])
                # load spectrogram info
                if len(hdu_list) > 4:
                    self.spectrogram = hdu_list["S_DATA"].data
                    self.spectrogram_err = hdu_list["S_ERR"].data
                    self.spectrogram_bgd = hdu_list["S_BGD"].data
                    if len(hdu_list) > 7:
                        self.spectrogram_bgd_rms = hdu_list["S_BGD_ER"].data
                        self.spectrogram_fit = hdu_list["S_FIT"].data
                        self.spectrogram_residuals = hdu_list["S_RES"].data
                elif os.path.isfile(spectrogram_file_name):
                    self.my_logger.info(f'\n\tLoading spectrogram from {spectrogram_file_name}...')
                    self.load_spectrogram(spectrogram_file_name)
                else:
                    raise FileNotFoundError(f"\n\tNo spectrogram info in {input_file_name} "
                                            f"and not even a spectrogram file {spectrogram_file_name}.")
                if "PSF_TAB" in hdu_list:
                    self.chromatic_psf.init_from_table(Table.read(hdu_list["PSF_TAB"]),
                                                       saturation=self.spectrogram_saturation)
                elif os.path.isfile(psf_file_name):  # retro-compatibility
                    self.my_logger.info(f'\n\tLoading PSF from {psf_file_name}...')
                    self.load_chromatic_psf(psf_file_name)
                else:
                    raise FileNotFoundError(f"\n\tNo PSF info in {input_file_name} "
                                            f"and not even a PSF file {psf_file_name}.")
                if "LINES" in hdu_list:
                    self.lines.table = Table.read(hdu_list["LINES"], unit_parse_strict="silent")

    def load_spectrum_latest(self, input_file_name):
        """Load the spectrum from a FITS file (data, error and wavelengths) from Spectrum files generated
        with Spectractor software above or equal 2.4 version. The parameters are loaded via the FITS file header
        and overwrites those loaded via the config file.

        Parameters
        ----------
        input_file_name: str
            Path to the input fits file

        Examples
        --------
        >>> s = Spectrum(config="")
        >>> s.load_spectrum('tests/data/reduc_20170530_134_spectrum.fits')
        >>> print(s.units)
        erg/s/cm$^2$/nm

        .. doctest::
            :hide:

            >>> assert parameters.OBS_CAMERA_ROTATION == s.header["CAM_ROT"]
            >>> assert parameters.CCD_REBIN == s.header["REBIN"]
            >>> assert parameters.OBS_OBJECT_TYPE == "STAR"
            >>> assert s.parallactic_angle == s.header["PARANGLE"]

        """
        self.header, raw_data = load_fits(input_file_name)
        self.lambdas = raw_data[0]
        self.lambdas_binwidths = np.gradient(self.lambdas)
        self.data = raw_data[1]
        if len(raw_data) > 2:
            self.err = raw_data[2]
            self.cov_matrix = np.diag(self.err ** 2)

        # set the config parameters first
        param_header, _ = load_fits(input_file_name, hdu_index="CONFIG")
        for key, value in param_header.items():
            if "X_" not in key and (not isinstance(param_header[key], str) or (isinstance(param_header[key], str) and "X_" not in param_header[key])):
                setattr(parameters, key, value)
            elif "X_" in key:
                continue
            elif "X_" in param_header[key]:
                setattr(parameters, key, param_header[value])
            else:
                continue
        update_derived_parameters()
        # loaded parameters have already been rebinned normally
        # if parameters.CCD_REBIN > 1:
        #     apply_rebinning_to_parameters()

        # set the simple items from the mappings. More complex items, i.e.
        # those needing function calls, follow
        for attribute, header_key in fits_mappings.items():
            if self.header.get(header_key) is not None:
                setattr(self, attribute, self.header.get(header_key))
            else:
                self.my_logger.warning(f'\n\tFailed to set spectrum attribute {attribute} using header {header_key}')

        # set the more complex items by hand here
        if self.header.get('TARGET'):
            self.target = load_target(self.header.get('TARGET'), verbose=parameters.VERBOSE)
            self.lines = self.target.lines
        if self.header.get('TARGETX') and self.header.get('TARGETY'):
            self.x0 = [self.header.get('TARGETX'), self.header.get('TARGETY')]  # should be a tuple not a list
        self.my_logger.info(f'\n\tLoading disperser {self.disperser_label}...')
        self.disperser = Hologram(self.disperser_label, D=parameters.DISTANCE2CCD,
                                  data_dir=parameters.DISPERSER_DIR, verbose=parameters.VERBOSE)
        self.my_logger.info(f'\n\tSpectrum loaded from {input_file_name}')
        if parameters.OBS_OBJECT_TYPE == "STAR":
            self.adr_params = [self.dec, self.hour_angle, self.temperature,
                               self.pressure, self.humidity, self.airmass]

        self.psf = load_PSF(psf_type=parameters.PSF_TYPE, target=self.target)
        self.chromatic_psf = ChromaticPSF(self.psf, self.spectrogram_Nx, self.spectrogram_Ny,
                                          x0=self.spectrogram_x0, y0=self.spectrogram_y0,
                                          deg=self.spectrogram_deg, saturation=self.spectrogram_saturation)
        if 'PSF_REG' in self.header and float(self.header["PSF_REG"]) > 0:
            self.chromatic_psf.opt_reg = float(self.header["PSF_REG"])

        if not self.fast_load:
            with fits.open(input_file_name) as hdu_list:
                # load other spectrum info
                self.cov_matrix = hdu_list["SPEC_COV"].data
                _, self.data_next_order, self.err_next_order = hdu_list["ORDER2"].data
                self.target.image = hdu_list["ORDER0"].data
                self.target.image_x0 = float(hdu_list["ORDER0"].header["IM_X0"])
                self.target.image_y0 = float(hdu_list["ORDER0"].header["IM_Y0"])
                # load spectrogram info
                self.spectrogram = hdu_list["S_DATA"].data
                self.spectrogram_err = hdu_list["S_ERR"].data
                self.spectrogram_bgd = hdu_list["S_BGD"].data
                self.spectrogram_bgd_rms = hdu_list["S_BGD_ER"].data
                self.spectrogram_fit = hdu_list["S_FIT"].data
                self.spectrogram_residuals = hdu_list["S_RES"].data
                self.chromatic_psf.init_from_table(Table.read(hdu_list["PSF_TAB"]),
                                                   saturation=self.spectrogram_saturation)
                self.lines.table = Table.read(hdu_list["LINES"], unit_parse_strict="silent")

    def load_spectrogram(self, input_file_name):  # pragma: no cover
        """OBSOLETE: Load the spectrum from a fits file (data, error and wavelengths).

        Parameters
        ----------
        input_file_name: str
            Path to the input fits file

        Examples
        --------
        >>> s = Spectrum(config="./config/ctio.ini")
        >>> s.load_spectrum('tests/data/reduc_20170605_028_spectrum.fits')
        """
        if os.path.isfile(input_file_name):
            with fits.open(input_file_name) as hdu_list:
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
            self.my_logger.info('\n\tSpectrogram loaded from %s' % input_file_name)
        else:
            self.my_logger.warning('\n\tSpectrogram file %s not found' % input_file_name)

    def load_chromatic_psf(self, input_file_name):  # pragma: no cover
        """OBSOLETE: Load the spectrum from a fits file (data, error and wavelengths).

        Parameters
        ----------
        input_file_name: str
            Path to the input fits file

        Examples
        --------
        >>> s = Spectrum('./tests/data/reduc_20170530_134_spectrum.fits')
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

    def compute_disp_axis_in_spectrogram(self, shift_x, shift_y, angle):
        """Compute the dispersion axis position in a spectrogram.
        Origin is the order 0 centroid.

        Parameters
        ----------
        shift_x: float
            Shift in the x axis direction for order 0 position in pixel.
        shift_y: float
            Shift in the y axis direction for order 0 position in pixel.
        angle: float
            Main dispersion axis angle in degrees.

        Returns
        -------
        Dx: array_like
            Position array along x axis of spectrogram
        Dy_disp_axis: array_like
            Dispersion axis position along y axis of spectrogram with respect to Dx.

        Examples
        --------
        >>> s = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> Dx, Dy_disp_axis = s.compute_disp_axis_in_spectrogram(0, 0, 0)
        >>> Dx[0] == -s.spectrogram_x0
        True
        >>> Dy_disp_axis[:4]
        array([0., 0., 0., 0.])
        """
        # Distance in x and y with respect to the TRUE order 0 position at lambda_ref
        Dx = np.arange(self.spectrogram_Nx) - self.spectrogram_x0 - shift_x  # distance in (x,y) spectrogram frame for column x
        Dy_disp_axis = np.tan(angle * np.pi / 180) * Dx - shift_y  # disp axis height in spectrogram frame for x with respect to order 0
        return Dx, Dy_disp_axis

    def compute_lambdas_in_spectrogram(self, D, shift_x, shift_y, angle, niter=3, with_adr=True, order=1):
        """Compute the dispersion relation in a spectrogram, using grating dispersion model and ADR,
        for a given diffraction order. Origin is the order 0 centroid.

        Parameters
        ----------
        D: float
            The distance between the CCD and the disperser in mm.
        shift_x: float
            Shift in the x axis direction for order 0 position in pixel.
        shift_y: float
            Shift in the y axis direction for order 0 position in pixel.
        angle: float
            Main dispersion axis angle in degrees.
        niter: int, optional
            Number of iterations to compute ADR (default: 3).
        with_adr: bool, optional
            If True, add ADR effect to grating dispersion model (default: True).
        order: int, optional
            Diffraction order (default: 1).

        Returns
        -------
        lambdas: array_like
            Wavelength array for the given diffraction order.

        Examples
        --------
        >>> s = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> s.x0 = [743, 683]
        >>> s.spectrogram_x0 = -280
        >>> lambdas = s.compute_lambdas_in_spectrogram(58, 0, 0, 0)
        >>> lambdas[:4]  #doctest: +ELLIPSIS
        array([334.874..., 336.022..., 337.170..., 338.318...])
        >>> lambdas_order2 = s.compute_lambdas_in_spectrogram(58, 0, 0, 0, order=2)
        >>> lambdas_order2[:4]  #doctest: +ELLIPSIS
        array([175.248..., 175.661..., 176.088..., 176.527...])
        """
        # Distance in x and y with respect to the true order 0 position at lambda_ref
        Dx, Dy_disp_axis = self.compute_disp_axis_in_spectrogram(shift_x=shift_x, shift_y=shift_y, angle=angle)
        distance = np.sign(Dx) * np.sqrt(Dx * Dx + Dy_disp_axis * Dy_disp_axis)  # algebraic distance along dispersion axis

        # Wavelengths using the order 0 shifts (ADR has no impact as it shifts order 0 and order p equally)
        new_x0 = [self.x0[0] + shift_x, self.x0[1] + shift_y]
        # First guess of wavelengths
        self.disperser.D = np.copy(D)
        lambdas = self.disperser.grating_pixel_to_lambda(distance, new_x0, order=order)
        # Evaluate ADR
        if with_adr:
            for k in range(niter):
                adr_ra, adr_dec = adr_calib(lambdas, self.adr_params, parameters.OBS_LATITUDE,
                                            lambda_ref=self.lambda_ref)
                adr_u, adr_v = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=angle)

                # Compute lambdas at pixel column x
                lambdas = self.disperser.grating_pixel_to_lambda(distance - adr_u, new_x0, order=order)
        return lambdas

    def compute_dispersion_in_spectrogram(self, lambdas, shift_x, shift_y, angle, niter=3, with_adr=True, order=1):
        """Compute the dispersion relation in a spectrogram, using grating dispersion model and ADR, for a given
        diffraction order. Origin is the order 0 centroid.

        Parameters
        ----------
        lambdas: array_like
            Wavelength array for the given diffraction order.
        shift_x: float
            Shift in the x axis direction for order 0 position in pixel.
        shift_y: float
            Shift in the y axis direction for order 0 position in pixel.
        angle: float
            Main dispersion axis angle in degrees.
        niter: int, optional
            Number of iterations to compute ADR (default: 3).
        with_adr: bool, optional
            If True, add ADR effect to grating dispersion model (default: True).
        order: int, optional
            Diffraction order (default: 1).

        Returns
        -------
        dispersion_law: array_like
            Complex array coding the 2D dispersion relation in the spectrogram for the given diffraction order.

        Examples
        --------
        >>> s = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> s.x0 = [743, 683]
        >>> s.spectrogram_x0 = -280
        >>> lambdas = s.compute_lambdas_in_spectrogram(58, 0, 0, 0)
        >>> lambdas[:4]  #doctest: +ELLIPSIS
        array([334.874..., 336.022..., 337.170..., 338.318...])
        >>> dispersion_law = s.compute_dispersion_in_spectrogram(lambdas, 0, 0, 0, order=1)
        >>> dispersion_law[:4]  #doctest: +ELLIPSIS
        array([280.0... +1.0...j, 281.0...+1.0...j,
               282.0...+1.0...j, 283.0... +1.0...j])
        >>> dispersion_law_order2 = s.compute_dispersion_in_spectrogram(lambdas, 0, 0, 0, order=2)
        >>> dispersion_law_order2[:4]  #doctest: +ELLIPSIS
        array([573.6...+1.0...j, 575.8...+1.0...j,
               577.9...+1.0...j, 580.0...+1.0...j])

        """
        new_x0 = [self.x0[0] + shift_x, self.x0[1] + shift_y]
        # Distance (not position) in pixel of wavelength lambda centroid in the (x,y) spectrogram frame
        # with respect to order 0 centroid
        distance_along_disp_axis = self.disperser.grating_lambda_to_pixel(lambdas, x0=new_x0, order=order)
        Dx = distance_along_disp_axis * np.cos(angle * np.pi / 180)
        Dy_disp_axis = distance_along_disp_axis * np.sin(angle * np.pi / 180)
        # Evaluate ADR
        adr_x = np.zeros_like(Dx)
        adr_y = np.zeros_like(Dy_disp_axis)
        if with_adr:
            for k in range(niter):
                adr_ra, adr_dec = adr_calib(lambdas, self.adr_params, parameters.OBS_LATITUDE,
                                            lambda_ref=self.lambda_ref)
                adr_x, adr_y = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=0)

        # Position (not distance) in pixel of wavelength lambda centroid in the (x,y) spectrogram frame
        # with respect to order 0 initial centroid position.
        dispersion_law = (Dx + shift_x + with_adr * adr_x) + 1j * (Dy_disp_axis + with_adr * adr_y + shift_y)
        return dispersion_law


class MultigaussAndBgdFitWorkspace(FitWorkspace):
    def __init__(self, guess, x, data, err, bounds, file_name="", verbose=False, plot=False, live_fit=False, truth=None):
        """

        Parameters
        ----------

        Examples
        --------

        >>> from spectractor.config import load_config
        >>> load_config("default.ini")
        >>> x = np.arange(600.,800.,1)
        >>> x_norm = rescale_x_to_legendre(x)
        >>> p = [20, 0, 0, 0, 20, 650, 3, 40, 750, 5]
        >>> y = multigauss_and_bgd(np.array([x_norm, x]), *p)
        >>> print(f'{y[0]:.2f}')
        20.00
        >>> err = 0.1 * np.sqrt(y)
        >>> guess = (10,0,0,0.1,10,640,2,20,750,7)
        >>> bounds = ((-np.inf,-np.inf,-np.inf,-np.inf,1,600,1,1,600,1),(np.inf,np.inf,np.inf,np.inf,100,800,100,100,800,100))
        >>> w = MultigaussAndBgdFitWorkspace(guess, x, y, err, np.array(bounds).T)
        >>> w = run_multigaussandbgd_minimisation(w, method="newton")
        >>> popt = w.params.values
        >>> assert np.allclose(p, w.params.values, rtol=1e-4, atol=1e-5)
        >>> _ = w.plot_fit()

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np
            from spectractor.tools import multigauss_and_bgd, fit_multigauss_and_bgd
            x = np.arange(600.,800.,1)
            x_norm = rescale_x_to_legendre(x)
            p = [20, 0, 0, 0, 20, 650, 3, 40, 750, 5]
            y = multigauss_and_bgd(np.array([x_norm, x]), *p)
            err = 0.1 * np.sqrt(y)
            guess = (10,0,0,0.1,10,640,2,20,750,7)
            bounds = ((-np.inf,-np.inf,-np.inf,-np.inf,1,600,1,1,600,1),(np.inf,np.inf,np.inf,np.inf,100,800,100,100,800,100))
            w = MultigaussAndBgdFitWorkspace(guess, x, y, err, np.array(bounds).T)
            w = run_multigaussandbgd_minimisation(w, method="newton")
            w.plot_fit()

        """
        bgd_nparams = parameters.CALIB_BGD_NPARAMS
        labels = [f"b_{k}" for k in range(bgd_nparams)]
        for ngauss in range((len(guess) - bgd_nparams) // 3):
            labels += [f"A_{ngauss}", f"x0_{ngauss}", f"sigma_{ngauss}"]

        params = FitParameters(values=guess,labels=labels,bounds=bounds,truth=truth)
        FitWorkspace.__init__(self, params, file_name=file_name, verbose=verbose, plot=plot,
                              live_fit=live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        if data.shape != err.shape:
            raise ValueError(f"Data and uncertainty arrays must have the same shapes. "
                             f"Here data.shape={data.shape} and data_errors.shape={err.shape}.")
        self.x = x
        self.x_norm = rescale_x_to_legendre(x)
        self.xs = np.array([self.x_norm, x])
        self.data = data
        self.err = err

    def simulate(self, *p):
        self.model = multigauss_and_bgd(self.xs, *p)
        return self.x, self.model, np.zeros_like(self.model)

    def jacobian(self, params, epsilon, model_input=None):
        return multigauss_and_bgd_jacobian(self.xs, *params).T


def run_multigaussandbgd_minimisation(w, method="newton"):
    run_minimisation(w, method=method, ftol=1 / w.x.size, xtol=1e-6, niter=50)
    return w


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
        # bgd_mean = float(np.mean(spec_smooth[bgd_index]))
        # bgd_std = float(np.std(spec_smooth[bgd_index]))
        for n in range(bgd_npar):
            guess[n] = fit[n]
            b = abs(baseline_prior * guess[n])
            # b = abs(baseline_prior * np.sqrt(cov[n,n]))
            # CHECK: following is completely inefficient as rtol has no effect when second argument is 0...
            # if np.isclose(b, 0, rtol=1e-2 * bgd_mean):
            #     b = baseline_prior * bgd_std
            #     if np.isclose(b, 0, rtol=1e-2 * bgd_mean):
            #         b = np.inf
            bounds[0][n] = guess[n] - b
            bounds[1][n] = guess[n] + b
        for j in range(len(new_lines_list[k])):
            idx = new_peak_index_list[k][j]
            x_norm = rescale_x_to_legendre(lambdas[idx])
            guess[bgd_npar + 3 * j] = np.sign(guess[bgd_npar + 3 * j]) * abs(spec_smooth[idx] - np.polynomial.legendre.legval(x_norm, guess[:bgd_npar]))
            # guess[bgd_npar + 3 * j] = np.sign(guess[bgd_npar + 3 * j]) * abs(spec_smooth[idx] - np.polyval(guess[:bgd_npar], lambdas[idx]))
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
        # w = MultigaussAndBgdFitWorkspace(guess, lambdas[index], spec[index], sigma, np.array(bounds).T)
        # w = run_multigaussandbgd_minimisation(w, method="newton")
        # popt = w.params.values
        # pcov = w.params.cov
        popt, pcov = fit_multigauss_and_bgd(lambdas[index], spec[index], guess=guess, bounds=bounds, sigma=sigma)
        # noise level defined as the std of the residuals if no error
        x_norm = rescale_x_to_legendre(lambdas[index])
        best_fit_model = multigauss_and_bgd(np.array([x_norm, lambdas[index]]), *popt)
        noise_level = np.std(spec[index] - best_fit_model)
        # otherwise mean of error bars of bgd lateral bands
        if sigma is not None:
            chisq = np.sum((best_fit_model - spec[index]) ** 2 / (sigma * sigma))
        else:
            chisq = np.sum((best_fit_model - spec[index]) ** 2)
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
            signal_level = popt[bgd_npar + 3 * j]  # multigauss_and_bgd(peak_pos, *popt) - np.polyval(popt[:bgd_npar], peak_pos)
            snr = np.abs(signal_level / noise_level)
            # save fit results
            line.fitted = True
            line.fit_index = index
            line.fit_lambdas = lambdas[index]

            x_norm = rescale_x_to_legendre(lambdas[index])

            x_step = 0.1  # nm
            x_int = np.arange(max(np.min(lambdas), peak_pos - 5 * np.abs(popt[bgd_npar + 3 * j + 2])),
                              min(np.max(lambdas), peak_pos + 5 * np.abs(popt[bgd_npar + 3 * j + 2])), x_step)
            x_int_norm = rescale_x_to_legendre(x_int)

            # jmin and jmax a bit larger than x_int to avoid extrapolation
            jmin = max(0, int(np.argmin(np.abs(lambdas - (x_int[0] - x_step))) - 2))
            jmax = min(len(lambdas), int(np.argmin(np.abs(lambdas - (x_int[-1] + x_step))) + 2))
            if jmax-2 < jmin+2:  # decreasing order
                jmin, jmax = max(0, jmax-4), min(len(lambdas), jmin+4)
            spectr_data = interp1d(lambdas[jmin:jmax], spec[jmin:jmax],
                                   bounds_error=False, fill_value="extrapolate")(x_int)

            Continuum = np.polynomial.legendre.legval(x_int_norm, popt[:bgd_npar])
            # Continuum = np.polyval(popt[:bgd_npar], x_int)
            Gauss = gauss(x_int, *popt[bgd_npar + 3 * j:bgd_npar + 3 * j + 3])

            Y = -Gauss / Continuum
            Ydata = 1 - spectr_data / Continuum

            line.fit_eqwidth_mod = integrate.simpson(Y, x=x_int)  # sol1
            line.fit_eqwidth_data = integrate.simpson(Ydata, x=x_int)  # sol2

            line.fit_popt = popt
            line.fit_pcov = pcov
            line.fit_popt_gaussian = popt[bgd_npar + 3 * j:bgd_npar + 3 * j + 3]
            line.fit_pcov_gaussian = pcov[bgd_npar + 3 * j:bgd_npar + 3 * j + 3, bgd_npar + 3 * j:bgd_npar + 3 * j + 3]
            line.fit_gauss = gauss(lambdas[index], *popt[bgd_npar + 3 * j:bgd_npar + 3 * j + 3])

            line.fit_bgd = np.polynomial.legendre.legval(x_norm, popt[:bgd_npar])
            # line.fit_bgd = np.polyval(popt[:bgd_npar], x_int)
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
        lines.plot_detected_lines(ax)
    lines.table = lines.build_detected_line_table()
    lines.my_logger.debug(f"\n{lines.table}")
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


def calibrate_spectrum(spectrum, with_adr=False, niter=5, grid_search=False):
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
    niter: int, optional
        Number of iterations for ADR accurate evaluation (default: 5).

    Returns
    -------
    lambdas: array_like
        The new wavelength array in nm.

    Examples
    --------
    >>> spectrum = Spectrum('tests/data/reduc_20170530_134_spectrum.fits', config="")
    >>> parameters.LAMBDA_MIN = 550
    >>> parameters.LAMBDA_MAX = 800
    >>> lambdas = calibrate_spectrum(spectrum, with_adr=False)
    >>> spectrum.plot_spectrum()

    """
    with_adr = int(with_adr)
    if spectrum.units != "ADU/s":  # go back in ADU/s to remove previous lambda*dlambda normalisation
        spectrum.convert_from_flam_to_ADUrate()
    distance = spectrum.chromatic_psf.get_algebraic_distance_along_dispersion_axis()
    spectrum.lambdas = spectrum.disperser.grating_pixel_to_lambda(distance, spectrum.x0, order=spectrum.order)
    # ADR is x>0 westward and y>0 northward while CTIO images are x>0 westward and y>0 southward
    # Must project ADR along dispersion axis
    if with_adr > 0:
        for k in range(niter):
            adr_ra, adr_dec = adr_calib(spectrum.lambdas, spectrum.adr_params, parameters.OBS_LATITUDE,
                                        lambda_ref=spectrum.lambda_ref)
            adr_u, _ = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec,
                                                                   dispersion_axis_angle=spectrum.rotation_angle)
            spectrum.lambdas = spectrum.disperser.grating_pixel_to_lambda(distance - adr_u, spectrum.x0, order=spectrum.order)
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
        chisq += (shift / parameters.PIXSHIFT_PRIOR) ** 2
        spectrum.convert_from_flam_to_ADUrate()
        return chisq

    # grid exploration of the parameters
    # necessary because of the line detection algo
    D = parameters.DISTANCE2CCD
    pixel_shift = 0
    if 'D2CCD' in spectrum.header:
        D = spectrum.header['D2CCD']
    if 'PIXSHIFT' in spectrum.header:
        pixel_shift = spectrum.header['PIXSHIFT']
    D_err = parameters.DISTANCE2CCD_ERR
    D_step = D_err / 2
    pixel_shift_step = parameters.PIXSHIFT_PRIOR / 5
    pixel_shift_prior = parameters.PIXSHIFT_PRIOR
    if grid_search:
        Ds = np.arange(D - 5 * D_err, D + 6 * D_err, D_step)
        pixel_shifts = np.arange(-pixel_shift_prior, pixel_shift_prior + pixel_shift_step, pixel_shift_step)
        chisq_grid = np.zeros((len(Ds), len(pixel_shifts)))
        for i, D in enumerate(Ds):
            for j, pixel_shift in enumerate(pixel_shifts):
                chisq_grid[i, j] = shift_minimizer([D, pixel_shift])
        imin, jmin = np.unravel_index(chisq_grid.argmin(), chisq_grid.shape)
        D = Ds[imin]
        pixel_shift = pixel_shifts[jmin]
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
            fig.tight_layout()
            if parameters.DISPLAY:  # pragma: no cover
                plt.show()
            if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
                fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'D2CCD_x0_fit.pdf'))
    start = np.array([D, pixel_shift])

    # now minimize around the global minimum found previously
    res = optimize.minimize(shift_minimizer, start, args=(), method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-3},
                            bounds=((D - 5 * parameters.DISTANCE2CCD_ERR, D + 5 * parameters.DISTANCE2CCD_ERR), (-2, 2)))
    # error = [parameters.DISTANCE2CCD_ERR, pixel_shift_step]
    # fix = [False, False]
    # m = Minuit(shift_minimizer, start)
    # m.errors = error
    # m.errordef = 1
    # m.fixed = fix
    # m.print_level = 0
    # m.limits = ((D - 5 * parameters.DISTANCE2CCD_ERR, D + 5 * parameters.DISTANCE2CCD_ERR), (-2, 2))
    # m.migrad()
    # D, pixel_shift = np.array(m.values[:])
    D, pixel_shift = res.x
    spectrum.disperser.D = D
    x0 = [x0[0] + pixel_shift, x0[1]]
    spectrum.x0 = x0
    # check success, xO or D on the edge of their priors
    distance = spectrum.chromatic_psf.get_algebraic_distance_along_dispersion_axis(shift_x=pixel_shift)
    lambdas = spectrum.disperser.grating_pixel_to_lambda(distance - with_adr * adr_u, x0=x0, order=spectrum.order)
    spectrum.lambdas = lambdas
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

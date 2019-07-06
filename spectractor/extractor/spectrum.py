from scipy.signal import argrelextrema, savgol_filter

from spectractor.extractor.images import *


class Spectrum:

    def __init__(self, file_name="", image=None, order=1, target=None):
        """ Spectrum class used to store information and methods
        relative to spectra nd their extraction.

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
        >>> image = Image('tests/data/reduc_20170605_028.fits', target='3C273')
        >>> s = Spectrum(image=image)
        >>> print(s.target.label)
        3C273
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.target = target
        self.data = None
        self.err = None
        self.x0 = None
        self.pixels = None
        self.lambdas = None
        self.lambdas_binwidths = None
        self.lambdas_indices = None
        self.order = order
        self.chromatic_psf = None
        self.filter = None
        self.filters = None
        self.units = 'ADU/s'
        self.gain = parameters.CCD_GAIN
        self.chromatic_psf = ChromaticPSF1D(1, 1, deg=1, saturation=1)
        self.rotation_angle = 0
        if file_name != "":
            self.filename = file_name
            self.load_spectrum(file_name)
        if image is not None:
            self.header = image.header
            self.date_obs = image.date_obs
            self.airmass = image.airmass
            self.expo = image.expo
            self.filters = image.filters
            self.filter = image.filter
            self.disperser_label = image.disperser_label
            self.disperser = image.disperser
            self.target = image.target
            self.lines = self.target.lines
            self.x0 = image.target_pixcoords
            self.target_pixcoords = image.target_pixcoords
            self.target_pixcoords_rotated = image.target_pixcoords_rotated
            self.units = image.units
            self.gain = image.gain
            self.my_logger.info('\n\tSpectrum info copied from image')
        self.load_filter()

    def convert_from_ADUrate_to_flam(self):
        """Convert units from ADU/s to erg/s/cm^2/nm.
        The SED is supposed to be in flam units ie erg/s/cm^2/nm

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
        >>> s.convert_from_ADUrate_to_flam()
        >>> assert np.max(s.data) < 1e-2
        >>> assert np.max(s.err) < 1e-2

        """

        self.data = self.data / parameters.FLAM_TO_ADURATE
        self.data /= self.lambdas * self.lambdas_binwidths
        if self.err is not None:
            self.err = self.err / parameters.FLAM_TO_ADURATE
            self.err /= (self.lambdas * self.lambdas_binwidths)
        self.units = 'erg/s/cm$^2$/nm'

    def convert_from_flam_to_ADUrate(self):
        """Convert units from erg/s/cm^2/nm to ADU/s.
        The SED is supposed to be in flam units ie erg/s/cm^2/nm

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
        >>> s.convert_from_flam_to_ADUrate()
        >>> assert np.max(s.data) > 1e-2
        >>> assert np.max(s.err) > 1e-2

        """
        self.data = self.data * parameters.FLAM_TO_ADURATE
        self.data *= self.lambdas_binwidths * self.lambdas
        if self.err is not None:
            self.err = self.err * parameters.FLAM_TO_ADURATE
            self.err *= self.lambdas_binwidths * self.lambdas
        self.units = 'ADU/s'

    def load_filter(self):
        """Load filter properties and set relevant LAMBDA_MIN and LAMBDA_MAX values.

        Examples
        --------
        >>> s = Spectrum()
        >>> s.filter = 'FGB37'
        >>> s.load_filter()
        >>> assert parameters.LAMBDA_MIN == parameters.FGB37['min']
        >>> assert parameters.LAMBDA_MAX == parameters.FGB37['max']

        """
        for f in parameters.FILTERS:
            if f['label'] == self.filter:
                parameters.LAMBDA_MIN = f['min']
                parameters.LAMBDA_MAX = f['max']
                self.my_logger.info('\n\tLoad filter %s: lambda between %.1f and %.1f' % (
                    f['label'], parameters.LAMBDA_MIN, parameters.LAMBDA_MAX))
                break

    def plot_spectrum(self, ax=None, xlim=None, ylim=None,live_fit=False, label='', force_lines=False,drawclose=True):
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
            label = f'Order {self.order:d} spectrum\nD={self.disperser.D:.2f}mm'
        if self.x0 is not None:
            label += f', x0={self.x0[0]:.2f}pix'
        title = self.target.label
        plot_spectrum_simple(ax, self.lambdas, self.data, data_err=self.err, xlim=xlim, ylim=ylim,label=label,
                             title=title, units=self.units)
        if len(self.target.spectra) > 0:
            for k in range(len(self.target.spectra)):
                s = self.target.spectra[k] / np.max(self.target.spectra[k]) * np.max(self.data)
                #ax.plot(self.target.wavelengths[k], s, lw=2, label='Tabulated spectra #%d' % k)
                ax.plot(self.target.wavelengths[k], s, lw=2)
        if self.lambdas is not None:
            self.lines.plot_detected_lines(ax, print_table=parameters.VERBOSE)
        if self.lambdas is not None and self.lines is not None:
            self.lines.plot_atomic_lines(ax, fontsize=12, force=force_lines)
        ax.legend(loc='upper right')
        if self.filters is not None:
            ax.get_legend().set_title(self.filters)
        if parameters.DISPLAY:
            if live_fit and drawclose:
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
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
        >>> s.save_spectrum('./tests/test.fits')
        >>> assert os.path.isfile('./tests/test.fits')

        Overwrite previous file:
        >>> s.save_spectrum('./tests/test.fits', overwrite=True)
        >>> assert os.path.isfile('./tests/test.fits')
        >>> os.remove('./tests/test.fits')
        """
        self.header['UNIT1'] = "nanometer"
        self.header['UNIT2'] = self.units
        self.header['COMMENTS'] = 'First column gives the wavelength in unit UNIT1, ' \
                                  'second column gives the spectrum in unit UNIT2, ' \
                                  'third column the corresponding errors.'
        save_fits(output_file_name, self.header, [self.lambdas, self.data, self.err], overwrite=overwrite)
        self.my_logger.info('\n\tSpectrum saved in %s' % output_file_name)

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
        hdu2 = fits.ImageHDU()
        hdu3 = fits.ImageHDU()
        hdu1.header = self.header
        hdu1.data = self.spectrogram
        hdu2.data = self.spectrogram_err
        hdu3.data = self.spectrogram_bgd
        hdu = fits.HDUList([hdu1, hdu2, hdu3])
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

            if parameters.OBS_NAME == 'CTIO' or parameters.OBS_NAME == 'LPNHE':
                extract_info_from_CTIO_header(self, self.header)
            elif parameters.OBS_NAME == 'PICDUMIDI':
                extract_info_from_PDM_header(self, self.header)
                self.header['LSHIFT'] = 0.
                self.header['D2CCD'] = parameters.DISTANCE2CCD
                self.header['EXPTIME'] = self.header['EXPOSURE']
                self.header['FILTERS'] = self.header['FILTERS']
                self.header['FILTER1'] = self.header['FILTER1']
                self.header['FILTER2'] = self.header['FILTER2']
                # image.header["FILTER2"] = image.disperser_label #better take the name in the logbook
                self.header['TARGET'] = self.header['OBJECT']
                self.filters=self.header['FILTERS']



            if parameters.OBS_NAME == 'CTIO' or parameters.OBS_NAME == 'LPNHE':
                if self.header['TARGET'] != "":
                    self.target = load_target(self.header['TARGET'], verbose=parameters.VERBOSE)
                    self.lines = self.target.lines
                if self.header['UNIT2'] != "":
                    self.units = self.header['UNIT2']
                if self.header['ROTANGLE'] != "":
                    self.rotation_angle = self.header['ROTANGLE']
                if self.header['TARGETX'] != "" and self.header['TARGETY'] != "":
                    self.x0 = [self.header['TARGETX'], self.header['TARGETY']]



            elif parameters.OBS_NAME == 'PICDUMIDI':
                if self.header['OBJECT'] != "":
                    self.target = load_target(self.header['OBJECT'], verbose=parameters.VERBOSE)
                    self.lines = self.target.lines


            self.my_logger.info('\n\tLoading disperser %s...' % self.disperser_label)
            self.disperser = Hologram(self.header['FILTER2'], data_dir=parameters.HOLO_DIR, verbose=parameters.VERBOSE)



            self.my_logger.info('\n\tSpectrum loaded from %s' % input_file_name)
            self.load_spectrogram(input_file_name.replace('spectrum', 'spectrogram'))
            
            self.my_logger.info('\n\tLoading chromatic psf %s...' % input_file_name.replace('spectrum.fits', 'table.csv'))
            
            self.load_chromatic_psf(input_file_name.replace('spectrum.fits', 'table.csv'))
        else:
            self.my_logger.warning('\n\tSpectrum file %s not found' % input_file_name)

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
            self.spectrogram_x0 = header['S_X0']
            self.spectrogram_y0 = header['S_Y0']
            self.spectrogram_xmin = header['S_XMIN']
            self.spectrogram_xmax = header['S_XMAX']
            self.spectrogram_ymin = header['S_YMIN']
            self.spectrogram_ymax = header['S_YMAX']
            self.spectrogram_deg = header['S_DEG']
            self.spectrogram_saturation = header['S_SAT']
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
        >>> s.load_spectrum('outputs/reduc_20170530_134_spectrum.fits')
        """
        if os.path.isfile(input_file_name):
            self.chromatic_psf = ChromaticPSF1D(self.spectrogram_Nx, self.spectrogram_Ny,
                                                deg=self.spectrogram_deg, saturation=self.spectrogram_saturation)
            self.chromatic_psf.table = Table.read(input_file_name)
            self.my_logger.info('\n\tSpectrogram loaded from %s' % input_file_name)
        else:
            self.my_logger.warning('\n\tSpectrogram file %s not found' % input_file_name)


def calibrate_spectrum(spectrum, xlim=None):
    """Convert pixels into wavelengths given the position of the order 0,
    the data for the spectrum, and the properties of the disperser. Convert the
    spectrum amplitude from ADU rate to flams. Truncate the outputs to the wavelenghts
    between parameters.LAMBDA_MIN and parameters.LAMBDA_MAX.

    Parameters
    ----------
    spectrum: Spectrum
        Spectrum object to calibrate
    xlim: list, optional
        List of minimum and maximum abscisses

    """
    my_logger = set_logger(__name__)
    if xlim is None:
        left_cut, right_cut = [0, spectrum.data.size]
    else:
        left_cut, right_cut = xlim
    spectrum.data = spectrum.data[left_cut:right_cut]
    pixels = spectrum.pixels[left_cut:right_cut] - spectrum.target_pixcoords_rotated[0]
    spectrum.lambdas = spectrum.disperser.grating_pixel_to_lambda(pixels, spectrum.target_pixcoords,
                                                                  order=spectrum.order)
    spectrum.lambdas_binwidths = np.gradient(spectrum.lambdas)
    # Cut spectra
    spectrum.lambdas_indices = \
        np.where(np.logical_and(spectrum.lambdas > parameters.LAMBDA_MIN, spectrum.lambdas < parameters.LAMBDA_MAX))[0]
    spectrum.lambdas = spectrum.lambdas[spectrum.lambdas_indices]
    spectrum.lambdas_binwidths = spectrum.lambdas_binwidths[spectrum.lambdas_indices]
    spectrum.data = spectrum.data[spectrum.lambdas_indices]
    if spectrum.err is not None:
        spectrum.err = spectrum.err[spectrum.lambdas_indices]
    spectrum.convert_from_ADUrate_to_flam()


def detect_lines(lines, lambdas, spec, spec_err=None, fwhm_func=None, snr_minlevel=3, ax=None,
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
    fwhm_func: callable, optional
        The fwhm of the cross spectrum to reset CALIB_PEAK_WIDTH parameter as a function of lambda (default: None)
    snr_minlevel: float
        The minimum signal over noise ratio to consider using a fitted line in the computation of the mean
        shift output and to print it in the outpur table (default: 3)
    ax: Axes, optional
        An Axes instance to over plot the result of the fit (default: None).
    xlim: array, optional
        (min, max) list limiting the wavelength interval where to detect spectral lines (default:
        (parameters.LAMBDA_MIN, parameters.LAMBDA_MAX))

    Returns
    -------
    shift: float
        The mean shift (in nm) between the detected and tabulated lines

    Examples
    --------

    Creation of a mock spectrum with emission and absorption lines
    >>> import numpy as np
    >>> lambdas = np.arange(300,1000,1)
    >>> spectrum = 1e4*np.exp(-((lambdas-600)/200)**2)
    >>> spectrum += HALPHA.gaussian_model(lambdas, A=5000, sigma=3)
    >>> spectrum += HBETA.gaussian_model(lambdas, A=3000, sigma=2)
    >>> spectrum += O2.gaussian_model(lambdas, A=-3000, sigma=7)
    >>> spectrum_err = np.sqrt(spectrum)
    >>> spectrum = np.random.poisson(spectrum)
    >>> spec = Spectrum()
    >>> spec.lambdas = lambdas
    >>> spec.data = spectrum
    >>> spec.err = spectrum_err
    >>> fwhm_func = interp1d(lambdas, 0.01 * lambdas)

    Detect the lines
    >>> lines = Lines([HALPHA, HBETA, O2], hydrogen_only=True,
    ... atmospheric_lines=True, redshift=0, emission_spectrum=True)
    >>> global_chisq = detect_lines(lines, lambdas, spectrum, spectrum_err, fwhm_func=fwhm_func)
    >>> assert(global_chisq < 1)

    Plot the result
    >>> spec.lines = lines
    >>> fig = plt.figure()
    >>> plot_spectrum_simple(plt.gca(), lambdas, spec.data, data_err=spec.err)
    >>> lines.plot_detected_lines(plt.gca())
    >>> if parameters.DISPLAY: plt.show()
    """

    # main settings
    my_logger = set_logger(__name__)
    bgd_npar = parameters.CALIB_BGD_NPARAMS
    peak_width = parameters.CALIB_PEAK_WIDTH
    bgd_width = parameters.CALIB_BGD_WIDTH
    if lines.hydrogen_only:
        peak_width = 7
        bgd_width = 15
    fwhm_to_peak_width_factor = 3
    len_index_to_bgd_npar_factor = 0.12
    baseline_prior = 0.1  # *sigma gaussian prior on base line fit
    # filter the noise
    #plt.errorbar(lambdas,spec,yerr=spec_err)
    spec = np.copy(spec)
    spec = savgol_filter(spec, 5, 2)
    #plt.plot(lambdas,spec)
    #plt.show()
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
        if np.any(np.isnan(spec[index])):
            continue
        extrema = argrelextrema(spec[index], line_strategy)
        if len(extrema[0]) == 0:
            continue
        peak_index = index[0] + extrema[0][0]
        # if several extrema, look for the greatest
        if len(extrema[0]) > 1:
            if line_strategy == np.greater:
                test = -1e20
                for m in extrema[0]:
                    idx = index[0] + m
                    if spec[idx] > test:
                        peak_index = idx
                        test = spec[idx]
            elif line_strategy == np.less:
                test = 1e20
                for m in extrema[0]:
                    idx = index[0] + m
                    if spec[idx] < test:
                        peak_index = idx
                        test = spec[idx]
        # search for first local minima around the local maximum
        # or for first local maxima around the local minimum
        # around +/- 3*peak_width
        index_inf = peak_index - 1  # extrema on the left
        while index_inf > max(0, peak_index - 3 * peak_width):
            test_index = np.arange(index_inf, peak_index, 1).astype(int)
            minm = argrelextrema(spec[test_index], bgd_strategy)
            if len(minm[0]) > 0:
                index_inf = index_inf + minm[0][0]
                break
            else:
                index_inf -= 1
        index_sup = peak_index + 1  # extrema on the right
        while index_sup < min(len(spec) - 1, peak_index + 3 * peak_width):
            test_index = np.arange(peak_index, index_sup, 1).astype(int)
            minm = argrelextrema(spec[test_index], bgd_strategy)
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
        if np.any(np.isnan(spec[index])):
            continue
        # first guess and bounds to fit the line properties and
        # the background with CALIB_BGD_ORDER order polynom
        # guess = [0] * bgd_npar + [0.5 * np.max(spec[index]), lambdas[peak_index],
        #                          0.5 * (line.width_bounds[0] + line.width_bounds[1])]
        bgd_npar = max(parameters.CALIB_BGD_NPARAMS, int(len_index_to_bgd_npar_factor * (index[-1] - index[0])))
        bgd_npar_list.append(bgd_npar)
        guess = [0] * bgd_npar + [0.5 * np.max(spec[index]), line_wavelength,
                                  0.5 * (line.width_bounds[0] + line.width_bounds[1])]
        if line_strategy == np.less:
            guess[bgd_npar] = -0.5 * np.max(spec[index])  # look for abosrption under bgd
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
        if index_list[idx][-1] > index_list[idx + 1][0]:
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
                fit, cov, model = fit_poly1d_legendre(lambdas[bgd_index], spec[bgd_index],
                                                      order=bgd_npar - 1, w=1. / spec_err[bgd_index])
            except:
                fit, cov, model = fit_poly1d_legendre(lambdas[index], spec[index],
                                                      order=bgd_npar - 1, w=1. / spec_err[index])
        else:
            fit, cov, model = fit_poly1d_legendre(lambdas[index], spec[index],
                                                  order=bgd_npar - 1, w=1. / spec_err[index])
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
            if np.isclose(b, 0, rtol=1e-2*float(np.mean(spec[bgd_index]))):
                b = baseline_prior * np.std(spec[bgd_index])
                if np.isclose(b, 0, rtol=1e-2 * float(np.mean(spec[bgd_index]))):
                    b = np.inf
            bounds[0][n] = guess[n] - b
            bounds[1][n] = guess[n] + b
        for j in range(len(new_lines_list[k])):
            idx = new_peak_index_list[k][j]
            x_norm = rescale_x_for_legendre(lambdas[idx])
            guess[bgd_npar + 3 * j] = np.sign(guess[bgd_npar + 3 * j]) * abs(
                spec[idx] - np.polynomial.legendre.legval(x_norm, guess[:bgd_npar]))
            if np.sign(guess[bgd_npar + 3 * j]) < 0:  # absorption
                bounds[0][bgd_npar + 3 * j] = 2 * guess[bgd_npar + 3 * j]
            else:  # emission
                bounds[1][bgd_npar + 3 * j] = 2 * guess[bgd_npar + 3 * j]
        # fit local extrema with a multigaussian + CALIB_BGD_ORDER polynom
        # account for the spectrum uncertainties if provided
        sigma = None
        if spec_err is not None:
            sigma = spec_err[index]
        # my_logger.warning(f'\n{guess} {np.mean(spec[bgd_index])} {np.std(spec[bgd_index])}')
        popt, pcov = fit_multigauss_and_bgd(lambdas[index], spec[index], guess=guess, bounds=bounds, sigma=sigma,
                                            fix_centroids=True)
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
            line.fit_lambdas = lambdas[index]
            line.fit_popt = popt
            line.fit_gauss = gauss(lambdas[index], *popt[bgd_npar + 3 * j:bgd_npar + 3 * j + 3])
            x_norm = rescale_x_for_legendre(lambdas[index])
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


# noinspection PyArgumentList
def calibrate_spectrum_with_lines(spectrum):
    """Convert pixels into wavelengths given the position of the order 0,
    the data for the spectrum, the properties of the disperser. Fit the absorption
    (and eventually the emission) lines to perform a second calibration of the
    distance between the CCD and the disperser. The number of fitting steps is
    limited to 30.

    Prerequisites: a first calibration from pixels to wavelengths must have been performed before

    Parameters
    ----------
    spectrum: Spectrum
        Spectrum object to calibrate

    Returns
    -------
    lambdas: array_like
        The new wavelength array in nm.

    Examples
    --------
    >>> spectrum = Spectrum('tests/data/reduc_20170605_028_spectrum.fits')
    >>> parameters.LAMBDA_MIN = 550
    >>> parameters.LAMBDA_MAX = 800
    >>> lambdas = calibrate_spectrum_with_lines(spectrum)
    >>> spectrum.plot_spectrum()
    """
    my_logger = set_logger(__name__)

    # Convert wavelength array into original pixels
    x0 = spectrum.x0
    if x0 is None:
        x0 = spectrum.target_pixcoords
        spectrum.x0 = x0
    D = parameters.DISTANCE2CCD
    if spectrum.header['D2CCD'] != '':
        D = spectrum.header['D2CCD']
    spectrum.disperser.D = D
    delta_pixels = spectrum.disperser.grating_lambda_to_pixel(spectrum.lambdas, x0=x0, order=spectrum.order)

    # Detect emission/absorption lines and calibrate pixel/lambda
    D = parameters.DISTANCE2CCD  # - parameters.DISTANCE2CCD_ERR
    D_err = parameters.DISTANCE2CCD_ERR
    fwhm_func = interp1d(spectrum.chromatic_psf.table['lambdas'],
                         spectrum.chromatic_psf.table['fwhm'],
                         fill_value=(parameters.CALIB_PEAK_WIDTH, parameters.CALIB_PEAK_WIDTH), bounds_error=False)

    def shift_minimizer(params):
        spectrum.disperser.D, shift = params
        lambdas_test = spectrum.disperser.grating_pixel_to_lambda(delta_pixels - shift,
                                                                  x0=[x0[0] + shift, x0[1]], order=spectrum.order)

        if parameters.OBS_NAME != 'PICDUMIDI':
            chisq = detect_lines(spectrum.lines, lambdas_test, spectrum.data, spec_err=spectrum.err,
                             fwhm_func=fwhm_func, ax=None)
        else:
            # for Pic du Midi restriction of absorption line in range 430-800 nm (Hgamma to O2) because  of wiggles
            chisq = detect_lines(spectrum.lines, lambdas_test, spectrum.data, spec_err=spectrum.err,
                                 fwhm_func=fwhm_func, ax=None, xlim=(parameters.LAMBDA_MIN, 800.))
                                 #fwhm_func=fwhm_func, ax=None,xlim = (430.,800.))




        chisq += (shift * shift) / (parameters.PIXSHIFT_PRIOR / 2) ** 2
        if parameters.DEBUG and parameters.DISPLAY:
            spectrum.lambdas = lambdas_test
            spectrum.plot_spectrum(live_fit=True, label=f'Order {spectrum.order:d} spectrum'
                                                        f'\nD={D:.2f}mm, shift={shift:.2f}pix')
        return chisq

    #--------------------------------------------------
    # grid exploration of the parameters
    # necessary because of the  line detection algo
    #---------------------------------------------------------

    if parameters.OBS_NAME != 'PICDUMIDI':
        D_step = D_err / 2
        pixel_shift_step = 0.5
        pixel_shift_prior = parameters.PIXSHIFT_PRIOR
        Ds = np.arange(D - 5 * D_err, D + 6 * D_err, D_step)
        pixel_shifts = np.arange(-pixel_shift_prior, pixel_shift_prior + pixel_shift_step, pixel_shift_step)
    else:
        D_step = D_err
        pixel_shift_step = 0.5
        pixel_shift_prior = parameters.PIXSHIFT_PRIOR
        Ds = np.arange(D - 7 * D_err, D + 8* D_err, D_step)
        pixel_shifts = np.arange(-2*pixel_shift_prior, 2*pixel_shift_prior + pixel_shift_step, pixel_shift_step)

    #if parameters.DEBUG:
    #    spectrum.my_logger.info('\n\tDs={}'.format(Ds))
    #    spectrum.my_logger.info('\n\tpixel_shifts={}'.format(pixel_shifts))

    # pixel_shifts = np.array([0])
    chisq_grid = np.zeros((len(Ds), len(pixel_shifts)))
    for i, D in enumerate(Ds):
        for j, pixel_shift in enumerate(pixel_shifts):
            #spectrum.my_logger.info('\n\tD={:2.2f} , pixel_shift={:2.2f}'.format(D,pixel_shift))
            chisq_grid[i, j] = shift_minimizer([D, pixel_shift])

    # values for initialization
    imin, jmin = np.unravel_index(chisq_grid.argmin(), chisq_grid.shape)
    D = Ds[imin]
    pixel_shift = pixel_shifts[jmin]
    start = np.array([D, pixel_shift])


    if imin == 0 or imin == Ds.size or jmin == 0 or jmin == pixel_shifts.size:
        spectrum.my_logger.warning('\n\tMinimum chisq is on the edge of the exploration grid.')

    #if parameters.DEBUG and parameters.DISPLAY:
    if (parameters.VERBOSE or parameters.DEBUG) and parameters.DISPLAY:
        im = plt.imshow(np.log10(chisq_grid), origin='lower', aspect='auto',
                        extent=(
                            np.min(pixel_shifts) - pixel_shift_step / 2, np.max(pixel_shifts) + pixel_shift_step / 2,
                            np.min(Ds) - D_step / 2, np.max(Ds) + D_step / 2))
        plt.gca().scatter(pixel_shift, D, marker='o', s=100, edgecolors='red', facecolors='none',
                          label='Minimum', linewidth=2)
        c = plt.colorbar(im)
        c.set_label('Log10(chisq)')
        plt.xlabel('Pixel shift [pix]')
        plt.ylabel('D [mm]')
        plt.legend()
        plt.show()

    # now minimize around the global minimum found previously
    # res = opt.minimize(shift_minimizer, start, args=(), method='L-BFGS-B',
    #                    options={'maxiter': 200, 'ftol': 1e-3},
    #                    bounds=((D - 5 * parameters.DISTANCE2CCD_ERR, D + 5 * parameters.DISTANCE2CCD_ERR), (-2, 2)))

    error = [parameters.DISTANCE2CCD_ERR, pixel_shift_step]


    fix = [False, False]

    if parameters.OBS_NAME != 'PICDUMIDI':
        m = Minuit.from_array_func(fcn=shift_minimizer, start=start, error=error, errordef=1, fix=fix, print_level=0,
                               limit=((D - 5 * parameters.DISTANCE2CCD_ERR, D + 5 * parameters.DISTANCE2CCD_ERR),
                                      (-2, 2)))
    else:
        #m = Minuit.from_array_func(fcn=shift_minimizer, start=start, error=error, errordef=1, fix=fix, print_level=0,
        #                           limit=((D - 7 * parameters.DISTANCE2CCD_ERR, D + 7 * parameters.DISTANCE2CCD_ERR),
        #                                  (-5, 5)))
        m = Minuit.from_array_func(fcn=shift_minimizer, start=start, error=error, errordef=1, fix=fix, print_level=0,
                                   limit=((D - 5 * parameters.DISTANCE2CCD_ERR, D + 5 * parameters.DISTANCE2CCD_ERR),
                                          (pixel_shift-3,pixel_shift+3)))
    m.migrad()
    #if parameters.DEBUG:
    #    print(m.pri)
    #if not res.success:
    #    spectrum.my_logger.warning('\n\tMinimizer failed.')
    #    print(res)

    D, pixel_shift = m.np_values()
    spectrum.disperser.D = D
    x0 = [x0[0] + pixel_shift, x0[1]]
    spectrum.x0 = x0
    # check success, xO ou D sur les bords du prior
    lambdas = spectrum.disperser.grating_pixel_to_lambda(delta_pixels - pixel_shift, x0=x0, order=spectrum.order)
    spectrum.lambdas = lambdas
    spectrum.pixels = delta_pixels - pixel_shift
    spectrum.my_logger.info(
        '\n\tOrder0 total shift: {:.2f}pix'
        '\n\tD = {:.2f} mm (default: DISTANCE2CCD = {:.2f} +/- {:.2f} mm, {:.1f} sigma shift)'.format(
            pixel_shift, D, parameters.DISTANCE2CCD, parameters.DISTANCE2CCD_ERR,
            (D - parameters.DISTANCE2CCD) / parameters.DISTANCE2CCD_ERR))
    spectrum.header['PIXSHIFT'] = pixel_shift
    spectrum.header['D2CCD'] = D
    return lambdas


def extract_spectrum_from_image(image, spectrum, w=10, ws=(20, 30), right_edge=parameters.CCD_IMSIZE - 200):
    """Extract the 1D spectrum from the image.

    Method : remove a uniform background estimated from the rectangular lateral bands

    The spectrum amplitude is the sum of the pixels in the 2*w rectangular window
    centered on the order 0 y position.
    The up and down backgrounds are estimated as the median in rectangular regions
    above and below the spectrum, in the ws-defined rectangular regions; stars are filtered
    as nan values using an hessian analysis of the image to remove structures.
    The subtracted background is the mean of the two up and down backgrounds.
    Stars are filtered.

    Prerequisites: the target position must have been found before, and the
        image turned to have an horizontal dispersion line

    Parameters
    ----------
    image: Image
        Image object from which to extract the spectrum
    spectrum: Spectrum
        Spectrum object to store new wavelengths, data and error arrays
    w: int
        Half width of central region where the spectrum is extracted and summed (default: 10)
    ws: list
        up/down region extension where the sky background is estimated with format [int, int] (default: [20,30])
    right_edge: int
        Right-hand pixel position above which no pixel should be used (default: 1800)
    """

    my_logger = set_logger(__name__)
    if ws is None:
        ws = [20, 30]
    my_logger.info(
        f'\n\tExtracting spectrum from image: spectrum with width 2*{w:d} pixels '
        f'and background from {ws[0]:d} to {ws[1]:d} pixels')

    # Make a data copy
    data = np.copy(image.data_rotated)[:, 0:right_edge]
    err = np.copy(image.stat_errors_rotated)[:, 0:right_edge]




    # Lateral bands to remove sky background
    Ny, Nx = data.shape
    x0 = int(image.target_pixcoords_rotated[0])
    y0 = int(image.target_pixcoords_rotated[1])
    ymax = min(Ny, y0 + ws[1])
    ymin = max(0, y0 - ws[1])

    my_logger.info(
        f'\n\tExtracting spectrum from image: extract_spectrum_from_image : Ny,Nx= {Ny}, {Nx}, x0,y0= {x0}, {y0} '
        f'and ymin, ymax= {ymin}, {ymax}')


    if parameters.DEBUG:
        plt.figure(figsize=(6, 6))
        plt.title("DEBUG 1 : extract_spectrum_from_image")
        plt.imshow(data, cmap="jet", origin="lower",vmin=0,vmax=data.flatten().max()*0.1)
        plt.scatter(x0, y0, marker='o', s=100, edgecolors='y', facecolors='none',
               label='Target', linewidth=2)
        plt.grid(True,color="white")
        plt.show()



    # Roughly estimates the wavelengths and set start 50 nm before parameters.LAMBDA_MIN
    # and end 50 nm after parameters.LAMBDA_MAX

    if parameters.OBS_NAME != 'PICDUMIDI':
        lambdas = image.disperser.grating_pixel_to_lambda(np.arange(Nx) - image.target_pixcoords_rotated[0],
                                                      x0=image.target_pixcoords)
    else:
        #lambdas = image.disperser.grating_pixel_to_lambda(np.arange(Nx) - image.target_pixcoords_rotated[0],
        #                                                  x0=image.target_pixcoords_rotated)
        lambdas = image.disperser.grating_pixel_to_lambda(np.arange(Nx) - image.target_pixcoords_rotated[0],
                                                          x0=image.target_pixcoords)

    if parameters.DEBUG:
        plt.figure(figsize=(6, 6))
        plt.title("DEBUG 2 : extract_spectrum_from_image show dispersion relation ")
        plt.plot(np.arange(Nx), lambdas,"b-")
        plt.grid()
        plt.xlabel("pixels")
        plt.ylabel("wavelength (nm)")
        plt.show()


    pixel_start = int(np.argmin(np.abs(lambdas - (parameters.LAMBDA_MIN - 0))))
    pixel_end = min(right_edge, int(np.argmin(np.abs(lambdas - (parameters.LAMBDA_MAX + 0)))))

    my_logger.info(
        f'\n\tExtracting spectrum from image: extract_spectrum_from_image : pixel_start = {pixel_start} pixel_end = {pixel_end} ')



    if (pixel_end - pixel_start) % 2 == 0:  # spectrogram table must have odd size in x for the fourier simulation
        pixel_end -= 1

    my_logger.info(
        f'\n\tExtracting spectrum from image: extract_spectrum_from_image : pixel_start = {pixel_start} pixel_end = {pixel_end} ')


    # Create spectrogram
    data = data[ymin:ymax, pixel_start:pixel_end]
    err = err[ymin:ymax, pixel_start:pixel_end]
    Ny, Nx = data.shape
    my_logger.info(
        f'\n\tExtract spectrogram: crop rotated image [{pixel_start}:{pixel_end},{ymin}:{ymax}] (size ({Nx}, {Ny}))')

    # Extract the abckground on the rotated image
    bgd_model_func = extract_background(data, deg=1, ws=ws, pixel_step=1, sigma=3)

    # Fit the transverse profile
    my_logger.info(f'\n\tStart PSF1D transverse fit...')
    s = fit_transverse_PSF1D_profile(data, err, w, ws, pixel_step=1, sigma=5, deg=2,bgd_model_func=bgd_model_func,
                                                     saturation=image.saturation, live_fit=parameters.DEBUG)

    # Fill spectrum object
    spectrum.pixels = np.arange(pixel_start, pixel_end, 1).astype(int)
    spectrum.data = np.copy(s.table['flux_sum'])
    spectrum.err = np.copy(s.table['flux_err'])
    my_logger.debug(f'\n\tTransverse fit table:\n{s.table}')
    if parameters.DEBUG:
        s.plot_summary()

    # Fit the data:
    my_logger.info(f'\n\tStart ChromaticPSF1D polynomial fit...')
    s = fit_chromatic_PSF1D(data, s, bgd_model_func=bgd_model_func, data_errors=err)
    spectrum.chromatic_psf = s
    spectrum.data = np.copy(s.table['flux_integral'])
    s.table['Dx_rot'] = spectrum.pixels.astype(float) - image.target_pixcoords_rotated[0]
    s.table['Dx'] = np.copy(s.table['Dx_rot'])
    s.table['Dy'] = s.table['x_mean'] - (image.target_pixcoords_rotated[1] - ymin)
    s.table['Dy_fwhm_inf'] = s.table['Dy'] - 0.5 * s.table['fwhm']
    s.table['Dy_fwhm_sup'] = s.table['Dy'] + 0.5 * s.table['fwhm']
    s.table['x_mean'] = s.table['x_mean'] - (image.target_pixcoords_rotated[1] - ymin)
    my_logger.debug(f"\n\tTransverse fit table before derotation:\n{s.table[['Dx_rot', 'Dx', 'x_mean', 'Dy']]}")

    # rotate and save the table
    s.rotate_table(-image.rotation_angle)
    my_logger.debug(f"\n\tTransverse fit table after derotation:\n{s.table[['Dx_rot', 'Dx', 'x_mean', 'Dy']]}")

    # Extract the spectrogram edges
    data = np.copy(image.data)[:, 0:right_edge]
    err = np.copy(image.stat_errors)[:, 0:right_edge]
    Ny, Nx = data.shape
    x0 = int(image.target_pixcoords[0])
    y0 = int(image.target_pixcoords[1])
    ymax = min(Ny, y0 + int(s.table['Dy_mean'].max()) + ws[1] + 1)  # +1 to  include edges
    ymin = max(0, y0 + int(s.table['Dy_mean'].min()) - ws[1])
    distance = s.get_distance_along_dispersion_axis()
    lambdas = image.disperser.grating_pixel_to_lambda(distance, x0=image.target_pixcoords)
    lambda_min_index = int(np.argmin(np.abs(lambdas - (parameters.LAMBDA_MIN - 0))))
    lambda_max_index = int(np.argmin(np.abs(lambdas - (parameters.LAMBDA_MAX + 0))))
    xmin = int(s.table['Dx'][lambda_min_index] + x0)
    xmax = min(right_edge, int(s.table['Dx'][lambda_max_index] + x0) + 1)  # +1 to  include edges
    if (xmax - xmin) % 2 == 0:  # spectrogram must have odd size in x for the fourier simulation
        xmax -= 1
        s.table.remove_row(-1)

    # Position of the order 0 in the spectrogram coordinates
    target_pixcoords_spectrogram = [image.target_pixcoords[0] - xmin, image.target_pixcoords[1] - ymin]

    # Create spectrogram
    data = data[ymin:ymax, xmin:xmax]
    err = err[ymin:ymax, xmin:xmax]
    Ny, Nx = data.shape
    # Extract the non rotated background
    bgd_model_func = extract_background(data, deg=1, ws=ws, sigma=3)
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny))
    bgd = bgd_model_func(xx, yy)

    # Crop the background lateral regions
    bgd_width = ws[1] - w
    yeven = 0
    if (Ny - 2 * bgd_width) % 2 == 0:  # spectrogram must have odd size in y for the fourier simulation
        yeven = 1
    ymax = ymax - bgd_width + yeven
    ymin += bgd_width
    bgd = bgd[bgd_width:-bgd_width + yeven, :]
    data = data[bgd_width:-bgd_width + yeven, :]
    err = err[bgd_width:-bgd_width + yeven, :]
    Ny, Nx = data.shape
    # First guess for lambdas
    first_guess_lambdas = image.disperser.grating_pixel_to_lambda(s.get_distance_along_dispersion_axis(),
                                                                  x0=image.target_pixcoords)
    s.table['lambdas'] = first_guess_lambdas
    spectrum.lambdas = np.array(first_guess_lambdas)
    my_logger.warning(f"\n\tTransverse fit table after derotation:\n{s.table[['lambdas', 'Dx_rot', 'Dx']]}")

    # Position of the order 0 in the spectrogram coordinates
    target_pixcoords_spectrogram[1] -= bgd_width
    my_logger.info(f'\n\tExtract spectrogram: crop image [{xmin}:{xmax},{ymin}:{ymax}] (size ({Nx}, {Ny}))'
                   f'\n\tNew target position in spectrogram frame: {target_pixcoords_spectrogram}')

    # Save results
    spectrum.spectrogram = data
    spectrum.spectrogram_err = err
    spectrum.spectrogram_bgd = bgd
    spectrum.spectrogram_x0 = target_pixcoords_spectrogram[0]
    spectrum.spectrogram_y0 = target_pixcoords_spectrogram[1]
    spectrum.spectrogram_xmin = xmin
    spectrum.spectrogram_xmax = xmax
    spectrum.spectrogram_ymin = ymin
    spectrum.spectrogram_ymax = ymax
    spectrum.spectrogram_deg = spectrum.chromatic_psf.deg
    spectrum.spectrogram_saturation = spectrum.chromatic_psf.saturation

    # Summary plot
    if parameters.DEBUG:
        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(12, 6))
        x = np.arange(Nx)
        xx = np.arange(s.table['Dx_rot'].size)
        plot_image_simple(ax[2], data=data, scale="log", title='', units=image.units, aspect='auto')
        ax[2].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_mean'], label='Dispersion axis')
        ax[2].scatter(xx, target_pixcoords_spectrogram[1] + s.table['Dy'],
                      c=s.table['lambdas'], edgecolors='None', cmap=from_lambda_to_colormap(s.table['lambdas']),
                      label='Fitted spectrum centers', marker='o', s=10)
        ax[2].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_fwhm_inf'], 'k-', label='Fitted FWHM')
        ax[2].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_fwhm_sup'], 'k-', label='')
        ax[2].set_ylim(0, Ny)
        ax[2].set_xlim(0, xx.size)
        ax[2].legend(loc='best')
        plot_spectrum_simple(ax[0], np.arange(spectrum.data.size), spectrum.data, data_err=spectrum.err,
                             units=image.units, label='Fitted spectrum', xlim=[0, spectrum.data.size])
        ax[0].plot(xx, s.table['flux_sum'], 'k-', label='Cross spectrum')
        ax[0].set_xlim(0, xx.size)
        ax[0].legend(loc='best')
        ax[1].plot(xx, (s.table['flux_sum'] - s.table['flux_integral']) / s.table['flux_sum'],
                   label='(model_integral-cross_sum)/cross_sum')
        ax[1].legend()
        ax[1].grid(True)
        ax[1].set_ylim(-1, 1)
        ax[1].set_ylabel('Relative difference')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        pos0 = ax[0].get_position()
        pos1 = ax[1].get_position()
        pos2 = ax[2].get_position()
        ax[0].set_position([pos2.x0, pos0.y0, pos2.width, pos0.height])
        ax[1].set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])
        if parameters.DISPLAY:
            plt.suptitle("spectrum::extract_spectrum_from_image")
            plt.show()
    return spectrum


if __name__ == "__main__":
    import doctest
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    doctest.testmod()

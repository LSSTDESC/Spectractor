from astropy.table import Table
from scipy.signal import argrelextrema
import scipy.optimize as opt

from spectractor.extractor.images import *


class Line:
    """Class modeling the emission or absorption lines."""

    def __init__(self, wavelength, label, atmospheric=False, emission=False, label_pos=[0.007, 0.02],
                 width_bounds=[1, 6], use_for_calibration=False):
        """Class modeling the emission or absorption lines. lines attributes contains main spectral lines
        sorted in wavelength.

        Parameters
        ----------
        wavelength: float
            Wavelength of the spectral line in nm
        label: str

        atmospheric: bool
            Set True if the spectral line is atmospheric (default: False)
        emission: bool
            Set True if the spectral line has to be detected in emission. Can't be true if the line is atmospheric.
            (default: False)
        label_pos: [float, float]
            Position of the label in the plot with respect to the vertical lin (default: [0.007,0.02])
        width_bounds: [float, float]
            Minimum and maximum width (in nm) of the line for fitting procedures (default: [1,7])
        use_for_calibration: bool
            Use this line for the dispersion relation calibration, bright line recommended (default: False)

        Examples
        --------
        >>> l = Line(550, label='test', atmospheric=True, emission=True)
        >>> print(l.wavelength)
        550
        >>> print(l.label)
        test
        >>> print(l.atmospheric)
        True
        >>> print(l.emission)
        False
        """
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.wavelength = wavelength  # in nm
        self.label = label
        self.label_pos = label_pos
        self.atmospheric = atmospheric
        self.emission = emission
        if self.atmospheric:
            self.emission = False
        self.width_bounds = width_bounds
        self.fitted = False
        self.use_for_calibration = use_for_calibration
        self.high_snr = False
        self.fit_lambdas = None
        self.fit_gauss = None
        self.fit_bgd = None
        self.fit_snr = None
        self.fit_fwhm = None
        self.fit_popt = None
        self.fit_chisq = None

    def gaussian_model(self, lambdas, A=1, sigma=2, use_fit=False):
        """Return a Gaussian model of the spectral line.

        Parameters
        ----------
        lambdas: float, array
            Wavelength array of float in nm
        A: float
            Amplitude of the Gaussian (default: +1)
        sigma: float
            Standard deviation of the Gaussian (default: 2)
        use_fit: bool, optional
            If True, it overrides the previous setting values and use the Gaussian fit made on data, if ti exists.

        Returns
        -------
        model: float, array
            The amplitude array of float of the Gaussian model of the line.

        Examples
        --------

        Give lambdas as a float:
        >>> l = Line(656.3, atmospheric=False, label='$H\\alpha$')
        >>> sigma = 2.
        >>> model = l.gaussian_model(656.3, A=1, sigma=sigma, use_fit=False)
        >>> print(model)
        1.0
        >>> model = l.gaussian_model(656.3+sigma*np.sqrt(2*np.log(2)), A=1, sigma=sigma, use_fit=False)
        >>> print(model)
        0.5

        Use a fit (for the example we create a mock fit result):
        >>> l.fit_lambdas = np.arange(600,700,2)
        >>> l.fit_gauss = gauss(l.fit_lambdas, 1e-10, 650, 2.3)
        >>> l.fit_fwhm = 2.3*2*np.sqrt(2*np.log(2))
        >>> lambdas = np.arange(500,1000,1)
        >>> model = l.gaussian_model(lambdas, A=1, sigma=sigma, use_fit=True)
        >>> print(model[:5])
        [ 0.  0.  0.  0.  0.]

        """
        if use_fit and self.fit_gauss is not None:
            interp = interpolate.interp1d(self.fit_lambdas, self.fit_gauss, bounds_error=False, fill_value=0.)
            return interp(lambdas)
        else:
            return gauss(lambdas, A=A, x0=self.wavelength, sigma=sigma)


class Lines:
    """Class gathering all the lines and associated methods."""

    def __init__(self, redshift=0, atmospheric_lines=True, hydrogen_only=False, emission_spectrum=False):
        """ Main emission/absorption lines in nm. Sorted lines are sorted in self.lines.
        See http://www.pa.uky.edu/~peter/atomic/ or https://physics.nist.gov/PhysRefData/ASD/lines_form.html

        Parameters
        ----------
        redshift: float
            Red shift the spectral lines. Must be positive or null (default: 0)
        atmospheric_lines: bool
            Set True if the atmospheric spectral lines must be included (default: True)
        hydrogen_only: bool
            Set True to gather only the hydrogen spectral lines, atmospheric lines still included (default: False)
        emission_spectrum: bool
            Set True if the spectral line has to be detected in emission (default: False)

        Examples
        --------
        The default first five lines:
        >>> lines = Lines(redshift=0, atmospheric_lines=False, hydrogen_only=False, emission_spectrum=False)
        >>> print([lines.lines[i].wavelength for i in range(5)])
        [353.1, 388.8, 410.2, 434.0, 447.1]

        The four hydrogen lines only:
        >>> lines = Lines(redshift=0, atmospheric_lines=False, hydrogen_only=True, emission_spectrum=True)
        >>> print([lines.lines[i].wavelength for i in range(4)])
        [410.2, 434.0, 486.3, 656.3]
        >>> print(lines.emission_spectrum)
        True

        Redshift the hydrogen lines, the atmospheric lines stay unchanged:
        >>> lines = Lines(redshift=1, atmospheric_lines=True, hydrogen_only=True, emission_spectrum=True)
        >>> print([lines.lines[i].wavelength for i in range(5)])
        [382.044, 393.366, 396.847, 430.79, 438.355]

        Redshift all the spectral lines, except the atmospheric lines:
        >>> lines = Lines(redshift=1, atmospheric_lines=True, hydrogen_only=False, emission_spectrum=True)
        >>> print([lines.lines[i].wavelength for i in range(5)])
        [382.044, 393.366, 396.847, 430.79, 438.355]
        """
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        if redshift < 0:
            self.my_logger.warning(f'Redshift must be positive or null. Got {redshift}')
            sys.exit()
        HALPHA = Line(656.3, atmospheric=False, label='$H\\alpha$', label_pos=[-0.016, 0.02], use_for_calibration=True)
        HBETA = Line(486.3, atmospheric=False, label='$H\\beta$', label_pos=[0.007, 0.02], use_for_calibration=True)
        HGAMMA = Line(434.0, atmospheric=False, label='$H\\gamma$', label_pos=[0.007, 0.02], use_for_calibration=True)
        HDELTA = Line(410.2, atmospheric=False, label='$H\\delta$', label_pos=[0.007, 0.02])
        OIII = Line(500.7, atmospheric=False, label=r'$O_{III}$', label_pos=[0.007, 0.02])
        CII1 = Line(723.5, atmospheric=False, label=r'$C_{II}$', label_pos=[0.005, 0.88])
        CII2 = Line(711.0, atmospheric=False, label=r'$C_{II}$', label_pos=[0.005, 0.02])
        CIV = Line(706.0, atmospheric=False, label=r'$C_{IV}$', label_pos=[-0.016, 0.88])
        CII3 = Line(679.0, atmospheric=False, label=r'$C_{II}$', label_pos=[0.005, 0.02])
        CIII1 = Line(673.0, atmospheric=False, label=r'$C_{III}$', label_pos=[-0.016, 0.88])
        CIII2 = Line(570.0, atmospheric=False, label=r'$C_{III}$', label_pos=[0.007, 0.02])
        CIII3 = Line(970.5, atmospheric=False, label=r'$C_{III}$', label_pos=[0.007, 0.02])
        FEII1 = Line(463.8, atmospheric=False, label=r'$Fe_{II}$', label_pos=[-0.016, 0.02])
        FEII2 = Line(515.8, atmospheric=False, label=r'$Fe_{II}$', label_pos=[0.007, 0.02])
        FEII3 = Line(527.3, atmospheric=False, label=r'$Fe_{II}$', label_pos=[0.007, 0.02])
        FEII4 = Line(534.9, atmospheric=False, label=r'$Fe_{II}$', label_pos=[0.007, 0.02])
        HEI1 = Line(388.8, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI2 = Line(447.1, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI3 = Line(587.5, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI4 = Line(750.0, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI5 = Line(776.0, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI6 = Line(781.6, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI7 = Line(848.2, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI8 = Line(861.7, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI9 = Line(906.5, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI10 = Line(923.5, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI11 = Line(951.9, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI12 = Line(1023.5, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        HEI13 = Line(353.1, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
        OI = Line(630.0, atmospheric=False, label=r'$O_{II}$', label_pos=[0.007, 0.02])
        OII = Line(732.5, atmospheric=False, label=r'$O_{II}$', label_pos=[0.007, 0.02])
        HEII1 = Line(468.6, atmospheric=False, label=r'$He_{II}$', label_pos=[0.007, 0.02])
        HEII2 = Line(611.8, atmospheric=False, label=r'$He_{II}$', label_pos=[0.007, 0.02])
        HEII3 = Line(617.1, atmospheric=False, label=r'$He_{II}$', label_pos=[0.007, 0.02])
        HEII4 = Line(856.7, atmospheric=False, label=r'$He_{II}$', label_pos=[0.007, 0.02])
        HI = Line(833.9, atmospheric=False, label=r'$He_{II}$', label_pos=[0.007, 0.02])
        FE1 = Line(382.044, atmospheric=True, label=r'$Fe$',
                   label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        FE2 = Line(430.790, atmospheric=True, label=r'$Fe$',
                   label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        FE3 = Line(438.355, atmospheric=True, label=r'$Fe$',
                   label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        CAII1 = Line(393.366, atmospheric=True, label=r'$Ca_{II}$',
                     label_pos=[0.007, 0.02],
                     use_for_calibration=True)  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        CAII2 = Line(396.847, atmospheric=True, label=r'$Ca_{II}$',
                     label_pos=[0.007, 0.02],
                     use_for_calibration=True)  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        O2 = Line(762.1, atmospheric=True, label=r'$O_2$',
                  label_pos=[0.007, 0.02],
                  use_for_calibration=True)  # http://onlinelibrary.wiley.com/doi/10.1029/98JD02799/pdf
        # O2_1 = Line( 760.6,atmospheric=True,label='',label_pos=[0.007,0.02]) # libradtran paper fig.3
        # O2_2 = Line( 763.2,atmospheric=True,label='$O_2$',label_pos=[0.007,0.02])  # libradtran paper fig.3
        O2B = Line(686.719, atmospheric=True, label=r'$O_2(B)$',
                   label_pos=[0.007, 0.02], use_for_calibration=True)  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        O2Y = Line(898.765, atmospheric=True, label=r'$O_2(Y)$',
                   label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        O2Z = Line(822.696, atmospheric=True, label=r'$O_2(Z)$',
                   label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        # H2O = Line( 960,atmospheric=True,label='$H_2 O$',label_pos=[0.007,0.02],width_bounds=(1,50))  #
        H2O_1 = Line(935, atmospheric=True, label=r'$H_2 O$', label_pos=[0.007, 0.02],
                     width_bounds=[5, 30])  # libradtran paper fig.3, broad line
        H2O_2 = Line(960, atmospheric=True, label=r'$H_2 O$', label_pos=[0.007, 0.02],
                     width_bounds=[5, 30])  # libradtran paper fig.3, broad line

        self.lines = [HALPHA, HBETA, HGAMMA, HDELTA, O2, O2B, O2Y, O2Z, H2O_1, H2O_2, OIII, CII1, CII2, CIV, CII3,
                      CIII1, CIII2, CIII3, HEI1, HEI2, HEI3, HEI4, HEI5, HEI6, HEI7, HEI8, HEI9, HEI10, HEI11, HEI12,
                      HEI13, OI, OII, HEII1, HEII2, HEII3, HEII4, CAII1, CAII2, FE1, FE2, FE3, HI, FEII1, FEII2, FEII3,
                      FEII4]
        self.redshift = redshift
        self.atmospheric_lines = atmospheric_lines
        self.hydrogen_only = hydrogen_only
        self.emission_spectrum = emission_spectrum
        self.lines = self.sort_lines()

    def sort_lines(self):
        """Sort the lines in increasing order of wavelength, and add the redshift effect.

        Returns
        -------
        sorted_lines: list
            List of the sorted lines

        Examples
        --------
        >>> lines = Lines()
        >>> sorted_lines = lines.sort_lines()

        """
        sorted_lines = []
        for line in self.lines:
            if self.hydrogen_only:
                if not self.atmospheric_lines:
                    if line.atmospheric:
                        continue
                    if '$H\\' not in line.label:
                        continue
                else:
                    if not line.atmospheric and '$H\\' not in line.label:
                        continue
            else:
                if not self.atmospheric_lines and line.atmospheric:
                    continue
            sorted_lines.append(line)
        if self.redshift > 0:
            for line in sorted_lines:
                if not line.atmospheric:
                    line.wavelength *= (1 + self.redshift)
        sorted_lines = sorted(sorted_lines, key=lambda x: x.wavelength)
        return sorted_lines

    def plot_atomic_lines(self, ax, color_atomic='g', color_atmospheric='b', fontsize=12):
        """Over plot the atomic lines as vertical lines.

        Parameters
        ----------
        ax: Axes
            An Axes instance on which plot the spectral lines
        color_atomic: str
            Color of the atomic lines (default: 'g')
        color_atmospheric: str
            Color of the atmospheric lines (default: 'b')
        fontsize: int
            Font size of the spectral line labels (default: 12)

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> f, ax = plt.subplots(1,1)
        >>> ax.set_xlim(300,1000)
        (300, 1000)
        >>> lines = Lines()
        >>> ax = lines.plot_atomic_lines(ax)
        >>> assert ax is not None
        """
        xlim = ax.get_xlim()
        for l in self.lines:
            if l.fitted and not l.high_snr:
                continue
            color = color_atomic
            if l.atmospheric:
                color = color_atmospheric
            ax.axvline(l.wavelength, lw=2, color=color)
            xpos = (l.wavelength - xlim[0]) / (xlim[1] - xlim[0]) + l.label_pos[0]
            if 0 < xpos < 1:
                ax.annotate(l.label, xy=(xpos, l.label_pos[1]), rotation=90, ha='left', va='bottom',
                            xycoords='axes fraction', color=color, fontsize=fontsize)
        return ax

    def detect_lines(self, lambdas, spec, spec_err=None, snr_minlevel=3, ax=None,
                     xlim=(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX)):
        """Detect and fit the lines in a spectrum. The method is to look at maxima or minima
        around emission or absorption tabulated lines, and to select surrounding pixels
        to fit a (positive or negative) gaussian and a polynomial background. If several regions
        overlap, a multi-gaussian fit is performed above a common polynomial background.
        The mean global shift (in nm) between the detected and tabulated lines is returned, considering
        only the lines with a signal-to-noise ratio above a threshold.
        The order of the polynomial background is set in parameters.py with BGD_ORDER.

        Parameters
        ----------
        lambdas: float array
            The wavelength array (in nm)
        spec: float array
            The spectrum amplitude array
        spec_err: float array, optional
            The spectrum amplitude uncertainty array (default: None)
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
        >>> HALPHA = Line(656.3, atmospheric=False, label=r'$H\alpha$')
        >>> HBETA = Line(486.3, atmospheric=False, label=r'$H\beta$')
        >>> O2 = Line(762.1, atmospheric=True, label=r'$O_2$')
        >>> spectrum += HALPHA.gaussian_model(lambdas, A=5000, sigma=3)
        >>> spectrum += HBETA.gaussian_model(lambdas, A=3000, sigma=2)
        >>> spectrum += O2.gaussian_model(lambdas, A=-3000, sigma=3)
        >>> spectrum_err = np.sqrt(spectrum)
        >>> spec = Spectrum()
        >>> spec.lambdas = lambdas
        >>> spec.data = spectrum
        >>> spec.err = spectrum_err

        Detect the lines
        >>> lines = Lines(hydrogen_only=True, atmospheric_lines=True, redshift=0, emission_spectrum=True)
        >>> global_chisq = lines.detect_lines(lambdas, spectrum, spectrum_err)
        >>> print('{:.1f}'.format(global_chisq))
        0.2

        Plot the result
        >>> spec.lines = lines
        >>> spec.plot_spectrum()
        """

        # main settings
        bgd_npar = parameters.BGD_NPARAMS
        peak_look = 7  # half range to look for local maximum in pixels
        bgd_width = 10  # size of the peak sides to use to fit spectrum base line
        if self.hydrogen_only:
            peak_look = 7
            bgd_width = 15
        baseline_prior = 1  # *sigma gaussian prior on base line fit

        # initialisation
        lambda_shifts = []
        snrs = []
        index_list = []
        peak_index_list = []
        guess_list = []
        bounds_list = []
        lines_list = []
        for line in self.lines:
            # reset line fit attributes
            line.fitted = False
            line.fit_popt = None
            line.high_snr = False
            # wavelength of the line: find the nearest pixel index
            line_wavelength = line.wavelength
            if line_wavelength < xlim[0] or line_wavelength > xlim[1]:
                continue
            l_index, l_lambdas = find_nearest(lambdas, line_wavelength)
            # reject if pixel index is too close to image bounds
            if l_index < peak_look or l_index > len(lambdas) - peak_look:
                continue
            # search for local extrema to detect emission or absorption line
            # around pixel index +/- peak_look
            line_strategy = np.greater  # look for emission line
            bgd_strategy = np.less
            if not self.emission_spectrum or line.atmospheric:
                line_strategy = np.less  # look for absorption line
                bgd_strategy = np.greater
            index = np.arange(l_index - peak_look, l_index + peak_look, 1).astype(int)
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
            # around +/- 3*peak_look
            index_inf = peak_index - 1  # extrema on the left
            while index_inf > max(0, peak_index - 3 * peak_look):
                test_index = np.arange(index_inf, peak_index, 1).astype(int)
                minm = argrelextrema(spec[test_index], bgd_strategy)
                if len(minm[0]) > 0:
                    index_inf = index_inf + minm[0][0]
                    break
                else:
                    index_inf -= 1
            index_sup = peak_index + 1  # extrema on the right
            while index_sup < min(len(spec) - 1, peak_index + 3 * peak_look):
                test_index = np.arange(peak_index, index_sup, 1).astype(int)
                minm = argrelextrema(spec[test_index], bgd_strategy)
                if len(minm[0]) > 0:
                    index_sup = peak_index + minm[0][0]
                    break
                else:
                    index_sup += 1
            # pixel range to consider around the peak, adding bgd_width pixels
            # to fit for background around the peak
            index = list(np.arange(max(0, index_inf - bgd_width),
                                   min(len(lambdas), index_sup + bgd_width), 1).astype(int))
            # skip if data is masked with NaN
            if np.any(np.isnan(spec[index])):
                continue
            # first guess and bounds to fit the line properties and
            # the background with BGD_ORDER order polynom
            # guess = [0] * bgd_npar + [0.5 * np.max(spec[index]), lambdas[peak_index],
            #                          0.5 * (line.width_bounds[0] + line.width_bounds[1])]
            guess = [0] * bgd_npar + [0.5 * np.max(spec[index]), line_wavelength,
                                      0.5 * (line.width_bounds[0] + line.width_bounds[1])]
            if line_strategy == np.less:
                guess[bgd_npar] = -0.5 * np.max(spec[index])  # look for abosrption under bgd
            # bounds = [[-np.inf] * bgd_npar + [-abs(np.max(spec[index])), lambdas[index_inf], line.width_bounds[0]],
            #          [np.inf] * bgd_npar + [abs(np.max(spec[index])), lambdas[index_sup], line.width_bounds[1]]]
            bounds = [[-np.inf] * bgd_npar + [-abs(np.max(spec[index])), line_wavelength - peak_look / 2,
                                              line.width_bounds[0]],
                      [np.inf] * bgd_npar + [abs(np.max(spec[index])), line_wavelength + peak_look / 2,
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
            tmp_guess = [guess_list[i][bgd_npar + 1] for i in merge]
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
                if i == merge[0]:
                    new_guess_list[-1] += guess_list[i][:bgd_npar]
                    new_bounds_list[-1][0] += bounds_list[i][0][:bgd_npar]
                    new_bounds_list[-1][1] += bounds_list[i][1][:bgd_npar]
                # add the gauss parameters
                new_peak_index_list[-1].append(peak_index_list[i])
                new_index_list[-1] += index_list[i]
                new_guess_list[-1] += guess_list[i][bgd_npar:]
                new_bounds_list[-1][0] += bounds_list[i][0][bgd_npar:]
                new_bounds_list[-1][1] += bounds_list[i][1][bgd_npar:]
                new_lines_list[-1].append(lines_list[i])
            # set central peak bounds exactly between two close lines
            for k in range(len(merge) - 1):
                new_bounds_list[-1][0][bgd_npar + 3 * (k + 1) + 1] = 0.5 * (
                        new_guess_list[-1][bgd_npar + 3 * k + 1] + new_guess_list[-1][bgd_npar + 3 * (k + 1) + 1])
                new_bounds_list[-1][1][bgd_npar + 3 * k + 1] = 0.5 * (
                        new_guess_list[-1][bgd_npar + 3 * k + 1] + new_guess_list[-1][
                    bgd_npar + 3 * (k + 1) + 1]) + 1e-3
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
            for i in index:
                is_close_to_peak = False
                for j in peak_index:
                    if abs(i - j) < peak_look:
                        is_close_to_peak = True
                        break
                if not is_close_to_peak:
                    bgd_index.append(i)
            fit, cov, model = fit_poly1d(lambdas[bgd_index], spec[bgd_index],
                                         order=parameters.BGD_ORDER, w=1. / spec_err[bgd_index])
            for n in range(bgd_npar):
                # guess[n] = getattr(bgd, bgd.param_names[parameters.BGD_ORDER - n]).value
                guess[n] = fit[n]
                b = abs(baseline_prior * guess[n])
                bounds[0][n] = guess[n] - b
                bounds[1][n] = guess[n] + b
            for j in range(len(new_lines_list[k])):
                idx = new_peak_index_list[k][j]
                guess[bgd_npar + 3 * j] = np.sign(guess[bgd_npar + 3 * j]) * abs(
                    spec[idx] - np.polyval(guess[:bgd_npar], lambdas[idx]))
                if np.sign(guess[bgd_npar + 3 * j]) < 0:  # absorption
                    bounds[0][bgd_npar + 3 * j] = 2 * guess[bgd_npar + 3 * j]
                else:  # emission
                    bounds[1][bgd_npar + 3 * j] = 2 * guess[bgd_npar + 3 * j]
            # fit local extrema with a multigaussian + BGD_ORDER polynom
            # account for the spectrum uncertainties if provided
            sigma = None
            if spec_err is not None:
                sigma = spec_err[index]
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
                line.fit_lambdas = lambdas[index]
                line.fit_popt = popt
                line.fit_gauss = gauss(lambdas[index], *popt[bgd_npar + 3 * j:bgd_npar + 3 * j + 3])
                line.fit_bgd = np.polyval(popt[:bgd_npar], lambdas[index])
                line.fit_snr = snr
                line.fit_chisq = chisq
                line.fit_fwhm = FWHM
                if snr < snr_minlevel:
                    continue
                line.high_snr = True
                if line.use_for_calibration:
                    # wavelength shift between tabulate and observed lines
                    lambda_shifts.append(peak_pos - line.wavelength)
                    snrs.append(snr)
        if ax is not None:
            self.plot_detected_lines(ax, print_table=parameters.DEBUG)
        if len(lambda_shifts) > 0:
            global_chisq /= len(lambda_shifts)
            shift = np.average(np.abs(lambda_shifts) ** 2, weights=np.array(snrs) ** 2)
            # if guess values on tabulated lines have not moved: penalize the chisq
            global_chisq += shift
            self.my_logger.debug(f'\n\tNumber of calibration lines detected {len(lambda_shifts):d}'
                                 f'\n\tTotal chisq: {global_chisq:.3f} with shift {shift:.3f}pix')
        else:
            global_chisq = 2 * len(parameters.LAMBDAS)
            self.my_logger.debug(
                f'\n\tNumber of calibration lines detected {len(lambda_shifts):d}\n\tTotal chisq: {global_chisq:.3f}')
        return global_chisq

    def plot_detected_lines(self, ax=None, print_table=False):
        lambdas = np.zeros(1)
        rows = []
        j = 0
        bgd_npar = parameters.BGD_NPARAMS
        for line in self.lines:
            if line.fitted is True:
                # look for lines in subset fit
                if lambdas.shape != line.fit_lambdas.shape or not np.allclose(lambdas, line.fit_lambdas, 1e-3):
                    j = 0
                    lambdas = np.copy(line.fit_lambdas)
                    if ax is not None:
                        ax.plot(lambdas, multigauss_and_bgd(lambdas, *line.fit_popt), lw=2, color='b')
                        ax.plot(lambdas, np.polyval(line.fit_popt[0:bgd_npar], lambdas), lw=2, color='b',
                                linestyle='--')
                popt = line.fit_popt
                peak_pos = popt[bgd_npar + 3 * j + 1]
                FWHM = np.abs(popt[bgd_npar + 3 * j + 2]) * 2.355
                signal_level = popt[bgd_npar + 3 * j]
                if line.high_snr:
                    rows.append((line.label, line.wavelength, peak_pos, peak_pos - line.wavelength,
                                 FWHM, signal_level, line.fit_snr, line.fit_chisq))
                j += 1
        if print_table and len(rows) > 0:
            t = Table(rows=rows, names=('Line', 'Tabulated', 'Detected', 'Shift', 'FWHM', 'Amplitude', 'SNR', 'Chisq'),
                      dtype=('a10', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
            for col in t.colnames[1:-2]:
                t[col].unit = 'nm'
            t[t.colnames[-1]].unit = 'reduced'
            print(t)


class Spectrum(object):

    def __init__(self, file_name="", image=None, atmospheric_lines=True, order=1, target=None):
        """ Spectrum class used to store information and methods
        relative to spectra nd their extraction.

        Parameters
        ----------
        file_name: str, optional
            Path to the spectrum file (default: "").
        image: Image, optional
            Image object from which to create the Spectrum object:
            copy the information from the Image header (default: None).
        atmospheric_lines: bool
            If True atmospheric absorption lines are plotted and eventually
            used for the spectrum calibration (default: True).
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
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.target = target
        self.data = None
        self.err = None
        self.x0 = None
        self.lambdas = None
        self.lambdas_binwidths = None
        self.lambdas_indices = None
        self.order = order
        self.filter = None
        self.filters = None
        self.units = 'ADU/s'
        self.gain = parameters.GAIN
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
            self.x0 = image.target_pixcoords
            self.target_pixcoords = image.target_pixcoords
            self.target_pixcoords_rotated = image.target_pixcoords_rotated
            self.units = image.units
            self.gain = image.gain
            self.my_logger.info('\n\tSpectrum info copied from image')
        self.atmospheric_lines = atmospheric_lines
        self.lines = None
        if self.target is not None:
            self.lines = Lines(self.target.redshift, atmospheric_lines=self.atmospheric_lines,
                               hydrogen_only=self.target.hydrogen_only, emission_spectrum=self.target.emission_spectrum)
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

    def plot_spectrum_simple(self, ax, xlim=None, color='r', label='', lambdas=None):
        """Simple function to plot a spectrum with error bars and labels.

        Parameters
        ----------
        ax: Axes
            Axes instance to make the plot
        xlim: list, optional
            List of minimum and maximum abscisses
        color: str
            String for the color of the spectrum (default: 'r')
        label: str
            String label for the plot legend
        lambdas: array, optional
            The wavelengths array if it has been given externally (default: None)

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> f, ax = plt.subplots(1,1)
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
        >>> s.plot_spectrum_simple(ax, xlim=[500,700], color='r', label='test')
        >>> if parameters.DISPLAY: plt.show()
        """
        xs = self.lambdas
        if lambdas is not None:
            xs = lambdas
        if label == '':
            label = f'Order {self.order:d} spectrum\nD={self.disperser.D:.2f}mm'
            if self.x0 is not None:
                label += f', x0={self.x0[0]:.2f}pix'
        if xs is None:
            xs = np.arange(self.data.shape[0])
        if self.err is not None:
            ax.errorbar(xs, self.data, yerr=self.err, fmt='{}o'.format(color), lw=1,
                        label=label, zorder=0, markersize=2)
        else:
            ax.plot(xs, self.data, '{}-'.format(color), lw=2, label=label)
        ax.grid(True)
        if xlim is None:
            xlim = [parameters.LAMBDA_MIN, parameters.LAMBDA_MAX]
        ax.set_xlim(xlim)
        ax.set_ylim(0., np.nanmax(self.data) * 1.2)
        ax.set_xlabel('$\lambda$ [nm]')
        ax.set_ylabel(f'Flux [{self.units}]')
        ax.set_title(self.target.label)

    def plot_spectrum(self, xlim=None, label='', live_fit=False):
        """Plot spectrum with emission and absorption lines.

        Parameters
        ----------
        xlim: list, optional
            List of minimum and maximum abscisses (default: None)
        label: str, optional
            Label for the plot legend (default: '')
        live_fit: bool, optional
            If True the spectrum is plotted in live during the fitting procedures
            (default: False).

        Examples
        --------
        >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
        >>> s.plot_spectrum(xlim=[500,700])
        >>> if parameters.DISPLAY: plt.show()
        """
        plt.figure(figsize=[12, 6])
        self.plot_spectrum_simple(plt.gca(), xlim=xlim, label=label)
        # if len(self.target.spectra)>0:
        #    for k in range(len(self.target.spectra)):
        #        s = self.target.spectra[k]/np.max(self.target.spectra[k])*np.max(self.data)
        #        plt.plot(self.target.wavelengths[k],s,lw=2,label='Tabulated spectra #%d' % k)
        if self.lambdas is not None:
            # self.lines.detect_lines(self.lambdas, self.data, spec_err=self.err, ax=plt.gca(),
            #                        print_table=parameters.VERBOSE)
            self.lines.plot_detected_lines(plt.gca(), print_table=parameters.VERBOSE)
        if self.lambdas is not None and self.lines is not None:
            self.lines.plot_atomic_lines(plt.gca(), fontsize=12)
        plt.legend(loc='best')
        if self.filters is not None:
            plt.gca().get_legend().set_title(self.filters)
        if parameters.DISPLAY:
            if live_fit:
                plt.draw()
                plt.pause(1e-8)
                plt.close()
            else:
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
            extract_info_from_CTIO_header(self, self.header)
            if self.header['TARGET'] != "":
                self.target = Target(self.header['TARGET'], verbose=parameters.VERBOSE)
            if self.header['UNIT2'] != "":
                self.units = self.header['UNIT2']
            if self.header['TARGETX'] != "" and self.header['TARGETY'] != "":
                self.x0 = [self.header['TARGETX'], self.header['TARGETY']]
            self.my_logger.info('\n\tLoading disperser %s...' % self.disperser_label)
            self.disperser = Hologram(self.header['FILTER2'], data_dir=parameters.HOLO_DIR, verbose=parameters.VERBOSE)
            self.my_logger.info('\n\tSpectrum loaded from %s' % input_file_name)
        else:
            self.my_logger.warning('\n\tSpectrum file %s not found' % input_file_name)


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
    if xlim is None:
        left_cut, right_cut = [0, spectrum.data.shape[0]]
    else:
        left_cut, right_cut = xlim
    spectrum.data = spectrum.data[left_cut:right_cut]
    pixels = np.arange(left_cut, right_cut, 1) - spectrum.target_pixcoords_rotated[0]
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
    D: float
        The new distance between the CCD and the disperser.

    """
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

    def shift_minimizer(params, spectrum, x0):
        spectrum.disperser.D = params[0]
        pixel_shift = params[1]
        lambdas_test = spectrum.disperser.grating_pixel_to_lambda(delta_pixels - pixel_shift,
                                                                  x0=[x0[0] + pixel_shift, x0[1]], order=spectrum.order)
        chisq = spectrum.lines.detect_lines(lambdas_test, spectrum.data, spec_err=spectrum.err, ax=None)
        chisq += pixel_shift * pixel_shift
        if parameters.DEBUG:
            spectrum.lambdas = lambdas_test
            spectrum.plot_spectrum(live_fit=True, label=f'Order {spectrum.order:d} spectrum'
                                                        f'\nD={D:.2f}mm, shift={pixel_shift:.2f}pix')
        return chisq

    # grid exploration of the parameters
    D_step = D_err / 2
    pixel_shift_step = 0.5
    Ds = np.arange(D - 5 * D_err, D + 6 * D_err, D_step)
    pixel_shifts = np.arange(-2, 2.5, pixel_shift_step)
    chisq_grid = np.zeros((len(Ds), len(pixel_shifts)))
    for i, D in enumerate(Ds):
        for j, pixel_shift in enumerate(pixel_shifts):
            chisq_grid[i, j] = shift_minimizer([D, pixel_shift], spectrum, x0)
    imin, jmin = np.unravel_index(chisq_grid.argmin(), chisq_grid.shape)
    D = Ds[imin]
    pixel_shift = pixel_shifts[jmin]
    start = np.array([D, pixel_shift])
    if imin == 0 or imin == Ds.size or jmin == 0 or jmin == pixel_shifts.size:
        spectrum.my_logger.warning('\n\tMinimum chisq is on the edge of the exploration grid.')
    if parameters.DEBUG and parameters.DISPLAY:
        im = plt.imshow(np.log10(chisq_grid), origin='lower',
                        extent=(
                        np.min(pixel_shifts) - pixel_shift_step / 2, np.max(pixel_shifts) + pixel_shift_step / 2,
                        np.min(Ds) - D_step / 2, np.max(Ds) + D_step / 2))
        plt.gca().scatter(pixel_shift, D, marker='o', s=100, edgecolors='k', facecolors='none',
                          label='Minimum', linewidth=2)
        c = plt.colorbar(im)
        c.set_label('Log10(chisq)')
        plt.xlabel('Pixel shift [pix]')
        plt.ylabel('D [mm]')
        plt.legend()
        plt.show()
    # now minimize around the global minimum found previously
    res = opt.minimize(shift_minimizer, start, args=(spectrum, x0), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-3},
                       bounds=((D - 5 * parameters.DISTANCE2CCD_ERR, D + 5 * parameters.DISTANCE2CCD_ERR), (-2, 2)))
    if parameters.DEBUG:
        print(res)
    if not res.success:
        spectrum.my_logger.warning('\n\tMinimizer failed.')
        print(res)
    D = res.x[0]
    spectrum.disperser.D = D
    pixel_shift = res.x[1]
    x0 = [x0[0] + res.x[1], x0[1]]
    spectrum.x0 = x0
    # check success, xO ou D sur les bords du prior
    lambdas = spectrum.disperser.grating_pixel_to_lambda(delta_pixels - pixel_shift, x0=x0, order=spectrum.order)
    spectrum.lambdas = lambdas
    spectrum.my_logger.info(
        '\n\tOrder0 total shift: {:.2f}pix'
        '\n\tD = {:.2f} mm (default: DISTANCE2CCD = {:.2f} +/- {:.2f} mm, {:.1f} sigma shift)'.format(
            pixel_shift, D, parameters.DISTANCE2CCD, parameters.DISTANCE2CCD_ERR,
            (D - parameters.DISTANCE2CCD) / parameters.DISTANCE2CCD_ERR))
    spectrum.header['PIXSHIFT'] = pixel_shift
    spectrum.header['D2CCD'] = D
    return lambdas


def extract_spectrum_from_image(image, spectrum, w=10, ws=(20, 30), right_edge=1800):
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
    ws: [int,int]
        up/down region extension where the sky background is estimated (default: [20,30])
    right_edge: int
        Right-hand pixel position above which no pixel should be used (default: 1800)
    """
    if ws is None:
        ws = [20, 30]
    image.my_logger.info(
        '\n\tExtracting spectrum from image: spectrum with width 2*{:d} pixels'
        ' and background from {:d} to {:d} pixels'.format(
            w, ws[0], ws[1]))
    # Make a data copy
    data = np.copy(image.data_rotated)[:, 0:right_edge]
    err = np.copy(image.stat_errors_rotated)[:, 0:right_edge]
    # Lateral bands to remove sky background
    Ny, Nx = data.shape
    y0 = int(image.target_pixcoords_rotated[1])
    ymax = min(Ny, y0 + ws[1])
    ymin = max(0, y0 - ws[1])
    spectrum2DUp = np.copy(data[y0 + ws[0]:ymax, :])
    spectrum2DUp = filter_stars_from_bgd(spectrum2DUp, margin_cut=1)
    err_spectrum2DUp = np.copy(err[y0 + ws[0]:ymax, :])
    err_spectrum2DUp = filter_stars_from_bgd(err_spectrum2DUp, margin_cut=1)
    xprofileUp = np.nanmedian(spectrum2DUp, axis=0)
    xprofileUp_err = np.sqrt(np.nanmean(err_spectrum2DUp ** 2, axis=0))
    spectrum2DDown = np.copy(data[ymin:y0 - ws[0], :])
    spectrum2DDown = filter_stars_from_bgd(spectrum2DDown, margin_cut=1)
    err_spectrum2DDown = np.copy(err[ymin:y0 - ws[0], :])
    err_spectrum2DDown = filter_stars_from_bgd(err_spectrum2DDown, margin_cut=1)
    xprofileDown = np.nanmedian(spectrum2DDown, axis=0)
    xprofileDown_err = np.sqrt(np.nanmean(err_spectrum2DDown ** 2, axis=0))
    # Sum rotated image profile along y axis
    # Subtract mean lateral profile
    xprofile_background = 0.5 * (xprofileUp + xprofileDown)
    xprofile_background_err = np.sqrt(0.5 * (xprofileUp_err ** 2 + xprofileDown_err ** 2))
    spectrum2D = np.copy(data[y0 - w:y0 + w, :])
    xprofile = np.sum(spectrum2D, axis=0) - 2 * w * xprofile_background
    # Sum uncertainties in quadrature
    err2D = np.copy(err[y0 - w:y0 + w, :])
    xprofile_err = np.sqrt(np.sum(err2D ** 2, axis=0) + (2 * w * xprofile_background_err) ** 2)
    # Fill spectrum object
    spectrum.data = xprofile
    spectrum.err = xprofile_err
    if parameters.DEBUG:
        spectrum.plot_spectrum()
    return spectrum


if __name__ == "__main__":
    import doctest

    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    doctest.testmod()

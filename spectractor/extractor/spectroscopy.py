from astropy.io import fits
from astropy.table import Table
from scipy.signal import argrelextrema

from spectractor.extractor.dispersers import *
from spectractor.extractor.filters import *
from spectractor.extractor.targets import *


class Line:
    """Class modeling the emission or absorption lines."""

    # noinspection PyDefaultArgument
    def __init__(self, wavelength, label, atmospheric=False, emission=False, label_pos=[0.007, 0.02],
                 width_bounds=[1, 7]):
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

        Examples
        --------
        >>> l = Line(550, label='test', atmospheric=True, emission=True)
        >>> print(l.wavelength)
        550
        >>> print(l.label)
        'test'
        >>> print(l.atmospheric)
        True
        >>> print(l.emission)
        False

        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.wavelength = wavelength  # in nm
        self.label = label
        self.label_pos = label_pos
        self.atmospheric = atmospheric
        self.emission = emission
        if self.atmospheric:
            self.emission = False
        self.width_bounds = width_bounds
        self.fitted = False
        self.high_snr = False


class Lines:
    """Class gathering all the lines and associated methods."""

    def __init__(self, redshift=0, atmospheric_lines=True, hydrogen_only=False, emission_spectrum=False):
        """ Main emission/absorption lines in nm
        see http://www.pa.uky.edu/~peter/atomic/
        see https://physics.nist.gov/PhysRefData/ASD/lines_form.html

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
        """
        self.my_logger = set_logger(self.__class__.__name__)
        if redshift < 0:
            self.my_logger.warning(f'Redshift must be positive or null. Got {redshift}')
            sys.exit()
        HALPHA = Line(656.3, atmospheric=False, label='$H\\alpha$', label_pos=[-0.016, 0.02])
        HBETA = Line(486.3, atmospheric=False, label='$H\\beta$', label_pos=[0.007, 0.02])
        HGAMMA = Line(434.0, atmospheric=False, label='$H\\gamma$', label_pos=[0.007, 0.02])
        HDELTA = Line(410.2, atmospheric=False, label='$H\\delta$', label_pos=[0.007, 0.02])
        OIII = Line(500.7, atmospheric=False, label='$O_{III}$', label_pos=[0.007, 0.02])
        CII1 = Line(723.5, atmospheric=False, label='$C_{II}$', label_pos=[0.005, 0.92])
        CII2 = Line(711.0, atmospheric=False, label='$C_{II}$', label_pos=[0.005, 0.02])
        CIV = Line(706.0, atmospheric=False, label='$C_{IV}$', label_pos=[-0.016, 0.92])
        CII3 = Line(679.0, atmospheric=False, label='$C_{II}$', label_pos=[0.005, 0.02])
        CIII1 = Line(673.0, atmospheric=False, label='$C_{III}$', label_pos=[-0.016, 0.92])
        CIII2 = Line(570.0, atmospheric=False, label='$C_{III}$', label_pos=[0.007, 0.02])
        CIII3 = Line(970.5, atmospheric=False, label='$C_{III}$', label_pos=[0.007, 0.02])
        FEII1 = Line(515.8, atmospheric=False, label='$Fe_{II}$', label_pos=[0.007, 0.02])
        FEII2 = Line(527.3, atmospheric=False, label='$Fe_{II}$', label_pos=[0.007, 0.02])
        FEII3 = Line(534.9, atmospheric=False, label='$Fe_{II}$', label_pos=[0.007, 0.02])
        HEI1 = Line(388.8, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI2 = Line(447.1, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI3 = Line(587.5, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI4 = Line(750.0, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI5 = Line(776.0, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI6 = Line(781.6, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI7 = Line(848.2, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI8 = Line(861.7, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI9 = Line(906.5, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI10 = Line(923.5, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI11 = Line(951.9, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI12 = Line(1023.5, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        HEI13 = Line(353.1, atmospheric=False, label='$He_{I}$', label_pos=[0.007, 0.02])
        OI = Line(630.0, atmospheric=False, label='$O_{II}$', label_pos=[0.007, 0.02])
        OII = Line(732.5, atmospheric=False, label='$O_{II}$', label_pos=[0.007, 0.02])
        HEII1 = Line(468.6, atmospheric=False, label='$He_{II}$', label_pos=[0.007, 0.02])
        HEII2 = Line(611.8, atmospheric=False, label='$He_{II}$', label_pos=[0.007, 0.02])
        HEII3 = Line(617.1, atmospheric=False, label='$He_{II}$', label_pos=[0.007, 0.02])
        HEII4 = Line(856.7, atmospheric=False, label='$He_{II}$', label_pos=[0.007, 0.02])
        HI = Line(833.9, atmospheric=False, label='$He_{II}$', label_pos=[0.007, 0.02])
        CAII1 = Line(393.366, atmospheric=True, label='$Ca_{II}$',
                     label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        CAII2 = Line(396.847, atmospheric=True, label='$Ca_{II}$',
                     label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        O2 = Line(762.1, atmospheric=True, label='$O_2$',
                  label_pos=[0.007, 0.02])  # http://onlinelibrary.wiley.com/doi/10.1029/98JD02799/pdf
        # O2_1 = Line( 760.6,atmospheric=True,label='',label_pos=[0.007,0.02]) # libradtran paper fig.3
        # O2_2 = Line( 763.2,atmospheric=True,label='$O_2$',label_pos=[0.007,0.02])  # libradtran paper fig.3
        O2B = Line(686.719, atmospheric=True, label='$O_2(B)$',
                   label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        O2Y = Line(898.765, atmospheric=True, label='$O_2(Y)$',
                   label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        O2Z = Line(822.696, atmospheric=True, label='$O_2(Z)$',
                   label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
        # H2O = Line( 960,atmospheric=True,label='$H_2 O$',label_pos=[0.007,0.02],width_bounds=(1,50))  #
        H2O_1 = Line(935, atmospheric=True, label='$H_2 O$', label_pos=[0.007, 0.02],
                     width_bounds=[5, 30])  # libradtran paper fig.3, broad line
        H2O_2 = Line(960, atmospheric=True, label='$H_2 O$', label_pos=[0.007, 0.02],
                     width_bounds=[5, 30])  # libradtran paper fig.3, broad line

        self.lines = [HALPHA, HBETA, HGAMMA, HDELTA, O2, O2B, O2Y, O2Z, H2O_1, H2O_2, OIII, CII1, CII2, CIV, CII3,
                      CIII1, CIII2, CIII3, HEI1, HEI2, HEI3, HEI4, HEI5, HEI6, HEI7, HEI8, HEI9, HEI10, HEI11, HEI12,
                      HEI13, OI, OII, HEII1, HEII2, HEII3, HEII4, CAII1, CAII2, HI, FEII1, FEII2, FEII3]
        self.redshift = redshift
        self.atmospheric_lines = atmospheric_lines
        self.hydrogen_only = hydrogen_only
        self.emission_spectrum = emission_spectrum
        self.lines = self.sort_lines()

    def sort_lines(self):
        """Sort the lines in increasing order of wavelength."""
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
        """Plot the atomic lines as vertical lines.

        Args:
            ax: an Axes instance
            color_atomic: color of the atomic lines (default: 'g')
            color_atmospheric: color of the atmospheric lines (default: 'b')
            fontsize: font size of the labels (default: 12)
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

    def detect_lines(self, lambdas, spec, spec_err=None, snr_minlevel=3, ax=None, verbose=False):
        """Detect and fit the lines in a spectrum. The method is to look at maxima or minima
        around emission or absorption tabulated lines, and to select surrounding pixels
        to fit a (positive or negative) gaussian and a polynomial background. If several regions
        overlap, a multi-gaussian fit is performed above a common polynomial background.
        The mean global shift (in nm) between the detected and tabulated lines is returned, considering
        only the lines with a signal-to-noise ratio above a threshold.
        The order of the polynomial bakcground is set in parameters.py with BGD_ORDER.

        Args:
            lambdas: the wavelength array (in nm)
            spec: the spectrum array
            spec_err: the spectrum uncertainty array
            snr_minlevel: minimum signal-to-noise ratio for line detection
            ax: an Axes instance (optional: for plot only)
            verbose: if True print many stuff.

        Returns:
            shift: the mean shift (in nm) between the detected and tabulated lines
        """

        # main settings
        bgd_npar = parameters.BGD_NPARAMS
        peak_look = 7  # half range to look for local maximum in pixels
        bgd_width = 7  # size of the peak sides to use to fit spectrum base line
        if self.hydrogen_only:
            peak_look = 15
            bgd_width = 20
        baseline_prior = 1e-10  # *sigma gaussian prior on base line fit

        # initialisation
        lambda_shifts = []
        snrs = []
        index_list = []
        peak_index_list = []
        guess_list = []
        bounds_list = []
        lines_list = []
        for line in self.lines:
            # wavelength of the line: find the nearest pixel index
            line_wavelength = line.wavelength
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
            index = list(range(l_index - peak_look, l_index + peak_look))
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
                test_index = list(range(index_inf, peak_index))
                minm = argrelextrema(spec[test_index], bgd_strategy)
                if len(minm[0]) > 0:
                    index_inf = index_inf + minm[0][0]
                    break
                else:
                    index_inf -= 1
            index_sup = peak_index + 1  # extrema on the right
            while index_sup < min(len(spec) - 1, peak_index + 3 * peak_look):
                test_index = list(range(peak_index, index_sup))
                minm = argrelextrema(spec[test_index], bgd_strategy)
                if len(minm[0]) > 0:
                    index_sup = peak_index + minm[0][0]
                    break
                else:
                    index_sup += 1
            # pixel range to consider around the peak, adding bgd_width pixels
            # to fit for background around the peak
            index = list(range(max(0, index_inf - bgd_width), min(len(lambdas), index_sup + bgd_width)))
            # first guess and bounds to fit the line properties and
            # the background with BGD_ORDER order polynom
            guess = [0] * bgd_npar + [0.5 * np.max(spec[index]), lambdas[peak_index],
                                      0.5 * (line.width_bounds[0] + line.width_bounds[1])]
            if line_strategy == np.less:
                guess[bgd_npar] = -0.5 * np.max(spec[index])  # look for abosrption under bgd
            bounds = [[-np.inf] * bgd_npar + [-abs(np.max(spec[index])), lambdas[index_inf], line.width_bounds[0]],
                      [np.inf] * bgd_npar + [abs(np.max(spec[index])), lambdas[index_sup], line.width_bounds[1]]]
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
                        new_guess_list[-1][bgd_npar + 3 * k + 1] + new_guess_list[-1][bgd_npar + 3 * (
                        k + 1) + 1]) + 1e-3  # last term is to avoid equalities between bounds in some pathological case
            # sort pixel indices and remove doublons
            new_index_list[-1] = sorted(list(set(new_index_list[-1])))
        # fit the line subsets and background
        rows = []
        for k in range(len(new_index_list)):
            # first guess for the base line with the lateral bands
            peak_index = new_peak_index_list[k]
            index = new_index_list[k]
            guess = new_guess_list[k]
            bounds = new_bounds_list[k]
            bgd_index = index[:bgd_width] + index[-bgd_width:]
            try:
                bgd = fit_poly1d_outlier_removal(lambdas[bgd_index], spec[bgd_index],
                                                 order=BGD_ORDER, sigma=2, niter=300)
            except:
                bgd = fit_poly1d_outlier_removal(lambdas[bgd_index], spec[bgd_index],
                                                 order=BGD_ORDER, sigma=3, niter=300)
            # f = plt.figure()
            # plt.errorbar(lambdas[index],spec[index],yerr=spec_err[index])
            # plt.plot(lambdas[bgd_index],spec[bgd_index],'r-')
            # plt.plot(lambdas[index],bgd(lambdas[index]),'b--')
            # plt.show()
            for n in range(bgd_npar):
                guess[n] = getattr(bgd, bgd.param_names[BGD_ORDER - n]).value
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
            if spec_err is not None:
                noise_level = np.sqrt(np.mean(spec_err[index]**2))
            # f = plt.figure()
            # plt.errorbar(lambdas[index],spec[index],yerr=spec_err[index])
            # plt.plot(lambdas[bgd_index],spec[bgd_index])
            # plt.plot(lambdas[index],np.polyval(popt[:bgd_npar],lambdas[index]),'b--')
            # plt.plot(lambdas[index],multigauss_and_bgd(lambdas[index],*popt),'b-')
            # plt.plot(lambdas[index],multigauss_and_bgd(lambdas[index],*guess),'g-')
            # plt.plot(lambdas[index],base_line,'r-')
            # plt.show()
            plot_line_subset = False
            for j in range(len(new_lines_list[k])):
                line = new_lines_list[k][j]
                l = line.wavelength
                peak_pos = popt[bgd_npar + 3 * j + 1]
                # FWHM
                FWHM = np.abs(popt[bgd_npar + 3 * j + 2]) * 2.355
                # SNR computation
                # signal_level = popt[bgd_npar+3*j]
                signal_level = popt[bgd_npar + 3 * j] #multigauss_and_bgd(peak_pos, *popt) - np.polyval(popt[:bgd_npar], peak_pos)
                snr = np.abs(signal_level / noise_level)
                # save fit results
                line.fitted = True
                line.fit_lambdas = lambdas[index]
                line.fit_gauss = gauss(lambdas[index], *popt[bgd_npar + 3 * j:bgd_npar + 3 * j + 3])
                line.fit_bgd = np.polyval(popt[:bgd_npar], lambdas[index])
                line.fit_snr = snr
                line.fit_fwhm = FWHM
                if snr < snr_minlevel:
                    continue
                line.high_snr = True
                plot_line_subset = True
                rows.append((line.label, l, peak_pos, peak_pos - l, FWHM, signal_level, snr))
                # wavelength shift between tabulate and observed lines
                lambda_shifts.append(peak_pos - l)
                snrs.append(snr)
            if ax is not None and plot_line_subset:
                ax.plot(lambdas[index], multigauss_and_bgd(lambdas[index], *popt), lw=2, color='b')
                ax.plot(lambdas[index], np.polyval(popt[:bgd_npar], lambdas[index]), lw=2, color='b', linestyle='--')
        if len(rows) > 0:
            t = Table(rows=rows, names=('Line', 'Tabulated', 'Detected', 'Shift', 'FWHM', 'Amplitude', 'SNR'),
                      dtype=('a10', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
            for col in t.colnames[1:-2]:
                t[col].unit = 'nm'
            if verbose:
                print(t)
            shift = np.average(lambda_shifts, weights=np.array(snrs) ** 2)
            # remove lines with low SNR from plot
            for line in self.lines:
                if line.label not in t['Line'][:]:
                    line.high_snr = False
        else:
            shift = 0
        return shift


class Spectrum(object):
    """ Spectrum class used to store information and methods
    relative to spectra nd their extraction.
    """

    def __init__(self, filename="", Image=None, atmospheric_lines=True, order=1, target=None):
        """
        Args:
            atmospheric_lines (bool): if True atmospheric absorption lines
                are plotted and eventually used for the spectrum calibration
            order (int): order of the spectrum
            target (:obj:`str`): name of the image target
            filename (:obj:`str`): path to the image file
            Image (:obj:`Image`): copy info from Image object
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.target = target
        self.data = None
        self.err = None
        self.lambdas = None
        self.lambdas_binwidths = None
        self.lambdas_indices = None
        self.order = order
        self.filter = None
        self.filters = None
        self.units = 'ADU/s'
        self.gain = parameters.GAIN
        if filename != "":
            self.filename = filename
            self.load_spectrum(filename)
        if Image is not None:
            self.header = Image.header
            self.date_obs = Image.date_obs
            self.airmass = Image.airmass
            self.expo = Image.expo
            self.filters = Image.filters
            self.filter = Image.filter
            self.disperser = Image.disperser
            self.target = Image.target
            self.target_pixcoords = Image.target_pixcoords
            self.target_pixcoords_rotated = Image.target_pixcoords_rotated
            self.units = Image.units
            self.gain = Image.gain
            self.my_logger.info('\n\tSpectrum info copied from Image')
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
        >>> s = Spectrum()
        >>> s.lambdas = np.linspace(parameters.LAMBDA_MIN,parameters.LAMBDA_MAX,100)
        >>> s.lambdas_binwidths = np.gradient(s.lambdas)
        >>> s.data = np.ones_like(s.lambdas)
        >>> s.convert_from_ADUrate_to_flam()
        >>> assert np.max(s.data) < 1e-14

        """

        self.data = self.data / FLAM_TO_ADURATE
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
        >>> s = Spectrum()
        >>> s.lambdas = np.linspace(parameters.LAMBDA_MIN,parameters.LAMBDA_MAX,100)
        >>> s.lambdas_binwidths = np.gradient(s.lambdas)
        >>> s.data = np.ones_like(s.lambdas)
        >>> s.convert_from_flam_to_ADUrate()
        >>> assert np.max(s.data) > 1e14

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
        for f in FILTERS:
            if f['label'] == self.filter:
                parameters.LAMBDA_MIN = f['min']
                parameters.LAMBDA_MAX = f['max']
                self.my_logger.info('\n\tLoad filter %s: lambda between %.1f and %.1f' % (
                    f['label'], parameters.LAMBDA_MIN, parameters.LAMBDA_MAX))
                break

    def plot_spectrum_simple(self, ax, xlim=None, color='r', label='', lambdas=None):
        """Simple function to plot a spectrum with error bars and labels.

        Args:
            label: string label for the plot legend
            color: character for the color of the spectrum (default: 'r')
            ax: Axes instance of a figure
            xlim: (optional) list of minimum and maximum abscisses
        """
        xs = self.lambdas
        if lambdas is not None: xs = lambdas
        if label == '':
            label = 'Order {:d} spectrum'.format(self.order)
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
        ax.set_ylim(0., np.max(self.data) * 1.2)
        ax.set_xlabel('$\lambda$ [nm]')
        ax.set_ylabel('Flux [%s]' % self.units)

    def plot_spectrum(self, xlim=None, fit=True):
        """Plot spectrum with emission and absorption lines.

        Args:
            xlim: (optional) list of minimum and maximum abscisses
            fit: if True lines are fitted (default: True).
        """
        fig = plt.figure(figsize=[12, 6])
        self.plot_spectrum_simple(plt.gca(), xlim=xlim)
        # if len(self.target.spectra)>0:
        #    for k in range(len(self.target.spectra)):
        #        s = self.target.spectra[k]/np.max(self.target.spectra[k])*np.max(self.data)
        #        plt.plot(self.target.wavelengths[k],s,lw=2,label='Tabulated spectra #%d' % k)
        if fit and self.lambdas is not None:
            self.lines.detect_lines(self.lambdas, self.data, spec_err=self.err, ax=plt.gca(),
                                    verbose=parameters.VERBOSE)
        if self.lambdas is not None and self.lines is not None:
            self.lines.plot_atomic_lines(plt.gca(), fontsize=12)
        plt.legend(loc='best')
        if self.filters is not None:
            plt.gca().get_legend().set_title(self.filters)
        plt.show()

    def save_spectrum(self, output_file_name, overwrite=False):
        """Save the spectrum into a fits file (data, error and wavelengths).

        Args:
            output_file_name: path to the output fits file
            overwrite: if True overwrite the output file if needed.
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

        Args:
            input_file_name: path to the input fits file
        """
        if os.path.isfile(input_file_name):
            self.header, raw_data = load_fits(input_file_name)
            self.lambdas = raw_data[0]
            self.data = raw_data[1]
            if len(raw_data) > 2:
                self.err = raw_data[2]
            extract_info_from_CTIO_header(self, self.header)
            if self.header['TARGET'] != "":
                self.target = Target(self.header['TARGET'], verbose=parameters.VERBOSE)
            if self.header['UNIT2'] != "":
                self.units = self.header['UNIT2']
            self.my_logger.info('\n\tSpectrum loaded from %s' % input_file_name)
        else:
            self.my_logger.warning('\n\tSpectrum file %s not found' % input_file_name)


def calibrate_spectrum(spectrum, xlim=None):
    """Convert pixels into wavelengths given the position of the order 0,
    the data for the spectrum, and the properties of the disperser.

    Args:
        xlim: (optional) list of minimum and maximum abscisses
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

    Returns:
        float: the final shift value of the spectrum in nm

    """
    # Detect emission/absorption lines and calibrate pixel/lambda
    D = DISTANCE2CCD - DISTANCE2CCD_ERR
    shifts = []
    counts = 0
    D_step = DISTANCE2CCD_ERR / 4
    delta_pixels = spectrum.lambdas_indices - int(spectrum.target_pixcoords_rotated[0])
    lambdas_test = spectrum.disperser.grating_pixel_to_lambda(delta_pixels, spectrum.target_pixcoords,
                                                              order=spectrum.order)
    while DISTANCE2CCD + 4 * DISTANCE2CCD_ERR > D > DISTANCE2CCD - 4 * DISTANCE2CCD_ERR and counts < 30:
        spectrum.disperser.D = D
        lambdas_test = spectrum.disperser.grating_pixel_to_lambda(delta_pixels,
                                                                  spectrum.target_pixcoords, order=spectrum.order)
        lambda_shift = spectrum.lines.detect_lines(lambdas_test, spectrum.data, spec_err=spectrum.err, ax=None,
                                                   verbose=parameters.DEBUG)
        shifts.append(lambda_shift)
        counts += 1
        if abs(lambda_shift) < 0.1:
            break
        elif lambda_shift > 2:
            D_step = DISTANCE2CCD_ERR
        elif 0.5 < lambda_shift < 2:
            D_step = DISTANCE2CCD_ERR / 4
        elif 0 < lambda_shift < 0.5:
            D_step = DISTANCE2CCD_ERR / 10
        elif 0 > lambda_shift > -0.5:
            D_step = -DISTANCE2CCD_ERR / 20
        elif lambda_shift < -0.5:
            D_step = -DISTANCE2CCD_ERR / 6
        D += D_step
    shift = np.mean(lambdas_test - spectrum.lambdas)
    spectrum.lambdas = lambdas_test
    lambda_shift = spectrum.lines.detect_lines(spectrum.lambdas, spectrum.data, spec_err=spectrum.err, ax=None,
                                               verbose=parameters.DEBUG)
    spectrum.my_logger.info(
        '\n\tWavelenght total shift: {:.2f}nm (after {:d} steps)'
        '\n\twith D = {:.2f} mm (DISTANCE2CCD = {:.2f} +/- {:.2f} mm, {:.1f} sigma shift)'.format(
            shift, len(shifts), D, DISTANCE2CCD, DISTANCE2CCD_ERR, (D - DISTANCE2CCD) / DISTANCE2CCD_ERR))
    spectrum.header['LSHIFT'] = shift
    spectrum.header['D2CCD'] = D
    return lambda_shift


if __name__ == "__main__":
    import doctest

    doctest.testmod()

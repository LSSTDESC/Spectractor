from astropy.table import Table
from scipy.interpolate import interp1d
import numpy as np

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.tools import gauss, multigauss_and_bgd, rescale_x_for_legendre


class Line:
    """Class modeling the emission or absorption lines."""

    def __init__(self, wavelength, label, atmospheric=False, emission=False, label_pos=[0.007, 0.02],
                 width_bounds=[0.5, 6], use_for_calibration=False):
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
        self.use_for_calibration = use_for_calibration
        self.high_snr = False
        self.fit_lambdas = None
        self.fit_gauss = None
        self.fit_bgd = None
        self.fit_snr = None
        self.fit_fwhm = None
        self.fit_popt = None
        self.fit_chisq = None
        self.fit_bgd_npar = parameters.CALIB_BGD_NPARAMS

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
            interp = interp1d(self.fit_lambdas, self.fit_gauss, bounds_error=False, fill_value=0.)
            return interp(lambdas)
        else:
            return gauss(lambdas, A=A, x0=self.wavelength, sigma=sigma)


class Lines:
    """Class gathering all the lines and associated methods."""

    def __init__(self, lines, redshift=0, atmospheric_lines=True, hydrogen_only=False, emission_spectrum=False):
        """ Main emission/absorption lines in nm. Sorted lines are sorted in self.lines.
        See http://www.pa.uky.edu/~peter/atomic/ or https://physics.nist.gov/PhysRefData/ASD/lines_form.html

        Parameters
        ----------
        lines: list
            List of Line objects to gather and sort.
        redshift: float, optional
            Red shift the spectral lines. Must be positive or null (default: 0)
        atmospheric_lines: bool, optional
            Set True if the atmospheric spectral lines must be included (default: True)
        hydrogen_only: bool, optional
            Set True to gather only the hydrogen spectral lines, atmospheric lines still included (default: False)
        emission_spectrum: bool, optional
            Set True if the spectral line has to be detected in emission (default: False)

        Examples
        --------
        The default first five lines:
        >>> lines = Lines(ISM_LINES+HYDROGEN_LINES, redshift=0, atmospheric_lines=False, hydrogen_only=False, emission_spectrum=False)
        >>> print([lines.lines[i].wavelength for i in range(5)])
        [353.1, 388.8, 410.2, 434.0, 447.1]

        The four hydrogen lines only:
        >>> lines = Lines(ISM_LINES+HYDROGEN_LINES+ATMOSPHERIC_LINES, redshift=0, atmospheric_lines=False, hydrogen_only=True, emission_spectrum=True)
        >>> print([lines.lines[i].wavelength for i in range(4)])
        [410.2, 434.0, 486.3, 656.3]
        >>> print(lines.emission_spectrum)
        True

        Redshift the hydrogen lines, the atmospheric lines stay unchanged:
        >>> lines = Lines(ISM_LINES+HYDROGEN_LINES+ATMOSPHERIC_LINES, redshift=1, atmospheric_lines=True, hydrogen_only=True, emission_spectrum=True)
        >>> print([lines.lines[i].wavelength for i in range(7)])
        [382.044, 393.366, 396.847, 430.79, 438.355, 686.719, 762.1]

        Redshift all the spectral lines, except the atmospheric lines:
        >>> lines = Lines(ISM_LINES+HYDROGEN_LINES+ATMOSPHERIC_LINES, redshift=1, atmospheric_lines=True, hydrogen_only=False, emission_spectrum=True)
        >>> print([lines.lines[i].wavelength for i in range(5)])
        [382.044, 393.366, 396.847, 430.79, 438.355]

        Negative redshift:
        >>> lines = Lines(HYDROGEN_LINES, redshift=-0.5)

        """
        self.my_logger = set_logger(self.__class__.__name__)
        if redshift < 0:
            self.my_logger.error(f'\n\tRedshift must be positive or null. Got redshift={redshift}.')
        self.lines = lines
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
        >>> lines = Lines(HYDROGEN_LINES+ATMOSPHERIC_LINES, redshift=0)
        >>> sorted_lines = lines.sort_lines()
        >>> print([l.wavelength for l in sorted_lines][:5])
        [382.044, 393.366, 396.847, 410.2, 430.79]
        """
        sorted_lines = []
        import copy
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
            sorted_lines.append(copy.copy(line))
        if self.redshift > 0:
            for line in sorted_lines:
                if not line.atmospheric:
                    line.wavelength *= (1 + self.redshift)
        sorted_lines = sorted(sorted_lines, key=lambda x: x.wavelength)
        return sorted_lines

    def plot_atomic_lines(self, ax, color_atomic='g', color_atmospheric='b', fontsize=12, force=False):
        """Over plot the atomic lines as vertical lines, only if they are fitted or with high
        signal to  noise ratio, unless force keyword is set to True.

        Parameters
        ----------
        ax: Axes
            An Axes instance on which plot the spectral lines.
        color_atomic: str
            Color of the atomic lines (default: 'g').
        color_atmospheric: str
            Color of the atmospheric lines (default: 'b').
        fontsize: int
            Font size of the spectral line labels (default: 12).
        force: bool
            Force the plot of vertical lines if set to True (default: False).

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> f, ax = plt.subplots(1,1)
        >>> ax.set_xlim(300,1000)
        (300, 1000)
        >>> lines = Lines(HYDROGEN_LINES+ATMOSPHERIC_LINES)
        >>> lines.lines[5].fitted = True
        >>> lines.lines[5].high_snr = True
        >>> lines.lines[-1].fitted = True
        >>> lines.lines[-1].high_snr = True
        >>> ax = lines.plot_atomic_lines(ax)
        >>> assert ax is not None
        >>> if parameters.DISPLAY: plt.show()
        """
        xlim = ax.get_xlim()
        for l in self.lines:
            if (not l.fitted or not l.high_snr) and not force:
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

    def plot_detected_lines(self, ax=None, print_table=False):
        """Detect and fit the lines in a spectrum. The method is to look at maxima or minima
        around emission or absorption tabulated lines, and to select surrounding pixels
        to fit a (positive or negative) gaussian and a polynomial background. If several regions
        overlap, a multi-gaussian fit is performed above a common polynomial background.
        The mean global shift (in nm) between the detected and tabulated lines is returned, considering
        only the lines with a signal-to-noise ratio above a threshold.
        The order of the polynomial background is set in parameters.py with CALIB_BGD_ORDER.

        Parameters
        ----------
        ax: Axes
            The Axes instance if needed (default: None).
        print_table: bool, optional
            If True, print a summary table (default: False).

        Examples
        --------

        Creation of a mock spectrum with emission and absorption lines
        >>> from spectractor.extractor.spectrum import Spectrum, detect_lines
        >>> lambdas = np.arange(300,1000,1)
        >>> spectrum = 1e4*np.exp(-((lambdas-600)/200)**2)
        >>> spectrum += HALPHA.gaussian_model(lambdas, A=5000, sigma=3)
        >>> spectrum += HBETA.gaussian_model(lambdas, A=3000, sigma=2)
        >>> spectrum += O2.gaussian_model(lambdas, A=-3000, sigma=7)
        >>> spectrum_err = np.sqrt(spectrum)
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
        >>> import matplotlib.pyplot as plt
        >>> from spectractor.tools import plot_spectrum_simple
        >>> spec.lines = lines
        >>> fig = plt.figure()
        >>> plot_spectrum_simple(plt.gca(), lambdas, spec.data, data_err=spec.err)
        >>> lines.plot_detected_lines(plt.gca())
        >>> if parameters.DISPLAY: plt.show()
        """
        lambdas = np.zeros(1)
        rows = []
        j = 0
        for line in self.lines:
            if line.fitted is True:
                # look for lines in subset fit
                bgd_npar = line.fit_bgd_npar
                parameters.CALIB_BGD_NPARAMS = bgd_npar
                if lambdas.shape != line.fit_lambdas.shape or not np.allclose(lambdas, line.fit_lambdas, 1e-3):
                    j = 0
                    lambdas = np.copy(line.fit_lambdas)
                    if ax is not None:
                        ax.plot(lambdas, multigauss_and_bgd(lambdas, *line.fit_popt), lw=2, color='b')
                        x_norm = rescale_x_for_legendre(lambdas)
                        bgd = np.polynomial.legendre.legval(x_norm, line.fit_popt[0:bgd_npar])
                        ax.plot(lambdas, bgd, lw=2, color='b', linestyle='--')
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
            for col in t.colnames[1:-3]:
                t[col].unit = 'nm'
            t[t.colnames[-1]].unit = 'reduced'
            print(t)


# Line catalog

# Hydrogen lines
HALPHA = Line(656.3, atmospheric=False, label='$H\\alpha$', label_pos=[-0.016, 0.02], use_for_calibration=True)
HBETA = Line(486.3, atmospheric=False, label='$H\\beta$', label_pos=[0.007, 0.02], use_for_calibration=True)
HGAMMA = Line(434.0, atmospheric=False, label='$H\\gamma$', label_pos=[0.007, 0.02], use_for_calibration=True)
HDELTA = Line(410.2, atmospheric=False, label='$H\\delta$', label_pos=[0.007, 0.02])
HYDROGEN_LINES = [HALPHA, HBETA, HGAMMA, HDELTA]

# Atmospheric lines
FE1 = Line(382.044, atmospheric=True, label=r'$Fe$',
           label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
FE2 = Line(430.790, atmospheric=True, label=r'$Fe$',
           label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
FE3 = Line(438.355, atmospheric=True, label=r'$Fe$',
           label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
CAII1 = Line(393.366, atmospheric=True, label=r'$Ca_{II}$',
             label_pos=[0.007, 0.02],
             use_for_calibration=False)  # https://en.wikipedia.org/wiki/Fraunhofer_lines
CAII2 = Line(396.847, atmospheric=True, label=r'$Ca_{II}$',
             label_pos=[0.007, 0.02],
             use_for_calibration=False)  # https://en.wikipedia.org/wiki/Fraunhofer_lines
# O2 = Line(762.1, atmospheric=True, label=r'$O_2$',
#           label_pos=[0.007, 0.02],
#           use_for_calibration=True)  # http://onlinelibrary.wiley.com/doi/10.1029/98JD02799/pdf
O2_1 = Line(760.6, atmospheric=True, label='', label_pos=[0.007, 0.02],
            use_for_calibration=True)  # libradtran paper fig.3
O2_2 = Line(763.2, atmospheric=True, label='$O_2$', label_pos=[0.007, 0.02],
            use_for_calibration=True)  # libradtran paper fig.3
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
ATMOSPHERIC_LINES = [O2_1, O2_2, O2B, O2Y, O2Z, H2O_1, H2O_2, CAII1, CAII2, FE1, FE2, FE3]

# ISM lines
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
HI = Line(833.9, atmospheric=False, label=r'$H_{I}$', label_pos=[0.007, 0.02])
ISM_LINES = [OIII, CII1, CII2, CIV, CII3, CIII1, CIII2, CIII3, HEI1, HEI2, HEI3, HEI4, HEI5, HEI6, HEI7, HEI8,
             HEI9, HEI10, HEI11, HEI12, HEI13, OI, OII, HEII1, HEII2, HEII3, HEII4, HI, FEII1, FEII2, FEII3, FEII4]

# HG-AR lines https://oceanoptics.com/wp-content/uploads/hg1.pdf
HG1 = Line(253.652, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02])
HG2 = Line(296.728, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02])
HG3 = Line(302.150, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02])
HG4 = Line(313.155, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02])
HG5 = Line(334.148, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02])
HG6 = Line(365.015, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02], use_for_calibration=True)
HG7 = Line(404.656, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02], use_for_calibration=True)
HG8 = Line(407.783, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02], use_for_calibration=True)
HG9 = Line(435.833, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02])
HG10 = Line(546.074, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02], use_for_calibration=True)
HG11 = Line(576.960, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02], use_for_calibration=True)
HG12 = Line(579.066, atmospheric=False, label=r'$Hg$', label_pos=[0.007, 0.02], use_for_calibration=True)
AR1 = Line(696.543, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR2 = Line(706.722, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR3 = Line(714.704, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR4 = Line(727.294, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR5 = Line(738.398, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR6 = Line(750.387, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR7 = Line(763.511, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR8 = Line(772.376, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR9 = Line(794.818, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR10 = Line(800.616, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR11 = Line(811.531, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR12 = Line(826.452, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR13 = Line(842.465, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR14 = Line(852.144, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR15 = Line(866.794, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR16 = Line(912.297, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
AR17 = Line(922.450, atmospheric=False, label=r'$Ar$', label_pos=[0.007, 0.02])
HGAR_LINES = [HG1, HG2, HG3, HG4, HG5, HG6, HG7, HG8, HG9, HG10, HG11, HG12,
              AR1, AR2, AR3, AR4, AR5, AR6, AR7, AR8, AR9, AR10, AR11, AR12, AR13, AR14, AR15, AR16, AR17]

if __name__ == "__main__":
    import doctest

    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    doctest.testmod()

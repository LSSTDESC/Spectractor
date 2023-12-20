import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import interp1d
import getCalspec
import copy

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.extractor.spectrum import Spectrum
from spectractor.simulation.simulator import SpectrumSimulation
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.fit.fitter import (FitWorkspace, FitParameters, run_minimisation_sigma_clipping, run_minimisation,
                                    write_fitparameter_json)
from spectractor.tools import plot_spectrum_simple


class SpectrumFitWorkspace(FitWorkspace):

    def __init__(self, spectrum, atmgrid_file_name="", fit_angstrom_exponent=False,
                 verbose=False, plot=False, live_fit=False, truth=None):
        """Class to fit a spectrum extracted with Spectractor.

        The spectrum is supposed to be the product of the star SED, the instrument throughput and the atmospheric
        transmission, contaminated eventually by a second order diffraction.
        The truth parameters are loaded from the file header if provided.
        If provided, the atmospheric grid is used for the atmospheric transmission simulations and interpolated
        with splines, otherwise Libradtran is called at each step (slower).

        Parameters
        ----------
        spectrum: Spectrum
            Spectrum object.
        atmgrid_file_name: str, optional
            Atmospheric grid file name (default: "").
        fit_angstrom_exponent: bool, optional
            If True, fit angstrom exponent (default: False).
        verbose: bool, optional
            Verbosity level (default: False).
        plot: bool, optional
            If True, many plots are produced (default: False).
        live_fit: bool, optional
            If True, many plots along the fitting procedure are produced to see convergence in live (default: False).
        truth: array_like, optional
            Array of truth parameters to compare with the best fit result (default: None).

        Examples
        --------

        >>> from spectractor.config import load_config
        >>> load_config("config/ctio.ini")
        >>> spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
        >>> atmgrid_filename = spec.filename.replace('spectrum', 'atmsim')
        >>> w = SpectrumFitWorkspace(spec, atmgrid_file_name=atmgrid_filename, verbose=True, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.params.values)
        >>> w.plot_fit()

        """
        self.my_logger = set_logger(self.__class__.__name__)
        if not getCalspec.is_calspec(spectrum.target.label):
            raise ValueError(f"{spectrum.target.label=} must be a CALSPEC star according to getCalspec package.")
        self.spectrum = spectrum
        p = np.array([1, 0, 0.05, 1.2, 400, 5, 1, self.spectrum.header['D2CCD'], self.spectrum.header['PIXSHIFT'], 0])
        fixed = [False] * p.size
        # fixed[0] = True
        fixed[1] = "A2_T" not in self.spectrum.header  # fit A2 only on sims to evaluate extraction biases
        fixed[5] = False
        # fixed[6:8] = [True, True]
        fixed[8] = True
        fixed[9] = True
        # fixed[-1] = True
        if not fit_angstrom_exponent:
            fixed[3] = True  # angstrom_exponent
        bounds = [(0, 2), (0, 2/parameters.GRATING_ORDER_2OVER1), (0, 0.1), (0, 3), (100, 700), (0, 20),
                       (0.1, 10),(p[7] - 5 * parameters.DISTANCE2CCD_ERR, p[7] + 5 * parameters.DISTANCE2CCD_ERR),
                  (-2, 2), (-np.inf, np.inf)]
        params = FitParameters(p, labels=["A1", "A2", "VAOD", "angstrom_exp", "ozone [db]", "PWV [mm]",
                                          "reso [pix]", r"D_CCD [mm]", r"alpha_pix [pix]", "B"],
                               axis_names=["$A_1$", "$A_2$", "VAOD", r'$\"a$', "ozone [db]", "PWV [mm]",
                                           "reso [pix]", r"$D_{CCD}$ [mm]", r"$\alpha_{\mathrm{pix}}$ [pix]", "$B$"],
                               bounds=bounds, fixed=fixed, truth=truth, filename=spectrum.filename)
        FitWorkspace.__init__(self, params, verbose=verbose, plot=plot, live_fit=live_fit, file_name=spectrum.filename)
        if atmgrid_file_name == "":
            self.atmosphere = Atmosphere(self.spectrum.airmass, self.spectrum.pressure, self.spectrum.temperature)
            if self.atmosphere.emulator is not None:
                self.params.bounds[self.params.get_index("ozone [db]")] = (self.atmosphere.emulator.OZMIN, self.atmosphere.emulator.OZMAX)
                self.params.bounds[self.params.get_index("PWV [mm]")] = (self.atmosphere.emulator.PWVMIN, self.atmosphere.emulator.PWVMAX)
        else:
            self.use_grid = True
            self.atmosphere = AtmosphereGrid(spectrum_filename=spectrum.filename, atmgrid_filename=atmgrid_file_name)
            self.params.bounds[2] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
            self.params.bounds[4] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.params.bounds[5] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.params.fixed[self.params.get_index("angstrom_exp")] = True  # angstrom exponent
            if parameters.VERBOSE:
                self.my_logger.info(f'\n\tUse atmospheric grid models from file {atmgrid_file_name}. ')
        self.params.values[self.params.get_index("angstrom_exp")] = self.atmosphere.angstrom_exponent_default
        self.lambdas = self.spectrum.lambdas
        self.data = self.spectrum.data
        self.err = self.spectrum.err
        self.data_cov = self.spectrum.cov_matrix
        self.fit_angstrom_exponent = fit_angstrom_exponent
        self.params.values[self.params.get_index("angstrom_exp")] = self.atmosphere.angstrom_exponent_default
        self.simulation = SpectrumSimulation(self.spectrum, atmosphere=self.atmosphere, fast_sim=True, with_adr=True)
        self.amplitude_truth = None
        self.lambdas_truth = None
        self.get_truth()

    def get_truth(self):
        """Load the truth parameters (if provided) from the file header.

        """
        if 'A1_T' in list(self.spectrum.header.keys()):
            A1_truth = self.spectrum.header['A1_T']
            A2_truth = 0 * self.spectrum.header['A2_T']
            ozone_truth = self.spectrum.header['OZONE_T']
            pwv_truth = self.spectrum.header['PWV_T']
            aerosols_truth = self.spectrum.header['VAOD_T']
            reso_truth = -1
            D_truth = self.spectrum.header['D2CCD_T']
            shift_truth = 0
            B_truth = 0
            self.truth = (A1_truth, A2_truth, aerosols_truth, ozone_truth, pwv_truth,
                          reso_truth, D_truth, shift_truth, B_truth)
            self.lambdas_truth = np.fromstring(self.spectrum.header['LBDAS_T'][1:-1], sep=' ', dtype=float)
            self.amplitude_truth = np.fromstring(self.spectrum.header['AMPLIS_T'][1:-1], sep=' ', dtype=float)
        else:
            self.truth = None

    def plot_spectrum_comparison_simple(self, ax, title='', extent=None, size=0.4):
        """Method to plot a spectrum issued from data and compare it with simulations.

        Parameters
        ----------
        ax: Axes
            Axes instance of shape (4, 2).
        title: str, optional
            Title for the simulation plot (default: '').
        extent: array_like, optional
            Extent argument for imshow to crop plots (default: None).
        size: float, optional
            Relative size of the residual pad (default: 0.4).
        """
        lambdas = self.spectrum.lambdas
        sub = np.where((lambdas > parameters.LAMBDA_MIN) & (lambdas < parameters.LAMBDA_MAX))
        if extent is not None:
            sub = np.where((lambdas > extent[0]) & (lambdas < extent[1]))
        plot_spectrum_simple(ax, lambdas=lambdas, data=self.data, data_err=self.err,
                             units=self.spectrum.units)
        p0 = ax.plot(lambdas, self.model, label='model')
        ax.fill_between(lambdas, self.model - self.model_err,
                        self.model + self.model_err, alpha=0.3, color=p0[0].get_color())
        if self.amplitude_truth is not None:
            ax.plot(self.lambdas_truth, self.amplitude_truth, 'g', label="truth")
        # ax.plot(self.lambdas, self.model_noconv, label='before conv')
        if title != '':
            ax.set_title(title, fontsize=10)
        ax.legend()
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("bottom", size=size, pad=0, sharex=ax)
        ax.figure.add_axes(ax2)
        min_positive = np.min(self.model[self.model > 0])
        idx = np.logical_not(np.isclose(self.model[sub], 0, atol=0.01 * min_positive))
        norm = np.sqrt(self.err[sub][idx]**2 + self.model_err[sub][idx]**2)
        residuals = (self.spectrum.data[sub][idx] - self.model[sub][idx]) / norm
        residuals_err = self.spectrum.err[sub][idx] / norm
        ax2.errorbar(lambdas[sub][idx], residuals, yerr=residuals_err, fmt='ro', markersize=2, label='(Data-Model)/Err')
        ax2.axhline(0, color=p0[0].get_color())
        ax2.grid(True)
        ylim = ax2.get_ylim()
        residuals_model = self.model_err[sub][idx] / self.err[sub][idx]
        ax2.fill_between(lambdas[sub][idx], -residuals_model, residuals_model, alpha=0.3, color=p0[0].get_color())
        std = np.nanstd(residuals)  # max(np.std(residuals), np.std(residuals_model))
        ax2.set_ylim(-5*std, 5*std)
        ax2.set_xlabel(ax.get_xlabel())
        # ax2.set_ylabel('(Data-Model)/Err', fontsize=10)
        ax2.legend()
        ax2.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_ylim((0.9 * np.min(self.spectrum.data[sub]), 1.1 * np.max(self.spectrum.data[sub])))
        ax.set_xticks(ax2.get_xticks()[1:-1])
        ax.get_yaxis().set_label_coords(-0.08, 0.6)
        # ax2.get_yaxis().set_label_coords(-0.11, 0.5)

    def simulate(self, A1, A2, aerosols, angstrom_exponent, ozone, pwv, reso, D, shift_x, B):
        """Interface method to simulate a spectrogram.

        Parameters
        ----------
        A1: float
            Main amplitude parameter.
        A2: float
            Relative amplitude of the order 2 spectrogram.
        aerosols: float
            Vertical Aerosols Optical Depth quantity for Libradtran (no units).
        angstrom_exponent: float
            Angstrom exponent for aerosols.
        ozone: float
            Ozone parameter for Libradtran (in db).
        pwv: float
            Precipitable Water Vapor quantity for Libradtran (in mm).
        reso: float
            Width of the Gaussian kernel to convolve the spectrum.
        D: float
            Distance between the CCD and the disperser (in mm).
        shift_x: float
            Shift of the order 0 position along the X axis (in pixels).
        B: float
            Amplitude of the simulated background (considered flat in ADUs).

        Returns
        -------
        lambdas: array_like
            Array of wavelengths (1D).
        model: array_like
            2D array of the spectrogram simulation.
        model_err: array_like
            2D array of the spectrogram simulation uncertainty.

        Examples
        --------
        >>> parameters.VERBOSE = True
        >>> spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
        >>> atmgrid_filename = spec.filename.replace('spectrum', 'atmsim')
        >>> w = SpectrumFitWorkspace(spec, atmgrid_file_name=atmgrid_filename, verbose=True, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.params.values)
        >>> w.plot_fit()

        """
        if not self.fit_angstrom_exponent:
            angstrom_exponent = None
        lambdas, model, model_err = self.simulation.simulate(A1, A2, aerosols, angstrom_exponent, ozone, pwv, reso, D, shift_x, B)
        self.model = model
        self.model_err = model_err
        return lambdas, model, model_err

    def plot_fit(self):
        """Plot the fit result.

        Examples
        --------
        >>> parameters.VERBOSE = True
        >>> spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
        >>> atmgrid_filename = spec.filename.replace('spectrum', 'atmsim')
        >>> w = SpectrumFitWorkspace(spec, atmgrid_file_name=atmgrid_filename, verbose=True, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.params.values)
        >>> w.plot_fit()

        .. plot::
            :include-source:

            from spectractor.fit.fit_spectrum import SpectrumFitWorkspace
            spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
            atmgrid_filename = spec.filename.replace('spectrum', 'atmsim')
            fit_workspace = SpectrumFitWorkspace(spec, atmgrid_file_name=atmgrid_filename, verbose=True, plot=True, live_fit=False)
            A1, A2, aerosols, ozone, pwv, reso, D, shift_x = fit_workspace.p
            lambdas, model, model_err = fit_workspace.simulation.simulate(A1, A2, aerosols, ozone, pwv, reso, D, shift_x)
            fit_workspace.plot_fit()

        """
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(222)
        ax2 = plt.subplot(224)
        ax3 = plt.subplot(121)
        A1, A2, aerosols, angstrom_exponent, ozone, pwv, reso, D, shift, B = self.params.values
        self.title = f'A1={A1:.3f}, A2={A2:.3f}, PWV={pwv:.3f}, OZ={ozone:.3g}, VAOD={aerosols:.3f},\n ' \
                     f'reso={reso:.2f}pix, D={D:.2f}mm, shift={shift:.2f}pix, B={B:.2g}'
        # main plot
        self.plot_spectrum_comparison_simple(ax3, title="", size=0.8)
        # zoom O2
        self.plot_spectrum_comparison_simple(ax2, extent=[730, 800], title='Zoom $O_2$', size=0.8)
        # zoom H2O
        self.plot_spectrum_comparison_simple(ax1, extent=[870, 1000], title='Zoom $H_2 O$', size=0.8)
        fig.tight_layout()
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            if parameters.DISPLAY and parameters.VERBOSE:
                plt.show()
            if parameters.PdfPages:
                parameters.PdfPages.savefig()
            if parameters.SAVE:
                figname = self.filename.replace('.fits', '_bestfit.pdf')
                self.my_logger.info(f'Save figure {figname}.')
                fig.savefig(figname, dpi=100, bbox_inches='tight', transparent=True)

    def decontaminate_order2(self):  # pragma: no cover
        lambdas = self.spectrum.lambdas
        lambdas_order2 = self.simulation.lambdas_order2
        A1, A2, aerosols, ozone, pwv, reso, D, shift, B = self.params.values
        lambdas_binwidths_order2 = np.gradient(lambdas_order2)
        lambdas_binwidths = np.gradient(lambdas)
        sim_conv = interp1d(lambdas, self.model * lambdas, kind="linear", bounds_error=False, fill_value=(0, 0))
        err_conv = interp1d(lambdas, self.model_err * lambdas, kind="linear", bounds_error=False, fill_value=(0, 0))
        spectrum_order2 = sim_conv(lambdas_order2) * lambdas_binwidths_order2 / lambdas_binwidths
        err_order2 = err_conv(lambdas_order2) * lambdas_binwidths_order2 / lambdas_binwidths
        self.model = (sim_conv(lambdas) - A2 * spectrum_order2) / lambdas
        self.model_err = (err_conv(lambdas) - A2 * err_order2) / lambdas


def lnprob_spectrum(p):  # pragma: no cover
    """Logarithmic likelihood function to maximize in MCMC exploration.

    Parameters
    ----------
    p: array_like
        Array of SpectrumFitWorkspace parameters.

    Returns
    -------
    lp: float
        Log of the likelihood function.

    """
    global w
    lp = w.lnprior(p)
    if not np.isfinite(lp):
        return -1e20
    return lp + w.lnlike(p)


def run_spectrum_minimisation(fit_workspace, method="newton"):
    """Interface function to fit spectrum simulation parameters to data.

    Parameters
    ----------
    fit_workspace: SpectrumFitWorkspace
        An instance of the SpectrogramFitWorkspace class.
    method: str, optional
        Fitting method (default: 'newton').

    Examples
    --------

    >>> spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
    >>> atmgrid_filename = spec.filename.replace('spectrum', 'atmsim')
    >>> w = SpectrumFitWorkspace(spec, atmgrid_file_name=atmgrid_filename, verbose=True, plot=True, live_fit=False)
    >>> parameters.VERBOSE = True
    >>> run_spectrum_minimisation(w, method="newton")

    """
    my_logger = set_logger(__name__)
    guess = np.asarray(fit_workspace.params.values)
    if method != "newton":
        run_minimisation(fit_workspace, method=method)
    else:
        # fit_workspace.simulation.fast_sim = True
        fit_workspace.chisq(guess)
        if parameters.DISPLAY and (parameters.DEBUG or fit_workspace.live_fit):
            fit_workspace.plot_fit()

        # params_table = np.array([guess])
        my_logger.info(f"\n\tStart guess: {guess}\n\twith {fit_workspace.params.labels}")
        epsilon = 1e-4 * guess
        epsilon[epsilon == 0] = 1e-4
        epsilon[-1] = 0.001 * np.max(fit_workspace.data)

        # fit_workspace.simulation.fast_sim = True
        # fit_workspace.simulation.fix_psf_cube = False
        # run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=epsilon, fix=fit_workspace.fixed,
        #                                 xtol=1e-4, ftol=1 / fit_workspace.data.size, sigma_clip=10, niter_clip=3,
        #                                 verbose=False)

        fit_workspace.simulation.fast_sim = False
        # fit_workspace.fixed[0] = True
        fixed = copy.copy(fit_workspace.params.fixed)
        fit_workspace.params.fixed = [True] * len(fit_workspace.params.values)
        fit_workspace.params.fixed[0] = False
        run_minimisation(fit_workspace, method="newton", epsilon=epsilon, xtol=1e-3, ftol=100 / fit_workspace.data.size,
                         verbose=False)
        # fit_workspace.fixed[0] = False
        fit_workspace.params.fixed = fixed
        run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=epsilon, xtol=1e-6,
                                        ftol=1 / fit_workspace.data.size, sigma_clip=20, niter_clip=3, verbose=False)

        fit_workspace.params.plot_correlation_matrix()
        fit_workspace.plot_fit()
        if fit_workspace.filename != "":
            write_fitparameter_json(fit_workspace.params.json_filename, fit_workspace.params,
                                    extra={"chi2": fit_workspace.costs[-1] / fit_workspace.data.size,
                                           "date-obs": fit_workspace.spectrum.date_obs})
            # save_gradient_descent(fit_workspace, costs, params_table)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

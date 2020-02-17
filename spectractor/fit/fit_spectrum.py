import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import interp1d

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.simulation.simulator import SimulatorInit, SpectrumSimulation
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.fit.fitter import FitWorkspace, run_minimisation_sigma_clipping
from spectractor.tools import plot_spectrum_simple


class SpectrumFitWorkspace(FitWorkspace):

    def __init__(self, file_name, atmgrid_file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        """Class to fit a spectrum extracted with Spectractor.

        The spectrum is supposed to be the product of the star SED, the instrument throughput and the atmospheric
        transmission, contaminated eventually by a second order diffraction.
        The truth parameters are loaded from the file header if provided.
        If provided, the atmospheric grid is used for the atmospheric transmission simulations and interpolated
        with splines, otherwise Libradtran is called at each step (slower).

        Parameters
        ----------
        file_name: str
            Spectrum file name.
        atmgrid_file_name: str, optional
            Atmospheric grid file name (default: "").
        nwalkers: int, optional
            Number of walkers for MCMC fitting.
        nsteps: int, optional
            Number of steps for MCMC fitting.
        burnin: int, optional
            Number of burn-in steps for MCMC fitting.
        nbins: int, optional
            Number of bins for MCMC chains analysis.
        verbose: int, optional
            Verbosity level (default: 0).
        plot: bool, optional
            If True, many plots are produced (default: False).
        live_fit: bool, optional
            If True, many plots along the fitting procedure are produced to see convergence in live (default: False).
        truth: array_like, optional
            Array of truth parameters to compare with the best fit result (default: None).

        Examples
        --------

        >>> filename = 'tests/data/reduc_20170530_134_spectrum.fits'
        >>> atmgrid_filename = filename.replace('spectrum', 'atmsim')
        >>> load_config("config/ctio.ini")
        >>> w = SpectrumFitWorkspace(filename, atmgrid_file_name=atmgrid_filename, nsteps=1000,
        ... burnin=2, nbins=10, verbose=1, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.p)
        >>> w.plot_fit()

        """
        FitWorkspace.__init__(self, file_name, nwalkers, nsteps, burnin, nbins, verbose, plot, live_fit, truth=truth)
        if "spectrum" not in file_name:
            raise ValueError("file_name argument must contain spectrum keyword and be an output from Spectractor.")
        self.my_logger = set_logger(self.__class__.__name__)
        self.spectrum, self.telescope, self.disperser, self.target = SimulatorInit(file_name)
        self.airmass = self.spectrum.header['AIRMASS']
        self.pressure = self.spectrum.header['OUTPRESS']
        self.temperature = self.spectrum.header['OUTTEMP']
        if atmgrid_file_name == "":
            self.atmosphere = Atmosphere(self.airmass, self.pressure, self.temperature)
        else:
            self.use_grid = True
            self.atmosphere = AtmosphereGrid(file_name, atmgrid_file_name)
            if parameters.VERBOSE:
                self.my_logger.info(f'\n\tUse atmospheric grid models from file {atmgrid_file_name}. ')
        self.lambdas = self.spectrum.lambdas
        self.data = self.spectrum.data
        self.err = self.spectrum.err
        self.data_cov = self.spectrum.cov_matrix
        self.A1 = 1.0
        self.A2 = 0
        self.ozone = 400.
        self.pwv = 3
        self.aerosols = 0.05
        self.reso = -1
        self.D = self.spectrum.header['D2CCD']
        self.shift_x = self.spectrum.header['PIXSHIFT']
        self.B = 0
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols, self.reso, self.D,
                           self.shift_x, self.B])
        self.fixed = [False] * self.p.size
        # self.fixed[0] = True
        self.fixed[1] = "A2_T" not in self.spectrum.header  # fit A2 only on sims to evaluate extraction biases
        self.fixed[5] = True
        # self.fixed[6:8] = [True, True]
        self.fixed[7] = True
        self.fixed[8] = True
        # self.fixed[-1] = True
        self.input_labels = ["A1", "A2", "ozone", "PWV", "VAOD", "reso [pix]", r"D_CCD [mm]",
                             r"alpha_pix [pix]", "B"]
        self.axis_names = ["$A_1$", "$A_2$", "ozone", "PWV", "VAOD", "reso [pix]", r"$D_{CCD}$ [mm]",
                           r"$\alpha_{\mathrm{pix}}$ [pix]", "$B$"]
        bounds_D = (self.D - 5 * parameters.DISTANCE2CCD_ERR, self.D + 5 * parameters.DISTANCE2CCD_ERR)
        self.bounds = [(0, 2), (0, 2/parameters.GRATING_ORDER_2OVER1), (100, 700), (0, 10), (0, 0.1),
                       (0.1, 10), bounds_D, (-2, 2), (-np.inf, np.inf)]
        if atmgrid_file_name != "":
            self.bounds[2] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.bounds[3] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.bounds[4] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.simulation = SpectrumSimulation(self.spectrum, self.atmosphere, self.telescope, self.disperser)
        self.amplitude_truth = None
        self.lambdas_truth = None
        self.output_file_name = file_name.replace('_spectrum', '_spectrum_A2=0')
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
            self.truth = (A1_truth, A2_truth, ozone_truth, pwv_truth, aerosols_truth,
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
        residuals = (self.spectrum.data[sub][idx] - self.model[sub][idx]) / self.err[sub][idx]
        residuals_err = self.spectrum.err[sub][idx] / self.err[sub][idx]
        ax2.errorbar(lambdas[sub][idx], residuals, yerr=residuals_err, fmt='ro', markersize=2, label='(Data-Model)/Err')
        ax2.axhline(0, color=p0[0].get_color())
        ax2.grid(True)
        ylim = ax2.get_ylim()
        residuals_model = self.model_err[sub][idx] / self.err[sub][idx]
        ax2.fill_between(lambdas[sub][idx], -residuals_model, residuals_model, alpha=0.3, color=p0[0].get_color())
        # std = np.std(residuals)
        ax2.set_ylim(ylim)
        ax2.set_xlabel(ax.get_xlabel())
        # ax2.set_ylabel('(Data-Model)/Err', fontsize=10)
        ax2.legend()
        ax2.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_ylim((0.9 * np.min(self.spectrum.data[sub]), 1.1 * np.max(self.spectrum.data[sub])))
        ax.set_xticks(ax2.get_xticks()[1:-1])
        ax.get_yaxis().set_label_coords(-0.08, 0.6)
        # ax2.get_yaxis().set_label_coords(-0.11, 0.5)

    def simulate(self, A1, A2, ozone, pwv, aerosols, reso, D, shift_x, B):
        """Interface method to simulate a spectrogram.

        Parameters
        ----------
        A1: float
            Main amplitude parameter.
        A2: float
            Relative amplitude of the order 2 spectrogram.
        ozone: float
            Ozone parameter for Libradtran (in db).
        pwv: float
            Precipitable Water Vapor quantity for Libradtran (in mm).
        aerosols: float
            Vertical Aerosols Optical Depth quantity for Libradtran (no units).
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

        >>> filename = 'tests/data/sim_20170530_134_spectrum.fits'
        >>> atmgrid_filename = filename.replace('spectrum', 'atmsim')
        >>> load_config("config/ctio.ini")
        >>> w = SpectrumFitWorkspace(filename, atmgrid_filename, verbose=1, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.p)
        >>> w.plot_fit()

        """
        lambdas, model, model_err = self.simulation.simulate(A1, A2, ozone, pwv, aerosols, reso, D, shift_x, B)
        self.model = model
        self.model_err = model_err
        return lambdas, model, model_err

    def plot_fit(self):
        """Plot the fit result.

        Examples
        --------

        >>> filename = 'tests/data/reduc_20170530_134_spectrum.fits'
        >>> atmgrid_filename = filename.replace('spectrum', 'atmsim')
        >>> load_config("config/ctio.ini")
        >>> w = SpectrumFitWorkspace(filename, atmgrid_filename, verbose=1, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.p)
        >>> w.plot_fit()

        .. plot::
            :include-source:

            from spectractor.fit.fit_spectrum import SpectrumFitWorkspace
            file_name = 'tests/data/reduc_20170530_134_spectrum.fits'
            atmgrid_file_name = file_name.replace('spectrum', 'atmsim')
            fit_workspace = SpectrumFitWorkspace(file_name, atmgrid_file_name=atmgrid_file_name, verbose=True)
            A1, A2, ozone, pwv, aerosols, reso, D, shift_x = fit_workspace.p
            lambdas, model, model_err = fit_workspace.simulation.simulate(A1,A2,ozone, pwv, aerosols, reso, D, shift_x)
            fit_workspace.lambdas = lambdas
            fit_workspace.model = model
            fit_workspace.model_err = model_err
            fit_workspace.plot_fit()

        """
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(222)
        ax2 = plt.subplot(224)
        ax3 = plt.subplot(121)
        A1, A2, ozone, pwv, aerosols, reso, D, shift, B = self.p
        self.title = f'A1={A1:.3f}, A2={A2:.3f}, PWV={pwv:.3f}, OZ={ozone:.3g}, VAOD={aerosols:.3f},\n ' \
                     f'reso={reso:.2f}pix, D={D:.2f}mm, shift={shift:.2f}pix, B={B:.2g}'
        # main plot
        self.plot_spectrum_comparison_simple(ax3, title=self.title, size=0.8)
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
                fig.savefig(figname, dpi=100, bbox_inches='tight')

    def decontaminate_order2(self):  # pragma: no cover
        lambdas = self.spectrum.lambdas
        lambdas_order2 = self.simulation.lambdas_order2
        A1, A2, ozone, pwv, aerosols, reso, D, shift, B = self.p
        lambdas_binwidths_order2 = np.gradient(lambdas_order2)
        lambdas_binwidths = np.gradient(lambdas)
        sim_conv = interp1d(lambdas, self.model * lambdas, kind="linear", bounds_error=False, fill_value=(0, 0))
        err_conv = interp1d(lambdas, self.model_err * lambdas, kind="linear", bounds_error=False, fill_value=(0, 0))
        spectrum_order2 = sim_conv(lambdas_order2) * lambdas_binwidths_order2 / lambdas_binwidths
        err_order2 = err_conv(lambdas_order2) * lambdas_binwidths_order2 / lambdas_binwidths
        self.model = (sim_conv(lambdas) - A2 * spectrum_order2) / lambdas
        self.model_err = (err_conv(lambdas) - A2 * err_order2) / lambdas

    def get_truth_without_order2(self):  # pragma: no cover
        lambdas, model, model_err = self.simulation.simulate(1., 0., self.ozone, self.pwv, self.aerosols, self.reso,
                                                             self.D, self.shift_x)
        self.lambdas_truth = lambdas
        self.amplitude_truth = model


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

    >>> filename = 'tests/data/sim_20170530_134_spectrum.fits'
    >>> atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')
    >>> load_config("config/ctio.ini")
    >>> w = SpectrumFitWorkspace(filename, atmgrid_file_name=atmgrid_filename, verbose=1, plot=True, live_fit=False)
    >>> parameters.VERBOSE = True
    >>> run_spectrum_minimisation(w, method="newton")

    """
    my_logger = set_logger(__name__)
    guess = np.asarray(fit_workspace.p)
    if method != "newton":
        run_minimisation(fit_workspace, method=method)
    else:
        # fit_workspace.simulation.fast_sim = True
        # costs = np.array([fit_workspace.chisq(guess)])
        # if parameters.DISPLAY and (parameters.DEBUG or fit_workspace.live_fit):
        #     fit_workspace.plot_fit()
        # params_table = np.array([guess])
        my_logger.info(f"\n\tStart guess: {guess}\n\twith {fit_workspace.input_labels}")
        epsilon = 1e-4 * guess
        epsilon[epsilon == 0] = 1e-4
        epsilon[-1] = 0.001 * np.max(fit_workspace.data)

        # fit_workspace.simulation.fast_sim = True
        # fit_workspace.simulation.fix_psf_cube = False
        # run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=epsilon, fix=fit_workspace.fixed,
        #                                 xtol=1e-4, ftol=1 / fit_workspace.data.size, sigma_clip=10, niter_clip=3,
        #                                 verbose=False)

        fit_workspace.simulation.fast_sim = False
        fit_workspace.fixed[0] = True
        run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=epsilon, fix=fit_workspace.fixed,
                                        xtol=1e-6, ftol=1 / fit_workspace.data.size, sigma_clip=20, niter_clip=3,
                                        verbose=False)
        fit_workspace.fixed[0] = False
        run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=epsilon, fix=fit_workspace.fixed,
                                        xtol=1e-6, ftol=1 / fit_workspace.data.size, sigma_clip=20, niter_clip=3,
                                        verbose=False)

        parameters.SAVE = True
        ipar = np.array(np.where(np.array(fit_workspace.fixed).astype(int) == 0)[0])
        fit_workspace.plot_correlation_matrix(ipar)
        fit_workspace.plot_fit()
        if fit_workspace.filename != "":
            header = f"{fit_workspace.spectrum.date_obs}\nchi2: {fit_workspace.costs[-1] / fit_workspace.data.size}"
            fit_workspace.save_parameters_summary(ipar, header=header)
            # save_gradient_descent(fit_workspace, costs, params_table)
        parameters.SAVE = False


if __name__ == "__main__":
    from argparse import ArgumentParser
    from spectractor.config import load_config
    from spectractor.fit.fitter import run_minimisation

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path', default=["tests/data/reduc_20170530_134_spectrum.fits"],
                        help="Input fits file name. It can be a list separated by spaces, or it can use * as wildcard.",
                        nargs='*')
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=True)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                        help="Write results in given output directory (default: ./outputs/).")
    parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    parser.add_argument("-c", "--config", dest="config", default="config/ctio.ini",
                        help="INI config file. (default: config.ctio.ini).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    load_config(args.config)

    filenames = ['outputs/sim_20170530_134_spectrum.fits']
    filenames = [
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_104_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_109_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_114_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_119_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_124_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_129_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_134_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_139_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_144_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_149_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_154_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_159_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_164_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_169_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_174_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_179_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_184_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_189_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_194_spectrum.fits',
                 'outputs/data_30may17_HoloAmAg_prod6.9/sim_20170530_199_spectrum.fits']
    params = []
    chisqs = []
    filenames = ['outputs/reduc_20170530_176_spectrum.fits']
    # filenames = ['../test_176_Moffat4/reduc_20170530_176_spectrum.fits']
    # filenames = ['../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_176_spectrum.fits']
    # filenames = ['../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_061_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_066_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_071_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_076_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_081_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_086_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_091_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_096_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_101_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_106_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_111_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_116_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_121_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_126_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_131_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_136_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_141_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_146_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_151_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_156_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_161_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_166_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_171_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_176_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_181_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_186_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_191_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.2/data_30may17_A2=0.1/reduc_20170530_196_spectrum.fits']


    for filename in filenames:
        atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')

        w = SpectrumFitWorkspace(filename, atmgrid_file_name=atmgrid_filename, nsteps=1000,
                                 burnin=200, nbins=10, verbose=1, plot=True, live_fit=False)
        run_spectrum_minimisation(w, method="newton")
        params.append(w.p)
        chisqs.append(w.costs[-1])
    params = np.asarray(params).T

    # fig, ax = plt.subplots(1, len(params))
    # for ip, p in enumerate(params):
    #     print(f"{w.input_labels[ip]}:", np.mean(p), np.std(p))
    #     ax[ip].plot(p, label=f"{w.input_labels[ip]}")
    #     ax[ip].grid()
    #     ax[ip].legend()
    # plt.show()

    # w.decontaminate_order2()
    # fit_workspace.simulate(*fit_workspace.p)
    # fit_workspace.plot_fit()
    # run_emcee(fit_workspace, ln=lnprob_spectrum)
    # fit_workspace.analyze_chains()

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.simulation.simulator import SimulatorInit, SpectrumSimulation
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.fit.fitter import FitWorkspace, run_gradient_descent, save_gradient_descent
from spectractor.tools import plot_spectrum_simple


class SpectrumFitWorkspace(FitWorkspace):

    def __init__(self, file_name, atmgrid_file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None, fast_sim=True):
        FitWorkspace.__init__(self, file_name, nwalkers, nsteps, burnin, nbins, verbose, plot, live_fit, truth=truth)
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
        self.A1 = 1.0
        self.A2 = 0.05
        self.ozone = 300.
        self.pwv = 5
        self.aerosols = 0.03
        self.reso = -1
        self.D = self.spectrum.header['D2CCD']
        self.shift_x = self.spectrum.header['PIXSHIFT']
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols, self.reso, self.D, self.shift_x])
        self.fixed = [False] * self.p.size
        self.fixed[5] = True
        self.input_labels = ["A1", "A2", "ozone", "PWV", "VAOD", "reso [pix]", r"D_CCD [mm]",
                             r"alpha_pix [pix]"]
        self.axis_names = ["$A_1$", "$A_2$", "ozone", "PWV", "VAOD", "reso [pix]", r"$D_{CCD}$ [mm]",
                           r"$\alpha_{\mathrm{pix}}$ [pix]"]
        self.bounds = [(0, 2), (0, 0.5), (300, 700), (0, 10), (0, 0.01), (-2, 2), (50, 60), (-0.1, 0.1)]
        if atmgrid_file_name != "":
            self.bounds[2] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.bounds[3] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.bounds[4] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.simulation = SpectrumSimulation(self.spectrum, self.atmosphere, self.telescope, self.disperser,
                                             fast_sim=fast_sim)
        self.amplitude_truth = None
        self.lambdas_truth = None
        self.get_truth()

    def get_truth(self):
        if 'A1_T' in list(self.spectrum.header.keys()):
            A1_truth = self.spectrum.header['A1_T']
            A2_truth = self.spectrum.header['A2_T']
            ozone_truth = self.spectrum.header['OZONE_T']
            pwv_truth = self.spectrum.header['PWV_T']
            aerosols_truth = self.spectrum.header['VAOD_T']
            reso_truth = -1
            D_truth = self.spectrum.header['D2CCD_T']
            shift_truth = self.spectrum.header['X0_T']
            self.truth = (A1_truth, A2_truth, ozone_truth, pwv_truth, aerosols_truth,
                          reso_truth, D_truth, shift_truth)
            self.lambdas_truth = np.fromstring(self.spectrum.header['LBDAS_T'][1:-1], sep=' ', dtype=float)
            self.amplitude_truth = np.fromstring(self.spectrum.header['AMPLIS_T'][1:-1], sep=' ', dtype=float)
        else:
            self.truth = None

    def plot_spectrum_comparison_simple(self, ax, title='', extent=None, size=0.4):
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
        ax2 = divider.append_axes("bottom", size=size, pad=0)
        ax.figure.add_axes(ax2)
        min_positive = np.min(self.model[self.model > 0])
        idx = np.logical_not(np.isclose(self.model[sub], 0, atol=0.01 * min_positive))
        residuals = (self.spectrum.data[sub][idx] - self.model[sub][idx]) / self.model[sub][idx]
        residuals_err = self.spectrum.err[sub][idx] / self.model[sub][idx]
        ax2.errorbar(lambdas[sub][idx], residuals, yerr=residuals_err, fmt='ro', markersize=2)
        ax2.axhline(0, color=p0[0].get_color())
        ax2.grid(True)
        residuals_model = self.model_err[sub][idx] / self.model[sub][idx]
        ax2.fill_between(lambdas[sub][idx], -residuals_model, residuals_model, alpha=0.3, color=p0[0].get_color())
        std = np.std(residuals)
        ax2.set_ylim([-2. * std, 2. * std])
        ax2.set_xlabel(ax.get_xlabel())
        ax2.set_ylabel('(data-fit)/fit')
        ax2.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_ylim((0.9 * np.min(self.spectrum.data[sub]), 1.1 * np.max(self.spectrum.data[sub])))
        ax.set_xticks(ax2.get_xticks()[1:-1])
        ax.get_yaxis().set_label_coords(-0.15, 0.6)
        ax2.get_yaxis().set_label_coords(-0.15, 0.5)

    def simulate(self, A1, A2, ozone, pwv, aerosols, reso, D, shift_x):
        lambdas, model, model_err = self.simulation.simulate(A1, A2, ozone, pwv, aerosols, reso, D, shift_x)
        # if self.live_fit:
        #    self.plot_fit()
        self.model = model
        self.model_err = model_err
        return lambdas, model, model_err

    def weighted_residuals(self, p):
        x, model, model_err = self.simulate(*p)
        if len(self.outliers) > 0:
            raise NotImplementedError("Weighted residuals function not implemented for outlier rejection.")
        else:
            cov = self.spectrum.cov_matrix[:-1,:-1] + np.diag(model_err * model_err)
            try:
                L = np.linalg.inv(np.linalg.cholesky(cov))
                inv_cov = L.T @ L
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.inv(cov)
            res = inv_cov @ (model - self.data)
        return res

    def plot_fit(self):
        """
        Examples
        --------

        .. plot::
            :include-source:

            >>> from spectractor.fit.fit_spectrum import SpectrumFitWorkspace
            >>> parameters.VERBOSE = True
            >>> file_name = 'tests/data/sim_20170530_134_spectrum.fits'
            >>> atmgrid_file_name = file_name.replace('spectrum', 'atmsim')
            >>> fit_workspace = SpectrumFitWorkspace(file_name, atmgrid_file_name=atmgrid_file_name, verbose=True)
            >>> A1, A2, ozone, pwv, aerosols, reso, D, shift_x = fit_workspace.p
            >>> lambdas, model, model_err = fit_workspace.simulation.simulate(A1, A2,
            ... ozone, pwv, aerosols, reso, D, shift_x)
            >>> fit_workspace.lambdas = lambdas
            >>> fit_workspace.model = model
            >>> fit_workspace.model_err = model_err
            >>> fit_workspace.plot_fit()

        """
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(222)
        ax2 = plt.subplot(224)
        ax3 = plt.subplot(121)
        A1, A2, ozone, pwv, aerosols, reso, D, shift = self.p
        self.title = f'A1={A1:.3f}, A2={A2:.3f}, PWV={pwv:.3f}, OZ={ozone:.3g}, VAOD={aerosols:.3f},\n ' \
                     f'reso={reso:.2f}pix, D={D:.2f}mm, shift={shift:.2f}pix '
        # main plot
        self.plot_spectrum_comparison_simple(ax3, title=self.title, size=0.8)
        # zoom O2
        self.plot_spectrum_comparison_simple(ax2, extent=[730, 800], title='Zoom $O_2$', size=0.8)
        # zoom H2O
        self.plot_spectrum_comparison_simple(ax1, extent=[870, 1000], title='Zoom $H_2 O$', size=0.8)
        fig.tight_layout()
        if self.live_fit:
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            if parameters.DISPLAY and parameters.VERBOSE:
                plt.show()
            if parameters.SAVE:
                figname = self.filename.replace('.fits', '_bestfit.pdf')
                self.my_logger.info(f'Save figure {figname}.')
                fig.savefig(figname, dpi=100, bbox_inches='tight')


def lnprob_spectrum(p):
    global fit_workspace
    lp = fit_workspace.lnprior(p)
    if not np.isfinite(lp):
        return -1e20
    return lp + fit_workspace.lnlike(p)


def run_spectrum_minimisation(fit_workspace, method="newton"):
    my_logger = set_logger(__name__)
    guess = np.asarray(fit_workspace.p)
    if method != "newton":
        run_minimisation(fit_workspace, method=method)
    else:
        fit_workspace.simulation.fast_sim = True
        costs = np.array([fit_workspace.chisq(guess)])
        if parameters.DISPLAY and (parameters.DEBUG or fit_workspace.live_fit):
            fit_workspace.plot_fit()
        params_table = np.array([guess])
        my_logger.info(f"\n\tStart guess: {guess}\n\twith {fit_workspace.input_labels}")
        epsilon = 1e-4 * guess
        epsilon[epsilon == 0] = 1e-4

        fit_workspace.simulation.fast_sim = True
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = run_gradient_descent(fit_workspace, guess, epsilon, params_table, costs,
                                                   fix=fit_workspace.fixed, xtol=1e-4, ftol=1e-4, niter=40)
        if fit_workspace.filename != "":
            parameters.SAVE = True
            ipar = np.array(np.where(np.array(fit_workspace.fixed).astype(int) == 0)[0])
            fit_workspace.plot_correlation_matrix(ipar)
            fit_workspace.save_parameters_summary(header=fit_workspace.spectrum.date_obs)
            save_gradient_descent(fit_workspace, costs, params_table)
            fit_workspace.plot_fit()
            parameters.SAVE = False


if __name__ == "__main__":
    from argparse import ArgumentParser
    from spectractor.config import load_config
    from spectractor.fit.fitter import run_minimisation, run_emcee

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

    # filename = 'outputs/reduc_20170530_130_spectrum.fits'
    filename = 'outputs/sim_20170530_134_spectrum.fits'
    # 062
    # filename = './outputs/reduc_20170530_134_spectrum.fits'
    atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')

    fit_workspace = SpectrumFitWorkspace(filename, atmgrid_file_name=atmgrid_filename, nsteps=1000,
                                         burnin=200, nbins=10, verbose=1, plot=True, live_fit=False, fast_sim=False)
    # run_spectrum_minimisation(fit_workspace, method="newton")
    fit_workspace.simulate(*fit_workspace.p)
    fit_workspace.plot_fit()
    # run_emcee(fit_workspace, ln=lnprob_spectrum)
    # fit_workspace.analyze_chains()

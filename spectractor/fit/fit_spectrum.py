import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import time
from spectractor import parameters
from spectractor.config import set_logger
from spectractor.simulation.simulator import SimulatorInit, SpectrumModel
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.fit.fitter import FitWorkspace, run_minimisation, run_gradient_descent, save_gradient_descent
from spectractor.tools import plot_spectrum_simple

plot_counter = 0


class SpectrumFitWorkspace(FitWorkspace):

    def __init__(self, file_name, atmgrid_file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        FitWorkspace.__init__(self, file_name, nwalkers, nsteps, burnin, nbins, verbose, plot, live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        self.spectrum, self.telescope, self.disperser, self.target = SimulatorInit(file_name)
        plt.plot(parameters.LAMBDA_TRUTH, parameters.AMPLITUDE_TRUTH)
        plt.plot(self.spectrum.lambdas, self.spectrum.data)
        plt.show()
        self.spectrum.data = parameters.AMPLITUDE_TRUTH
        self.spectrum.lambdas = parameters.LAMBDA_TRUTH
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
        self.D = self.spectrum.header['D2CCD']
        self.shift = self.spectrum.header['PIXSHIFT']
        self.reso = 1.5
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols, self.D, self.shift, self.reso])
        self.ndim = len(self.p)
        self.input_labels = ["A1", "A2", "ozone", "PWV", "VAOD", r"D_CCD [mm]",
                             r"alpha_pix [pix]", "reso [pix]"]
        self.axis_names = ["$A_1$", "$A_2$", "ozone", "PWV", "VAOD", r"$D_{CCD}$ [mm]",
                           r"$\alpha_{\mathrm{pix}}$ [pix]", "reso [pix]"]
        self.bounds = [(0, 2), (0, 0.5), (0, 800), (0, 10), (0, 1), (50, 60), (-2, 2), (1, 10)]
        if atmgrid_file_name != "":
            self.bounds[2] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.bounds[3] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.bounds[4] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.simulation = SpectrumModel(self.spectrum, self.atmosphere, self.telescope, self.disperser)
        # self.get_truth()

    def get_truth(self):
        if 'A1' in list(self.spectrum.header.keys()):
            A1_truth = self.spectrum.header['A1']
            A2_truth = self.spectrum.header['A2']
            ozone_truth = self.spectrum.header['OZONE']
            pwv_truth = self.spectrum.header['PWV']
            aerosols_truth = self.spectrum.header['VAOD']
            reso_truth = self.spectrum.header['RESO']
            D_truth = self.spectrum.header['D2CCD']
            shift_truth = self.spectrum.header['X0SHIFT']
            self.truth = (A1_truth, A2_truth, ozone_truth, pwv_truth, aerosols_truth,
                          D_truth, shift_truth, reso_truth)
        else:
            self.truth = None

    def simulate(self, A1, A2, ozone, pwv, aerosols, D, shift_x, reso):
        global plot_counter
        lambdas, model, model_err = \
            self.simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x, reso)
        self.p = np.array([A1, A2, ozone, pwv, aerosols, D, shift_x, reso])
        if lambdas.size > self.data.size:
            lambdas = lambdas[lambdas.size-self.data.size:]
        self.lambdas = lambdas
        self.model = model(lambdas)
        self.model_err = model_err(lambdas)
        if self.live_fit and (plot_counter % 30) == 0:
            self.plot_fit()
        plot_counter += 1
        return lambdas, self.model, self.model_err

    # def jacobian(self, params, epsilon, fixed_params=None):
    #     start = time.time()
    #     lambdas, model, model_err = self.simulate(*params)
    #     model = model.flatten()
    #     J = np.zeros((params.size, model.size))
    #     for ip, p in enumerate(params):
    #         if fixed_params[ip]:
    #             continue
    #         tmp_p = np.copy(params)
    #         if tmp_p[ip] + epsilon[ip] < self.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.bounds[ip][1]:
    #             epsilon[ip] = - epsilon[ip]
    #         tmp_p[ip] += epsilon[ip]
    #         tmp_lambdas, tmp_model, tmp_model_err = self.simulate(*tmp_p)
    #         J[ip] = (tmp_model.flatten() - model) / epsilon[ip]
    #         # print(ip, self.input_labels[ip], p, tmp_p[ip] + epsilon[ip], J[ip])
    #     # if False:
    #     #     plt.imshow(J, origin="lower", aspect="auto")
    #     #     plt.show()
    #     print(f"\tjacobian time computation = {time.time() - start:.1f}s")
    #     return J

    def plot_spectrum_comparison_simple(self, ax, title='', extent=None, size=0.4):
        lambdas = self.spectrum.lambdas
        sub = np.where((lambdas > parameters.LAMBDA_MIN) & (lambdas < parameters.LAMBDA_MAX))
        if extent is not None:
            sub = np.where((lambdas > extent[0]) & (lambdas < extent[1]))
        plot_spectrum_simple(ax, lambdas[sub], self.data[sub], data_err=self.err[sub])
        p0 = ax.plot(lambdas, self.model, label='model')
        ax.fill_between(lambdas[sub], self.model[sub] - self.model_err[sub],
                        self.model[sub] + self.model_err[sub], alpha=0.3, color=p0[0].get_color())
        # ax.plot(self.lambdas, self.model_noconv, label='before conv')
        if title != '':
            ax.set_title(title, fontsize=10)
        ax.legend()
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("bottom", size=size, pad=0)
        ax.figure.add_axes(ax2)
        if not np.all(self.model[sub] == np.zeros_like(self.model[sub])):
            residuals = (self.spectrum.data[sub] - self.model[sub]) / self.model[sub]
            residuals_err = self.spectrum.err[sub] / self.model[sub]
            ax2.errorbar(lambdas[sub], residuals, yerr=residuals_err, fmt='ro', markersize=2)
            residuals_model = self.model_err[sub] / self.model[sub]
            ax2.fill_between(lambdas[sub], -residuals_model, residuals_model, alpha=0.3, color=p0[0].get_color())
            std = np.std(residuals)
            ax2.set_ylim([-2. * std, 2. * std])
        ax2.axhline(0, color=p0[0].get_color())
        ax2.grid(True)
        ax2.set_xlabel(ax.get_xlabel())
        ax2.set_ylabel('(data-fit)/fit')
        ax2.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_ylim((0.9 * np.min(self.spectrum.data[sub]), 1.1 * np.max(self.spectrum.data[sub])))
        ax.set_xticks(ax2.get_xticks()[1:-1])
        ax.get_yaxis().set_label_coords(-0.15, 0.6)
        ax2.get_yaxis().set_label_coords(-0.15, 0.5)

    # def simulate(self, A1, A2, ozone, pwv, aerosols, D, shift, reso):
    #     lambdas, model, model_err = self.simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift, reso)
    #     # if self.live_fit:
    #     #    self.plot_fit()
    #     self.model = model
    #     self.model_err = model_err
    #     return model, model_err

    def plot_fit(self):
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(222)
        ax2 = plt.subplot(224)
        ax3 = plt.subplot(121)
        A1, A2, ozone, pwv, aerosols, D, shift, reso = self.p
        self.title = f'A1={A1:.3f}, A2={A2:.3f}, PWV={pwv:.3f}, OZ={ozone:.3g}, VAOD={aerosols:.3f},\n ' \
                     f'D={D:.2f}mm, shift={shift:.2f}pix, reso={reso:.2f}pix'
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
    return lp + fit_workspace.lnlike_spectrum(p)


def run_spectrum_minimisation(fit_workspace, method="newton"):
    my_logger = set_logger(__name__)
    bounds = fit_workspace.bounds

    nll = lambda params: -fit_workspace.lnlike(params)

    guess = fit_workspace.p
    if method != "newton":
        run_minimisation(fit_workspace, method=method)
    else:
        costs = np.array([fit_workspace.chisq(guess)])
        if parameters.DISPLAY:
            fit_workspace.plot_fit()
        params_table = np.array([guess])
        start = time.time()
        epsilon = 1e-4 * guess
        epsilon[epsilon == 0] = 1e-3
        epsilon[0] = 1e-3  # A1
        epsilon[1] = 1e-4  # A2
        epsilon[2] = 1  # ozone
        epsilon[3] = 0.01  # pwv
        epsilon[4] = 0.001  # aerosols
        epsilon[5] = 0.001  # DCCD
        epsilon[6] = 0.0005  # shift_x
        epsilon[7] = 0.005  # reso
        my_logger.info(f"\n\tStart guess: {guess}")

        # fit all
        guess = np.array(fit_workspace.p)
        fix = [False] * guess.size
        parameters.SAVE = True
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = run_gradient_descent(fit_workspace, guess, epsilon, params_table, costs,
                                                   fix=fix, xtol=1e-5, ftol=1e-5, niter=40)
        if fit_workspace.filename != "":
            fit_workspace.save_parameters_summary(header=fit_workspace.spectrum.date_obs)
            save_gradient_descent(fit_workspace, costs, params_table)
        print(f"Newton: total computation time: {time.time() - start}s")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from spectractor.config import load_config

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path', default=["tests/data/reduc_20170530_134_spectrum.fits"],
                        help="Input fits file name. It can be a list separated by spaces, or it can use * as wildcard.",
                        nargs='*')
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                        help="Write results in given output directory (default: ./outputs/).")
    parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    parser.add_argument("-c", "--config", dest="config", default="config/ctio.ini",
                        help="INI config file. (default: config.ctio.ini).")
    args = parser.parse_args()

    from spectractor.extractor.images import Image
    image = Image("tests/data/sim_20170530_134.fits")
    lambda_truth = np.fromstring(image.header['LAMBDAS'][1:-1], sep=' ', dtype=float)
    amplitude_truth = np.fromstring(image.header['PSF_POLY'][1:-1], sep=' ', dtype=float)[:lambda_truth.size]
    parameters.AMPLITUDE_TRUTH = np.copy(amplitude_truth)
    parameters.LAMBDA_TRUTH = np.copy(lambda_truth)

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    load_config(args.config)

    # filename = 'outputs/reduc_20170530_130_spectrum.fits'
    # filename = 'outputs/sim_20170530_134_spectrum.fits'
    # 062
    filename = "tests/data/sim_20170530_134_spectrum.fits"
    atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')


    w = SpectrumFitWorkspace(filename, atmgrid_file_name=atmgrid_filename, nsteps=1000,
                             burnin=2, nbins=10, verbose=1, plot=True, live_fit=True)
    run_spectrum_minimisation(w, method="minimize")
    # run_emcee(w, ln=lnprob_spectrogram)
    # w.analyze_chains()

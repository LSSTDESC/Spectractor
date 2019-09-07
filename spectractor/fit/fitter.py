from iminuit import Minuit
from scipy import optimize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from schwimmbad import MPIPool
import emcee
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import multiprocessing

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.tools import formatting_numbers, plot_image_simple, from_lambda_to_colormap
from spectractor.simulation.simulator import SimulatorInit, SpectrumSimulation, SpectrogramModel
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.fit.statistics import Likelihood

plot_counter = 0


class FitWorkspace:

    def __init__(self, file_name, nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        self.my_logger = set_logger(self.__class__.__name__)
        self.filename = file_name
        self.ndim = 0
        self.truth = truth
        self.verbose = verbose
        self.plot = plot
        self.live_fit = live_fit
        self.p = np.array([])
        self.cov = np.array([[]])
        self.rho = np.array([[]])
        self.ndim = len(self.p)
        self.data = None
        self.err = None
        self.x = None
        self.model = None
        self.model_err = None
        self.model_noconv = None
        self.input_labels = []
        self.axis_names = []
        self.input_labels = []
        self.axis_names = []
        self.bounds = ((), ())
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.nsteps = nsteps
        self.nbins = nbins
        self.burnin = burnin
        self.start = []
        self.likelihood = np.array([[]])
        self.gelmans = np.array([])
        self.chains = np.array([[]])
        self.lnprobs = np.array([[]])
        self.flat_chains = np.array([[]])
        self.valid_chains = [False] * self.nwalkers
        self.global_average = None
        self.global_std = None
        self.title = ""
        self.use_grid = False
        if "." in self.filename:
            self.emcee_filename = self.filename.split('.')[0] + "_emcee.h5"
        else:
            self.my_logger.warning("\n\tFile name must have an extension.")

    def set_start(self):
        self.start = np.array(
            [np.random.uniform(self.p[i] - 0.02 * self.p[i], self.p[i] + 0.02 * self.p[i], self.nwalkers)
             for i in range(self.ndim)]).T
        self.start[self.start == 0] = 1e-5 * np.random.uniform(0, 1)
        return self.start

    def load_chains(self):
        self.chains = [[]]
        self.lnprobs = [[]]
        self.nsteps = 0
        # tau = -1
        reader = emcee.backends.HDFBackend(self.emcee_filename)
        try:
            tau = reader.get_autocorr_time()
        except emcee.autocorr.AutocorrError:
            tau = -1
        self.chains = reader.get_chain(discard=0, flat=False, thin=1)
        self.lnprobs = reader.get_log_prob(discard=0, flat=False, thin=1)
        self.nsteps = self.chains.shape[0]
        self.nwalkers = self.chains.shape[1]
        print(f"Auto-correlation time: {tau}")
        print(f"Burn-in: {self.burnin}")
        print(f"Chains shape: {self.chains.shape}")
        print(f"Log prob shape: {self.lnprobs.shape}")
        return self.chains, self.lnprobs

    def build_flat_chains(self):
        self.flat_chains = self.chains[self.burnin:, self.valid_chains, :].reshape((-1, self.ndim))
        return self.flat_chains

    def simulate(self, *p):
        x = np.array([])
        self.model = np.array([])
        self.model_err = np.array([])
        return x, self.model, self.model_err

    def analyze_chains(self):
        self.load_chains()
        self.set_chain_validity()
        self.convergence_tests()
        self.build_flat_chains()
        self.likelihood = self.chain2likelihood()
        self.cov = self.likelihood.cov_matrix
        self.rho = self.likelihood.rho_matrix
        self.p = self.likelihood.mean_vec
        self.simulate(*self.p)
        self.plot_fit()
        figure_name = self.emcee_filename.replace('.h5', '_triangle.pdf')
        self.likelihood.triangle_plots(output_filename=figure_name)

    def plot_fit(self):
        plt.errorbar(self.x, self.data, yerr=self.err, fmt='ko', label='Data')
        plt.plot(self.x, self.model, label='Best fitting model')
        if self.truth is not None:
            x, truth, truth_err = self.simulate(*self.truth)
            plt.plot(self.x, truth, label="Truth")
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        title = ""
        for i, label in enumerate(self.input_labels):
            title += f"{label} = {self.p[i]:.3g}"
            if self.cov.size > 0:
                title += rf" $\pm$ {np.sqrt(self.cov[i, i]):.3g}"
            if i < len(self.input_labels)-1:
                title += ", "
        plt.title(title)
        plt.legend()
        plt.grid()
        if parameters.DISPLAY:
            plt.show()

    def chain2likelihood(self, pdfonly=False, walker_index=-1):
        if walker_index >= 0:
            chains = self.chains[self.burnin:, walker_index, :]
        else:
            chains = self.flat_chains
        rangedim = range(chains.shape[1])
        centers = []
        for i in rangedim:
            centers.append(np.linspace(np.min(chains[:, i]), np.max(chains[:, i]), self.nbins - 1))
        likelihood = Likelihood(centers, labels=self.input_labels, axis_names=self.axis_names, truth=self.truth)
        if walker_index < 0:
            for i in rangedim:
                likelihood.pdfs[i].fill_histogram(chains[:, i], weights=None)
                if not pdfonly:
                    for j in rangedim:
                        if i != j:
                            likelihood.contours[i][j].fill_histogram(chains[:, i], chains[:, j], weights=None)
            output_file = self.filename.replace('.fits', '_bestfit.txt')
            likelihood.stats(output=output_file)
        else:
            for i in rangedim:
                likelihood.pdfs[i].fill_histogram(chains[:, i], weights=None)
        return likelihood

    def compute_local_acceptance_rate(self, start_index, last_index, walker_index):
        frequences = []
        test = -2 * self.lnprobs[start_index, walker_index]
        counts = 1
        for index in range(start_index + 1, last_index):
            chi2 = -2 * self.lnprobs[index, walker_index]
            if np.isclose(chi2, test):
                counts += 1
            else:
                frequences.append(float(counts))
                counts = 1
                test = chi2
        frequences.append(counts)
        return 1.0 / np.mean(frequences)

    def set_chain_validity(self):
        nchains = [k for k in range(self.nwalkers)]
        chisq_averages = []
        chisq_std = []
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            # if np.mean(chisqs) < 1e5:
            chisq_averages.append(np.mean(chisqs))
            chisq_std.append(np.std(chisqs))
        self.global_average = np.mean(chisq_averages)
        self.global_std = np.mean(chisq_std)
        self.valid_chains = [False] * self.nwalkers
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            chisq_average = np.mean(chisqs)
            chisq_std = np.std(chisqs)
            if 3 * self.global_std + self.global_average < chisq_average < 1e5:
                self.valid_chains[k] = False
            elif chisq_std < 0.1 * self.global_std:
                self.valid_chains[k] = False
            else:
                self.valid_chains[k] = True
        return self.valid_chains

    def convergence_tests(self):
        chains = self.chains[self.burnin:, :, :]  # .reshape((-1, self.ndim))
        nchains = [k for k in range(self.nwalkers)]
        fig, ax = plt.subplots(self.ndim + 1, 2, figsize=(16, 7), sharex='all')
        fontsize = 8
        steps = np.arange(self.burnin, self.nsteps)
        # Chi2 vs Index
        print("Chisq statistics:")
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            text = f"\tWalker {k:d}: {float(np.mean(chisqs)):.3f} +/- {float(np.std(chisqs)):.3f}"
            if not self.valid_chains[k]:
                text += " -> excluded"
                ax[self.ndim, 0].plot(steps, chisqs, c='0.5', linestyle='--')
            else:
                ax[self.ndim, 0].plot(steps, chisqs)
            print(text)
        # global_average = np.mean(-2*self.lnprobs[self.valid_chains, self.burnin:])
        # global_std = np.std(-2*self.lnprobs[self.valid_chains, self.burnin:])
        ax[self.ndim, 0].set_ylim(
            [self.global_average - 5 * self.global_std, self.global_average + 5 * self.global_std])
        # Parameter vs Index
        print("Computing Parameter vs Index plots...")
        for i in range(self.ndim):
            ax[i, 0].set_ylabel(self.axis_names[i], fontsize=fontsize)
            for k in nchains:
                if self.valid_chains[k]:
                    ax[i, 0].plot(steps, chains[:, k, i])
                else:
                    ax[i, 0].plot(steps, chains[:, k, i], c='0.5', linestyle='--')
                ax[i, 0].get_yaxis().set_label_coords(-0.05, 0.5)
        ax[self.ndim, 0].set_ylabel(r'$\chi^2$', fontsize=fontsize)
        ax[self.ndim, 0].set_xlabel('Steps', fontsize=fontsize)
        ax[self.ndim, 0].get_yaxis().set_label_coords(-0.05, 0.5)
        # Acceptance rate vs Index
        print("Computing acceptance rate...")
        min_len = self.nsteps
        window = 100
        if min_len > window:
            for k in nchains:
                ARs = []
                indices = []
                for l in range(self.burnin + window, self.nsteps, window):
                    ARs.append(self.compute_local_acceptance_rate(l - window, l, k))
                    indices.append(l)
                if self.valid_chains[k]:
                    ax[self.ndim, 1].plot(indices, ARs, label=f'Walker {k:d}')
                else:
                    ax[self.ndim, 1].plot(indices, ARs, label=f'Walker {k:d}', c='gray', linestyle='--')
                ax[self.ndim, 1].set_xlabel('Steps', fontsize=fontsize)
                ax[self.ndim, 1].set_ylabel('Aceptance rate', fontsize=fontsize)
                # ax[self.dim + 1, 2].legend(loc='upper left', ncol=2, fontsize=10)
        # Parameter PDFs by chain
        print("Computing chain by chain PDFs...")
        for k in nchains:
            likelihood = self.chain2likelihood(pdfonly=True, walker_index=k)
            likelihood.stats(pdfonly=True, verbose=False)
            # for i in range(self.dim):
            # ax[i, 1].plot(likelihood.pdfs[i].axe.axis, likelihood.pdfs[i].grid, lw=var.LINEWIDTH,
            #               label=f'Walker {k:d}')
            # ax[i, 1].set_xlabel(self.axis_names[i])
            # ax[i, 1].set_ylabel('PDF')
            # ax[i, 1].legend(loc='upper right', ncol=2, fontsize=10)
        # Gelman-Rubin test.py
        if len(nchains) > 1:
            step = max(1, (self.nsteps - self.burnin) // 20)
            self.gelmans = []
            print(f'Gelman-Rubin tests (burnin={self.burnin:d}, step={step:d}, nsteps={self.nsteps:d}):')
            for i in range(self.ndim):
                Rs = []
                lens = []
                for l in range(self.burnin + step, self.nsteps, step):
                    chain_averages = []
                    chain_variances = []
                    global_average = np.mean(self.chains[self.burnin:l, self.valid_chains, i])
                    for k in nchains:
                        if not self.valid_chains[k]:
                            continue
                        chain_averages.append(np.mean(self.chains[self.burnin:l, k, i]))
                        chain_variances.append(np.var(self.chains[self.burnin:l, k, i], ddof=1))
                    W = np.mean(chain_variances)
                    B = 0
                    for n in range(len(chain_averages)):
                        B += (chain_averages[n] - global_average) ** 2
                    B *= ((l + 1) / (len(chain_averages) - 1))
                    R = (W * l / (l + 1) + B / (l + 1) * (len(chain_averages) + 1) / len(chain_averages)) / W
                    Rs.append(R - 1)
                    lens.append(l)
                print(f'\t{self.input_labels[i]}: R-1 = {Rs[-1]:.3f} (l = {lens[-1] - 1:d})')
                self.gelmans.append(Rs[-1])
                ax[i, 1].plot(lens, Rs, lw=1, label=self.axis_names[i])
                ax[i, 1].axhline(0.03, c='k', linestyle='--')
                ax[i, 1].set_xlabel('Walker length', fontsize=fontsize)
                ax[i, 1].set_ylabel('$R-1$', fontsize=fontsize)
                ax[i, 1].set_ylim(0, 0.6)
                # ax[self.dim, 3].legend(loc='best', ncol=2, fontsize=10)
        self.gelmans = np.array(self.gelmans)
        fig.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.show()
        figure_name = self.emcee_filename.replace('.h5', '_convergence.pdf')
        print(f'Save figure: {figure_name}')
        fig.savefig(figure_name, dpi=100)

    def print_settings(self):
        print('************************************')
        print(f"Input file: {self.filename}\nWalkers: {self.nwalkers}\t Steps: {self.nsteps}")
        print(f"Output file: {self.emcee_filename}")
        print('************************************')

    def save_parameters_summary(self, header=""):
        output_filename = self.filename.replace(self.filename.split('.')[-1], "_bestfit.txt")
        f = open(output_filename, 'w')
        txt = self.filename + "\n"
        if header != "":
            txt += header + "\n"
        for ip in np.arange(0, self.cov.shape[0]).astype(int):
            txt += "%s: %s +%s -%s\n" % formatting_numbers(self.p[ip], np.sqrt(self.cov[ip, ip]),
                                                           np.sqrt(self.cov[ip, ip]),
                                                           label=self.input_labels[ip])
        for row in self.cov:
            txt += np.array_str(row, max_line_width=20 * self.cov.shape[0]) + '\n'
        self.my_logger.info(f"\n\tSave best fit parameters in {output_filename}.")
        f.write(txt)
        f.close()

    def compute_correlation_matrix(self):
        rho = np.zeros_like(self.cov)
        for i in range(self.cov.shape[0]):
            for j in range(self.cov.shape[1]):
                rho[i, j] = self.cov[i, j] / np.sqrt(self.cov[i, i] * self.cov[j, j])
        self.rho = rho
        return rho

    def plot_correlation_matrix(self, ipar=None):
        fig = plt.figure()
        rho = self.compute_correlation_matrix()
        im = plt.imshow(rho, interpolation="nearest", cmap='bwr', vmin=-1, vmax=1)
        if ipar is None:
            ipar = np.arange(0, self.cov.shape[0]).astype(int)
        self.rho = rho
        plt.title("Correlation matrix")
        axis_names = [self.axis_names[ip] for ip in ipar]
        plt.xticks(np.arange(ipar.size), axis_names, rotation='vertical', fontsize=11)
        plt.yticks(np.arange(ipar.size), axis_names, fontsize=11)
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=9)
        fig.tight_layout()
        if parameters.SAVE:
            figname = self.filename.replace(self.filename.split('.')[-1], "_correlation.pdf")
            self.my_logger.info(f"Save figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')
        if parameters.DISPLAY:
            if self.live_fit:
                plt.draw()
                plt.pause(1e-8)
            else:
                plt.show()

    def weighted_residuals(self, p):
        lambdas, model, model_err = self.simulate(*p)
        res = ((model - self.data) / np.sqrt(model_err ** 2 + self.err ** 2)).flatten()
        return res

    def chisq(self, p):
        res = self.weighted_residuals(p)
        chisq = np.sum(res ** 2)
        return chisq

    def lnlike(self, p):
        return -0.5 * self.chisq(p)

    def lnprior(self, p):
        in_bounds = True
        for npar, par in enumerate(p):
            if par < self.bounds[npar][0] or par > self.bounds[npar][1]:
                in_bounds = False
                break
        if in_bounds:
            return 0.0
        else:
            return -np.inf

    def jacobian(self, params, epsilon, fixed_params=None):
        lambdas, model, model_err = self.simulate(*params)
        model = model.flatten()
        J = np.zeros((params.size, model.size))
        for ip, p in enumerate(params):
            if fixed_params[ip]:
                continue
            tmp_p = np.copy(params)
            if tmp_p[ip] + epsilon[ip] < self.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.bounds[ip][1]:
                epsilon[ip] = - epsilon[ip]
            tmp_p[ip] += epsilon[ip]
            tmp_lambdas, tmp_model, tmp_model_err = self.simulate(*tmp_p)
            J[ip] = (tmp_model.flatten() - model) / epsilon[ip]
        return J

    @staticmethod
    def hessian(J, W):
        # algebra
        JT_W = J.T * W
        JT_W_J = JT_W @ J
        L = np.linalg.inv(np.linalg.cholesky(JT_W_J))
        inv_JT_W_J = L.T @ L
        return inv_JT_W_J


class SpectrumFitWorkspace(FitWorkspace):

    def __init__(self, file_name, atmgrid_file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
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
        self.pwv = 3
        self.aerosols = 0.03
        self.reso = 1.5
        self.D = self.spectrum.header['D2CCD']
        self.shift = self.spectrum.header['PIXSHIFT']
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols, self.reso, self.D, self.shift])
        self.ndim = len(self.p)
        self.input_labels = ["A1", "A2", "ozone", "PWV", "VAOD", "reso [pix]", r"D_CCD [mm]",
                             r"alpha_pix [pix]"]
        self.axis_names = ["$A_1$", "$A_2$", "ozone", "PWV", "VAOD", "reso [pix]", r"$D_{CCD}$ [mm]",
                           r"$\alpha_{\mathrm{pix}}$ [pix]"]
        self.bounds = [(0, 2), (0, 0.5), (0, 800), (0, 10), (0, 1), (1, 10), (50, 60), (-20, 20)]
        if atmgrid_filename != "":
            self.bounds[2] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.bounds[3] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.bounds[4] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.simulation = SpectrumSimulation(self.spectrum, self.atmosphere, self.telescope, self.disperser)
        self.get_truth()

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
                          reso_truth, D_truth, shift_truth)
        else:
            self.truth = None

    def plot_spectrum_comparison_simple(self, ax, title='', extent=None, size=0.4):
        lambdas = self.spectrum.lambdas
        sub = np.where((lambdas > parameters.LAMBDA_MIN) & (lambdas < parameters.LAMBDA_MAX))
        if extent is not None:
            sub = np.where((lambdas > extent[0]) & (lambdas < extent[1]))
        self.spectrum.plot_spectrum_simple(ax, lambdas=lambdas)
        p0 = ax.plot(lambdas, self.model(lambdas), label='model')
        ax.fill_between(lambdas, self.model - self.model_err,
                        self.model(lambdas) + self.model_err, alpha=0.3, color=p0[0].get_color())
        # ax.plot(self.lambdas, self.model_noconv, label='before conv')
        if title != '':
            ax.set_title(title, fontsize=10)
        ax.legend()
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("bottom", size=size, pad=0)
        ax.figure.add_axes(ax2)
        residuals = (self.spectrum.data - self.model(lambdas)) / self.model(lambdas)
        residuals_err = self.spectrum.err / self.model(lambdas)
        ax2.errorbar(lambdas, residuals, yerr=residuals_err, fmt='ro', markersize=2)
        ax2.axhline(0, color=p0[0].get_color())
        ax2.grid(True)
        residuals_model = self.model_err / self.model(lambdas)
        ax2.fill_between(lambdas, -residuals_model, residuals_model, alpha=0.3, color=p0[0].get_color())
        std = np.std(residuals[sub])
        ax2.set_ylim([-2. * std, 2. * std])
        ax2.set_xlabel(ax.get_xlabel())
        ax2.set_ylabel('(data-fit)/fit')
        ax2.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_xlim((lambdas[sub][0], lambdas[sub][-1]))
        ax.set_ylim((0.9 * np.min(self.spectrum.data[sub]), 1.1 * np.max(self.spectrum.data[sub])))
        ax.set_xticks(ax2.get_xticks()[1:-1])
        ax.get_yaxis().set_label_coords(-0.15, 0.6)
        ax2.get_yaxis().set_label_coords(-0.15, 0.5)

    def simulate(self, A1, A2, ozone, pwv, aerosols, reso, D, shift):
        lambdas, model, model_err = self.simulation.simulate(A1, A2, ozone, pwv, aerosols, reso, D, shift)
        # if self.live_fit:
        #    self.plot_fit()
        self.model = model
        self.model_err = model_err
        return model, model_err

    def plot_fit(self):
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


class SpectrogramFitWorkspace(FitWorkspace):

    def __init__(self, file_name, atmgrid_file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        FitWorkspace.__init__(self, file_name, nwalkers, nsteps, burnin, nbins, verbose, plot,
                              live_fit, truth=truth)
        self.spectrum, self.telescope, self.disperser, self.target = SimulatorInit(file_name)
        self.airmass = self.spectrum.header['AIRMASS']
        self.pressure = self.spectrum.header['OUTPRESS']
        self.temperature = self.spectrum.header['OUTTEMP']
        self.my_logger = set_logger(self.__class__.__name__)
        if atmgrid_file_name == "":
            self.atmosphere = Atmosphere(self.airmass, self.pressure, self.temperature)
        else:
            self.use_grid = True
            self.atmosphere = AtmosphereGrid(file_name, atmgrid_file_name)
            if parameters.VERBOSE:
                self.my_logger.info(f'\n\tUse atmospheric grid models from file {atmgrid_file_name}. ')
        self.crop_spectrogram()
        self.lambdas = self.spectrum.lambdas
        self.data = self.spectrum.spectrogram
        self.err = self.spectrum.spectrogram_err
        self.A1 = 1.0
        self.A2 = 0.01
        self.ozone = 400.
        self.pwv = 3
        self.aerosols = 0.05
        self.D = self.spectrum.header['D2CCD']
        self.psf_poly_params = self.spectrum.chromatic_psf.from_table_to_poly_params()
        length = len(self.spectrum.chromatic_psf.table)
        self.psf_poly_params = self.psf_poly_params[length:-1]  # remove saturation (fixed parameter)
        self.psf_poly_params_labels = np.copy(self.spectrum.chromatic_psf.poly_params_labels[length:-1])
        self.psf_poly_params_names = np.copy(self.spectrum.chromatic_psf.poly_params_names[length:-1])
        self.psf_poly_params_bounds = self.spectrum.chromatic_psf.set_bounds(data=None)
        self.shift_x = self.spectrum.header['PIXSHIFT']
        self.shift_y = 0.
        self.angle = self.spectrum.rotation_angle
        self.saturation = self.spectrum.spectrogram_saturation
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols,
                           self.D, self.shift_x, self.shift_y, self.angle])
        self.psf_params_start_index = self.p.size
        self.p = np.concatenate([self.p, self.psf_poly_params])
        self.ndim = self.p.size
        self.input_labels = ["A1", "A2", "ozone [db]", "PWV [mm]", "VAOD", r"D_CCD [mm]",
                             r"shift_x [pix]", r"shift_y [pix]", r"angle [deg]"] + list(self.psf_poly_params_labels)
        self.axis_names = ["$A_1$", "$A_2$", "ozone [db]", "PWV [mm]", "VAOD", r"$D_{CCD}$ [mm]",
                           r"$\Delta_{\mathrm{x}}$ [pix]", r"$\Delta_{\mathrm{y}}$ [pix]",
                           r"$\theta$ [deg]"] + list(self.psf_poly_params_names)
        self.bounds = np.concatenate([np.array([(0, 2), (0, 0.5), (0, 800), (1, 10), (0, 1),
                                                (50, 60), (-3, 3), (-3, 3), (-90, 90)]),
                                      self.psf_poly_params_bounds[:-1]])  # remove saturation
        if atmgrid_file_name != "":
            self.bounds[2] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.bounds[3] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.bounds[4] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.simulation = SpectrogramModel(self.spectrum, self.atmosphere, self.telescope, self.disperser,
                                           with_background=True, fast_sim=False)
        self.get_spectrogram_truth()

    def crop_spectrogram(self):
        bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        # spectrogram must have odd size in y for the fourier simulation
        yeven = 0
        if (self.spectrum.spectrogram_Ny - 2 * bgd_width) % 2 == 0:
            yeven = 1
        self.spectrum.spectrogram_ymax = self.spectrum.spectrogram_ymax - bgd_width + yeven
        self.spectrum.spectrogram_ymin += bgd_width
        self.spectrum.spectrogram_bgd = self.spectrum.spectrogram_bgd[bgd_width:-bgd_width + yeven, :]
        self.spectrum.spectrogram = self.spectrum.spectrogram[bgd_width:-bgd_width + yeven, :]
        self.spectrum.spectrogram_err = self.spectrum.spectrogram_err[bgd_width:-bgd_width + yeven, :]
        self.spectrum.spectrogram_y0 -= bgd_width
        self.spectrum.spectrogram_Ny, self.spectrum.spectrogram_Nx = self.spectrum.spectrogram.shape
        self.my_logger.debug(f'\n\tSize of the spectrogram region after cropping: '
                             f'({self.spectrum.spectrogram_Nx},{self.spectrum.spectrogram_Ny})')

    def get_spectrogram_truth(self):
        if 'A1' in list(self.spectrum.header.keys()):
            A1_truth = self.spectrum.header['A1']
            A2_truth = self.spectrum.header['A2']
            ozone_truth = self.spectrum.header['OZONE']
            pwv_truth = self.spectrum.header['PWV']
            aerosols_truth = self.spectrum.header['VAOD']
            D_truth = self.spectrum.header['D2CCD']
            # shift_x_truth = self.spectrum.header['X0SHIFT']
            # shift_y_truth = self.spectrum.header['Y0SHIFT']
            angle_truth = self.spectrum.header['ROTANGLE']
            self.truth = (A1_truth, A2_truth, ozone_truth, pwv_truth, aerosols_truth,
                          D_truth, angle_truth)
        else:
            self.truth = None
        self.truth = None

    def plot_spectrogram_comparison_simple(self, ax, title='', extent=None, dispersion=False):
        lambdas = self.spectrum.lambdas
        sub = np.where((lambdas > parameters.LAMBDA_MIN) & (lambdas < parameters.LAMBDA_MAX))[0]
        sub = np.where(sub < self.spectrum.spectrogram_Nx)[0]
        if extent is not None:
            sub = np.where((lambdas > extent[0]) & (lambdas < extent[1]))[0]
        if len(sub) > 0:
            norm = np.max(self.spectrum.spectrogram[:, sub])
            plot_image_simple(ax[0, 0], data=self.model[:, sub] / norm, aspect='auto', cax=ax[0, 1], vmin=0, vmax=1,
                              units='1/max(data)')
            if dispersion:
                x = self.spectrum.chromatic_psf.table['Dx'][sub[2:-3]] + self.spectrum.spectrogram_x0 - sub[0]
                y = np.ones_like(x)
                ax[0, 0].scatter(x, y, cmap=from_lambda_to_colormap(self.lambdas[sub[2:-3]]), edgecolors='None',
                                 c=self.lambdas[sub[2:-3]],
                                 label='', marker='o', s=10)
            # p0 = ax.plot(lambdas, self.model(lambdas), label='model')
            # # ax.plot(self.lambdas, self.model_noconv, label='before conv')
            if title != '':
                ax[0, 0].set_title(title, fontsize=10, loc='center', color='white', y=0.8)
            plot_image_simple(ax[1, 0], data=self.spectrum.spectrogram[:, sub] / norm, title='Data', aspect='auto',
                              cax=ax[1, 1], vmin=0, vmax=1, units='1/max(data)')
            ax[1, 0].set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
            residuals = (self.spectrum.spectrogram - self.model)
            # residuals_err = self.spectrum.spectrogram_err / self.model
            norm = self.spectrum.spectrogram_err
            residuals /= norm
            std = float(np.std(residuals[:, sub]))
            plot_image_simple(ax[2, 0], data=residuals[:, sub], vmin=-5 * std, vmax=5 * std, title='(Data-Model)/Err',
                              aspect='auto', cax=ax[2, 1], units='', cmap="bwr")
            ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, loc='center', color='black', y=0.8)
            ax[2, 0].text(0.05, 0.05, f'mean={np.mean(residuals[:, sub]):.3f}\nstd={np.std(residuals[:, sub]):.3f}',
                          horizontalalignment='left', verticalalignment='bottom',
                          color='black', transform=ax[2, 0].transAxes)
            ax[0, 0].set_xticks(ax[2, 0].get_xticks()[1:-1])
            ax[0, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[1, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[2, 1].get_yaxis().set_label_coords(3.5, 0.5)
            # ax[0, 0].get_yaxis().set_label_coords(-0.15, 0.6)
            # ax[2, 0].get_yaxis().set_label_coords(-0.15, 0.5)
            # remove the underlying axes
            # for ax in ax[3, 1]:
            ax[3, 1].remove()
            ax[3, 0].plot(self.lambdas[sub], self.spectrum.spectrogram.sum(axis=0)[sub], label='Data')
            ax[3, 0].plot(self.lambdas[sub], self.model.sum(axis=0)[sub], label='Model')
            ax[3, 0].set_ylabel('Cross spectrum')
            ax[3, 0].set_xlabel(r'$\lambda$ [nm]')
            ax[3, 0].legend(fontsize=7)
            ax[3, 0].grid(True)

    def simulate(self, A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, angle, *psf_poly_params):
        global plot_counter
        self.simulation.fix_psf_cube = False
        if np.all(np.isclose(psf_poly_params, self.p[self.psf_params_start_index:], rtol=1e-6)):
            self.simulation.fix_psf_cube = True
        lambdas, model, model_err = \
            self.simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, angle, psf_poly_params)
        self.p = np.array([A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, angle] + list(psf_poly_params))
        self.lambdas = lambdas
        self.model = model
        self.model_err = model_err
        if self.live_fit and (plot_counter % 30) == 0:
            self.plot_fit()
        plot_counter += 1
        return lambdas, model, model_err

    def jacobian(self, params, epsilon, fixed_params=None):
        start = time.time()
        lambdas, model, model_err = self.simulate(*params)
        model = model.flatten()
        J = np.zeros((params.size, model.size))
        for ip, p in enumerate(params):
            if fixed_params[ip]:
                continue
            if ip < self.psf_params_start_index:
                self.simulation.fix_psf_cube = True
            else:
                self.simulation.fix_psf_cube = False
            tmp_p = np.copy(params)
            if tmp_p[ip] + epsilon[ip] < self.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.bounds[ip][1]:
                epsilon[ip] = - epsilon[ip]
            tmp_p[ip] += epsilon[ip]
            tmp_lambdas, tmp_model, tmp_model_err = self.simulate(*tmp_p)
            J[ip] = (tmp_model.flatten() - model) / epsilon[ip]
            # print(ip, self.input_labels[ip], p, tmp_p[ip] + epsilon[ip], J[ip])
        # if False:
        #     plt.imshow(J, origin="lower", aspect="auto")
        #     plt.show()
        print(f"\tjacobian time computation = {time.time() - start:.1f}s")
        return J

    def plot_fit(self):
        """
        Examples
        --------
        >>> file_name = 'outputs/reduc_20170530_130_spectrum.fits'
        >>> atmgrid_filename = file_name.replace('sim', 'reduc').replace('spectrum', 'atmsim')
        >>> fit_workspace = SpectrogramFitWorkspace(file_name, atmgrid_filename=atmgrid_filename,
        ... nwalkers=28, nsteps=20000, burnin=10000, nbins=10, verbose=1, plot=True, live_fit=False)
        >>> A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, angle, *psf = fit_workspace.p
        >>> lambdas, model, model_err = fit_workspace.simulation.simulate(A1, A2, ozone, pwv, aerosols,
        ... D, shift_x, shift_y, shift_t, angle, psf)
        >>> fit_workspace.lambdas = lambdas
        >>> fit_workspace.model = model
        >>> fit_workspace.model_err = model_err
        >>> fit_workspace.plot_fit()
        """
        gs_kw = dict(width_ratios=[3, 0.15, 1, 0.15, 1, 0.15], height_ratios=[1, 1, 1, 1])
        fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(12, 8), constrained_layout=True, gridspec_kw=gs_kw)

        A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, *psf = self.p
        plt.suptitle(f'A1={A1:.3f}, A2={A2:.3f}, PWV={pwv:.3f}, OZ={ozone:.3g}, VAOD={aerosols:.3f}, '
                     f'D={D:.2f}mm, shift_x={shift_x:.2f}pix, shift_y={shift_y:.2f}pix')
        # main plot
        self.plot_spectrogram_comparison_simple(ax[:, 0:2], title='Spectrogram model', dispersion=True)
        # zoom O2
        self.plot_spectrogram_comparison_simple(ax[:, 2:4], extent=[730, 800], title='Zoom $O_2$', dispersion=True)
        # zoom H2O
        self.plot_spectrogram_comparison_simple(ax[:, 4:6], extent=[870, 1000], title='Zoom $H_2 O$', dispersion=True)
        # fig.tight_layout()
        if self.live_fit:
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            if parameters.DISPLAY and self.verbose:
                plt.show()
        if parameters.SAVE:
            figname = self.filename.replace(self.filename.split('.')[-1], "_bestfit.pdf")
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')


def lnprob_spectrogram(p):
    global fit_workspace
    lp = fit_workspace.lnprior(p)
    if not np.isfinite(lp):
        return -1e20
    return lp + fit_workspace.lnlike_spectrogram(p)


def lnprob(p):
    global fit_workspace
    lp = fit_workspace.lnprior(p)
    if not np.isfinite(lp):
        return -1e20
    return lp + fit_workspace.lnlike(p)


def gradient_descent(fit_workspace, params, epsilon, niter=10, fixed_params=None, xtol=1e-3, ftol=1e-3):
    tmp_params = np.copy(params)
    W = 1 / fit_workspace.err.flatten() ** 2
    ipar = np.arange(params.size)
    if fixed_params is not None:
        ipar = np.array(np.where(np.array(fixed_params).astype(int) == 0)[0])
    costs = []
    params_table = []
    inv_JT_W_J = np.zeros((len(ipar), len(ipar)))
    for i in range(niter):
        print(f"start iteration={i}", end=' ')  # , tmp_params)
        start = time.time()
        tmp_lambdas, tmp_model, tmp_model_err = fit_workspace.simulate(*tmp_params)
        # if fit_workspace.live_fit:
        #    fit_workspace.plot_fit()
        residuals = (tmp_model - fit_workspace.data).flatten()
        cost = np.sum((residuals ** 2) * W)
        print(f"cost={cost:.3f} chisq_red={cost / tmp_model.size:.3f}")
        J = fit_workspace.jacobian(tmp_params, epsilon, fixed_params=fixed_params)
        # remove parameters with unexpected null Jacobian vectors
        for ip in range(J.shape[0]):
            if ip not in ipar:
                continue
            if np.all(J[ip] == np.zeros(J.shape[1])):
                ipar = np.delete(ipar, list(ipar).index(ip))
                tmp_params[ip] = 0
                print(f"Step {i}: {fit_workspace.input_labels[ip]} has a null Jacobian; parameter is fixed at 0 "
                      f"in the following instead of its current value ({tmp_params[ip]}).")
        # remove fixed parameters
        J = J[ipar].T
        # algebra
        JT_W = J.T * W
        JT_W_J = JT_W @ J
        L = np.linalg.inv(np.linalg.cholesky(JT_W_J))
        inv_JT_W_J = L.T @ L
        if fit_workspace.live_fit:
            fit_workspace.cov = inv_JT_W_J
            fit_workspace.plot_correlation_matrix(ipar)
        JT_W_R0 = JT_W @ residuals
        dparams = - inv_JT_W_J @ JT_W_R0

        def line_search(alpha):
            tmp_params_2 = np.copy(tmp_params)
            tmp_params_2[ipar] = tmp_params[ipar] + alpha * dparams
            lbd, mod, err = fit_workspace.simulate(*tmp_params_2)
            return np.sum(((mod - fit_workspace.data) / fit_workspace.err) ** 2)

        # tol parameter acts on alpha (not func)
        alpha_min, fval, iter, funcalls = optimize.brent(line_search, full_output=True, tol=1e-2)
        print(f"\talpha_min={alpha_min:.3g} iter={iter} funcalls={funcalls}")
        tmp_params[ipar] += alpha_min * dparams
        print(f"shift: {alpha_min * dparams}")
        print(f"new params: {tmp_params[ipar]}")
        # check bounds
        for ip, p in enumerate(tmp_params):
            if p < fit_workspace.bounds[ip][0]:
                # print(ip, fit_workspace.axis_names[ip], tmp_params[ip], fit_workspace.bounds[ip][0])
                tmp_params[ip] = fit_workspace.bounds[ip][0]
            if p > fit_workspace.bounds[ip][1]:
                # print(ip, fit_workspace.axis_names[ip], tmp_params[ip], fit_workspace.bounds[ip][1])
                tmp_params[ip] = fit_workspace.bounds[ip][1]
        # in_bounds, penalty, outbound_parameter_name = \
        #     fit_workspace.spectrum.chromatic_psf.check_bounds(tmp_params[fit_workspace.psf_params_start_index:])
        # prepare outputs
        costs.append(fval)
        params_table.append(np.copy(tmp_params))
        print(f"end iteration={i} in {time.time() - start:.2f}s cost={fval:.3f}")
        # if np.sum(np.abs(alpha_min * dparams)) / np.sum(np.abs(tmp_params[ipar])) < xtol \
        #         or (len(costs) > 1 and np.abs(costs[-2] - fval) / np.max([np.abs(fval), np.abs(costs[-2]), 1]) < ftol):
        #     break
        if len(costs) > 1 and np.abs(costs[-2] - fval) / np.max([np.abs(fval), np.abs(costs[-2]), 1]) < ftol:
            break
    plt.close()
    return tmp_params, inv_JT_W_J, np.array(costs), np.array(params_table)


def plot_psf_poly_params(psf_poly_params):
    from spectractor.extractor.psf import PSF1D
    psf = PSF1D()
    truth_psf_poly_params = [0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955,
                             -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901,
                             0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673,
                             528.3594585697788, 628.4966480821147, 12.438043546369354, 499.99999999999835]

    x = np.linspace(-1, 1, 100)
    for i in range(5):
        plt.plot(x, np.polynomial.legendre.legval(x, truth_psf_poly_params[3 * i:3 * i + 3]),
                 label="truth " + psf.param_names[1 + i])
        plt.plot(x, np.polynomial.legendre.legval(x, psf_poly_params[3 * i:3 * i + 3]),
                 label="fit " + psf.param_names[1 + i])

        plt.legend()
        plt.show()


def print_parameter_summary(params, cov, labels):
    for ip in np.arange(0, cov.shape[0]).astype(int):
        txt = "%s: %s +%s -%s" % formatting_numbers(params[ip], np.sqrt(cov[ip, ip]), np.sqrt(cov[ip, ip]),
                                                    label=labels[ip])
        print(txt)


def plot_gradient_descent(fit_workspace, costs, params_table):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex="all")
    iterations = np.arange(params_table.shape[0])
    ax[0].plot(iterations, costs, lw=2)
    for ip in range(params_table.shape[1]):
        ax[1].plot(iterations, params_table[:, ip], label=f"{fit_workspace.axis_names[ip]}")
    ax[1].set_yscale("symlog")
    ax[1].legend(ncol=6, loc=9)
    ax[1].grid()
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\chi^2$")
    ax[1].set_ylabel("Parameters")
    ax[0].grid()
    ax[1].set_xlabel("Iterations")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if parameters.SAVE:
        figname = fit_workspace.filename.replace(fit_workspace.filename.split('.')[-1], "_fitting.pdf")
        fit_workspace.my_logger.info(f"\n\tSave figure {figname}.")
        fig.savefig(figname, dpi=100, bbox_inches='tight')
    if parameters.DISPLAY:
        plt.show()

    fit_workspace.simulate(*fit_workspace.p)
    fit_workspace.live_fit = False
    fit_workspace.plot_fit()


def save_gradient_descent(fit_workspace, costs, params_table):
    iterations = np.arange(params_table.shape[0]).astype(int)
    t = np.zeros((params_table.shape[1] + 2, params_table.shape[0]))
    t[0] = iterations
    t[1] = costs
    t[2:] = params_table.T
    h = 'iter,costs,' + ','.join(fit_workspace.input_labels)
    output_filename = fit_workspace.filename.replace(fit_workspace.filename.split('.')[-1], "_fitting.txt")
    np.savetxt(output_filename, t.T, header=h, delimiter=",")
    fit_workspace.my_logger.info(f"\n\tSave gradient descent log {output_filename}.")


def run_minimisation(fit_workspace, method="newton"):
    my_logger = set_logger(__name__)
    bounds = fit_workspace.bounds

    nll = lambda params: -fit_workspace.lnlike(params)

    # sim_134
    # guess = fit_workspace.p
    # truth sim_134
    # guess = np.array([1., 0.05, 300, 5, 0.03, 55.45, 0.0, 0.0, -1.54, 0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955, -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901, 0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673, 528.3594585697788, 628.4966480821147, 12.438043546369354])
    guess = np.array(
        [1., 0.05, 300, 5, 0.03, 55.45, -0.275, 0.0, -1.54, -1.47570237e-01, -5.00195918e-01, 4.74296776e-01,
         2.85776501e+00, -1.86436219e+00, 1.83899390e+00, 1.89342052e+00,
         -9.43239034e-01, 1.06985560e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 1.44368271e+00, -9.95896258e-01, 1.59015965e+00])
    # 5.00000000e+02
    guess = fit_workspace.p
    if method == "minimize":
        start = time.time()
        result = optimize.minimize(nll, fit_workspace.p, method='L-BFGS-B',
                                   options={'ftol': 1e-20, 'xtol': 1e-20, 'gtol': 1e-20, 'disp': True,
                                            'maxiter': 100000,
                                            'maxls': 50, 'maxcor': 30},
                                   bounds=bounds)
        fit_workspace.p = result['x']
        print(f"Minimize: total computation time: {time.time() - start}s")
        fit_workspace.simulate(*fit_workspace.p)
        fit_workspace.live_fit = False
        fit_workspace.plot_fit()

    elif method == "least_squares":
        start = time.time()
        x_scale = np.abs(guess)
        x_scale[x_scale == 0] = 0.1
        p = optimize.least_squares(fit_workspace.weighted_residuals, guess, verbose=2, ftol=1e-6, x_scale=x_scale,
                                   diff_step=0.001, bounds=bounds.T)
        fit_workspace.p = p.x  # m.np_values()
        print(f"Least_squares: total computation time: {time.time() - start}s")
        fit_workspace.simulate(*fit_workspace.p)
        fit_workspace.live_fit = False
        fit_workspace.plot_fit()
    elif method == "minuit":
        start = time.time()
        # fit_workspace.simulation.fix_psf_cube = False
        error = 0.1 * np.abs(guess) * np.ones_like(guess)
        error[2:5] = 0.3 * np.abs(guess[2:5]) * np.ones_like(guess[2:5])
        z = np.where(np.isclose(error, 0.0, 1e-6))
        error[z] = 1.
        fix = [False] * guess.size
        # noinspection PyArgumentList
        m = Minuit.from_array_func(fcn=nll, start=guess, error=error, errordef=1,
                                   fix=fix, print_level=2, limit=bounds)
        m.tol = 10
        m.migrad()
        fit_workspace.p = m.np_values()
        print(f"Minuit: total computation time: {time.time() - start}s")
        fit_workspace.simulate(*fit_workspace.p)
        fit_workspace.live_fit = False
        fit_workspace.plot_fit()
    elif method == "newton":

        def bloc_gradient_descent(guess, epsilon, params_table, costs, fix, xtol, ftol, niter):
            fit_workspace.p, fit_workspace.cov, tmp_costs, tmp_params_table = gradient_descent(fit_workspace, guess,
                                                                                               epsilon, niter=niter,
                                                                                               fixed_params=fix,
                                                                                               xtol=xtol, ftol=ftol)
            params_table = np.concatenate([params_table, tmp_params_table])
            costs = np.concatenate([costs, tmp_costs])
            ipar = np.array(np.where(np.array(fix).astype(int) == 0)[0])
            print_parameter_summary(fit_workspace.p[ipar], fit_workspace.cov,
                                    [fit_workspace.input_labels[ip] for ip in ipar])
            if True:
                # plot_psf_poly_params(fit_workspace.p[fit_workspace.psf_params_start_index:])
                plot_gradient_descent(fit_workspace, costs, params_table)
                fit_workspace.plot_correlation_matrix(ipar=ipar)
            return params_table, costs

        fit_workspace.simulation.fast_sim = True
        costs = np.array([fit_workspace.chisq_spectrogram(guess)])
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
        my_logger.info(f"\n\tStart guess: {guess}")

        # cancel the Gaussian part of the PSF
        # TODO: solve this Gaussian PSF part issue
        guess[-6:] = 0

        # fit trace
        fix = [True] * guess.size
        fix[0] = False  # A1
        fix[1] = False  # A2
        fix[6] = True  # x0
        fix[7] = True  # y0
        fix[8] = True  # angle
        fit_workspace.simulation.fast_sim = True
        fix[fit_workspace.psf_params_start_index:fit_workspace.psf_params_start_index + 3] = [False] * 3
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = bloc_gradient_descent(guess, epsilon, params_table, costs,
                                                    fix=fix, xtol=1e-2, ftol=1e-2, niter=20)

        # fit PSF
        guess = np.array(fit_workspace.p)
        fix = [True] * guess.size
        fix[0] = False  # A1
        fix[1] = False  # A2
        fit_workspace.simulation.fast_sim = True
        fix[fit_workspace.psf_params_start_index:fit_workspace.psf_params_start_index + 9] = [False] * 9
        # fix[fit_workspace.psf_params_start_index:] = [False] * (guess.size - fit_workspace.psf_params_start_index)
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = bloc_gradient_descent(guess, epsilon, params_table, costs,
                                                    fix=fix, xtol=1e-2, ftol=1e-2, niter=20)

        # fit dispersion
        guess = np.array(fit_workspace.p)
        fix = [True] * guess.size
        fix[0] = False
        fix[1] = False
        fix[5] = False  # DCCD
        fix[6] = False  # x0
        fit_workspace.simulation.fix_psf_cube = True
        fit_workspace.simulation.fast_sim = True
        params_table, costs = bloc_gradient_descent(guess, epsilon, params_table, costs,
                                                    fix=fix, xtol=1e-3, ftol=1e-2, niter=10)

        # fit all except Gaussian part of the PSF
        # TODO: solve this Gaussian PSF part issue
        guess = np.array(fit_workspace.p)
        fit_workspace.simulation.fast_sim = False
        fix = [False] * guess.size
        fix[6] = False  # x0
        fix[7] = True  # y0
        fix[8] = True  # angle
        fix[-6:] = [True] * 6  # gaussian part
        parameters.SAVE = True
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = bloc_gradient_descent(guess, epsilon, params_table, costs,
                                                    fix=fix, xtol=1e-5, ftol=1e-5, niter=40)
        fit_workspace.save_parameters_summary(header=fit_workspace.spectrum.date_obs)
        save_gradient_descent(fit_workspace, costs, params_table)
        print(f"Newton: total computation time: {time.time() - start}s")


def run_emcee(fit_workspace, ln=lnprob):
    my_logger = set_logger(__name__)
    fit_workspace.print_settings()
    nsamples = fit_workspace.nsteps
    p0 = fit_workspace.set_start()
    filename = fit_workspace.emcee_filename
    backend = emcee.backends.HDFBackend(filename)
    try:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(fit_workspace.nwalkers, fit_workspace.ndim, ln, args=(),
                                        pool=pool, backend=backend)
        print(f"Initial size: {backend.iteration}")
        if backend.iteration > 0:
            p0 = backend.get_last_sample()
        if nsamples - backend.iteration > 0:
            sampler.run_mcmc(p0, nsteps=max(0, nsamples - backend.iteration), progress=True)
        pool.close()
    except ValueError:
        sampler = emcee.EnsembleSampler(fit_workspace.nwalkers, fit_workspace.ndim, ln, args=(),
                                        threads=multiprocessing.cpu_count(), backend=backend)
        print(f"Initial size: {backend.iteration}")
        if backend.iteration > 0:
            p0 = sampler.get_last_sample()
        for _ in sampler.sample(p0, iterations=max(0, nsamples - backend.iteration), progress=True, store=True):
            continue
    fit_workspace.chains = sampler.chain
    fit_workspace.lnprobs = sampler.lnprobability


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

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    load_config(args.config)

    # filename = 'outputs/reduc_20170530_130_spectrum.fits'
    # filename = 'outputs/sim_20170530_134_spectrum.fits'
    # 062
    filename = 'CTIODataJune2017_reduced_RG715_v2_prod6/data_30may17/sim_20170530_067_spectrum.fits'
    atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')

    w = SpectrogramFitWorkspace(filename, atmgrid_file_name=atmgrid_filename, nsteps=1000,
                                burnin=2, nbins=10, verbose=1, plot=True, live_fit=False)
    run_minimisation(w, method="newton")
    # run_emcee(w, ln=lnprob_spectrogram)
    # w.analyze_chains()

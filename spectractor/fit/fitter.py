from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectractor.simulation.simulator import *
from spectractor.fit.statistics import *

from spectractor.parameters import FIT_WORKSPACE as fit_workspace

from iminuit import Minuit
from scipy import optimize

import emcee
from schwimmbad import MPIPool

import time

plot_counter = 0


class FitWorkspace:

    def __init__(self, file_name, atmgrid_file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False):
        self.my_logger = set_logger(self.__class__.__name__)
        self.filename = file_name
        self.ndim = 0
        self.truth = None
        self.verbose = verbose
        self.plot = plot
        self.live_fit = live_fit
        self.p = np.array([])
        self.cov = np.array([[]])
        self.rho = np.array([[]])
        self.ndim = len(self.p)
        self.lambdas = None
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
        self.chains = np.array([[]])
        self.lnprobs = np.array([[]])
        self.flat_chains = np.array([[]])
        self.valid_chains = [False] * self.nwalkers
        self.title = ""
        self.spectrum, self.telescope, self.disperser, self.target = SimulatorInit(file_name)
        self.airmass = self.spectrum.header['AIRMASS']
        self.pressure = self.spectrum.header['OUTPRESS']
        self.temperature = self.spectrum.header['OUTTEMP']
        self.use_grid = False
        if atmgrid_file_name == "":
            self.atmosphere = Atmosphere(self.airmass, self.pressure, self.temperature)
        else:
            self.use_grid = True
            self.atmosphere = AtmosphereGrid(file_name, atmgrid_file_name)
            if parameters.VERBOSE:
                self.my_logger.info(f'\n\tUse atmospheric grid models from file {atmgrid_file_name}. ')
        self.truth = None
        self.simulation = None
        self.emcee_filename = self.filename.replace("_spectrum.fits", "_emcee.h5")
        if parameters.DEBUG:
            for k in range(10):
                atmo = self.atmosphere.simulate(300, k, 0.05)
                plt.plot(self.atmosphere.lambdas, atmo, label='pwv=%dmm' % k)
            plt.grid()
            plt.xlabel('$\lambda$ [nm]')
            plt.ylabel('Atmospheric transmission')
            plt.legend(loc='best')
            if parameters.DISPLAY:
                plt.show()

    def set_start(self):
        self.start = np.array(
            [np.random.uniform(self.p[i] - 0.02 * self.p[i], self.p[i] + 0.02 * self.p[i], self.nwalkers)
             for i in range(self.ndim)]).T
        self.start[self.start == 0] = 1e-5 * np.random.uniform(0, 1)
        return self.start

    def load_chains(self):
        self.chains = [[]]
        self.lnprobs = [[]]
        self.nbetafits = [[]]
        self.nsteps = 0
        tau = -1
        reader = emcee.backends.HDFBackend(self.emcee_filename)
        try:
            tau = reader.get_autocorr_time()
            burnin = int(2 * np.max(tau))
            thin = 1 #int(0.5 * np.min(tau))
        except emcee.autocorr.AutocorrError:
            tau = -1
        self.chains = reader.get_chain(discard=0, flat=False, thin=1)
        self.lnprobs = reader.get_log_prob(discard=0, flat=False, thin=1)
        self.nsteps = self.chains.shape[0]
        self.nwalkers = self.chains.shape[1]
        print(f"Auto-correlation time: {tau}")
        print(f"Burn-in: {burnin}")
        print(f"Thin: {thin}")
        print(f"Chains shape: {self.chains.shape}")
        print(f"Log prob shape: {self.lnprobs.shape}")
        return self.chains, self.lnprobs

    def build_flat_chains(self):
        self.flat_chains = self.chains[self.burnin:, self.valid_chains, :].reshape((-1, self.ndim))
        return self.flat_chains

    def simulate(self, *p):
        pass

    def analyze_chains(self):
        self.load_chains()
        self.set_chain_validity()
        self.convergence_tests()
        self.build_flat_chains()
        likelihood = self.chain2likelihood()
        self.p = likelihood.mean_vec
        self.simulate(*self.p)
        if isinstance(self, SpectrumFitWorkspace):
            self.plot_fit()
        elif isinstance(self, SpectrogramFitWorkspace):
            self.plot_spectrogram_fit()
        figure_name = self.filename.replace('.fits', '_triangle.pdf')
        likelihood.triangle_plots(output_filename=figure_name)
        self.cov = likelihood.cov_matrix
        self.rho = likelihood.rho_matrix

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
        test = -2 * self.lnprobs[walker_index, start_index]
        counts = 1
        for index in range(start_index + 1, last_index):
            chi2 = -2 * self.lnprobs[walker_index, index]
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
                ax[i, 1].plot(lens, Rs, lw=1, label=self.axis_names[i])
                ax[i, 1].axhline(0.03, c='k', linestyle='--')
                ax[i, 1].set_xlabel('Walker length', fontsize=fontsize)
                ax[i, 1].set_ylabel('$R-1$', fontsize=fontsize)
                ax[i, 1].set_ylim(0, 0.6)
                # ax[self.dim, 3].legend(loc='best', ncol=2, fontsize=10)
        fig.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.show()
        figure_name = self.emcee_filename.replace('.h5', '_convergence.pdf').replace('.txt', '_convergence.pdf')
        print(f'Save figure: {figure_name}')
        fig.savefig(figure_name, dpi=100)

    def print_settings(self):
        print('************************************')
        print(f"Input file: {self.filename}\nWalkers: {self.nwalkers}\t Steps: {self.nsteps}")
        print(f"Output file: {self.emcee_filename}")
        print('************************************')

    def save_parameters_summary(self):
        output_filename = self.filename.replace(".fits", "_bestfit.txt")
        f = open(output_filename, 'w')
        txt = self.spectrum.date_obs + "\n"
        for ip in np.arange(0, self.cov.shape[0]).astype(int):
            txt += "%s: %s +%s -%s\n" % formatting_numbers(self.p[ip], np.sqrt(self.cov[ip, ip]), np.sqrt(self.cov[ip, ip]),
                                                           label=self.input_labels[ip])
        for row in self.cov:
            txt += np.array_str(row, max_line_width=20*self.cov.shape[0]) + '\n'
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
            figname = self.filename.replace(".fits", "_correlation.pdf")
            self.my_logger.info(f"Save figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')
        if parameters.DISPLAY:
            if self.live_fit:
                plt.draw()
                plt.pause(1e-8)
            else:
                plt.show()


class SpectrumFitWorkspace(FitWorkspace):

    def __init__(self, filename, atmgrid_filename="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False):
        FitWorkspace.__init__(self, filename, atmgrid_filename, nwalkers, nsteps, burnin, nbins, verbose, plot,
                              live_fit)
        self.my_logger = set_logger(self.__class__.__name__)
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
        ax.fill_between(lambdas, self.model(lambdas) - self.model_err(lambdas),
                        self.model(lambdas) + self.model_err(lambdas), alpha=0.3, color=p0[0].get_color())
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
        residuals_model = self.model_err(lambdas) / self.model(lambdas)
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
        return model(self.spectrum.lambdas), model_err(self.spectrum.lambdas)

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

    def __init__(self, filename, atmgrid_filename="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False):
        FitWorkspace.__init__(self, filename, atmgrid_filename, nwalkers, nsteps, burnin, nbins, verbose, plot,
                              live_fit)
        self.my_logger = set_logger(self.__class__.__name__)
        self.crop_spectrogram()
        self.A1 = 1.0
        self.A2 = 0.01
        self.ozone = 400.
        self.pwv = 3
        self.aerosols = 0.05
        self.D = self.spectrum.header['D2CCD']
        self.psf_poly_params = self.spectrum.chromatic_psf.from_table_to_poly_params()
        l = len(self.spectrum.chromatic_psf.table)
        self.psf_poly_params = self.psf_poly_params[l:-1]  # remove saturation (fixed parameter)
        self.psf_poly_params_labels = np.copy(self.spectrum.chromatic_psf.poly_params_labels[l:-1])
        self.psf_poly_params_names = np.copy(self.spectrum.chromatic_psf.poly_params_names[l:-1])
        self.psf_poly_params_bounds = self.spectrum.chromatic_psf.set_bounds(data=None)
        self.shift_x = self.spectrum.header['PIXSHIFT']
        self.shift_y = 0.
        self.angle = self.spectrum.rotation_angle
        self.saturation = self.spectrum.spectrogram_saturation
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols,
                           self.D, self.shift_x, self.shift_y])
        self.psf_params_start_index = self.p.size
        self.p = np.concatenate([self.p, self.psf_poly_params])
        self.ndim = self.p.size
        self.input_labels = ["A1", "A2", "ozone [db]", "PWV [mm]", "VAOD", r"D_CCD [mm]",
                             r"shift_x [pix]", r"shift_y [pix]"] + list(self.psf_poly_params_labels)
        self.axis_names = ["$A_1$", "$A_2$", "ozone [db]", "PWV [mm]", "VAOD", r"$D_{CCD}$ [mm]",
                           r"$\Delta_{\mathrm{x}}$ [pix]", r"$\Delta_{\mathrm{y}}$ [pix]"] \
                           + list(self.psf_poly_params_names)
        self.bounds = np.concatenate([np.array([(0, 2), (0, 0.5), (0, 800), (0, 10), (0, 1),
                                                (50, 60), (-3, 3), (-3, 3)]),
                                      self.psf_poly_params_bounds[:-1]])  # remove saturation
        if atmgrid_filename != "":
            self.bounds[2] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.bounds[3] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.bounds[4] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.simulation = SpectrogramModel(self.spectrum, self.atmosphere, self.telescope, self.disperser)
        self.get_spectrogram_truth()

    def crop_spectrogram(self):
        bgd_width = parameters.PIXDIST_BACKGROUND + parameters.PIXWIDTH_BACKGROUND - parameters.PIXWIDTH_SIGNAL
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
                y = self.spectrum.chromatic_psf.table['Dy'][sub[2:-3]] + self.spectrum.spectrogram_y0
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
            residuals_err = self.spectrum.spectrogram_err / self.model
            norm = self.spectrum.spectrogram_err[:, sub]
            std = float(np.std(residuals[:, sub] / norm))
            plot_image_simple(ax[2, 0], data=residuals[:, sub] / norm, vmin=-5 * std, vmax=5 * std, title='Data-Model',
                              aspect='auto', cax=ax[2, 1], units='') #1/max(data)
            ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, loc='center', color='white', y=0.8)
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
            ax[3, 0].set_xlabel('$\lambda$ [nm]')
            ax[3, 0].legend(fontsize=7)
            ax[3, 0].grid(True)

    def simulate(self, A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, *psf_poly_params):
        global plot_counter
        self.simulation.fix_psf_cube = False
        if np.all(np.isclose(psf_poly_params, self.p[self.psf_params_start_index:], rtol=1e-6)):
            self.simulation.fix_psf_cube = True
        lambdas, model, model_err = \
            self.simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, psf_poly_params)
        self.p = np.array([A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y] + list(psf_poly_params))
        self.lambdas = lambdas
        self.model = model
        self.model_err = model_err
        if self.live_fit and (plot_counter % 30) == 0:
            self.plot_spectrogram_fit()
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
        if False:
            plt.imshow(J, origin="lower", aspect="auto")
            plt.show()
        print(f"\tjacobian time computation = {time.time() - start:.1f}s")
        return J

    def plot_spectrogram_fit(self):
        """
        Examples
        --------
        >>> file_name = 'outputs/reduc_20170530_130_spectrum.fits'
        >>> atmgrid_filename = file_name.replace('sim', 'reduc').replace('spectrum', 'atmsim')
        >>> fit_workspace = SpectrogramFitWorkspace(file_name, atmgrid_filename=atmgrid_filename, nwalkers=28, nsteps=20000, burnin=10000,
        ... nbins=10, verbose=1, plot=True, live_fit=False)
        >>> A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, angle, *psf = fit_workspace.p
        >>> lambdas, model, model_err = fit_workspace.simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, angle, psf)
        >>> fit_workspace.lambdas = lambdas
        >>> fit_workspace.model = model
        >>> fit_workspace.model_err = model_err
        >>> fit_workspace.plot_spectrogram_fit()
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
            figname = self.filename.replace(".fits", "_bestfit.pdf")
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')


def spectrogram_weighted_res(p, fit_workspace):
    lambdas, model, model_err = fit_workspace.simulate(*p)
    res = ((model - fit_workspace.spectrum.spectrogram) / fit_workspace.spectrum.spectrogram_err).flatten()
    return res


def chisq_spectrogram(p, fit_workspace):
    res = spectrogram_weighted_res(p, fit_workspace)
    chisq = np.sum(res ** 2)
    return chisq


def lnlike_spectrogram(p, fit_workspace):
    return -0.5 * chisq_spectrogram(p, fit_workspace)


def lnprob_spectrogram(p):
    global fit_workspace
    lp = lnprior(p, fit_workspace.bounds)
    if not np.isfinite(lp):
        return -1e20
    return lp + lnlike_spectrogram(p, fit_workspace)


def chisq(p, fit_workspace):
    model, err = fit_workspace.simulate(*p)
    chisquare = np.sum((model - fit_workspace.spectrum.data) ** 2 / (err ** 2 + fit_workspace.spectrum.err ** 2))
    # chisq /= self.spectrum.data.size
    # print '\tReduced chisq =',chisq/self.spectrum.data.size
    return chisquare


def lnprior(p, bounds):
    in_bounds = True
    for npar, par in enumerate(p):
        if par < bounds[0][npar] or par > bounds[1][npar]:
            in_bounds = False
            break
    if in_bounds:
        return 0.0
    else:
        return -np.inf


def lnlike(p, fit_workspace):
    return -0.5 * chisq(p, fit_workspace)


def lnprob(p):
    lp = lnprior(p, fit_workspace.bounds)
    if not np.isfinite(lp):
        return -1e20
    return lp + lnlike(p, fit_workspace)


def gradient_descent(fit_workspace, params, epsilon, niter=10, fixed_params=None, xtol=1e-3, ftol=1e-3):
    tmp_params = np.copy(params)
    W = 1 / fit_workspace.spectrum.spectrogram_err.flatten() ** 2
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
        #    fit_workspace.plot_spectrogram_fit()
        residuals = (tmp_model - fit_workspace.spectrum.spectrogram).flatten()
        cost = np.sum((residuals ** 2) * W)
        print(f"cost={cost:.3f} ({tmp_model.size:d} pixels)")
        J = fit_workspace.jacobian(tmp_params, epsilon, fixed_params=fixed_params)
        # remove parameters with unexpected null Jacobian vectors
        for ip in range(J.shape[0]):
            if ip not in ipar:
                continue
            if np.all(J[ip] == np.zeros(J.shape[1])):
                ipar = np.delete(ipar, list(ipar).index(ip))
                print(f"Step {i}: {fit_workspace.input_labels[ip]} has a null Jacobian; parameter is fixed at its "
                      f"current value ({tmp_params[ip]}) in the following.")
        # remove fixed parameters
        J = J[ipar].T
        # algebra
        JT_W = J.T * W
        JT_W_J = JT_W @ J
        JT_W_R0 = JT_W @ residuals
        L = np.linalg.inv(np.linalg.cholesky(JT_W_J))
        inv_JT_W_J = L.T @ L
        if fit_workspace.live_fit:
            fit_workspace.cov = inv_JT_W_J
            fit_workspace.plot_correlation_matrix(ipar)
        dparams = - inv_JT_W_J @ JT_W_R0

        def line_search(alpha):
            tmp_params_2 = np.copy(tmp_params)
            tmp_params_2[ipar] = tmp_params[ipar] + alpha * dparams
            lbd, mod, err = fit_workspace.simulate(*tmp_params_2)
            return np.sum(((mod - fit_workspace.spectrum.spectrogram) / fit_workspace.spectrum.spectrogram_err) ** 2)

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
    ax[0].set_ylabel("$\chi^2$")
    ax[1].set_ylabel("Parameters")
    ax[0].grid()
    ax[1].set_xlabel("Iterations")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if parameters.SAVE:
        figname = fit_workspace.filename.replace(".fits", "_fitting.pdf")
        fit_workspace.my_logger.info(f"\n\tSave figure {figname}.")
        fig.savefig(figname, dpi=100, bbox_inches='tight')
    if parameters.DISPLAY:
        plt.show()

    fit_workspace.simulate(*fit_workspace.p)
    fit_workspace.live_fit = False
    fit_workspace.plot_spectrogram_fit()


def save_gradient_descent(fit_workspace, costs, params_table):
    iterations = np.arange(params_table.shape[0]).astype(int)
    t = np.zeros((params_table.shape[1]+2,params_table.shape[0]))
    t[0] = iterations
    t[1] = costs
    t[2:] = params_table.T
    h = 'iter,costs,'+ ','.join(fit_workspace.input_labels)
    output_filename = fit_workspace.filename.replace(".fits", "_fitting.txt")
    np.savetxt(output_filename, t.T, header=h, delimiter=",")
    fit_workspace.my_logger.info(f"\n\tSave gradient descent log {output_filename}.")


def run_minimisation(fit_workspace, method="newton"):
    my_logger = set_logger(__name__)
    bounds = fit_workspace.bounds

    nll = lambda p: -lnlike(p,  fit_workspace)

    # sim_134
    # guess = fit_workspace.p
    # truth sim_134
    # guess = np.array([1., 0.05, 300, 5, 0.03, 55.45, 0.0, 0.0, 0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955, -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901, 0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673, 528.3594585697788, 628.4966480821147, 12.438043546369354])
    guess = fit_workspace.p
    if method == "minimize":
        start = time.time()
        result = optimize.minimize(nll, fit_workspace.p, method='L-BFGS-B',
                                   options={'ftol': 1e-20, 'xtol': 1e-20, 'gtol': 1e-20, 'disp': True,
                                            'maxiter': 100000,
                                            'maxls': 50, 'maxcor': 30},
                                   bounds=bounds)
        fit_workspace.p = result['x']
        print(f"Minimize: total computation time: {time.time()-start}s")
        fit_workspace.simulate(*fit_workspace.p)
        fit_workspace.live_fit = False
        fit_workspace.plot_spectrogram_fit()

    elif method == "least_squares":
        start = time.time()
        x_scale = np.abs(guess)
        x_scale[x_scale == 0] = 0.1
        p = optimize.least_squares(spectrogram_weighted_res, guess, verbose=2, ftol=1e-6, x_scale=x_scale,
                                   diff_step=0.001, bounds=bounds.T, args=fit_workspace)
        fit_workspace.p = p.x  # m.np_values()
        print(f"Least_squares: total computation time: {time.time()-start}s")
        fit_workspace.simulate(*fit_workspace.p)
        fit_workspace.live_fit = False
        fit_workspace.plot_spectrogram_fit()
    elif method == "minuit":
        start = time.time()
        fit_workspace.simulation.fix_psf_cube = False
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
        print(f"Minuit: total computation time: {time.time()-start}s")
        fit_workspace.simulate(*fit_workspace.p)
        fit_workspace.live_fit = False
        fit_workspace.plot_spectrogram_fit()
    elif method == "newton":

        def bloc_gradient_descent(guess, epsilon, params_table, costs, fix, xtol, ftol, niter):
            fit_workspace.p, fit_workspace.cov, tmp_costs, tmp_params_table = gradient_descent(fit_workspace, guess, epsilon, niter=niter,
                                                                                 fixed_params=fix,
                                                                                 xtol=xtol, ftol=ftol)
            params_table = np.concatenate([params_table, tmp_params_table])
            costs = np.concatenate([costs, tmp_costs])
            ipar = np.array(np.where(np.array(fix).astype(int) == 0)[0])
            print_parameter_summary(fit_workspace.p[ipar], fit_workspace.cov,
                                    [fit_workspace.input_labels[ip] for ip in ipar])
            if True:
                #plot_psf_poly_params(fit_workspace.p[fit_workspace.psf_params_start_index:])
                plot_gradient_descent(fit_workspace, costs, params_table)
                fit_workspace.plot_correlation_matrix(ipar=ipar)
            return params_table, costs

        fit_workspace.simulation.fast_sim = True
        costs = np.array([chisq_spectrogram(guess, fit_workspace)])
        params_table = np.array([guess])
        start = time.time()
        epsilon = 1e-4 * guess
        epsilon[epsilon == 0] = 1e-3
        epsilon[0] = 1e-3 # A1
        epsilon[1] = 1e-4 # A2
        epsilon[2] = 1 # ozone
        epsilon[3] = 0.01 # pwv
        epsilon[4] = 0.0005 # aerosols
        epsilon[5] = 0.001 # DCCD
        epsilon[6] = 0.005 # shift_x
        my_logger.info(f"\n\tStart guess: {guess}")

        # fit trace
        fix = [True] * guess.size
        fix[0] = False  # A1
        fix[1] = True  # A2
        fix[6] = True  # x0
        fix[7] = True  # y0
        fit_workspace.simulation.fast_sim = True
        fix[fit_workspace.psf_params_start_index:fit_workspace.psf_params_start_index+3] = [False] * 3
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = bloc_gradient_descent(guess, epsilon, params_table, costs,
                                                    fix=fix, xtol=1e-2, ftol=1e-2, niter=20)

        # fit PSF
        guess = np.array(fit_workspace.p)
        fix = [True] * guess.size
        fix[0] = False  # A1
        fix[1] = False  # A2
        fit_workspace.simulation.fast_sim = True
        fix[fit_workspace.psf_params_start_index:] = [False] * (guess.size - fit_workspace.psf_params_start_index)
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = bloc_gradient_descent(guess, epsilon, params_table, costs,
                                                    fix=fix, xtol=1e-2, ftol=1e-2, niter=20)

        # fit dispersion
        guess = np.array(fit_workspace.p)
        fix = [True] * guess.size
        fix[0] = False
        fix[1] = False
        fix[5] = False # DCCD
        fix[6] = False # x0
        fit_workspace.simulation.fix_psf_cube = True
        fit_workspace.simulation.fast_sim = True
        params_table, costs = bloc_gradient_descent(guess, epsilon, params_table, costs,
                                                    fix=fix, xtol=1e-3, ftol=1e-2, niter=10)

        # fit all
        guess = np.array(fit_workspace.p)
        fit_workspace.simulation.fast_sim = False
        fix = [False] * guess.size
        fix[6] = False  # x0
        fix[7] = True  # y0
        parameters.SAVE = True
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = bloc_gradient_descent(guess, epsilon, params_table, costs,
                                                    fix=fix, xtol=1e-5, ftol=1e-5, niter=40)
        fit_workspace.save_parameters_summary()
        save_gradient_descent(fit_workspace, costs, params_table)
        print(f"Newton: total computation time: {time.time()-start}s")


def run_emcee(fit_workspace):
    my_logger = set_logger(__name__)
    fit_workspace.print_settings()
    nsamples = fit_workspace.nsteps
    p0 = fit_workspace.set_start()
    filename = fit_workspace.filename.replace("_spectrum.fits", "_emcee.h5")
    backend = emcee.backends.HDFBackend(filename)
    try:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(fit_workspace.nwalkers, fit_workspace.ndim, lnprob_spectrogram, args=(),
                                        pool=pool, backend=backend)
        print(f"Initial size: {backend.iteration}")
        if backend.iteration > 0:
            p0 = backend.get_last_sample()
        if nsamples - backend.iteration > 0:
            for i, result in enumerate(sampler.sample(p0, iterations=max(0, nsamples - backend.iteration))):
                if pool.is_master():
                    if (i + 1) % 100 == 0:
                        print("{0:5.1%}".format(float(i) / nsamples))
        pool.close()
    except ValueError:
        sampler = emcee.EnsembleSampler(fit_workspace.nwalkers, fit_workspace.ndim, lnprob_spectrogram, args=(),
                                        threads=4, backend=backend)
        print(f"Initial size: {backend.iteration}")
        if backend.iteration > 0:
            p0 = backend.get_last_sample()
        if nsamples - backend.iteration > 0:
            for i, result in enumerate(
                    sampler.sample(p0, iterations=max(0, nsamples - backend.iteration), progress=True)):
                if (i + 1) % 100 == 0:
                    print("{0:5.1%}".format(float(i) / nsamples))
    fit_workspace.chains = sampler.chain
    fit_workspace.lnprobs = sampler.lnprobability


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--debug", dest="debug", action="store_true",
                      help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="Enter verbose (print more stuff).", default=False)
    parser.add_option("-o", "--output_directory", dest="output_directory", default="./outputs/",
                      help="Write results in given output directory (default: ./outputs/).")
    (opts, args) = parser.parse_args()

    filename = 'outputs/reduc_20170530_134_spectrum.fits'
    filename = 'outputs/sim_20170530_134_spectrum.fits'
    atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')

    w = SpectrogramFitWorkspace(filename, atmgrid_filename=atmgrid_filename, nsteps=1000,
                                            burnin=2, nbins=10, verbose=1, plot=True, live_fit=False)
    run_minimisation(w, method="newton")
    # fit_workspace = w
    # run_emcee(w)
    # w.analyze_chains()

from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectractor.simulation.simulator import *
from spectractor.fit.statistics import *

from spectractor.parameters import FIT_WORKSPACE as fit_workspace

from iminuit import Minuit
from scipy import optimize

import emcee
from schwimmbad import MPIPool

import time


class FitWorkspace:

    def __init__(self, filename, atmgrid_filename="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False):
        self.my_logger = set_logger(self.__class__.__name__)
        self.filename = filename
        self.ndim = 0
        self.truth = None
        self.verbose = verbose
        self.plot = plot
        self.live_fit = live_fit
        self.p = np.array([])
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
        self.spectrum, self.telescope, self.disperser, self.target = SimulatorInit(filename)
        self.airmass = self.spectrum.header['AIRMASS']
        self.pressure = self.spectrum.header['OUTPRESS']
        self.temperature = self.spectrum.header['OUTTEMP']
        self.use_grid = False
        if atmgrid_filename == "":
            self.atmosphere = Atmosphere(self.airmass, self.pressure, self.temperature)
        else:
            self.use_grid = True
            self.atmosphere = AtmosphereGrid(filename, atmgrid_filename)
            if parameters.VERBOSE:
                self.my_logger.info(f'\n\tUse atmospheric grid models from file {atmgrid_filename}. ')
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
        self.start[self.start == 0] = 1e-5*np.random.uniform(0,1)
        return self.start

    def load_chains(self):
        self.chains = [[]]
        self.lnprobs = [[]]
        self.nbetafits = [[]]
        self.nsteps = 0
        tau = -1
        burnin = self.burnin
        thin = 1
        reader = emcee.backends.HDFBackend(self.emcee_filename)
        try:
            tau = reader.get_autocorr_time()
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
        except emcee.autocorr.AutocorrError:
            tau = -1
            burnin=self.burnin
            thin=1
        self.chains = reader.get_chain(discard=burnin, flat=False, thin=thin)
        self.lnprobs = reader.get_log_prob(discard=burnin, flat=False, thin=thin)
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

    def analyze_chains(self):
        self.load_chains()
        self.set_chain_validity()
        self.convergence_tests()
        self.build_flat_chains()
        likelihood = self.chain2likelihood()
        self.p = likelihood.mean_vec
        simulate_spectrogram(*self.p)
        if isinstance(self, SpectrumFitWorkspace):
            self.plot_fit()
        elif isinstance(self, SpectrogramFitWorkspace):
            self.plot_spectrogram_fit()
        figure_name = self.filename.replace('.fits', '_triangle.pdf')
        likelihood.triangle_plots(output_filename=figure_name)

    def save_bestfit_parameters(self, likelihood):
        pass

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
            #if np.mean(chisqs) < 1e5:
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
        ax[self.ndim, 0].set_ylim([self.global_average-5*self.global_std, self.global_average+5*self.global_std])
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
                for l in range(self.burnin+window, self.nsteps, window):
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
            step = max(1, (self.nsteps-self.burnin) // 20)
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
        figname = fit_workspace.filename.replace('.fits', '_bestfit.pdf')
        print(f'Save figure: {figname}')
        fig.savefig(figname, dpi=100)


class SpectrogramFitWorkspace(FitWorkspace):

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
        self.D = self.spectrum.header['D2CCD']
        self.psf_poly_params = self.spectrum.chromatic_psf.from_table_to_poly_params()
        self.psf_poly_params = self.psf_poly_params[
                               self.spectrum.spectrogram_Nx:-1]  # remove saturation (fixed parameter)
        self.psf_poly_params_labels = [f"a{k}" for k in range(self.psf_poly_params.size)]
        self.psf_poly_params_names = ["$a_{" + str(k) + "}$" for k in range(self.psf_poly_params.size)]
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
        self.input_labels = ["A1", "A2", "ozone", "PWV", "VAOD", r"D_CCD [mm]",
                             r"shift_x [pix]", r"shift_y [pix]"] + self.psf_poly_params_labels
        self.axis_names = ["$A_1$", "$A_2$", "ozone", "PWV", "VAOD", r"$D_{CCD}$ [mm]",
                           r"$\Delta_{\mathrm{x}}$ [pix]", r"$\Delta_{\mathrm{y}}$ [pix]"] + self.psf_poly_params_names
        self.bounds = np.concatenate([np.array([(0, 2), (0, 0.5), (0, 800), (0, 10), (0, 1),
                                                (50, 60), (-3, 3), (-3, 3)]),
                                      self.psf_poly_params_bounds])
        if atmgrid_filename != "":
            self.bounds[2] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.bounds[3] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.bounds[4] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.simulation = SpectrogramModel(self.spectrum, self.atmosphere, self.telescope, self.disperser)
        self.get_spectrogram_truth()

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
        if extent is not None:
            sub = np.where((lambdas > extent[0]) & (lambdas < extent[1]))[0]
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
        residuals = (self.spectrum.spectrogram - self.model)
        residuals_err = self.spectrum.spectrogram_err / self.model
        std = np.std(residuals[:, sub] / norm)
        plot_image_simple(ax[2, 0], data=residuals[:, sub] / norm, vmin=-5 * std, vmax=5 * std, title='Data-Model',
                          aspect='auto', cax=ax[2, 1], units='1/max(data)')
        ax[2, 0].set_title('Data-Model', fontsize=10, loc='center', color='white', y=0.8)
        ax[0, 0].set_xticks(ax[2, 0].get_xticks()[1:-1])
        ax[0, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[1, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[2, 1].get_yaxis().set_label_coords(3.5, 0.5)
        # ax[0, 0].get_yaxis().set_label_coords(-0.15, 0.6)
        # ax[2, 0].get_yaxis().set_label_coords(-0.15, 0.5)
        plot_image_simple(ax[1, 0], data=self.spectrum.spectrogram[:, sub] / norm, title='Data', aspect='auto',
                          cax=ax[1, 1], vmin=0, vmax=1, units='1/max(data)')
        ax[1, 0].set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
        # remove the underlying axes
        # for ax in ax[3, 1]:
        ax[3, 1].remove()
        ax[3, 0].plot(self.lambdas[sub], self.spectrum.spectrogram.sum(axis=0)[sub], label='Data')
        ax[3, 0].plot(self.lambdas[sub], self.model.sum(axis=0)[sub], label='Model')
        ax[3, 0].set_ylabel('Cross spectrum')
        ax[3, 0].set_xlabel('$\lambda$ [nm]')
        ax[3, 0].legend(fontsize=7)
        ax[3, 0].grid(True)

    def jacobian(self, params, epsilon, fixed_params=None):
        # print("start")
        start = time.time()
        lambdas, model, model_err = simulate_spectrogram(*params)
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
            tmp_lambdas, tmp_model, tmp_model_err = simulate_spectrogram(*tmp_p)
            J[ip] = (tmp_model.flatten() - model) / epsilon[ip]
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
        figname = self.filename.replace('.fits', '_bestfit.pdf')
        self.my_logger.info(f'\n\tSave figure: {figname}')
        fig.savefig(figname, dpi=100)


plot_counter = 0


def simulate_spectrogram(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, *psf_poly_params):
    global plot_counter
    # print('tttt', A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, psf_poly_params)
    fit_workspace.simulation.fix_psf_cube = False
    if np.all(np.isclose(psf_poly_params, fit_workspace.p[fit_workspace.psf_params_start_index:], rtol=1e-6)):
        fit_workspace.simulation.fix_psf_cube = True
    lambdas, model, model_err = \
        fit_workspace.simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, psf_poly_params)
    fit_workspace.p = np.array([A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y] + list(psf_poly_params))
    fit_workspace.lambdas = lambdas
    fit_workspace.model = model
    fit_workspace.model_err = model_err
    if fit_workspace.live_fit and (plot_counter % 30) == 0:
        fit_workspace.plot_spectrogram_fit()
    plot_counter += 1
    return lambdas, model, model_err


def spectrogram_weighted_res(p):
    lambdas, model, model_err = simulate_spectrogram(*p)
    res = ((model - fit_workspace.spectrum.spectrogram) / fit_workspace.spectrum.spectrogram_err).flatten()
    return res


def chisq_spectrogram(p):
    res = spectrogram_weighted_res(p)
    chisq = np.sum(res ** 2)
    return chisq


def lnlike_spectrogram(p):
    return -0.5 * chisq_spectrogram(p)


def lnprob_spectrogram(p):
    lp = lnprior(p, fit_workspace.bounds)
    if not np.isfinite(lp):
        return -1e20
    return lp + lnlike_spectrogram(p)


def simulate(A1, A2, ozone, pwv, aerosols, reso, D, shift):
    lambdas, model, model_err = fit_workspace.simulation.simulate(A1, A2, ozone, pwv, aerosols, reso, D, shift)
    # if fit_workspace.live_fit:
    #    fit_workspace.plot_fit()
    fit_workspace.model = model
    fit_workspace.model_err = model_err
    return model(fit_workspace.spectrum.lambdas), model_err(fit_workspace.spectrum.lambdas)


def chisq(p):
    model, err = simulate(*p)
    chisq = np.sum((model - fit_workspace.spectrum.data) ** 2 / (err ** 2 + fit_workspace.spectrum.err ** 2))
    # chisq /= self.spectrum.data.size
    # print '\tReduced chisq =',chisq/self.spectrum.data.size
    return chisq


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


def lnlike(p):
    return -0.5 * chisq(p)


def lnprob(p):
    lp = lnprior(p, fit_workspace.bounds)
    if not np.isfinite(lp):
        return -1e20
    return lp + lnlike(p)


def gradient_descent(params, epsilon, niter=10, fixed_params=None, tol = 1e-3):
    tmp_params = np.copy(params)
    W = 1 / fit_workspace.spectrum.spectrogram_err.flatten() ** 2
    ipar = np.arange(params.size)
    if fixed_params is not None:
        ipar = np.array(np.where(np.array(fixed_params).astype(int) == 0)[0])
    costs = []
    params_table = []
    for i in range(niter):
        print(f"start iteration={i}", end=' ')  # , tmp_params)
        start = time.time()
        tmp_lambdas, tmp_model, tmp_model_err = simulate_spectrogram(*tmp_params)
        # if fit_workspace.live_fit:
        #    fit_workspace.plot_spectrogram_fit()
        residuals = (tmp_model - fit_workspace.spectrum.spectrogram).flatten()
        cost = np.sum((residuals ** 2) * W)
        print(f"cost={cost:.3f}")
        J = fit_workspace.jacobian(tmp_params, epsilon, fixed_params=fixed_params)
        J = J[ipar].T  # remove unfixed parameter (null Jacobian vectors)
        JT_W = J.T * W
        JT_W_J = JT_W @ J
        if fit_workspace.live_fit:
            plt.close()
            fig = plt.figure()
            im = plt.imshow(JT_W_J)
            plt.title("$J^T W J$")
            plt.xticks(np.arange(ipar.size), [fit_workspace.axis_names[ip] for ip in ipar], rotation='vertical',
                       fontsize=11)
            plt.yticks(np.arange(ipar.size), [fit_workspace.axis_names[ip] for ip in ipar], fontsize=11)
            cbar = fig.colorbar(im)
            cbar.ax.tick_params(labelsize=9)
            fig.tight_layout()
            plt.draw()
            plt.pause(1e-8)
        JT_W_R0 = JT_W @ residuals
        L = np.linalg.inv(np.linalg.cholesky(JT_W_J))
        inv_JT_W_J = L.T @ L
        dparams = - inv_JT_W_J @ JT_W_R0

        def line_search(alpha):
            tmp_params_2 = np.copy(tmp_params)
            tmp_params_2[ipar] = tmp_params[ipar] + alpha * dparams
            lbd, mod, err = simulate_spectrogram(*tmp_params_2)
            return np.sum(((mod - fit_workspace.spectrum.spectrogram) / fit_workspace.spectrum.spectrogram_err) ** 2)

        # tol parameter acts on alpha (not func)
        alpha_min, fval, iter, funcalls = optimize.brent(line_search, full_output=True, tol=1e-2)
        print(f"\talpha_min={alpha_min:.3g} iter={iter} funcalls={funcalls}")
        tmp_params[ipar] += alpha_min * dparams
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
        if np.abs(alpha_min) < tol or (len(costs) > 1 and np.abs(costs[-2] - fval) / fval < tol):
            break
    plt.close()
    return tmp_params, np.array(costs), np.array(params_table)


def plot_gradient_descent(costs, params_table):
    fig, ax = plt.subplots(2, 1, figsize=(10,6), sharex="all")
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
    plt.show()

    if isinstance(fit_workspace, SpectrumFitWorkspace):
        simulate(*fit_workspace.p)
    elif isinstance(fit_workspace, SpectrogramFitWorkspace):
        simulate_spectrogram(*fit_workspace.p)
    fit_workspace.live_fit = False
    fit_workspace.plot_spectrogram_fit()


def run_minimisation(method="newton"):
    my_logger = set_logger(__name__)
    bounds = fit_workspace.bounds

    # sim_134
    # guess = fit_workspace.p
    # truth sim_134
    # guess = np.array([1., 0.01, 300, 5, 0.03, 55.45, 0.0, 0.0, 0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955, -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901, 0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673, 528.3594585697788, 628.4966480821147, 12.438043546369354])
    guess = fit_workspace.p

    if method == "minimize":
        if isinstance(fit_workspace, SpectrumFitWorkspace):
            nll = lambda p: -lnlike(p)
        elif isinstance(fit_workspace, SpectrogramFitWorkspace):
            nll = lambda p: -lnlike_spectrogram(p)
        else:
            my_logger.error(f'\n\tUnknown fit_workspace class type {type(fit_workspace)}.\n'
                            f'Choose either "SpectrumFitWorkspace" of "SpectrogramFitWorkspace"')
            sys.exit()
        result = optimize.minimize(nll, fit_workspace.p, method='L-BFGS-B',
                                   options={'ftol': 1e-20, 'xtol': 1e-20, 'gtol': 1e-20, 'disp': True,
                                            'maxiter': 100000,
                                            'maxls': 50, 'maxcor': 30},
                                   bounds=bounds)
        fit_workspace.p = result['x']
        if isinstance(fit_workspace, SpectrumFitWorkspace):
            simulate(*fit_workspace.p)
        elif isinstance(fit_workspace, SpectrogramFitWorkspace):
            simulate_spectrogram(*fit_workspace.p)
        fit_workspace.live_fit = False
        fit_workspace.plot_spectrogram_fit()

    elif method == "minuit":
        x_scale = np.abs(guess)
        x_scale[x_scale == 0] = 0.1
        p = optimize.least_squares(spectrogram_weighted_res, guess, verbose=2, ftol=1e-6, x_scale=x_scale,
                                   diff_step=0.001, bounds=bounds.T)
        fit_workspace.p = p.x  # m.np_values()
        if isinstance(fit_workspace, SpectrumFitWorkspace):
            simulate(*fit_workspace.p)
        elif isinstance(fit_workspace, SpectrogramFitWorkspace):
            simulate_spectrogram(*fit_workspace.p)
        fit_workspace.live_fit = False
        fit_workspace.plot_spectrogram_fit()

    elif method == "newton":
        costs = np.array([chisq_spectrogram(guess)])
        params_table = np.array([guess])

        epsilon = 1e-3 * guess
        epsilon[epsilon == 0] = 1e-3

        # fit shape
        fix = [True] * guess.size
        fix[0] = False  # A1
        fix[1] = False  # A2
        fix[7] = False  # y0
        fix[fit_workspace.psf_params_start_index:] = [False] * (guess.size - fit_workspace.psf_params_start_index)
        fit_workspace.simulation.fix_psf_cube = False
        fit_workspace.p, tmp_costs, tmp_params_table = gradient_descent(guess, epsilon, niter=20, fixed_params=fix, tol=1e-2)
        params_table = np.concatenate([params_table, tmp_params_table])
        costs = np.concatenate([costs, tmp_costs])
        print(fit_workspace.p)
        #plot_gradient_descent(costs, params_table)

        # fit dispersion
        guess = np.array(fit_workspace.p)
        # guess =  np.array([ 9.57341110e-01,  3.21149660e-02,  3.00000000e+02,  3.00000000e+00,
        #   3.00000000e-02,  5.51650191e+01,  1.99999618e+00,  0.00000000e+00,
        #   1.76245582e+00,  5.82214509e-01,  2.27827769e-01,
        #   2.12741265e+00, -1.32583056e+00,  1.01407834e+00,  1.72903242e+00,
        #  -6.72668605e-01,  4.23255049e-01, -3.17253279e-03, -3.44687146e-03,
        #  -3.59027314e-03,  2.14896817e+01, -5.76766492e+00,  1.07628816e+00])
        fix = [True] * guess.size
        fix[0] = False
        fix[1] = False
        fix[5:8] = [False] * 3  # dispersion params
        fit_workspace.simulation.fix_psf_cube = True
        fit_workspace.p, tmp_costs, tmp_params_table = gradient_descent(guess, epsilon, niter=2, fixed_params=fix, tol=1e-3)
        params_table = np.concatenate([params_table, tmp_params_table])
        costs = np.concatenate([costs, tmp_costs])
        print(fit_workspace.p)
        #plot_gradient_descent(costs, params_table)

        # fit all
        guess = np.array(fit_workspace.p)
        # guess = np.array([ 9.70636227e-01,  2.15812703e-02,  3.00000000e+02,  3.00000000e+00,
        #  3.00000000e-02,  5.54560254e+01,  1.51615293e+00,  0.00000000e+00,
        #  1.76245582e+00,  5.82214509e-01,  2.27827769e-01,
        #  2.12741265e+00, -1.32583056e+00,  1.01407834e+00,  1.72903242e+00,
        # -6.72668605e-01,  4.23255049e-01, -3.17253279e-03, -3.44687146e-03,
        # -3.59027314e-03,  2.14896817e+01, -5.76766492e+00,,  1.07628816e+00])
        fix = [False] * guess.size
        fit_workspace.simulation.fix_psf_cube = False
        fit_workspace.p, tmp_costs, tmp_params_table = gradient_descent(guess, epsilon, niter=20, fixed_params=fix, tol=1e-5)
        params_table = np.concatenate([params_table, tmp_params_table])
        costs = np.concatenate([costs, tmp_costs])
        print(fit_workspace.p)
        plot_gradient_descent(costs, params_table)


def run_emcee():
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
        sampler = emcee.EnsembleSampler(fit_workspace.nwalkers, fit_workspace.ndim, lnlike_spectrogram, args=(),
                                        pool=pool, backend=backend)
        print(f"Initial size: {backend.iteration}")
        if nsamples-backend.iteration > 0:
            for i, result in enumerate(sampler.sample(p0, iterations=max(0, nsamples-backend.iteration))):
                if pool.is_master():
                    if (i + 1) % 100 == 0:
                        print("{0:5.1%}".format(float(i) / nsamples))
        pool.close()
    except ValueError:
        sampler = emcee.EnsembleSampler(fit_workspace.nwalkers, fit_workspace.ndim, lnlike_spectrogram, args=(),
                                        threads=4, backend=backend)
        print(f"Initial size: {backend.iteration}")
        if nsamples-backend.iteration > 0:
            for i, result in enumerate(sampler.sample(p0, iterations=max(0, nsamples-backend.iteration), progress=True)):
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

    filename = 'outputs/reduc_20170530_130_spectrum.fits'
    filename = 'outputs/sim_20170530_134_spectrum.fits'
    atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')

    fit_workspace = SpectrogramFitWorkspace(filename, atmgrid_filename=atmgrid_filename, nsteps=20,
                                            burnin=1, nbins=10, verbose=1, plot=True, live_fit=False)
    run_minimisation()
    run_emcee()
    fit_workspace.analyze_chains()

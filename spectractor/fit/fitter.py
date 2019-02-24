from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectractor.simulation.simulator import *
from spectractor.fit.statistics import *

from spectractor.parameters import FIT_WORKSPACE as fit_workspace

from iminuit import Minuit

import emcee
from emcee.utils import MPIPool


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
            [np.random.uniform(self.p[i] - 0.02 * (self.bounds[1][i] - self.bounds[0][i]),
                               self.p[i] + 0.02 * (self.bounds[1][i] - self.bounds[0][i]),
                               self.nwalkers)
             for i in range(self.ndim)]).T
        return self.start

    def build_flat_chains(self):
        self.flat_chains = self.chains[self.valid_chains, self.burnin:, :].reshape((-1, self.ndim))
        return self.flat_chains

    def analyze_chains(self):
        self.set_chain_validity()
        self.convergence_tests()
        self.build_flat_chains()
        likelihood = self.chain2likelihood()
        self.p = likelihood.mean_vec
        simulate(*self.p)
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
            chains = self.chains[walker_index, self.burnin:, :]
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
            chisqs = -2 * self.lnprobs[k, self.burnin:]
            if np.mean(chisqs) < 1e5:
                chisq_averages.append(np.mean(chisqs))
                chisq_std.append(np.std(chisqs))
        global_average = np.mean(chisq_averages)
        global_std = np.mean(chisq_std)
        self.valid_chains = [False] * self.nwalkers
        for k in nchains:
            chisqs = -2 * self.lnprobs[k, self.burnin:]
            chisq_average = np.mean(chisqs)
            chisq_std = np.std(chisqs)
            if chisq_average > 3 * global_std + global_average:
                self.valid_chains[k] = False
            elif chisq_std < 0.1 * global_std:
                self.valid_chains[k] = False
            else:
                self.valid_chains[k] = True
        return self.valid_chains

    def convergence_tests(self):
        chains = self.chains[:, self.burnin:, :]  # .reshape((-1, self.ndim))
        nchains = [k for k in range(self.nwalkers)]
        fig, ax = plt.subplots(self.ndim + 1, 2, figsize=(16, 7), sharex='all')
        fontsize = 8
        steps = np.arange(self.burnin, self.nsteps)
        # Chi2 vs Index
        print("Chisq statistics:")
        for k in nchains:
            chisqs = -2 * self.lnprobs[k, self.burnin:]
            text = f"\tWalker {k:d}: {float(np.mean(chisqs)):.3f} +/- {float(np.std(chisqs)):.3f}"
            if not self.valid_chains[k]:
                text += " -> excluded"
                ax[self.ndim, 0].plot(steps, chisqs, c='0.5', linestyle='--')
            else:
                ax[self.ndim, 0].plot(steps, chisqs)
            print(text)
        global_average = np.mean(-2 * self.lnprobs[self.valid_chains, self.burnin:])
        global_std = np.std(-2 * self.lnprobs[self.valid_chains, self.burnin:])
        ax[self.ndim, 0].set_ylim([global_average - 5 * global_std, global_average + 5 * global_std])
        # Parameter vs Index
        print("Computing Parameter vs Index plots...")
        for i in range(self.ndim):
            ax[i, 0].set_ylabel(self.axis_names[i], fontsize=fontsize)
            for k in nchains:
                if self.valid_chains[k]:
                    ax[i, 0].plot(steps, chains[k, :, i])
                else:
                    ax[i, 0].plot(steps, chains[k, :, i], c='0.5', linestyle='--')
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
        # Gelman-Rubin test
        if len(nchains) > 1:
            step = max(1, (self.nsteps - self.burnin) // 20)
            gelman_rubins = []
            print(f'Gelman-Rubin tests (burnin={self.burnin:d}, step={step:d}, nsteps={self.nsteps:d}):')
            for i in range(self.ndim):
                Rs = []
                lens = []
                for l in range(self.burnin + step, self.nsteps, step):
                    chain_averages = []
                    chain_variances = []
                    global_average = np.mean(self.chains[self.valid_chains, self.burnin:l, i])
                    for k in nchains:
                        if not self.valid_chains[k]:
                            continue
                        chain_averages.append(np.mean(self.chains[k, self.burnin:l, i]))
                        chain_variances.append(np.var(self.chains[k, self.burnin:l, i], ddof=1))
                    W = np.mean(chain_variances)
                    B = 0
                    for n in range(len(chain_averages)):
                        B += (chain_averages[n] - global_average) ** 2
                    B *= ((l + 1) / (len(chain_averages) - 1))
                    R = (W * l / (l + 1) + B / (l + 1) * (len(chain_averages) + 1) / len(chain_averages)) / W
                    Rs.append(R - 1)
                    lens.append(l)
                gelman_rubins.append(Rs[-1])
                print(f'\t{self.input_labels[i]}: R-1 = {Rs[-1]:.3f} (l = {lens[-1] - 1:d})')
                ax[i, 1].plot(lens, Rs, lw=2, label=self.axis_names[i])
                ax[i, 1].axhline(0.03, c='k', linestyle='--')
                ax[i, 1].set_xlabel('Walker length', fontsize=fontsize)
                ax[i, 1].set_ylabel('$R-1$', fontsize=fontsize)
                ax[i, 1].set_ylim(0, 0.6)
                # ax[self.dim, 3].legend(loc='best', ncol=2, fontsize=10)
        fig.tight_layout()
        plt.subplots_adjust(hspace=0)
        if parameters.DISPLAY and parameters.VERBOSE:
            plt.show()
        figure_name = self.filename.replace('.fits', '_convergence.pdf')
        print(f'Save figure: {figure_name}')
        fig.savefig(figure_name, dpi=100)
        output_file = self.filename.replace('.fits', '_convergence.txt')
        print(f'Save: {output_file}')
        txt = ''
        for i in range(self.ndim):
            txt += f'{self.input_labels[i]} {gelman_rubins[i]}\n'
        f = open(output_file, 'w')
        f.write(txt)
        f.close()

    def print_settings(self):
        print('************************************')
        print(f"Input file: {self.filename}\nWalkers: {self.nwalkers}\t Steps: {self.nsteps}")
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
        self.bounds = ((0, 2), (0, 0.5), (0, 800), (0, 10), (0, 1), (1, 10), (50, 60), (-20, 20))
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
        self.psf_poly_params = self.psf_poly_params[self.spectrum.spectrogram_Nx:]
        self.psf_poly_params_labels = [f"a{k}" for k in range(self.psf_poly_params.size)]
        self.psf_poly_params_names = [f"$a_{k}$" for k in range(self.psf_poly_params.size)]
        self.psf_poly_params_bounds = self.spectrum.chromatic_psf.set_bounds(data=None)
        self.shift_x = self.spectrum.header['PIXSHIFT']
        self.shift_y = 0.
        self.shift_t = 0.
        self.angle = self.spectrum.rotation_angle
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols,
                           self.D, self.shift_x, self.shift_y, self.shift_t, self.angle] + list(self.psf_poly_params))
        self.ndim = self.p.size
        self.input_labels = ["A1", "A2", "ozone", "PWV", "VAOD", r"D_CCD [mm]",
                             r"shift_x [pix]", r"shift_y [pix]", r"shift_T [nm]", r"angle"] + self.psf_poly_params_labels
        self.axis_names = ["$A_1$", "$A_2$", "ozone", "PWV", "VAOD", r"$D_{CCD}$ [mm]",
                           r"$\Delta_{\mathrm{x}}$ [pix]", r"$\Delta_{\mathrm{y}}$ [pix]", r"$\Delta_{\mathrm{T}}$ [nm]", r"\theta"] + \
                          self.psf_poly_params_names
        self.bounds = np.concatenate([np.array([(0, 2), (0, 0.5), (0, 800), (0, 10), (0, 1),
                                                (50, 60), (-2, 2), (-2, 2), (-40, 40), (-3, 3)]), self.psf_poly_params_bounds])
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

    def plot_spectrogram_comparison_simple(self, ax, title='', extent=None, dispersion=False):
        lambdas = self.spectrum.lambdas
        sub = np.where((lambdas > parameters.LAMBDA_MIN) & (lambdas < parameters.LAMBDA_MAX))[0]
        if extent is not None:
            sub = np.where((lambdas > extent[0]) & (lambdas < extent[1]))[0]
        plot_image_simple(ax[0, 0], data=self.model[:, sub], aspect='auto', cax=ax[0, 1])
        if dispersion:
            ax[0, 0].scatter(self.spectrum.chromatic_psf.table['Dx'][sub[2:-3]] + self.spectrum.spectrogram_x0 - sub[0],
                             self.spectrum.chromatic_psf.table['Dy'][sub[2:-3]] + self.spectrum.spectrogram_y0,
                             cmap=from_lambda_to_colormap(self.lambdas[sub[2:-3]]), edgecolors='None',
                             c=self.lambdas[sub[2:-3]],
                             label='', marker='o', s=3)

        # p0 = ax.plot(lambdas, self.model(lambdas), label='model')
        # # ax.plot(self.lambdas, self.model_noconv, label='before conv')
        if title != '':
            ax[0, 0].set_title(title, fontsize=10, loc='center', color='white', y=0.8)
        residuals = (self.spectrum.spectrogram - self.model)
        residuals_err = self.spectrum.spectrogram_err / self.model
        std = np.std(residuals[:, sub])
        plot_image_simple(ax[2, 0], data=residuals[:, sub], vmin=-5 * std, vmax=5 * std, title='Data-Model',
                          aspect='auto', cax=ax[2, 1])
        ax[2, 0].set_title('Data-Model',  fontsize=10, loc='center', color='white', y=0.8)
        ax[0, 0].set_xticks(ax[2, 0].get_xticks()[1:-1])
        ax[0, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[1, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[2, 1].get_yaxis().set_label_coords(3.5, 0.5)
        # ax[0, 0].get_yaxis().set_label_coords(-0.15, 0.6)
        # ax[2, 0].get_yaxis().set_label_coords(-0.15, 0.5)
        plot_image_simple(ax[1, 0], data=self.spectrum.spectrogram[:, sub], title='Data', aspect='auto',
                          cax=ax[1, 1])
        ax[1, 0].set_title('Data',  fontsize=10, loc='center', color='white', y=0.8)
        # remove the underlying axes
        #for ax in ax[3, 1]:
        ax[3, 1].remove()
        ax[3, 0].plot(self.lambdas[sub], self.spectrum.spectrogram.sum(axis=0)[sub], label='Data')
        ax[3, 0].plot(self.lambdas[sub], self.model.sum(axis=0)[sub], label='Model')
        ax[3, 0].set_ylabel('Cross spectrum')
        ax[3, 0].set_xlabel('$\lambda$ [nm]')
        ax[3, 0].legend(fontsize=7)
        ax[3, 0].grid(True)

    def plot_spectrogram_fit(self):
        """
        Examples
        --------
        >>> filename = 'outputs/reduc_20170530_130_spectrum.fits'
        >>> atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')
        >>> fit_workspace = SpectrogramFitWorkspace(filename, atmgrid_filename=atmgrid_filename, nwalkers=28, nsteps=20000, burnin=10000,
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

        A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, angle, *psf = self.p
        plt.suptitle(f'A1={A1:.3f}, A2={A2:.3f}, PWV={pwv:.3f}, OZ={ozone:.3g}, VAOD={aerosols:.3f}, '
                     f'D={D:.2f}mm, shift_x={shift_x:.2f}pix, shift_y={shift_y:.2f}pix, shift_t={shift_t:.2f}nm, angle={angle:.2f} ')
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


def simulate_spectrogram(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, angle, *psf_poly_params):
    print('tttt', A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, angle, psf_poly_params)
    fit_workspace.simulation.fix_psf_cube = False
    if np.all(np.isclose(psf_poly_params, fit_workspace.p[10:], rtol=1e-6)):
        fit_workspace.simulation.fix_psf_cube = True
    lambdas, model, model_err = \
        fit_workspace.simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, angle, psf_poly_params)
    fit_workspace.p = [A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, angle] + list(psf_poly_params)
    fit_workspace.lambdas = lambdas
    fit_workspace.model = model
    fit_workspace.model_err = model_err
    if fit_workspace.live_fit:
        fit_workspace.plot_spectrogram_fit()
    return lambdas, model, model_err


def chisq_spectrogram(p):
    lambdas, model, model_err = simulate_spectrogram(*p)
    chisq = np.sum((model - fit_workspace.spectrum.spectrogram) ** 2 / (
            model_err ** 2 + fit_workspace.spectrum.spectrogram_err ** 2))
    print('cc', chisq)
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


def sort_on_runtime(p, depth_index=5):
    p = np.atleast_2d(p)
    idx = np.argsort(p[:, depth_index])[::-1]
    return p[idx], idx


def run_minimisation():
    my_logger = set_logger(__name__)
    bounds = fit_workspace.bounds
    if isinstance(fit_workspace, SpectrumFitWorkspace):
        nll = lambda p: -lnlike(p)
    elif isinstance(fit_workspace, SpectrogramFitWorkspace):
        nll = lambda p: -lnlike_spectrogram(p)
    else:
        my_logger.error(f'\n\tUnknown fit_workspace class type {type(fit_workspace)}.\n'
                        f'Choose either "SpectrumFitWorkspace" of "SpectrogramFitWorkspace"')
        sys.exit()
    # result = minimize(nll, fit_workspace.p, method='L-BFGS-B',
    #                   options={'ftol': 1e-20, 'xtol': 1e-20, 'gtol': 1e-20, 'disp': True, 'maxiter': 100000,
    #                            'maxls': 50, 'maxcor': 30},
    #                   bounds=bounds)
    # fit_workspace.p = result['x']
    # for _130.fits
    # guess = np.array(
    #     [1.0221355184586634, 0.05, 300.0, 3.0, 0.03, 55.45, 0.0, 0.0, -13.9, -0.6305218866359683, 0.9697783937167165,
    #      -0.23418568011585317, 0.33884422382705437, 6.491300580087118, 7.23813399139167, 3.4442522038396097,
    #      3.2040421136282338, 2.3246058070491347, 0.8560985887811235, -0.24436185918630957, -0.24696445586018434,
    #      -0.0028070102515497844, 3.028241966129593, 2.8582943432553085, -0.3948830196355327, 999.9999999999964])
    # guess = np.array(
    #     [1.0739594429903858, 0.05, 300.0, 3.0, 0.03, 55.45, 0.0, 0.0, -13.9, -0.6305218866359683, 0.9697783937167165,
    #     -0.23418568011585317, 0.33884422382705437, 6.446415469477438,
    #                          7.729767353344003, 3.398518168281213, 3.246584721057184, 2.1288614628936395,
    #                          0.878365118777165, -0.22805156531233226, -0.26791683345602524, -0.03629293223130861,
    #                          3.570264949549503, 2.498371135708351, -0.29661704517488957, 999.9999999999964])
    # guess = np.array([1.1999374575144732, 0.05, 300.0, 3.0, 0.03, 55.082071522198056, 0.0, 0.0, -13.9, -0.6305218866359683, 1.0395299151149904, 0.1275725960085727, 0.5280831133684005, 6.235160344247447, 6.784917251773036, 4.2369845643653115, 3.448993463233748, 1.6421707683237179, 1.9074156099667425, -0.09281914503246441, -0.27894475572169325, 0.12240788960743397, 3.9261326346235434, 2.7938537263795604, 0.43486802733644325, 999.9999999999964])
    # guess = np.array([1.2045483348683912, 0.05, 300.0, 3.0, 0.03, 55.07937441283125, 0.0, 0.0, -13.9, -0.6305218866359683, 1.042808541712434, 0.13530636527489748, 0.5503986690804731, 5.33368295428954, 5.0841705471792755, 3.7713637509390625, 3.063271361527012, 0.8486787059487655, 1.9719045741976886, -0.05858775474963545, -0.16956856815728952, 0.06644620580104998, 4.319269043951657, 2.866341144875256, -0.08709501136597646, 520.5744613957937])
    # guess = np.array([1.2045483348683912, 0.05, 300.0, 3.0, 0.03, 55.07937441283125, 0.0, 0.0, -13.9, -0.6305218866359683, 1.042808541712434, 0.13530636527489748, 0.5503986690804731, 5.33368295428954, 5.0841705471792755, 3.7713637509390625, 3.063271361527012, 0.8486787059487655, 1.9719045741976886, -0.05858775474963545, -0.16956856815728952, 0.06644620580104998, 4.319269043951657, 2.866341144875256, -0.08709501136597646, 520.5744613957937])
    # reduc_134
    # guess = np.array([1.1783004945625857, 0.028771175829755774, 300.0, 3.5788603010605713, 0.03, 56.31035877782502, 0.001324080414817609, 0.0, -27.69406155413207, -1.542396612306326, 0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955, -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901, 0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673, 528.3594585697788, 628.4966480821147, 12.438043546369354, 499.99999999999835])
    # sim_134
    guess = np.array([1.0200367670106933, 0.0363572740671097, 237.88100163825928, 9.99985583070645, 0.04881224044620258, 55.19390495103511, -1.18481117217888, -0.15902033562208118, 1.720820828790302, -1.681859821533847, 1.4118025344599496, 0.5302750543501547, 0.05486954668604792, 3.72604016525854, -1.296968852142376, 1.0703672456033204, 3.0205592886126813, -0.06300056342451373, 0.9697073189216313, -0.37263561988400884, -0.1330268748184035, -0.05678495440244433, 2.1345811085002606, -0.9857266387263521, -0.07773569454909918, 499.99999999999835])
    guess = np.array([1.0177449875857436, 0.03273899280095122, 244.28900108972238, 9.99985328071115, 0.05574572930987309, 55.2113222597155, -1.9999934481655455, -0.16588415261849976, 1.3518973026372976, -1.681859821533847, 1.418683896878165, 0.5533962728465286, 0.06744270896451146, 3.7242663590096523, -1.2946316193457752, 1.0714560991986213, 3.027495524307809, -0.07472328983133317, 0.9555378331632957, -0.3653217756888738, -0.1428741221990274, -0.06995171218400704, 2.1270602543705697, -0.9765558807737923, -0.05544098513783982, 499.99999999999835])
    guess = np.array([1.008504591340807, 0.01608223007207482, 360.55203093205034, 9.999396124992284, 0.07408687185771112, 55.278549773179, -1.9999880367300542, -0.19831397139122675 ,1.731835570968407, -1.681859821533847, 1.4512075119943928, 0.6298897489162698, 0.1715121918150413, 3.6949934638991815, -1.2787266139872775, 1.1050818585131204, 3.0678599353279385, -0.141000223367532, 0.8878220330543275, -0.34365173970314067, -0.17666624793314062, -0.0778543688708959, 2.0961259559036165, -0.912009316148657, 0.058847183286068916, 499.99999999999835])
    guess = np.array([1.021105038594662, 0.004522557159414742, 399.1804795838472, 7.151266092418873, 0.06532628682748798, 55.28129937643125, -1.5796083498014732, -0.214022484422576, 1.5532729238980494, -1.681859821533847, 1.4669017969146076, 0.6167895273613202, 0.2240612421558033, 3.636185977901238, -1.252269816889516, 1.2319551727230675, 3.1212613617244513, -0.15132200936139598, 0.8692478333014879, -0.34168336262027477, -0.21066308766825492, 0.08067864362906049, 1.9588266937720888, -0.7975072607096434, 0.32389616177447655, 499.99999999999835])
    # guess = fit_workspace.p
    fix = [True] * guess.size
    fix[0] = False # A1
    fix[1] = False # A2
    fix[2:5] = [False, False, False] # LIBRADTRAN
    # fix[3] = False
    fix[5] = False #DCCD
    fix[5:8] = [False, False, False] #DCCD
    # fix[10:13] = [False, False, False] # centers
    fix[10:] = [False] * (guess.size-10)
    fix[8] = False # shift_t
    fix[-1] = True
    fit_workspace.simulation.fix_psf_cube = False
    error = 0.1 * np.abs(guess) * np.ones_like(guess)
    z = np.where(np.isclose(error,0.0,1e-6))
    error[z] = 1.
    m = Minuit.from_array_func(fcn=nll, start=guess, error=error, errordef=1,
                               fix=fix, print_level=2, limit=bounds)
    m.tol = 10
    m.migrad()
    fit_workspace.p = m.np_values()
    print(fit_workspace.p)
    if isinstance(fit_workspace, SpectrumFitWorkspace):
        simulate(*fit_workspace.p)
    elif isinstance(fit_workspace, SpectrogramFitWorkspace):
        simulate_spectrogram(*fit_workspace.p)
    fit_workspace.live_fit = False
    fit_workspace.plot_spectrogram_fit()


def run_emcee(w):
    global fit_workspace
    my_logger = set_logger(__name__)
    fit_workspace = w
    fit_workspace.print_settings()
    nsamples = fit_workspace.nsteps
    p0 = fit_workspace.set_start()
    try:
        pool = MPIPool(loadbalance=True, debug=False)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(fit_workspace.nwalkers, fit_workspace.ndim, lnprob, args=(), pool=pool,
                                        runtime_sortingfn=sort_on_runtime)
        for i, result in enumerate(sampler.sample(p0, iterations=max(0, nsamples), storechain=True)):
            if pool.is_master():
                if (i + 1) % 100 == 0:
                    print("{0:5.1%}".format(float(i) / nsamples))
        pool.close()
    except ValueError:
        sampler = emcee.EnsembleSampler(fit_workspace.nwalkers, fit_workspace.ndim, lnprob, args=(), threads=8,
                                        runtime_sortingfn=sort_on_runtime)
        for i, result in enumerate(sampler.sample(p0, iterations=max(0, nsamples), storechain=True)):
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

    fit_workspace = SpectrogramFitWorkspace(filename, atmgrid_filename=atmgrid_filename, nwalkers=28, nsteps=20000,
                                            burnin=10000,
                                            nbins=10, verbose=1, plot=True, live_fit=True)
    run_minimisation()
    # run_emcee()
    # fit_workspace.analyze_chains()

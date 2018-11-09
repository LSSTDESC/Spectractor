from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectractor.simulation.simulator import *
from spectractor.fit.statistics import *

from spectractor.parameters import FIT_WORKSPACE as fit_workspace

from scipy.optimize import minimize

import emcee
from emcee.utils import MPIPool


class FitWorkspace:

    def __init__(self, filename, atmgrid_filename="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.filename = filename
        self.ndim = 0
        self.truth = None
        self.verbose = verbose
        self.plot = plot
        self.live_fit = live_fit
        self.A1 = 1.0
        self.A2 = 0.05
        self.ozone = 300.
        self.pwv = 3
        self.aerosols = 0.03
        self.reso = 1.5
        self.D = parameters.DISTANCE2CCD
        self.shift = 0.
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols, self.reso, self.D, self.shift])
        self.ndim = len(self.p)
        self.lambdas = None
        self.model = None
        self.model_err = None
        self.model_noconv = None
        self.input_labels = ["A1", "A2", "ozone", "PWV", "VAOD", "reso [pix]", r"D_CCD [mm]",
                             r"alpha_pix [pix]"]
        self.axis_names = ["$A_1$", "$A_2$", "ozone", "PWV", "VAOD", "reso [pix]", r"$D_{CCD}$ [mm]",
                           r"$\alpha_{\mathrm{pix}}$ [pix]"]
        self.bounds = ((0, 0, 0, 0, 0, 1, 50, -20), (2, 0.5, 800, 10, 1.0, 10, 60, 20))
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
        self.get_truth()
        self.simulation = SpectrumSimulation(self.spectrum, self.atmosphere, self.telescope, self.disperser)
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
        self.plot_fit()
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
    bounds = tuple([(fit_workspace.bounds[0][i], fit_workspace.bounds[1][i]) for i in range(fit_workspace.ndim)])
    nll = lambda p: -lnlike(p)
    result = minimize(nll, fit_workspace.p, method='L-BFGS-B',
                      options={'ftol': 1e-20, 'xtol': 1e-20, 'gtol': 1e-20, 'disp': True, 'maxiter': 100000,
                               'maxls': 50, 'maxcor': 30},
                      bounds=bounds)
    fit_workspace.p = result['x']
    print(fit_workspace.p)
    simulate(*fit_workspace.p)
    fit_workspace.plot_fit()


def run_emcee(w):
    global fit_workspace
    fit_workspace = w
    fit_workspace.print_settings()
    nsamples = fit_workspace.nsteps
    p0 = fit_workspace.set_start()
    pool = MPIPool(loadbalance=True, debug=True)
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

    filename = 'outputs/data_30may17/reduc_20170530_130_spectrum.fits'
    atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')

    fit_workspace = FitWorkspace(filename, atmgrid_filename=atmgrid_filename, nwalkers=28, nsteps=20000, burnin=10000,
                                 nbins=10, verbose=0, plot=False, live_fit=False)
    run_emcee()
    fit_workspace.analyze_chains()

from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectractor.simulation.simulator import *
from spectractor.extractor.extractor import *
from spectractor.fit.mcmc import *
from spectractor import parameters

from scipy.optimize import minimize
import corner

# import pymc as pm
import emcee
from multiprocessing import Pool


class Extractor:

    def __init__(self, filename, atmgrid_filename="", live_fit=False):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.filename = filename
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
        self.truth = None
        self.lambdas = None
        self.model = None
        self.model_err = None
        self.model_noconv = None
        self.labels = ["$A_1$", "$A_2$", "ozone", "PWV", "VAOD", "reso", r"$D_{CCD}$", r"$\alpha_{\mathrm{pix}}$"]
        self.bounds = ((0, 0, 0, 0, 0, 1, 50, -20), (2, 0.5, 800, 10, 1.0, 10, 60, 20))
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
                self.my_logger.info('\n\tUse atmospheric grid models from file %s. ' % atmgrid_filename)
        # self.p[0] *= np.max(self.spectrum.data) / np.max(self.simulation(self.A1,self.A2,self.ozone,self.pwv,self.aerosols,self.reso,self.shift))
        self.get_truth()
        # if 0. in self.spectrum.err:
        #    self.spectrum.err = np.ones_like(self.spectrum.err)
        self.simulation = SpectrumSimulation(self.spectrum, self.atmosphere, self.telescope, self.disperser)

        if parameters.DEBUG:
            for k in range(10):
                atmo = self.atmosphere.simulate(300, k, 0.05)
                plt.plot(self.atmosphere.lambdas, atmo, label='pwv=%dmm' % k)
            plt.grid()
            plt.xlabel('$\lambda$ [nm]')
            plt.ylabel('Atmospheric transmission')
            plt.legend(loc='best')
            if parameters.DISPLAY: plt.show()

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
            self.truth = (A1_truth, A2_truth, ozone_truth, pwv_truth, aerosols_truth, reso_truth, D_truth, shift_truth)
        else:
            self.truth = None

    def plot_spectrum_comparison_simple(self, ax, title='', extent=None, size=0.4):
        l = self.spectrum.lambdas
        sub = np.where((l > parameters.LAMBDA_MIN) & (l < parameters.LAMBDA_MAX))
        if extent is not None:
            sub = np.where((l > extent[0]) & (l < extent[1]))
        self.spectrum.plot_spectrum_simple(ax, lambdas=l)
        p0 = ax.plot(l, self.model(l), label='model')
        ax.fill_between(l, self.model(l) - self.model_err(l),
                        self.model(l) + self.model_err(l), alpha=0.3, color=p0[0].get_color())
        # ax.plot(self.lambdas, self.model_noconv, label='before conv')
        if title != '':
            ax.set_title(title, fontsize=10)
        ax.legend()
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("bottom", size=size, pad=0)
        ax.figure.add_axes(ax2)
        residuals = (self.spectrum.data - self.model(l)) / self.model(l)
        residuals_err = self.spectrum.err / self.model(l)
        ax2.errorbar(l, residuals, yerr=residuals_err, fmt='ro', markersize=2)
        ax2.axhline(0, color=p0[0].get_color())
        ax2.grid(True)
        residuals_model = self.model_err(l) / self.model(l)
        ax2.fill_between(l, -residuals_model, residuals_model, alpha=0.3, color=p0[0].get_color())
        std = np.std(residuals[sub])
        ax2.set_ylim([-2. * std, 2. * std])
        ax2.set_xlabel(ax.get_xlabel())
        ax2.set_ylabel('(data-fit)/fit')
        ax2.set_xlim((l[sub][0], l[sub][-1]))
        ax.set_xlim((l[sub][0], l[sub][-1]))
        ax.set_ylim((0.9 * np.min(self.spectrum.data[sub]), 1.1 * np.max(self.spectrum.data[sub])))
        ax.set_xticks(ax2.get_xticks()[1:-1])

    def plot_fit(self):
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(222)
        ax2 = plt.subplot(224)
        ax3 = plt.subplot(121)
        A1, A2, ozone, pwv, aerosols, reso, D, shift = self.p
        self.title = 'A1={:.3f}, A2={:.3f}, PWV={:.3f}, OZ={:.3g}, ' \
                     'VAOD={:.3f}, reso={:.2f}, D={:.2f}, shift={:.2f}'.format(A1, A2,
                                                                               pwv, ozone, aerosols, reso, D, shift)
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
            if parameters.DISPLAY: plt.show()


class Extractor_MCMC(Extractor):

    def __init__(self, filename, covfile, nchains=1, nsteps=1000, burnin=100, nbins=10, exploration_time=100,
                 atmgrid_filename="", live_fit=False):
        Extractor.__init__(self, filename, atmgrid_filename=atmgrid_filename, live_fit=live_fit)
        self.nwalkers = 5 * self.ndim
        self.nchains = nchains
        self.nsteps = nsteps
        self.covfile = covfile
        self.nbins = nbins
        self.burnin = burnin
        self.exploration_time = exploration_time
        self.chains = Chains(filename, covfile, nchains, nsteps, burnin, nbins, truth=self.truth)
        self.covfile = filename.replace('spectrum.fits', 'cov.txt')
        self.results = []
        self.results_err = []
        for k in range(self.chains.dim):
            self.results.append(ParameterList(self.chains.labels[k], self.chains.axis_names[k]))
            self.results_err.append([])


    def run_mcmc(self):
        complete = self.chains.check_completness()
        if not complete:
            for k in range(self.nchains):
                self.chains.chains.append(
                    Chain(self.chains.chains_filename, self.covfile, nchain=k, nsteps=self.nsteps))
            pool = Pool(processes=self.nchains)
            try:
                # Without the .get(9999), you can't interrupt this with Ctrl+C.
                pool.map_async(self.mcmc, self.chains.chains).get(999999)
                pool.close()
                pool.join()
                # to skip lines after the progress bars
                print('\n' * self.nchains)
            except KeyboardInterrupt:
                pool.terminate()
        self.likelihood = self.chains.chains_to_likelihood()
        self.likelihood.stats(self.covfile)
        # [self.results[i].append(self.likelihood.pdfs[i].mean) for i in range(self.chains.dim)]
        # self.p = [self.likelihood.pdfs[i].mean for i in range(self.chains.dim)]
        self.p = self.chains.best_row_params
        self.simulate(*self.p)
        # [self.results_err[i].append([self.likelihood.pdfs[i].error_high,
        # self.likelihood.pdfs[i].error_low]) for i in range(self.chains.dim)]
        # if(self.plot):
        self.likelihood.triangle_plots()
        self.plot_fit()
        # if convergence_test :
        self.chains.convergence_tests()
        return self.likelihood


def simulate(A1, A2, ozone, pwv, aerosols, reso, D, shift):
    '''
    self.atmosphere.simulate(ozone, pwv, aerosols)
    self.disperser.D = DISTANCE2CCD
    pixels = self.disperser.grating_lambda_to_pixel(self.spectrum.lambdas, self.x0, order=1)
    new_x0 = [ self.x0[0] - shift, self.x0[1] ]
    pixels = pixels - shift
    self.disperser.D = D
    self.lambdas = self.disperser.grating_pixel_to_lambda(pixels, new_x0, order=1)
    '''
    lambdas, model, model_err = fit_workspace.simulation.simulate(A1, A2, ozone, pwv, aerosols, reso, D, shift)
    #if fit_workspace.live_fit:
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


def run_minimisation(fit_workspace):
    bounds = tuple([(fit_workspace.bounds[0][i], fit_workspace.bounds[1][i]) for i in range(fit_workspace.ndim)])
    nll = lambda p: -lnlike(p)
    result = minimize(nll, fit_workspace.p, method='L-BFGS-B',
                         options={'ftol': 1e-20, 'xtol': 1e-20, 'gtol': 1e-20, 'disp': True, 'maxiter': 100000,
                                  'maxls': 50, 'maxcor': 30},
                         bounds=bounds)
    fit_workspace.p = result['x']
    print(fit_workspace.p)
    fit_workspace.simulate(*fit_workspace.p)
    fit_workspace.plot_fit()


def run_emcee():
    start = np.array([np.random.uniform(fit_workspace.p[i]-0.01*(fit_workspace.bounds[1][i]-fit_workspace.bounds[0][i]),
                                        fit_workspace.p[i]+0.01*(fit_workspace.bounds[1][i]-fit_workspace.bounds[0][i]),
                                        fit_workspace.nwalkers)
                      for i in range(fit_workspace.ndim)]).T
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    # file_name = "tutorial.h5"
    # backend = emcee.backends.HDFBackend(file_name)
    # backend.reset(self.nwalkers, self.ndim)
    nwalkers = 5*fit_workspace.ndim
    nsamples = 1000
    sampler = emcee.EnsembleSampler(nwalkers, fit_workspace.ndim, lnprob, args=(), threads=4)
    for i, result in enumerate(sampler.sample(start, iterations=nsamples)):
        if (i + 1) % 100 == 0:
            print("{0:5.1%}".format(float(i) / nsamples))
    # self.sampler.run_mcmc(start, nsamples)
    # tau = sampler.get_autocorr_time()
    burnin = 500  # int(nsamples / 2)
    thin = int(nsamples / 4)
    # self.chains = self.sampler(discard=burnin, flat=True, thin=thin)
    chains = sampler.chain[:, burnin:, :]  # .reshape((-1, self.ndim))
    fit_workspace.p = [np.mean(sampler.flatchain[burnin:, i]) for i in range(fit_workspace.ndim)]
    print(fit_workspace.p)
    fig, ax = plt.subplots(fit_workspace.ndim + 1, 1, figsize=(16, 8), sharex=True)
    steps = np.arange(0, nsamples - burnin)
    for i in range(fit_workspace.ndim):
        ax[i].set_ylabel(fit_workspace.labels[i])
        for k in range(nwalkers):
            ax[i].plot(steps, chains[k, :, i])
    for k in range(nwalkers):
        ax[fit_workspace.ndim].plot(steps, -2 * sampler.lnprobability[k, burnin:])
    ax[fit_workspace.ndim].set_ylabel(r'$\chi^2$')
    ax[fit_workspace.ndim].set_xlabel('Steps')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if parameters.DISPLAY:
        plt.show()
    print(("burn-in: {0}".format(burnin)))
    print(("thin: {0}".format(thin)))
    # print("flat chain shape: {0}".format(samples.shape))
    # print("flat log prob shape: {0}".format(log_prob_samples.shape))
    # print("flat log prior shape: {0}".format(log_prior_samples.shape))
    simulate(*fit_workspace.p)
    fit_workspace.plot_fit()
    fig = corner.corner(sampler.flatchain[burnin*nwalkers:, :], labels=fit_workspace.labels, truths=fit_workspace.truth,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True)
    #fig.set_size_inches(5, 5)
    #fig.tight_layout()
    print(sampler.acceptance_fraction)
    # print(self.sampler.acor)
    if parameters.DISPLAY: plt.show()
    fig.savefig("triangle.png")



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

    parameters.VERBOSE = False
    filename = 'outputs/data_30may17/reduc_20170530_134_spectrum.fits'
    atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')
    filename = 'outputs/data_30may17/reduc_20170530_134_sim.fits'

    # m = Extractor(file_name,atmgrid_filename)
    # m.minimizer(live_fit=True)
    covfile = 'covariances/proposal.txt'
    fit_workspace = Extractor_MCMC(filename, covfile, nchains=4, nsteps=10000, burnin=2000, nbins=10, exploration_time=500,
                       atmgrid_filename=atmgrid_filename, live_fit=False)
    #run_minimisation(m)
    run_emcee()

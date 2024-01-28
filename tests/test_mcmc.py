import matplotlib as mpl
mpl.use('Agg')  # must be run first! But therefore requires noqa E402 on all other imports

import numpy as np  # noqa: E402
from spectractor.fit.fitter import FitParameters  # noqa: E402
from spectractor.fit.mcmc import (MCMCFitWorkspace, run_emcee)  # noqa: E402
from spectractor.config import set_logger  # noqa: E402
from spectractor import parameters  # noqa: E402

import os  # noqa: E402


class LineFitWorkspace(MCMCFitWorkspace):

    def __init__(self, x, y, yerr, file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        params = FitParameters(np.array([1, 1]), labels=["a", "b"], axis_names=["$a$", "$b$"],
                               bounds=[[-100, 100], [-100, 100]], truth=truth, filename=file_name)
        MCMCFitWorkspace.__init__(self, params, file_name, nwalkers, nsteps, burnin, nbins, verbose, plot, live_fit)
        self.my_logger = set_logger(self.__class__.__name__)
        self.x = x
        self.data = y
        self.err = yerr

    def simulate(self, a, b):
        self.model = a * self.x + b
        self.model_err = np.zeros_like(self.x)
        return self.x, self.model, self.model_err


def test_fitworkspace(seed=42):
    # Create mock data
    np.random.seed(42)
    N = 100
    a = 5
    b = -1
    truth = (a, b)
    sigma = 0.1
    x = np.linspace(0, 1, N)
    y = a * x + b
    yerr = sigma * np.ones_like(y)
    y += np.random.normal(scale=sigma, size=N)
    parameters.VERBOSE = True

    # Do the fits
    file_name = "./outputs/test_linefitworkspace.txt"
    w = LineFitWorkspace(x, y, yerr, file_name, truth=truth, nwalkers=20, nsteps=5000, burnin=1000, nbins=20)

    def lnprob(p):
        lp = w.lnprior(p)
        if not np.isfinite(lp):
            return -1e20
        return lp + w.lnlike(p)

    run_emcee(w, ln=lnprob)
    w.analyze_chains()

    assert w.chains.shape == (5000, 20, 2)
    assert np.all(w.gelmans < 0.03)
    assert os.path.exists(file_name.replace(".txt", "_emcee.h5"))
    assert os.path.exists(file_name.replace(".txt", "_emcee_convergence.pdf"))
    assert os.path.exists(file_name.replace(".txt", "_emcee_triangle.pdf"))
    assert np.all([np.abs(w.params.values[i] - truth[i]) / np.sqrt(w.params.cov[i, i]) < 3 for i in range(w.params.ndim)])


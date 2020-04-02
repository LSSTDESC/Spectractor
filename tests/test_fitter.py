from numpy.testing import run_module_suite
import numpy as np
from spectractor.fit.fitter import FitWorkspace, run_minimisation, run_emcee, run_minimisation_sigma_clipping
from spectractor.config import set_logger
from spectractor import parameters

import os


class LineFitWorkspace(FitWorkspace):

    def __init__(self, x, y, yerr, file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        FitWorkspace.__init__(self, file_name, nwalkers, nsteps, burnin, nbins, verbose, plot, live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        self.x = x
        self.data = y
        self.err = yerr
        self.a = 1
        self.b = 1
        self.p = np.array([self.a, self.b])
        self.input_labels = ["a", "b"]
        self.axis_names = ["$a$", "$b$"]
        self.bounds = np.array([(-100, 100), (-100, 100)])
        self.nwalkers = max(2 * self.ndim, nwalkers)

    def simulate(self, a, b):
        self.model = a * self.x + b
        self.model_err = np.zeros_like(self.x)
        return self.x, self.model, self.model_err


def test_fitworkspace():
    # Create mock data
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
    file_name = "test_linefitworkspace.txt"
    w = LineFitWorkspace(x, y, yerr, file_name, truth=truth, nwalkers=20, nsteps=5000, burnin=1000, nbins=20)
    run_minimisation(w, method="minimize")
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    w.p = np.array([1, 1])
    run_minimisation(w, method="basinhopping")
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    w.p = np.array([1, 1])
    run_minimisation(w, method="least_squares")
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    w.p = np.array([1, 1])
    run_minimisation(w, method="minuit")
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    w.p = np.array([1, 1])
    run_minimisation(w, method="newton")
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])

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
    assert np.all([np.abs(w.p[i] - truth[i]) / np.sqrt(w.cov[i,i]) < 2 for i in range(w.ndim)])


def test_minimisation_sigma_clipping():
    # Create mock data
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

    # Add outliers
    outliers = [3, 4, 5, 6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 80, 81]
    y[outliers[:10]] = 4
    y[outliers[10:]] = -3

    # Do the fits
    sigma = 5
    clip_niter = 4
    w = LineFitWorkspace(x, y, yerr, file_name="", truth=truth, nwalkers=20, nsteps=5000, burnin=1000, nbins=20)
    run_minimisation_sigma_clipping(w, method="minimize", sigma_clip=sigma, niter_clip=clip_niter)
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="basinhopping", sigma_clip=sigma, niter_clip=clip_niter)
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="least_squares", sigma_clip=sigma, niter_clip=clip_niter)
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="minuit", sigma_clip=sigma, niter_clip=clip_niter)
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="newton", sigma_clip=sigma, niter_clip=clip_niter)
    assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    assert np.all(outliers == w.outliers)


if __name__ == "__main__":
    run_module_suite()

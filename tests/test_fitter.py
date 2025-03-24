import matplotlib as mpl
mpl.use('Agg')  # must be run first! But therefore requires noqa E402 on all other imports

import numpy as np  # noqa: E402
from spectractor.fit.fitter import (FitParameters, FitWorkspace, run_minimisation,  # noqa: E402
                                    run_minimisation_sigma_clipping)  # noqa: E402
from spectractor.config import set_logger  # noqa: E402
from spectractor import parameters  # noqa: E402


class LineFitWorkspace(FitWorkspace):

    def __init__(self, x, y, yerr, file_name="", epsilon=None, verbose=0, plot=False, live_fit=False, truth=None):
        params = FitParameters(np.array([1, 1]), labels=["a", "b"], axis_names=["$a$", "$b$"],
                               bounds=[[-100, 100], [-100, 100]], truth=truth, filename=file_name)
        FitWorkspace.__init__(self, params, x=x, data=y, err=yerr, epsilon=epsilon,
                              file_name=file_name, verbose=verbose, plot=plot, live_fit=live_fit)
        self.my_logger = set_logger(self.__class__.__name__)

    def simulate(self, *params):
        a, b = params
        self.model = a * self.x + b
        self.model_err = np.zeros_like(self.x)
        return self.x, self.model, self.model_err

    def jacobian(self, params, model_input=None):
        a, b = params
        return np.array([self.x, np.ones_like(self.x)])


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
    parameters.DEBUG = True

    # Do the fits
    file_name = "./outputs/test_linefitworkspace.txt"
    w = LineFitWorkspace(x, y, yerr, file_name=file_name, epsilon=None, truth=truth, verbose=True)
    w.my_logger.info(f"Test minimize method...")
    run_minimisation(w, method="minimize")
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    w.my_logger.info(f"Test bainhopping method...")
    w.params.values = np.array([1, 1])
    run_minimisation(w, method="basinhopping")
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    w.my_logger.info(f"Test curve_fit method...")
    w.params.values = np.array([1, 1])
    run_minimisation(w, method="curve_fit")
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    w.my_logger.info(f"Test Levenberg-Marquardt method...")
    w.params.values = np.array([1, 1])
    run_minimisation(w, method="lm")
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    w.my_logger.info(f"Test least_squares method...")
    w.params.values = np.array([1, 1])
    run_minimisation(w, method="least_squares")
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    w.my_logger.info(f"Test Newton-Raphson method...")
    w.params.values = np.array([1, 1])
    run_minimisation(w, method="newton", with_line_search=True)
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])


def test_minimisation_sigma_clipping(seed=42):
    # Create mock data
    np.random.seed(seed)
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
    w = LineFitWorkspace(x, y, yerr, file_name="", truth=truth)
    run_minimisation_sigma_clipping(w, method="minimize", sigma_clip=sigma, niter_clip=clip_niter)
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="basinhopping", sigma_clip=sigma, niter_clip=clip_niter)
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    assert np.all(outliers == w.outliers)
    # w.p = np.array([1, 1])
    # w.outliers = []
    # run_minimisation_sigma_clipping(w, method="least_squares", sigma_clip=sigma, niter_clip=clip_niter)
    # assert np.all([np.abs(w.p[i] - truth[i]) / sigma < 1 for i in range(w.ndim)])
    # assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="minuit", sigma_clip=sigma, niter_clip=clip_niter)
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="newton", sigma_clip=sigma, niter_clip=clip_niter)
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    assert np.all(outliers == w.outliers)

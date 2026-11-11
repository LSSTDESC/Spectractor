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
        FitWorkspace.__init__(self, params, data=y, err=yerr, epsilon=epsilon,
                              file_name=file_name, verbose=verbose, plot=plot, live_fit=live_fit)
        self.x = x
        self.epsilon = np.array([1e-6, 1e-6])
        self.my_logger = set_logger(self.__class__.__name__)

    def simulate(self, *params):
        a, b = params
        self.model = a * self.x + b
        self.model_err = np.zeros_like(self.x)
        return self.x, self.model, self.model_err

    def jacobian(self, params, model_input=None):
        a, b = params
        return np.array([self.x, np.ones_like(self.x)])


def test_fitworkspace(seed=42, sigma_test=5):
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
    assert np.all([np.abs(w.params.values[i] - truth[i]) /sigma < 1 for i in range(w.params.ndim)])
    w.my_logger.info(f"Test Levenberg-Marquardt method...")
    w.params.values = np.array([1, 1])
    run_minimisation(w, method="lm")
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / w.params.err[i] < sigma_test for i in range(w.params.ndim)])
    w.my_logger.info(f"Test least_squares method...")
    w.params.values = np.array([1, 1])
    run_minimisation(w, method="least_squares")
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / w.params.err[i] < sigma_test for i in range(w.params.ndim)])
    w.my_logger.info(f"Test Newton-Raphson method...")
    w.params.values = np.array([1, 1])
    run_minimisation(w, method="newton", with_line_search=True)
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / w.params.err[i] < sigma_test for i in range(w.params.ndim)])
    w.my_logger.info(f"Test lm method...")
    w.params.values = np.array([1, 1])
    run_minimisation(w, method="lm")
    w.my_logger.info(f"\n{w.params}")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / w.params.err[i] < sigma_test for i in range(w.params.ndim)])


def test_minimisation_sigma_clipping(seed=42, sigma_test=5, sigma_clip=5):
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
    run_minimisation_sigma_clipping(w, method="minimize", sigma_clip=sigma_clip, niter_clip=clip_niter)
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="basinhopping", sigma_clip=sigma_clip, niter_clip=clip_niter)
    assert np.all([np.abs(w.params.values[i] - truth[i]) / sigma < 1 for i in range(w.params.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="least_squares", sigma_clip=sigma_clip, niter_clip=clip_niter)
    assert np.all([np.abs(w.params.values[i] - truth[i]) / w.params.err[i] < sigma_test for i in range(w.params.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="minuit", sigma_clip=sigma_clip, niter_clip=clip_niter)
    assert np.all([np.abs(w.params.values[i] - truth[i]) / w.params.err[i] < sigma_test for i in range(w.params.ndim)])
    assert np.all(outliers == w.outliers)
    w.p = np.array([1, 1])
    w.outliers = []
    run_minimisation_sigma_clipping(w, method="newton", sigma_clip=sigma_clip, niter_clip=clip_niter)
    assert np.all([np.abs(w.params.values[i] - truth[i]) / w.params.err[i] < sigma_test for i in range(w.params.ndim)])
    assert np.all(outliers == w.outliers)



def test_weighted_residuals_jac(seed=42, sigma_test=5):
    # Check that weighted_residuals_jac is the Jacobian of weighted_residuals and
    # that least_squares can use it through run_minimisation.
    np.random.seed(seed)
    N = 50
    a, b = 5.0, -1.0
    sigma = 0.1
    truth = (a, b)
    x = np.linspace(0, 1, N)
    y = a * x + b
    yerr = sigma * np.ones_like(y)
    y += np.random.normal(scale=sigma, size=N)

    epsilon = 1e-6
    w = LineFitWorkspace(x, y, yerr, epsilon=epsilon, truth=truth)

    # 1. weighted_residuals_jac must be the (numerical) Jacobian of weighted_residuals
    params = np.array([1.0, 1.0])
    res = w.weighted_residuals(params)
    J = w.weighted_residuals_jac(params)
    assert J.shape == (len(params), len(res)), f"J has shape {J.shape}"
    J_num = np.zeros_like(J)
    for ip in range(len(params)):
        pp = np.copy(params)
        pp[ip] += epsilon
        J_num[ip] = (w.weighted_residuals(pp) - res) / epsilon
    assert np.allclose(J, J_num, rtol=1e-4, atol=1e-4), "weighted_residuals_jac mismatch"

    # 2. least_squares method using the analytic jacobian must recover the truth
    w.params.values = np.array([1.0, 1.0])
    run_minimisation(w, method="least_squares")
    assert np.all([np.abs(w.params.values[i] - truth[i]) / w.params.err[i] < sigma_test for i in range(w.params.ndim)])

    # 3. check it handles fixed parameters: fix b at its true value
    w = LineFitWorkspace(x, y, yerr, epsilon=epsilon, truth=truth)
    ipb = w.params.get_index("b")
    w.params.values[ipb] = b
    w.params.fixed[ipb] = True
    run_minimisation(w, method="least_squares")
    assert np.abs(w.params.values[0] - a) / w.params.err[0] < sigma_test, "fixed-b least_squares failed"
    assert w.params.values[1] == b, "fixed parameter b should be unchanged"


class MultiLineFitWorkspace(FitWorkspace):
    """Fit workspace for several blocks of unequal lengths sharing a common model y = a*x + b."""

    def __init__(self, xs, ys, covs, epsilon=None, verbose=0, truth=None):
        params = FitParameters(np.array([1.0, 1.0]), labels=["a", "b"],
                               bounds=[[-100, 100], [-100, 100]], truth=truth, filename="")
        # Like MultiSpectraFitWorkspace: only pass params/epsilon to the base __init__
        # (so prepare_weight_matrices is not auto-called), then build the object arrays
        # of blocks and the per-block weight matrices manually.
        FitWorkspace.__init__(self, params, epsilon=epsilon, verbose=verbose)
        self.data = np.empty(len(xs), dtype=object)
        self.err = np.empty(len(xs), dtype=object)
        self.data_cov = np.empty(len(xs), dtype=object)
        self.W = np.empty(len(xs), dtype=object)
        for k in range(len(xs)):
            self.data[k] = np.asarray(ys[k], dtype=float)
            self.err[k] = np.sqrt(np.diag(np.asarray(covs[k], dtype=float)))
            self.data_cov[k] = np.asarray(covs[k], dtype=float)
            L = np.linalg.inv(np.linalg.cholesky(self.data_cov[k]))
            self.W[k] = L.T @ L
        self.xs = [np.asarray(x, dtype=float) for x in xs]
        self.epsilon = np.array([1e-6, 1e-6])

    def simulate(self, *params):
        a, b = params
        model = np.empty(len(self.xs), dtype=object)
        model_err = np.empty(len(self.xs), dtype=object)
        for k in range(len(self.xs)):
            model[k] = a * self.xs[k] + b
            model_err[k] = np.zeros_like(self.xs[k])
        self.model = np.asarray(model, dtype=object)
        self.model_err = np.asarray(model_err, dtype=object)
        return self.xs, self.model, self.model_err


def test_weighted_residuals_jac_multiblock(seed=42, sigma_test=5):
    # Check that weighted_residuals_jac works when fitting a common model to several
    # data blocks of unequal lengths (multi-spectra / block diagonal covariance case).
    np.random.seed(seed)
    a, b = 3.0, 2.0
    truth = (a, b)
    sigma = 0.05
    xs = [np.linspace(0, 1, 30), np.linspace(0, 1.5, 20), np.linspace(0, 0.5, 10)]
    ys, covs = [], []
    for x in xs:
        y = a * x + b + np.random.normal(scale=sigma, size=len(x))
        ys.append(y)
        # correlated (non diagonal) covariance per block
        cov = np.eye(len(x)) * sigma ** 2
        for i in range(len(x) - 1):
            cov[i, i + 1] = cov[i + 1, i] = 0.5 * sigma ** 2
        covs.append(cov)

    epsilon = 1e-6
    w = MultiLineFitWorkspace(xs, ys, covs, epsilon=epsilon, truth=truth)
    total_m = sum(len(x) for x in xs)

    # 1. weighted_residuals_jac must have shape (n_params, total_m) and match the
    #    numerical Jacobian of weighted_residuals
    params = np.array([1.0, 1.0])
    res = w.weighted_residuals(params)
    assert res.shape == (total_m,), f"residuals have shape {res.shape}"
    J = w.weighted_residuals_jac(params)
    assert J.shape == (len(params), total_m), f"J has shape {J.shape}"
    J_num = np.zeros_like(J)
    for ip in range(len(params)):
        pp = np.copy(params)
        pp[ip] += epsilon
        J_num[ip] = (w.weighted_residuals(pp) - res) / epsilon
    assert np.allclose(J, J_num, rtol=1e-4, atol=1e-4), "multi-block weighted_residuals_jac mismatch"

    # 2. jacobian() returns a list of per-block Jacobians (object array (n_params, n_blocks))
    Jb = w.jacobian(params)
    assert Jb.shape == (len(params), len(xs)), f"jacobian has shape {Jb.shape}"
    for ip in range(len(params)):
        for k in range(len(xs)):
            assert len(Jb[ip][k]) == len(xs[k]), f"block {k} jacobian length mismatch"

    # 3. least_squares with the analytic jacobian must recover the truth
    w.params.values = np.array([1.0, 1.0])
    run_minimisation(w, method="least_squares")
    assert np.abs(w.params.values[0] - a) < sigma_test * sigma, f"a={w.params.values[0]} != {a}"
    assert np.abs(w.params.values[1] - b) < sigma_test * sigma, f"b={w.params.values[1]} != {b}"

    # 3bis. gradient descent (newton) must also recover the truth in the multi-block case
    w1 = MultiLineFitWorkspace(xs, ys, covs, epsilon=epsilon, truth=truth)
    w1.params.values = np.array([1.0, 1.0])
    run_minimisation(w1, method="newton", niter=50)
    assert np.abs(w1.params.values[0] - a) < sigma_test * sigma, f"newton a={w1.params.values[0]} != {a}"
    assert np.abs(w1.params.values[1] - b) < sigma_test * sigma, f"newton b={w1.params.values[1]} != {b}"

    # 4. fixed parameter handling: fix b at its true value
    w2 = MultiLineFitWorkspace(xs, ys, covs, epsilon=epsilon, truth=truth)
    ipb = w2.params.get_index("b")
    w2.params.values[ipb] = b
    w2.params.fixed[ipb] = True
    run_minimisation(w2, method="least_squares")
    assert np.abs(w2.params.values[0] - a) < sigma_test * sigma, f"fixed-b a={w2.params.values[0]} != {a}"
    assert w2.params.values[1] == b, "fixed parameter b should be unchanged"

    # 4bis. same fixed-parameter check with newton
    w3 = MultiLineFitWorkspace(xs, ys, covs, epsilon=epsilon, truth=truth)
    ipb = w3.params.get_index("b")
    w3.params.values[ipb] = b
    w3.params.fixed[ipb] = True
    run_minimisation(w3, method="newton", niter=50)
    assert np.abs(w3.params.values[0] - a) < sigma_test * sigma, f"newton fixed-b a={w3.params.values[0]} != {a}"
    assert w3.params.values[1] == b, "newton fixed parameter b should be unchanged"

import os, sys
from scipy.optimize import curve_fit
import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import MaxNLocator

import warnings
from scipy.signal import fftconvolve, gaussian
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from skimage.feature import hessian_matrix
from spectractor.config import *
from spectractor import parameters
from math import floor


def gauss(x, A, x0, sigma):
    """Evaluate the Gaussian function.

    Parameters
    ----------
    x: array_like
        Abscisse array to evaluate the function of size Nx.
    A: float
        Amplitude of the Gaussian function.
    x0: float
        Mean of the Gaussian function.
    sigma: float
        Standard deviation of the Gaussian function.

    Returns
    -------
    m: array_like
        The Gaussian function evaluated on the x array.

    Examples
    --------
    >>> x = np.arange(50)
    >>> y = gauss(x, 10, 25, 3)
    >>> print(y.shape)
    (50,)
    >>> y[25]
    10.0
    """
    return A * np.exp(-(x - x0)*(x - x0) / (2 * sigma * sigma))


def gauss_jacobian(x, A, x0, sigma):
    """Compute the Jacobian matrix of the Gaussian function.

    Parameters
    ----------
    x: array_like
        Abscisse array to evaluate the function of size Nx.
    A: float
        Amplitude of the Gaussian function.
    x0: float
        Mean of the Gaussian function.
    sigma: float
        Standard deviation of the Gaussian function.

    Returns
    -------
    m: array_like
        The Jacobian matrix of size 3 x Nx.

    Examples
    --------
    >>> x = np.arange(50)
    >>> jac = gauss_jacobian(x, 10, 25, 3)
    >>> print(jac.shape)
    (50, 3)
    """
    dA = gauss(x, A, x0, sigma) / A
    dx0 = A * (x - x0) / (sigma * sigma) * dA
    dsigma = A * (x-x0)*(x-x0) / (sigma ** 3) * dA
    return np.array([dA, dx0, dsigma]).T


def line(x, a, b):
    return a * x + b


# noinspection PyTypeChecker
def fit_gauss(x, y, guess=[10, 1000, 1], bounds=(-np.inf, np.inf), sigma=None):
    """Fit a Gaussian profile to data, using curve_fit. The mean guess value of the Gaussian
    must not be far from the truth values. Boundaries helps a lot also.

    Parameters
    ----------
    x: np.array
        The x data values.
    y: np.array
        The y data values.
    guess: list, [amplitude, mean, sigma], optional
        List of first guessed values for the Gaussian fit (default: [10, 1000, 1]).
    bounds: list, optional
        List of boundaries for the parameters [[minima],[maxima]] (default: (-np.inf, np.inf)).
    sigma: np.array, optional
        The y data uncertainties.

    Returns
    -------
    popt: list
        Best fitting parameters of curve_fit.
    pcov: list
        Best fitting parameters covariance matrix from curve_fit.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(600.,700.,2)
    >>> p = [10, 650, 10]
    >>> y = gauss(x, *p)
    >>> y_err = np.ones_like(y)
    >>> print(y[25])
    10.0
    >>> guess = (2,630,2)
    >>> popt, pcov = fit_gauss(x, y, guess=guess, bounds=((1,600,1),(100,700,100)), sigma=y_err)
    >>> assert np.all(np.isclose(p,popt))
    """
    popt, pcov = curve_fit(gauss, x, y, p0=guess, bounds=bounds,  tr_solver='exact', jac=gauss_jacobian,
                           sigma=sigma, method='dogbox', verbose=0, xtol=1e-20, ftol=1e-20)
    return popt, pcov


def multigauss_and_line(x, *params):
    """Multiple Gaussian profile plus a straight line to data.
    The order of the parameters is line slope, line intercept,
    and then block of 3 parameters for the Gaussian profiles like amplitude, mean and standard
    deviation.

    Parameters
    ----------
    x: array
        The x data values.
    *params: list of float parameters as described above.

    Returns
    -------
    y: array
        The y profile values.

    Examples
    --------
    >>> x = np.arange(600.,800.,1)
    >>> y = multigauss_and_line(x, 1, 10, 20, 650, 3, 40, 750, 10)
    >>> print(y[0])
    610.0
    """
    out = line(x, params[0], params[1])
    for k in range((len(params) - 2) // 3):
        out += gauss(x, *params[2 + 3 * k:2 + 3 * k + 3])
    return out


# noinspection PyTypeChecker
def fit_multigauss_and_line(x, y, guess=[0, 1, 10, 1000, 1, 0], bounds=(-np.inf, np.inf)):
    """Fit a multiple Gaussian profile plus a straight line to data, using curve_fit.
    The mean guess value of the Gaussian must not be far from the truth values.
    Boundaries helps a lot also. The order of the parameters is line slope, line intercept,
    and then block of 3 parameters for the Gaussian profiles like amplitude, mean and standard
    deviation.

    Parameters
    ----------
    x: array
        The x data values.
    y: array
        The y data values.
    guess: list, [slope, intercept, amplitude, mean, sigma]
        List of first guessed values for the Gaussian fit (default: [0, 1, 10, 1000, 1]).
    bounds: 2D-list
        List of boundaries for the parameters [[minima],[maxima]] (default: (-np.inf, np.inf)).

    Returns
    -------
    popt: list
        Best fitting parameters of curve_fit.
    pcov: 2D-list
        Best fitting parameters covariance matrix from curve_fit.

    Examples
    --------
    >>> x = np.arange(600.,800.,1)
    >>> y = multigauss_and_line(x, 1, 10, 20, 650, 3, 40, 750, 10)
    >>> print(y[0])
    610.0
    >>> bounds = ((-np.inf,-np.inf,1,600,1,1,600,1),(np.inf,np.inf,100,800,100,100,800,100))
    >>> popt, pcov = fit_multigauss_and_line(x, y, guess=(0,1,3,630,3,3,770,3), bounds=bounds)
    >>> print(popt)
    [   1.   10.   20.  650.    3.   40.  750.   10.]
    """
    maxfev = 1000
    popt, pcov = curve_fit(multigauss_and_line, x, y, p0=guess, bounds=bounds, maxfev=maxfev, absolute_sigma=True)
    return popt, pcov


def rescale_x_for_legendre(x):
    middle = 0.5*(np.max(x) + np.min(x))
    x_norm = x - middle
    if np.max(x_norm) != 0:
        return x_norm / np.max(x_norm)
    else:
        return x_norm


# noinspection PyTypeChecker
def multigauss_and_bgd(x, *params):
    """Multiple Gaussian profile plus a polynomial background to data.
    Polynomial function is based on the orthogonal Legendre polynomial basis.
    The degree of the polynomial background is fixed by parameters.CALIB_BGD_NPARAMS.
    The order of the parameters is a first block CALIB_BGD_NPARAMS parameters (from low to high Legendre polynome degree,
    contrary to np.polyval), and then block of 3 parameters for the Gaussian profiles like amplitude, mean and standard
    deviation.

    Parameters
    ----------
    x: array
        The x data values.
    *params: list of float parameters as described above.

    Returns
    -------
    y: array
        The y profile values.

    Examples
    --------
    >>> import spectractor.parameters as parameters
    >>> parameters.CALIB_BGD_NPARAMS = 4
    >>> x = np.arange(600.,800.,1)
    >>> p = [20, 1, -1, -1, 20, 650, 3, 40, 750, 5]
    >>> y = multigauss_and_bgd(x, *p)
    >>> print(f'{y[0]:.2f}')
    19.00

    ..plot:
        import matplotlib.pyplot as plt
        plt.plot(x,y,'r-')
        plt.show()

    """
    bgd_nparams = parameters.CALIB_BGD_NPARAMS
    # out = np.polyval(params[0:bgd_nparams], x)
    x_norm = rescale_x_for_legendre(x)
    out = np.polynomial.legendre.legval(x_norm, params[0:bgd_nparams])
    for k in range((len(params) - bgd_nparams) // 3):
        out += gauss(x, *params[bgd_nparams + 3 * k:bgd_nparams + 3 * k + 3])
    return out


# noinspection PyTypeChecker
def multigauss_and_bgd_jacobian(x, *params):
    """Jacobien of the multiple Gaussian profile plus a polynomial background to data.
    The degree of the polynomial background is fixed by parameters.CALIB_BGD_NPARAMS.
    The order of the parameters is a first block CALIB_BGD_NPARAMS parameters (from low to high Legendre polynome degree,
    contrary to np.polyval), and then block of 3 parameters for the Gaussian profiles like amplitude, mean and standard
    deviation. x values are renormalised on the [-1, 1] interval for the background.

    Parameters
    ----------
    x: array
        The x data values.
    *params: list of float parameters as described above.

    Returns
    -------
    y: array
        The jacobian values.

    Examples
    --------
    >>> import spectractor.parameters as parameters
    >>> parameters.CALIB_BGD_NPARAMS = 4
    >>> x = np.arange(600.,800.,1)
    >>> p = [20, 1, -1, -1, 20, 650, 3, 40, 750, 5]
    >>> y = multigauss_and_bgd_jacobian(x, *p)
    >>> assert(np.all(np.isclose(y.T[0],np.ones_like(x))))
    >>> print(y.shape)
    (200, 10)
    """
    bgd_nparams = parameters.CALIB_BGD_NPARAMS
    out = []
    x_norm = rescale_x_for_legendre(x)
    for k in range(bgd_nparams):
        # out.append(params[k]*(parameters.CALIB_BGD_ORDER-k)*x**(parameters.CALIB_BGD_ORDER-(k+1)))
        # out.append(x ** (bgd_nparams - 1 - k))
        c = np.zeros(bgd_nparams)
        c[k] = 1
        out.append(np.polynomial.legendre.legval(x_norm, c))
    for k in range((len(params) - bgd_nparams) // 3):
        jac = gauss_jacobian(x, *params[bgd_nparams + 3 * k:bgd_nparams + 3 * k + 3]).T
        for j in jac:
            out.append(list(j))
    return np.array(out).T


# noinspection PyTypeChecker
def fit_multigauss_and_bgd(x, y, guess=[0, 1, 10, 1000, 1, 0], bounds=(-np.inf, np.inf), sigma=None, fix_centroids=False):
    """Fit a multiple Gaussian profile plus a polynomial background to data, using iminuit.
    The mean guess value of the Gaussian must not be far from the truth values.
    Boundaries helps a lot also. The degree of the polynomial background is fixed by parameters.CALIB_BGD_NPARAMS.
    The order of the parameters is a first block CALIB_BGD_NPARAMS parameters (from high to low monomial terms,
    same as np.polyval), and then block of 3 parameters for the Gaussian profiles like amplitude, mean and standard
    deviation. x values are renormalised on the [-1, 1] interval for the background.

    Parameters
    ----------
    x: array
        The x data values.
    y: array
        The y data values.
    guess: list, [CALIB_BGD_ORDER+1 parameters, 3*number of Gaussian parameters]
        List of first guessed values for the Gaussian fit (default: [0, 1, 10, 1000, 1]).
    bounds: array
        List of boundaries for the parameters [[minima],[maxima]] (default: (-np.inf, np.inf)).
    sigma: array, optional
        The uncertainties on the y values (default: None).

    Returns
    -------
    popt: array
        Best fitting parameters of curve_fit.
    pcov: array
        Best fitting parameters covariance matrix from curve_fit.

    Examples
    --------
    >>> x = np.arange(600.,800.,1)
    >>> p = [20, 1, -1, -1, 20, 650, 3, 40, 750, 5]
    >>> y = multigauss_and_bgd(x, *p)
    >>> print(f'{y[0]:.2f}')
    19.00
    >>> err = 0.1 * np.sqrt(y)
    >>> guess = (15,0,0,0,10,640,2,20,750,7)
    >>> bounds = ((-np.inf,-np.inf,-np.inf,-np.inf,1,600,1,1,600,1),(np.inf,np.inf,np.inf,np.inf,100,800,100,100,800,100))
    >>> popt, pcov = fit_multigauss_and_bgd(x, y, guess=guess, bounds=bounds, sigma=err)
    >>> assert np.all(np.isclose(p,popt,rtol=1e-4))
    >>> fit = multigauss_and_bgd(x, *popt)

    ..plot:
        import matplotlib.pyplot as plt
        plt.errorbar(x,y,yerr=err,linestyle='None')
        plt.plot(x,fit,'r-')
        plt.plot(x,multigauss_and_bgd(x, *guess),'k--')
        plt.show()
    """
    maxfev = 10000
    popt, pcov = curve_fit(multigauss_and_bgd, x, y, p0=guess, bounds=bounds, maxfev=maxfev, sigma=sigma,
                           absolute_sigma=True, method='trf', xtol=1e-4, ftol=1e-4, verbose=0,
                           jac=multigauss_and_bgd_jacobian, x_scale='jac')
    # error = 0.1 * np.abs(guess) * np.ones_like(guess)
    # z = np.where(np.isclose(error,0.0,1e-6))
    # error[z] = 0.01
    # bounds = np.array(bounds)
    # if bounds.shape[0] == 2 and bounds.shape[1] > 2:
    #     bounds = bounds.T
    # guess = np.array(guess)
    #
    # def chisq_multigauss_and_bgd(params):
    #     if sigma is None:
    #         return np.nansum((multigauss_and_bgd(x, *params) - y)**2)
    #     else:
    #         return np.nansum(((multigauss_and_bgd(x, *params) - y)/sigma)**2)
    #
    # def chisq_multigauss_and_bgd_jac(params):
    #     diff = multigauss_and_bgd(x, *params) - y
    #     jac = multigauss_and_bgd_jacobian(x, *params)
    #     if sigma is None:
    #         return np.array([np.nansum(2 * jac[p] * diff) for p in range(len(params))])
    #     else:
    #         return np.array([np.nansum(2 * jac[p] * diff / (sigma*sigma)) for p in range(len(params))])
    #
    # fix = [False] * error.size
    # if fix_centroids:
    #     for k in range(parameters.CALIB_BGD_NPARAMS, len(fix), 3):
    #        fix[k+1] = True
    # # noinspection PyArgumentList
    # m = Minuit.from_array_func(fcn=chisq_multigauss_and_bgd, start=guess, error=error, errordef=1,
    #                            fix=fix, print_level=0, limit=bounds, grad=chisq_multigauss_and_bgd_jac)
    #
    # m.tol = 0.001
    # m.migrad()
    # try:
    #     pcov = m.np_covariance()
    # except:
    #     pcov = None
    # popt = m.np_values()
    return popt, pcov


# noinspection PyTupleAssignmentBalance
def fit_poly1d(x, y, order, w=None):
    """Fit a 1D polynomial function to data. Use np.polyfit.

    Parameters
    ----------
    x: array
        The x data values.
    y: array
        The y data values.
    order: int
        The degree of the polynomial function.
    w: array, optional
        Weights on the y data (default: None).

    Returns
    -------
    fit: array
        The best fitting parameter values.
    cov: 2D-array
        The covariance matrix
    model: array
        The best fitting profile

    Examples
    --------
    >>> x = np.arange(500., 1000., 1)
    >>> p = [3, 2, 1, 1]
    >>> y = np.polyval(p, x)
    >>> err = np.ones_like(y)
    >>> fit, cov, model = fit_poly1d(x, y, order=3)
    >>> assert np.all(np.isclose(p, fit, 1e-5))
    >>> assert np.all(np.isclose(model, y))
    >>> assert cov.shape == (4, 4)
    >>> fit, cov2, model2 = fit_poly1d(x, y, order=3, w=err)
    >>> assert np.all(np.isclose(p, fit, 1e-5))
    >>> fit, cov3, model3 = fit_poly1d([0, 1], [1, 1], order=3, w=err)
    >>> print(fit)
    [0 0 0 0]
    """
    cov = np.array([])
    if len(x) > order:
        if w is None:
            fit, cov = np.polyfit(x, y, order, cov=True)
        else:
            fit, cov = np.polyfit(x, y, order, cov=True, w=w)
        model = np.polyval(fit, x)
    else:
        fit = np.array([0] * (order + 1))
        model = y
    return fit, cov, model


# noinspection PyTupleAssignmentBalance
def fit_poly1d_legendre(x, y, order, w=None):
    """Fit a 1D polynomial function to data using Legendre polynomial orthogonal basis.

    Parameters
    ----------
    x: array
        The x data values.
    y: array
        The y data values.
    order: int
        The degree of the polynomial function.
    w: array, optional
        Weights on the y data (default: None).

    Returns
    -------
    fit: array
        The best fitting parameter values.
    cov: 2D-array
        The covariance matrix
    model: array
        The best fitting profile

    Examples
    --------
    >>> x = np.arange(500., 1000., 1)
    >>> p = [-1e-6, -1e-4, 1, 1]
    >>> y = np.polyval(p, x)
    >>> err = np.ones_like(y)
    >>> fit, cov, model = fit_poly1d_legendre(x,y,order=3)
    >>> assert np.all(np.isclose(p,fit,3))
    >>> fit, cov2, model2 = fit_poly1d_legendre(x,y,order=3,w=err)
    >>> assert np.all(np.isclose(p,fit,3))
    >>> fit, cov3, model3 = fit_poly1d([0, 1], [1, 1], order=3, w=err)
    >>> print(fit)
    [0 0 0 0]

    ..plot:
        import matplotlib.pyplot as plt
        plt.errorbar(x,y,yerr=err,fmt='ro')
        plt.plot(x,model2)
        plt.show()
    """
    cov = -1
    x_norm = rescale_x_for_legendre(x)
    if len(x) > order:
        if w is None:
            fit, cov = np.polynomial.legendre.legfit(x_norm, y, deg=order, full=True)
        else:
            fit, cov = np.polynomial.legendre.legfit(x_norm, y, deg=order, full=True, w=w)
        model = np.polynomial.legendre.legval(x_norm, fit)
    else:
        fit = np.array([0] * (order + 1))
        model = y
    return fit, cov, model


# noinspection PyTypeChecker,PyUnresolvedReferences
def fit_poly2d(x, y, z, order):
    """Fit a 2D polynomial function to data. Use astropy.modeling.

    Parameters
    ----------
    x: array
        The x data values.
    y: array
        The y data values.
    z: array
        The z data values.
    order: int
        The degree of the polynomial function.

    Returns
    -------
    model: Astropy model
        The best fitting astropy polynomial model

    Examples
    --------
    >>> x, y = np.mgrid[:50,:50]
    >>> z = x**2 + y**2 - 2*x*y
    >>> fit = fit_poly2d(x, y, z, order=2)
    >>> assert np.isclose(fit.c0_0.value, 0)
    >>> assert np.isclose(fit.c1_0.value, 0)
    >>> assert np.isclose(fit.c2_0.value, 1)
    >>> assert np.isclose(fit.c0_1.value, 0)
    >>> assert np.isclose(fit.c0_2.value, 1)
    >>> assert np.isclose(fit.c1_1.value, -2)
    """
    p_init = models.Polynomial2D(degree=order)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)
        return p


def fit_poly1d_outlier_removal(x, y, order=2, sigma=3.0, niter=3):
    """Fit a 1D polynomial function to data. Use astropy.modeling.

    Parameters
    ----------
    x: array
        The x data values.
    y: array
        The y data values.
    order: int
        The degree of the polynomial function (default: 2).
    sigma: float
        Value of the sigma-clipping (default: 3.0).
    niter: int
        The number of iterations to converge (default: 3).

    Returns
    -------
    model: Astropy model
        The best fitting astropy model.
    outliers: array_like
        List of the outlier points.

    Examples
    --------
    >>> x = np.arange(500., 1000., 1)
    >>> p = [3,2,1,0]
    >>> y = np.polyval(p, x)
    >>> y[::10] = 0.
    >>> model, outliers = fit_poly1d_outlier_removal(x,y,order=3,sigma=3)
    >>> print('{:.2f}'.format(model.c0.value))
    0.00
    >>> print('{:.2f}'.format(model.c1.value))
    1.00
    >>> print('{:.2f}'.format(model.c2.value))
    2.00
    >>> print('{:.2f}'.format(model.c3.value))
    3.00

    """
    my_logger = set_logger(__name__)
    gg_init = models.Polynomial1D(order)
    gg_init.c0.min = np.min(y)
    gg_init.c0.max = 2 * np.max(y)
    gg_init.c1 = 0
    gg_init.c2 = 0
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        or_fitted_model, filtered_data = or_fit(gg_init, x, y)
        outliers = [] # not working
        '''
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.plot(x, y, 'gx', label="original data")
        plt.plot(x, filtered_data, 'r+', label="filtered data")
        plt.plot(x, or_fitted_model(x), 'r--',
                 label="model fitted w/ filtered data")
        plt.legend(loc=2, numpoints=1)
        if parameters.DISPLAY: plt.show()
        '''
        # my_logger.info(f'\n\t{or_fitted_model}')
        # my_logger.debug(f'\n\t{fit.fit_info}')
        return or_fitted_model, outliers


def fit_poly2d_outlier_removal(x, y, z, order=2, sigma=3.0, niter=30):
    """Fit a 2D polynomial function to data. Use astropy.modeling.

    Parameters
    ----------
    x: array
        The x data values.
    y: array
        The y data values.
    z: array
        The z data values.
    order: int
        The degree of the polynomial function (default: 2).
    sigma: float
        Value of the sigma-clipping (default: 3.0).
    niter: int
        The number of iterations to converge (default: 30).

    Returns
    -------
    model: Astropy model
        The best fitting astropy model.

    Examples
    --------
    >>> x, y = np.mgrid[:50,:50]
    >>> z = x**2 + y**2 - 2*x*y
    >>> z[::10,::10] = 0.
    >>> fit = fit_poly2d_outlier_removal(x,y,z,order=2,sigma=3)
    >>> assert np.isclose(fit.c0_0.value, 0)
    >>> assert np.isclose(fit.c1_0.value, 0)
    >>> assert np.isclose(fit.c2_0.value, 1)
    >>> assert np.isclose(fit.c0_1.value, 0)
    >>> assert np.isclose(fit.c0_2.value, 1)
    >>> assert np.isclose(fit.c1_1.value, -2)

    """
    my_logger = set_logger(__name__)
    gg_init = models.Polynomial2D(order)
    gg_init.c0_0.min = np.min(z)
    gg_init.c0_0.max = 2 * np.max(z)
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        or_fitted_model, filtered_data = or_fit(gg_init, x, y, z)
        my_logger.info(f'\n\t{or_fitted_model}')
        # my_logger.debug(f'\n\t{fit.fit_info}')
        return or_fitted_model


def tied_circular_gauss2d(g1):
    std = g1.x_stddev
    return std


def fit_gauss2d_outlier_removal(x, y, z, sigma=3.0, niter=3, guess=None, bounds=None, circular=False):
    """Fit an astropy Gaussian 2D model with parameters :
        amplitude, x_mean,y_mean,x_stddev, y_stddev,theta
    using outlier removal methods.

    Parameters
    ----------
    x: np.array
        2D array of the x coordinates from meshgrid.
    y: np.array
        2D array of the y coordinates from meshgrid.
    z: np.array
        the 2D array image.
    sigma: float
        value of sigma for the sigma rejection of outliers (default: 3)
    niter: int
        maximum number of iterations for the outlier detection (default: 3)
    guess: list, optional
        List containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    circular: bool, optional
        If True, force the Gaussian shape to be circular (default: False)

    Returns
    -------
    fitted_model: Fittable
        Astropy Gaussian2D model

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.modeling import models
    >>> X, Y = np.mgrid[:50,:50]
    >>> PSF = models.Gaussian2D()
    >>> p = (50, 25, 25, 5, 5, 0)
    >>> Z = PSF.evaluate(X, Y, *p)

    ..plot:
        plt.imshow(Z, origin='loxer') #doctest: +ELLIPSIS
        plt.show()
    >>> guess = (45, 20, 20, 7, 7, 0)
    >>> bounds = ((1, 10, 10, 1, 1, -90), (100, 40, 40, 10, 10, 90))
    >>> fit = fit_gauss2d_outlier_removal(X, Y, Z, guess=guess, bounds=bounds, circular=True)
    >>> res = [getattr(fit, p).value for p in fit.param_names]
    >>> print(res)
    [50.0, 25.0, 25.0, 5.0, 5.0, 0.0]

    ..plot:
        plt.imshow(Z-fit(X, Y), origin='loxer') #doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at 0x...>
        plt.show()

    """
    my_logger = set_logger(__name__)
    gg_init = models.Gaussian2D()
    if guess is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).value = guess[ip]
    if bounds is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).min = bounds[0][ip]
            getattr(gg_init, p).max = bounds[1][ip]
    if circular:
        gg_init.y_stddev.tied = tied_circular_gauss2d
        gg_init.theta.fixed = True
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        or_fitted_model, filtered_data = or_fit(gg_init, x, y, z)
        my_logger.info(f'\n\t{or_fitted_model}')
        # my_logger.debug(f'\n\t{fit.fit_info}')
        return or_fitted_model


def fit_moffat2d_outlier_removal(x, y, z, sigma=3.0, niter=3, guess=None, bounds=None):
    """Fit an astropy Moffat 2D model with parameters :
        amplitude, x_mean, y_mean, gamma, alpha
    using outlier removal methods.

    Parameters
    ----------
    x: np.array
        2D array of the x coordinates from meshgrid.
    y: np.array
        2D array of the y coordinates from meshgrid.
    z: np.array
        the 2D array image.
    sigma: float
        value of sigma for the sigma rejection of outliers (default: 3)
    niter: int
        maximum number of iterations for the outlier detection (default: 3)
    guess: list, optional
        List containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)

    Returns
    -------
    fitted_model: Fittable
        Astropy Moffat2D model

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.modeling import models
    >>> X, Y = np.mgrid[:100,:100]
    >>> PSF = models.Moffat2D()
    >>> p = (50, 50, 50, 5, 2)
    >>> Z = PSF.evaluate(X, Y, *p)

    ..plot:
        plt.imshow(Z, origin='loxer')
        plt.show()
    >>> guess = (45, 48, 52, 4, 2)
    >>> bounds = ((1, 10, 10, 1, 1), (100, 90, 90, 10, 10))
    >>> fit = fit_moffat2d_outlier_removal(X, Y, Z, guess=guess, bounds=bounds, niter=3)
    >>> res = [getattr(fit, p).value for p in fit.param_names]
    >>> assert(np.all(np.isclose(p, res, 1e-1)))

    ..plot:
        plt.imshow(Z-fit(X, Y), origin='loxer')
        plt.show()
    """
    my_logger = set_logger(__name__)
    gg_init = models.Moffat2D()
    if guess is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).value = guess[ip]
    if bounds is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).min = bounds[0][ip]
            getattr(gg_init, p).max = bounds[1][ip]
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        or_fitted_model, filtered_data = or_fit(gg_init, x, y, z)
        my_logger.info(f'\n\t{or_fitted_model}')
        # my_logger.debug(f'\n\t{fit.fit_info}')
        return or_fitted_model


def fit_moffat1d_outlier_removal(x, y, sigma=3.0, niter=3, guess=None, bounds=None):
    """Fit an astropy Moffat 1D model with parameters :
        amplitude, x_mean, gamma, alpha
    using outlier removal methods.

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates from meshgrid.
    y: np.array
        the 1D array amplitudes.
    sigma: float
        value of sigma for the sigma rejection of outliers (default: 3)
    niter: int
        maximum number of iterations for the outlier detection (default: 3)
    guess: list, optional
        List containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)

    Returns
    -------
    fitted_model: Fittable
        Astropy Moffat1D model

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.modeling import models
    >>> X = np.arange(100)
    >>> PSF = models.Moffat1D()
    >>> p = (50, 50, 5, 2)
    >>> Y = PSF.evaluate(X, *p)

    ..plot:
        plt.imshow(Z, origin='loxer')
        plt.show()
    >>> guess = (45, 48, 4, 2)
    >>> bounds = ((1, 10, 1, 1), (100, 90, 10, 10))
    >>> fit = fit_moffat1d_outlier_removal(X, Y, guess=guess, bounds=bounds, niter=3)
    >>> res = [getattr(fit, p).value for p in fit.param_names]
    >>> assert(np.all(np.isclose(p, res, 1e-6)))

    ..plot:
        plt.imshow(Z-fit(X, Y), origin='loxer')
        plt.show()
    """
    my_logger = set_logger(__name__)
    gg_init = models.Moffat1D()
    if guess is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).value = guess[ip]
    if bounds is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).min = bounds[0][ip]
            getattr(gg_init, p).max = bounds[1][ip]
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        or_fitted_model, filtered_data = or_fit(gg_init, x, y)
        my_logger.info(f'\n\t{or_fitted_model}')
        # my_logger.debug(f'\n\t{fit.fit_info}')
        return or_fitted_model


def fit_moffat1d(x, y, guess=None, bounds=None):
    """Fit an astropy Moffat 1D model with parameters :
        amplitude, x_mean, gamma, alpha

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates from meshgrid.
    y: np.array
        the 1D array amplitudes.
    guess: list, optional
        List containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)

    Returns
    -------
    fitted_model: Fittable
        Astropy Moffat1D model

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.modeling import models
    >>> X = np.arange(100)
    >>> PSF = models.Moffat1D()
    >>> p = (50, 50, 5, 2)
    >>> Y = PSF.evaluate(X, *p)

    ..plot:
        plt.imshow(Z, origin='loxer')
        plt.show()
    >>> guess = (45, 48, 4, 2)
    >>> bounds = ((1, 10, 1, 1), (100, 90, 10, 10))
    >>> fit = fit_moffat1d(X, Y, guess=guess, bounds=bounds)
    >>> res = [getattr(fit, p).value for p in fit.param_names]
    >>> assert(np.all(np.isclose(p, res, 1e-6)))

    ..plot:
        plt.imshow(Z-fit(X, Y), origin='loxer')
        plt.show()
    """
    my_logger = set_logger(__name__)
    gg_init = models.Moffat1D()
    if guess is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).value = guess[ip]
    if bounds is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).min = bounds[0][ip]
            getattr(gg_init, p).max = bounds[1][ip]
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        fitted_model = fit(gg_init, x, y)
        my_logger.info(f'\n\t{fitted_model}')
        # my_logger.debug(f'\n\t{fit.fit_info}')
        return fitted_model


class LevMarLSQFitterWithNan(fitting.LevMarLSQFitter):

    def objective_function(self, fps, *args):
        """
        Function to minimize.

        Parameters
        ----------
        fps : list
            parameters returned by the fitter
        args : list
            [model, [weights], [input coordinates]]
        """

        model = args[0]
        weights = args[1]
        fitting._fitter_to_model_params(model, fps)
        meas = args[-1]
        if weights is None:
            a = np.ravel(model(*args[2: -1]) - meas)
            a[np.isfinite(a)] = 0
            return a
        else:
            a = np.ravel(weights * (model(*args[2: -1]) - meas))
            a[~np.isfinite(a)] = 0
            return a


def find_nearest(array, value):
    """Find the nearest index and value in an array.

    Parameters
    ----------
    array: array
        The array to inspect.
    value: float
        The value to look for.

    Returns
    -------
    index: int
        The array index of the nearest value close to *value*
    val: float
        The value fo the array at index.

    Examples
    --------
    >>> x = np.arange(0.,10.)
    >>> idx, val = find_nearest(x, 3.3)
    >>> print(idx, val)
    3 3.0
    """
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def ensure_dir(directory_name):
    """Ensure that *directory_name* directory exists. If not, create it.

    Parameters
    ----------
    directory_name: str
        The directory name.

    Examples
    --------
    >>> ensure_dir('tests')
    >>> os.path.exists('tests')
    True
    >>> ensure_dir('tests/mytest')
    >>> os.path.exists('tests/mytest')
    True
    >>> os.rmdir('./tests/mytest')
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

        
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    
    For example for the PSF
    
    x=pixel number
    y=Intensity in pixel
    
    values-x
    weights=y=f(x)
    
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
    return average, np.sqrt(variance)


def hessian_and_theta(data, margin_cut=1):
    # compute hessian matrices on the image
    Hxx, Hxy, Hyy = hessian_matrix(data, sigma=3, order='xy')
    lambda_plus = 0.5 * ((Hxx + Hyy) + np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy * Hxy))
    lambda_minus = 0.5 * ((Hxx + Hyy) - np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy * Hxy))
    theta = 0.5 * np.arctan2(2 * Hxy, Hyy - Hxx) * 180 / np.pi
    # remove the margins
    lambda_minus = lambda_minus[margin_cut:-margin_cut, margin_cut:-margin_cut]
    lambda_plus = lambda_plus[margin_cut:-margin_cut, margin_cut:-margin_cut]
    theta = theta[margin_cut:-margin_cut, margin_cut:-margin_cut]
    return lambda_plus, lambda_minus, theta


def filter_stars_from_bgd(data, margin_cut=1):
    lambda_plus, lambda_minus, theta = hessian_and_theta(np.copy(data), margin_cut=margin_cut)
    # thresholds
    lambda_threshold = np.median(lambda_minus) - 2 * np.std(lambda_minus)
    mask = np.where(lambda_minus < lambda_threshold)
    data[mask] = np.nan
    return data


def fftconvolve_gaussian(array, reso):
    """Convolve an 1D or 2D array with a Gaussian profile of given standard deviation.

    Parameters
    ----------
    array: array
        The array to convolve.
    reso: float
        The standard deviation of the Gaussian profile.

    Returns
    -------
    convolved: array
        The convolved array, same size and shape as input.

    Examples
    --------
    >>> array = np.ones(100)
    >>> output = fftconvolve_gaussian(array, 3)
    >>> print(output[:3])
<<<<<<< HEAD
    [0.5        0.63125312 0.74870357]
=======
    [ 0.5         0.63114657  0.74850168]
    >>> array = np.ones((100, 100))
    >>> output = fftconvolve_gaussian(array, 3)
    >>> print(output[0][:3])
    [ 0.5         0.63114657  0.74850168]
    >>> array = np.ones((100, 100, 100))
    >>> output = fftconvolve_gaussian(array, 3)
>>>>>>> 0e123b8e16fd5d6e5d5995d961478e73bc23105c
    """
    my_logger = set_logger(__name__)
    if array.ndim == 2:
        kernel = gaussian(array.shape[1], reso)
        kernel /= np.sum(kernel)
        for i in range(array.shape[0]):
            array[i] = fftconvolve(array[i], kernel, mode='same')
    elif array.ndim == 1:
        kernel = gaussian(array.size, reso)
        kernel /= np.sum(kernel)
        array = fftconvolve(array, kernel, mode='same')
    else:
        my_logger.error('\n\tArray dimension must be 1 or 2.')
    return array


def formatting_numbers(value, error_high, error_low, std=None, label=None):
    """Format a physical value and its uncertainties. Round the uncertainties
    to the first significant digit, and do the same for the physical value.

    Parameters
    ----------
    value: float
        The physical value.
    error_high: float
        Upper uncertainty.
    error_low: float
        Lower uncertainty
    std: float, optional
        The RMS of the physical parameter (default: None).
    label: str, optional
        The name of the physical parameter to output (default: None).

    Returns
    -------
    text: tuple
        The formatted output strings inside a tuple.

    Examples
    --------
    >>> formatting_numbers(3., 0.789, 0.500, std=0.45, label='test')
    ('test', '3.0', '0.8', '0.5', '0.5')
    >>> formatting_numbers(3., 0.07, 0.008, std=0.03, label='test')
    ('test', '3.000', '0.07', '0.008', '0.03')
    >>> formatting_numbers(3240., 0.2, 0.4, std=0.3)
    ('3240.0', '0.2', '0.4', '0.3')
    >>> formatting_numbers(3240., 230, 420, std=330)
    ('3240', '230', '420', '330')
    >>> formatting_numbers(0, 0.008, 0.04, std=0.03)
    ('0.000', '0.008', '0.040', '0.030')
    >>> formatting_numbers(-55, 0.008, 0.04, std=0.03)
    ('-55.000', '0.008', '0.04', '0.03')
    """
    str_std = ""
    out = []
    if label is not None:
        out.append(label)
    power10 = min(int(floor(np.log10(np.abs(error_high)))), int(floor(np.log10(np.abs(error_low)))))
    if np.isclose(0.0, float("%.*f" % (abs(power10), value))):
        str_value = "%.*f" % (abs(power10), 0)
        str_error_high = "%.*f" % (abs(power10), error_high)
        str_error_low = "%.*f" % (abs(power10), error_low)
        if std is not None:
            str_std = "%.*f" % (abs(power10), std)
    elif power10 > 0:
        str_value = f"{value:.0f}"
        str_error_high = f"{error_high:.0f}"
        str_error_low = f"{error_low:.0f}"
        if std is not None:
            str_std = f"{std:.0f}"
    else:
        if int(floor(np.log10(np.abs(error_high)))) == int(floor(np.log10(np.abs(error_low)))):
            str_value = "%.*f" % (abs(power10), value)
            str_error_high = f"{error_high:.1g}"
            str_error_low = f"{error_low:.1g}"
            if std is not None:
                str_std = f"{std:.1g}"
        elif int(floor(np.log10(np.abs(error_high)))) > int(floor(np.log10(np.abs(error_low)))):
            str_value = "%.*f" % (abs(power10), value)
            str_error_high = f"{error_high:.2g}"
            str_error_low = f"{error_low:.1g}"
            if std is not None:
                str_std = f"{std:.2g}"
        else:
            str_value = "%.*f" % (abs(power10), value)
            str_error_high = f"{error_high:.1g}"
            str_error_low = f"{error_low:.2g}"
            if std is not None:
                str_std = f"{std:.2g}"
    out += [str_value, str_error_high]
    if not np.isclose(error_high, error_low):
        out += [str_error_low]
    if std is not None:
        out += [str_std]
    out = tuple(out)
    return out


def pixel_rotation(x, y, theta, x0=0, y0=0):
    """Rotate a 2D vector (x,y) of an angle theta clockwise.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    theta: float
        angle in radians
    x0: float, optional
        x position of the center of rotation (default: 0)
    y0: float, optional
        y position of the center of rotation (default: 0)

    Returns
    -------
    u: float
        rotated x coordinate
    v: float
        rotated y coordinate

    Examples
    --------
    >>> pixel_rotation(0, 0, 45)
    (0.0, 0.0)
    >>> u, v = pixel_rotation(1, 0, np.pi/4)
    >>> assert np.isclose(u, 1/np.sqrt(2))
    >>> assert np.isclose(v, -1/np.sqrt(2))
    >>> u, v = pixel_rotation(1, 2, -np.pi/2, x0=1, y0=0)
    >>> assert np.isclose(u, -2)
    >>> assert np.isclose(v, 0)
    """
    u = np.cos(theta) * (x - x0) + np.sin(theta) * (y - y0)
    v = -np.sin(theta) * (x - x0) + np.cos(theta) * (y - y0)
    return u, v


def detect_peaks(image):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise).
    Only positive peaks are detected (take absolute value or negative value of the
    image to detect the negative ones).

    Parameters
    ----------
    image: array_like
        The image 2D array.

    Returns
    -------
    detected_peaks: array_like
        Boolean maskof the peaks.

    Examples
    --------
    >>> im = np.zeros((50,50))
    >>> im[4,6] = 2
    >>> im[10,20] = -3
    >>> im[49,49] = 1
    >>> detected_peaks = detect_peaks(im)
    >>> assert detected_peaks[4,6]
    >>> assert not detected_peaks[10,20]
    >>> assert detected_peaks[49,49]
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=50)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def clean_target_spikes(data, saturation):
    saturated_pixels = np.where(data > saturation)
    data[saturated_pixels] = saturation
    NY, NX = data.shape
    delta = len(saturated_pixels[0])
    while delta > 0:
        delta = len(saturated_pixels[0])
        grady, gradx = np.gradient(data)
        for iy in range(1, NY - 1):
            for ix in range(1, NX - 1):
                # if grady[iy,ix]  > 0.8*np.max(grady) :
                #    data[iy,ix] = data[iy-1,ix]
                # if grady[iy,ix]  < 0.8*np.min(grady) :
                #    data[iy,ix] = data[iy+1,ix]
                if gradx[iy, ix] > 0.8 * np.max(gradx):
                    data[iy, ix] = data[iy, ix - 1]
                if gradx[iy, ix] < 0.8 * np.min(gradx):
                    data[iy, ix] = data[iy, ix + 1]
        saturated_pixels = np.where(data >= saturation)
        delta = delta - len(saturated_pixels[0])
    return data


def plot_image_simple(ax, data, scale="lin", title="", units="Image units", cmap="jet",
                      target_pixcoords=None, vmin=None, vmax=None, aspect=None, cax=None):
    """Simple function to plot a spectrum with error bars and labels.

    Parameters
    ----------
    ax: Axes
        Axes instance to make the plot
    data: array_like
        The image data 2D array.
    scale: str
        Scaling of the image (choose between: lin, log or log10) (default: lin)
    title: str
        Title of the image (default: "")
    units: str
        Units of the image to be written in the color bar label (default: "Image units")
    cmap: colormap
        Color map label (default: None)
    target_pixcoords: array_like, optional
        2D array  giving the (x,y) coordinates of the targets on the image: add a scatter plot (default: None)
    vmin: float
        Minimum value of the image (default: None)
    vmax: float
        Maximum value of the image (default: None)
    aspect: str
        Aspect keyword to be passed to imshow (default: None)
    cax: Axes, optional
        Color bar axes if necessary (default: None).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from spectractor.extractor.images import Image
    >>> f, ax = plt.subplots(1,1)
    >>> im = Image('tests/data/reduc_20170605_028.fits')
    >>> plot_image_simple(ax, im.data, scale="log10", units="ADU", target_pixcoords=(815,580),
    ...                     title="tests/data/reduc_20170605_028.fits")
    >>> if parameters.DISPLAY: plt.show()
    """
    if scale == "log" or scale == "log10":
        # removes the zeros and negative pixels first
        zeros = np.where(data <= 0)
        min_noz = np.min(data[np.where(data > 0)])
        data[zeros] = min_noz
        # apply log
        data = np.log10(data)

    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    ax.grid(color='silver', ls='solid')
    ax.grid(True)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    cb = plt.colorbar(im, ax=ax, cax=cax)
    cb.formatter.set_powerlimits((0, 0))
    cb.locator = MaxNLocator(7, prune=None)
    cb.update_ticks()
    cb.set_label('%s (%s scale)' % (units, scale))  # ,fontsize=16)
    if title != "":
        ax.set_title(title)
    if target_pixcoords is not None:
        ax.scatter(target_pixcoords[0], target_pixcoords[1], marker='o', s=100, edgecolors='k', facecolors='none',
                   label='Target', linewidth=2)


def plot_spectrum_simple(ax, lambdas, data, data_err=None, xlim=None, color='r', label='', title='', units=''):
    """Simple function to plot a spectrum with error bars and labels.

    Parameters
    ----------
    ax: Axes
        Axes instance to make the plot
    xlim: list, optional
        List of minimum and maximum abscisses
    color: str
        String for the color of the spectrum (default: 'r')
    label: str
        String label for the plot legend
    lambdas: array, optional
        The wavelengths array if it has been given externally (default: None)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from spectractor.extractor.spectrum import Spectrum
    >>> f, ax = plt.subplots(1,1)
    >>> s = Spectrum(file_name='tests/data/reduc_20170605_028_spectrum.fits')
    >>> plot_spectrum_simple(ax, s.lambdas, s.data, data_err=s.err, xlim=[500,700], color='r', label='test')
    >>> if parameters.DISPLAY: plt.show()
    """
    xs = lambdas
    if xs is None:
        xs = np.arange(data.size)
    if data_err is not None:
        ax.errorbar(xs, data, yerr=data_err, fmt=f'{color}o', lw=1, label=label, zorder=0, markersize=2)
    else:
        ax.plot(xs, data, f'{color}-', lw=2, label=label)
    ax.grid(True)
    if xlim is None and lambdas is not None:
        xlim = [parameters.LAMBDA_MIN, parameters.LAMBDA_MAX]
    ax.set_xlim(xlim)
    ax.set_ylim(0., np.nanmax(data) * 1.2)
    if lambdas is not None:
        ax.set_xlabel('$\lambda$ [nm]')
    else:
        ax.set_xlabel('X [pixels]')
    if units != '':
        ax.set_ylabel(f'Flux [{units}]')
    else:
        ax.set_ylabel(f'Flux')
    if title != '':
        ax.set_title(title)


def load_fits(file_name, hdu_index=0):
    hdu_list = fits.open(file_name)
    header = hdu_list[0].header
    data = hdu_list[hdu_index].data
    hdu_list.close()  # need to free allocation for file descripto
    return header, data


def extract_info_from_CTIO_header(obj, header):
    obj.date_obs = header['DATE-OBS']
    obj.airmass = header['AIRMASS']
    obj.expo = header['EXPTIME']
    obj.filters = header['FILTERS']
    obj.filter_label = header['FILTER1']
    obj.disperser_label = header['FILTER2']

def extract_info_from_PDM_header(obj, header):
    obj.date_obs = header['DATE-OBS']
    # SDC : note AIRMASS is in logbook, but it will be moved later in reduced image
    obj.airmass = header['AIRMASS']
    # SDC: note reduction already divides by exposure
    obj.expo = header['EXPOSURE']
    # SDC: note information is more in the filename
    obj.filters = header['FILTER']
    obj.filter_label = header['FILTER']
    obj.disperser_label = header['FILTER']


def save_fits(file_name, header, data, overwrite=False):
    hdu = fits.PrimaryHDU()
    hdu.header = header
    hdu.data = data
    output_directory = '/'.join(file_name.split('/')[:-1])
    ensure_dir(output_directory)
    hdu.writeto(file_name, overwrite=overwrite)


def dichotomie(f, a, b, epsilon):
    """
    Dichotomie method to find a function root.

    Parameters
    ----------
    f: callable
        The function
    a: float
        Left bound to the expected root
    b: float
        Right bound to the expected root
    epsilon: float
        Precision

    Returns
    -------
    root: float
        The root of the function.

    Examples
    --------

    Search for the Gaussian FWHM:
    >>> p = [1,0,1]
    >>> xx = np.arange(-10,10,0.1)
    >>> PSF = gauss(xx, *p)
    >>> def eq(x):
    ...     return np.interp(x, xx, PSF) - 0.5
    >>> root = dichotomie(eq, 0, 10, 1e-6)
    >>> assert np.isclose(2*root, 2.355*p[2], 1e-3)
    """
    x = 0.5 * (a + b)
    N = 1
    while b - a > epsilon and N < 100:
        x = 0.5 * (a + b)
        if f(x) * f(a) > 0:
            a = x
        else:
            b = x
        N += 1
    return x


def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if 380 <= wavelength <= 750:
        A = 1.
    else:
        A = 0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength > 750:
        wavelength = 750.
    if 380 <= wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B, A)


def from_lambda_to_colormap(lambdas):
    colorlist = [wavelength_to_rgb(lbda) for lbda in lambdas]
    spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)
    return spectralmap


if __name__ == "__main__":
    import doctest
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    doctest.testmod()


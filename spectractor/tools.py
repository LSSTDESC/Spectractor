import os
import shutil
from photutils.detection import IRAFStarFinder
from scipy.optimize import curve_fit
import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.io import fits
from astropy import wcs as WCS

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import MaxNLocator

import json
import warnings
from scipy.signal import fftconvolve, gaussian
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_erosion
from scipy.interpolate import interp1d
from scipy.integrate import quad

from skimage.feature import hessian_matrix
from spectractor.config import set_logger
from spectractor import parameters
from math import floor

from numba import njit


_SCIKIT_IMAGE_NEW_HESSIAN = None


# do not increase speed:
# @njit(fastmath=True, cache=True)
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
    return A * np.exp(-(x - x0) * (x - x0) / (2 * sigma * sigma))


# do not increase speed:
# @njit(fastmath=True, cache=True)
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
    >>> print(np.array(jac).T.shape)
    (50, 3)
    """
    dA = gauss(x, 1, x0, sigma)
    dx0 = A * (x - x0) / (sigma * sigma) * dA
    dsigma = A * (x - x0) * (x - x0) / (sigma ** 3) * dA
    # return np.array([dA, dx0, dsigma]).T
    return dA, dx0, dsigma


@njit(fastmath=True, cache=True)
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

    .. doctest::
        :hide:

        >>> assert np.all(np.isclose(p,popt))
    """
    def gauss_jacobian_wrapper(*params):
        return np.array(gauss_jacobian(*params)).T
    popt, pcov = curve_fit(gauss, x, y, p0=guess, bounds=bounds, tr_solver='exact', jac=gauss_jacobian_wrapper,
                           sigma=sigma, method='dogbox', verbose=0, xtol=1e-15, ftol=1e-15)
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
    [  1.  10.  20. 650.   3.  40. 750.  10.]
    """
    maxfev = 1000
    popt, pcov = curve_fit(multigauss_and_line, x, y, p0=guess, bounds=bounds, maxfev=maxfev, absolute_sigma=True)
    return popt, pcov


def rescale_x_to_legendre(x):
    """Rescale array between -1 and 1 for Legendre polynomial evaluation.

    Parameters
    ----------
    x: np.ndarray

    Returns
    -------
    x_norm: np.ndarray

    See Also
    --------
    rescale_x_from_legendre

    Examples
    --------
    >>> x = np.linspace(0, 10, 101)
    >>> x_norm = rescale_x_to_legendre(x)
    >>> x_norm[0], x_norm[-1], x_norm.size
    (-1.0, 1.0, 101)
    >>> x_new = rescale_x_from_legendre(x_norm, 0, 10)
    >>> assert np.allclose(x_new, x)

    """
    middle = 0.5 * (np.max(x) + np.min(x))
    x_norm = x - middle
    if np.max(x_norm) != 0:
        return x_norm / np.max(x_norm)
    else:
        return x_norm


def rescale_x_from_legendre(x_norm, Xmin, Xmax):
    """Rescale normalized array set between -1 and 1 for Legendre polynomial evaluation to normal array
    between Xmin and Xmax.

    Parameters
    ----------
    x: np.ndarray
    Xmin: float
    Xmax: float

    Returns
    -------
    x: np.ndarray

    See Also
    --------
    rescale_x_to_legendre

    Examples
    --------
    >>> x = np.linspace(0, 10, 101)
    >>> x_norm = rescale_x_to_legendre(x)
    >>> x_norm[0], x_norm[-1], x_norm.size
    (-1.0, 1.0, 101)
    >>> x_new = rescale_x_from_legendre(x_norm, 0, 10)
    >>> assert np.allclose(x_new, x)

    """
    X = 0.5 * x_norm * (Xmax - Xmin) + 0.5 * (Xmax + Xmin)
    return X


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
    >>> parameters.CALIB_BGD_NPARAMS = 4
    >>> x = np.arange(600., 800., 1)
    >>> x_norm = rescale_x_to_legendre(x)
    >>> p = [20, 0, 0, 0, 20, 650, 3, 40, 750, 5]
    >>> y = multigauss_and_bgd(np.array([x_norm, x]), *p)
    >>> print(f'{y[0]:.2f}')
    20.00
    >>> print(f'{np.max(y):.2f}')
    60.00
    >>> print(f'{np.argmax(y)}')
    150

    .. plot::

        from spectractor import parameters
        from spectractor.tools import multigauss_and_bgd
        import numpy as np
        parameters.CALIB_BGD_NPARAMS = 4
        x = np.arange(600., 800., 1)
        x_norm = rescale_x_to_legendre(x)
        p = [20, 0, 0, 0, 20, 650, 3, 40, 750, 5]
        y = multigauss_and_bgd(np.array([x_norm, x]), *p)
        plt.plot(x,y,'r-')
        plt.show()

    """
    bgd_nparams = parameters.CALIB_BGD_NPARAMS
    x_norm, x_gauss = x
    # x_norm = rescale_x_to_legendre(x)
    out = np.polynomial.legendre.legval(x_norm, params[0:bgd_nparams])
    # out = np.polyval(params[0:bgd_nparams], x)
    for k in range((len(params) - bgd_nparams) // 3):
        out += gauss(x_gauss, *params[bgd_nparams + 3 * k:bgd_nparams + 3 * k + 3])
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
    >>> x_norm = rescale_x_to_legendre(x)
    >>> p = [20, 0, 0, 0, 20, 650, 3, 40, 750, 5]
    >>> jac = multigauss_and_bgd_jacobian(np.array([x_norm, x]), *p)
    >>> assert(np.all(np.isclose(jac.T[0],np.ones_like(x))))
    >>> print(jac.shape)
    (200, 10)
    """
    bgd_nparams = parameters.CALIB_BGD_NPARAMS
    x_norm, x_gauss = x
    out = np.zeros((len(params), len(x_norm)))
    # x_norm = rescale_x_to_legendre(x)
    for k in range(0, bgd_nparams):
        # out[k] = x ** (bgd_nparams - 1 - k)
        c = np.zeros(bgd_nparams)
        c[k] = 1
        out[k] = np.polynomial.legendre.legval(x_norm, c)  # np.eye(1, bgd_nparams, k)[0])
    for ngauss in range((len(params) - bgd_nparams) // 3):
        out[bgd_nparams + 3 * ngauss:bgd_nparams + 3 * ngauss + 3] = gauss_jacobian(x_gauss, *params[bgd_nparams + 3 * ngauss:bgd_nparams + 3 * ngauss + 3])
    return out.T


# noinspection PyTypeChecker
def fit_multigauss_and_bgd(x, y, guess=[0, 1, 10, 1000, 1, 0], bounds=(-np.inf, np.inf), sigma=None):
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

    >>> from spectractor.config import load_config
    >>> load_config("default.ini")
    >>> x = np.arange(600.,800.,1)
    >>> x_norm = rescale_x_to_legendre(x)
    >>> p = [20, 0, 0, 0, 20, 650, 3, 40, 750, 5]
    >>> y = multigauss_and_bgd(np.array([x_norm, x]), *p)
    >>> print(f'{y[0]:.2f}')
    20.00
    >>> err = 0.1 * np.sqrt(y)
    >>> guess = (10,0,0,0.1,10,640,2,20,750,7)
    >>> bounds = ((-np.inf,-np.inf,-np.inf,-np.inf,1,600,1,1,600,1),(np.inf,np.inf,np.inf,np.inf,100,800,100,100,800,100))
    >>> popt, pcov = fit_multigauss_and_bgd(x, y, guess=guess, bounds=bounds, sigma=err)
    >>> assert np.allclose(p,popt,rtol=1e-4, atol=1e-5)
    >>> fit = multigauss_and_bgd(np.array([x_norm, x]), *popt)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from spectractor.tools import multigauss_and_bgd, fit_multigauss_and_bgd
        x = np.arange(600.,800.,1)
        x_norm = rescale_x_to_legendre(x)
        p = [20, 0, 0, 0, 20, 650, 3, 40, 750, 5]
        y = multigauss_and_bgd(np.array([x_norm, x]), *p)
        err = 0.1 * np.sqrt(y)
        guess = (10,0,0,0.1,10,640,2,20,750,7)
        bounds = ((-np.inf,-np.inf,-np.inf,-np.inf,1,600,1,1,600,1),(np.inf,np.inf,np.inf,np.inf,100,800,100,100,800,100))
        popt, pcov = fit_multigauss_and_bgd(x, y, guess=guess, bounds=bounds, sigma=err)
        fit = multigauss_and_bgd(np.array([x_norm, x]), *popt)
        fig = plt.figure()
        plt.errorbar(x,y,yerr=err,linestyle='None',label="data")
        plt.plot(x,fit,'r-',label="best fit")
        plt.plot(x,multigauss_and_bgd(np.array([x_norm, x]), *guess),'k--',label="guess")
        plt.legend()
        plt.show()
    """
    maxfev = 10000
    x_norm = rescale_x_to_legendre(x)
    popt, pcov = curve_fit(multigauss_and_bgd, np.array([x_norm, x]), y, p0=guess, bounds=bounds, maxfev=maxfev, sigma=sigma,
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

    .. doctest::
        :hide:

        >>> assert np.all(np.isclose(p, fit, 1e-5))
        >>> assert np.all(np.isclose(model, y))
        >>> assert cov.shape == (4, 4)

    With uncertainties:

    >>> fit, cov2, model2 = fit_poly1d(x, y, order=3, w=err)

    .. doctest::
        :hide:

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

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from spectractor.tools import fit_poly1d_legendre
        p = [-1e-6, -1e-4, 1, 1]
        x = np.arange(500., 1000., 1)
        y = np.polyval(p, x)
        err = np.ones_like(y)
        fit, cov2, model2 = fit_poly1d_legendre(x,y,order=3,w=err)
        plt.errorbar(x,y,yerr=err,fmt='ro')
        plt.plot(x,model2)
        plt.show()
    """
    cov = -1
    x_norm = rescale_x_to_legendre(x)
    if len(x) > order:
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

    .. doctest::
        :hide:

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
    gg_init = models.Polynomial1D(order)
    gg_init.c1 = 0
    gg_init.c2 = 0
    fit = fitting.LinearLSQFitter()
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
    # get fitted model and filtered data
    or_fitted_model, filtered_data = or_fit(gg_init, x, y)
    outliers = []  # not working
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(x, y, 'gx', label="original data")
    plt.plot(x, gg_init(x), 'k.', label="guess")
    plt.plot(x, filtered_data, 'r+', label="filtered data")
    plt.plot(x, or_fitted_model(x), 'r--',
             label="model fitted w/ filtered data")
    plt.legend(loc=2, numpoints=1)
    if parameters.DISPLAY: plt.show()
    """
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

    .. doctest::
        :hide:

        >>> assert np.isclose(fit.c0_0.value, 0)
        >>> assert np.isclose(fit.c1_0.value, 0)
        >>> assert np.isclose(fit.c2_0.value, 1)
        >>> assert np.isclose(fit.c0_1.value, 0)
        >>> assert np.isclose(fit.c0_2.value, 1)
        >>> assert np.isclose(fit.c1_1.value, -2)

    """
    my_logger = set_logger(__name__)
    gg_init = models.Polynomial2D(order)
    fit = fitting.LinearLSQFitter()
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
    """
    Fit an astropy Gaussian 2D model with parameters : amplitude, x_mean, y_mean, x_stddev, y_stddev, theta
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

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.modeling import models
        X, Y = np.mgrid[:50,:50]
        PSF = models.Gaussian2D()
        p = (50, 25, 25, 5, 5, 0)
        Z = PSF.evaluate(X, Y, *p)
        plt.imshow(Z, origin='lower')
        plt.show()

    >>> guess = (45, 20, 20, 7, 7, 0)
    >>> bounds = ((1, 10, 10, 1, 1, -90), (100, 40, 40, 10, 10, 90))
    >>> fit = fit_gauss2d_outlier_removal(X, Y, Z, guess=guess, bounds=bounds, circular=True)
    >>> res = [getattr(fit, p).value for p in fit.param_names]
    >>> print(res)
    [50.0, 25.0, 25.0, 5.0, 5.0, 0.0]

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.modeling import models
        from spectractor.tools import fit_gauss2d_outlier_removal
        X, Y = np.mgrid[:50,:50]
        PSF = models.Gaussian2D()
        p = (50, 25, 25, 5, 5, 0)
        Z = PSF.evaluate(X, Y, *p)
        guess = (45, 20, 20, 7, 7, 0)
        bounds = ((1, 10, 10, 1, 1, -90), (100, 40, 40, 10, 10, 90))
        fit = fit_gauss2d_outlier_removal(X, Y, Z, guess=guess, bounds=bounds, circular=True)
        plt.imshow(Z-fit(X, Y), origin='lower')
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
    """
    Fit an astropy Moffat 2D model with parameters: amplitude, x_mean, y_mean, gamma, alpha
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

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.modeling import models
        X, Y = np.mgrid[:100,:100]
        PSF = models.Moffat2D()
        p = (50, 50, 50, 5, 2)
        Z = PSF.evaluate(X, Y, *p)
        plt.imshow(Z, origin='loxer')
        plt.show()

    >>> guess = (45, 48, 52, 4, 2)
    >>> bounds = ((1, 10, 10, 1, 1), (100, 90, 90, 10, 10))
    >>> fit = fit_moffat2d_outlier_removal(X, Y, Z, guess=guess, bounds=bounds, niter=3)
    >>> res = [getattr(fit, p).value for p in fit.param_names]

    .. doctest::
        :hide:

        >>> assert(np.all(np.isclose(p, res, 1e-1)))

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.modeling import models
        from spectractor.tools import fit_moffat2d_outlier_removal
        X, Y = np.mgrid[:100,:100]
        PSF = models.Moffat2D()
        p = (50, 50, 50, 5, 2)
        Z = PSF.evaluate(X, Y, *p)
        guess = (45, 48, 52, 4, 2)
        bounds = ((1, 10, 10, 1, 1), (100, 90, 90, 10, 10))
        fit = fit_moffat2d_outlier_removal(X, Y, Z, guess=guess, bounds=bounds, niter=3)
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
    """
    Fit an astropy Moffat 1D model with parameters: amplitude, x_mean, gamma, alpha
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

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.modeling import models
        X = np.arange(100)
        PSF = models.Moffat1D()
        p = (50, 50, 5, 2)
        Y = PSF.evaluate(X, *p)
        plt.plot(X, Y)
        plt.show()

    >>> guess = (45, 48, 4, 2)
    >>> bounds = ((1, 10, 1, 1), (100, 90, 10, 10))
    >>> fit = fit_moffat1d_outlier_removal(X, Y, guess=guess, bounds=bounds, niter=3)
    >>> res = [getattr(fit, p).value for p in fit.param_names]

    .. doctest::
        :hide:

        >>> assert(np.all(np.isclose(p, res, 1e-6)))

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.modeling import models
        from spectractor.tools import fit_moffat1d_outlier_removal
        X = np.arange(100)
        PSF = models.Moffat1D()
        p = (50, 50, 5, 2)
        Y = PSF.evaluate(X, *p)
        guess = (45, 48, 4, 2)
        bounds = ((1, 10, 1, 1), (100, 90, 10, 10))
        fit = fit_moffat1d_outlier_removal(X, Y, guess=guess, bounds=bounds, niter=3)
        plt.plot(X, Y-fit(X))
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
        my_logger.debug(f'\n\t{or_fitted_model}')
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

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.modeling import models
        X = np.arange(100)
        PSF = models.Moffat1D()
        p = (50, 50, 5, 2)
        Y = PSF.evaluate(X, *p)
        plt.plot(X, Y)
        plt.show()

    >>> guess = (45, 48, 4, 2)
    >>> bounds = ((1, 10, 1, 1), (100, 90, 10, 10))
    >>> fit = fit_moffat1d(X, Y, guess=guess, bounds=bounds)
    >>> res = [getattr(fit, p).value for p in fit.param_names]
    >>> assert(np.all(np.isclose(p, res, 1e-6)))

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.modeling import models
        from spectractor.tools import fit_moffat1d
        X = np.arange(100)
        PSF = models.Moffat1D()
        p = (50, 50, 5, 2)
        Y = PSF.evaluate(X, *p)
        guess = (45, 48, 4, 2)
        bounds = ((1, 10, 1, 1), (100, 90, 10, 10))
        fit = fit_moffat1d(X, Y, guess=guess, bounds=bounds)
        plt.plot(X, Y-fit(X))
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


def compute_fwhm(x, y, minimum=0, center=None, full_output=False, epsilon=1e-3):
    """
    Compute the full width half maximum of y(x) curve,
    using an interpolation of the data points and dichotomie method.

    Parameters
    ----------
    x: array_like
        The abscisse array.
    y: array_like
        The function array.
    minimum: float, optional
        The minimum reference from which to compyte half the height (default: 0).
    center: float, optional
        The center of the curve. If None, the weighted averageof the y(x) distribution is computed (default: None).
    full_output: bool, optional
        If True, half maximum, the edges of the curve and the curve center are given in output (default: False).
    epsilon: float, optional
        Dichotomie algorithm stop if difference is smaller than epsilon (default: 1e-3).

    Returns
    -------
    FWHM: float
        The full width half maximum of the curve.
    half: float, optional
        The half maximum value. Only if full_output=True.
    center: float, optional
        The y(x) center value. Only if full_output=True.
    left_edge: float, optional
        The left_edge value at half maximum. Only if full_output=True.
    right_edge: float, optional
        The right_edge value at half maximum. Only if full_output=True.

    Examples
    --------

    Gaussian example

    >>> x = np.arange(0, 100, 1)
    >>> stddev = 4
    >>> middle = 40
    >>> psf = gauss(x, 1, middle, stddev)
    >>> fwhm, half, center, a, b = compute_fwhm(x, psf, full_output=True)
    >>> print(f"{fwhm:.4f} {2.355*stddev:.4f} {center:.4f}")
    9.4329 9.4200 40.0000

    .. doctest::
        :hide:

        >>> assert np.isclose(fwhm, 2.355*stddev, atol=2e-1)
        >>> assert np.isclose(center, middle, atol=1e-3)

    .. plot ::

        import matplotlib.pyplot as plt
        import numpy as np
        from spectractor.tools import gauss, compute_fwhm
        x = np.arange(0, 100, 1)
        stddev = 4
        middle = 40
        psf = gauss(x, 1, middle, stddev)
        fwhm, half, center, a, b = compute_fwhm(x, psf, full_output=True)
        plt.figure()
        plt.plot(x, psf, label="function")
        plt.axvline(center, color="gray", label="center")
        plt.axvline(a, color="k", label="edges at half max")
        plt.axvline(b, color="k", label="edges at half max")
        plt.axhline(half, color="r", label="half max")
        plt.legend()
        plt.title(f"FWHM={fwhm:.3f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    Defocused PSF example

    >>> from spectractor.extractor.psf import MoffatGauss
    >>> p = [2,40,40,4,2,-0.4,1,10]
    >>> psf = MoffatGauss(p)
    >>> fwhm, half, center, a, b = compute_fwhm(x, psf.evaluate(x), full_output=True)

    .. doctest::
        :hide:

        >>> assert np.isclose(fwhm, 7.05, atol=1e-2)
        >>> assert np.isclose(center, p[1], atol=1e-2)

    .. plot ::

        import matplotlib.pyplot as plt
        import numpy as np
        from spectractor.tools import gauss, compute_fwhm
        from spectractor.extractor.psf import MoffatGauss
        x = np.arange(0, 100, 1)
        p = [2,40,40,4,2,-0.4,1,10]
        psf = MoffatGauss(p)
        fwhm, half, center, a, b = compute_fwhm(x, psf.evaluate(x), full_output=True)
        plt.figure()
        plt.plot(x, psf.evaluate(x, p), label="function")
        plt.axvline(center, color="gray", label="center")
        plt.axvline(a, color="k", label="edges at half max")
        plt.axvline(b, color="k", label="edges at half max")
        plt.axhline(half, color="r", label="half max")
        plt.legend()
        plt.title(f"FWHM={fwhm:.3f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    """
    if y.ndim > 1:
        # TODO: implement fwhm for 2D curves
        return -1
    interp = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    maximum = np.max(y) - minimum
    imax = np.argmax(y)
    a = x[imax + np.argmin(np.abs(y[imax:] - 0.9 * maximum))]
    b = x[imax + np.argmin(np.abs(y[imax:] - 0.1 * maximum))]

    def eq(xx):
        return interp(xx) - 0.5 * maximum

    res = dichotomie(eq, a, b, epsilon)
    if center is None:
        center = np.average(x, weights=y)
    fwhm = abs(2 * (res - center))
    if not full_output:
        return fwhm
    else:
        return fwhm, 0.5 * maximum, center, res, center - abs(res - center)


def compute_integral(x, y, bounds=None):
    """
    Compute the integral of an y(x) curve. The curve is interpolated and extrapolated with cubic splines.
    If not provided, bounds are set to the x array edges.

    Parameters
    ----------
    x: array_like
        The abscisse array.
    y: array_like
        The function array.
    bounds: array_like, optional
        The bounds of the integral. If None, the edges of thex array are taken (default bounds=None).

    Returns
    -------
    result: float
        The integral of the PSF model.

    Examples
    --------

    Gaussian example

    .. doctest::

        >>> x = np.arange(0, 100, 1)
        >>> stddev = 4
        >>> middle = 40
        >>> psf = gauss(x, 1/(stddev*np.sqrt(2*np.pi)), middle, stddev)
        >>> integral = compute_integral(x, psf)
        >>> print(f"{integral:.6f}")
        1.000000

    Defocused PSF example

    .. doctest::

        >>> from spectractor.extractor.psf import MoffatGauss
        >>> p = [2,30,30,4,2,-0.5,1,10]
        >>> psf = MoffatGauss(p)
        >>> integral = compute_integral(x, psf.evaluate(x))
        >>> assert np.isclose(integral, p[0], atol=1e-2)

    """
    if bounds is None:
        bounds = (np.min(x), np.max(x))
    interp = interp1d(x, y, kind="cubic", bounds_error=False, fill_value="extrapolate")
    integral = quad(interp, bounds[0], bounds[1], limit=200)
    return integral[0]


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
    # Check for unannounced API change on hessian_matrix in scikit-image>=0.20
    # See https://github.com/scikit-image/scikit-image/pull/6624
    global _SCIKIT_IMAGE_NEW_HESSIAN

    if _SCIKIT_IMAGE_NEW_HESSIAN is None:
        from importlib import metadata
        import packaging

        vers = packaging.version.parse(metadata.version("scikit-image"))
        if vers < packaging.version.parse("0.20.0"):
            _SCIKIT_IMAGE_NEW_HESSIAN = False
        else:
            _SCIKIT_IMAGE_NEW_HESSIAN = True

    # compute hessian matrices on the image
    order = "xy" if _SCIKIT_IMAGE_NEW_HESSIAN else "rc"
    Hxx, Hxy, Hyy = hessian_matrix(data, sigma=3, order=order)
    lambda_plus = 0.5 * ((Hxx + Hyy) + np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy * Hxy))
    lambda_minus = 0.5 * ((Hxx + Hyy) - np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy * Hxy))
    theta = 0.5 * np.arctan2(2 * Hxy, Hxx - Hyy) * 180 / np.pi
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
    [0.5        0.63114657 0.74850168]
    >>> array = np.ones((100, 100))
    >>> output = fftconvolve_gaussian(array, 3)
    >>> print(output[0][:3])
    [0.5        0.63114657 0.74850168]
    >>> array = np.ones((100, 100, 100))
    >>> output = fftconvolve_gaussian(array, 3)
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
        my_logger.error(f'\n\tArray dimension must be 1 or 2. Here I have array.ndim={array.ndim}.')
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
    # if not np.isclose(error_high, error_low):
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

    .. doctest::
        :hide:

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

    .. doctest::
        :hide:

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


def clean_target_spikes(data, saturation):  # pragma: no cover
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


def plot_image_simple(ax, data, scale="lin", title="", units="Image units", cmap=None,
                      target_pixcoords=None, vmin=None, vmax=None, aspect=None, cax=None):
    """Simple function to plot a spectrum with error bars and labels.

    Parameters
    ----------
    ax: Axes
        Axes instance to make the plot
    data: array_like
        The image data 2D array.
    scale: str
        Scaling of the image (choose between: lin, log or log10, symlog) (default: lin)
    title: str
        Title of the image (default: "")
    units: str
        Units of the image to be written in the color bar label (default: "Image units")
    cmap: colormap
        Color map label (default: None)
    target_pixcoords: array_like, optional
        2D array giving the (x,y) coordinates of the targets on the image: add a scatter plot (default: None)
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

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from spectractor.extractor.images import Image
        >>> from spectractor import parameters
        >>> from spectractor.tools import plot_image_simple
        >>> f, ax = plt.subplots(1,1)
        >>> im = Image('tests/data/reduc_20170605_028.fits', config="./config/ctio.ini")
        >>> plot_image_simple(ax, im.data, scale="symlog", units="ADU", target_pixcoords=(815,580),
        ...                     title="tests/data/reduc_20170605_028.fits")
        >>> if parameters.DISPLAY: plt.show()
    """
    if scale == "log" or scale == "log10":
        # removes the zeros and negative pixels first
        zeros = np.where(data <= 0)
        min_noz = np.min(data[np.where(data > 0)])
        data[zeros] = min_noz
        # apply log
        # data = np.log10(data)
    if scale == "log10" or scale == "log":
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    elif scale == "symlog":
        norm = matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=10, base=10)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(data, origin='lower', cmap=cmap, norm=norm, aspect=aspect)
    ax.grid(color='silver', ls='solid')
    ax.grid(True)
    ax.set_xlabel(parameters.PLOT_XLABEL)
    ax.set_ylabel(parameters.PLOT_YLABEL)
    cb = plt.colorbar(im, ax=ax, cax=cax)
    if scale == "lin":
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7, prune=None)
        cb.update_ticks()
        cb.set_label(f'{units}')  # ,fontsize=16)
    else:
        cb.set_label(f'{units} ({scale} scale)')  # ,fontsize=16)
    if title != "":
        ax.set_title(title)
    if target_pixcoords is not None:
        ax.scatter(target_pixcoords[0], target_pixcoords[1], marker='o', s=100, edgecolors='k', facecolors='none',
                   label='Target', linewidth=2)


def plot_spectrum_simple(ax, lambdas, data, data_err=None, xlim=None, color='r', linestyle='none', lw=2, label='',
                         title='', units='', marker='o'):
    """Simple function to plot a spectrum with error bars and labels.

    Parameters
    ----------
    ax: Axes
        Axes instance to make the plot.
    lambdas: array
        The wavelengths array.
    data: array
        The spectrum data array.
    data_err: array, optional
        The spectrum uncertainty array (default: None).
    xlim: list, optional
        List of minimum and maximum abscisses (default: None).
    color: str, optional
        String for the color of the spectrum (default: 'r').
    linestyle: str, optional
        String for the linestyle of the spectrum (default: 'none').
    lw: int, optional
        Integer for line width (default: 2).
    marker: str, optional
        Character for marker style (default: 'o').
    label: str, optional
        String label for the plot legend (default: '').
    title: str, optional
        String label for the plot title (default: '').
    units: str, optional
        String label for the plot units (default: '').


    Examples
    --------

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from spectractor.extractor.spectrum import Spectrum
        >>> from spectractor import parameters
        >>> from spectractor.tools import plot_spectrum_simple
        >>> f, ax = plt.subplots(1,1)
        >>> s = Spectrum(file_name='tests/data/reduc_20170530_134_spectrum.fits')
        >>> plot_spectrum_simple(ax, s.lambdas, s.data, data_err=s.err, xlim=None, color='r', label='test')
        >>> if parameters.DISPLAY: plt.show()
    """
    xs = lambdas
    if xs is None:
        xs = np.arange(data.size)
    if data_err is not None:
        ax.errorbar(xs, data, yerr=data_err, color=color, marker=marker, lw=lw, label=label,
                    zorder=0, markersize=2, linestyle=linestyle)
    else:
        ax.plot(xs, data, color=color, lw=lw, label=label, linestyle=linestyle)
    ax.grid(True)
    if xlim is None and lambdas is not None:
        xlim = [parameters.LAMBDA_MIN, parameters.LAMBDA_MAX]
    ax.set_xlim(xlim)
    try:
        ax.set_ylim(0., np.nanmax(data[np.logical_and(xs > xlim[0], xs < xlim[1])]) * 1.2)
    except ValueError:
        pass
    if lambdas is not None:
        ax.set_xlabel(r'$\lambda$ [nm]')
    else:
        ax.set_xlabel('X [pixels]')
    if units != '':
        ax.set_ylabel(f'Flux [{units}]')
    else:
        ax.set_ylabel(f'Flux')
    if title != '':
        ax.set_title(title)


def plot_compass_simple(ax, parallactic_angle=None, arrow_size=0.1, origin=[0.15, 0.15]):
    """Plot small (N,W) compass, and optionally zenith direction.

    Parameters
    ----------
    ax: Axes
        Axes instance to make the plot.
    parallactic_angle: float, optional
        Value is the parallactic angle with respect to North eastward and plot the zenith direction (default: None).
    arrow_size: float, optional
        Length of the arrow as a fraction of axe sizes (default: 0.1)
    origin: array_like, optional
        (x0, y0) position of the compass as axes fraction (default: [0.15, 0.15]).

    Examples
    --------

    >>> from spectractor.extractor.images import Image
    >>> from spectractor import parameters
    >>> from spectractor.tools import plot_image_simple, plot_compass_simple
    >>> f, ax = plt.subplots(1,1)
    >>> im = Image('tests/data/reduc_20170605_028.fits', config="./config/ctio.ini")
    >>> plot_image_simple(ax, im.data, scale="symlog", units="ADU", target_pixcoords=(750,700),
    ...                   title='tests/data/reduc_20170530_134.fits')
    >>> plot_compass_simple(ax, im.parallactic_angle)
    >>> if parameters.DISPLAY: plt.show()

    """
    # North arrow
    N_arrow = [0, arrow_size]
    N_xy = np.asarray(flip_and_rotate_radec_vector_to_xy_vector(N_arrow[0], N_arrow[1],
                                                                camera_angle=parameters.OBS_CAMERA_ROTATION,
                                                                flip_ra_sign=parameters.OBS_CAMERA_RA_FLIP_SIGN,
                                                                flip_dec_sign=parameters.OBS_CAMERA_DEC_FLIP_SIGN))
    ax.annotate("N", xy=origin, xycoords='axes fraction', xytext=N_xy + origin, textcoords='axes fraction',
                arrowprops=dict(arrowstyle="<|-", fc="yellow", ec="yellow"), color="yellow",
                horizontalalignment='center', verticalalignment='center')
    # West arrow
    W_arrow = [arrow_size, 0]
    W_xy = np.asarray(flip_and_rotate_radec_vector_to_xy_vector(W_arrow[0], W_arrow[1],
                                                                camera_angle=parameters.OBS_CAMERA_ROTATION,
                                                                flip_ra_sign=parameters.OBS_CAMERA_RA_FLIP_SIGN,
                                                                flip_dec_sign=parameters.OBS_CAMERA_DEC_FLIP_SIGN))
    ax.annotate("W", xy=origin, xycoords='axes fraction', xytext=W_xy + origin, textcoords='axes fraction',
                arrowprops=dict(arrowstyle="<|-", fc="yellow", ec="yellow"), color="yellow",
                horizontalalignment='center', verticalalignment='center')
    # Central dot
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.scatter(origin[0] * xmax, origin[1] * ymax, color="yellow", s=20)
    # Zenith direction
    if parallactic_angle is not None:
        p_arrow = [0, arrow_size]  # angle with respect to North in RADEC counterclockwise
        angle = parameters.OBS_CAMERA_ROTATION + parameters.OBS_CAMERA_RA_FLIP_SIGN * parallactic_angle
        p_xy = np.asarray(flip_and_rotate_radec_vector_to_xy_vector(p_arrow[0], p_arrow[1],
                                                                    camera_angle=angle,
                                                                    flip_ra_sign=parameters.OBS_CAMERA_RA_FLIP_SIGN,
                                                                    flip_dec_sign=parameters.OBS_CAMERA_DEC_FLIP_SIGN))
        ax.annotate("Z", xy=origin, xycoords='axes fraction', xytext=p_xy + origin, textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="<|-", fc="lightgreen", ec="lightgreen"), color="lightgreen",
                    horizontalalignment='center', verticalalignment='center')


def plot_table_in_axis(ax, table):
    def getRoundedValues(df):
        toReturn = []
        rows = [r for r in df.values]

        for row in rows:
            rowItems = []
            for itemNum, item in enumerate(row):
                if itemNum == 0:
                    rowItems.append(str(item))
                else:
                    assert isinstance(item, float)
                    rowItems.append(f'{item:.4}')
            toReturn.append(rowItems)
        return toReturn

    def getColumnNames(df):
        headerRow = [headerRowMap[col] for col in df.columns]
        return headerRow

    headerRowMap = {
        'Line': 'Line',
        'Tabulated': 'Tabulated\n(nm)',
        'Detected': 'Detected\n(nm)',
        'Shift': 'Shift\n(nm)',
        'Err': 'Err\n(nm)',
        'FWHM': 'FWHM\n(nm)',
        'Amplitude': 'Amplitude',
        'SNR': 'SNR',
        'Chisq': '${\chi}^2$',
        'Eqwidth_mod': 'Eq. Width\n(model)',
        'Eqwidth_data': 'Eq. Width\n(data)',
    }

    # Hide the axes
    ax.axis('off')

    table.convert_bytestring_to_unicode()
    df = table.to_pandas()

    # Create a table
    table_data = ax.table(cellText=getRoundedValues(df),
                          colLabels=getColumnNames(df),
                          cellLoc='left',
                          loc='center',
                          edges='horizontal')

    # Style the table
    table_data.auto_set_font_size(False)
    table_data.set_fontsize(10)
    table_data.scale(1.5, 1.1)  # Adjust the table size if needed

    # Set header row height to be double the default height
    header_cells = table_data.get_celld()
    for key, cell in header_cells.items():
        cell.set_text_props(fontfamily='serif', fontsize=10)
        if key[0] == 0:
            cell.set_text_props(weight='bold')
            cell.set_height(0.15)  # Adjust the height as needed
        if key[1] == 0:  # First column
            if key[0] == 0:  # first line of first column
                continue
            cell.set_text_props(fontstyle='italic', fontfamily='serif', fontsize=11)
            cell._text.set_horizontalalignment('left')


def load_fits(file_name, hdu_index=0):
    """Generic function to load a FITS file.

    Parameters
    ----------
    file_name: str
        The FITS file name.
    hdu_index: int, str, optional
        The HDU index in the file (default: 0).

    Returns
    -------
    header: fits.Header
        Header of the FITS file.
    data: np.array
        The data array.

    Examples
    --------
    >>> header, data = load_fits("./tests/data/reduc_20170530_134.fits")
    >>> header["DATE-OBS"]
    '2017-05-31T02:53:52.356'
    >>> data.shape
    (2048, 2048)

    """
    with fits.open(file_name) as hdu_list:
        header = hdu_list[hdu_index].header
        data = hdu_list[hdu_index].data
    return header, data


def save_fits(file_name, header, data, overwrite=False):
    """Generic function to save a FITS file.

    Parameters
    ----------
    file_name: str
        The FITS file name.
    header: fits.Header
        Header of the FITS file.
    data: np.array
        The data array.
    overwrite: bool, optional
        If True and the file already exists, it is overwritten (default: False).

    Examples
    --------

    >>> header, data = load_fits("./tests/data/reduc_20170530_134.fits")
    >>> save_fits("./outputs/save_fits_test.fits", header, data, overwrite=True)
    >>> assert os.path.isfile("./outputs/save_fits_test.fits")

    .. doctest:
        :hide:

        >>> os.remove("./outputs/save_fits_test.fits")

    """
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
    """ taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    """
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
    return R, G, B, A


def from_lambda_to_colormap(lambdas):
    """Convert an array of wavelength in nm into a color map.

    Parameters
    ----------
    lambdas: array_like
        Wavelength array in nm.

    Returns
    -------
    spectral_map: matplotlib.colors.LinearSegmentedColormap
        Color map.

    Examples
    --------
    >>> lambdas = np.arange(300, 1000, 10)
    >>> spec = from_lambda_to_colormap(lambdas)
    >>> plt.scatter(lambdas, np.zeros(lambdas.size), cmap=spec, c=lambdas)  #doctest: +ELLIPSIS
    <matplotlib.collections.PathCollection object at ...>
    >>> plt.grid()
    >>> plt.xlabel("Wavelength [nm]")  #doctest: +ELLIPSIS
    Text(..., 'Wavelength [nm]')
    >>> plt.show()

    ..plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from spectractor.tools import from_lambda_to_colormap
        lambdas = np.arange(300, 1000, 10)
        spec = from_lambda_to_colormap(lambdas)
        plt.scatter(lambdas, np.zeros(lambdas.size), cmap=spec, c=lambdas)
        plt.xlabel("Wavelength [nm]")
        plt.grid()
        plt.show()

    """
    colorlist = [wavelength_to_rgb(lbda) for lbda in lambdas]
    spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)
    return spectralmap


def rebin(arr, new_shape, FLAG_MAKESUM=True):
    """Rebin and reshape a numpy array.

    Parameters
    ----------
    arr: np.array
        Numpy array to be reshaped.
    new_shape: array_like
        New shape of the array.

    Returns
    -------
    arr_rebinned: np.array
        Rebinned array.

    Examples
    --------
    >>> a = 4 * np.ones((10, 10))
    >>> b = rebin(a, (5, 5))
    >>> b
    array([[4., 4., 4., 4., 4.],
           [4., 4., 4., 4., 4.],
           [4., 4., 4., 4., 4.],
           [4., 4., 4., 4., 4.],
           [4., 4., 4., 4., 4.]])
    """



    if np.any(new_shape * parameters.CCD_REBIN != arr.shape):
        shape_cropped = new_shape * parameters.CCD_REBIN
        margins = np.asarray(arr.shape) - shape_cropped
        arr = arr[:-margins[0], :-margins[1]]
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])

    if FLAG_MAKESUM:
        # SDC : conservation of energy
        return arr.reshape(shape).sum(-1).sum(1)

    else:
        # SDC : sure of conservation of energy
        return arr.reshape(shape).mean(-1).mean(1)


def set_wcs_output_directory(file_name, output_directory=""):
    """Returns the WCS output directory corresponding to the analyzed image. The name of the directory is
    the anme of the image with the suffix _wcs.

    Parameters
    ----------
    file_name: str
        File name of the image.
    output_directory: str, optional
        If not set, the main output directory is the one the image,
        otherwise the specified directory is taken (default: "").

    Returns
    -------
    output: str
        The name of the output directory

    Examples
    --------
    >>> set_wcs_output_directory("image.fits", output_directory="")
    'image_wcs'
    >>> set_wcs_output_directory("image.png", output_directory="outputs")
    'outputs/image_wcs'

    """
    outdir = os.path.dirname(file_name)
    if output_directory != "":
        outdir = output_directory
    output_directory = os.path.join(outdir, os.path.splitext(os.path.basename(file_name))[0]) + "_wcs"
    return output_directory


def set_wcs_tag(file_name):
    """Returns the WCS tag name associated to the analyzed image: the file name without the extension.

    Parameters
    ----------
    file_name: str
        File name of the image.

    Returns
    -------
    tag: str
        The tag.

    Examples
    --------
    >>> set_wcs_tag("image.fits")
    'image'

    """
    tag = os.path.splitext(os.path.basename(file_name))[0]
    return tag


def set_wcs_file_name(file_name, output_directory=""):
    """Returns the WCS file name associated to the analyzed image, placed in the output directory.
    The extension is .wcs.

    Parameters
    ----------
    file_name: str
        File name of the image.
    output_directory: str, optional
        If not set, the main output directory is the one the image,
        otherwise the specified directory is taken (default: "").

    Returns
    -------
    wcs_file_name: str
        The WCS file name.

    Examples
    --------
    >>> set_wcs_file_name("image.fits", output_directory="")
    'image_wcs/image.wcs'
    >>> set_wcs_file_name("image.png", output_directory="outputs")
    'outputs/image_wcs/image.wcs'

    """
    output_directory = set_wcs_output_directory(file_name, output_directory=output_directory)
    tag = set_wcs_tag(file_name)
    return os.path.join(output_directory, tag + '.wcs')


def set_sources_file_name(file_name, output_directory=""):
    """Returns the file name containing the deteted sources associated to the analyzed image,
    placed in the output directory. The suffix is _source.fits.

    Parameters
    ----------
    file_name: str
        File name of the image.
    output_directory: str, optional
        If not set, the main output directory is the one the image,
        otherwise the specified directory is taken (default: "").

    Returns
    -------
    sources_file_name: str
        The detected sources file name.

    Examples
    --------
    >>> set_sources_file_name("image.fits", output_directory="")
    'image_wcs/image.xyls'
    >>> set_sources_file_name("image.png", output_directory="outputs")
    'outputs/image_wcs/image.xyls'

    """
    output_directory = set_wcs_output_directory(file_name, output_directory=output_directory)
    tag = set_wcs_tag(file_name)
    return os.path.join(output_directory, f"{tag}.xyls")


def set_gaia_catalog_file_name(file_name, output_directory=""):
    """Returns the file name containing the Gaia catalog associated to the analyzed image,
    placed in the output directory. The suffix is _gaia.ecsv.

    Parameters
    ----------
    file_name: str
        File name of the image.
    output_directory: str, optional
        If not set, the main output directory is the one the image,
        otherwise the specified directory is taken (default: "").

    Returns
    -------
    sources_file_name: str
        The Gaia catalog file name.

    Examples
    --------
    >>> set_gaia_catalog_file_name("image.fits", output_directory="")
    'image_wcs/image_gaia.ecsv'
    >>> set_gaia_catalog_file_name("image.png", output_directory="outputs")
    'outputs/image_wcs/image_gaia.ecsv'

    """
    output_directory = set_wcs_output_directory(file_name, output_directory=output_directory)
    tag = set_wcs_tag(file_name)
    return os.path.join(output_directory, f"{tag}_gaia.ecsv")


def load_wcs_from_file(file_name):
    """Open the WCS FITS file and returns a WCS astropy object.

    Parameters
    ----------
    file_name: str
        File name of the WCS FITS file.

    Returns
    -------
    wcs: WCS
        WCS Astropy object.

    """
    # Load the FITS hdulist using astropy.io.fits
    with fits.open(file_name) as hdulist:
        # Parse the WCS keywords in the primary HDU
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            wcs = WCS.WCS(hdulist[0].header, fix=False)
    return wcs


def imgslice(slicespec):
    """
    Utility function: convert a FITS slice specification (1-based)
    into the corresponding numpy array slice spec (0-based as python does, xy swapped).

    Parameters
    ----------
    slicespec: str
        FITS slice specification with the format [xmin:xmax,ymin:ymax]

    Returns
    -------
    slice: slice
        Slice object to be injected in a np.array for instance.

    Examples
    --------
    >>> imgslice('[11:522,1:2002]')
    (slice(0, 2002, None), slice(10, 522, None))
    """

    parts = slicespec.replace('[', '').replace(']', '').split(',')
    xbegin, xend = [int(i) for i in parts[0].split(':')]
    ybegin, yend = [int(i) for i in parts[1].split(':')]
    xbegin -= 1
    ybegin -= 1
    return np.s_[ybegin:yend, xbegin:xend]


def compute_correlation_matrix(cov):
    """Compute correlation matrix from a covariance matrix.

    Parameters
    ----------
    cov: np.ndarray
        Covariance matrix.

    Returns
    -------
    rho: np.ndarray
        Correlation matrix.

    Examples
    --------
    >>> cov = np.array([[4, 1], [1, 16]])
    >>> compute_correlation_matrix(cov)
    array([[1.   , 0.125],
           [0.125, 1.   ]])


    """
    rho = np.zeros_like(cov, dtype="float")
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            rho[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    return rho


def plot_correlation_matrix_simple(ax, rho, axis_names=None, ipar=None):  # pragma: no cover
    if ipar is None:
        ipar = np.arange(rho.shape[0]).astype(int)
    im = plt.imshow(rho[ipar[:, None], ipar], interpolation="nearest", cmap='bwr', vmin=-1, vmax=1)
    ax.set_title("Correlation matrix")
    if axis_names is not None:
        names = [axis_names[ip] for ip in ipar]
        plt.xticks(np.arange(ipar.size), names, rotation='vertical', fontsize=15)
        plt.yticks(np.arange(ipar.size), names, fontsize=15)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=15)
    plt.gcf().tight_layout()


def resolution_operator(cov, Q, reg):
    N = cov.shape[0]
    return np.eye(N) - reg * cov @ Q


def flip_and_rotate_radec_vector_to_xy_vector(ra, dec, camera_angle=0, flip_ra_sign=1, flip_dec_sign=1):
    """Flip and rotate the vectors in pixels along (RA,DEC) directions to (x, y) image coordinates.
    The parity transformations are applied first, then rotation.

    Parameters
    ----------
    ra: array_like
        Vector coordinates along RA direction.
    dec: array_like
        Vector coordinates along DEC direction.
    camera_angle: float
        Angle of the camera between y axis and the North Celestial Pole counterclockwise, or equivalently between
        the x axis and the West direction counterclokwise. Units are degrees. (default: 0).
    flip_ra_sign: -1, 1, optional
        Flip RA axis is value is -1 (default: 1).
    flip_dec_sign: -1, 1, optional
        Flip DEC axis is value is -1 (default: 1).

    Returns
    -------
    x: array_like
       Vector coordinates along the x direction.
    y: array_like
       Vector coordinates along the y direction.

    Examples
    --------

    >>> from spectractor import parameters
    >>> parameters.OBS_CAMERA_ROTATION = 180
    >>> parameters.OBS_CAMERA_DEC_FLIP_SIGN = 1
    >>> parameters.OBS_CAMERA_RA_FLIP_SIGN = 1

    North vector

    >>> N_ra, N_dec = [0, 1]

    Compute North direction in (x, y) frame

    >>> flip_and_rotate_radec_vector_to_xy_vector(N_ra, N_dec, 0, flip_ra_sign=1, flip_dec_sign=1)
    (0.0, 1.0)
    >>> "%.1f, %.1f" % flip_and_rotate_radec_vector_to_xy_vector(N_ra, N_dec, 180, flip_ra_sign=1, flip_dec_sign=1)
    '-0.0, -1.0'
    >>> "%.1f, %.1f" % flip_and_rotate_radec_vector_to_xy_vector(N_ra, N_dec, 90, flip_ra_sign=1, flip_dec_sign=1)
    '-1.0, 0.0'
    >>> "%.1f, %.1f" % flip_and_rotate_radec_vector_to_xy_vector(N_ra, N_dec, 90, flip_ra_sign=1, flip_dec_sign=-1)
    '1.0, -0.0'
    >>> "%.1f, %.1f" % flip_and_rotate_radec_vector_to_xy_vector(N_ra, N_dec, 90, flip_ra_sign=-1, flip_dec_sign=-1)
    '1.0, -0.0'
    >>> "%.1f, %.1f" % flip_and_rotate_radec_vector_to_xy_vector(N_ra, N_dec, 0, flip_ra_sign=1, flip_dec_sign=-1)
    '0.0, -1.0'
    >>> "%.1f, %.1f" % flip_and_rotate_radec_vector_to_xy_vector(N_ra, N_dec, 0, flip_ra_sign=-1, flip_dec_sign=1)
    '0.0, 1.0'

    """
    flip = np.array([[flip_ra_sign, 0], [0, flip_dec_sign]], dtype=float)
    a = - camera_angle * np.pi / 180
    # minus sign as rotation matrix is apply on the right on the adr vector
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=float)
    transformation = flip @ rotation
    x, y = (np.asarray([ra, dec]).T @ transformation).T
    return x, y


def get_uvspec_binary():
    """Get the path to the libradtran uvspec binary if available.

    Returns
    -------
    uvspec_binary : `str`
        Path to the uvspec binary if available, else ``None``.
    """
    return shutil.which('uvspec')


def uvspec_available():
    """Check if the uvspec binary is available.

    Returns
    -------
    is_available : `bool`
        Is the binary available?
    """
    return get_uvspec_binary() is not None


if __name__ == "__main__":
    import doctest

    doctest.testmod()


def iraf_source_detection(data_wo_bkg, sigma=3.0, fwhm=3.0, threshold_std_factor=5, mask=None):
    """Function to detect point-like sources in a data array.

    This function use the photutils IRAFStarFinder module to search for sources in an image. This finder
    is better than DAOStarFinder for the astrometry of isolated sources but less good for photometry.

    Parameters
    ----------
    data_wo_bkg: array_like
        The image data array. It works better if the background was subtracted before.
    sigma: float
        Standard deviation value for sigma clipping function before finding sources (default: 3.0).
    fwhm: float
        Full width half maximum for the source detection algorithm (default: 3.0).
    threshold_std_factor: float
        Only sources with a flux above this value times the RMS of the images are kept (default: 5).
    mask: array_like, optional
        Boolean array to mask image pixels (default: None).

    Returns
    -------
    sources: Table
        Astropy table containing the source centroids and fluxes, ordered by decreasing magnitudes.

    Examples
    --------

    >>> N = 100
    >>> data = np.ones((N, N))
    >>> yy, xx = np.mgrid[:N, :N]
    >>> x_center, y_center = 20, 30
    >>> data += 10*np.exp(-((x_center-xx)**2+(y_center-yy)**2)/10)
    >>> sources = iraf_source_detection(data)
    >>> print(float(sources["xcentroid"]), float(sources["ycentroid"]))
    20.0 30.0

    .. doctest:
        :hide:

        >>> assert len(sources) == 1
        >>> assert sources["xcentroid"] == x_center
        >>> assert sources["ycentroid"] == y_center

    .. plot:

        from spectractor.tools import plot_image_simple
        from spectractor.astrometry import source_detection
        import numpy as np
        import matplotlib.pyplot as plt

        N = 100
        data = np.ones((N, N))
        yy, xx = np.mgrid[:N, :N]
        x_center, y_center = 20, 30
        data += 10*np.exp(-((x_center-xx)**2+(y_center-yy)**2)/10)
        sources = iraf_source_detection(data)
        fig = plt.figure(figsize=(6,5))
        plot_image_simple(plt.gca(), data, target_pixcoords=(sources["xcentroid"], sources["ycentroid"]))
        fig.tight_layout()
        plt.show()

    """
    mean, median, std = sigma_clipped_stats(data_wo_bkg, sigma=sigma)
    #fwhm = 5
    #threshold_std_factor = 3
    if mask is None:
        mask = np.zeros(data_wo_bkg.shape, dtype=bool)
    # daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_std_factor * std, exclude_border=True)
    # sources = daofind(data_wo_bkg - median, mask=mask)
    iraffind = IRAFStarFinder(fwhm=fwhm, threshold=threshold_std_factor * std, exclude_border=True)
    sources = iraffind(data_wo_bkg - median, mask=mask)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
    sources.sort('mag')
    if parameters.DEBUG:
        positions = np.array((sources['xcentroid'], sources['ycentroid']))
        plot_image_simple(plt.gca(), data_wo_bkg, scale="symlog", target_pixcoords=positions)
        if parameters.DISPLAY:
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()

    return sources


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super().default(obj)
        #return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

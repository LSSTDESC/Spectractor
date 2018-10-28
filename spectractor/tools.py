import os, sys
from scipy.optimize import curve_fit
import numpy as np
from astropy.modeling import models, fitting, Fittable2DModel, Parameter
from astropy.stats import sigma_clip
from astropy.io import fits

import warnings
from scipy.signal import fftconvolve, gaussian
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from skimage.feature import hessian_matrix

from spectractor import parameters
from math import floor


def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0)*(x - x0) / (2 * sigma * sigma))


def gauss_jacobian(x, A, x0, sigma):
    dA = np.exp(-(x - x0)*(x - x0) / (2 * sigma * sigma))
    dx0 = - A * (x - x0) / (sigma * sigma) * dA
    dsigma = A * (x-x0)*(x-x0) / (sigma ** 3) * dA
    return [dA, dx0, dsigma]


def line(x, a, b):
    return a * x + b


# noinspection PyTypeChecker
def fit_gauss(x, y, guess=[10, 1000, 1], bounds=(-np.inf, np.inf)):
    """Fit a Gaussian profile to data, using curve_fit. The mean guess value of the Gaussian
    must not be far from the truth values. Boundaries helps a lot also.

    Parameters
    ----------
    x: 1D-array
        The x data values.
    y: 1D-array
        The y data values.
    guess: list, [amplitude, mean, sigma]
        List of first guessed values for the Gaussian fit (default: [10, 1000, 1]).
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
    >>> y = gauss(x, 10, 600, 10)
    >>> print(y[0])
    10.0
    >>> popt, pcov = fit_gauss(x, y, guess=(3,630,3), bounds=((1,600,1),(100,800,100)))
    >>> print(popt)
    [  10.  600.   10.]
    """
    popt, pcov = curve_fit(gauss, x, y, p0=guess, bounds=bounds)
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


# noinspection PyTypeChecker
def multigauss_and_bgd(x, *params):
    """Multiple Gaussian profile plus a polynomial background to data, using curve_fit.
    The degree of the polynomial background is fixed by parameters.BGD_ORDER.
    The order of the parameters is a first block BGD_ORDER+1 parameters (from high to low monomial terms,
    same as np.polyval), and then block of 3 parameters for the Gaussian profiles like amplitude, mean and standard
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
    >>> p = [-1e-6, -1e-4, 1, 1, 20, 650, 3, 40, 750, 5]
    >>> y = multigauss_and_bgd(x, *p)
    >>> print(y[0])
    349.0
    """
    bgd_nparams = parameters.BGD_NPARAMS
    out = np.polyval(params[0:bgd_nparams], x)
    for k in range((len(params) - bgd_nparams) // 3):
        out += gauss(x, *params[bgd_nparams + 3 * k:bgd_nparams + 3 * k + 3])
    return out


# noinspection PyTypeChecker
def multigauss_and_bgd_jacobian(x, *params):
    """Jacobien of the multiple Gaussian profile plus a polynomial background to data, using curve_fit.
    The degree of the polynomial background is fixed by parameters.BGD_ORDER.
    The order of the parameters is a first block BGD_ORDER+1 parameters (from high to low monomial terms,
    same as np.polyval), and then block of 3 parameters for the Gaussian profiles like amplitude, mean and standard
    deviation.

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
    >>> x = np.arange(600.,800.,1)
    >>> p = [-1e-6, -1e-4, 1, 1, 20, 650, 3, 40, 750, 5]
    >>> y = multigauss_and_bgd_jacobian(x, *p)
    >>> print(y[0][0])
    2.16000000e+008
    """
    bgd_nparams = parameters.BGD_NPARAMS
    out = []
    for k in range(bgd_nparams):
        # out.append(params[k]*(parameters.BGD_ORDER-k)*x**(parameters.BGD_ORDER-(k+1)))
        out.append(x ** (parameters.BGD_ORDER - k))
    for k in range((len(params) - bgd_nparams) // 3):
        out += gauss_jacobian(x, *params[bgd_nparams + 3 * k:bgd_nparams + 3 * k + 3])
    return np.array(out).T


# noinspection PyTypeChecker
def fit_multigauss_and_bgd(x, y, guess=[0, 1, 10, 1000, 1, 0], bounds=(-np.inf, np.inf), sigma=None):
    """Fit a multiple Gaussian profile plus a polynomial background to data, using curve_fit.
    The mean guess value of the Gaussian must not be far from the truth values.
    Boundaries helps a lot also. The degree of the polynomial background is fixed by parameters.BGD_ORDER.
    The order of the parameters is a first block BGD_ORDER+1 parameters (from high to low monomial terms,
    same as np.polyval), and then block of 3 parameters for the Gaussian profiles like amplitude, mean and standard
    deviation.

    Parameters
    ----------
    x: array
        The x data values.
    y: array
        The y data values.
    guess: list, [BGD_ORDER+1 parameters, 3*number of Gaussian parameters]
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
    >>> p = [-1e-6, -1e-4, 1, 1, 20, 650, 3, 40, 750, 5]
    >>> y = multigauss_and_bgd(x, *p)
    >>> print(y[0])
    349.0
    >>> err = 0.1 * np.sqrt(y)
    >>> bounds = ((-np.inf,-np.inf,-np.inf,-np.inf,1,600,1,1,600,1),(np.inf,np.inf,np.inf,np.inf,100,800,100,100,800,100))
    >>> popt, pcov = fit_multigauss_and_bgd(x, y, guess=(0,1,-1,1,10,640,3,20,760,5), bounds=bounds, sigma=err)
    >>> assert np.all(np.isclose(p,popt))
    >>> fit = multigauss_and_bgd(x, *popt)

    .. plot::

        import matplotlib.pyplot as plt
        plt.errorbar(x,y,yerr=err,linestyle='None')
        plt.plot(x,fit,'r-')
        plt.show()
    """
    maxfev = 10000
    popt, pcov = curve_fit(multigauss_and_bgd, x, y, p0=guess, bounds=bounds, maxfev=maxfev, sigma=sigma,
                           absolute_sigma=True, method='trf', xtol=1e-4, ftol=1e-4, verbose=0,
                           jac=multigauss_and_bgd_jacobian, x_scale='jac')
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
    >>> p = [3,2,1,0]
    >>> y = np.polyval(p, x)
    >>> err = np.ones_like(y)
    >>> fit, cov, model = fit_poly1d(x,y,order=3)
    >>> assert np.all(np.isclose(p,fit,3))
    >>> fit, cov2, model2 = fit_poly1d(x,y,order=3,w=err)
    >>> assert np.all(np.isclose(p,fit,3))
    """
    cov = -1
    if len(x) > order:
        if w is None:
            fit, cov = np.polyfit(x, y, order, cov=True)
        else:
            fit, cov = np.polyfit(x, y, order, cov=True, w=w)
        model = lambda xx: np.polyval(fit, xx)
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

    Examples
    --------
    >>> x = np.arange(500., 1000., 1)
    >>> p = [3,2,1,0]
    >>> y = np.polyval(p, x)
    >>> y[::10] = 0.
    >>> model = fit_poly1d_outlier_removal(x,y,order=3,sigma=3)
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
        filtered_data, or_fitted_model = or_fit(gg_init, x, y)
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
        return or_fitted_model


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
    gg_init = models.Polynomial2D(order)
    gg_init.c0_0.min = np.min(z)
    gg_init.c0_0.max = 2 * np.max(z)
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        filtered_data, or_fitted_model = or_fit(gg_init, x, y, z)
        return or_fitted_model


def tied_circular_gauss2d(g1):
    std = g1.x_stddev
    return std


def fit_gauss2d_outlier_removal(x, y, z, sigma=3.0, niter=50, guess=None, bounds=None, circular=False):
    """Gauss2D parameters: amplitude, x_mean,y_mean,x_stddev, y_stddev,theta"""
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
        filtered_data, or_fitted_model = or_fit(gg_init, x, y, z)
        if parameters.VERBOSE:
            print(or_fitted_model)
        return or_fitted_model


def fit_moffat2d_outlier_removal(x, y, z, sigma=3.0, niter=50, guess=None, bounds=None):
    """Moffat2D parameters: amplitude, x_mean,y_mean,gamma,alpha"""
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
        filtered_data, or_fitted_model = or_fit(gg_init, x, y, z)
        if parameters.VERBOSE:
            print(or_fitted_model)
        return or_fitted_model


class Star2D(Fittable2DModel):
    inputs = ('x','y',)
    outputs = ('z',)

    amplitude = Parameter('amplitude', default=1)
    x_mean = Parameter('x_mean', default=0)
    y_mean = Parameter('y_mean', default=0)
    stddev = Parameter('stddev', default=1)
    eta = Parameter('eta', default=0.5)
    alpha = Parameter('alpha', default=3)
    gamma = Parameter('gamma', default=3)
    saturation = Parameter('saturation', default=1)

    #def __init__(self, amplitude=amplitude.default, x_mean=x_mean.default, y_mean=y_mean.default, stddev=stddev.default,
    #             eta=eta.default, alpha=alpha.default, gamma=gamma.default,saturation=saturation.default, **kwargs):
    #    super(Fittable2DModel, self).__init__(**kwargs)

    @property
    def fwhm(self):
        return self.stddev / 2.335

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, stddev, eta, alpha, gamma, saturation):
        rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
        rr_gg = rr/(gamma*gamma)
        a = amplitude * ( np.exp(-(rr / (2. * stddev * stddev))) + eta * (1 + rr_gg) ** (-alpha))
        # print(amplitude, x_mean, y_mean, stddev, eta, alpha, gamma, saturation)
        if isinstance(x, float) and isinstance(y, float):
            if a > saturation:
                return saturation
            else:
                return a
        else:
            a[np.where(a >= saturation)] = saturation
            return a

    @staticmethod
    def fit_deriv(x, y, amplitude, x_mean, y_mean, stddev, eta, alpha, gamma, saturation):
        rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
        rr_gg = rr/(gamma*gamma)
        d_amplitude_gauss = np.exp(-(rr / (2. * stddev * stddev)))
        d_amplitude_moffat =  eta * (1 + rr_gg) ** (-alpha)
        d_amplitude = d_amplitude_gauss + d_amplitude_moffat
        d_x_mean = - amplitude * (x - x_mean) / (stddev * stddev) * d_amplitude_gauss \
                   - amplitude * alpha * d_amplitude_moffat * (-2 * x + 2 * x_mean) / (gamma ** 2 * (1 + rr_gg))
        d_y_mean = - amplitude * (y - y_mean) / (stddev * stddev) * d_amplitude_gauss \
                   - amplitude * alpha * d_amplitude_moffat * (-2 * y + 2 * y_mean) / (gamma ** 2 * (1 + rr_gg))
        d_stddev = amplitude * rr / (stddev ** 3) * d_amplitude_gauss
        d_eta = amplitude * d_amplitude_moffat / eta
        d_alpha = - amplitude * d_amplitude_moffat * np.log(1 + rr_gg)
        d_gamma = 2 * amplitude * alpha * d_amplitude_moffat * (rr_gg / (gamma * (1 + rr_gg)))
        d_saturation = saturation * np.ones_like(x)
        return [d_amplitude, d_x_mean, d_y_mean, d_stddev, d_eta, d_alpha, d_gamma, d_saturation]


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


def fit_star2d_outlier_removal(x, y, z, sigma=5.0, niter=10, guess=None, bounds=None):
    """Star2D parameters: amplitude, x_mean,y_mean,stddev,saturation"""
    gg_init = Star2D()
    if guess is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).value = guess[ip]
    if bounds is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).min = bounds[0][ip]
            getattr(gg_init, p).max = bounds[1][ip]
    # gg_init.saturation.fixed = True
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        filtered_data, or_fitted_model = or_fit(gg_init, x, y, z)
        if parameters.VERBOSE:
            print(or_fitted_model)
        if parameters.DEBUG:
            print(fit.fit_info)
        return or_fitted_model


def fit_star2d(x, y, z, guess=None, bounds=None, sub_errors=None):
    """Star2D parameters: amplitude, x_mean,y_mean,stddev,saturation"""
    gg_init = Star2D()
    if guess is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).value = guess[ip]
    if bounds is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).min = bounds[0][ip]
            getattr(gg_init, p).max = bounds[1][ip]
    # gg_init.saturation.fixed = True
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        # fit = fitting.LevMarLSQFitter()
        fit = LevMarLSQFitterWithNan()
        fitted_model = fit(gg_init, x, y, z, acc=1e-20, epsilon=1e-20, weights=1./sub_errors)
        if parameters.VERBOSE:
            print(fitted_model)
        if parameters.DEBUG:
            print(fit.fit_info)
        return fitted_model


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
    >>> array = np.ones(20)
    >>> output = fftconvolve_gaussian(array, 3)
    >>> print(output[:3])
    [ 0.5         0.63125312  0.74870357]
    """
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
        sys.exit('fftconvolve_gaussian: array dimension must be 1 or 2.')
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
    >>> text = formatting_numbers(3., 0.789, 0.500, std=0.45, label='test')
    >>> print(text)
    ('test', '3.0', '0.8', '0.5', '0.5')
    >>> text = formatting_numbers(3., 0.07, 0.008, std=0.03, label='test')
    >>> print(text)
    ('test', '3.000', '0.07', '0.008', '0.03')
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
    u = np.cos(theta) * (x - x0) + np.sin(theta) * (y - y0)
    v = -np.sin(theta) * (x - x0) + np.cos(theta) * (y - y0)
    return u, v


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
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


def load_fits(file_name, hdu_index=0):
    hdu_list = fits.open(file_name)
    header = hdu_list[hdu_index].header
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


def save_fits(file_name, header, data, overwrite=False):
    hdu = fits.PrimaryHDU()
    hdu.header = header
    hdu.data = data
    output_directory = '/'.join(file_name.split('/')[:-1])
    ensure_dir(output_directory)
    hdu.writeto(file_name, overwrite=overwrite)


if __name__ == "__main__":
    import doctest
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    doctest.testmod()

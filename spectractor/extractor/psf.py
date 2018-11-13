import warnings
import sys
import numpy as np
from scipy.optimize import basinhopping, minimize, newton
from scipy.integrate import quad
from astropy.modeling import fitting, Fittable1DModel, Fittable2DModel, Parameter
from astropy.stats import sigma_clip
from spectractor.tools import LevMarLSQFitterWithNan
from spectractor import parameters
from spectractor.config import set_logger


class PSF1D(Fittable1DModel):
    inputs = ('x',)
    outputs = ('y',)

    amplitude_gauss = Parameter('amplitude_gauss', default=1)
    x_mean = Parameter('x_mean', default=0)
    stddev = Parameter('stddev', default=1)
    amplitude_moffat = Parameter('amplitude_moffat', default=0.5)
    alpha = Parameter('alpha', default=3)
    gamma = Parameter('gamma', default=3)
    saturation = Parameter('saturation', default=1)

    def fwhm(self):
        """
        Compute the full width half maximum of the PSF1D model.

        Returns
        -------
        FWHM: float
            The full width half maximum of the PSF1D model.

        Examples
        --------
        >>> p = [1,0,2,2,2,2,10]
        >>> PSF = PSF1D(*p)
        >>> fwhm = PSF.fwhm()
        >>> assert np.isclose(fwhm, 3.1409218870190476)

        """
        eq = lambda x, *args: self.evaluate(x, *args) - 0.5*(self.amplitude_gauss+self.amplitude_moffat)
        fwhm = 2*newton(eq, self.x_mean+self.gamma,
                        args=(self.amplitude_gauss, self.x_mean, self.stddev,
                           self.amplitude_moffat, self.alpha, self.gamma, self.saturation))
        return fwhm

    @staticmethod
    def evaluate(x, amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation):
        rr = (x - x_mean) ** 2
        rr_gg = rr / (gamma * gamma)
        a = amplitude_gauss * np.exp(-(rr / (2. * stddev * stddev))) + amplitude_moffat * (1 + rr_gg) ** (-alpha)
        if isinstance(x, float):
            if a > saturation:
                return saturation
            else:
                return a
        else:
            a[np.where(a >= saturation)] = saturation
            return a

    @staticmethod
    def fit_deriv(x, amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation):
        rr = (x - x_mean) * (x - x_mean)
        rr_gg = rr / (gamma * gamma)
        d_amplitude_gauss = np.exp(-(rr / (2. * stddev * stddev)))
        d_amplitude_moffat = (1 + rr_gg) ** (-alpha)
        d_x_mean = amplitude_gauss * (x - x_mean) / (stddev * stddev) * d_amplitude_gauss \
                   - amplitude_moffat * alpha * d_amplitude_moffat * (-2 * x + 2 * x_mean) / (gamma * gamma * (1 + rr_gg))
        d_stddev = amplitude_gauss * rr / (stddev ** 3) * d_amplitude_gauss
        d_alpha = - amplitude_moffat * d_amplitude_moffat * np.log(1 + rr_gg)
        d_gamma = 2 * amplitude_moffat * alpha * d_amplitude_moffat * (rr_gg / (gamma * (1 + rr_gg)))
        d_saturation = saturation * np.zeros_like(x)
        return np.array([d_amplitude_gauss, d_x_mean, d_stddev, d_amplitude_moffat, d_alpha, d_gamma, d_saturation])

    def integral(self):
        """
        Compute the wide integral of the PSF1D model. Bounds are -5*FWHM and +5*FWHM.

        Returns
        -------
        result: float
            The integral of the PSF1D model.

        Examples
        --------
        >>> p = [1,0,2,2,2,2,10]
        >>> PSF = PSF1D(*p)
        >>> i = PSF.integral()
        >>> assert np.isclose(i, 11.291039428010016)

        """
        fwhm = self.fwhm()
        i = quad(self.evaluate, self.x_mean-5*fwhm, self.x_mean+5*fwhm,
                 args=(self.amplitude_gauss, self.x_mean, self.stddev,
                       self.amplitude_moffat, self.alpha, self.gamma, self.saturation))
        return i[0]


class PSF2D(Fittable2DModel):
    inputs = ('x', 'y',)
    outputs = ('z',)
    # TODO: decouple the two amplitudes to be able to fit defocused object
    amplitude = Parameter('amplitude', default=1)
    x_mean = Parameter('x_mean', default=0)
    y_mean = Parameter('y_mean', default=0)
    stddev = Parameter('stddev', default=1)
    eta = Parameter('eta', default=0.5)
    alpha = Parameter('alpha', default=3)
    gamma = Parameter('gamma', default=3)
    saturation = Parameter('saturation', default=1)

    @property
    def fwhm(self):
        return self.stddev / 2.335

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, stddev, eta, alpha, gamma, saturation):
        rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
        rr_gg = rr / (gamma * gamma)
        a = amplitude * (np.exp(-(rr / (2. * stddev * stddev))) + eta * (1 + rr_gg) ** (-alpha))
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
        rr_gg = rr / (gamma * gamma)
        d_amplitude_gauss = np.exp(-(rr / (2. * stddev * stddev)))
        d_amplitude_moffat = eta * (1 + rr_gg) ** (-alpha)
        d_amplitude = d_amplitude_gauss + d_amplitude_moffat
        d_x_mean = amplitude * (x - x_mean) / (stddev * stddev) * d_amplitude_gauss \
                   - amplitude * alpha * d_amplitude_moffat * (-2 * x + 2 * x_mean) / (gamma ** 2 * (1 + rr_gg))
        d_y_mean = amplitude * (y - y_mean) / (stddev * stddev) * d_amplitude_gauss \
                   - amplitude * alpha * d_amplitude_moffat * (-2 * y + 2 * y_mean) / (gamma ** 2 * (1 + rr_gg))
        d_stddev = amplitude * rr / (stddev ** 3) * d_amplitude_gauss
        d_eta = amplitude * d_amplitude_moffat / eta
        d_alpha = - amplitude * d_amplitude_moffat * np.log(1 + rr_gg)
        d_gamma = 2 * amplitude * alpha * d_amplitude_moffat * (rr_gg / (gamma * (1 + rr_gg)))
        d_saturation = saturation * np.zeros_like(x)
        return [d_amplitude, d_x_mean, d_y_mean, d_stddev, d_eta, d_alpha, d_gamma, d_saturation]


def PSF2D_chisq(params, model, xx, yy, zz, zz_err=None):
    mod = model.evaluate(xx, yy, *params)
    if zz_err is None:
        return np.nansum((mod - zz) ** 2)
    else:
        return np.nansum(((mod - zz) / zz_err) ** 2)


def PSF2D_chisq_jac(params, model, xx, yy, zz, zz_err=None):
    diff = model.evaluate(xx, yy, *params) - zz
    jac = model.fit_deriv(xx, yy, *params)
    if zz_err is None:
        return np.array([np.nansum(2 * jac[p] * diff) for p in range(len(params))])
    else:
        zz_err2 = zz_err * zz_err
        return np.array([np.nansum(2 * jac[p] * diff / zz_err2) for p in range(len(params))])


def fit_PSF2D_outlier_removal(x, y, z, sigma=5.0, niter=10, guess=None, bounds=None):
    """Star2D parameters: amplitude, x_mean,y_mean,stddev,saturation"""
    gg_init = PSF2D()
    if guess is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).value = guess[ip]
    if bounds is not None:
        for ip, p in enumerate(gg_init.param_names):
            getattr(gg_init, p).min = bounds[0][ip]
            getattr(gg_init, p).max = bounds[1][ip]
    gg_init.saturation.fixed = True
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = LevMarLSQFitterWithNan()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        filtered_data, or_fitted_model = or_fit(gg_init, x, y, z)
        if parameters.VERBOSE:
            print(or_fitted_model)
        if parameters.DEBUG:
            print(fit.fit_info)
        return or_fitted_model


def fit_PSF2D(x, y, z, guess=None, bounds=None, sub_errors=None, method='minimize'):
    """Fit a PSF 2D model with parameters :
        amplitude, x_mean, y_mean, stddev, eta, alpha, gamma, saturation
    using basin hopping global minimization method.

    Parameters
    ----------
    x: np.array
        2D array of the x coordinates from meshgrid.
    y: np.array
        2D array of the y coordinates from meshgrid.
    z: np.array
        the 2D array image.
    guess: array_like, optional
        List containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    sub_errors: np.array
        the 2D array uncertainties.

    Returns
    -------
    fitted_model: PSF2D
        the PSF2D fitted model.

    Examples
    --------

    Create the model:
    >>> import numpy as np
    >>> X, Y = np.mgrid[:50,:50]
    >>> PSF = PSF2D()
    >>> p = (50, 25, 25, 5, 1, 1, 5, 60)
    >>> Z = PSF.evaluate(X, Y, *p)
    >>> Z_err = np.sqrt(Z)/10.

    Prepare the fit:
    >>> guess = (50, 20, 20, 3, 0.5, 1.2, 2.2, 60)
    >>> bounds = ((1, 200), (10, 40), (10, 40), (1, 10), (0.5, 2), (0.5, 5), (0.5, 10), (0, 400))

    Fit with error bars:
    >>> model = fit_PSF2D(X, Y, Z, guess=guess, bounds=bounds, sub_errors=Z_err)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit without error bars:
    >>> model = fit_PSF2D(X, Y, Z, guess=guess, bounds=bounds, sub_errors=None)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit with error bars and basin hopping method:
    >>> model = fit_PSF2D(X, Y, Z, guess=guess, bounds=bounds, sub_errors=Z_err, method='basinhopping')
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    """

    model = PSF2D()
    my_logger = set_logger(__name__)
    if method == 'minimize':
        res = minimize(PSF2D_chisq, guess, method="L-BFGS-B", bounds=bounds,
                       args=(model, x, y, z, sub_errors), jac=PSF2D_chisq_jac)
    elif method == 'basinhopping':
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, jac=PSF2D_chisq_jac,
                            args=(model, x, y, z, sub_errors))
        res = basinhopping(PSF2D_chisq, guess, niter=20, minimizer_kwargs=minimizer_kwargs)
    else:
        my_logger.error(f'\n\tUnknown method {method}.')
        sys.exit()
    my_logger.debug(f'\n{res}')
    PSF = PSF2D(*res.x)
    my_logger.info(f'\n\tPSF best fitting parameters:\n{PSF}')
    return PSF


def PSF1D_chisq(params, model, xx, yy, yy_err=None):
    diff = model.evaluate(xx, *params) - yy
    if yy_err is None:
        return np.nansum(diff * diff)
    else:
        return np.nansum((diff / yy_err) ** 2)


def PSF1D_chisq_jac(params, model, xx, yy, yy_err=None):
    diff = model.evaluate(xx, *params) - yy
    jac = model.fit_deriv(xx, *params)
    if yy_err is None:
        return np.array([np.nansum(2 * jac[p] * diff) for p in range(len(params))])
    else:
        yy_err2 = yy_err * yy_err
        return np.array([np.nansum(2 * jac[p] * diff / yy_err2) for p in range(len(params))])


def fit_PSF1D(x, y, guess=None, bounds=None, sub_errors=None, method='minimize'):
    """Fit a PSF 1D model with parameters :
        amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation
    using basin hopping global minimization method.

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates.
    y: np.array
        the 1D array profile.
    guess: array_like, optional
        list containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    sub_errors: np.array
        the 1D array uncertainties.
    method: str, optional
        method to use for the minimisation: choose between minimize and basinhopping.

    Returns
    -------
    fitted_model: PSF1D
        the PSF1D fitted model.

    Examples
    --------

    Create the model:
    >>> import numpy as np
    >>> X = np.arange(0, 50)
    >>> PSF = PSF1D()
    >>> p = (-10, 25, 5, 50, 1, 5, 60)
    >>> Y = PSF.evaluate(X, *p)
    >>> Y_err = np.sqrt(Y)/10.

    Prepare the fit:
    >>> guess = (-5, 20, 4, 60, 1.2, 3.2, 60)
    >>> bounds = ((-100, 100), (10, 40), (1, 10), (0, 200), (0.5, 5), (0.5, 10), (0, 400))

    Fit with error bars:
    >>> model = fit_PSF1D(X, Y, guess=guess, bounds=bounds, sub_errors=Y_err)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit without error bars:
    >>> model = fit_PSF1D(X, Y, guess=guess, bounds=bounds, sub_errors=None)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit with error bars and basin hopping method:
    >>> model = fit_PSF1D(X, Y, guess=guess, bounds=bounds, sub_errors=Y_err, method='basinhopping')
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    """
    my_logger = set_logger(__name__)
    model = PSF1D()
    if method == 'minimize':
        res = minimize(PSF1D_chisq, guess, method="L-BFGS-B", bounds=bounds,
                       args=(model, x, y, sub_errors), jac=PSF1D_chisq_jac)
    elif method == 'basinhopping':
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds,
                                args=(model, x, y, sub_errors), jac=PSF1D_chisq_jac)
        res = basinhopping(PSF1D_chisq, guess, niter=20, minimizer_kwargs=minimizer_kwargs)
    else:
        my_logger.error(f'\n\tUnknown method {method}.')
        sys.exit()
    my_logger.debug(f'\n{res}')
    PSF = PSF1D(*res.x)
    my_logger.info(f'\n\tPSF best fitting parameters:\n{PSF}')
    return PSF


def fit_PSF1D_outlier_removal(x, y, sub_errors=None, sigma=3.0, niter=3, guess=None, bounds=None, method='minimize'):
    """Star2D parameters: amplitude, x_mean,y_mean,stddev,saturation"""

    my_logger = set_logger(__name__)
    indices = np.mgrid[:x.shape[0]]
    outliers = np.array([])
    model = PSF1D()

    for step in range(niter):
        # first fit
        if sub_errors is None:
            err = None
        else:
            err = sub_errors[indices]
        if method == 'minimize':
            res = minimize(PSF1D_chisq, guess, method="L-BFGS-B", bounds=bounds, jac=PSF1D_chisq_jac,
                           args=(model, x[indices], y[indices], err))
        elif method == 'basinhopping':
            # TODO: add jacobian
            minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds,
                                    args=(model, x[indices], y[indices], err))
            res = basinhopping(PSF1D_chisq, guess, niter=20, minimizer_kwargs=minimizer_kwargs)
        else:
            my_logger.error(f'\n\tUnknown method {method}.')
            sys.exit()
        if parameters.DEBUG:
            my_logger.debug(f'\n\tniter={step}\n{res}')
        # update the model and the guess
        for ip, p in enumerate(model.param_names):
            setattr(model, p, res.x[ip])
        guess = res.x
        # remove outliers
        if sub_errors is not None:
            outliers = np.where(np.abs(model(x) - y) / sub_errors > sigma)[0]
        else:
            std = np.std(model(x) - y)
            outliers = np.where(np.abs(model(x) - y) / std > sigma)[0]
        if len(outliers) > 0:
            indices = [i for i in range(x.shape[0]) if i not in outliers]
    my_logger.info(f'\n\tPSF best fitting parameters:\n{model}')
    return model, outliers

import warnings
import numpy as np
from astropy.modeling import fitting, Fittable2DModel, Parameter
from astropy.stats import sigma_clip
from spectractor.tools import LevMarLSQFitterWithNan
from spectractor import parameters


class PSF2D(Fittable2DModel):
    inputs = ('x', 'y', )
    outputs = ('z', )

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
        rr_gg = rr/(gamma*gamma)
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
        rr_gg = rr/(gamma*gamma)
        d_amplitude_gauss = np.exp(-(rr / (2. * stddev * stddev)))
        d_amplitude_moffat = eta * (1 + rr_gg) ** (-alpha)
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


def fit_PSF2D(x, y, z, guess=None, bounds=None, sub_errors=None):
    """Star2D parameters: amplitude, x_mean,y_mean,stddev,saturation"""
    gg_init = PSF2D()
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

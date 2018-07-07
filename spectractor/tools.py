import os, sys
import scipy
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

from . import parameters
from math import floor


def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_and_line(x, a, b, A, x0, sigma):
    return gauss(x, A, x0, sigma) + line(x, a, b)


def line(x, a, b):
    return a * x + b


def parabola(x, a, b, c):
    return a * x * x + b * x + c


def fit_gauss(x, y, guess=[10, 1000, 1], bounds=(-np.inf, np.inf)):
    popt, pcov = curve_fit(gauss, x, y, p0=guess, bounds=bounds)
    return popt, pcov


def multigauss_and_line(x, *params):
    out = line(x, params[0], params[1])
    for k in range((len(params) - 2) // 3):
        out += gauss(x, *params[2 + 3 * k:2 + 3 * k + 3])
    return out


def fit_multigauss_and_line(x, y, guess=[10, 1000, 1, 0, 0, 1], bounds=(-np.inf, np.inf)):
    maxfev = 100000
    popt, pcov = curve_fit(multigauss_and_line, x, y, p0=guess, bounds=bounds, maxfev=maxfev)
    return popt, pcov


def multigauss_and_bgd(x, *params):
    bgd_nparams = parameters.BGD_NPARAMS
    out = np.polyval(params[0:bgd_nparams], x)
    for k in range((len(params) - bgd_nparams) // 3):
        out += gauss(x, *params[bgd_nparams + 3 * k:bgd_nparams + 3 * k + 3])
    return out


def fit_multigauss_and_bgd(x, y, guess=[10, 1000, 1, 0, 0, 1], bounds=(-np.inf, np.inf), sigma=None):
    maxfev = 1000000
    popt, pcov = curve_fit(multigauss_and_bgd, x, y, p0=guess, bounds=bounds, maxfev=maxfev, sigma=sigma)
    return popt, pcov


def fit_line(x, y, guess=[1, 1], bounds=(-np.inf, np.inf), sigma=None):
    popt, pcov = curve_fit(line, x, y, p0=guess, bounds=bounds, sigma=sigma)
    return popt, pcov


def fit_bgd(x, y, guess=[1] * parameters.BGD_NPARAMS, bounds=(-np.inf, np.inf), sigma=None):
    bgd = lambda x, *p: np.polyval(p, x)
    popt, pcov = curve_fit(bgd, x, y, p0=guess, bounds=bounds, sigma=sigma)
    return popt, pcov


def fit_poly(x, y, degree, w=None):
    cov = -1
    if len(x) > degree:
        if w is None:
            fit, cov = np.polyfit(x, y, degree, cov=True)
        else:
            fit, cov = np.polyfit(x, y, degree, cov=True, w=w)
        model = lambda x: np.polyval(fit, x)
    else:
        fit = [0] * (degree + 1)
        model = y
    return fit, cov, model


def fit_poly2d(x, y, z, degree):
    # Fit the data using astropy.modeling
    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)
        return p


def fit_poly1d_outlier_removal(x, y, order=2, sigma=3.0, niter=3):
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
        plt.show()
        '''
        return or_fitted_model


def fit_poly2d_outlier_removal(x, y, z, order=2, sigma=3.0, niter=30):
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
        if parameters.VERBOSE: print(or_fitted_model)
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
        if parameters.VERBOSE: print(or_fitted_model)
        return or_fitted_model


class Star2D(Fittable2DModel):
    amplitude = Parameter('amplitude', default=1)
    x_mean = Parameter('x_mean', default=0)
    y_mean = Parameter('y_mean', default=0)
    stddev = Parameter('stddev', default=1)
    saturation = Parameter('saturation', default=1)

    def __init__(self, amplitude=amplitude.default, x_mean=x_mean.default, y_mean=y_mean.default, stddev=stddev.default,
                 saturation=saturation.default, **kwargs):
        super(Fittable2DModel, self).__init__(**kwargs)

    @property
    def fwhm(self):
        return self.stddev / 2.335

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, stddev, saturation):
        a = amplitude * np.exp(
            -(1 / (2. * stddev ** 2)) * (x - x_mean) ** 2 - (1 / (2. * stddev ** 2)) * (y - y_mean) ** 2)
        if isinstance(x, float) and isinstance(y, float):
            if a > saturation:
                return saturation
            else:
                return a
        else:
            a[np.where(a >= saturation)] = saturation
            return a

    @staticmethod
    def fit_deriv(x, y, amplitude, x_mean, y_mean, stddev, saturation):
        d_amplitude = np.exp(
            -((1 / (2. * stddev ** 2)) * (x - x_mean) ** 2 + (1 / (2. * stddev ** 2)) * (y - y_mean) ** 2))
        d_x_mean = - amplitude * (x - x_mean) / (stddev ** 2) * np.exp(
            -(1 / (2. * stddev ** 2)) * (x - x_mean) ** 2 - (1 / (2. * stddev ** 2)) * (y - y_mean) ** 2)
        d_y_mean = - amplitude * (y - y_mean) / (stddev ** 2) * np.exp(
            -(1 / (2. * stddev ** 2)) * (x - x_mean) ** 2 - (1 / (2. * stddev ** 2)) * (y - y_mean) ** 2)
        d_stddev = amplitude * ((x - x_mean) ** 2 + (y - y_mean) ** 2) / (stddev ** 3) * np.exp(
            -(1 / (2. * stddev ** 2)) * (x - x_mean) ** 2 - (1 / (2. * stddev ** 2)) * (y - y_mean) ** 2)
        d_saturation = np.zeros_like(x)
        return [d_amplitude, d_x_mean, d_y_mean, d_stddev, d_saturation]


def fit_star2d_outlier_removal(x, y, z, sigma=3.0, niter=50, guess=None, bounds=None):
    """Star2D parameters: amplitude, x_mean,y_mean,stddev,saturation"""
    gg_init = Star2D()
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
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        filtered_data, or_fitted_model = or_fit(gg_init, x, y, z)
        if parameters.VERBOSE: print(or_fitted_model)
        return or_fitted_model


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)

        
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
    lambda_plus, lambda_minus, theta = hessian_and_theta(np.copy(data), margin_cut=1)
    # thresholds
    lambda_threshold = np.median(lambda_minus) - 2 * np.std(lambda_minus)
    mask = np.where(lambda_minus < lambda_threshold)
    data[mask] = np.nan
    return data


def fftconvolve_gaussian(array, reso):
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


def restrict_lambdas(lambdas):
    lambdas_indices = \
        np.where(np.logical_and(lambdas > parameters.LAMBDA_MIN, lambdas < parameters.LAMBDA_MAX))[0]
    lambdas = lambdas[lambdas_indices]
    return lambdas


def formatting_numbers(value, errorhigh, errorlow, std=None, label=None):
    str_value = ""
    str_errorhigh = ""
    str_errorlow = ""
    str_std = ""
    out = []
    if label is not None: out.append(label)
    power10 = min(int(floor(np.log10(np.abs(errorhigh)))), int(floor(np.log10(np.abs(errorlow)))))
    if np.isclose(0.0, float("%.*f" % (abs(power10), value))):
        str_value = "%.*f" % (abs(power10), 0)
        str_errorhigh = "%.*f" % (abs(power10), errorhigh)
        str_errorlow = "%.*f" % (abs(power10), errorlow)
        if std is not None:
            str_std = "%.*f" % (abs(power10), std)
    elif power10 > 0:
        str_value = "%d" % value
        str_errorhigh = "%d" % errorhigh
        str_errorlow = "%d" % errorlow
        if std is not None:
            str_std = "%d" % std
    else:
        if int(floor(np.log10(np.abs(errorhigh)))) == int(floor(np.log10(np.abs(errorlow)))):
            str_value = "%.*f" % (abs(power10), value)
            str_errorhigh = "%.1g" % errorhigh
            str_errorlow = "%.1g" % errorlow
            if std is not None:
                str_std = "%.1g" % std
        elif int(floor(np.log10(np.abs(errorhigh)))) > int(floor(np.log10(np.abs(errorlow)))):
            str_value = "%.*f" % (abs(power10), value)
            str_errorhigh = "%.2g" % errorhigh
            str_errorlow = "%.1g" % errorlow
            if std is not None:
                str_std = "%.2g" % std
        else:
            str_value = "%.*f" % (abs(power10), value)
            str_errorhigh = "%.1g" % errorhigh
            str_errorlow = "%.2g" % errorlow
            if std is not None:
                str_std = "%.2g" % std
    out += [str_value, str_errorhigh]
    if not np.isclose(errorhigh, errorlow): out += [str_errorlow]
    if std is not None: out += [str_std]
    out = tuple(out)
    return out


def pixel_rotation(x, y, theta, x0=0, y0=0):
    u = np.cos(theta) * (x - x0) + np.sin(theta) * (y - y0)
    v = -np.sin(theta) * (x - x0) + np.cos(theta) * (y - y0)
    x = u + x0
    y = v + y0
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
    obj.filter = header['FILTER1']
    obj.disperser = header['FILTER2']


def save_fits(file_name, header, data, overwrite=False):
    hdu = fits.PrimaryHDU()
    hdu.header = header
    hdu.data = data
    output_directory = file_name.split('/')[0]
    ensure_dir(output_directory)
    hdu.writeto(file_name, overwrite=overwrite)

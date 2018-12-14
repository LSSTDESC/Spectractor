import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import basinhopping, minimize
from scipy.interpolate import interp1d
from scipy.integrate import quad

from astropy.modeling import fitting, Fittable1DModel, Fittable2DModel, Parameter
from astropy.modeling.models import Moffat1D
from astropy.stats import sigma_clip

from spectractor.tools import LevMarLSQFitterWithNan, dichotomie, fit_poly1d_outlier_removal, \
    fit_poly1d, fit_moffat1d_outlier_removal
from spectractor import parameters
from spectractor.config import set_logger


class PSF1D(Fittable1DModel):
    inputs = ('x',)
    outputs = ('y',)

    amplitude_moffat = Parameter('amplitude_moffat', default=0.5)
    x_mean = Parameter('x_mean', default=0)
    gamma = Parameter('gamma', default=3)
    alpha = Parameter('alpha', default=3)
    eta_gauss = Parameter('eta_gauss', default=1)
    stddev = Parameter('stddev', default=1)
    saturation = Parameter('saturation', default=1)

    def fwhm(self, x_array=None):
        """
        Compute the full width half maximum of the PSF1D model with a dichotomie method.

        Parameters
        ----------
        x_array: array_like, optional
            An abscisse array is one wants to find FWHM on the interpolated PSF1D model
            (to smooth the spikes from spurious parameter sets).

        Returns
        -------
        FWHM: float
            The full width half maximum of the PSF1D model.

        Examples
        --------
        >>> x = np.arange(0, 60, 1)
        >>> p = [2,30,4,2,-0.4,1,10]
        >>> PSF = PSF1D(*p)
        >>> a, b =  p[1], p[1]+3*max(p[-2], p[2])
        >>> fwhm = PSF.fwhm(x_array=None)
        >>> assert np.isclose(fwhm, 7.25390625)
        >>> fwhm = PSF.fwhm(x_array=x)
        >>> assert np.isclose(fwhm, 7.083984375)
        >>> print(fwhm)
        7.083984375
        >>> import matplotlib.pyplot as plt
        >>> x = np.arange(0, 60, 0.01)
        >>> plt.plot(x, PSF.evaluate(x, *p))
        >>> plt.show()
        """
        params = [getattr(self, p).value for p in self.param_names]
        interp = None
        if x_array is not None:
            interp = self.interpolation(x_array)
            values = self.evaluate(x_array, *params)
            maximum = np.max(values)
            imax = np.argmax(values)
            a = imax + np.argmin(np.abs(values[imax:] - 0.95 * maximum))
            b = imax + np.argmin(np.abs(values[imax:] - 0.05 * maximum))

            def eq(x):
                return interp(x) - 0.5 * maximum
        else:
            maximum = self.amplitude_moffat.value * (1 + self.eta_gauss.value)
            a = self.x_mean.value
            b = self.x_mean.value + 3 * max(self.gamma.value, self.stddev.value)

            def eq(x):
                return self.evaluate(x, *params) - 0.5 * maximum
        res = dichotomie(eq, a, b, 1e-2)
        # res = newton()
        return abs(2 * (res - self.x_mean.value))

    @staticmethod
    def evaluate(x, amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = (x - x_mean) * (x - x_mean)
        rr_gg = rr / (gamma * gamma)
        a = amplitude_moffat * ((1 + rr_gg) ** (-alpha) + eta_gauss * np.exp(-(rr / (2. * stddev * stddev))))
        if isinstance(x, float) or isinstance(x, int):
            if a > saturation:
                return saturation
            elif a < 0.:
                return 0.
            else:
                return a
        else:
            a[np.where(a >= saturation)] = saturation
            a[np.where(a < 0.)] = 0.
            return a

    @staticmethod
    def fit_deriv(x, amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = (x - x_mean) * (x - x_mean)
        rr_gg = rr / (gamma * gamma)
        gauss_norm = np.exp(-(rr / (2. * stddev * stddev)))
        d_eta_gauss = amplitude_moffat * gauss_norm
        moffat_norm = (1 + rr_gg) ** (-alpha)
        d_amplitude_moffat = moffat_norm + eta_gauss * gauss_norm
        d_x_mean = amplitude_moffat * (eta_gauss * (x - x_mean) / (stddev * stddev) * gauss_norm
                                       - alpha * moffat_norm * (-2 * x + 2 * x_mean) / (
                                               gamma * gamma * (1 + rr_gg)))
        d_stddev = amplitude_moffat * eta_gauss * rr / (stddev ** 3) * gauss_norm
        d_alpha = - amplitude_moffat * moffat_norm * np.log(1 + rr_gg)
        d_gamma = 2 * amplitude_moffat * alpha * moffat_norm * (rr_gg / (gamma * (1 + rr_gg)))
        d_saturation = saturation * np.zeros_like(x)
        return np.array([d_amplitude_moffat, d_x_mean, d_gamma, d_alpha, d_eta_gauss, d_stddev, d_saturation])

    @staticmethod
    def deriv(x, amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = (x - x_mean) * (x - x_mean)
        rr_gg = rr / (gamma * gamma)
        d_eta_gauss = np.exp(-(rr / (2. * stddev * stddev)))
        d_gauss = - eta_gauss * (x - x_mean) / (stddev * stddev) * d_eta_gauss
        d_moffat = -  alpha * 2 * (x - x_mean) / (gamma * gamma * (1 + rr_gg) ** (alpha + 1))
        return amplitude_moffat * (d_gauss + d_moffat)

    def interpolation(self, x_array):
        """

        Parameters
        ----------
        x_array: array_like
            The abscisse array to interpolate the model.

        Returns
        -------
        interp: callable
            Function corresponding to the interpolated model on the x_array array.

        Examples
        --------
        >>> x = np.arange(0, 60, 1)
        >>> p = [2,0,2,2,1,2,10]
        >>> PSF = PSF1D(*p)
        >>> interp = PSF.interpolation(x)
        >>> assert np.isclose(interp(p[1]), PSF.evaluate(p[1], *p))

        """
        params = [getattr(self, p).value for p in self.param_names]
        return interp1d(x_array, self.evaluate(x_array, *params), fill_value=0, bounds_error=False)

    def integrate(self, bounds=(-np.inf, np.inf), x_array=None):
        """
        Compute the integral of the PSF1D model. Bounds are -np.inf, np.inf by default, or provided
        if no x_array is provided. Otherwise the bounds comes from x_array edges.

        Parameters
        ----------
        x_array: array_like, optional
            If not None, the interpoalted PSF1D modelis used for integration (default: None).
        bounds: array_like, optional
            The bounds of the integral (default bounds=(-np.inf, np.inf)).

        Returns
        -------
        result: float
            The integral of the PSF1D model.

        Examples
        --------
        >>> x = np.arange(0, 60, 1)
        >>> p = [2,30,4,2,-0.5,1,10]
        >>> PSF = PSF1D(*p)
        >>> i = PSF.integrate()
        >>> assert np.isclose(i, 10.059742339728174)
        >>> i = PSF.integrate(bounds=(0,60), x_array=x)
        >>> assert np.isclose(i, 10.046698028728645)

        >>> import matplotlib.pyplot as plt
        >>> xx = np.arange(0, 60, 0.01)
        >>> plt.plot(xx, PSF.evaluate(xx, *p))
        >>> plt.plot(x, PSF.evaluate(x, *p))
        >>> plt.show()

        """
        params = [getattr(self, p).value for p in self.param_names]
        if x_array is None:
            i = quad(self.evaluate, bounds[0], bounds[1], args=tuple(params), limit=200)
            return i[0]
        else:
            return np.trapz(self.evaluate(x_array, *params), x_array)


class PSF2D(Fittable2DModel):
    inputs = ('x', 'y',)
    outputs = ('z',)

    amplitude = Parameter('amplitude', default=1)
    x_mean = Parameter('x_mean', default=0)
    y_mean = Parameter('y_mean', default=0)
    gamma = Parameter('gamma', default=3)
    alpha = Parameter('alpha', default=3)
    eta_gauss = Parameter('eta_gauss', default=0.5)
    stddev = Parameter('stddev', default=1)
    saturation = Parameter('saturation', default=1)

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
        rr_gg = rr / (gamma * gamma)
        a = amplitude * ((1 + rr_gg) ** (-alpha) + eta_gauss * np.exp(-(rr / (2. * stddev * stddev))))
        if isinstance(x, float) and isinstance(y, float):
            if a > saturation:
                return saturation
            elif a < 0.:
                return 0.
            else:
                return a
        else:
            a[np.where(a >= saturation)] = saturation
            a[np.where(a < 0.)] = 0.
            return a

    @staticmethod
    def fit_deriv(x, y, amplitude, x_mean, y_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
        rr_gg = rr / (gamma * gamma)
        gauss_norm = np.exp(-(rr / (2. * stddev * stddev)))
        d_eta_gauss = amplitude * gauss_norm
        moffat_norm = (1 + rr_gg) ** (-alpha)
        d_amplitude = moffat_norm + eta_gauss * gauss_norm
        d_x_mean = amplitude * eta_gauss * (x - x_mean) / (stddev * stddev) * gauss_norm \
                   - amplitude * alpha * moffat_norm * (-2 * x + 2 * x_mean) / (gamma ** 2 * (1 + rr_gg))
        d_y_mean = amplitude * eta_gauss * (y - y_mean) / (stddev * stddev) * gauss_norm \
                   - amplitude * alpha * moffat_norm * (-2 * y + 2 * y_mean) / (gamma ** 2 * (1 + rr_gg))
        d_stddev = amplitude * eta_gauss * rr / (stddev ** 3) * gauss_norm
        d_alpha = - amplitude * moffat_norm * np.log(1 + rr_gg)
        d_gamma = 2 * amplitude * alpha * moffat_norm * (rr_gg / (gamma * (1 + rr_gg)))
        d_saturation = saturation * np.zeros_like(x)
        return [d_amplitude, d_x_mean, d_y_mean, d_gamma, d_alpha, d_eta_gauss, d_stddev, d_saturation]


class ChromaticPSF:

    def __init__(self):
        # file_name="", image=None, order=1, target=None):
        # Spectrum.__init__(self, file_name=file_name, image=image, order=order, target=target)
        # self.PSF_params = PSF_params
        self.PSF1D = PSF1D()
        self.polynomial_orders = {key: 4 for key in self.PSF1D.param_names}
        self.polynomial_orders['saturation'] = 0

    def fit_params_from_transverse_profile_fit(self, Nx, PSF_params, pixels):
        """
        Transform the PSF_params array from fit_transverse_profile into a set of parameters
        for the chromatic PSF parameterisation.
        Fit polynomial functions across the pixels for each PSF1D
        parameters. The order of the function is given by self.polynomial_orders.

        Parameters
        ----------
        Nx: int
            Size in x direction
        PSF_params: array_like
            The output PSF_params array from fit_transverse_profile.
        pixels: array
            The output pixels array from fit_transverse_profile.

        Returns
        -------
        parameters: array_like
            A set of parameters that can be evaluated by the chromatic PSF class evaluate function.

        Examples
        --------

        # Build a mock spectrogram with random Poisson noise:
        >>> Nx = 100
        >>> Ny = 100
        >>> s = ChromaticPSF()
        >>> params = s.test_params(Nx, Ny)
        >>> data = s.evaluate(Nx, Ny, params)
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        # Fit the transverse profile:
        >>> PSF_params, flux, flux_integral, flux_err, fwhms, pixels = \
        fit_transverse_profile(data, data_errors, w=20, ws=[30,50], saturation=None, npixels=50, live_fit=True)

        # Fit the transverse profile parameters:
        >>> params = s.fit_params_from_transverse_profile_fit(Nx, PSF_params, pixels)
        >>> assert(params is not None)

        """
        nparams = PSF_params.shape[0]
        test = PSF1D()
        # all_pixels = np.arange(0, Nx)
        pixels = np.array(pixels) / Nx
        all_pixels = np.linspace(0, 1, Nx)
        chromatic_psf_params = np.array([])
        for i in range(0, nparams):
            if test.param_names[i] is 'amplitude_moffat':
                amplitude = np.interp(all_pixels, pixels, PSF_params[i]) #*(1+PSF_params[4]))
                #eta_gauss = np.polyval(np.polyfit(pixels, PSF_params[4], self.polynomial_orders['eta_gauss']),
                #                           all_pixels)
                #eta_gauss[eta_gauss < -0.9] = -0.9
                #amplitude /= (1+eta_gauss)
                chromatic_psf_params = np.concatenate([chromatic_psf_params, amplitude])
                # plt.plot(pixels, PSF_params[i], 'ro')
                # plt.plot(all_pixels, amplitude, 'b-')
                # plt.show()
            else:
                fit, cov, model = fit_poly1d(pixels, PSF_params[i], order=self.polynomial_orders[test.param_names[i]])
                chromatic_psf_params = np.concatenate([chromatic_psf_params, fit])
                # plt.plot(PSF_params[2], PSF_params[i], 'ro')
                # plt.plot(PSF_params[2], np.polyval(np.polyfit(PSF_params[2], PSF_params[i], 1), PSF_params[2]), 'b-')
                # plt.xlabel(test.param_names[2])
                # plt.ylabel(test.param_names[i])
                # plt.show()
                # plt.plot(pixels, PSF_params[i], 'ro')
                # plt.plot(pixels, np.polyval(fit, pixels), 'b-')
                # plt.xlabel('X [pixels]')
                # plt.ylabel(test.param_names[i])
                # plt.show()

        return chromatic_psf_params

    def set_bounds(self, data, saturation=None):
        if saturation is None:
            saturation = 2*np.max(data)
        Ny, Nx = data.shape
        bounds = []
        bounds.append([0.1 * np.max(data[:, x]) for x in range(Nx)])
        bounds.append([3.0 * np.max(data[:, x]) for x in range(Nx)])
        for ip, p in enumerate(self.PSF1D.param_names):
            tmp_bounds = []
            tmp_bounds.append([None] * (self.polynomial_orders[p]))
            tmp_bounds.append([None] * (self.polynomial_orders[p]))
            if p is "x_mean":
                tmp_bounds[0].append(0)
                tmp_bounds[1].append(Ny)
            elif p is "gamma":
                tmp_bounds[0].append(0)
                tmp_bounds[1].append(Ny/2)
            elif p is "alpha":
                tmp_bounds[0].append(1)
                tmp_bounds[1].append(10)
            elif p is "eta_gauss":
                tmp_bounds[0].append(-1)
                tmp_bounds[1].append(2)
            elif p is "stddev":
                tmp_bounds[0].append(0.1)
                tmp_bounds[1].append(Ny/2)
            elif p is "saturation":
                tmp_bounds = [[0], [2 * saturation]]
            elif p is "amplitude_moffat":
                continue
            else:
                sys.exit(f'Unknown parameter name {p} in set_bounds.')
            bounds[0] += tmp_bounds[0]
            bounds[1] += tmp_bounds[1]
        return np.array(bounds).T

    def evaluate(self, Nx, Ny, params):
        """
        Simulate a 2D spectrogram of size Nx times Ny.

        Parameters
        ----------
        Nx: int
            Size in x direction
        Ny: int
            Size in y direction
        params: array_like
            Parameter array of the model, in the form:
                * Nx first parameters are amplitudes for the Moffat transverse profiles
                * next parameters are polynomial coefficients for all the PSF1D parameters
                in the same order as in PSF1D definition, except amplitude_moffat

        Returns
        -------
        output: array
            A 2D array with the model

        Examples
        --------
        >>> Nx = 100
        >>> Ny = 20
        >>> s = ChromaticPSF()
        >>> params = s.test_params(Nx, Ny)
        >>> output = s.evaluate(Nx, Ny, params)
        >>> print(output)
        >>> import matplotlib.pyplot as plt
        >>> im = plt.imshow(output, origin='lower')
        >>> plt.colorbar(im)
        >>> plt.show()

        """
        # pixels = np.arange(Nx).astype(int)
        pixels = np.linspace(0, 1, Nx)
        PSF_params = []
        shift = 0
        for k, name in enumerate(self.PSF1D.param_names):
            if name == 'amplitude_moffat':
                PSF_params.append(params[:Nx])
            else:
                PSF_params.append(np.polyval(params[Nx + shift:Nx + shift + self.polynomial_orders[name] + 1], pixels))
                shift = shift + self.polynomial_orders[name] + 1
        PSF_params = np.array(PSF_params).T
        y = np.arange(Ny)
        output = np.zeros((Ny, Nx))
        #for x in pixels:
        for k in range(Nx):
            output[:, k] = PSF1D.evaluate(y, *PSF_params[k])
        return output

    def test_params(self, Nx, Ny):
        """
        A set of parameters to define a test spectrogram

        Parameters
        ----------
        Nx: int
            The size of the spectrogram along the dispersion direction.
        Ny: int
            The size of the spectrogram along the transverse direction.

        Returns
        -------
        parameters: list
            The list of the test parameters

        Examples
        --------
        >>> s = ChromaticPSF()
        >>> Nx = 5
        >>> Ny = 4
        >>> params = s.test_params(Nx, Ny)
        >>> print(params)
        [0, 100, 200, 300, 400, 0, 0, 0, 0, 2.0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 2, 0, 0, 0, -0.8, 1, 0, 0, 0, 0, 2, 8000]
        """
        params = [100 * i for i in range(Nx)]
        params += [0] * (self.polynomial_orders['x_mean']-1) + [0, Ny/2]  # y mean
        params += [0] * (self.polynomial_orders['gamma']-1) + [ 0, 5]  # gamma
        params += [0] * (self.polynomial_orders['alpha']-1) + [ 0, 2]  # alpha
        params += [0] * (self.polynomial_orders['eta_gauss']-1) + [ -0.8, 0]  # eta_gauss
        params += [0] * (self.polynomial_orders['stddev']-1) + [ 0, 2]  # stddev
        params += [8000]  # saturation
        return params


def fit_transverse_profile(data, err, w, ws, saturation=None, npixels=50, live_fit=False):
    """
    Fit the transverse profile of a 2D data image with a PSF1D profile.
    Loop is done on the x-axis direction, with npixels steps only to save time.
    An order 1 polynomial function is fitted to subtract the background for each pixel
    with a 3*sigma outlier removal procedure to remove background stars.
    
    Parameters
    ----------
    data: array
        The 2D array image. The transverse profile is fitted on the y direction for npixels pixels along the x direction.
    err: array 
        The uncertainties related to the data array.
    w: int
        Half width of central region where the spectrum is extracted and summed (default: 10)
    ws: list
        up/down region extension where the sky background is estimated with format [int, int] (default: [20,30])
    saturation: float, optional
        The saturation level of the image. Default is set to twice the maximum of the data array and has no effect.
    npixels: int, optional
        The number of transverse profiles to fit along the x direction (default: 50).
    live_fit: bool, optional
        If True, the transverse profilt fit is plotted in live accross the loop (default: False).

    Returns
    -------
    PSF_params: array
        The PSF1D best fit parameters along the x direction
    flux: array
        The sum of the pixel values along the y direction after background subtraction.
    flux_integral: array
        The integral of the fitted PSF1D profile.
    flux_err: array
        The uncertainty on the flux estimate, summing the square of the uncertainties along the y direction.
    fwhms: array
        The full width half maximum of the fitted PSF1D profile.
    pixels: array
        The pixel indices where the profile has been evaluated.

    Examples
    --------

    # Build a mock spectrogram with random Poisson noise:
    >>> Nx = 100
    >>> Ny = 100
    >>> s = ChromaticPSF()
    >>> params = s.test_params(Nx, Ny)
    >>> saturation = params[-1]
    >>> data = s.evaluate(Nx, Ny, params)
    >>> data = np.random.poisson(data)
    >>> data_errors = np.sqrt(data+1)

    # Fit the transverse profile:
    >>> import spectractor.parameters as parameters
    >>> parameters.DEBUG = True
    >>> PSF_params, flux, flux_integral, flux_err, fwhms, pixels = \
    fit_transverse_profile(data, data_errors, w=20, ws=[30,50], saturation=saturation, npixels=50, live_fit=True)
    >>> assert(np.all(np.isclose(pixels[:5], np.arange(0, Nx, Nx//50)[:5], rtol=1e-3)))

    """
    my_logger = set_logger(__name__)
    if saturation is None:
        saturation = 2*np.max(data)
    Ny, Nx = data.shape
    middle = Ny // 2
    index = np.arange(Ny)
    # Prepare the fit
    bgd_index = np.concatenate((np.arange(0, middle - ws[0]), np.arange(middle + ws[0], Ny))).astype(int)
    y = data[:, 0]
    guess = [np.nanmax(y) - np.nanmean(y), middle, 5, 2, 0, 2, saturation]
    maxi = np.abs(np.nanmax(y))
    bounds = [(0.1 * maxi, 3 * maxi), (middle - w, middle + w), (1, w), (1, 10), (-1, 0), (0.1, w),
              (0, 2 * saturation)]
    PSF_params = []
    flux = []
    flux_integral = []
    flux_err = []
    fwhms = []
    # set the pixel array where to fit the transverse profile
    # 50 steps starting from the middle to the edges
    pixels = np.arange(0, Nx, Nx // npixels)
    for x in pixels:
        guess = np.copy(guess)
        # fit the background with a polynomial function
        y = data[:, x]
        bgd = data[bgd_index, x]
        bgd_err = err[bgd_index, x]
        bgd_fit = fit_poly1d_outlier_removal(bgd_index, bgd, order=1, sigma=3.0, niter=2)
        signal = y - bgd_fit(index)
        # in case guess amplitude is too low
        pdf = np.abs(signal)
        sum = np.nansum(np.abs(signal))
        if sum > 0:
            pdf /= sum
        mean = np.nansum(pdf * index)
        std = np.sqrt(np.nansum(pdf * (index - mean) ** 2))
        maxi = np.abs(np.nanmax(signal))
        bounds[0] = (np.nanstd(bgd), 3 * maxi)
        # if guess[4] > -1:
        #    guess[0] = np.max(signal) / (1 + guess[4])
        if guess[0] * (1 + guess[4]) < 3 * np.nanstd(bgd):
            guess = [0.9*maxi, mean, std, 2, 0, std, saturation]
        if guess[0] * (1 + guess[4]) > 1.2 * maxi:
            guess = [0.9*maxi, mean, std, 2, 0, std, saturation]
        PSF_guess = PSF1D(*guess)
        # fit with outlier removal to clean background stars
        # first a simple moffat to get the general shape
        moffat1d = fit_moffat1d_outlier_removal(index, signal, sigma=5, niter=2,
                                                guess=guess[:4], bounds=np.array(bounds[:4]).T)
        guess[:4] = [getattr(moffat1d, p).value for p in moffat1d.param_names]
        # then PSF1D model using the result from the Moffat1D fit
        fit, outliers = fit_PSF1D_outlier_removal(index, signal, sub_errors=err[:, x], method='basinhopping',
                                                  guess=guess, bounds=bounds, sigma=5, niter=2,
                                                  niter_basinhopping=5, T_basinhopping=1)
        # test if 3 consecutive pixels are in the outlier list
        test = 0
        consecutive_outliers = False
        for o in range(1, len(outliers)):
            t = outliers[o] - outliers[o - 1]
            if t == 1:
                test += t
            else:
                test = 0
            if test > 1:
                consecutive_outliers = True
                break
        # test if the fit has badly fitted the two highest data points or the middle points
        test = np.copy(signal)
        max_badfit = False
        max_index = signal.argmax()
        if max_index in outliers:
            test[max_index] = 0
            if test.argmax() in outliers:
                max_badfit = True
        # if Ny//2 in outliers or Ny//2-1 in outliers or Ny//2+1 in outliers:
        #    max_badfit = True
        # if there are consecutive outliers or max is badly fitted, re-estimate the guess and refi
        if consecutive_outliers: # or max_badfit:
            my_logger.warning(f'\n\tRefit because of max_badfit={max_badfit} or '
                              f'consecutive_outliers={consecutive_outliers}')
            # tmp_guess = [getattr(fit, p).value for p in fit.param_names]
            # if guess[4] < 0 or fit.evaluate(float(max_index), *tmp_guess) > signal[max_index]:  # defocus
            guess = [1.3 * maxi, middle, guess[2], guess[3], -0.3, std/2, saturation]
            # else:
            #    guess = [0.9 * maxi, middle, guess[2], guess[3], 0.1, std, saturation]
            # bounds[0] = (np.nanstd(bgd), 3 * maxi)
            # bounds[3] = (np.nanstd(bgd), 2 * maxi)
            fit, outliers = fit_PSF1D_outlier_removal(index, signal, sub_errors=err[:, x], method='basinhopping',
                                                      guess=guess, bounds=bounds, sigma=5, niter=2,
                                                      niter_basinhopping=20, T_basinhopping=1)
        # compute the flux
        guess = [getattr(fit, p).value for p in fit.param_names]
        fwhm = fit.fwhm(x_array=index)
        PSF_params.append(guess)
        flux_integral.append(fit.integrate(bounds=(-10 * fwhm + guess[1], 10 * fwhm + guess[1]), x_array=index))
        flux_err.append(np.sqrt(np.sum(err[:, x] ** 2)))
        flux.append(np.sum(signal))
        fwhms.append(fwhm)
        if live_fit:
            plt.figure(figsize=(6, 6))
            plt.errorbar(np.arange(Ny), y, yerr=err[:, x], fmt='ro',
                         label="bgd data")
            plt.errorbar(bgd_index, bgd, yerr=bgd_err, fmt='bo', label="original data")
            plt.errorbar(outliers, data[outliers, x], yerr=err[outliers, x], fmt='go', label="outliers")
            plt.plot(bgd_index, bgd_fit(bgd_index), 'b--',
                     label="fitted bgd")
            plt.plot(index, PSF_guess(index) + bgd_fit(index), 'k--',
                     label="guessed profile")
            plt.plot(index, fit(index) + bgd_fit(index), 'b-',
                     label="fitted profile")
            ylim = plt.gca().get_ylim()
            PSF_moffat = Moffat1D(*guess[:4])
            plt.plot(index, PSF_moffat(index) + bgd_fit(index), 'b+',
                     label="fitted moffat")
            plt.gca().set_ylim(ylim)
            plt.legend(loc=2, numpoints=1)
            txt = ""
            for ip, p in enumerate(fit.param_names):
                txt += f'{p}: {getattr(fit, p).value:.4g}\n'
            plt.text(0.95, 0.95, txt, horizontalalignment='right',
                     verticalalignment='top',transform=plt.gca().transAxes)
            plt.title(f'x={x}')
            if parameters.DISPLAY:
                plt.draw()
                plt.pause(1e-8)
                plt.close()
    # End of loop, prepare the outputs
    flux = np.array(flux)
    flux_integral = np.array(flux_integral)
    flux_err = np.array(flux_err)
    fwhms = np.array(fwhms)
    PSF_params = np.array(PSF_params).T
    # Summary plot
    if parameters.DEBUG or True:
        fig, ax = plt.subplots(2, 1, sharex='all', figsize=(12, 6))
        test = PSF1D()
        PSF_models = []
        all_pixels = np.arange(Nx)
        for i in range(PSF_params.shape[0]):
            fit, cov, model = fit_poly1d(pixels, PSF_params[i], order=4)
            PSF_models.append(np.polyval(fit, all_pixels))
        for i in range(2, PSF_params.shape[0] - 1):
            p = ax[0].plot(pixels, PSF_params[i], label=test.param_names[i], marker='o', linestyle='none')
            ax[0].plot(all_pixels, PSF_models[i], color=p[0].get_color())
        img = np.zeros_like(data).astype(float)
        yy, xx = np.mgrid[:Ny, :Nx]
        # print('x', test.param_names)
        for x in pixels[::5]:
            params = [PSF_models[p][x] for p in range(len(PSF_params))]
            psf = PSF2D.evaluate(xx, yy, 1, x, Ny // 2, *params[2:])
            psf /= np.max(psf)
            img += psf
        ax[1].imshow(img, origin='lower')
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Y [pixels]')
        ax[0].set_ylabel('PSF1D parameters')
        ax[0].grid()
        ax[1].grid(color='white', ls='solid')
        ax[1].grid(True)
        ax[1].legend(title='PSF(x)')
        ax[0].legend()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        if parameters.DISPLAY:
            plt.show()

    return PSF_params, flux, flux_integral, flux_err, fwhms, pixels


def fit_chromatic_psf(Nx, Ny, data, guess, bounds=None, data_errors=None, ):
    """
    Fit a chromatic PSF model on 2D data.

    Parameters
    ----------
    Nx: int
        Size in x direction
    Ny: int
        Size in y direction
    data: array_like
        2D array containing the image data.
    guess: array_like
        First guess for the parameters, mandatory.
    bounds: array_like, optional
        Bounds for the parameters.
    data_errors: np.array
        the 2D array uncertainties.

    Returns
    -------
    parameters: array_like
        The best fit spectrogram parameter list.

    Examples
    --------

    # Build a mock spectrogram with random Poisson noise:
    >>> Nx = 100
    >>> Ny = 100
    >>> s = ChromaticPSF()
    >>> params = s.test_params(Nx, Ny)
    >>> saturation = params[-1]
    >>> data = s.evaluate(Nx, Ny, params)
    >>> data = np.random.poisson(data)
    >>> data_errors = np.sqrt(data+1)

    # Estimate the first guess values
    >>> PSF_params, flux, flux_integral, flux_err, fwhms, pixels = \
    fit_transverse_profile(data, data_errors, w=20, ws=[30,50], saturation=saturation, npixels=Nx//2, live_fit=False)
    >>> guess = s.fit_params_from_transverse_profile_fit(Nx, PSF_params, pixels)
    >>> im_guess = s.evaluate(Nx, Ny, guess)

    # Set bounds
    >>> bounds = s.set_bounds(data, saturation=saturation)

    # Fit the data:
    >>> p = fit_chromatic_psf(Nx, Ny, data, guess, bounds=bounds, data_errors=data_errors)
    >>> fit = s.evaluate(Nx, Ny, p)
    >>> print(guess)
    >>> print(p, np.array(p).shape)

    # Plot data, best fit model and residuals:
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(5, 1, sharex='all')
    >>> im0 = ax[0].imshow(data, origin='lower', aspect='auto')
    >>> plt.colorbar(im0, ax=ax[0])
    >>> im1 = ax[1].imshow(im_guess, origin='lower', aspect='auto')
    >>> plt.colorbar(im1, ax=ax[1])
    >>> im2 = ax[2].imshow((data-im_guess)/data_errors, origin='lower', aspect='auto')
    >>> plt.colorbar(im2, ax=ax[2])
    >>> im3 = ax[3].imshow(fit, origin='lower', aspect='auto')
    >>> plt.colorbar(im3, ax=ax[3])
    >>> im4 = ax[4].imshow((data-fit)/data_errors, origin='lower', aspect='auto')
    >>> plt.colorbar(im4, ax=ax[4])
    >>> plt.show()
    """
    my_logger = set_logger(__name__)
    s = ChromaticPSF()
    fit = s.evaluate(Nx, Ny, guess)

    def spectrogram_chisq(params, model, Nx, Ny, zz, zz_err=None):
        mod = model.evaluate(Nx, Ny, params)
        if zz_err is None:
            return np.nansum((mod - zz) ** 2)
        else:
            return np.nansum(((mod - zz) / zz_err) ** 2)

    my_logger.warning(f'\n\tStart chisq: {spectrogram_chisq(guess, s, Nx, Ny, data, data_errors)}')

    # TODO: add jacobian except if Nelder-Mead
    # TODO: test methods: Nelder_Mead OK but need to increase maxiter, Powell much too slow,
    # CG not so bad, BFGS not reach minimum
    my_logger = set_logger(__name__)
    #res = minimize(spectrogram_chisq, guess, method="Nelder-Mead", bounds=None,
    #               args=(s, Nx, Ny, data, data_errors), jac=None, options={'maxiter': 2000*len(guess), 'adaptive': True})
    my_logger.warning(f'\n{res}')
    fit = s.evaluate(Nx, Ny, res.x)
    my_logger.debug(f'\n\tSpectrogram best fitting parameters:\n{res.x}')

    return res.x


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


def fit_PSF2D(x, y, data, guess=None, bounds=None, data_errors=None, method='minimize'):
    """Fit a PSF 2D model with parameters :
        amplitude, x_mean, y_mean, stddev, eta, alpha, gamma, saturation
    using basin hopping global minimization method.

    Parameters
    ----------
    x: np.array
        2D array of the x coordinates from meshgrid.
    y: np.array
        2D array of the y coordinates from meshgrid.
    data: np.array
        the 2D array image.
    guess: array_like, optional
        List containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    data_errors: np.array
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
    >>> p = (50, 25, 25, 5, 1, -0.4, 1, 60)
    >>> Z = PSF.evaluate(X, Y, *p)
    >>> Z_err = np.sqrt(Z)/10.

    Prepare the fit:
    >>> guess = (52, 22, 22, 3.2, 1.2, -0.1, 2, 60)
    >>> bounds = ((1, 200), (10, 40), (10, 40), (0.5, 10), (0.5, 5), (-100, 200), (0.01, 10), (0, 400))

    Fit with error bars:
    >>> model = fit_PSF2D(X, Y, Z, guess=guess, bounds=bounds, data_errors=Z_err)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit without error bars:
    >>> model = fit_PSF2D(X, Y, Z, guess=guess, bounds=bounds, data_errors=None)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit with error bars and basin hopping method:
    >>> model = fit_PSF2D(X, Y, Z, guess=guess, bounds=bounds, data_errors=Z_err, method='basinhopping')
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    """

    model = PSF2D()
    my_logger = set_logger(__name__)
    if method == 'minimize':
        res = minimize(PSF2D_chisq, guess, method="L-BFGS-B", bounds=bounds,
                       args=(model, x, y, data, data_errors), jac=PSF2D_chisq_jac)
    elif method == 'basinhopping':
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, jac=PSF2D_chisq_jac,
                                args=(model, x, y, data, data_errors))
        res = basinhopping(PSF2D_chisq, guess, niter=20, minimizer_kwargs=minimizer_kwargs)
    else:
        my_logger.error(f'\n\tUnknown method {method}.')
        sys.exit()
    my_logger.debug(f'\n{res}')
    PSF = PSF2D(*res.x)
    my_logger.debug(f'\n\tPSF best fitting parameters:\n{PSF}')
    return PSF


def PSF1D_chisq(params, model, xx, yy, yy_err=None):
    m = model.evaluate(xx, *params)
    if len(m) == 0 or len(yy) == 0:
        return 1e20
    if np.any(m < 0) or np.any(m > 1.5 * np.max(yy)) or np.max(m) < 0.5 * np.max(yy):
        return 1e20
    diff = m - yy
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


def fit_PSF1D(x, data, guess=None, bounds=None, data_errors=None, method='minimize'):
    """Fit a PSF 1D model with parameters :
        amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation
    using basin hopping global minimization method.

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates.
    data: np.array
        the 1D array profile.
    guess: array_like, optional
        list containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    data_errors: np.array
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
    >>> p = (50, 25, 5, 1, -0.2, 1, 60)
    >>> Y = PSF.evaluate(X, *p)
    >>> Y_err = np.sqrt(Y)/10.

    Prepare the fit:
    >>> guess = (60, 20, 3.2, 1.2, -0.1, 2,  60)
    >>> bounds = ((0, 200), (10, 40), (0.5, 10), (0.5, 5), (-10, 200), (0.01, 10), (0, 400))

    Fit with error bars:
    >>> model = fit_PSF1D(X, Y, guess=guess, bounds=bounds, data_errors=Y_err)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit without error bars:
    >>> model = fit_PSF1D(X, Y, guess=guess, bounds=bounds, data_errors=None)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit with error bars and basin hopping method:
    >>> model = fit_PSF1D(X, Y, guess=guess, bounds=bounds, data_errors=Y_err, method='basinhopping')
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    """
    my_logger = set_logger(__name__)
    model = PSF1D()
    if method == 'minimize':
        res = minimize(PSF1D_chisq, guess, method="L-BFGS-B", bounds=bounds,
                       args=(model, x, data, data_errors), jac=PSF1D_chisq_jac)
    elif method == 'basinhopping':
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds,
                                args=(model, x, data, data_errors), jac=PSF1D_chisq_jac)
        res = basinhopping(PSF1D_chisq, guess, niter=20, minimizer_kwargs=minimizer_kwargs)
    else:
        my_logger.error(f'\n\tUnknown method {method}.')
        sys.exit()
    my_logger.debug(f'\n{res}')
    PSF = PSF1D(*res.x)
    my_logger.debug(f'\n\tPSF best fitting parameters:\n{PSF}')
    return PSF


def fit_PSF1D_outlier_removal(x, y, sub_errors=None, sigma=3.0, niter=3, guess=None, bounds=None, method='minimize',
                              niter_basinhopping=5, T_basinhopping=0.2):
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
            minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, jac=PSF1D_chisq_jac,
                                    args=(model, x[indices], y[indices], err))
            res = basinhopping(PSF1D_chisq, guess, T=T_basinhopping, niter=niter_basinhopping,
                               minimizer_kwargs=minimizer_kwargs)
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
        indices_no_nan = ~np.isnan(y)
        diff = model(x[indices_no_nan]) - y[indices_no_nan]
        if sub_errors is not None:
            outliers = np.where(np.abs(diff) / sub_errors[indices_no_nan] > sigma)[0]
        else:
            std = np.std(diff)
            outliers = np.where(np.abs(diff) / std > sigma)[0]
        if len(outliers) > 0:
            indices = [i for i in range(x.shape[0]) if i not in outliers]
        else:
            break
    my_logger.debug(f'\n\tPSF best fitting parameters:\n{model}')
    return model, outliers

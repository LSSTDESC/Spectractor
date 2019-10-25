import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import basinhopping, minimize
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
from iminuit import Minuit

from astropy.modeling import Fittable1DModel, Fittable2DModel, Parameter
from astropy.modeling.models import Moffat1D
from astropy.table import Table

from spectractor.tools import dichotomie, fit_poly1d, fit_moffat1d_outlier_removal, plot_image_simple
from spectractor.extractor.background import extract_background_photutils
from spectractor import parameters
from spectractor.config import set_logger
from spectractor.fit.fitter import FitWorkspace, run_minimisation


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

    param_titles = ["A", "y", r"\gamma", r"\alpha", r"\eta", r"\sigma", "saturation"]

    @staticmethod
    def evaluate(x, amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = (x - x_mean) * (x - x_mean)
        rr_gg = rr / (gamma * gamma)
        # use **(-alpha) instead of **(alpha) to avoid overflow power errors due to high alpha exponents
        # import warnings
        # warnings.filterwarnings('error')
        try:
            a = amplitude_moffat * ((1 + rr_gg) ** (-alpha) + eta_gauss * np.exp(-(rr / (2. * stddev * stddev))))
        except RuntimeWarning:  # pragma: no cover
            my_logger = set_logger(__name__)
            my_logger.warning(f"{[amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation]}")
            a = amplitude_moffat * eta_gauss * np.exp(-(rr / (2. * stddev * stddev)))
        return np.clip(a, 0, saturation)

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
        Compute the integral of the PSF model. Bounds are -np.inf, np.inf by default, or provided
        if no x_array is provided. Otherwise the bounds comes from x_array edges.

        Parameters
        ----------
        x_array: array_like, optional
            If not None, the interpoalted PSF modelis used for integration (default: None).
        bounds: array_like, optional
            The bounds of the integral (default bounds=(-np.inf, np.inf)).

        Returns
        -------
        result: float
            The integral of the PSF model.

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
        >>> plt.plot(xx, PSF.evaluate(xx, *p)) #doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object at 0x...>]
        >>> plt.plot(x, PSF.evaluate(x, *p)) #doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object at 0x...>]
        >>> if parameters.DISPLAY: plt.show()

        """
        params = [getattr(self, p).value for p in self.param_names]
        if x_array is None:
            i = quad(self.evaluate, bounds[0], bounds[1], args=tuple(params), limit=200)
            return i[0]
        else:
            return np.trapz(self.evaluate(x_array, *params), x_array)

    def fwhm(self, x_array=None):
        """
        Compute the full width half maximum of the PSF model with a dichotomie method.

        Parameters
        ----------
        x_array: array_like, optional
            An abscisse array is one wants to find FWHM on the interpolated PSF model
            (to smooth the spikes from spurious parameter sets).

        Returns
        -------
        FWHM: float
            The full width half maximum of the PSF model.

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
        >>> plt.plot(x, PSF.evaluate(x, *p)) #doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object at 0x...>]
        >>> if parameters.DISPLAY: plt.show()
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


class PSF2D(Fittable2DModel):
    inputs = ('x', 'y',)
    outputs = ('z',)

    amplitude_moffat = Parameter('amplitude_moffat', default=1)
    x_mean = Parameter('x_mean', default=0)
    y_mean = Parameter('y_mean', default=0)
    gamma = Parameter('gamma', default=3)
    alpha = Parameter('alpha', default=3)
    eta_gauss = Parameter('eta_gauss', default=0.)
    stddev = Parameter('stddev', default=1)
    saturation = Parameter('saturation', default=1)

    param_titles = ["A", "x", "y", "\gamma", r"\alpha", r"\eta", r"\sigma", "saturation"]

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
        rr_gg = rr / (gamma * gamma)
        a = amplitude * ((1 + rr_gg) ** (-alpha) + eta_gauss * np.exp(-(rr / (2. * stddev * stddev))))
        return np.clip(a, 0, saturation)

    @staticmethod
    def normalisation(amplitude, gamma, alpha, eta_gauss, stddev):
        return amplitude * ((np.pi * gamma * gamma) / (alpha - 1) + eta_gauss * 2 * np.pi * stddev * stddev)

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

    def interpolation(self, x_array, y_array):
        """

        Parameters
        ----------
        x_array: array_like
            The x array to interpolate the model.
        y_array: array_like
            The y array to interpolate the model.

        Returns
        -------
        interp: callable
            Function corresponding to the interpolated model on the (x_array,y_array) array.

        Examples
        --------
        >>> x = np.arange(0, 60, 1)
        >>> y = np.arange(0, 30, 1)
        >>> p = [2,30,15,2,2,1,2,10]
        >>> PSF = PSF2D(*p)
        >>> interp = PSF.interpolation(x, y)
        >>> assert np.isclose(interp(p[1], p[2]), PSF.evaluate(p[1], p[2], *p))

        """
        params = [getattr(self, p).value for p in self.param_names]
        xx, yy = np.meshgrid(x_array, y_array)
        return interp2d(x_array, y_array, self.evaluate(xx, yy, *params), fill_value=0, bounds_error=False)

    def integrate(self, bounds=(-np.inf, np.inf), x_array=None):
        """
        Ths PSF2D model is normalized to 1. The return of this function is trivially 1.

        Parameters
        ----------
        x_array: array_like, optional
            If not None, the interpoalted PSF modelis used for integration (default: None).
        bounds: array_like, optional
            The bounds of the integral (default bounds=(-np.inf, np.inf)).

        Returns
        -------
        result: float
            The integral of the PSF2D model, equal to one.

        Examples
        --------
        >>> p = [2,30,4,2,-0.5,1,10]
        >>> PSF = PSF2D(*p)
        >>> i = PSF.integrate()
        >>> assert np.isclose(i, 1)
        """
        return 1

    def fwhm(self, x_array=None, y_array=None):
        """
        Compute the full width half maximum of the PSF model with a dichotomie method.

        Parameters
        ----------
        x_array: array_like (optional)
            An x array is one wants to find FWHM on the interpolated PSF model
            (to smooth the spikes from spurious parameter sets).
        y_array: array_like
            An y array is one wants to find FWHM on the interpolated PSF model
            (to smooth the spikes from spurious parameter sets).

        Returns
        -------
        FWHM: float
            The full width half maximum of the PSF model.

        Examples
        --------
        >>> x = np.arange(0, 60, 1)
        >>> y = np.arange(0, 60, 1)
        >>> p = [2,30,30,4,2,-0.4,1,10]
        >>> PSF = PSF2D(*p)
        >>> fwhm = PSF.fwhm(x_array=x, y_array=y)
        >>> print(fwhm)
        -1
        """
        return -1


class ChromaticPSF:

    def __init__(self, PSF, Nx, Ny, deg=4, saturation=None, file_name=''):
        self.my_logger = set_logger(self.__class__.__name__)
        self.PSF = PSF
        self.deg = -1
        self.degrees = {}
        self.set_polynomial_degrees(deg)
        self.Nx = Nx
        self.Ny = Ny
        self.profile_params = np.zeros((Nx, len(self.PSF.param_names)))
        self.pixels = np.arange(Nx).astype(int)
        if file_name == '':
            arr = np.zeros((Nx, len(self.PSF.param_names) + 11))
            self.table = Table(arr, names=['lambdas', 'Dx', 'Dy', 'Dy_mean', 'flux_sum', 'flux_integral',
                                           'flux_err', 'fwhm', 'Dy_fwhm_sup', 'Dy_fwhm_inf', 'Dx_rot'] +
                                            list(self.PSF.param_names))
        else:
            self.table = Table.read(file_name)
        self.n_poly_params = len(self.table)
        self.fitted_pixels = np.arange(len(self.table)).astype(int)
        self.saturation = saturation
        if saturation is None:
            self.saturation = 1e20
            self.my_logger.warning(f"\n\tSaturation level should be given to instanciate the ChromaticPSF "
                                   f"object. self.saturation is set arbitrarily to 1e20. Good luck.")
        for name in self.PSF.param_names:
            self.n_poly_params += self.degrees[name] + 1
        self.poly_params = np.zeros(self.n_poly_params)
        self.alpha_max = 10
        self.poly_params_labels = []  # [f"a{k}" for k in range(self.poly_params.size)]
        self.poly_params_names = []  # "$a_{" + str(k) + "}$" for k in range(self.poly_params.size)]
        for ip, p in enumerate(self.PSF.param_names):
            if ip == 0:
                self.poly_params_labels += [f"{p}_{k}" for k in range(len(self.table))]
                self.poly_params_names += \
                    ["$" + self.PSF.param_titles[ip] + "_{(" + str(k) + ")}$" for k in range(len(self.table))]
            else:
                for k in range(self.degrees[p] + 1):
                    self.poly_params_labels.append(f"{p}_{k}")
                    self.poly_params_names.append("$" + self.PSF.param_titles[ip] + "_{(" + str(k) + ")}$")

    def set_polynomial_degrees(self, deg):
        self.deg = deg
        self.degrees = {key: deg for key in self.PSF.param_names}
        self.degrees['saturation'] = 0

    def fill_table_with_profile_params(self, profile_params):
        """
        Fill the table with the profile parameters.

        Parameters
        ----------
        profile_params: array
           a Nx * len(self.PSF.param_names) numpy array containing the PSF parameters as a function of pixels.

        Examples
        --------
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()
        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test)
        >>> s.fill_table_with_profile_params(profile_params)
        >>> assert(np.all(np.isclose(s.table['stddev'], 2*np.ones(100))))
        """
        for k, name in enumerate(self.PSF.param_names):
            self.table[name] = profile_params[:, k]

    def rotate_table(self, angle_degree):
        """
        In self.table, rotate the columns Dx, Dy, Dy_fwhm_inf and Dy_fwhm_sup by an angle
        given in degree. The results overwrite the previous columns in self.table.

        Parameters
        ----------
        angle_degree: float
            Rotation angle in degree

        Examples
        --------
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=8000)
        >>> s.table['Dx_rot'] = np.arange(100)
        >>> s.rotate_table(45)
        >>> assert(np.all(np.isclose(s.table['Dy'], -np.arange(100)/np.sqrt(2))))
        >>> assert(np.all(np.isclose(s.table['Dx'], np.arange(100)/np.sqrt(2))))
        >>> assert(np.all(np.isclose(s.table['Dy_fwhm_inf'], -np.arange(100)/np.sqrt(2))))
        >>> assert(np.all(np.isclose(s.table['Dy_fwhm_sup'], -np.arange(100)/np.sqrt(2))))
        """
        angle = angle_degree * np.pi / 180.
        rotmat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        # finish with Dy_mean to get correct Dx
        for name in ['Dy', 'Dy_fwhm_inf', 'Dy_fwhm_sup', 'Dy_mean']:
            vec = list(np.array([self.table['Dx_rot'], self.table[name]]).T)
            rot_vec = np.array([np.dot(rotmat, v) for v in vec])
            self.table[name] = rot_vec.T[1]
        self.table['Dx'] = rot_vec.T[0]

    def from_profile_params_to_poly_params(self, profile_params):
        """
        Transform the profile_params array from fit_transverse_PSF1D into a set of parameters
        for the chromatic PSF parameterisation.
        Fit polynomial functions across the pixels for each PSF
        parameters. The order of the function is given by self.degrees.

        Parameters
        ----------
        profile_params: array
            a Nx * len(self.PSF.param_names) numpy array containing the PSF parameters as a function of pixels.

        Returns
        -------
        profile_params: array_like
            A set of parameters that can be evaluated by the chromatic PSF class evaluate function.

        Examples
        --------

        # Build a mock spectrogram with random Poisson noise:
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()
        >>> data = s.evaluate(poly_params_test)
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        # From the polynomial parameters to the profile parameters:
        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test)
        >>> assert(np.all(np.isclose(profile_params[0], [0, 50, 5, 2, 0, 2, 8e3])))

        # From the profile parameters to the polynomial parameters:
        >>> profile_params = s.from_profile_params_to_poly_params(profile_params)
        >>> assert(np.all(np.isclose(profile_params, poly_params_test)))
        """
        pixels = np.linspace(-1, 1, len(self.table))
        poly_params = np.array([])
        amplitude = None
        for k, name in enumerate(self.PSF.param_names):
            if name is 'amplitude_moffat':
                amplitude = profile_params[:, k]
                poly_params = np.concatenate([poly_params, amplitude])
        if amplitude is None:
            self.my_logger.warning('\n\tAmplitude array not initialized. '
                                   'Polynomial fit for shape parameters will be unweighted.')
        for k, name in enumerate(self.PSF.param_names):
            if name is not 'amplitude_moffat':
                weights = np.copy(amplitude)
                if name is 'stddev':
                    i_eta = list(self.PSF.param_names).index('eta_gauss')
                    weights = np.abs(amplitude * profile_params[:, i_eta])
                fit = np.polynomial.legendre.legfit(pixels, profile_params[:, k], deg=self.degrees[name], w=weights)
                poly_params = np.concatenate([poly_params, fit])
        return poly_params

    def from_table_to_profile_params(self):
        """
        Extract the profile parameters from self.table and fill an array of profile parameters.

        Parameters
        ----------

        Returns
        -------
        profile_params: array
            Nx * len(self.PSF.param_names) numpy array containing the PSF parameters as a function of pixels.

        Examples
        --------
        >>> from spectractor.extractor.spectrum import Spectrum
        >>> s = Spectrum('./tests/data/reduc_20170530_134_spectrum.fits')
        >>> profile_params = s.chromatic_psf.from_table_to_profile_params()
        >>> assert(profile_params.shape == (s.chromatic_psf.Nx, len(s.chromatic_psf.PSF.param_names)))
        >>> assert not np.all(np.isclose(profile_params, np.zeros_like(profile_params)))
        """
        profile_params = np.zeros((len(self.table), len(self.PSF.param_names)))
        for k, name in enumerate(self.PSF.param_names):
            profile_params[:, k] = self.table[name]
        return profile_params

    def from_table_to_poly_params(self):
        """
        Extract the polynomial parameters from self.table and fill an array with polynomial parameters.

        Parameters
        ----------

        Returns
        -------
        poly_params: array_like
            A set of polynomial parameters that can be evaluated by the chromatic PSF class evaluate function.

        Examples
        --------
        >>> from spectractor.extractor.spectrum import Spectrum
        >>> s = Spectrum('./tests/data/reduc_20170530_134_spectrum.fits')
        >>> poly_params = s.chromatic_psf.from_table_to_poly_params()
        >>> assert(poly_params.size > s.chromatic_psf.Nx)
        >>> assert(len(poly_params.shape)==1)
        >>> assert not np.all(np.isclose(poly_params, np.zeros_like(poly_params)))
        """
        profile_params = self.from_table_to_profile_params()
        poly_params = self.from_profile_params_to_poly_params(profile_params)
        return poly_params

    def from_poly_params_to_profile_params(self, poly_params, force_positive=False):
        """
        Evaluate the PSF profile parameters from the polynomial coefficients. If poly_params length is smaller
        than self.Nx, it is assumed that the amplitude_moffat parameters are not included and set to arbitrarily to 1.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
                * Nx first parameters are amplitudes for the Moffat transverse profiles
                * next parameters are polynomial coefficients for all the PSF parameters
                in the same order as in PSF definition, except amplitude_moffat
        force_positive: bool, optional
            Force some profile parameters to be positive despite the polynomial coefficients (default: False)

        Returns
        -------
        profile_params: array
            Nx * len(self.PSF.param_names) numpy array containing the PSF parameters as a function of pixels.

        Examples
        --------

        # Build a mock spectrogram with random Poisson noise:
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=1, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()
        >>> data = s.evaluate(poly_params_test)
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        # From the polynomial parameters to the profile parameters:
        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test)
        >>> assert(np.all(np.isclose(profile_params[0], [0, 50, 5, 2, 0, 2, 8e3])))

        # From the profile parameters to the polynomial parameters:
        >>> profile_params = s.from_profile_params_to_poly_params(profile_params)
        >>> assert(np.all(np.isclose(profile_params, poly_params_test)))

        # From the polynomial parameters to the profile parameters without Moffat amplitudes:
        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test[100:])
        >>> assert(np.all(np.isclose(profile_params[0], [1, 50, 5, 2, 0, 2, 8e3])))

        """
        l = len(self.table)
        pixels = np.linspace(-1, 1, l)
        profile_params = np.zeros((l, len(self.PSF.param_names)))
        shift = 0
        for k, name in enumerate(self.PSF.param_names):
            if name == 'amplitude_moffat':
                if len(poly_params) > l:
                    profile_params[:, k] = poly_params[:l]
                else:
                    profile_params[:, k] = np.ones(l)
            else:
                if len(poly_params) > l:
                    profile_params[:, k] = \
                        np.polynomial.legendre.legval(pixels,
                                                      poly_params[
                                                      l + shift:l + shift + self.degrees[name] + 1])
                else:
                    p = poly_params[shift:shift + self.degrees[name] + 1]
                    if len(p) > 0:  # to avoid saturation parameters in case not set
                        profile_params[:, k] = np.polynomial.legendre.legval(pixels, p)
                shift = shift + self.degrees[name] + 1
        if force_positive:
            for k, name in enumerate(self.PSF.param_names):
                # if name == "x_mean":
                #    profile_params[profile_params[:, k] <= 0.1, k] = 1e-1
                #    profile_params[profile_params[:, k] >= self.Ny, k] = self.Ny
                if name == "alpha":
                    profile_params[profile_params[:, k] <= 1, k] = 1
                    # profile_params[profile_params[:, k] >= 6, k] = 6
                if name == "gamma":
                    profile_params[profile_params[:, k] <= 0.1, k] = 1e-1
                if name == "stddev":
                    profile_params[profile_params[:, k] <= 0.1, k] = 1e-1
                if name == "eta_gauss":
                    profile_params[profile_params[:, k] > 0, k] = 0
                    profile_params[profile_params[:, k] < -1, k] = -1
        return profile_params

    def from_profile_params_to_shape_params(self, profile_params):
        """
        Compute the PSF integrals and FWHMS given the profile_params array and fill the table.

        Parameters
        ----------
        profile_params: array
         a Nx * len(self.PSF.param_names) numpy array containing the PSF parameters as a function of pixels.

        Examples
        --------
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()
        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test)
        >>> s.from_profile_params_to_shape_params(profile_params)
        >>> assert(np.isclose(s.table['fwhm'][-1], 12.7851))
        """
        self.fill_table_with_profile_params(profile_params)
        pixel_y = np.arange(self.Ny).astype(int)
        pixel_x = np.arange(self.Nx).astype(int)
        for x in pixel_x:
            p = profile_params[x, :]
            self.PSF.parameters = p
            fwhm = self.PSF.fwhm(x_array=pixel_y)
            integral = self.PSF.integrate(bounds=(-10 * fwhm + p[1], 10 * fwhm + p[1]), x_array=pixel_y)
            self.table['flux_integral'][x] = integral
            self.table['fwhm'][x] = fwhm
            self.table['Dy_mean'][x] = 0

    def set_bounds(self, data=None):
        """
        This function returns an array of bounds for iminuit. It is very touchy, change the values with caution !

        Parameters
        ----------
        data: array_like, optional
            The data array, to set the bounds for the amplitude using its maximum.
            If None is provided, no bounds are provided for the amplitude parameters.

        Returns
        -------
        bounds: array_like
            2D array containing the pair of bounds for each polynomial parameters.

        """
        if self.saturation is None:
            self.saturation = 2 * np.max(data)
        if data is not None:
            Ny, Nx = data.shape
            bounds = [[0.1 * np.max(data[:, x]) for x in range(Nx)], [100.0 * np.max(data[:, x]) for x in range(Nx)]]
        else:
            bounds = [[], []]
        for k, name in enumerate(self.PSF.param_names):
            tmp_bounds = [[-np.inf] * (1 + self.degrees[name]), [np.inf] * (1 + self.degrees[name])]
            # if name is "x_mean":
            #      tmp_bounds[0].append(0)
            #      tmp_bounds[1].append(Ny)
            # elif name is "gamma":
            #      tmp_bounds[0].append(0)
            #      tmp_bounds[1].append(None) # Ny/2
            # elif name is "alpha":
            #      tmp_bounds[0].append(1)
            #      tmp_bounds[1].append(None) # 10
            # elif name is "eta_gauss":
            #     tmp_bounds[0].append(-1)
            #     tmp_bounds[1].append(0)
            # elif name is "stddev":
            #     tmp_bounds[0].append(0.1)
            #     tmp_bounds[1].append(Ny / 2)
            if name is "saturation":
                if data is not None:
                    tmp_bounds = [[0.1 * np.max(data)], [2 * self.saturation]]
                else:
                    tmp_bounds = [[0], [2 * self.saturation]]
            elif name is "amplitude_moffat":
                continue
            # else:
            #     self.my_logger.error(f'Unknown parameter name {name} in set_bounds.')
            bounds[0] += tmp_bounds[0]
            bounds[1] += tmp_bounds[1]
        return np.array(bounds).T

    def check_bounds(self, poly_params, noise_level=0):
        """
        Evaluate the PSF profile parameters from the polynomial coefficients and check if they are within priors.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
                * Nx first parameters are amplitudes for the Moffat transverse profiles
                * next parameters are polynomial coefficients for all the PSF parameters
                in the same order as in PSF definition, except amplitude_moffat
        noise_level: float, optional
            Noise level to set minimal boundary for amplitudes (negatively).

        Returns
        -------
        in_bounds: bool
            Return True if all parameters respect the model parameter priors.

        """
        in_bounds = True
        penalty = 0
        outbound_parameter_name = ""
        profile_params = self.from_poly_params_to_profile_params(poly_params)
        for k, name in enumerate(self.PSF.param_names):
            p = profile_params[:, k]
            if name == 'amplitude_moffat':
                if np.any(p < -noise_level):
                    in_bounds = False
                    penalty += np.abs(np.sum(profile_params[p < -noise_level, k]))  # / np.mean(np.abs(p))
                    outbound_parameter_name += name + ' '
            # elif name is "x_mean":
            #     if np.any(p < 0):
            #         penalty += np.abs(np.sum(profile_params[p < 0, k])) / np.abs(np.mean(p))
            #         in_bounds = False
            #         outbound_parameter_name += name + ' '
            #     if np.any(p > self.Ny):
            #         penalty += np.sum(profile_params[p > self.Ny, k] - self.Ny) / np.abs(np.mean(p))
            #         in_bounds = False
            #         outbound_parameter_name += name + ' '
            # elif name is "gamma":
            #     if np.any(p < 0) or np.any(p > self.Ny):
            #         in_bounds = False
            #         penalty = 1
            #         break
            elif name is "alpha":
                if np.any(p > self.alpha_max):
                    penalty += np.sum(profile_params[p > self.alpha_max, k] - self.alpha_max) / np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
                if np.any(p < 1.1):
                    penalty += np.sum(1.1 - profile_params[p < 1.1, k])
                    penalty /= np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
            elif name is "eta_gauss":
                if np.any(p > 0):
                    penalty += np.sum(profile_params[p > 0, k])
                    # penalty /= np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
                if np.any(p < -1):
                    penalty += np.sum(-1 - profile_params[p < -1, k]) / np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
            # elif name is "stddev":
            #     if np.any(p < 0) or np.any(p > self.Ny):
            #         in_bounds = False
            #         penalty = 1
            #         break
            elif name is "saturation":
                continue
            else:
                continue
            # else:
            #    self.my_logger.error(f'Unknown parameter name {name} in set_bounds.')
        penalty *= self.Nx * self.Ny
        return in_bounds, penalty, outbound_parameter_name

    def get_distance_along_dispersion_axis(self, shift_x=0, shift_y=0):
        return np.sqrt((self.table['Dx'] - shift_x) ** 2 + (self.table['Dy_mean'] - shift_y) ** 2)

    def evaluate(self, poly_params):  # pragma: no cover
        """
        Dummy function to simulate a 2D spectrogram of size Nx times Ny.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
                * Nx first parameters are amplitudes for the Moffat transverse profiles
                * next parameters are polynomial coefficients for all the PSF parameters
                in the same order as in PSF definition, except amplitude_moffat

        Returns
        -------
        output: array
            A 2D array with the model

        Examples
        --------
        >>> s = ChromaticPSF1D(Nx=100, Ny=20, deg=4, saturation=8000)
        >>> poly_params = s.generate_test_poly_params()
        >>> output = s.evaluate(poly_params)
        >>> assert not np.all(np.isclose(output, 0))

        >>> import matplotlib.pyplot as plt
        >>> im = plt.imshow(output, origin='lower')  #doctest: +ELLIPSIS
        >>> plt.colorbar(im)  #doctest: +ELLIPSIS
        <matplotlib.colorbar.Colorbar object at 0x...>
        >>> if parameters.DISPLAY: plt.show()

        """
        output = np.zeros((self.Ny, self.Nx))
        return output

    def plot_summary(self, truth=None):
        fig, ax = plt.subplots(2, 1, sharex='all', figsize=(12, 6))
        PSF_models = []
        PSF_truth = []
        if truth is not None:
            PSF_truth = truth.from_poly_params_to_profile_params(truth.poly_params)
        all_pixels = np.arange(self.profile_params.shape[0])
        for i, name in enumerate(self.PSF.param_names):
            fit, cov, model = fit_poly1d(all_pixels, self.profile_params[:, i], order=self.degrees[name])
            PSF_models.append(np.polyval(fit, all_pixels))
        for i, name in enumerate(self.PSF.param_names):
            p = ax[0].plot(all_pixels, self.profile_params[:, i], marker='+', linestyle='none')
            ax[0].plot(self.fitted_pixels, self.profile_params[self.fitted_pixels, i], label=name,
                       marker='o', linestyle='none', color=p[0].get_color())
            if i > 0:
                ax[0].plot(all_pixels, PSF_models[i], color=p[0].get_color())
            if truth is not None:
                ax[0].plot(all_pixels, PSF_truth[:, i], color=p[0].get_color(), linestyle='--')
        img = np.zeros((self.Ny, self.Nx)).astype(float)
        yy, xx = np.mgrid[:self.Ny, :self.Nx]
        for x in all_pixels[::self.Nx // 10]:
            params = [PSF_models[p][x] for p in range(len(self.PSF.param_names))]
            psf = PSF2D.evaluate(xx, yy, 1, x, self.Ny // 2, *params[-5:])
            psf /= np.max(psf)
            img += psf
        ax[1].imshow(img, origin='lower')
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Y [pixels]')
        ax[0].set_ylabel('PSF parameters')
        ax[0].grid()
        ax[1].grid(color='white', ls='solid')
        ax[1].grid(True)
        ax[0].set_yscale('symlog', linthreshy=10)
        ax[1].legend(title='PSF(x)')
        ax[0].legend()
        fig.tight_layout()
        # fig.subplots_adjust(hspace=0)
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()

    def plot_chromatic_PSF1D_residuals(self, bgd, data, data_errors, guess=None, live_fit=False, title=""):
        """Plot the residuals after fit_chromatic_PSF1D function.

        Parameters
        ----------
        bgd: array_like
            The 2D background array.
        data: array_like
            The 2D data array.
        data_errors: array_like
            The 2D data uncertainty array.
        guess: array_like, optional
            The guessed profile before the fit (default: None).
        live_fit: bool
            If True, the plot is shown during the fitting procedure (default: False).
        title: str, optional
            Title of the plot (default: "").
        """
        fig, ax = plt.subplots(5, 1, sharex='all', figsize=(6, 8))
        plt.title(title)
        im0 = ax[0].imshow(data, origin='lower', aspect='auto')
        ax[0].set_title('Data')
        plt.colorbar(im0, ax=ax[0])
        im_guess = self.evaluate(guess) + bgd
        im1 = ax[1].imshow(im_guess, origin='lower', aspect='auto')
        fit = self.evaluate(self.poly_params) + bgd
        ax[1].set_title('Guess')
        plt.colorbar(im1, ax=ax[1])
        im2 = ax[2].imshow((data - im_guess) / data_errors, origin='lower', aspect='auto')
        ax[2].set_title('(Data-Guess)/Data_errors')
        plt.colorbar(im2, ax=ax[2])
        im3 = ax[3].imshow(fit, origin='lower', aspect='auto')
        ax[3].set_title(title)
        plt.colorbar(im3, ax=ax[3])
        im4 = ax[4].imshow((data - fit) / data_errors, origin='lower', aspect='auto')
        ax[4].set_title('(Data-Fit)/Data_errors')
        plt.colorbar(im4, ax=ax[4])
        fig.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            if live_fit:
                plt.draw()
                plt.pause(1e-8)
                plt.close()
            else:
                plt.show()
        plt.close()

    def fit_transverse_PSF1D_profile(self, data, err, w, ws, pixel_step=1, bgd_model_func=None, saturation=None,
                                     live_fit=False, sigma=5):
        """
        Fit the transverse profile of a 2D data image with a PSF profile.
        Loop is done on the x-axis direction.
        An order 1 polynomial function is fitted to subtract the background for each pixel
        with a 3*sigma outlier removal procedure to remove background stars.

        Parameters
        ----------
        data: array
            The 2D array image. The transverse profile is fitted on the y direction
            for all pixels along the x direction.
        err: array
            The uncertainties related to the data array.
        w: int
            Half width of central region where the spectrum is extracted and summed (default: 10)
        ws: list
            up/down region extension where the sky background is estimated with format [int, int] (default: [20,30])
        pixel_step: int, optional
            The step in pixels between the slices to be fitted (default: 1).
            The values for the skipped pixels are interpolated with splines from the fitted parameters.
        bgd_model_func: callable, optional
            A 2D function to model the extracted background (default: None -> null background)
        saturation: float, optional
            The saturation level of the image. Default is set to twice the maximum of the data array and has no effect.
        live_fit: bool, optional
            If True, the transverse profile fit is plotted in live accross the loop (default: False).
        sigma: int
            Sigma for outlier rejection (default: 5).

        Examples
        --------

        # Build a mock spectrogram with random Poisson noise:
        >>> s0 = ChromaticPSF1D(Nx=100, Ny=100, saturation=1000)
        >>> params = s0.generate_test_poly_params()
        >>> saturation = params[-1]
        >>> data = s0.evaluate(params)
        >>> bgd = 10*np.ones_like(data)
        >>> xx, yy = np.meshgrid(np.arange(s0.Nx), np.arange(s0.Ny))
        >>> bgd += 1000*np.exp(-((xx-20)**2+(yy-10)**2)/(2*2))
        >>> data += bgd
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        # Extract the background
        >>> bgd_model_func = extract_background_photutils(data, data_errors, ws=[30,50])

        # Fit the transverse profile:
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50], pixel_step=10,
        ... bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False, sigma=5)
        >>> assert(np.all(np.isclose(s.pixels[:5], np.arange(s.Nx)[:5], rtol=1e-3)))
        >>> assert(not np.any(np.isclose(s.table['flux_sum'][3:6], np.zeros(s.Nx)[3:6], rtol=1e-3)))
        >>> assert(np.all(np.isclose(s.table['Dy'][-10:-1], np.zeros(s.Nx)[-10:-1], rtol=1e-2)))
        >>> s.plot_summary(truth=s0)
        """
        my_logger = set_logger(__name__)
        if saturation is None:
            saturation = 2 * np.max(data)
        Ny, Nx = data.shape
        middle = Ny // 2
        index = np.arange(Ny)
        # Prepare the fit: start with the maximum of the spectrum
        ymax_index = int(np.unravel_index(np.argmax(data[middle - ws[0]:middle + ws[0], :]), data.shape)[1])
        bgd_index = np.concatenate((np.arange(0, middle - ws[0]), np.arange(middle + ws[0], Ny))).astype(int)
        y = data[:, ymax_index]
        guess = [np.nanmax(y) - np.nanmean(y), middle, 5, 2, 0, 2, saturation]
        maxi = np.abs(np.nanmax(y))
        bounds = [(0.1 * maxi, 2 * maxi), (middle - w, middle + w), (0.1, Ny // 2), (0.1, self.alpha_max), (-1, 0),
                  (0.1, Ny // 2),
                  (0, 2 * saturation)]
        # first fit with moffat only to initialize the guess
        # hypothesis that max of spectrum if well describe by a focused PSF
        bgd = data[bgd_index, ymax_index]
        if bgd_model_func is not None:
            signal = y - bgd_model_func(ymax_index, index)[:, 0]
        else:
            signal = y
        fit = fit_moffat1d_outlier_removal(index, signal, sigma=sigma, niter=2,
                                           guess=guess[:4], bounds=np.array(bounds[:4]).T)
        moffat_guess = [getattr(fit, p).value for p in fit.param_names]
        signal_width_guess = moffat_guess[2]
        bounds[2] = (0.1, min(Ny // 2, 5*signal_width_guess))
        bounds[5] = (0.1, min(Ny // 2, 5*signal_width_guess))
        guess[:4] = moffat_guess
        init_guess = np.copy(guess)
        # Go from max to right, then from max to left
        # includes the boundaries to avoid Runge phenomenum in chromatic_fit
        pixel_range = list(np.arange(ymax_index, Nx, pixel_step).astype(int))
        if Nx - 1 not in pixel_range:
            pixel_range.append(Nx - 1)
        pixel_range += list(np.arange(ymax_index, -1, -pixel_step).astype(int))
        if 0 not in pixel_range:
            pixel_range.append(0)
        pixel_range = np.array(pixel_range)
        for x in pixel_range:
            guess = np.copy(guess)
            if x == ymax_index:
                guess = np.copy(init_guess)
            # fit the background with a polynomial function
            y = data[:, x]
            if bgd_model_func is not None:
                # x_array = [x] * index.size
                signal = y - bgd_model_func(x, index)[:, 0]
            else:
                signal = y
            # in case guess amplitude is too low
            pdf = np.abs(signal)
            signal_sum = np.nansum(np.abs(signal))
            if signal_sum > 0:
                pdf /= signal_sum
            mean = np.nansum(pdf * index)
            std = np.sqrt(np.nansum(pdf * (index - mean) ** 2))
            maxi = np.abs(np.nanmax(signal))
            bounds[0] = (0.1 * np.nanstd(bgd), 2 * np.nanmax(y[middle - ws[0]:middle + ws[0]]))
            # if guess[4] > -1:
            #    guess[0] = np.max(signal) / (1 + guess[4])
            if guess[0] * (1 + guess[4]) < 3 * np.nanstd(bgd):
                guess = [0.9 * maxi, mean, std, 2, 0, std, saturation]
            # if guess[0] * (1 + guess[4]) > 1.2 * maxi:
            #     guess = [0.9 * maxi, mean, std, 2, 0, std, saturation]
            PSF_guess = PSF1D(*guess)
            fit, outliers = fit_PSF1D_minuit_outlier_removal(index, signal, guess=guess, bounds=bounds,
                                                             data_errors=err[:, x], sigma=sigma, niter=2, consecutive=4)
            # It is better not to propagate the guess to further pixel columns
            # otherwise fit_chromatic_psf1D is more likely to get trapped in a local minimum
            # Randomness of the slice fit is better :
            # guess = [getattr(fit, p).value for p in fit.param_names]
            best_fit = [getattr(fit, p).value for p in fit.param_names]
            self.profile_params[x, 0] = best_fit[0]
            self.profile_params[x, -6:] = best_fit[1:]
            self.table['flux_err'][x] = np.sqrt(np.sum(err[:, x] ** 2))
            self.table['flux_sum'][x] = np.sum(signal)
            if live_fit and parameters.DISPLAY:  # pragma: no cover
                plot_transverse_PSF1D_profile(x, index, bgd_index, data, err, fit, bgd_model_func, best_fit,
                                              PSF_guess, outliers, sigma, live_fit)
        # interpolate the skipped pixels with splines
        x = np.arange(Nx)
        xp = np.array(sorted(set(list(pixel_range))))
        self.fitted_pixels = xp
        for i in range(len(self.PSF.param_names)):
            yp = self.profile_params[xp, i]
            self.profile_params[:, i] = interp1d(xp, yp, kind='cubic')(x)
        self.table['flux_sum'] = interp1d(xp, self.table['flux_sum'][xp], kind='cubic', bounds_error=False,
                                          fill_value='extrapolate')(x)
        self.table['flux_err'] = interp1d(xp, self.table['flux_err'][xp], kind='cubic', bounds_error=False,
                                          fill_value='extrapolate')(x)
        self.poly_params = self.from_profile_params_to_poly_params(self.profile_params)
        self.from_profile_params_to_shape_params(self.profile_params)


class ChromaticPSF1D(ChromaticPSF):

    def __init__(self, Nx, Ny, deg=4, saturation=None, file_name=''):
        PSF = PSF1D()
        ChromaticPSF.__init__(self, PSF, Nx=Nx, Ny=Ny, deg=deg, saturation=saturation, file_name=file_name)
        self.my_logger = set_logger(self.__class__.__name__)

    def generate_test_poly_params(self):
        """
        A set of parameters to define a test spectrogram

        Parameters
        ----------

        Returns
        -------
        profile_params: array
            The list of the test parameters

        Examples
        --------
        >>> s = ChromaticPSF1D(Nx=5, Ny=4, deg=1, saturation=8000)
        >>> params = s.generate_test_poly_params()
        >>> assert(np.all(np.isclose(params, [ 0, 50, 100, 150, 200, 2, 0, 5, 0, 2, 0, -0.4, -0.4, 2, 0, 8000])))
        """
        params = [50 * i for i in range(self.Nx)]
        params += [0.] * (self.degrees['x_mean'] - 1) + [0, self.Ny / 2]  # y mean
        params += [0.] * (self.degrees['gamma'] - 1) + [0, 5]  # gamma
        params += [0.] * (self.degrees['alpha'] - 1) + [0, 2]  # alpha
        params += [0.] * (self.degrees['eta_gauss'] - 1) + [-0.4, -0.4]  # eta_gauss
        params += [0.] * (self.degrees['stddev'] - 1) + [0, 2]  # stddev
        params += [8000.]  # saturation
        poly_params = np.zeros_like(params)
        poly_params[:self.Nx] = params[:self.Nx]
        index = self.Nx - 1
        self.saturation = 8000.
        for name in self.PSF.param_names:
            if name == 'amplitude_moffat':
                continue
            else:
                shift = self.degrees[name] + 1
                c = np.polynomial.legendre.poly2leg(params[index + shift:index:-1])
                coeffs = np.zeros(shift)
                coeffs[:c.size] = c
                poly_params[index + 1:index + shift + 1] = coeffs
                index = index + shift
        return poly_params

    def evaluate(self, poly_params):
        """
        Simulate a 2D spectrogram of size Nx times Ny with transverse 1D PSF profiles.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
                * Nx first parameters are amplitudes for the Moffat transverse profiles
                * next parameters are polynomial coefficients for all the PSF parameters
                in the same order as in PSF definition, except amplitude_moffat

        Returns
        -------
        output: array
            A 2D array with the model

        Examples
        --------
        >>> s = ChromaticPSF1D(Nx=100, Ny=20, deg=4, saturation=8000)
        >>> poly_params = s.generate_test_poly_params()
        >>> output = s.evaluate(poly_params)

        >>> import matplotlib.pyplot as plt
        >>> im = plt.imshow(output, origin='lower')  #doctest: +ELLIPSIS
        >>> plt.colorbar(im)  #doctest: +ELLIPSIS
        <matplotlib.colorbar.Colorbar object at 0x...>
        >>> if parameters.DISPLAY: plt.show()

        """
        profile_params = self.from_poly_params_to_profile_params(poly_params)
        y = np.arange(self.Ny)
        output = np.zeros((self.Ny, self.Nx))
        for k in range(self.Nx):
            output[:, k] = PSF1D.evaluate(y, *profile_params[k])
        return output

    def fit_chromatic_PSF1D_minuit(self, data, bgd_model_func=None, data_errors=None):
        """
        Fit a chromatic PSF model on 2D data.

        Parameters
        ----------
        data: array_like
            2D array containing the image data.
        bgd_model_func: callable, optional
            A 2D function to model the extracted background (default: None -> null background)
        data_errors: np.array
            the 2D array uncertainties.

        Examples
        --------

        # Set the parameters
        >>> parameters.PIXDIST_BACKGROUND = 40
        >>> parameters.PIXWIDTH_BACKGROUND = 10
        >>> parameters.PIXWIDTH_SIGNAL = 30

        # Build a mock spectrogram with random Poisson noise:
        >>> s0 = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=1000)
        >>> params = s0.generate_test_poly_params()
        >>> s0.poly_params = params
        >>> saturation = params[-1]
        >>> data = s0.evaluate(params)
        >>> bgd = 10*np.ones_like(data)
        >>> data += bgd
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        # Extract the background
        >>> bgd_model_func = extract_background_photutils(data, data_errors, ws=[30,50])

        # Estimate the first guess values
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
        ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)
        >>> s.plot_summary(truth=s0)

        # Fit the data:
        >>> s.fit_chromatic_PSF1D_minuit(data, bgd_model_func=bgd_model_func, data_errors=data_errors)
        >>> s.plot_summary(truth=s0)
        """
        Ny, Nx = data.shape
        if Ny != self.Ny:
            self.my_logger.error(f"\n\tData y shape {Ny} different from ChromaticPSF1D input self.Ny {self.Ny}.")
        if Nx != self.Nx:
            self.my_logger.error(f"\n\tData x shape {Nx} different from ChromaticPSF1D input self.Nx {self.Nx}.")
        guess = np.copy(self.poly_params)
        pixels = np.arange(Ny)

        bgd = np.zeros_like(data)
        if bgd_model_func is not None:
            # xx, yy = np.meshgrid(np.arange(Nx), pixels)
            bgd = bgd_model_func(np.arange(Nx), pixels)

        data_subtracted = data - bgd
        bgd_std = float(np.std(np.random.poisson(bgd)))

        # crop spectrogram to fit faster
        bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        data_subtracted = data_subtracted[bgd_width:-bgd_width, :]
        pixels = np.arange(data_subtracted.shape[0])
        data_errors_cropped = np.copy(data_errors[bgd_width:-bgd_width, :])

        # error matrix
        W = 1. / (data_errors_cropped * data_errors_cropped)
        W = [np.diag(W[:, x]) for x in range(Nx)]
        W_dot_data = [W[x] @ data_subtracted[:, x] for x in range(Nx)]
        poly_params = np.copy(guess)

        def spectrogram_chisq(shape_params):
            # linear regression for the amplitude parameters
            poly_params[Nx:] = np.copy(shape_params)
            profile_params = self.from_poly_params_to_profile_params(poly_params, force_positive=True)
            profile_params[:Nx, 0] = 1
            profile_params[:Nx, 1] -= bgd_width
            J = np.array([self.PSF.evaluate(pixels, *profile_params[x, :]) for x in range(Nx)])
            J_dot_W_dot_J = np.array([J[x].T @ W[x] @ J[x] for x in range(Nx)])
            amplitude_params = [
                J[x].T @ W_dot_data[x] / (J_dot_W_dot_J[x]) if J_dot_W_dot_J[x] > 0 else 0.1 * bgd_std
                for x in
                range(Nx)]
            poly_params[:Nx] = amplitude_params
            in_bounds, penalty, name = self.check_bounds(poly_params, noise_level=bgd_std)
            mod = self.evaluate(poly_params)[bgd_width:-bgd_width, :]
            self.poly_params = np.copy(poly_params)
            if data_errors is None:
                return np.nansum((mod - data_subtracted) ** 2) + penalty
            else:
                return np.nansum(((mod - data_subtracted) / data_errors_cropped) ** 2) + penalty

        self.my_logger.debug(f'\n\tStart chisq: {spectrogram_chisq(guess[Nx:])} with {guess[Nx:]}')
        error = 0.01 * np.abs(guess) * np.ones_like(guess)
        fix = [False] * (self.n_poly_params - Nx)
        fix[-1] = True
        bounds = self.set_bounds(data)
        # fix[:Nx] = [True] * Nx
        # noinspection PyArgumentList
        m = Minuit.from_array_func(fcn=spectrogram_chisq, start=guess[Nx:], error=error[Nx:], errordef=1,
                                   fix=fix, print_level=parameters.DEBUG, limit=bounds[Nx:])
        m.migrad()
        # m.hesse()
        # print(m.np_matrix())
        # print(m.np_matrix(correlation=True))
        self.poly_params[Nx:] = m.np_values()
        self.profile_params = self.from_poly_params_to_profile_params(self.poly_params,
                                                                      force_positive=True)
        self.fill_table_with_profile_params(self.profile_params)
        self.from_profile_params_to_shape_params(self.profile_params)
        if parameters.DEBUG:
            # Plot data, best fit model and residuals:
            self.plot_summary()
            self.plot_chromatic_PSF1D_residuals(bgd, data, data_errors, guess=guess, title='Best fit')

    def fit_chromatic_PSF1D(self, data, bgd_model_func=None, data_errors=None):
        """
        Fit a chromatic PSF model on 2D data.

        Parameters
        ----------
        data: array_like
            2D array containing the image data.
        bgd_model_func: callable, optional
            A 2D function to model the extracted background (default: None -> null background)
        data_errors: np.array
            the 2D array uncertainties.

        Examples
        --------

        # Set the parameters
        >>> parameters.PIXDIST_BACKGROUND = 40
        >>> parameters.PIXWIDTH_BACKGROUND = 10
        >>> parameters.PIXWIDTH_SIGNAL = 30
        >>> parameters.VERBOSE = True

        # Build a mock spectrogram with random Poisson noise:
        >>> s0 = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=1000)
        >>> params = s0.generate_test_poly_params()
        >>> s0.poly_params = params
        >>> saturation = params[-1]
        >>> data = s0.evaluate(params)
        >>> bgd = 10*np.ones_like(data)
        >>> data += bgd
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        # Extract the background
        >>> bgd_model_func = extract_background_photutils(data, data_errors, ws=[30,50])

        # Estimate the first guess values
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
        ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)
        >>> s.plot_summary(truth=s0)

        # Fit the data:
        >>> s.fit_chromatic_PSF1D(data, bgd_model_func=bgd_model_func, data_errors=data_errors)
        >>> s.plot_summary(truth=s0)
        """
        w = ChromaticPSF1DFitWorkspace(self, data, data_errors, bgd_model_func=bgd_model_func)

        guess = np.copy(self.poly_params)
        run_minimisation(w, method="newton", ftol=1e-4, xtol=1e-4)

        self.poly_params = w.poly_params
        self.profile_params = self.from_poly_params_to_profile_params(self.poly_params, force_positive=True)
        self.fill_table_with_profile_params(self.profile_params)
        self.from_profile_params_to_shape_params(self.profile_params)
        if parameters.DEBUG or True:
            # Plot data, best fit model and residuals:
            self.plot_summary()
            self.plot_chromatic_PSF1D_residuals(w.bgd, data, data_errors, guess=guess, title='Best fit')


class ChromaticPSF1DFitWorkspace(FitWorkspace):

    def __init__(self, chromatic_psf, data, data_errors, bgd_model_func=None, file_name="",
                 nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        FitWorkspace.__init__(self, file_name, nwalkers, nsteps, burnin, nbins, verbose, plot,
                              live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        self.chromatic_psf = chromatic_psf
        self.data = data
        self.err = data_errors
        self.bgd_model_func = bgd_model_func
        length = len(self.chromatic_psf.table)
        self.p = np.copy(self.chromatic_psf.poly_params[length:-1])  # remove saturation (fixed parameter))
        self.ndim = self.p.size
        self.poly_params = np.copy(self.chromatic_psf.poly_params)
        self.input_labels = list(np.copy(self.chromatic_psf.poly_params_labels[length:-1]))
        self.axis_names = list(np.copy(self.chromatic_psf.poly_params_names[length:-1]))
        self.bounds = self.chromatic_psf.set_bounds(data=None)[:-1]
        self.nwalkers = max(2 * self.ndim, nwalkers)

        # prepare the fit
        self.Ny, self.Nx = self.data.shape
        if self.Ny != self.chromatic_psf.Ny:
            self.my_logger.error(f"\n\tData y shape {self.Ny} different from ChromaticPSF input Ny {self.chromatic_psf.Ny}.")
        if self.Nx != self.chromatic_psf.Nx:
            self.my_logger.error(f"\n\tData x shape {self.Nx} different from ChromaticPSF input Nx {self.chromatic_psf.Nx}.")
        self.pixels = np.arange(self.Ny)

        # prepare the background, data and errors
        self.bgd = np.zeros_like(self.data)
        if self.bgd_model_func is not None:
            # xx, yy = np.meshgrid(np.arange(Nx), pixels)
            self.bgd = self.bgd_model_func(np.arange(self.Nx), self.pixels)
        self.data = self.data - self.bgd
        self.bgd_std = float(np.std(np.random.poisson(self.bgd)))

        # crop spectrogram to fit faster
        self.bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        self.data = self.data[self.bgd_width:-self.bgd_width, :]
        self.pixels = np.arange(self.data.shape[0])
        self.err = np.copy(self.err[self.bgd_width:-self.bgd_width, :])

        # error matrix
        self.W = 1. / (self.err * self.err)
        self.W = [np.diag(self.W[:, x]) for x in range(self.Nx)]
        self.W_dot_data = [self.W[x] @ self.data[:, x] for x in range(self.Nx)]

    def simulate(self, *shape_params):
        """
        Compute a ChromaticPSF model given PSF shape parameters and minimizing
        amplitude parameters given a spectrogram data array.

        Parameters
        ----------
        shape_params: array_like
            PSF shape polynomial parameter array.

        Examples
        --------

        # Set the parameters
        >>> parameters.PIXDIST_BACKGROUND = 40
        >>> parameters.PIXWIDTH_BACKGROUND = 10
        >>> parameters.PIXWIDTH_SIGNAL = 30

        # Build a mock spectrogram with random Poisson noise:
        >>> s0 = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=1000)
        >>> params = s0.generate_test_poly_params()
        >>> s0.poly_params = params
        >>> saturation = params[-1]
        >>> data = s0.evaluate(params)
        >>> bgd = 10*np.ones_like(data)
        >>> data += bgd
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        # Extract the background
        >>> bgd_model_func = extract_background_photutils(data, data_errors, ws=[30,50])

        # Estimate the first guess values
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
        ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)

        # Simulate the data:
        >>> w = ChromaticPSF1DFitWorkspace(s, data, data_errors, bgd_model_func=bgd_model_func, verbose=True)
        >>> y, mod, mod_err = w.simulate(s.poly_params[s.Nx:-1])
        >>> assert mod is not None
        >>> w.plot_fit()
        """
        # linear regression for the amplitude parameters
        poly_params = np.copy(self.chromatic_psf.poly_params)
        poly_params[self.Nx:-1] = np.copy(shape_params)
        profile_params = self.chromatic_psf.from_poly_params_to_profile_params(poly_params, force_positive=True)
        profile_params[:self.Nx, 0] = 1
        profile_params[:self.Nx, 1] -= self.bgd_width
        J = np.array([self.chromatic_psf.PSF.evaluate(self.pixels, *profile_params[x, :]) for x in range(self.Nx)])
        J_dot_W_dot_J = np.array([J[x].T @ self.W[x] @ J[x] for x in range(self.Nx)])
        amplitude_params = [
            J[x].T @ self.W_dot_data[x] / (J_dot_W_dot_J[x]) if J_dot_W_dot_J[x] > 0 else 0.1 * self.bgd_std
            for x in range(self.Nx)]
        poly_params[:self.Nx] = amplitude_params
        # in_bounds, penalty, name = self.chromatic_psf.check_bounds(poly_params, noise_level=self.bgd_std)
        self.model = self.chromatic_psf.evaluate(poly_params)[self.bgd_width:-self.bgd_width, :]
        self.model_err = np.zeros_like(self.model)
        self.poly_params = np.copy(poly_params)
        return self.pixels, self.model, self.model_err

    def plot_fit(self):
        gs_kw = dict(width_ratios=[3, 0.15], height_ratios=[1, 1, 1, 1])
        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7, 7), constrained_layout=True, gridspec_kw=gs_kw)
        norm = np.max(self.data)
        plot_image_simple(ax[0, 0], data=self.model / norm, aspect='auto', cax=ax[0, 1], vmin=0, vmax=1,
                          units='1/max(data)')
        ax[0, 0].set_title("Model", fontsize=10, loc='center', color='white', y=0.8)
        plot_image_simple(ax[1, 0], data=self.data / norm, title='Data', aspect='auto',
                          cax=ax[1, 1], vmin=0, vmax=1, units='1/max(data)')
        ax[1, 0].set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
        residuals = (self.data - self.model)
        # residuals_err = self.spectrum.spectrogram_err / self.model
        norm = self.err
        residuals /= norm
        std = float(np.std(residuals))
        plot_image_simple(ax[2, 0], data=residuals, vmin=-5 * std, vmax=5 * std, title='(Data-Model)/Err',
                          aspect='auto', cax=ax[2, 1], units='', cmap="bwr")
        ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, loc='center', color='black', y=0.8)
        ax[2, 0].text(0.05, 0.05, f'mean={np.mean(residuals):.3f}\nstd={np.std(residuals):.3f}',
                      horizontalalignment='left', verticalalignment='bottom',
                      color='black', transform=ax[2, 0].transAxes)
        ax[0, 0].set_xticks(ax[2, 0].get_xticks()[1:-1])
        ax[0, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[1, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[2, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[3, 1].remove()
        ax[3, 0].plot(np.arange(self.Nx), self.data.sum(axis=0), label='Data')
        ax[3, 0].plot(np.arange(self.Nx), self.model.sum(axis=0), label='Model')
        ax[3, 0].set_ylabel('Transverse sum')
        ax[3, 0].set_xlabel(r'X [pixels]')
        ax[3, 0].legend(fontsize=7)
        ax[3, 0].grid(True)
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            if parameters.DISPLAY and self.verbose:
                plt.show()
        if parameters.SAVE:  # pragma: no cover
            figname = self.filename.replace(self.filename.split('.')[-1], "_bestfit.pdf")
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')


class ChromaticPSF2D(ChromaticPSF):

    def __init__(self, Nx, Ny, deg=4, saturation=None, file_name=''):
        PSF = PSF2D()
        ChromaticPSF.__init__(self, PSF, Nx=Nx, Ny=Ny, deg=deg, saturation=saturation, file_name=file_name)
        self.my_logger = set_logger(self.__class__.__name__)

    def generate_test_poly_params(self):
        """
        A set of parameters to define a test spectrogram

        Parameters
        ----------

        Returns
        -------
        profile_params: array
            The list of the test parameters

        Examples
        --------
        >>> s = ChromaticPSF2D(Nx=5, Ny=4, deg=1, saturation=20000)
        >>> params = s.generate_test_poly_params()
        >>> assert(np.all(np.isclose(params, [ 0, 50, 100, 150, 200, 0, 1, 2, 0, 2, 0, 2, 0, -0.4, -0.4, 1, 0, 20000])))
        """
        params = [50 * i for i in range(self.Nx)]
        params += [0.] * (self.degrees['x_mean'] - 1) + [1, 0]  # y mean
        params += [0.] * (self.degrees['y_mean'] - 1) + [0, self.Ny / 2]  # y mean
        params += [0.] * (self.degrees['gamma'] - 1) + [0, 2]  # gamma
        params += [0.] * (self.degrees['alpha'] - 1) + [0, 2]  # alpha
        params += [0.] * (self.degrees['eta_gauss'] - 1) + [-0.4, -0.4]  # eta_gauss
        params += [0.] * (self.degrees['stddev'] - 1) + [0, 1]  # stddev
        params += [self.saturation]  # saturation
        poly_params = np.zeros_like(params)
        poly_params[:self.Nx] = params[:self.Nx]
        index = self.Nx - 1
        for name in self.PSF.param_names:
            if name == 'amplitude_moffat':
                continue
            else:
                shift = self.degrees[name] + 1
                c = np.polynomial.legendre.poly2leg(params[index + shift:index:-1])
                coeffs = np.zeros(shift)
                coeffs[:c.size] = c
                poly_params[index + 1:index + shift + 1] = coeffs
                index = index + shift
        return poly_params

    def evaluate(self, poly_params):
        """
        Simulate a 2D spectrogram of size Nx times Ny.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
                * Nx first parameters are amplitudes for the Moffat transverse profiles
                * next parameters are polynomial coefficients for all the PSF parameters
                in the same order as in PSF definition, except amplitude_moffat

        Returns
        -------
        output: array
            A 2D array with the model

        Examples
        --------
        >>> s = ChromaticPSF2D(Nx=100, Ny=20, deg=4, saturation=20000)
        >>> poly_params = s.generate_test_poly_params()
        >>> output = s.evaluate(poly_params)

        >>> import matplotlib.pyplot as plt
        >>> im = plt.imshow(output, origin='lower')  #doctest: +ELLIPSIS
        >>> plt.colorbar(im)  #doctest: +ELLIPSIS
        <matplotlib.colorbar.Colorbar object at 0x...>
        >>> if parameters.DISPLAY: plt.show()

        """
        profile_params = self.from_poly_params_to_profile_params(poly_params)
        # replace x_mean
        profile_params[:, 1] = np.arange(self.Nx)
        yy, xx = np.mgrid[:self.Ny, :self.Nx]
        output = np.zeros((self.Ny, self.Nx))
        for k in range(self.Nx):
            output += PSF2D.evaluate(xx, yy, *profile_params[k,])
        return output

    def fit_chromatic_PSF2D(self, data, bgd_model_func=None, data_errors=None):
        """
        Fit a chromatic PSF2D model on 2D data.

        Parameters
        ----------
        data: array_like
            2D array containing the image data.
        bgd_model_func: callable, optional
            A 2D function to model the extracted background (default: None -> null background)
        data_errors: np.array
            the 2D array uncertainties.

        Examples
        --------

        # Set the parameters
        >>> parameters.PIXDIST_BACKGROUND = 40
        >>> parameters.PIXWIDTH_BACKGROUND = 10
        >>> parameters.PIXWIDTH_SIGNAL = 30

        # Build a mock spectrogram with random Poisson noise:
        >>> s0 = ChromaticPSF2D(Nx=100, Ny=100, deg=4, saturation=20000)
        >>> params = s0.generate_test_poly_params()
        >>> s0.poly_params = params
        >>> saturation = params[-1]
        >>> data = s0.evaluate(params)
        >>> bgd = 10*np.ones_like(data)
        >>> data += bgd
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        # Extract the background
        >>> bgd_model_func = extract_background_photutils(data, data_errors, ws=[30,50])

        # Estimate the first guess values
        # >>> s = ChromaticPSF2D(Nx=100, Ny=100, deg=4, saturation=saturation)
        # >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
        # ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)
        # >>> s.plot_summary(truth=s0)

        # Fit the data:
        # >>> parameters.DEBUG = True
        # >>> s.fit_chromatic_PSF2D(data, bgd_model_func=bgd_model_func, data_errors=data_errors)
        # >>> s.plot_summary(truth=s0)
        """
        my_logger = set_logger(__name__)
        Ny, Nx = data.shape
        if Ny != self.Ny:
            my_logger.error(f"\n\tData y shape {Ny} different from ChromaticPSF1D input self.Ny {self.Ny}.")
        if Nx != self.Nx:
            my_logger.error(f"\n\tData x shape {Nx} different from ChromaticPSF1D input self.Nx {self.Nx}.")
        guess = np.copy(self.poly_params)
        pixels = np.arange(Ny)

        bgd = np.zeros_like(data)
        if bgd_model_func is not None:
            # xx, yy = np.meshgrid(np.arange(Nx), pixels)
            bgd = bgd_model_func(np.arange(Nx), pixels)

        data_subtracted = data - bgd
        bgd_std = float(np.std(np.random.poisson(bgd)))

        # crop spectrogram to fit faster
        bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        data_subtracted = data_subtracted[bgd_width:-bgd_width, :]
        Ny = data_subtracted.shape[0]
        yy, xx = np.mgrid[:Ny,:Nx]
        data_errors_cropped = np.copy(data_errors[bgd_width:-bgd_width, :])

        # error matrix
        W = 1. / (data_errors_cropped * data_errors_cropped)
        # W = [np.diag(W[:, x]) for x in range(Nx)]
        W = W.flatten()
        W_dot_data = np.diag(W) @ data_subtracted.flatten()
        poly_params = np.copy(guess)

        W_dot_J = np.zeros((Ny * Nx, Nx))
        J = np.zeros((Ny * Nx, Nx))

        def spectrogram_chisq(shape_params):
            # linear regression for the amplitude parameters
            poly_params[Nx:] = np.copy(shape_params)
            profile_params = self.from_poly_params_to_profile_params(poly_params, force_positive=True)
            profile_params[:Nx, 0] = 1
            profile_params[:Nx, 1] = np.arange(Nx)
            profile_params[:Nx, 2] -= bgd_width
            for x in range(Nx):
                # self.my_logger.warning(f'\n\t{x} {profile_params[x, :]}')
                J[:, x] = self.PSF.evaluate(xx, yy, *profile_params[x, :]).flatten()
                W_dot_J[:, x] = J[:, x] * W
            J_dot_W_dot_J = J.T @ W_dot_J
            #self.my_logger.warning(f'\n\tJWJ {J_dot_W_dot_J}')
            L = np.linalg.inv(np.linalg.cholesky(J_dot_W_dot_J))
            inv_J_dot_W_dot_J = L.T @ L # np.linalg.inv(J_dot_W_dot_J)
            amplitude_params = inv_J_dot_W_dot_J @ (J.T @ W_dot_data)
            amplitude_params[np.diagonal(J_dot_W_dot_J) <= 0] = 0.1 * bgd_std
            amplitude_params[amplitude_params < 0.1 * bgd_std] = 0.1 * bgd_std
            poly_params[:Nx] = amplitude_params
            in_bounds, penalty, name = self.check_bounds(poly_params, noise_level=bgd_std)
            mod = self.evaluate(poly_params)[bgd_width:-bgd_width, :]
            if not in_bounds:
                self.my_logger.warning(f'{in_bounds} {penalty} {name}')
            self.poly_params = np.copy(poly_params)
            if data_errors is None:
                return np.nansum((mod - data_subtracted) ** 2) + penalty
            else:
                return np.nansum(((mod - data_subtracted) / data_errors_cropped) ** 2) + penalty

        my_logger.debug(f'\n\tStart chisq: {spectrogram_chisq(guess[Nx:])} with {guess[Nx:]}')
        error = 0.01 * np.abs(guess) * np.ones_like(guess)
        fix = [False] * (self.n_poly_params - Nx)
        fix[Nx+self.degrees['amplitude_moffat']:Nx+self.degrees['amplitude_moffat']+self.degrees['x_mean']] = [True] * self.degrees['x_mean']
        fix[-1] = True
        bounds = self.set_bounds(data)
        # noinspection PyArgumentList
        m = Minuit.from_array_func(fcn=spectrogram_chisq, start=guess[Nx:], error=error[Nx:], errordef=1,
                                   fix=fix, print_level=2, limit=bounds[Nx:])
        m.migrad()
        self.poly_params[Nx:] = m.np_values()
        self.profile_params = self.from_poly_params_to_profile_params(self.poly_params,
                                                                      force_positive=True)
        self.fill_table_with_profile_params(self.profile_params)
        self.from_profile_params_to_shape_params(self.profile_params)
        if parameters.DEBUG:
            # Plot data, best fit model and residuals:
            self.plot_summary()
            self.plot_chromatic_PSF1D_residuals(bgd, data, data_errors, guess=guess, title='Best fit')


def plot_transverse_PSF1D_profile(x, indices, bgd_indices, data, err, fit=None, bgd_model_func=None, params=None,
                                  PSF_guess=None, outliers=[], sigma=3, live_fit=False):  # pragma: no cover
    """Plot the transverse profile of  the spectrogram.

    This plot function is called in transverse_PSF1D_profile if live_fit option is True.

    Parameters
    ----------
    x: int
        Pixel index along the dispersion axis.
    indices: array_like
        Pixel indices across the dispersion axis.
    bgd_indices: array_like
        Pixel indices across the dispersion axis for the background estimate.
    data: array_like
        The 2D spectrogram data array.
    err: array_like
        The 2D spectrogram uncertainty data array.
    fit: Fittable1DModel, optional
        Best fitting model function for the profile (default: None).
    bgd_model_func: callable, optional
        A 2D function to model the extracted background (default: None -> null background)
    params: array_like, optional
        Best fitting model parameter array (default: None).
    PSF_guess: callable, optional
        Guessed fitting model function for the profile before the fit (default: None).
    outliers: array_like, optional
        Pixel indices of the outliers across the dispersion axis (default: None).
    sigma: int, optional
        Value of the sigma-clipping rejection (default: 3).
        Necessary only if an outlier array is given with the outliers keyword.
    live_fit: bool
        If True, plot is shown  in live during the fitting procedure (default: False).

    Examples
    --------

    # Build a mock spectrogram with random Poisson noise:
    >>> s0 = ChromaticPSF1D(Nx=80, Ny=100, saturation=1000)
    >>> params = s0.generate_test_poly_params()
    >>> saturation = params[-1]
    >>> data = s0.evaluate(params)
    >>> bgd = 10*np.ones_like(data)
    >>> data += bgd
    >>> data = np.random.poisson(data)
    >>> data_errors = np.sqrt(data+1)

    # Extract the background
    >>> bgd_model_func = extract_background_photutils(data, data_errors, ws=[30,50])

    # Fit the transverse profile:
    >>> s = ChromaticPSF1D(Nx=80, Ny=100, deg=4, saturation=saturation)
    >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50], pixel_step=10,
    ... bgd_model_func=bgd_model_func, saturation=saturation, live_fit=True, sigma=5)

    """
    Ny = len(indices)
    y = data[:, x]
    bgd = data[bgd_indices, x]
    bgd_err = err[bgd_indices, x]
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex='all', gridspec_kw={'height_ratios': [5, 1]})
    ax[0].errorbar(np.arange(Ny), y, yerr=err[:, x], fmt='ro', label="original data")
    ax[0].errorbar(bgd_indices, bgd, yerr=bgd_err, fmt='bo', label="bgd data")
    if len(outliers) > 0:
        ax[0].errorbar(outliers, data[outliers, x], yerr=err[outliers, x], fmt='go',
                       label=f"outliers ({sigma}$\sigma$)")
    if bgd_model_func is not None:
        ax[0].plot(bgd_indices, bgd_model_func(x, bgd_indices)[:, 0], 'b--', label="fitted bgd")
    if PSF_guess is not None:
        if bgd_model_func is not None:
            ax[0].plot(indices, PSF_guess(indices) + bgd_model_func(x, indices)[:, 0], 'k--', label="guessed profile")
        else:
            ax[0].plot(indices, PSF_guess(indices), 'k--', label="guessed profile")
    if fit is not None and bgd_model_func is not None:
        model = fit(indices) + bgd_model_func(x, indices)[:, 0]
        ax[0].plot(indices, model, 'b-', label="fitted profile")
    ylim = ax[0].get_ylim()
    if params is not None:
        PSF_moffat = Moffat1D(*params[:4])
        ax[0].plot(indices, PSF_moffat(indices) + bgd_model_func(x, indices)[:, 0], 'b+', label="fitted moffat")
    ax[0].set_ylim(ylim)
    ax[0].set_ylabel('Transverse profile')
    ax[0].legend(loc=2, numpoints=1)
    ax[0].grid(True)
    if fit is not None:
        txt = ""
        for ip, p in enumerate(fit.param_names):
            txt += f'{p}: {getattr(fit, p).value:.4g}\n'
        ax[0].text(0.95, 0.95, txt, horizontalalignment='right', verticalalignment='top', transform=ax[0].transAxes)
        ax[0].set_title(f'x={x}')
    model = np.zeros_like(indices).astype(float)
    model_outliers = np.zeros_like(outliers).astype(float)
    if fit is not None:
        model += fit(indices)
        model_outliers += fit(outliers)
    if bgd_model_func is not None:
        model += bgd_model_func(x, indices)[:, 0]
        if len(outliers) > 0:
            model_outliers += bgd_model_func(x, outliers)[:, 0]
    if fit is not None or bgd_model_func is not None:
        residuals = (y - model) / err[:, x]  # / model
        residuals_err = err[:, x] / err[:, x]  # / model
        ax[1].errorbar(indices, residuals, yerr=residuals_err, fmt='ro')
        if len(outliers) > 0:
            residuals_outliers = (data[outliers, x] - model_outliers) / err[outliers, x]  # / model_outliers
            residuals_outliers_err = err[outliers, x] / err[outliers, x]  # / model_outliers
            ax[1].errorbar(outliers, residuals_outliers, yerr=residuals_outliers_err, fmt='go')
        ax[1].axhline(0, color='b')
        ax[1].grid(True)
        std = np.std(residuals)
        ax[1].set_ylim([-3. * std, 3. * std])
        ax[1].set_xlabel(ax[0].get_xlabel())
        ax[1].set_ylabel('(data-fit)/err')
        ax[0].set_xticks(ax[1].get_xticks()[1:-1])
        ax[0].get_yaxis().set_label_coords(-0.1, 0.5)
        ax[1].get_yaxis().set_label_coords(-0.1, 0.5)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    if parameters.DISPLAY:
        if live_fit:
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            plt.show()
    plt.close()


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


# DO NOT WORK
# def fit_PSF2D_outlier_removal(x, y, data, sigma=3.0, niter=3, guess=None, bounds=None):
#     """Fit a PSF 2D model with parameters:
#         amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation
#     using scipy. Find outliers data point above sigma*data_errors from the fit over niter iterations.
#
#     Parameters
#     ----------
#     x: np.array
#         2D array of the x coordinates.
#     y: np.array
#         2D array of the y coordinates.
#     data: np.array
#         the 1D array profile.
#     guess: array_like, optional
#         list containing a first guess for the PSF parameters (default: None).
#     bounds: list, optional
#         2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
#     sigma: int
#         the sigma limit to exclude data points (default: 3).
#     niter: int
#         the number of loop iterations to exclude  outliers and refit the model (default: 2).
#
#     Returns
#     -------
#     fitted_model: PSF2D
#         the PSF2D fitted model.
#
#     Examples
#     --------
#
#     Create the model:
#     >>> X, Y = np.mgrid[:50,:50]
#     >>> PSF = PSF2D()
#     >>> p = (1000, 25, 25, 5, 1, -0.2, 1, 6000)
#     >>> Z = PSF.evaluate(X, Y, *p)
#     >>> Z += 100*np.exp(-((X-10)**2+(Y-10)**2)/4)
#     >>> Z_err = np.sqrt(1+Z)
#
#     Prepare the fit:
#     >>> guess = (1000, 27, 23, 3.2, 1.2, -0.1, 2,  6000)
#     >>> bounds = np.array(((0, 6000), (10, 40), (10, 40), (0.5, 10), (0.5, 5), (-1, 0), (0.01, 10), (0, 8000)))
#     >>> bounds = bounds.T
#
#     Fit without bars:
#     >>> model = fit_PSF2D_outlier_removal(X, Y, Z, guess=guess, bounds=bounds, sigma=7, niter=5)
#     >>> res = [getattr(model, p).value for p in model.param_names]
#     >>> print(res, p)
#     >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))
#     """
#     gg_init = PSF2D()
#     if guess is not None:
#         for ip, p in enumerate(gg_init.param_names):
#             getattr(gg_init, p).value = guess[ip]
#     if bounds is not None:
#         for ip, p in enumerate(gg_init.param_names):
#             getattr(gg_init, p).min = bounds[0][ip]
#             getattr(gg_init, p).max = bounds[1][ip]
#     gg_init.saturation.fixed = True
#     with warnings.catch_warnings():
#         # Ignore model linearity warning from the fitter
#         warnings.simplefilter('ignore')
#         fit = LevMarLSQFitterWithNan()
#         or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
#         # get fitted model and filtered data
#         or_fitted_model, filtered_data = or_fit(gg_init, x, y, data)
#         if parameters.VERBOSE:
#             print(or_fitted_model)
#         if parameters.DEBUG:
#             print(fit.fit_info)
#         print(fit.fit_info)
#         return or_fitted_model


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
    method: str, optional
        the minimisation method: 'minimize' or 'basinhopping' (default: 'minimize').

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


def fit_PSF2D_minuit(x, y, data, guess=None, bounds=None, data_errors=None, fix=None):
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
    fix: array_like, optional
        A list of boolean to keep fix some parameters, in the same order as the list of parameters (default: None)

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
    >>> model = fit_PSF2D_minuit(X, Y, Z, guess=guess, bounds=bounds, data_errors=Z_err)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit without error bars:
    >>> model = fit_PSF2D_minuit(X, Y, Z, guess=guess, bounds=bounds, data_errors=None)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))
    """

    model = PSF2D()
    my_logger = set_logger(__name__)

    if bounds is not None:
        bounds = np.array(bounds)
        if bounds.shape[0] == 2 and bounds.shape[1] > 2:
            bounds = bounds.T

    guess = np.array(guess)
    error = 0.001 * np.abs(guess) * np.ones_like(guess)
    z = np.where(np.isclose(error, 0.0, 1e-6))
    error[z] = 0.001

    def chisq_PSF2D(params):
        return PSF2D_chisq(params, model, x, y, data, data_errors)

    def chisq_PSF2D_jac(params):
        return PSF2D_chisq_jac(params, model, x, y, data, data_errors)

    if fix is None:
        fix = [False] * error.size
    fix[-1] = True
    # noinspection PyArgumentList
    m = Minuit.from_array_func(fcn=chisq_PSF2D, start=guess, error=error, errordef=1,
                               fix=fix, print_level=0, limit=bounds, grad=chisq_PSF2D_jac)

    m.tol = 0.001
    m.migrad()
    popt = m.np_values()

    PSF = PSF2D(*popt)
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
        the PSF fitted model.

    Examples
    --------

    Create the model:
    >>> import numpy as np
    >>> X = np.arange(0, 50)
    >>> psf = PSF1D()
    >>> p = (50, 25, 5, 1, -0.2, 1, 60)
    >>> Y = psf.evaluate(X, *p)
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


def fit_PSF1D_outlier_removal(x, data, data_errors=None, sigma=3.0, niter=3, guess=None, bounds=None, method='minimize',
                              niter_basinhopping=5, T_basinhopping=0.2):
    """Fit a PSF 1D model with parameters:
        amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation
    using scipy. Find outliers data point above sigma*data_errors from the fit over niter iterations.

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates.
    data: np.array
        the 1D array profile.
    data_errors: np.array
        the 1D array uncertainties.
    guess: array_like, optional
        list containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    sigma: int
        the sigma limit to exclude data points (default: 3).
    niter: int
        the number of loop iterations to exclude  outliers and refit the model (default: 2).
    method: str
        Can be 'minimize' or 'basinhopping' (default: 'minimize').
    niter_basinhopping: int, optional
        The number of basin hops (default: 5)
    T_basinhopping: float, optional
        The temperature for the basin hops (default: 0.2)

    Returns
    -------
    fitted_model: PSF1D
        the PSF fitted model.
    outliers: list
        the list of the outlier indices.

    Examples
    --------

    Create the model:
    >>> import numpy as np
    >>> X = np.arange(0, 50)
    >>> PSF = PSF1D()
    >>> p = (1000, 25, 5, 1, -0.2, 1, 6000)
    >>> Y = PSF.evaluate(X, *p)
    >>> Y += 100*np.exp(-((X-10)/2)**2)
    >>> Y_err = np.sqrt(1+Y)

    Prepare the fit:
    >>> guess = (600, 27, 3.2, 1.2, -0.1, 2,  6000)
    >>> bounds = ((0, 6000), (10, 40), (0.5, 10), (0.5, 5), (-1, 0), (0.01, 10), (0, 8000))

    Fit without bars:
    >>> model, outliers = fit_PSF1D_outlier_removal(X, Y, guess=guess, bounds=bounds,
    ... sigma=3, niter=5, method="minimize")
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))

    Fit with error bars:
    >>> model, outliers = fit_PSF1D_outlier_removal(X, Y, guess=guess, bounds=bounds, data_errors=Y_err,
    ... sigma=3, niter=2, method="minimize")
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))

    Fit with error bars and basinhopping:
    >>> model, outliers = fit_PSF1D_outlier_removal(X, Y, guess=guess, bounds=bounds, data_errors=Y_err,
    ... sigma=3, niter=5, method="basinhopping", niter_basinhopping=20)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))
    """

    my_logger = set_logger(__name__)
    indices = np.mgrid[:x.shape[0]]
    outliers = np.array([])
    model = PSF1D()

    for step in range(niter):
        # first fit
        if data_errors is None:
            err = None
        else:
            err = data_errors[indices]
        if method == 'minimize':
            res = minimize(PSF1D_chisq, guess, method="L-BFGS-B", bounds=bounds, jac=PSF1D_chisq_jac,
                           args=(model, x[indices], data[indices], err))
        elif method == 'basinhopping':
            minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, jac=PSF1D_chisq_jac,
                                    args=(model, x[indices], data[indices], err))
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
        indices_no_nan = ~np.isnan(data)
        diff = model(x[indices_no_nan]) - data[indices_no_nan]
        if data_errors is not None:
            outliers = np.where(np.abs(diff) / data_errors[indices_no_nan] > sigma)[0]
        else:
            std = np.std(diff)
            outliers = np.where(np.abs(diff) / std > sigma)[0]
        if len(outliers) > 0:
            indices = [i for i in range(x.shape[0]) if i not in outliers]
        else:
            break
    my_logger.debug(f'\n\tPSF best fitting parameters:\n{model}')
    return model, outliers


def fit_PSF1D_minuit(x, data, guess=None, bounds=None, data_errors=None):
    """Fit a PSF 1D model with parameters:
        amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation
    using Minuit.

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

    Returns
    -------
    fitted_model: PSF1D
        the PSF fitted model.

    Examples
    --------

    Create the model:
    >>> import numpy as np
    >>> X = np.arange(0, 50)
    >>> PSF = PSF1D()
    >>> p = (50, 25, 5, 1, -0.2, 1, 60)
    >>> Y = PSF.evaluate(X, *p)
    >>> Y_err = np.sqrt(1+Y)

    Prepare the fit:
    >>> guess = (60, 20, 3.2, 1.2, -0.1, 2,  60)
    >>> bounds = ((0, 200), (10, 40), (0.5, 10), (0.5, 5), (-1, 0), (0.01, 10), (0, 400))

    Fit with error bars:
    >>> model = fit_PSF1D_minuit(X, Y, guess=guess, bounds=bounds, data_errors=Y_err)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-2))

    Fit without error bars:
    >>> model = fit_PSF1D_minuit(X, Y, guess=guess, bounds=bounds, data_errors=None)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-2))

    """

    my_logger = set_logger(__name__)
    model = PSF1D()

    def PSF1D_chisq_v2(params):
        mod = model.evaluate(x, *params)
        diff = mod - data
        if data_errors is None:
            return np.nansum(diff * diff)
        else:
            return np.nansum((diff / data_errors) ** 2)

    def PSF1D_chisq_v2_jac(params):
        diff = model.evaluate(x, *params) - data
        jac = model.fit_deriv(x, *params)
        if data_errors is None:
            return np.array([np.nansum(2 * jac[p] * diff) for p in range(len(params))])
        else:
            yy_err2 = data_errors * data_errors
            return np.array([np.nansum(2 * jac[p] * diff / yy_err2) for p in range(len(params))])

    error = 0.1 * np.abs(guess) * np.ones_like(guess)
    fix = [False] * len(guess)
    fix[-1] = True
    # noinspection PyArgumentList
    # 3 times faster with gradient
    m = Minuit.from_array_func(fcn=PSF1D_chisq_v2, start=guess, error=error, errordef=1, limit=bounds, fix=fix,
                               print_level=parameters.DEBUG, grad=PSF1D_chisq_v2_jac)
    m.migrad()
    PSF = PSF1D(*m.np_values())

    my_logger.debug(f'\n\tPSF best fitting parameters:\n{PSF}')
    return PSF


def fit_PSF1D_minuit_outlier_removal(x, data, data_errors, guess=None, bounds=None, sigma=3, niter=2, consecutive=3):
    """Fit a PSF 1D model with parameters:
        amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation
    using Minuit. Find outliers data point above sigma*data_errors from the fit over niter iterations.
    Only at least 3 consecutive outliers are considered.

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates.
    data: np.array
        the 1D array profile.
    data_errors: np.array
        the 1D array uncertainties.
    guess: array_like, optional
        list containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    sigma: int
        the sigma limit to exclude data points (default: 3).
    niter: int
        the number of loop iterations to exclude  outliers and refit the model (default: 2).
    consecutive: int
        the number of outliers that have to be consecutive to be considered (default: 3).

    Returns
    -------
    fitted_model: PSF1D
        the PSF fitted model.
    outliers: list
        the list of the outlier indices.

    Examples
    --------

    Create the model:
    >>> import numpy as np
    >>> X = np.arange(0, 50)
    >>> PSF = PSF1D()
    >>> p = (1000, 25, 5, 1, -0.2, 1, 6000)
    >>> Y = PSF.evaluate(X, *p)
    >>> Y += 100*np.exp(-((X-10)/2)**2)
    >>> Y_err = np.sqrt(1+Y)

    Prepare the fit:
    >>> guess = (600, 20, 3.2, 1.2, -0.1, 2,  6000)
    >>> bounds = ((0, 6000), (10, 40), (0.5, 10), (0.5, 5), (-1, 0), (0.01, 10), (0, 8000))

    Fit with error bars:
    >>> model, outliers = fit_PSF1D_minuit_outlier_removal(X, Y, guess=guess, bounds=bounds, data_errors=Y_err,
    ... sigma=3, niter=2, consecutive=3)
    >>> res = [getattr(model, p).value for p in model.param_names]
    >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))
    """

    my_logger = set_logger(__name__)
    PSF = PSF1D(*guess)
    model = PSF1D()
    outliers = np.array([])
    indices = [i for i in range(x.shape[0]) if i not in outliers]

    def PSF1D_chisq_v2(params):
        mod = model.evaluate(x, *params)
        diff = mod[indices] - data[indices]
        if data_errors is None:
            return np.nansum(diff * diff)
        else:
            return np.nansum((diff / data_errors[indices]) ** 2)

    def PSF1D_chisq_v2_jac(params):
        diff = model.evaluate(x, *params) - data
        jac = model.fit_deriv(x, *params)
        if data_errors is None:
            return np.array([np.nansum(2 * jac[p] * diff) for p in range(len(params))])
        else:
            yy_err2 = data_errors * data_errors
            return np.array([np.nansum(2 * jac[p] * diff / yy_err2) for p in range(len(params))])

    error = 0.1 * np.abs(guess) * np.ones_like(guess)
    fix = [False] * len(guess)
    fix[-1] = True

    consecutive_outliers = []
    for step in range(niter):
        # noinspection PyArgumentList
        # it seems that minuit with a jacobian function works less good...
        m = Minuit.from_array_func(fcn=PSF1D_chisq_v2, start=guess, error=error, errordef=1, limit=bounds, fix=fix,
                                   print_level=0, grad=None)
        m.migrad()
        guess = m.np_values()
        PSF = PSF1D(*m.np_values())
        for ip, p in enumerate(model.param_names):
            setattr(model, p, guess[ip])
        # remove outliers
        indices_no_nan = ~np.isnan(data)
        diff = model(x[indices_no_nan]) - data[indices_no_nan]
        if data_errors is not None:
            outliers = np.where(np.abs(diff) / data_errors[indices_no_nan] > sigma)[0]
        else:
            std = np.std(diff)
            outliers = np.where(np.abs(diff) / std > sigma)[0]
        if len(outliers) > 0:
            # test if 3 consecutive pixels are in the outlier list
            test = 0
            consecutive_outliers = []
            for o in range(1, len(outliers)):
                t = outliers[o] - outliers[o - 1]
                if t == 1:
                    test += t
                else:
                    test = 0
                if test >= consecutive - 1:
                    for i in range(consecutive):
                        consecutive_outliers.append(outliers[o - i])
            consecutive_outliers = list(set(consecutive_outliers))
            # my_logger.debug(f"\n\tConsecutive oultlier indices: {consecutive_outliers}")
            indices = [i for i in range(x.shape[0]) if i not in outliers]
        else:
            break

    # my_logger.debug(f'\n\tPSF best fitting parameters:\n{PSF}')
    return PSF, consecutive_outliers


if __name__ == "__main__":
    import doctest

    #if np.__version__ >= "1.14.0":
    #    np.set_printoptions(legacy="1.13")

    doctest.testmod()

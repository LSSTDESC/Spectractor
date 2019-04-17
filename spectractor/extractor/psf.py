import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import basinhopping, minimize
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
from iminuit import Minuit

from astropy.modeling import fitting, Fittable1DModel, Fittable2DModel, Parameter
from astropy.modeling.models import Moffat1D
from astropy.stats import sigma_clip
from astropy.table import Table

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

    @staticmethod
    def evaluate(x, amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = (x - x_mean) * (x - x_mean)
        rr_gg = rr / (gamma * gamma)
        # use **(-alpha) instead of **(alpha) to avoid overflow power errors due to high alpha exponents
        # import warnings
        # warnings.filterwarnings('error')
        try:
            a = amplitude_moffat * ((1 + rr_gg) ** (-alpha) + eta_gauss * np.exp(-(rr / (2. * stddev * stddev))))
        except RuntimeWarning:
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
        return np.clip(a, 0, saturation)

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


class ChromaticPSF1D:

    def __init__(self, Nx, Ny, deg=4, saturation=None):
        # file_name="", image=None, order=1, target=None):
        # Spectrum.__init__(self, file_name=file_name, image=image, order=order, target=target)
        # self.profile_params = profile_params
        self.my_logger = set_logger(self.__class__.__name__)
        self.PSF1D = PSF1D()
        self.deg = deg
        self.degrees = {key: deg for key in self.PSF1D.param_names}
        self.degrees['saturation'] = 0
        self.Nx = Nx
        self.Ny = Ny
        self.profile_params = np.zeros((Nx, len(self.PSF1D.param_names)))
        self.pixels = np.arange(Nx).astype(int)
        # self.flux_sum = np.zeros(Nx)
        # self.flux_integral = np.zeros(Nx)
        # self.flux_err = np.zeros(Nx)
        # self.fwhms = np.zeros(Nx)
        # self.pixels_fwhm_sup = np.zeros(Nx)
        # self.pixels_fwhm_inf = np.zeros(Nx)
        self.n_poly_params = Nx
        self.fitted_pixels = np.arange(Nx).astype(int)
        arr = np.zeros((Nx, len(self.PSF1D.param_names) + 11))
        self.table = Table(arr, names=['lambdas', 'Dx', 'Dy', 'Dy_mean', 'flux_sum', 'flux_integral',
                                       'flux_err', 'fwhm', 'Dy_fwhm_sup', 'Dy_fwhm_inf', 'Dx_rot'] + list(
            self.PSF1D.param_names))
        self.saturation = saturation
        if saturation is None:
            self.saturation = 1e20
            self.my_logger.warning(f"\n\tSaturation level should be given to instanciate the ChromaticPSF1D "
                                   f"object. self.saturation is set arbitrarily to 1e20. Good luck.")
        for name in self.PSF1D.param_names:
            self.n_poly_params += self.degrees[name] + 1
        self.poly_params = np.zeros(self.n_poly_params)

    def fill_table_with_profile_params(self, profile_params):
        """
        Fill the table with the profile parameters.

        Parameters
        ----------
        profile_params: array
           a Nx * len(self.PSF1D.param_names) numpy array containing the PSF1D parameters as a function of pixels.

        Examples
        --------
        >>> s = ChromaticPSF1D(Nx=100, Ny=100, deg=4, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()
        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test)
        >>> s.fill_table_with_profile_params(profile_params)
        >>> assert(np.all(np.isclose(s.table['stddev'], 2*np.ones(100))))
        """
        for k, name in enumerate(self.PSF1D.param_names):
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
        Fit polynomial functions across the pixels for each PSF1D
        parameters. The order of the function is given by self.degrees.

        Parameters
        ----------
        profile_params: array
            a Nx * len(self.PSF1D.param_names) numpy array containing the PSF1D parameters as a function of pixels.

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
        pixels = np.linspace(-1, 1, self.Nx)
        poly_params = np.array([])
        for k, name in enumerate(self.PSF1D.param_names):
            if name is 'amplitude_moffat':
                amplitude = profile_params[:, k]
                poly_params = np.concatenate([poly_params, amplitude])
            else:
                # fit, cov, model = fit_poly1d(pixels, profile_params[:, k], order=self.degrees[name])
                fit = np.polynomial.legendre.legfit(pixels, profile_params[:, k], deg=self.degrees[name])
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
            Nx * len(self.PSF1D.param_names) numpy array containing the PSF1D parameters as a function of pixels.

        Examples
        --------
        >>> from spectractor.extractor.spectrum import Spectrum
        >>> s = Spectrum('./tests/data/reduc_20170530_134_spectrum.fits')
        >>> profile_params = s.chromatic_psf.from_table_to_profile_params()
        >>> assert(profile_params.shape == (s.chromatic_psf.Nx, len(s.chromatic_psf.PSF1D.param_names)))
        >>> assert not np.all(np.isclose(profile_params, np.zeros_like(profile_params)))
        """
        profile_params = np.zeros((self.Nx, len(self.PSF1D.param_names)))
        for k, name in enumerate(self.PSF1D.param_names):
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
        Evaluate the PSF1D profile parameters from the polynomial coefficients. If poly_params length is smaller
        than self.Nx, it is assumed that the amplitude_moffat parameters are not included and set to arbitrarily to 1.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
                * Nx first parameters are amplitudes for the Moffat transverse profiles
                * next parameters are polynomial coefficients for all the PSF1D parameters
                in the same order as in PSF1D definition, except amplitude_moffat
        force_positive: bool, optional
            Force some profile parameters to be positive despite the polynomial coefficients (default: False)

        Returns
        -------
        profile_params: array
            Nx * len(self.PSF1D.param_names) numpy array containing the PSF1D parameters as a function of pixels.

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
        pixels = np.linspace(-1, 1, self.Nx)
        profile_params = np.zeros((self.Nx, len(self.PSF1D.param_names)))
        shift = 0
        for k, name in enumerate(self.PSF1D.param_names):
            if name == 'amplitude_moffat':
                if len(poly_params) > self.Nx:
                    profile_params[:, k] = poly_params[:self.Nx]
                else:
                    profile_params[:, k] = np.ones(self.Nx)
            else:
                if len(poly_params) > self.Nx:
                    # profile_params[:, k] = np.polyval(poly_params[Nx + shift:Nx + shift + self.degrees[name] + 1],
                    #                                  pixels)
                    profile_params[:, k] = \
                        np.polynomial.legendre.legval(pixels,
                                                      poly_params[
                                                      self.Nx + shift:self.Nx + shift + self.degrees[name] + 1])
                else:
                    profile_params[:, k] = \
                        np.polynomial.legendre.legval(pixels, poly_params[shift:shift + self.degrees[name] + 1])
                shift = shift + self.degrees[name] + 1
        if force_positive:
            for k, name in enumerate(self.PSF1D.param_names):
                if name == "x_mean":
                    profile_params[profile_params[:, k] <= 0.1, k] = 1e-1
                    profile_params[profile_params[:, k] >= self.Ny, k] = self.Ny
                if name == "alpha":
                    profile_params[profile_params[:, k] <= 0.1, k] = 1e-1
                    # profile_params[profile_params[:, k] >= 6, k] = 6
                if name == "gamma":
                    profile_params[profile_params[:, k] <= 0.1, k] = 1e-1
                if name == "stddev":
                    profile_params[profile_params[:, k] <= 0.1, k] = 1e-1
        return profile_params

    def from_profile_params_to_shape_params(self, profile_params):
        """
        Compute the PSF1D integrals and FWHMS given the profile_params array and fill the table.

        Parameters
        ----------
        profile_params: array
         a Nx * len(self.PSF1D.param_names) numpy array containing the PSF1D parameters as a function of pixels.

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
            PSF = PSF1D(*p)
            fwhm = PSF.fwhm(x_array=pixel_y)
            self.table['flux_integral'][x] = PSF.integrate(bounds=(-10 * fwhm + p[1], 10 * fwhm + p[1]),
                                                           x_array=pixel_y)
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
        for k, name in enumerate(self.PSF1D.param_names):
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
        Evaluate the PSF1D profile parameters from the polynomial coefficients and check if they are within priors.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
                * Nx first parameters are amplitudes for the Moffat transverse profiles
                * next parameters are polynomial coefficients for all the PSF1D parameters
                in the same order as in PSF1D definition, except amplitude_moffat
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
        for k, name in enumerate(self.PSF1D.param_names):
            p = profile_params[:, k]
            if name == 'amplitude_moffat':
                if np.any(p < -noise_level):
                    in_bounds = False
                    penalty += np.abs(np.sum(profile_params[p < -noise_level, k])) / np.abs(np.mean(p))
                    outbound_parameter_name += name + ' '
            elif name is "x_mean":
                if np.any(p < 0):
                    penalty += np.abs(np.sum(profile_params[p < 0, k])) / np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
                if np.any(p > self.Ny):
                    penalty += np.sum(profile_params[p > self.Ny, k] - self.Ny)
                    penalty /= np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
            # elif name is "gamma":
            #     if np.any(p < 0) or np.any(p > self.Ny):
            #         in_bounds = False
            #         penalty = 1
            #         break
            elif name is "alpha":
                if np.any(p > 10):
                    penalty += np.sum(profile_params[p > 10, k] - 10)
                    penalty /= np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
                # if np.any(p < 0.1):
                #     penalty += np.sum(0.1 - profile_params[p < 0.1, k])
                #     penalty /= np.abs(np.mean(p))
                #     in_bounds = False
                #     outbound_parameter_name += name + ' '
            elif name is "eta_gauss":
                if np.any(p > 0):
                    penalty += np.sum(profile_params[p > 0, k])
                    # penalty /= np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
                if np.any(p < -1):
                    penalty += np.sum(-1 - profile_params[p < -1, k])
                    penalty /= np.abs(np.mean(p))
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

    def evaluate(self, poly_params):
        """
        Simulate a 2D spectrogram of size Nx times Ny.

        Parameters
        ----------
        poly_params: array_like
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
            # self.my_logger.warning(f"{k} {profile_params[k]}")
            output[:, k] = PSF1D.evaluate(y, *profile_params[k])
        return output

    def get_distance_along_dispersion_axis(self, shift_x=0, shift_y=0):
        return np.sqrt((self.table['Dx'] - shift_x) ** 2 + (self.table['Dy_mean'] - shift_y) ** 2)

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
        for name in self.PSF1D.param_names:
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

    def plot_summary(self, truth=None):
        fig, ax = plt.subplots(2, 1, sharex='all', figsize=(12, 6))
        test = PSF1D()
        PSF_models = []
        PSF_truth = []
        if truth is not None:
            PSF_truth = truth.from_poly_params_to_profile_params(truth.poly_params)
        all_pixels = np.arange(self.Nx)
        for i, name in enumerate(self.PSF1D.param_names):
            fit, cov, model = fit_poly1d(self.pixels, self.profile_params[:, i], order=self.degrees[name])
            PSF_models.append(np.polyval(fit, all_pixels))
        for i, name in enumerate(self.PSF1D.param_names):
            p = ax[0].plot(self.pixels, self.profile_params[:, i], marker='+', linestyle='none')
            ax[0].plot(self.fitted_pixels, self.profile_params[self.fitted_pixels, i], label=test.param_names[i],
                       marker='o', linestyle='none', color=p[0].get_color())
            if i > 0:
                ax[0].plot(all_pixels, PSF_models[i], color=p[0].get_color())
            if truth is not None:
                ax[0].plot(all_pixels, PSF_truth[:, i], color=p[0].get_color(), linestyle='--')
        img = np.zeros((self.Ny, self.Nx)).astype(float)
        yy, xx = np.mgrid[:self.Ny, :self.Nx]
        for x in self.pixels[::self.Nx // 10]:
            params = [PSF_models[p][x] for p in range(len(self.PSF1D.param_names))]
            psf = PSF2D.evaluate(xx, yy, 1, x, self.Ny // 2, *params[2:])
            psf /= np.max(psf)
            img += psf
        ax[1].imshow(img, origin='lower')
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Y [pixels]')
        ax[0].set_ylabel('PSF1D parameters')
        ax[0].grid()
        ax[1].grid(color='white', ls='solid')
        ax[1].grid(True)
        ax[0].set_yscale('symlog', linthreshy=10)
        ax[1].legend(title='PSF(x)')
        ax[0].legend()
        fig.tight_layout()
        # fig.subplots_adjust(hspace=0)
        if parameters.DISPLAY:
            plt.show()


def extract_background(data, err, deg=1, ws=(20, 30), pixel_step=1, sigma=5, live_fit=False):
    """
    Fit a polynomial background slice per slice along the x axis,
    with outlier removal, on lateral bands defined by the ws parameter.

    Parameters
    ----------
    data: array
        The 2D array image. The transverse profile is fitted on the y direction for all pixels along the x direction.
    err: array
        The uncertainties related to the data array.
    deg: int
        Degree of the polynomial model for the background (default: 1).
    ws: list
        up/down region extension where the sky background is estimated with format [int, int] (default: [20,30])
    pixel_step: int, optional
        The step in pixels between the slices to be fitted (default: 1).
        The values for the skipped pixels are interpolated with splines from the fitted parameters.
    live_fit: bool, optional
        If True, the transverse profile fit is plotted in live across the loop (default: False).
    sigma: int
        Sigma for outlier rejection (default: 5).

    Returns
    -------
    bgd_model_func: callable
        A 2D function to model the extracted background

    Examples
    --------

    # Build a mock spectrogram with random Poisson noise:
    >>> s0 = ChromaticPSF1D(Nx=100, Ny=100, saturation=1000)
    >>> params = s0.generate_test_poly_params()
    >>> saturation = params[-1]
    >>> data = s0.evaluate(params)
    >>> bgd = 10*np.ones_like(data)
    >>> data += bgd
    >>> data = np.random.poisson(data)
    >>> data_errors = np.sqrt(data+1)

    # Fit the transverse profile:
    >>> bgd_model = extract_background(data, data_errors, deg=1, ws=[30,50], live_fit=True, sigma=5, pixel_step=50)
    """
    my_logger = set_logger(__name__)
    Ny, Nx = data.shape
    middle = Ny // 2
    index = np.arange(Ny)
    # Prepare the fit
    bgd_index = np.concatenate((np.arange(0, middle - ws[0]), np.arange(middle + ws[0], Ny))).astype(int)
    pixel_range = np.arange(0, Nx, pixel_step)
    bgd_model = np.zeros_like(data).astype(float)
    for x in pixel_range:
        # fit the background with a polynomial function
        bgd = data[bgd_index, x]
        bgd_fit, outliers = fit_poly1d_outlier_removal(bgd_index, bgd, order=deg, sigma=sigma, niter=2)
        bgd_model[:, x] = bgd_fit(index)
        if live_fit:
            plot_transverse_PSF1D_profile(x, index, bgd_index, data, err, bgd_fit=bgd_fit,
                                          sigma=sigma, live_fit=live_fit)
    # prepare the background model
    # interpolate the grid
    bgd_fit = bgd_model[:, pixel_range]
    bgd_model_func = interp2d(pixel_range, index, bgd_fit, kind='linear', bounds_error=False, fill_value=None)
    if parameters.DEBUG:
        # fig, ax = plt.subplots(1,3, figsize=(12,4))
        # noinspection PyTypeChecker
        b = bgd_model_func(pixel_range, index)
        im = plt.imshow(b, origin='auto', aspect="auto")
        plt.colorbar(im)
        plt.title('Fitted background')
        if parameters.DISPLAY:
            plt.show()

    return bgd_model_func


def fit_transverse_PSF1D_profile(data, err, w, ws, pixel_step=1, saturation=None, live_fit=False, sigma=5, deg=4):
    """
    Fit the transverse profile of a 2D data image with a PSF1D profile.
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
    saturation: float, optional
        The saturation level of the image. Default is set to twice the maximum of the data array and has no effect.
    live_fit: bool, optional
        If True, the transverse profile fit is plotted in live accross the loop (default: False).
    sigma: int
        Sigma for outlier rejection (default: 5).

    Returns
    -------
    s: ChromaticPSF1D
        The chromatic PSF containing all the information on the wavelength dependeces of the parameters adn the flux_sum.

    Examples
    --------

    # Build a mock spectrogram with random Poisson noise:
    >>> s0 = ChromaticPSF1D(Nx=100, Ny=100, saturation=1000)
    >>> params = s0.generate_test_poly_params()
    >>> saturation = params[-1]
    >>> data = s0.evaluate(params)
    >>> bgd = 10*np.ones_like(data)
    >>> data += bgd
    >>> data = np.random.poisson(data)
    >>> data_errors = np.sqrt(data+1)

    # Fit the transverse profile:
    >>> s, bgd_model = fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50], pixel_step=10,
    ... saturation=saturation, live_fit=False, sigma=5)
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
    # Prepare the outputs
    s = ChromaticPSF1D(Nx, Ny, deg=deg, saturation=saturation)
    # Prepare the fit: start with the maximum of the spectrum
    ymax_index = int(np.unravel_index(np.argmax(data), data.shape)[1])
    bgd_index = np.concatenate((np.arange(0, middle - ws[0]), np.arange(middle + ws[0], Ny))).astype(int)
    y = data[:, ymax_index]
    guess = [np.nanmax(y) - np.nanmean(y), middle, 5, 2, 0, 2, saturation]
    maxi = np.abs(np.nanmax(y))
    bounds = [(0.1 * maxi, 2 * maxi), (middle - w, middle + w), (0.1, Ny // 2), (1, 6), (-1, 0), (0.1, Ny // 2),
              (0, 2 * saturation)]
    # first fit with moffat only to initialize the guess
    # hypothesis that max of spectrum if well describe by a focused PSF
    bgd = data[bgd_index, ymax_index]
    bgd_fit, outliers = fit_poly1d_outlier_removal(bgd_index, bgd, order=1, sigma=sigma, niter=2)
    signal = y - bgd_fit(index)
    fit = fit_moffat1d_outlier_removal(index, signal, sigma=sigma, niter=2,
                                       guess=guess[:4], bounds=np.array(bounds[:4]).T)
    moffat_guess = [getattr(fit, p).value for p in fit.param_names]
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
    bgd_model = np.zeros_like(data).astype(float)
    for x in pixel_range:
        guess = np.copy(guess)
        if x == ymax_index:
            guess = np.copy(init_guess)
        # fit the background with a polynomial function
        y = data[:, x]
        bgd = data[bgd_index, x]
        bgd_err = err[bgd_index, x]
        bgd_fit, outliers = fit_poly1d_outlier_removal(bgd_index, bgd, order=parameters.BGD_ORDER, sigma=sigma, niter=2)
        bgd_model[:, x] = bgd_fit(index)
        signal = y - bgd_fit(index)
        # in case guess amplitude is too low
        pdf = np.abs(signal)
        signal_sum = np.nansum(np.abs(signal))
        if signal_sum > 0:
            pdf /= signal_sum
        mean = np.nansum(pdf * index)
        std = np.sqrt(np.nansum(pdf * (index - mean) ** 2))
        maxi = np.abs(np.nanmax(signal))
        bounds[0] = (0.1 * np.nanstd(bgd), 2 * np.nanmax(y))
        # if guess[4] > -1:
        #    guess[0] = np.max(signal) / (1 + guess[4])
        if guess[0] * (1 + guess[4]) < 3 * np.nanstd(bgd):
            guess = [0.9 * maxi, mean, std, 2, 0, std, saturation]
        # if guess[0] * (1 + guess[4]) > 1.2 * maxi:
        #     guess = [0.9 * maxi, mean, std, 2, 0, std, saturation]
        PSF_guess = PSF1D(*guess)
        outliers = []
        # fit with outlier removal to clean background stars
        # first a simple moffat to get the general shape
        # moffat1d = fit_moffat1d_outlier_removal(index, signal, sigma=5, niter=2,
        #                                        guess=guess[:4], bounds=np.array(bounds[:4]).T)
        # guess[:4] = [getattr(moffat1d, p).value for p in moffat1d.param_names]
        # then PSF1D model using the result from the Moffat1D fit
        # fit, outliers = fit_PSF1D_outlier_removal(index, signal, sub_errors=err[:, x], method='basinhopping',
        #                                            guess=guess, bounds=bounds, sigma=5, niter=2,
        #                                            niter_basinhopping=5, T_basinhopping=1)
        # fit = fit_PSF1D_minuit(index[good_indices], signal[good_indices], guess=guess, bounds=bounds,
        #                       data_errors=err[good_indices, x])
        fit, outliers = fit_PSF1D_minuit_outlier_removal(index, signal, guess=guess, bounds=bounds,
                                                         data_errors=err[:, x], sigma=sigma, niter=2, consecutive=4)
        # good_indices = [i for i in index if i not in outliers]
        # outliers = [i for i in index if np.abs((signal[i] - fit(i)) / err[i, x]) > sigma]
        """
        This part is relevant if outliers rejection above is activated
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
        # if there are consecutive outliers or max is badly fitted, re-estimate the guess and refi
        if consecutive_outliers:  # or max_badfit:
            my_logger.warning(f'\n\tRefit because of max_badfit={max_badfit} or '
                              f'consecutive_outliers={consecutive_outliers}')
            guess = [1.3 * maxi, middle, guess[2], guess[3], -0.3, std / 2, saturation]
            fit, outliers = fit_PSF1D_outlier_removal(index, signal, sub_errors=err[:, x], method='basinhopping',
                                                      guess=guess, bounds=bounds, sigma=5, niter=2,
                                                      niter_basinhopping=20, T_basinhopping=1)
        """
        # compute the flux_sum
        guess = [getattr(fit, p).value for p in fit.param_names]
        s.profile_params[x, :] = guess
        s.table['flux_err'][x] = np.sqrt(np.sum(err[:, x] ** 2))
        s.table['flux_sum'][x] = np.sum(signal)
        if live_fit:
            plot_transverse_PSF1D_profile(x, index, bgd_index, data, err, fit, bgd_fit, guess,
                                          PSF_guess,  outliers, sigma, live_fit)
    # interpolate the skipped pixels with splines
    x = np.arange(Nx)
    xp = np.array(sorted(set(list(pixel_range))))
    s.fitted_pixels = xp
    for i in range(len(s.PSF1D.param_names)):
        yp = s.profile_params[xp, i]
        s.profile_params[:, i] = interp1d(xp, yp, kind='cubic')(x)
    s.table['flux_sum'] = interp1d(xp, s.table['flux_sum'][xp], kind='cubic', bounds_error=False,
                                   fill_value='extrapolate')(x)
    s.table['flux_err'] = interp1d(xp, s.table['flux_err'][xp], kind='cubic', bounds_error=False,
                                   fill_value='extrapolate')(x)
    s.poly_params = s.from_profile_params_to_poly_params(s.profile_params)
    s.from_profile_params_to_shape_params(s.profile_params)
    # prepare the background model
    # interpolate the grid
    bgd_fit = bgd_model[:, xp]
    bgd_model_func = interp2d(xp, index, bgd_fit, kind='linear', bounds_error=False, fill_value=None)
    if parameters.DEBUG:
        # fig, ax = plt.subplots(1,3, figsize=(12,4))
        # noinspection PyTypeChecker
        b = bgd_model_func(x, index)
        im = plt.imshow(b, origin='auto', aspect="auto")
        plt.colorbar(im)
        plt.title('Fitted background')
        if parameters.DISPLAY:
            plt.show()

    return s, bgd_model_func


def plot_transverse_PSF1D_profile(x, indices, bgd_indices, data, err, fit=None, bgd_fit=None,
                                  params=None, PSF_guess=None, outliers=[], sigma=3, live_fit=False):
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
    bgd_fit: callable, optional
        Best fitting model function for the background of the profile (default: None).
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
    >>> s0 = ChromaticPSF1D(Nx=100, Ny=100, saturation=1000)
    >>> params = s0.generate_test_poly_params()
    >>> saturation = params[-1]
    >>> data = s0.evaluate(params)
    >>> bgd = 10*np.ones_like(data)
    >>> data += bgd
    >>> data = np.random.poisson(data)
    >>> data_errors = np.sqrt(data+1)

    # Fit the transverse profile:
    >>> s, bgd_model = fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50], pixel_step=50,
    ... saturation=saturation, live_fit=True, sigma=5)

    """
    Ny = len(indices)
    y = data[:, x]
    bgd = data[bgd_indices, x]
    bgd_err = err[bgd_indices, x]
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex='all', gridspec_kw={'height_ratios': [5, 1]})
    ax[0].errorbar(np.arange(Ny), y, yerr=err[:, x], fmt='ro', label="original data")
    ax[0].errorbar(bgd_indices, bgd, yerr=bgd_err, fmt='bo', label="bgd data")
    if len(outliers) >0:
        ax[0].errorbar(outliers, data[outliers, x], yerr=err[outliers, x], fmt='go', label=f"outliers ({sigma}$\sigma$)")
    ax[0].plot(bgd_indices, bgd_fit(bgd_indices), 'b--', label="fitted bgd")
    if PSF_guess is not None:
        ax[0].plot(indices, PSF_guess(indices) + bgd_fit(indices), 'k--', label="guessed profile")
    if fit is not None and bgd_fit is not None:
        model = fit(indices) + bgd_fit(indices)
        ax[0].plot(indices, model, 'b-', label="fitted profile")
    ylim = ax[0].get_ylim()
    if params is not None:
        PSF_moffat = Moffat1D(*params[:4])
        ax[0].plot(indices, PSF_moffat(indices) + bgd_fit(indices), 'b+', label="fitted moffat")
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
    if bgd_fit is not None:
        model += bgd_fit(indices)
        model_outliers += bgd_fit(outliers)
    if fit is not None or bgd_fit is not None:
        residuals = (y - model) / err[:, x]  # / model
        residuals_err = err[:, x] / err[:, x]  # / model
        residuals_outliers = (data[outliers, x] - model_outliers) / err[outliers, x]  # / model_outliers
        residuals_outliers_err = err[outliers, x] / err[outliers, x]  # / model_outliers
        ax[1].errorbar(indices, residuals, yerr=residuals_err, fmt='ro')
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


def fit_chromatic_PSF1D(data, chromatic_psf, bgd_model_func=None, data_errors=None):
    """
    Fit a chromatic PSF1D model on 2D data.

    Parameters
    ----------
    data: array_like
        2D array containing the image data.
    chromatic_psf: ChromaticPSF1D
        First guess for the chromatic PSF, filled with a first guess of the polynomial shape parameters, mandatory.
    bgd_model_func: callable, optional
        A 2D function to model the extracted background (default: None -> null background)
    data_errors: np.array
        the 2D array uncertainties.

    Returns
    -------
    s: ChromaticPSF1D
        The chromatic PSF containing all the information on the wavelength dependences of the parameters adn the flux_sum.

    Examples
    --------

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

    # Estimate the first guess values
    >>> s, bgd_model_func = fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
    ... pixel_step=1, saturation=saturation, live_fit=False, deg=4)
    >>> s.plot_summary(truth=s0)

    # Fit the data:
    >>> s = fit_chromatic_PSF1D(data, s, bgd_model_func=bgd_model_func, data_errors=data_errors)
    >>> s.plot_summary(truth=s0)
    """
    my_logger = set_logger(__name__)
    Ny, Nx = data.shape
    if Ny != chromatic_psf.Ny:
        my_logger.error(f"\n\tData y shape {Ny} different from ChromaticPSF1D input self.Ny {chromatic_psf.Ny}.")
    if Nx != chromatic_psf.Nx:
        my_logger.error(f"\n\tData x shape {Nx} different from ChromaticPSF1D input self.Nx {chromatic_psf.Nx}.")
    guess = np.copy(chromatic_psf.poly_params)
    pixels = np.arange(Ny)

    W = 1. / (data_errors * data_errors)
    W = [np.diag(W[:, x]) for x in range(Nx)]
    poly_params = np.copy(guess)
    bgd = np.zeros_like(data)
    if bgd_model_func is not None:
        bgd = bgd_model_func(np.arange(Nx), pixels)

    data_subtracted = data - bgd
    bgd_std = float(np.std(np.random.poisson(bgd)))

    # import warnings
    # warnings.filterwarnings('error')

    def spectrogram_chisq(shape_params):
        # linear regression for the amplitude parameters
        poly_params[Nx:] = np.copy(shape_params)
        profile_params = chromatic_psf.from_poly_params_to_profile_params(poly_params, force_positive=True)
        profile_params[:Nx, 0] = 1
        # try:
        J = np.array([chromatic_psf.PSF1D.evaluate(pixels, *profile_params[x, :]) for x in range(Nx)])
        # except:
        #     for x in range(Nx):
        #         my_logger.warning(f"{x} {profile_params[x, :]}")
        J_dot_W_dot_J = np.array([J[x].T.dot(W[x]).dot(J[x]) for x in range(Nx)])
        # my_logger.warning(f"{shape_params}")
        # if np.any(np.isclose(J_dot_W_dot_J, 0, rtol=1e-6)):
        #     pass
        # my_logger.warning(f"crasssshhhhh {shape_params}")
        # profile_params = chromatic_psf.from_poly_params_to_profile_params(poly_params, force_positive=True, verbose=True)
        # for x in range(Nx):
        #   my_logger.warning(f"{x} {profile_params[x, :]}")
        #   my_logger.warning(f"{x} {J[x]}")
        # J_dot_W_dot_J[np.isclose(J_dot_W_dot_J, 0, rtol=1e-3) ] = 1e-3
        # try:
        amplitude_params = [
            J[x].T.dot(W[x]).dot(data_subtracted[:, x]) / (J_dot_W_dot_J[x]) if J_dot_W_dot_J[x] > 0 else 0.1 * bgd_std
            for x in
            range(Nx)]
        # amplitude_params = [
        #    J[x].T.dot(W[x]).dot(data_subtracted[:, x]) / (J_dot_W_dot_J[x]) for x in
        #    range(Nx)]

        # except RuntimeWarning:
        #    my_logger.warning(f"{J_dot_W_dot_J} {profile_params} {W} {J}")

        poly_params[:Nx] = amplitude_params
        in_bounds, penalty, name = chromatic_psf.check_bounds(poly_params, noise_level=5 * bgd_std)
        # my_logger.warning(f"amplitude {amplitude_params}")
        # if in_bounds is False:
        #     for k, n in enumerate(chromatic_psf.PSF1D.param_names):
        #         if n in name:
        #             my_logger.warning(f"{name} {profile_params[:, k]}  {shape_params} {penalty} {bgd_std} {name}")
        # if False:
        #     my_logger.warning(f"crasssshhhhh {shape_params}")
        #     # profile_params = chromatic_psf.from_poly_params_to_profile_params(poly_params, force_positive=True, verbose=True)
        #     for x in range(Nx):
        #         my_logger.warning(f"\n{x} {profile_params[x, :]} {J[x]} {J_dot_W_dot_J[x]}")
        #     my_logger.warning(f"\n{J_dot_W_dot_J}")
        #     sys.exit()
        mod = chromatic_psf.evaluate(poly_params)
        # if bgd_model_func is not None:
        #     mod += bgd
        chromatic_psf.poly_params = np.copy(poly_params)
        if data_errors is None:
            return np.nansum((mod - data_subtracted) ** 2) + penalty
        else:
            return np.nansum(((mod - data_subtracted) / data_errors) ** 2) + penalty

    # def spectrogram_chisq_jacobian(shape_params):
    #     poly_params[Nx:] = shape_params
    #     profile_params = s.from_poly_params_to_profile_params(poly_params)
    #     J = np.array([s.PSF1D.evaluate(pixels, *profile_params[x, :]) for x in range(Nx)])
    #     grad_J = np.array([s.PSF1D.fit_deriv(pixels, *profile_params[x, :]).T for x in range(Nx)])
    #     amplitude_params = np.array([J[x].T.dot(W[x]).dot(data[:, x]) / (J[x].T.dot(W[x]).dot(J[x]))
    #  for x in range(Nx)])
    #     diff = np.array([J[x].dot(amplitude_params[x]) - data[:, x] for x in range(Nx)])
    #     grad_chi2_over_p = []
    #     my_logger.warning(f"{shape_params[:-1]} {grad_J.shape}")
    #     for p, name in enumerate(shape_params[:-1]):
    #         my_logger.warning(f"{p} {name} {2*grad_J[10, :, p]}")
    #         my_logger.warning(f"{p} {name} {2*grad_J[10, :, p].T.dot(np.outer(W[10].
    # dot(diff[10]),amplitude_params[10].T))}")
    #         grad_chi2_over_p.append([2*grad_J[x, :, p].T.dot(np.outer(W[x].dot(diff[x]),amplitude_params.T))
    #  for x in range(Nx)])
    #     return grad_chi2_over_p
    # grad = spectrogram_chisq_jacobian(guess[Nx:])
    # my_logger.warning(f"{grad.shape} {grad}")

    my_logger.debug(f'\n\tStart chisq: {spectrogram_chisq(guess[Nx:])} with {guess[Nx:]}')
    error = 0.01 * np.abs(guess) * np.ones_like(guess)
    fix = [False] * (chromatic_psf.n_poly_params - Nx)
    fix[-1] = True
    bounds = chromatic_psf.set_bounds(data)
    # fix[:Nx] = [True] * Nx
    # noinspection PyArgumentList
    m = Minuit.from_array_func(fcn=spectrogram_chisq, start=guess[Nx:], error=error[Nx:], errordef=1,
                               fix=fix, print_level=parameters.DEBUG, limit=bounds[Nx:])
    m.migrad()
    # m.hesse()
    # print(m.np_matrix())
    # print(m.np_matrix(correlation=True))
    chromatic_psf.poly_params[Nx:] = m.np_values()
    chromatic_psf.profile_params = chromatic_psf.from_poly_params_to_profile_params(chromatic_psf.poly_params,
                                                                                    force_positive=True)
    chromatic_psf.fill_table_with_profile_params(chromatic_psf.profile_params)
    chromatic_psf.from_profile_params_to_shape_params(chromatic_psf.profile_params)
    if parameters.DEBUG:
        # Plot data, best fit model and residuals:
        chromatic_psf.plot_summary()
        plot_chromatic_PSF1D_residuals(chromatic_psf, bgd, data, data_errors, guess=guess, title='Best fit')
    return chromatic_psf


def plot_chromatic_PSF1D_residuals(s, bgd, data, data_errors, guess=None, live_fit=False, title=""):
    """Plot the residuals after fit_chromatic_PSF1D function.

    Parameters
    ----------
    s: ChromaticPSF1D
        The chromatic PSF1D function that has been fitted to data.
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
    im_guess = s.evaluate(guess) + bgd
    im1 = ax[1].imshow(im_guess, origin='lower', aspect='auto')
    fit = s.evaluate(s.poly_params) + bgd
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


def fit_PSF2D_minuit(x, y, data, guess=None, bounds=None, data_errors=None):
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

    error = 0.001 * np.abs(guess) * np.ones_like(guess)
    z = np.where(np.isclose(error, 0.0, 1e-6))
    error[z] = 0.001
    bounds = np.array(bounds)
    if bounds.shape[0] == 2 and bounds.shape[1] > 2:
        bounds = bounds.T
    guess = np.array(guess)

    def chisq_PSF2D(params):
        return PSF2D_chisq(params, model, x, y, data, data_errors)

    def chisq_PSF2D_jac(params):
        return PSF2D_chisq_jac(params, model, x, y, data, data_errors)

    fix = [False] * error.size
    fix[-1] = True
    # noinspection PyArgumentList
    m = Minuit.from_array_func(fcn=chisq_PSF2D, start=guess, error=error, errordef=1,
                               fix=fix, print_level=0, limit=bounds, grad=chisq_PSF2D_jac)

    m.tol = 0.001
    m.migrad()
    popt = m.np_values()

    my_logger.debug(f'\n{popt}')
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
        the PSF1D fitted model.
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
        the PSF1D fitted model.

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
        the PSF1D fitted model.
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
        m = Minuit.from_array_func(fcn=PSF1D_chisq_v2, start=guess, error=error, errordef=1, limit=bounds, fix=fix,
                                   print_level=0, grad=PSF1D_chisq_v2_jac)
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
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    doctest.testmod()


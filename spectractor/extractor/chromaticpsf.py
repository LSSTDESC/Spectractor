import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.interpolate import interp1d

from astropy.table import Table

from spectractor.tools import (fit_poly1d, plot_image_simple, compute_fwhm)
from spectractor.extractor.background import extract_spectrogram_background_sextractor
from spectractor.extractor.psf import PSF, PSFFitWorkspace, MoffatGauss, Moffat
from spectractor import parameters
from spectractor.config import set_logger
from spectractor.fit.fitter import FitWorkspace, run_minimisation, run_minimisation_sigma_clipping, RegFitWorkspace


class ChromaticPSF:
    """Class to store a PSF evolving with wavelength.

    The wavelength evolution is stored in an Astropy table instance. Whatever the PSF model, the common keywords are:

    * lambdas: the wavelength [nm]
    * Dx: the distance along X axis to order 0 position of the PSF model centroid  [pixels]
    * Dy: the distance along Y axis to order 0 position of the PSF model centroid [pixels]
    * Dy_disp_axis: the distance along Y axis to order 0 position of the mean dispersion axis [pixels]
    * flux_sum: the transverse sum of the data flux [spectrogram units]
    * flux_integral: the integral of the best fitting PSF model to the data (should be equal to the amplitude parameter
    of the PSF model if the model is correclty normalized to one) [spectrogram units]
    * flux_err: the uncertainty on flux_sum [spectrogram units]
    * fwhm: the FWHM of the best fitting PSF model [pixels]
    * Dy_fwhm_sup: the distance along Y axis to order 0 position of the upper FWHM edge [pixels]
    * Dy_fwhm_inf: the distance along Y axis to order 0 position of the lower FWHM edge [pixels]

    Then all the specific parameter of the PSF model are stored in other columns with their wavelength evolution
    (read from PSF.param_names attribute).

    A saturation level should be specified in data units.

    """

    def __init__(self, psf, Nx, Ny, x0=0, y0=None, deg=4, saturation=None, file_name=''):
        """Initialize a ChromaticPSF instance.


        Parameters
        ----------
        psf: PSF
            The PSF class model to build the PSF.
        Nx: int
            The number of pixels along the dispersion axis.
        Ny: int
            The number of pixels transverse to the dispersion axis.
        x0: float
            Relative position to pixel (0, 0) of the order 0 position along x (default: 0).
        y0: float
            Relative position to pixel (0, 0) of the order 0 position along y.
            If None, y0 is set at Ny/2 (default: None).
        deg: int, optional
            The degree of the polynomial functions to model the chromatic evolution
            of the PSF shape parameters (default: 4).
        saturation: float, optional
            The image saturation level (default: None).
        file_name: str, optional
            A path to a CSV file containing a ChromaticPSF table and load it (default: "").
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.psf = psf
        self.deg = -1
        self.degrees = {}
        self.set_polynomial_degrees(deg)
        self.Nx = Nx
        self.Ny = Ny
        self.x0 = x0
        if y0 is None:
            y0 = Ny / 2
        self.y0 = y0
        self.profile_params = np.zeros((Nx, len(self.psf.param_names)))
        self.pixels = np.mgrid[:Nx, :Ny]
        if file_name == '':
            arr = np.zeros((Nx, len(self.psf.param_names) + 10))
            self.table = Table(arr, names=['lambdas', 'Dx', 'Dy', 'Dy_disp_axis', 'flux_sum', 'flux_integral',
                                           'flux_err', 'fwhm', 'Dy_fwhm_sup', 'Dy_fwhm_inf'] + list(
                self.psf.param_names))
        else:
            self.table = Table.read(file_name)
        self.psf_param_start_index = 10
        self.n_poly_params = len(self.table)
        self.fitted_pixels = np.arange(len(self.table)).astype(int)
        self.saturation = saturation
        if saturation is None:
            self.saturation = 1e20
            self.my_logger.warning(f"\n\tSaturation level should be given to instanciate the ChromaticPSF "
                                   f"object. self.saturation is set arbitrarily to 1e20. Good luck.")
        for name in self.psf.param_names:
            self.n_poly_params += self.degrees[name] + 1
        self.poly_params = np.zeros(self.n_poly_params)
        self.poly_params_labels = []  # [f"a{k}" for k in range(self.poly_params.size)]
        self.poly_params_names = []  # "$a_{" + str(k) + "}$" for k in range(self.poly_params.size)]
        for ip, p in enumerate(self.psf.param_names):
            if ip == 0:
                self.poly_params_labels += [f"{p}^{k}" for k in range(len(self.table))]
                self.poly_params_names += \
                    ["$" + self.psf.axis_names[ip].replace("$", "") + "{(" + str(k) + ")}$"
                     for k in range(len(self.table))]
            else:
                for k in range(self.degrees[p] + 1):
                    self.poly_params_labels.append(f"{p}_{k}")
                    self.poly_params_names.append("$" + self.psf.axis_names[ip].replace("$", "")
                                                  + "^{(" + str(k) + ")}$")
        self.opt_reg = parameters.PSF_FIT_REG_PARAM
        self.cov_matrix = np.zeros((Nx, Nx))

    def set_polynomial_degrees(self, deg):
        self.deg = deg
        self.degrees = {key: deg for key in self.psf.param_names}
        self.degrees['saturation'] = 0

    def generate_test_poly_params(self):
        """
        A set of parameters to define a test spectrogram. PSF function must be MoffatGauss
        for this test example.

        Returns
        -------
        profile_params: array
            The list of the test parameters

        Examples
        --------

        >>> psf = MoffatGauss()
        >>> s = ChromaticPSF(psf, Nx=5, Ny=4, deg=1, saturation=20000)
        >>> params = s.generate_test_poly_params()

        ..  doctest::
            :hide:

            >>> assert(np.all(np.isclose(params,[10, 50, 100, 150, 200, 0, 0, 2, 0, 5, 0, 2, 0, -0.4, -0.4,1,0,20000])))

        """
        if not isinstance(self.psf, MoffatGauss) and not isinstance(self.psf, Moffat):
            raise TypeError(f"In this test function, PSF model must be MoffatGauss or Moffat. Gave {type(self.psf)}.")
        params = [50 * i for i in range(self.Nx)]
        params[0] = 10
        # add absorption lines
        if self.Nx > 80:
            params = list(np.array(params)
                          - 3000 * np.exp(-((np.arange(self.Nx) - 70) / 2) ** 2)
                          - 2000 * np.exp(-((np.arange(self.Nx) - 50) / 2) ** 2))
        params += [0.] * (self.degrees['x_c'] - 1) + [0, 0]  # x mean
        params += [0.] * (self.degrees['y_c'] - 1) + [0, 0]  # y mean
        params += [0.] * (self.degrees['gamma'] - 1) + [0, 5]  # gamma
        params += [0.] * (self.degrees['alpha'] - 1) + [0, 2]  # alpha
        if isinstance(self.psf, MoffatGauss):
            params += [0.] * (self.degrees['eta_gauss'] - 1) + [-0.4, -0.4]  # eta_gauss
            params += [0.] * (self.degrees['stddev'] - 1) + [0, 1]  # stddev
        params += [self.saturation]  # saturation
        poly_params = np.zeros_like(params)
        poly_params[:self.Nx] = params[:self.Nx]
        index = self.Nx - 1
        for name in self.psf.param_names:
            if name == 'amplitude':
                continue
            else:
                shift = self.degrees[name] + 1
                c = np.polynomial.legendre.poly2leg(params[index + shift:index:-1])
                coeffs = np.zeros(shift)
                coeffs[:c.size] = c
                poly_params[index + 1:index + shift + 1] = coeffs
                index = index + shift
        return poly_params

    def evaluate(self, poly_params, mode="1D"):
        """Simulate a 2D spectrogram of size Nx times Ny.

        Given a set of polynomial parameters defining the chromatic PSF model, a 2D spectrogram
        is produced either summing transverse 1D PSF profiles along the dispersion axis, or
        full 2D PSF profiles.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
            - Nx first parameters are amplitudes for the Moffat transverse profiles
            - next parameters are polynomial coefficients for all the PSF parameters in the same order
            as in PSF definition, except amplitude.
        mode: str, optional
            Set the evaluation mode: either transverse 1D PSF profile (mode="1D") or full 2D PSF profile (mode="2D").

        Returns
        -------
        output: array
            A 2D array with the model.

        Examples
        --------

        >>> psf = MoffatGauss()
        >>> s = ChromaticPSF(psf, Nx=100, Ny=20, deg=4, saturation=20000)
        >>> poly_params = s.generate_test_poly_params()

        1D evaluation:

        .. doctest::

            >>> output = s.evaluate(poly_params, mode="1D")
            >>> im = plt.imshow(output, origin='lower')  # doctest: +ELLIPSIS
            >>> plt.colorbar(im)  # doctest: +ELLIPSIS
            <matplotlib.colorbar.Colorbar object at 0x...>
            >>> plt.show()

        .. plot ::

            from spectractor.extractor.chromaticpsf import ChromaticPSF
            from spectractor.extractor.psf import MoffatGauss
            psf = MoffatGauss()
            s = ChromaticPSF(psf, Nx=100, Ny=20, deg=4, saturation=20000)
            poly_params = s.generate_test_poly_params()
            output = s.evaluate(poly_params, mode="1D")
            im = plt.imshow(output, origin='lower')
            plt.colorbar(im)
            plt.show()

        2D evaluation:

        .. doctest::

            >>> output = s.evaluate(poly_params, mode="2D")
            >>> im = plt.imshow(output, origin='lower')  # doctest: +ELLIPSIS
            >>> plt.colorbar(im)  # doctest: +ELLIPSIS
            <matplotlib.colorbar.Colorbar object at 0x...>
            >>> plt.show()

        .. plot ::

            from spectractor.extractor.chromaticpsf import ChromaticPSF
            from spectractor.extractor.psf import MoffatGauss
            psf = MoffatGauss()
            s = ChromaticPSF(psf, Nx=100, Ny=20, deg=4, saturation=20000)
            poly_params = s.generate_test_poly_params()
            output = s.evaluate(poly_params, mode="2D")
            im = plt.imshow(output, origin='lower')
            plt.colorbar(im)
            plt.show()

        """
        if mode == "2D":
            yy, xx = np.mgrid[:self.Ny, :self.Nx]
            pixels = np.asarray([xx, yy])
        elif mode == "1D":
            pixels = np.arange(self.Ny)
        else:
            raise ValueError(f"Unknown evaluation mode={mode}. Must be '1D' or '2D'.")
        self.psf.apply_max_width_to_bounds(max_half_width=self.Ny)
        profile_params = self.from_poly_params_to_profile_params(poly_params, apply_bounds=True)
        profile_params[:, 1] = np.arange(self.Nx)  # replace x_c
        output = np.zeros((self.Ny, self.Nx))
        if mode == "1D":
            for x in range(self.Nx):
                output[:, x] = self.psf.evaluate(pixels, p=profile_params[x, :])
        elif mode == "2D":
            for x in range(self.Nx):
                output += self.psf.evaluate(pixels, p=profile_params[x, :])
        return np.clip(output, 0, self.saturation)

    def fill_table_with_profile_params(self, profile_params):
        """
        Fill the table with the profile parameters.

        Parameters
        ----------
        profile_params: array
           a Nx * len(self.psf.param_names) numpy array containing the PSF parameters as a function of pixels.

        Examples
        --------

        >>> psf = MoffatGauss()
        >>> s = ChromaticPSF(psf, Nx=100, Ny=100, deg=4, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()
        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test)
        >>> s.fill_table_with_profile_params(profile_params)

        ..  doctest::
            :hide:

            >>> assert(np.all(np.isclose(s.table['stddev'], 1*np.ones(100))))

        """
        for k, name in enumerate(self.psf.param_names):
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

        >>> psf = MoffatGauss()
        >>> s = ChromaticPSF(psf, Nx=100, Ny=100, deg=4, saturation=8000)
        >>> s.table['Dx'] = np.arange(100)
        >>> s.rotate_table(45)

        ..  doctest::
            :hide:

            >>> assert(np.all(np.isclose(s.table['Dy'], -np.arange(100)/np.sqrt(2))))
            >>> assert(np.all(np.isclose(s.table['Dx'], np.arange(100)/np.sqrt(2))))
            >>> assert(np.all(np.isclose(s.table['Dy_fwhm_inf'], -np.arange(100)/np.sqrt(2))))
            >>> assert(np.all(np.isclose(s.table['Dy_fwhm_sup'], -np.arange(100)/np.sqrt(2))))
        """
        angle = angle_degree * np.pi / 180.
        rotmat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        # finish with Dy_c to get correct Dx
        for name in ['Dy', 'Dy_fwhm_inf', 'Dy_fwhm_sup', 'Dy_disp_axis']:
            vec = list(np.array([self.table['Dx'], self.table[name]]).T)
            rot_vec = np.array([np.dot(rotmat, v) for v in vec])
            self.table[name] = rot_vec.T[1]
        self.table['Dx'] = rot_vec.T[0]

    def from_profile_params_to_poly_params(self, profile_params):
        """
        Transform the profile_params array into a set of parameters for the chromatic PSF parameterisation.
        Fit Legendre polynomial functions across the pixels for each PSF parameters.
        The order of the polynomial functions is given by the self.degrees array.

        Parameters
        ----------
        profile_params: array
            a Nx * len(self.psf.param_names) numpy array containing the PSF parameters as a function of pixels.

        Returns
        -------
        profile_params: array_like
            A set of parameters that can be evaluated by the chromatic PSF class evaluate function.

        Examples
        --------

        Build a mock spectrogram with random Poisson noise:

        >>> psf = MoffatGauss()
        >>> s = ChromaticPSF(psf, Nx=100, Ny=100, deg=4, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()
        >>> data = s.evaluate(poly_params_test, mode="1D")
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        From the polynomial parameters to the profile parameters:

        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test)

        ..  doctest::
            :hide:

            >>> assert(np.all(np.isclose(profile_params[0], [10, 0, 50, 5, 2, 0, 1, 8e3])))

        From the profile parameters to the polynomial parameters:

        >>> profile_params = s.from_profile_params_to_poly_params(profile_params)

        ..  doctest::
            :hide:

            >>> assert(np.all(np.isclose(profile_params, poly_params_test)))
        """
        pixels = np.linspace(-1, 1, len(self.table))
        poly_params = np.array([])
        amplitude = None
        for k, name in enumerate(self.psf.param_names):
            if name is 'amplitude':
                amplitude = profile_params[:, k]
                poly_params = np.concatenate([poly_params, amplitude])
        if amplitude is None:
            self.my_logger.warning('\n\tAmplitude array not initialized. '
                                   'Polynomial fit for shape parameters will be unweighted.')
        for k, name in enumerate(self.psf.param_names):
            delta = 0
            if name is not 'amplitude':
                weights = np.copy(amplitude)
                if name is 'x_c':
                    delta = self.x0
                if name is 'y_c':
                    delta = self.y0
                fit = np.polynomial.legendre.legfit(pixels, profile_params[:, k] - delta,
                                                    deg=self.degrees[name], w=weights)
                poly_params = np.concatenate([poly_params, fit])
        return poly_params

    def from_table_to_profile_params(self):  # pragma: nocover
        """
        Extract the profile parameters from self.table and fill an array of profile parameters.

        Parameters
        ----------

        Returns
        -------
        profile_params: array
            Nx * len(self.psf.param_names) numpy array containing the PSF parameters as a function of pixels.

        Examples
        --------

        >>> from spectractor.extractor.spectrum import Spectrum
        >>> s = Spectrum('./tests/data/reduc_20170530_134_spectrum.fits')
        >>> profile_params = s.chromatic_psf.from_table_to_profile_params()

        ..  doctest::
            :hide:

            >>> assert(profile_params.shape == (s.chromatic_psf.Nx, len(s.chromatic_psf.psf.param_names)))
            >>> assert not np.all(np.isclose(profile_params, np.zeros_like(profile_params)))
        """
        profile_params = np.zeros((len(self.table), len(self.psf.param_names)))
        for k, name in enumerate(self.psf.param_names):
            profile_params[:, k] = self.table[name]
        return profile_params

    def from_table_to_poly_params(self):  # pragma: nocover
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

        ..  doctest::
            :hide:

            >>> assert(poly_params.size > s.chromatic_psf.Nx)
            >>> assert(len(poly_params.shape)==1)
            >>> assert not np.all(np.isclose(poly_params, np.zeros_like(poly_params)))
        """
        profile_params = self.from_table_to_profile_params()
        poly_params = self.from_profile_params_to_poly_params(profile_params)
        return poly_params

    def from_poly_params_to_profile_params(self, poly_params, apply_bounds=False):
        """
        Evaluate the PSF profile parameters from the polynomial coefficients. If poly_params length is smaller
        than self.Nx, it is assumed that the amplitude  parameters are not included and set to arbitrarily to 1.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
                - Nx first parameters are amplitudes for the Moffat transverse profiles
                - next parameters are polynomial coefficients for all the PSF parameters in the same order
                as in PSF definition, except amplitude

        apply_bounds: bool, optional
            Force profile parameters to respect their boundary conditions if they lie outside (default: False)

        Returns
        -------
        profile_params: array
            Nx * len(self.psf.param_names) numpy array containing the PSF parameters as a function of pixels.

        Examples
        --------

        Build a mock spectrogram with random Poisson noise:

        >>> psf = MoffatGauss()
        >>> s = ChromaticPSF(psf, Nx=100, Ny=100, deg=1, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()
        >>> data = s.evaluate(poly_params_test, mode="1D")
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        From the polynomial parameters to the profile parameters:

        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test, apply_bounds=True)

        ..  doctest::
            :hide:

            >>> assert(np.all(np.isclose(profile_params[0], [10, 0, 50, 5, 2, 0, 1, 8e3])))

        From the profile parameters to the polynomial parameters:

        >>> profile_params = s.from_profile_params_to_poly_params(profile_params)

        ..  doctest::
            :hide:

            >>> assert(np.all(np.isclose(profile_params, poly_params_test)))

        From the polynomial parameters to the profile parameters without Moffat amplitudes:

        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test[100:])

        ..  doctest::
            :hide:

            >>> assert(np.all(np.isclose(profile_params[0], [1, 0, 50, 5, 2, 0, 1, 8e3])))

        """
        length = len(self.table)
        pixels = np.linspace(-1, 1, length)
        profile_params = np.zeros((length, len(self.psf.param_names)))
        shift = 0
        for k, name in enumerate(self.psf.param_names):
            if name == 'amplitude':
                if len(poly_params) > length:
                    profile_params[:, k] = poly_params[:length]
                else:
                    profile_params[:, k] = np.ones(length)
            else:
                if len(poly_params) > length:
                    profile_params[:, k] = \
                        np.polynomial.legendre.legval(pixels,
                                                      poly_params[
                                                      length + shift:length + shift + self.degrees[name] + 1])
                else:
                    p = poly_params[shift:shift + self.degrees[name] + 1]
                    if len(p) > 0:  # to avoid saturation parameters in case not set
                        profile_params[:, k] = np.polynomial.legendre.legval(pixels, p)
                shift += self.degrees[name] + 1
                if name == 'x_c':
                    profile_params[:, k] += self.x0
                if name == 'y_c':
                    profile_params[:, k] += self.y0
        if apply_bounds:
            for k, name in enumerate(self.psf.param_names):
                profile_params[profile_params[:, k] <= self.psf.bounds[k][0], k] = self.psf.bounds[k][0]
                profile_params[profile_params[:, k] > self.psf.bounds[k][1], k] = self.psf.bounds[k][1]
        return profile_params

    def from_profile_params_to_shape_params(self, profile_params):
        """
        Compute the PSF integrals and FWHMS given the profile_params array and fill the table.

        Parameters
        ----------
        profile_params: array
         a Nx * len(self.psf.param_names) numpy array containing the PSF parameters as a function of pixels.

        Examples
        --------

        >>> psf = MoffatGauss()
        >>> s = ChromaticPSF(psf, Nx=100, Ny=100, deg=4, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()
        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test)
        >>> s.from_profile_params_to_shape_params(profile_params)

        ..  doctest::
            :hide:

            >>> assert s.table['fwhm'][-1] > 0

        """
        self.fill_table_with_profile_params(profile_params)
        pixel_x = np.arange(self.Nx).astype(int)
        # oversampling for precise computation of the PSF
        # pixels.shape = (2, Nx, Ny): self.pixels[1<-y, 0<-first pixel value column, :]
        # TODO: account for rotation ad projection effects is PSF is not round
        pixel_eval = np.arange(self.pixels[1, 0, 0], self.pixels[1, 0, -1], 0.1)
        for x in pixel_x:
            p = profile_params[x, :]
            # compute FWHM transverse to dispersion axis (assuming revolution symmetry of the PSF)
            out = self.psf.evaluate(pixel_eval, p=p)
            fwhm = compute_fwhm(pixel_eval, out, center=p[2], minimum=0)
            self.table['flux_integral'][x] = p[0]  # if MoffatGauss1D normalized
            self.table['fwhm'][x] = fwhm

    def set_bounds(self):
        """
        This function returns an array of bounds for iminuit. It is very touchy, change the values with caution !

        Returns
        -------
        bounds: array_like
            2D array containing the pair of bounds for each polynomial parameters.

        """
        bounds = [[], []]
        for k, name in enumerate(self.psf.param_names):
            tmp_bounds = [[-np.inf] * (1 + self.degrees[name]), [np.inf] * (1 + self.degrees[name])]
            if name is "saturation":
                tmp_bounds = [[0], [2 * self.saturation]]
            elif name is "amplitude":
                continue
            bounds[0] += tmp_bounds[0]
            bounds[1] += tmp_bounds[1]
        return np.array(bounds).T

    def set_bounds_for_minuit(self, data=None):  # pragma: nocover
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
        for k, name in enumerate(self.psf.param_names):
            tmp_bounds = [[-np.inf] * (1 + self.degrees[name]), [np.inf] * (1 + self.degrees[name])]
            if name is "saturation":
                if data is not None:
                    tmp_bounds = [[0.1 * np.max(data)], [2 * self.saturation]]
                else:
                    tmp_bounds = [[0], [2 * self.saturation]]
            elif name is "amplitude":
                continue
            bounds[0] += tmp_bounds[0]
            bounds[1] += tmp_bounds[1]
        return np.array(bounds).T

    def check_bounds(self, poly_params, noise_level=0):  # pragma: nocover
        """
        Evaluate the PSF profile parameters from the polynomial coefficients and check if they are within priors.

        Parameters
        ----------
        poly_params: array_like
            Parameter array of the model, in the form:
            - Nx first parameters are amplitudes for the Moffat transverse profiles
            - next parameters are polynomial coefficients for all the PSF parameters
            in the same order as in PSF definition, except amplitude
        noise_level: float, optional
            Noise level to set minimal boundary for amplitudes (negatively).

        Returns
        -------
        in_bounds: bool
            True if all parameters respect the model parameter priors.
        penalty: float
            Float value to add to chi square evaluating the degree of departure of a parameter from its boundaries.
        outbound_parameter_name: str
            Names of the parameters that are out of their boundaries.

        Examples
        --------

        Build a mock spectrogram with random Poisson noise:

        >>> psf = MoffatGauss()
        >>> s = ChromaticPSF(psf, Nx=100, Ny=100, deg=1, saturation=8000)
        >>> poly_params_test = s.generate_test_poly_params()

        Check bounds:

        >>> in_bounds, penalty, outbound_parameter_name = s.check_bounds(poly_params_test)

        .. doctest::
            :hide:

            >>> assert in_bounds is True
            >>> assert np.isclose(penalty, 0, atol=1e-16)
            >>> assert outbound_parameter_name is not None

        """
        in_bounds = True
        penalty = 0
        outbound_parameter_name = ""
        profile_params = self.from_poly_params_to_profile_params(poly_params)
        for k, name in enumerate(self.psf.param_names):
            p = profile_params[:, k]
            if name == 'amplitude':
                if np.any(p < -noise_level):
                    in_bounds = False
                    penalty += np.abs(np.sum(profile_params[p < -noise_level, k]))  # / np.mean(np.abs(p))
                    outbound_parameter_name += name + ' '
            elif name is "saturation":
                continue
            else:
                if np.any(p > self.psf.bounds[k][1]):
                    penalty += np.sum(profile_params[p > self.psf.bounds[k][1], k] - self.psf.bounds[k][1])
                    if not np.isclose(np.mean(p), 0):
                        penalty /= np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
                if np.any(p < self.psf.bounds[k][0]):
                    penalty += np.sum(self.psf.bounds[k][0] - profile_params[p < self.psf.bounds[k][0], k])
                    if not np.isclose(np.mean(p), 0):
                        penalty /= np.abs(np.mean(p))
                    in_bounds = False
                    outbound_parameter_name += name + ' '
            # elif name is "stddev":
            #     if np.any(p < 0) or np.any(p > self.Ny):
            #         in_bounds = False
            #         penalty = 1
            #         break
            # else:
            #    raise ValueError(f'Unknown parameter name {name} in set_bounds_for_minuit.')
        penalty *= self.Nx * self.Ny
        return in_bounds, penalty, outbound_parameter_name

    def get_distance_along_dispersion_axis(self, shift_x=0, shift_y=0):
        return np.asarray(np.sqrt((self.table['Dx'] - shift_x) ** 2 + (self.table['Dy_disp_axis'] - shift_y) ** 2))

    def plot_summary(self, truth=None):
        fig, ax = plt.subplots(2, 1, sharex='all', figsize=(12, 6))
        PSF_models = []
        PSF_truth = []
        if truth is not None:
            PSF_truth = truth.from_poly_params_to_profile_params(truth.poly_params, apply_bounds=True)
        all_pixels = np.arange(self.profile_params.shape[0])
        for i, name in enumerate(self.psf.param_names):
            fit, cov, model = fit_poly1d(all_pixels, self.profile_params[:, i], order=self.degrees[name])
            PSF_models.append(np.polyval(fit, all_pixels))
        for i, name in enumerate(self.psf.param_names):
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
            params = [PSF_models[p][x] for p in range(len(self.psf.param_names))]
            params[:3] = [1, x, self.Ny // 2]
            out = self.psf.evaluate(np.asarray([xx, yy]), p=params)
            out /= np.max(out)
            img += out
        ax[1].imshow(img, origin='lower')  # , extent=[0, self.Nx,
        #        self.Ny//2-parameters.PIXWIDTH_SIGNAL,
        #        self.Ny//2+parameters.PIXWIDTH_SIGNAL])
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Y [pixels]')
        ax[0].set_ylabel('PSF parameters')
        ax[0].grid()
        ax[1].grid(color='white', ls='solid')
        ax[1].grid(True)
        ax[0].set_yscale('symlog', linthresh=10)
        ax[1].legend(title='PSF(x)')
        ax[0].legend()
        fig.tight_layout()
        # fig.subplots_adjust(hspace=0)
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()

    def fit_transverse_PSF1D_profile(self, data, err, w, ws, pixel_step=1, bgd_model_func=None, saturation=None,
                                     live_fit=False, sigma_clip=5):
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
        sigma_clip: int
            Sigma for outlier rejection (default: 5).

        Examples
        --------

        Build a mock spectrogram with random Poisson noise:

        >>> psf = MoffatGauss()
        >>> s0 = ChromaticPSF(psf, Nx=100, Ny=100, saturation=1000)
        >>> params = s0.generate_test_poly_params()
        >>> saturation = params[-1]
        >>> data = s0.evaluate(params, mode="1D")
        >>> bgd = 10*np.ones_like(data)
        >>> xx, yy = np.meshgrid(np.arange(s0.Nx), np.arange(s0.Ny))
        >>> bgd += 1000*np.exp(-((xx-20)**2+(yy-10)**2)/(2*2))
        >>> data += bgd
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        Extract the background:

        >>> bgd_model_func, _, _ = extract_spectrogram_background_sextractor(data, data_errors, ws=[30,50])

        Fit the transverse profile:

        >>> s = ChromaticPSF(psf, Nx=100, Ny=100, deg=4, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50], pixel_step=10,
        ... bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False, sigma_clip=5)
        >>> s.plot_summary(truth=s0)

        ..  doctest::
            :hide:

            >>> assert(not np.any(np.isclose(s.table['flux_sum'][3:6], np.zeros(s.Nx)[3:6], rtol=1e-3)))
            >>> assert(np.all(np.isclose(s.table['Dy'][-10:-1], np.zeros(s.Nx)[-10:-1], rtol=1e-2)))

        """
        if saturation is None:
            saturation = 2 * np.max(data)
        Ny, Nx = data.shape
        middle = Ny // 2
        index = np.arange(Ny)
        # Prepare the fit: start with the maximum of the spectrum
        xmax_index = int(np.unravel_index(np.argmax(data[middle - w:middle + w, :]), data.shape)[1])
        bgd_index = np.concatenate((np.arange(0, middle - ws[0]), np.arange(middle + ws[0], Ny))).astype(int)
        y = data[:, xmax_index]
        # first fit with moffat only to initialize the guess
        # hypothesis that max of spectrum if well describe by a focused PSF
        if bgd_model_func is not None:
            signal = y - bgd_model_func(xmax_index, index)[:, 0]
        else:
            signal = y
        # fwhm = compute_fwhm(index, signal, minimum=0)
        # Initialize PSF
        psf = self.psf
        guess = np.copy(psf.p_default).astype(float)
        # guess = [2 * np.nanmax(signal), middle, 0.5 * fwhm, 2, 0, 0.1 * fwhm, saturation]
        signal_sum = np.nanmax(signal)
        guess[0] = signal_sum
        guess[1] = xmax_index
        guess[2] = middle
        guess[-1] = saturation
        # bounds = [(0.1 * maxi, 10 * maxi), (middle - w, middle + w), (0.1, min(fwhm, Ny // 2)), (0.1, self.alpha_max),
        #           (-1, 0),
        #           (0.1, min(Ny // 2, fwhm)),
        #           (0, 2 * saturation)]
        psf.apply_max_width_to_bounds(max_half_width=Ny)
        bounds = np.copy(psf.bounds)
        bounds[0] = (0.1 * signal_sum, 2 * signal_sum)
        bounds[2] = (middle - w, middle + w)
        bounds[-1] = (0, 2 * saturation)
        # moffat_guess = [2 * np.nanmax(signal), middle, 0.5 * fwhm, 2]
        # moffat_bounds = [(0.1 * maxi, 10 * maxi), (middle - w, middle + w), (0.1, min(fwhm, Ny // 2)), (0.1, 10)]
        # fit = fit_moffat1d_outlier_removal(index, signal, sigma=sigma, niter=2,
        #                                    guess=moffat_guess, bounds=np.array(moffat_bounds).T)
        # moffat_guess = [getattr(fit, p).value for p in fit.param_names]
        # signal_width_guess = moffat_guess[2]
        # bounds[2] = (0.1, min(Ny // 2, 5 * signal_width_guess))
        # bounds[5] = (0.1, min(Ny // 2, 5 * signal_width_guess))
        # guess[:4] = moffat_guess
        init_guess = np.copy(guess)
        # Go from max to right, then from max to left
        # includes the boundaries to avoid Runge phenomenum in chromatic_fit
        pixel_range = list(np.arange(xmax_index, Nx, pixel_step).astype(int))
        if Nx - 1 not in pixel_range:
            pixel_range.append(Nx - 1)
        pixel_range += list(np.arange(xmax_index, -1, -pixel_step).astype(int))
        if 0 not in pixel_range:
            pixel_range.append(0)
        pixel_range = np.array(pixel_range)
        for x in tqdm(pixel_range, disable=not parameters.VERBOSE):
            guess = np.copy(guess)
            if x == xmax_index:
                guess = np.copy(init_guess)
            # fit the background with a polynomial function
            y = data[:, x]
            if bgd_model_func is not None:
                # x_array = [x] * index.size
                signal = y - bgd_model_func(x, index)[:, 0]
            else:
                signal = y
            if np.mean(signal[bgd_index]) < 0:
                signal -= np.mean(signal[bgd_index])
            # in case guess amplitude is too low
            # pdf = np.abs(signal)
            signal_sum = np.nansum(signal[middle - ws[0]:middle + ws[0]])
            # if signal_sum > 0:
            #     pdf /= signal_sum
            # mean = np.nansum(pdf * index)
            # bounds[0] = (0.1 * np.nanstd(bgd), 2 * np.nanmax(y[middle - ws[0]:middle + ws[0]]))
            bounds[0] = (0.1 * signal_sum, 1.5 * signal_sum)
            guess[0] = signal_sum
            guess[1] = x
            # if guess[4] > -1:
            #    guess[0] = np.max(signal) / (1 + guess[4])
            # std = np.sqrt(np.nansum(pdf * (index - mean) ** 2))
            # if guess[0] < 3 * np.nanstd(bgd):
            #     guess[0] = float(0.9 * np.abs(np.nanmax(signal)))
            # if guess[0] * (1 + 0*guess[4]) > 1.2 * maxi:
            #     guess[0] = 0.9 * maxi
            psf.p = guess
            w = PSFFitWorkspace(psf, signal, data_errors=err[:, x], bgd_model_func=None,
                                live_fit=False, verbose=False)
            try:
                run_minimisation_sigma_clipping(w, method="newton", sigma_clip=sigma_clip, niter_clip=2, verbose=False,
                                                fix=w.fixed)
            except:
                pass
            best_fit = w.psf.p
            # It is better not to propagate the guess to further pixel columns
            # otherwise fit_chromatic_psf1D is more likely to get trapped in a local minimum
            # Randomness of the slice fit is better :
            # guess = best_fit
            self.profile_params[x, :] = best_fit
            # TODO: propagate amplitude uncertainties from Newton fit
            self.table['flux_err'][x] = np.sqrt(np.sum(err[:, x] ** 2))
            self.table['flux_sum'][x] = np.sum(signal)
            if live_fit and parameters.DISPLAY:  # pragma: no cover
                w.live_fit = True
                w.plot_fit()
        # interpolate the skipped pixels with splines
        all_pixels = np.arange(Nx)
        xp = np.array(sorted(set(list(pixel_range))))
        self.fitted_pixels = xp
        for i in range(len(self.psf.param_names)):
            yp = self.profile_params[xp, i]
            self.profile_params[:, i] = interp1d(xp, yp, kind='cubic')(all_pixels)
        for x in all_pixels:
            y = data[:, x]
            if bgd_model_func is not None:
                signal = y - bgd_model_func(x, index)[:, 0]
            else:
                signal = y
            if np.mean(signal[bgd_index]) < 0:
                signal -= np.mean(signal[bgd_index])
            self.table['flux_err'][x] = np.sqrt(np.sum(err[:, x] ** 2))
            self.table['flux_sum'][x] = np.sum(signal)
        self.poly_params = self.from_profile_params_to_poly_params(self.profile_params)
        self.from_profile_params_to_shape_params(self.profile_params)

    def fit_chromatic_psf(self, data, bgd_model_func=None, data_errors=None, mode="1D",
                          amplitude_priors_method="noprior", verbose=False, live_fit=False):
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
        mode: str, optional
            Set the fitting mode: either transverse 1D PSF profile (mode="1D") or full 2D PSF profile (mode="2D").
        amplitude_priors_method: str, optional
            Prior method to use to constrain the amplitude parameters of the PSF (default: "noprior").
        verbose: bool, optional
            Set the verbosity of the fitting process (default: False).

        Returns
        -------
        w: ChromaticPSFFitWorkspace
            The ChromaticPSFFitWorkspace containing info abut the fitting process.

        Examples
        --------

        Set the parameters:

        >>> parameters.PIXDIST_BACKGROUND = 40
        >>> parameters.PIXWIDTH_BACKGROUND = 10
        >>> parameters.PIXWIDTH_SIGNAL = 30
        >>> parameters.DEBUG = True

        Build a mock spectrogram with random Poisson noise using the full 2D PSF model:

        >>> psf = Moffat()
        >>> s0 = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=10000000)
        >>> params = s0.generate_test_poly_params()
        >>> params[:s0.Nx] *= 1
        >>> s0.poly_params = params
        >>> saturation = params[-1]
        >>> data = s0.evaluate(params, mode="2D")
        >>> bgd = 10*np.ones_like(data)
        >>> data += bgd
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data)

        Extract the background:

        >>> bgd_model_func, _, _ = extract_spectrogram_background_sextractor(data, data_errors, ws=[30,50])

        Propagate background uncertainties:

        >>> data_errors = np.sqrt(data_errors**2 + bgd_model_func(np.arange(s0.Nx), np.arange(s0.Ny)))

        Estimate the first guess values:

        >>> s = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
        ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)
        >>> s.plot_summary(truth=s0)
        >>> amplitude_residuals = [ [s0.poly_params[:s0.Nx], np.array(s.table["amplitude"])-s0.poly_params[:s0.Nx],
        ... np.array(s.table['amplitude'] * s.table['flux_err'] / s.table['flux_sum'])] ]

        Fit the data using the transverse 1D PSF model only:

        >>> w = s.fit_chromatic_psf(data, mode="1D", data_errors=data_errors, bgd_model_func=bgd_model_func,
        ... amplitude_priors_method="noprior", verbose=True)
        >>> s.plot_summary(truth=s0)
        >>> amplitude_residuals.append([s0.poly_params[:s0.Nx], w.amplitude_params-s0.poly_params[:s0.Nx],
        ... w.amplitude_params_err])

        ..  doctest::
            :hide:

            >>> residuals = (w.data-w.model)/w.err
            >>> assert w.costs[-1] /(w.Nx*w.Ny) < 1.5
            >>> assert np.abs(np.mean(residuals)) < 0.15
            >>> assert np.std(residuals) < 1.2

        Fit the data using the full 2D PSF model

        >>> parameters.PSF_FIT_REG_PARAM = 0.002
        >>> w = s.fit_chromatic_psf(data, mode="2D", data_errors=data_errors, bgd_model_func=bgd_model_func,
        ... amplitude_priors_method="psf1d", verbose=True)
        >>> s.plot_summary(truth=s0)
        >>> amplitude_residuals.append([s0.poly_params[:s0.Nx], w.amplitude_params-s0.poly_params[:s0.Nx],
        ... w.amplitude_params_err])
        >>> for k, label in enumerate(["Transverse", "PSF1D", "PSF2D"]):
        ...     plt.errorbar(np.arange(s0.Nx), amplitude_residuals[k][1]/amplitude_residuals[k][2],
        ...                  yerr=amplitude_residuals[k][2]/amplitude_residuals[k][2],
        ...                  fmt="+", label=label)  # doctest: +ELLIPSIS
        <ErrorbarContainer ...>
        >>> plt.grid()
        >>> plt.legend()  # doctest: +ELLIPSIS
        <matplotlib.legend.Legend object at ...>
        >>> plt.show()

        ..  doctest::
            :hide:

            >>> residuals = (w.data-w.model)/w.err
            >>> assert w.costs[-1] /(w.Nx*w.Ny) < 1.2
            >>> assert np.abs(np.mean(residuals)) < 0.15
            >>> assert np.std(residuals) < 1.2
            >>> assert np.abs(np.mean((w.amplitude_params - s0.poly_params[:s0.Nx])/w.amplitude_params_err)) < 0.5

        """
        if mode == "1D":
            w = ChromaticPSF1DFitWorkspace(self, data, data_errors=data_errors, bgd_model_func=bgd_model_func,
                                           amplitude_priors_method=amplitude_priors_method, verbose=verbose,
                                           live_fit=live_fit)
            run_minimisation(w, method="newton", ftol=1 / (w.Nx * w.Ny), xtol=1e-6, niter=50, fix=w.fixed)
        elif mode == "2D":
            w = ChromaticPSF2DFitWorkspace(self, data, data_errors=data_errors, bgd_model_func=bgd_model_func,
                                           amplitude_priors_method=amplitude_priors_method, verbose=verbose,
                                           live_fit=live_fit)
            # run_minimisation(w, method="newton", ftol=1 / (w.Nx * w.Ny), xtol=1e-6, niter=50, fix=w.fixed)
            run_minimisation_sigma_clipping(w, method="newton", ftol=1 / (w.Nx * w.Ny), xtol=1e-6, niter=50,
                                            fix=w.fixed, sigma_clip=10, niter_clip=3, verbose=verbose)
        else:
            raise ValueError(f"Unknown fitting mode={mode}. Must be '1D' or '2D'.")

        if w.amplitude_priors_method == "psf1d" and mode == "2D":
            w_reg = RegFitWorkspace(w, opt_reg=parameters.PSF_FIT_REG_PARAM, verbose=verbose)
            run_minimisation(w_reg, method="minimize", ftol=1e-4, xtol=1e-2, verbose=verbose, epsilon=[1e-1],
                             minimizer_method="Nelder-Mead")
            w_reg.opt_reg = 10 ** w_reg.p[0]
            self.my_logger.info(f"\n\tOptimal regularisation parameter: {w_reg.opt_reg}")
            w.reg = np.copy(w_reg.opt_reg)
            w.simulate(*w.p)
            self.opt_reg = w_reg.opt_reg
            if np.trace(w.amplitude_cov_matrix) < np.trace(w.amplitude_priors_cov_matrix):
                self.my_logger.warning(f"\n\tTrace of final covariance matrix ({np.trace(w.amplitude_cov_matrix)}) is "
                                       f"below the trace of the prior covariance matrix "
                                       f"({np.trace(w.amplitude_priors_cov_matrix)}). This is probably due to a very "
                                       f"high regularisation parameter in case of a bad fit. Therefore the final "
                                       f"covariance matrix is mulitiplied by the ratio of the traces and "
                                       f"the amplitude parameters are very close the amplitude priors.")
                r = np.trace(w.amplitude_priors_cov_matrix) / np.trace(w.amplitude_cov_matrix)
                w.amplitude_cov_matrix *= r
                w.amplitude_params_err = np.array([np.sqrt(w.amplitude_cov_matrix[x, x]) for x in range(self.Nx)])

        self.poly_params = w.poly_params
        self.cov_matrix = np.copy(w.amplitude_cov_matrix)

        # add background crop to y_c
        self.poly_params[w.Nx + w.y_c_0_index] += w.bgd_width

        # fill results
        self.psf.apply_max_width_to_bounds(max_half_width=w.Ny + 2 * w.bgd_width)
        self.set_bounds()
        self.profile_params = self.from_poly_params_to_profile_params(self.poly_params, apply_bounds=True)
        self.profile_params[:self.Nx, 0] = w.amplitude_params
        self.profile_params[:self.Nx, 1] = np.arange(self.Nx)
        self.fill_table_with_profile_params(self.profile_params)
        self.from_profile_params_to_shape_params(self.profile_params)

        # save plots
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            w.plot_fit()
        return w


class ChromaticPSFFitWorkspace(FitWorkspace):

    def __init__(self, chromatic_psf, data, data_errors, bgd_model_func=None, file_name="",
                 amplitude_priors_method="noprior",
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
        self.p = np.copy(self.chromatic_psf.poly_params[length:])  # remove saturation (fixed parameter))
        self.poly_params = np.copy(self.chromatic_psf.poly_params)
        self.input_labels = list(np.copy(self.chromatic_psf.poly_params_labels[length:]))
        self.axis_names = list(np.copy(self.chromatic_psf.poly_params_names[length:]))
        self.fixed = [False] * self.p.size
        for k, par in enumerate(self.input_labels):
            if "x_c" in par or "saturation" in par:
                self.fixed[k] = True
        self.y_c_0_index = -1
        for k, par in enumerate(self.input_labels):
            if par == "y_c_0":
                self.y_c_0_index = k
                break
        self.nwalkers = max(2 * self.ndim, nwalkers)

        # prepare the fit
        self.Ny, self.Nx = self.data.shape
        if self.Ny != self.chromatic_psf.Ny:
            raise AttributeError(f"Data y shape {self.Ny} different from "
                                 f"ChromaticPSF input Ny {self.chromatic_psf.Ny}.")
        if self.Nx != self.chromatic_psf.Nx:
            raise AttributeError(f"Data x shape {self.Nx} different from "
                                 f"ChromaticPSF input Nx {self.chromatic_psf.Nx}.")
        self.pixels = np.arange(self.Ny)

        # prepare the background, data and errors
        self.bgd = np.zeros_like(self.data)
        if self.bgd_model_func is not None:
            # xx, yy = np.meshgrid(np.arange(Nx), pixels)
            self.bgd = self.bgd_model_func(np.arange(self.Nx), self.pixels)
        self.data = self.data - self.bgd
        self.bgd_std = float(np.std(np.random.poisson(np.abs(self.bgd))))

        # crop spectrogram to fit faster
        self.bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        self.data = self.data[self.bgd_width:-self.bgd_width, :]
        self.data_flat = self.data.flatten()
        self.pixels = np.arange(self.data.shape[0])
        self.err = np.copy(self.err[self.bgd_width:-self.bgd_width, :])
        self.Ny, self.Nx = self.data.shape

        # update the bounds
        self.chromatic_psf.psf.apply_max_width_to_bounds(max_half_width=self.Ny)
        self.bounds = self.chromatic_psf.set_bounds()

        # error matrix
        # here image uncertainties are assumed to be uncorrelated
        # (which is not exactly true in rotated images)
        self.W = 1. / (self.err * self.err)
        self.W = self.W.flatten()
        self.W_dot_data = self.W * self.data_flat  # np.diag(self.W) @ self.data.flatten()

        # design matrix
        self.M = np.zeros((self.Nx, self.data.size))
        self.M_dot_W_dot_M = np.zeros((self.Nx, self.Nx))

        # prepare results
        self.amplitude_params = np.zeros(self.Nx)
        self.amplitude_params_err = np.zeros(self.Nx)
        self.amplitude_cov_matrix = np.zeros((self.Nx, self.Nx))

        # priors on amplitude parameters
        self.amplitude_priors_list = ['noprior', 'positive', 'smooth', 'psf1d', 'fixed']
        self.amplitude_priors_method = amplitude_priors_method
        self.fwhm_priors = np.copy(self.chromatic_psf.table['fwhm'])
        self.reg = parameters.PSF_FIT_REG_PARAM
        self.Q = np.zeros((self.Nx, self.Nx))
        self.Q_dot_A0 = np.zeros(self.Nx)
        if amplitude_priors_method not in self.amplitude_priors_list:
            raise ValueError(f"Unknown prior method for the amplitude fitting: {self.amplitude_priors_method}. "
                             f"Must be either {self.amplitude_priors_list}.")
        if self.amplitude_priors_method == "psf1d":
            self.amplitude_priors = np.copy(self.chromatic_psf.poly_params[:self.Nx])
            self.amplitude_priors_cov_matrix = np.copy(self.chromatic_psf.cov_matrix)
            # self.amplitude_priors_err = np.copy(self.chromatic_psf.table["flux_err"])
            self.Q = np.diag([1 / np.sum(self.err[:, x] ** 2) for x in range(self.Nx)])
            self.Q_dot_A0 = self.Q @ self.amplitude_priors
        if self.amplitude_priors_method == "fixed":
            self.amplitude_priors = np.copy(self.chromatic_psf.poly_params[:self.Nx])

    def plot_fit(self):
        gs_kw = dict(width_ratios=[3, 0.15], height_ratios=[1, 1, 1, 1])
        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7, 7), gridspec_kw=gs_kw)
        norm = np.max(self.data)
        plot_image_simple(ax[1, 0], data=self.model / norm, aspect='auto', cax=ax[1, 1], vmin=0, vmax=1,
                          units='1/max(data)')
        ax[1, 0].set_title("Model", fontsize=10, loc='center', color='white', y=0.8)
        plot_image_simple(ax[0, 0], data=self.data / norm, title='Data', aspect='auto',
                          cax=ax[0, 1], vmin=0, vmax=1, units='1/max(data)')
        ax[0, 0].set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
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
        ax[3, 0].errorbar(np.arange(self.Nx), self.data.sum(axis=0), yerr=np.sqrt(np.sum(self.err ** 2, axis=0)),
                          label='Data', fmt='k.', markersize=0.1)
        ax[3, 0].plot(np.arange(self.Nx), self.model.sum(axis=0), label='Model')
        ax[3, 0].set_ylabel('Transverse sum')
        ax[3, 0].set_xlabel(parameters.PLOT_XLABEL)
        ax[3, 0].legend(fontsize=7)
        ax[3, 0].set_xlim((0, self.data.shape[1]))
        ax[3, 0].grid(True)
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:  # pragma: no cover
            if parameters.DISPLAY and self.verbose:
                plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH,
                                     f'fit_chromatic_psf_best_fit_{self.amplitude_priors_method}.pdf'),
                        dpi=100, bbox_inches='tight')


class ChromaticPSF1DFitWorkspace(ChromaticPSFFitWorkspace):

    def __init__(self, chromatic_psf, data, data_errors, bgd_model_func=None, file_name="",
                 amplitude_priors_method="noprior",
                 nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        ChromaticPSFFitWorkspace.__init__(self, chromatic_psf, data, data_errors, bgd_model_func,
                                          file_name, amplitude_priors_method, nwalkers, nsteps, burnin, nbins, verbose,
                                          plot, live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        self.pixels = np.arange(self.Ny)

        # error matrix
        # here image uncertainties are assumed to be uncorrelated
        # (which is not exactly true in rotated images)
        self.W = 1. / (self.err * self.err)
        self.W = [self.W[:, x] for x in range(self.Nx)]  # [np.diag(self.W[:, x]) for x in range(self.Nx)]
        self.W_dot_data = [self.W[x] * self.data[:, x] for x in range(self.Nx)]

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

        Set the parameters:

        >>> parameters.PIXDIST_BACKGROUND = 40
        >>> parameters.PIXWIDTH_BACKGROUND = 10
        >>> parameters.PIXWIDTH_SIGNAL = 30

        Build a mock spectrogram with random Poisson noise:

        >>> psf = MoffatGauss()
        >>> s0 = ChromaticPSF(psf, Nx=100, Ny=100, deg=4, saturation=1000)
        >>> params = s0.generate_test_poly_params()
        >>> s0.poly_params = params
        >>> saturation = params[-1]
        >>> data = s0.evaluate(params, mode="1D")
        >>> bgd = 10*np.ones_like(data)
        >>> data += bgd
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        Extract the background:

        >>> bgd_model_func, _, _ = extract_spectrogram_background_sextractor(data, data_errors, ws=[30,50])

        Estimate the first guess values:

        >>> s = ChromaticPSF(psf, Nx=100, Ny=100, deg=4, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
        ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)
        >>> guess = np.copy(s.table["amplitude"])

        Fit the amplitude of data without any prior:

        >>> w = ChromaticPSF1DFitWorkspace(s, data, data_errors, bgd_model_func=bgd_model_func, verbose=True,
        ... amplitude_priors_method="noprior")
        >>> y, mod, mod_err = w.simulate(*s.poly_params[s.Nx:])
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert mod is not None
            >>> assert np.mean(np.abs(mod-w.data)/w.err) < 1

        Fit the amplitude of data smoothing the result with a window of size 10 pixels:

        >>> w = ChromaticPSF1DFitWorkspace(s, data, data_errors, bgd_model_func=bgd_model_func, verbose=True,
        ... amplitude_priors_method="smooth")
        >>> y, mod, mod_err = w.simulate(*s.poly_params[s.Nx:])
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert mod is not None
            >>> assert np.mean(np.abs(mod-w.data)/w.err) < 1

        Fit the amplitude of data using the transverse PSF1D fit as a prior and with a
        Tikhonov regularisation parameter set by parameters.PSF_FIT_REG_PARAM:

        >>> w = ChromaticPSF1DFitWorkspace(s, data, data_errors, bgd_model_func=bgd_model_func, verbose=True,
        ... amplitude_priors_method="psf1d")
        >>> y, mod, mod_err = w.simulate(*s.poly_params[s.Nx:])
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert mod is not None
            >>> assert np.mean(np.abs(mod-w.data)/w.err) < 1

        Set the amplitude parameters fixing the transverse PSF1D fit amplitudes:

        >>> w = ChromaticPSF1DFitWorkspace(s, data, data_errors, bgd_model_func=bgd_model_func, verbose=True,
        ... amplitude_priors_method="fixed")
        >>> y, mod, mod_err = w.simulate(*s.poly_params[s.Nx:])
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert mod is not None
            >>> assert np.mean(np.abs(mod-w.data)/w.err) < 1

        """
        # linear regression for the amplitude parameters
        poly_params = np.copy(self.poly_params)
        poly_params[self.Nx:] = np.copy(shape_params)
        poly_params[self.Nx + self.y_c_0_index] -= self.bgd_width
        profile_params = self.chromatic_psf.from_poly_params_to_profile_params(poly_params, apply_bounds=True)
        profile_params[:self.Nx, 0] = 1
        profile_params[:self.Nx, 1] = np.arange(self.Nx)
        # profile_params[:self.Nx, 2] -= self.bgd_width
        if self.amplitude_priors_method != "fixed":
            # Matrix filling
            M = np.array([self.chromatic_psf.psf.evaluate(self.pixels, p=profile_params[x, :]) for x in range(self.Nx)])
            M_dot_W_dot_M = np.array([M[x].T @ (self.W[x] * M[x]) for x in range(self.Nx)])
            if self.amplitude_priors_method != "psf1d":
                cov_matrix = np.diag([1 / M_dot_W_dot_M[x] if M_dot_W_dot_M[x] > 0 else 0.1 * self.bgd_std
                                      for x in range(self.Nx)])
                amplitude_params = np.array([
                    M[x].T @ self.W_dot_data[x] / (M_dot_W_dot_M[x]) if M_dot_W_dot_M[x] > 0 else 0.1 * self.bgd_std
                    for x in range(self.Nx)])
                if self.amplitude_priors_method == "positive":
                    amplitude_params[amplitude_params < 0] = 0
                elif self.amplitude_priors_method == "smooth":
                    null_indices = np.where(amplitude_params < 0)[0]
                    for index in null_indices:
                        right = amplitude_params[index]
                        for i in range(index, min(index + 10, self.Nx)):
                            right = amplitude_params[i]
                            if i not in null_indices:
                                break
                        left = amplitude_params[index]
                        for i in range(index, max(0, index - 10), -1):
                            left = amplitude_params[i]
                            if i not in null_indices:
                                break
                        amplitude_params[index] = 0.5 * (right + left)
                elif self.amplitude_priors_method == "noprior":
                    pass
            else:
                M_dot_W_dot_M_plus_Q = [M_dot_W_dot_M[x] + self.reg * self.Q[x, x] for x in range(self.Nx)]
                cov_matrix = np.diag([1 / M_dot_W_dot_M_plus_Q[x] if M_dot_W_dot_M_plus_Q[x] > 0 else 0.1 * self.bgd_std
                                      for x in range(self.Nx)])
                amplitude_params = [cov_matrix[x, x] * (M[x].T @ self.W_dot_data[x] + self.reg * self.Q_dot_A0[x])
                                    for x in range(self.Nx)]
            self.M = M
            self.M_dot_W_dot_M = M_dot_W_dot_M
            self.model = np.zeros_like(self.data)
            for x in range(self.Nx):
                self.model[:, x] = M[x] * amplitude_params[x]
        else:
            amplitude_params = np.copy(self.amplitude_priors)
            err2 = np.copy(amplitude_params)
            err2[err2 <= 0] = np.min(np.abs(err2[err2 > 0]))
            cov_matrix = np.diag(err2)
        self.amplitude_params = np.copy(amplitude_params)
        self.amplitude_params_err = np.array([np.sqrt(cov_matrix[x, x])
                                              if cov_matrix[x, x] > 0 else 0 for x in range(self.Nx)])
        self.amplitude_cov_matrix = np.copy(cov_matrix)
        poly_params[:self.Nx] = amplitude_params
        self.poly_params = np.copy(poly_params)
        poly_params[self.Nx + self.y_c_0_index] += self.bgd_width
        if self.amplitude_priors_method == "fixed":
            self.model = self.chromatic_psf.evaluate(poly_params, mode="1D")[self.bgd_width:-self.bgd_width, :]
        self.model_err = np.zeros_like(self.model)
        return self.pixels, self.model, self.model_err


class ChromaticPSF2DFitWorkspace(ChromaticPSFFitWorkspace):

    def __init__(self, chromatic_psf, data, data_errors, bgd_model_func=None, amplitude_priors_method="noprior",
                 file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        ChromaticPSFFitWorkspace.__init__(self, chromatic_psf, data, data_errors, bgd_model_func,
                                          file_name, amplitude_priors_method, nwalkers, nsteps, burnin, nbins, verbose,
                                          plot,
                                          live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        yy, xx = np.mgrid[:self.Ny, :self.Nx]
        self.pixels = np.asarray([xx, yy])

        # error matrix
        # here image uncertainties are assumed to be uncorrelated
        # (which is not exactly true in rotated images)
        self.W = 1. / (self.err * self.err)
        self.W = self.W.flatten()
        self.W_dot_data = self.W * self.data_flat  # np.diag(self.W) @ self.data.flatten()

        # regularisation matrices
        self.reg = parameters.PSF_FIT_REG_PARAM
        if amplitude_priors_method == "psf1d":
            # U = np.diag([1 / np.sqrt(np.sum(self.err[:, x]**2)) for x in range(self.Nx)])
            self.U = np.diag([1 / np.sqrt(self.amplitude_priors_cov_matrix[x, x]) for x in range(self.Nx)])
            L = np.diag(-2 * np.ones(self.Nx)) + np.diag(np.ones(self.Nx), -1)[:-1, :-1] \
                + np.diag(np.ones(self.Nx), 1)[:-1, :-1]
            L.astype(float)
            L[0, 0] = -1
            L[-1, -1] = -1
            self.L = L
            self.Q = L.T @ self.U.T @ self.U @ L
            self.Q_dot_A0 = self.Q @ self.amplitude_priors

    def simulate(self, *shape_params):
        r"""
        Compute a ChromaticPSF2D model given PSF shape parameters and minimizing
        amplitude parameters using a spectrogram data array.

        The ChromaticPSF2D model :math:`\vec{m}(\vec{x},\vec{p})` can be written as

        .. math ::
            :label: chromaticpsf2d

            \vec{m}(\vec{x},\vec{p}) = \sum_{i=0}^{N_x} A_i \phi\left(\vec{x},\vec{p}_i\right)

        with :math:`\vec{x}` the 2D array  of the pixel coordinates, :math:`\vec{A}` the amplitude parameter array
        along the x axis of the spectrogram, :math:`\phi\left(\vec{x},\vec{p}_i\right)` the 2D PSF kernel whose integral
        is normalised to one parametrized with the :math:`\vec{p}_i` non-linear parameter array. If the :math:`\vec{x}`
        2D array is flatten in 1D, equation :eq:`chromaticpsf2d` is

        .. math ::
            :label: chromaticpsf2d_matrix
            :nowrap:

            \begin{align}
            \vec{m}(\vec{x},\vec{p}) & = \mathbf{M}\left(\vec{x},\vec{p}\right) \mathbf{A} \\

            \mathbf{M}\left(\vec{x},\vec{p}\right) & = \left(\begin{array}{cccc}
             \phi\left(\vec{x}_1,\vec{p}_1\right) & \phi\left(\vec{x}_2,\vec{p}_1\right) & ...
             & \phi\left(\vec{x}_{N_x},\vec{p}_1\right) \\
             ... & ... & ... & ...\\
             \phi\left(\vec{x}_1,\vec{p}_{N_x}\right) & \phi\left(\vec{x}_2,\vec{p}_{N_x}\right) & ...
             & \phi\left(\vec{x}_{N_x},\vec{p}_{N_x}\right) \\
            \end{array}\right)
            \end{align}


        with :math:`\mathbf{M}` the design matrix.

        The goal of this function is to perform a minimisation of the amplitude vector :math:`\mathbf{A}` given
        a set of non-linear parameters :math:`\mathbf{p}` and a spectrogram data array :math:`mathbf{y}` modelise as

        .. math:: \mathbf{y} = \mathbf{m}(\vec{x},\vec{p}) + \vec{\epsilon}

        with :math:`\vec{\epsilon}` a random noise vector. The :math:`\chi^2` function to minimise is

        .. math::
            :label: chromaticspsf2d_chi2

            \chi^2(\mathbf{A})= \left(\mathbf{y} - \mathbf{M}\left(\vec{x},\vec{p}\right) \mathbf{A}\right)^T \mathbf{W}
            \left(\mathbf{y} - \mathbf{M}\left(\vec{x},\vec{p}\right) \mathbf{A} \right)


        with :math:`\mathbf{W}` the weight matrix, inverse of the covariance matrix. In our case this matrix is diagonal
        as the pixels are considered all independent. The minimum of equation :eq:`chromaticspsf2d_chi2` is reached for
        a the set of amplitude parameters :math:`\hat{\mathbf{A}}` given by

        .. math::

            \hat{\mathbf{A}} =  (\mathbf{M}^T \mathbf{W} \mathbf{M})^{-1} \mathbf{M}^T \mathbf{W} \mathbf{y}

        The error matrix on the :math:`\hat{\mathbf{A}}` coefficient is simply
        :math:`(\mathbf{M}^T \mathbf{W} \mathbf{M})^{-1}`.

        See Also
        --------
        ChromaticPSF1DFitWorkspace.simulate

        Parameters
        ----------
        shape_params: array_like
            PSF shape polynomial parameter array.

        Examples
        --------

        Set the parameters:

        .. doctest::

            >>> parameters.PIXDIST_BACKGROUND = 40
            >>> parameters.PIXWIDTH_BACKGROUND = 10
            >>> parameters.PIXWIDTH_SIGNAL = 30

        Build a mock spectrogram with random Poisson noise:

        .. doctest::

            >>> psf = Moffat()
            >>> s0 = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=100000)
            >>> params = s0.generate_test_poly_params()
            >>> params[:s0.Nx] *= 10
            >>> s0.poly_params = params
            >>> saturation = params[-1]
            >>> data = s0.evaluate(params, mode="2D")
            >>> bgd = 10*np.ones_like(data)
            >>> data += bgd
            >>> data = np.random.poisson(data)
            >>> data_errors = np.sqrt(data+1)

        Extract the background:

        .. doctest::

            >>> bgd_model_func, _, _ = extract_spectrogram_background_sextractor(data, data_errors, ws=[30,50])

        Estimate the first guess values:

        .. doctest::

            >>> s = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=saturation)
            >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
            ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)
            >>> s.plot_summary(truth=s0)

        Simulate the data with fixed amplitude priors:

        .. doctest::

            >>> w = ChromaticPSF2DFitWorkspace(s, data, data_errors, bgd_model_func=bgd_model_func,
            ... amplitude_priors_method="fixed", verbose=True)
            >>> y, mod, mod_err = w.simulate(s.poly_params[s.Nx:])
            >>> w.plot_fit()

        .. doctest::
            :hide:

            >>> assert mod is not None

        Simulate the data with a Tikhonov prior on amplitude parameters:

        .. doctest::

            >>> parameters.PSF_FIT_REG_PARAM = 0.002
            >>> s.poly_params = s.from_table_to_poly_params()
            >>> w = ChromaticPSF2DFitWorkspace(s, data, data_errors, bgd_model_func=bgd_model_func,
            ... amplitude_priors_method="psf1d", verbose=True)
            >>> y, mod, mod_err = w.simulate(s.poly_params[s.Nx:])
            >>> w.plot_fit()

        .. doctest::
            :hide:

            >>> assert mod is not None
            >>> w = ChromaticPSF2DFitWorkspace(s0, data, data_errors, bgd_model_func=bgd_model_func,
            ... amplitude_priors_method="fixed", verbose=True)
            >>> y, mod, mod_err = w.simulate(s0.poly_params[s0.Nx:])
            >>> w.plot_fit()
            >>> assert np.abs(np.mean((w.amplitude_params-s0.poly_params[:s0.Nx])/w.amplitude_params_err)) < 0.05

        """
        # linear regression for the amplitude parameters
        # prepare the vectors
        poly_params = np.copy(self.poly_params)
        poly_params[self.Nx:] = np.copy(shape_params)
        poly_params[self.Nx + self.y_c_0_index] -= self.bgd_width
        profile_params = self.chromatic_psf.from_poly_params_to_profile_params(poly_params, apply_bounds=True)
        profile_params[:self.Nx, 0] = 1
        profile_params[:self.Nx, 1] = np.arange(self.Nx)
        # profile_params[:self.Nx, 2] -= self.bgd_width
        if self.amplitude_priors_method != "fixed":
            # Matrix filling
            M = np.array([self.chromatic_psf.psf.evaluate(self.pixels, p=profile_params[x, :]).flatten()
                          for x in range(self.Nx)]).T
            W_dot_M = np.array([M[:, x] * self.W for x in range(self.Nx)]).T
            # Compute the minimizing amplitudes
            M_dot_W_dot_M = M.T @ W_dot_M
            if self.amplitude_priors_method != "psf1d":
                try:
                    L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M))
                    cov_matrix = L.T @ L
                except np.linalg.LinAlgError:
                    cov_matrix = np.linalg.inv(M_dot_W_dot_M)
                amplitude_params = cov_matrix @ (M.T @ self.W_dot_data)
                if self.amplitude_priors_method == "positive":
                    amplitude_params[amplitude_params < 0] = 0
                elif self.amplitude_priors_method == "smooth":
                    null_indices = np.where(amplitude_params < 0)[0]
                    for index in null_indices:
                        right = amplitude_params[index]
                        for i in range(index, min(index + 10, self.Nx)):
                            right = amplitude_params[i]
                            if i not in null_indices:
                                break
                        left = amplitude_params[index]
                        for i in range(index, max(0, index - 10), -1):
                            left = amplitude_params[i]
                            if i not in null_indices:
                                break
                        amplitude_params[index] = 0.5 * (right + left)
                elif self.amplitude_priors_method == "noprior":
                    pass
            else:
                M_dot_W_dot_M_plus_Q = M_dot_W_dot_M + self.reg * self.Q
                try:
                    L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M_plus_Q))
                    cov_matrix = L.T @ L
                except np.linalg.LinAlgError:
                    cov_matrix = np.linalg.inv(M_dot_W_dot_M_plus_Q)
                amplitude_params = cov_matrix @ (M.T @ self.W_dot_data + self.reg * self.Q_dot_A0)
            self.M = M
            self.M_dot_W_dot_M = M_dot_W_dot_M
            self.model = (M @ amplitude_params).reshape((self.Ny, self.Nx))
        else:
            amplitude_params = np.copy(self.amplitude_priors)
            err2 = np.copy(amplitude_params)
            err2[err2 <= 0] = np.min(np.abs(err2[err2 > 0]))
            cov_matrix = np.diag(err2)
        poly_params[:self.Nx] = amplitude_params
        self.poly_params = np.copy(poly_params)
        poly_params[self.Nx + self.y_c_0_index] += self.bgd_width
        self.amplitude_params = np.copy(amplitude_params)
        # TODO: propagate and marginalize over the shape parameter uncertainties ?
        self.amplitude_params_err = np.array([np.sqrt(cov_matrix[x, x]) for x in range(self.Nx)])
        self.amplitude_cov_matrix = np.copy(cov_matrix)
        # in_bounds, penalty, name = self.chromatic_psf.check_bounds(poly_params, noise_level=self.bgd_std)
        if self.amplitude_priors_method == "fixed":
            self.model = self.chromatic_psf.evaluate(poly_params, mode="2D")[self.bgd_width:-self.bgd_width, :]
        self.model_err = np.zeros_like(self.model)
        return self.pixels, self.model, self.model_err


if __name__ == "__main__":
    import doctest

    doctest.testmod()

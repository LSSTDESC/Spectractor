import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import copy

from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy import sparse

from astropy.table import Table

from spectractor.tools import (rescale_x_to_legendre, plot_image_simple, compute_fwhm)
from spectractor.extractor.background import extract_spectrogram_background_sextractor
from spectractor.extractor.psf import PSF, PSFFitWorkspace, MoffatGauss, Moffat
from spectractor import parameters
from spectractor.config import set_logger
from spectractor.fit.fitter import (FitParameters, FitWorkspace, run_minimisation, run_minimisation_sigma_clipping,
                                    RegFitWorkspace)
try:
    import sparse_dot_mkl
except ModuleNotFoundError:
    sparse_dot_mkl = None


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
        self.profile_params = np.zeros((Nx, len(self.psf.params.labels)))
        yy, xx = np.mgrid[:self.Ny, :self.Nx]
        self.pixels = np.asarray([xx, yy])

        self.saturation = 1e20
        self.psf_param_start_index = 0
        self.n_poly_params = 0
        self.fitted_pixels = 0
        if file_name == '':
            arr = np.zeros((self.Nx, len(self.psf.params.labels) + 10))
            self.table = Table(arr, names=['lambdas', 'Dx', 'Dy', 'Dy_disp_axis', 'flux_sum', 'flux_integral', 'flux_err',
                                           'fwhm', 'Dy_fwhm_sup', 'Dy_fwhm_inf'] + list(self.psf.params.labels))
        else:
            self.table = Table.read(file_name)
        self.params = None
        self.init_from_table(self.table, saturation=saturation)
        self.opt_reg = parameters.PSF_FIT_REG_PARAM
        self.cov_matrix = np.zeros((Nx, Nx))
        if file_name != "":
            self.params.values = self.from_table_to_poly_params()

    def init_from_table(self, table, saturation=None):
        self.table = table
        self.n_poly_params = len(self.table)
        self.fitted_pixels = np.arange(len(self.table)).astype(int)
        self.saturation = saturation
        if saturation is None:
            self.saturation = 1e20
            self.my_logger.warning(f"\n\tSaturation level should be given to instanciate the ChromaticPSF "
                                   f"object. self.saturation is set arbitrarily to 1e20. Good luck.")
        for name in self.psf.params.labels:
            if "amplitude" in name:
                continue
            self.n_poly_params += self.degrees[name] + 1
        labels = []  # [f"a{k}" for k in range(self.poly_params.size)]
        axis_names = []  # "$a_{" + str(k) + "}$" for k in range(self.poly_params.size)]
        for ip, p in enumerate(self.psf.params.labels):
            if ip == 0:
                labels += [f"{p}_{k}" for k in range(len(self.table))]
                axis_names += ["$" + self.psf.params.axis_names[ip].replace("$", "") + "_{" + str(k) + "}$" for k in range(len(self.table))]
            else:
                for k in range(self.degrees[p] + 1):
                    labels.append(f"{p}_{k}")
                    axis_names.append("$" + self.psf.params.axis_names[ip].replace("$", "") + "^{(" + str(k) + ")}$")
        self.params = FitParameters(values=np.zeros(self.n_poly_params), labels=labels, axis_names=axis_names)

    def resize_table(self, new_Nx):
        """Resize the table and interpolate existing values to a new length size.

        Parameters
        ----------
        new_Nx: int
            New length of the ChromaticPSF on X axis.

        Examples
        --------

        >>> psf = Moffat()
        >>> s = ChromaticPSF(psf, Nx=20, Ny=5, deg=1, saturation=20000)
        >>> params = s.generate_test_poly_params()
        >>> s.fill_table_with_profile_params(s.from_poly_params_to_profile_params(params))
        >>> print(np.sum(s.table["gamma"]))
        100.0
        >>> print(s.table["gamma"].size)
        20
        >>> s.resize_table(10)
        >>> print(np.sum(s.table["gamma"]))
        50.0
        >>> print(s.table["gamma"].size)
        10

        """
        new_chromatic_psf = ChromaticPSF(psf=self.psf, Nx=new_Nx, Ny=self.Ny, x0=self.x0, y0=self.y0,
                                         deg=self.deg, saturation=self.saturation)
        old_x = np.linspace(0, 1, self.Nx)
        new_x = np.linspace(0, 1, new_Nx)
        for colname in self.table.colnames:
            new_chromatic_psf.table[colname] = np.interp(new_x, old_x, np.array(self.table[colname]))
        self.init_from_table(table=new_chromatic_psf.table, saturation=self.saturation)
        self.__dict__.update(new_chromatic_psf.__dict__)

    def crop_table(self, new_Nx):
        """Crop the table to a new length size.

        Parameters
        ----------
        new_Nx: int
            New length of the ChromaticPSF on X axis.

        Examples
        --------

        >>> psf = Moffat()
        >>> s = ChromaticPSF(psf, Nx=20, Ny=5, deg=1, saturation=20000)
        >>> params = s.generate_test_poly_params()
        >>> s.fill_table_with_profile_params(s.from_poly_params_to_profile_params(params))
        >>> print(np.sum(s.table["gamma"]))
        100.0
        >>> print(s.table["gamma"].size)
        20
        >>> s.crop_table(10)
        >>> print(np.sum(s.table["gamma"]))
        50.0
        >>> print(s.table["gamma"].size)
        10

        """
        new_chromatic_psf = ChromaticPSF(psf=self.psf, Nx=new_Nx, Ny=self.Ny, x0=self.x0, y0=self.y0,
                                         deg=self.deg, saturation=self.saturation)
        for colname in self.table.colnames:
            new_chromatic_psf.table[colname] = self.table[colname][:new_Nx]
        self.init_from_table(table=new_chromatic_psf.table, saturation=self.saturation)
        self.__dict__.update(new_chromatic_psf.__dict__)

    def set_polynomial_degrees(self, deg):
        self.deg = deg
        self.degrees = {key: deg for key in self.psf.params.labels}
        self.degrees["x_c"] = max(self.degrees["x_c"], 1)
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
            >>> assert(np.all(np.isclose(params,[10, 50, 100, 150, 200, 0, 0, 0, 0, 5, 0, 2, 0, -0.4, -0.4,1,0,20000])))

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
        for name in self.psf.params.labels:
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

    def set_pixels(self, mode):
        """Return the pixels array to evaluate ChromaticPSF.
        If mode='1D', one 1D array of pixels along y axis is returned.
        If mode='2D', two 2D meshgrid arrays of pixels are returned.

        Parameters
        ----------
        mode, str
            Must be '1D' or '2D'.

        Returns
        -------
        pixels: array_like
            The pixel array.

        Examples
        --------
        >>> psf = MoffatGauss()
        >>> s = ChromaticPSF(psf, Nx=5, Ny=4, deg=1, saturation=20000)
        >>> pixels = s.set_pixels(mode='1D')
        >>> pixels.shape
        (4,)
        >>> pixels = s.set_pixels(mode='2D')
        >>> pixels.shape
        (2, 4, 5)

        """
        if mode == "2D":
            yy, xx = np.mgrid[:self.Ny, :self.Nx]
            pixels = np.asarray([xx, yy])
        elif mode == "1D":
            pixels = np.arange(self.Ny)
        else:
            raise ValueError(f"Unknown evaluation mode={mode}. Must be '1D' or '2D'.")
        return pixels

    def evaluate(self, pixels, poly_params, fwhmx_clip=parameters.PSF_FWHM_CLIP,
                       fwhmy_clip=parameters.PSF_FWHM_CLIP, dtype="float64", mask=None, boundaries=None):
        """Simulate a 2D spectrogram of size Nx times Ny.

        Given a set of polynomial parameters defining the chromatic PSF model, a 2D spectrogram
        is produced either summing transverse 1D PSF profiles along the dispersion axis, or
        full 2D PSF profiles.

        Parameters
        ----------
        pixels: array_like
            The pixel array. If `pixels.ndim==1`, ChromaticPSF is evaluated using 1D PSF slices. Otherwise, `pixels`
            must have a shape like (2, Nx, Ny).
        poly_params: array_like
            Parameter array of the model, in the form:
            - Nx first parameters are amplitudes for the Moffat transverse profiles
            - next parameters are polynomial coefficients for all the PSF parameters in the same order
            as in PSF definition, except amplitude.
        fwhmx_clip: int, optional
            Clip PSF evaluation outside fwhmx*FWHM along x axis (default: parameters.PSF_FWHM_CLIP).
        fwhmy_clip: int, optional
            Clip PSF evaluation outside fwhmy*FWHM along y axis (default: parameters.PSF_FWHM_CLIP).
        dtype: str, optional
            Type of the output array (default: 'float64').
        mask: array_like, optional
            Cube of booleans where values are masked (default: None).
        boundaries: dict, optional
            Dictionary of boundaries for fast evaluation with keys ymin, ymax, xmin, xmax (default: None).

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

            >>> output = s.evaluate(s.set_pixels(mode="1D"), poly_params)
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
            output = s.evaluate(s.set_pixels(mode="1D"), poly_params)
            im = plt.imshow(output, origin='lower')
            plt.colorbar(im)
            plt.show()

        2D evaluation:

        .. doctest::

            >>> output = s.evaluate(s.set_pixels(mode="2D"), poly_params)
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
            output = s.evaluate(s.set_pixels(mode="2D"), poly_params)
            im = plt.imshow(output, origin='lower')
            plt.colorbar(im)
            plt.show()

        """
        profile_params = self.from_poly_params_to_profile_params(poly_params, apply_bounds=True)
        if pixels.ndim == 3:
            mode = "2D"
            Ny, Nx = pixels[0].shape
        elif pixels.ndim == 1:
            mode = "1D"
            Ny, Nx = pixels.size, profile_params.shape[0]
        else:
            raise ValueError(f"pixels argument must have shape (2, Nx, Ny) or (Ny). Got {pixels.shape=}.")
        self.params.values = np.asarray(poly_params).astype(float)
        self.psf.apply_max_width_to_bounds(max_half_width=Ny)
        profile_params[:, 1] = np.arange(Nx)  # replace x_c
        output = np.zeros((Ny, Nx), dtype=dtype)
        if mode == "1D":
            for x in range(Nx):
                output[:, x] = self.psf.evaluate(pixels, values=profile_params[x, :])
        elif mode == "2D":
            self.from_profile_params_to_shape_params(profile_params)
            psf_cube = self.build_psf_cube(pixels, profile_params, fwhmx_clip=fwhmx_clip, fwhmy_clip=fwhmy_clip,
                                           dtype=dtype, mask=mask, boundaries=boundaries)
            output = np.sum(psf_cube, axis=0)
        return np.clip(output, 0, self.saturation)

    def build_psf_cube(self, pixels, profile_params, fwhmx_clip=parameters.PSF_FWHM_CLIP,
                       fwhmy_clip=parameters.PSF_FWHM_CLIP, dtype="float64", mask=None, boundaries=None):
        """Build a cube, with one slice per wavelength evaluation which contains the PSF evaluation.

        Parameters
        ----------
        pixels: np.ndarray
            Array of pixels to evaluate ChromaticPSF.
        profile_params: array_like
            ChromaticPSF profile parameters.
        fwhmx_clip: int, optional
            Clip PSF evaluation outside fwhmx*FWHM along x axis (default: parameters.PSF_FWHM_CLIP).
        fwhmy_clip: int, optional
            Clip PSF evaluation outside fwhmy*FWHM along y axis (default: parameters.PSF_FWHM_CLIP).
        dtype: str, optional
            Type of the output array (default: 'float64').
        mask: array_like, optional
            Cube of booleans where values are masked (default: None).
        boundaries: dict, optional
            Dictionary of boundaries for fast evaluation with keys ymin, ymax, xmin, xmax (default: None).

        Returns
        -------
        psf_cube: np.ndarray
            Cube of chromatic PSF evaluations, each slice being a PSF for a given wavelength.

        Examples
        --------

        >>> s = ChromaticPSF(Moffat(), Nx=100, Ny=20, deg=2, saturation=20000)
        >>> profile_params = s.from_poly_params_to_profile_params(s.generate_test_poly_params(), apply_bounds=True)
        >>> profile_params[:, 1] = np.arange(s.Nx)
        >>> psf_cube = s.build_psf_cube(s.set_pixels(mode="2D"), profile_params)
        >>> psf_cube.shape
        (100, 20, 100)
        >>> plt.imshow(psf_cube[20], origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>
        >>> plt.imshow(psf_cube[80], origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>

        .. doctest::
            :hide:

            >>> assert psf_cube.shape == (s.Nx, s.Ny, s.Nx)
            >>> np.argmax(np.sum(psf_cube[20], axis=0))
            20
            >>> np.argmax(np.sum(psf_cube[80], axis=0))
            80

        """
        if pixels.ndim == 3:
            mode = "2D"
            Ny, Nx = pixels[0].shape
        elif pixels.ndim == 1:
            mode = "1D"
            Ny, Nx = pixels.size, profile_params.shape[0]
        else:
            raise ValueError(f"pixels argument must have shape (2, Nx, Ny) or (Ny). Got {pixels.shape=}.")
        psf_cube = np.zeros((len(profile_params), Ny, Nx), dtype=dtype)
        fwhms = self.table["fwhm"]
        for x in range(len(profile_params)):
            xc, yc = profile_params[x, 1:3]
            if xc < - fwhmx_clip * fwhms[x]:
                continue
            if xc > Nx + fwhmx_clip * fwhms[x]:
                break
            if mask is None and not boundaries:
                xmin = max(0, int(xc - max(1*parameters.PIXWIDTH_SIGNAL, fwhmx_clip * fwhms[x])))
                xmax = min(Nx, int(xc + max(1*parameters.PIXWIDTH_SIGNAL, fwhmx_clip * fwhms[x])))
                ymin = max(0, int(yc - max(1*parameters.PIXWIDTH_SIGNAL, fwhmy_clip * fwhms[x])))
                ymax = min(Ny, int(yc + max(1*parameters.PIXWIDTH_SIGNAL, fwhmy_clip * fwhms[x])))
            elif boundaries:
                xmin = boundaries["xmin"][x]
                xmax = boundaries["xmax"][x]
                ymin = boundaries["ymin"][x]
                ymax = boundaries["ymax"][x]
            else:
                maskx = np.any(mask[x], axis=0)
                masky = np.any(mask[x], axis=1)
                xmin = np.argmax(maskx)
                ymin = np.argmax(masky)
                xmax = len(maskx) - np.argmax(maskx[::-1])
                ymax = len(masky) - np.argmax(masky[::-1])
            if mode == "2D":
                psf_cube[x, ymin:ymax, xmin:xmax] = self.psf.evaluate(pixels[:, ymin:ymax, xmin:xmax], values=profile_params[x, :])
            else:
                psf_cube[x, ymin:ymax, x] = self.psf.evaluate(pixels[ymin:ymax], values=profile_params[x, :])
        return psf_cube

    def build_psf_cube_masked(self, pixels, profile_params, fwhmx_clip=parameters.PSF_FWHM_CLIP,
                              fwhmy_clip=parameters.PSF_FWHM_CLIP):
        """Build a boolean cube, with one slice per wavelength evaluation which contains booleans where PSF evaluation is non zero.

        Parameters
        ----------
        pixels: np.ndarray
            Array of pixels to evaluate ChromaticPSF.
        profile_params: array_like
            ChromaticPSF profile parameters.
        fwhmx_clip: int, optional
            Clip PSF evaluation outside fwhmx*FWHM along x axis (default: parameters.PSF_FWHM_CLIP).
        fwhmy_clip: int, optional
            Clip PSF evaluation outside fwhmy*FWHM along y axis (default: parameters.PSF_FWHM_CLIP).

        Returns
        -------
        psf_cube_masked: np.ndarray
            Cube of chromatic masked PSF evaluations, each slice being a PSF for a given wavelength.

        Examples
        --------

        >>> s = ChromaticPSF(Moffat(), Nx=100, Ny=20, deg=2, saturation=20000)
        >>> profile_params = s.from_poly_params_to_profile_params(s.generate_test_poly_params(), apply_bounds=True)
        >>> profile_params[:, 1] = np.arange(s.Nx)
        >>> psf_cube = s.build_psf_cube_masked(s.set_pixels(mode="2D"), profile_params)
        >>> psf_cube.shape
        (100, 20, 100)
        >>> plt.imshow(psf_cube[20], origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>
        >>> plt.imshow(psf_cube[80], origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>

        .. doctest::
            :hide:

            >>> assert psf_cube.shape == (s.Nx, s.Ny, s.Nx)
            >>> np.argmax(np.sum(psf_cube[20], axis=0))
            10
            >>> np.argmax(np.sum(psf_cube[80], axis=0))
            70

        """
        if pixels.ndim == 3:
            mode = "2D"
            Ny, Nx = pixels[0].shape
        elif pixels.ndim == 1:
            mode = "1D"
            Ny, Nx = pixels.size, profile_params.shape[0]
        else:
            raise ValueError(f"pixels argument must have shape (2, Nx, Ny) or (Ny). Got {pixels.shape=}.")
        psf_cube_masked = np.zeros((len(profile_params), Ny, Nx), dtype=bool)
        fwhms = self.table["fwhm"]
        for x in range(len(profile_params)):
            xc, yc = profile_params[x, 1:3]
            if xc < - fwhmx_clip * fwhms[x]:
                continue
            if xc > Nx + fwhmx_clip * fwhms[x]:
                break
            xmin = max(0, int(xc - max(1*parameters.PIXWIDTH_SIGNAL, fwhmx_clip * fwhms[x])))
            xmax = min(Nx, int(xc + max(1*parameters.PIXWIDTH_SIGNAL, fwhmx_clip * fwhms[x])))
            ymin = max(0, int(yc - max(1*parameters.PIXWIDTH_SIGNAL, fwhmy_clip * fwhms[x])))
            ymax = min(Ny, int(yc + max(1*parameters.PIXWIDTH_SIGNAL, fwhmy_clip * fwhms[x])))
            if mode == "2D":
                psf_cube_masked[x, ymin:ymax, xmin:xmax] = self.psf.evaluate(pixels[:, ymin:ymax, xmin:xmax], values=profile_params[x, :]) > 0
            else:
                psf_cube_masked[x, ymin:ymax, x] = self.psf.evaluate(pixels[ymin:ymax], values=profile_params[x, :]) > 0
        return psf_cube_masked

    @staticmethod
    def convolve_psf_cube_masked(psf_cube_masked):
        """Convolve the ChromaticPSF cube of boolean values to enlarge a bit the mask.

        Parameters
        ----------
        psf_cube_masked: np.ndarray
            A ChromaticPSF cube.

        Returns
        -------
        psf_cube_masked: np.ndarray
            Cube of boolean values where `psf_cube` cube is positive, eventually convolved.

        Examples
        --------

        >>> s = ChromaticPSF(Moffat(), Nx=100, Ny=20, deg=2, saturation=20000)
        >>> profile_params = s.from_poly_params_to_profile_params(s.generate_test_poly_params(), apply_bounds=True)
        >>> profile_params[:, 1] = np.arange(s.Nx)
        >>> psf_cube_masked = s.build_psf_cube_masked(s.set_pixels(mode="2D"), profile_params)
        >>> psf_cube_masked = s.convolve_psf_cube_masked(psf_cube_masked)
        >>> psf_cube_masked.dtype
        dtype('bool')
        >>> psf_cube_masked.shape
        (100, 20, 100)
        >>> plt.imshow(psf_cube_masked[20], origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>
        >>> plt.imshow(psf_cube_masked[80], origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>

        .. doctest::
            :hide:

            >>> assert psf_cube_masked.shape == (s.Nx, s.Ny, s.Nx)
            >>> assert np.sum(psf_cube_masked[20], axis=0)[20]
            >>> assert np.sum(psf_cube_masked[80], axis=0)[80]

        """
        wl_size = psf_cube_masked.shape[0]
        flat_spectrogram = np.sum(psf_cube_masked.reshape(wl_size, psf_cube_masked[0].size), axis=0)
        mask = flat_spectrogram == 0  # < 1e-2 * np.max(flat_spectrogram)
        mask = mask.reshape(psf_cube_masked[0].shape)
        kernel = np.ones((3, psf_cube_masked.shape[2]//10))  # enlarge a bit more the edges of the mask
        mask = convolve2d(mask, kernel, 'same').astype(bool)
        for k in range(wl_size):
            psf_cube_masked[k] *= ~mask
        return psf_cube_masked

    @staticmethod
    def get_boundaries(psf_cube_masked):
        """Compute the ChromaticPSF computation boundaries, as a dictionnary of integers giving
        the `"xmin"`, `"xmax"`, `"ymin"` and `"ymax"` edges where to compute the PSF for each wavelength.
        True regions are rectangular after this operation. The `psf_cube_masked` cube is updated accordingly and returned.

        Parameters
        ----------
        psf_cube_masked: np.ndarray
            Cube of boolean values where `psf_cube` cube is positive, eventually convolved.

        Returns
        -------
        boundaries: dict
            The dictionnary of PSF edges per wavelength.
        psf_cube_masked: np.ndarray
            Updated cube of boolean values where `psf_cube` cube is positive, eventually convolved.


        Examples
        --------

        >>> s = ChromaticPSF(Moffat(), Nx=100, Ny=20, deg=2, saturation=20000)
        >>> profile_params = s.from_poly_params_to_profile_params(s.generate_test_poly_params(), apply_bounds=True)
        >>> profile_params[:, 1] = np.arange(s.Nx)
        >>> psf_cube_masked = s.build_psf_cube_masked(s.set_pixels(mode="2D"), profile_params)
        >>> psf_cube_masked = s.convolve_psf_cube_masked(psf_cube_masked)
        >>> boundaries, psf_cube_masked = s.get_boundaries(psf_cube_masked)
        >>> boundaries["xmin"].shape
        (100,)
        >>> psf_cube_masked.shape
        (100, 20, 100)
        >>> plt.imshow(psf_cube_masked[20], origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>
        >>> plt.imshow(psf_cube_masked[80], origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>

        .. doctest::
            :hide:

            >>> assert psf_cube_masked.shape == (s.Nx, s.Ny, s.Nx)
            >>> assert np.sum(psf_cube_masked[20], axis=0)[20]
            >>> assert np.sum(psf_cube_masked[80], axis=0)[80]
            >>> assert np.all(boundaries["xmin"] < boundaries["xmax"])
            >>> assert np.all(boundaries["ymin"] < boundaries["ymax"])

        """
        wl_size = psf_cube_masked.shape[0]
        boundaries = {"xmin": np.zeros(wl_size, dtype=int), "xmax": np.zeros(wl_size, dtype=int),
                      "ymin": np.zeros(wl_size, dtype=int), "ymax": np.zeros(wl_size, dtype=int)}
        for k in range(wl_size):
            maskx = np.any(psf_cube_masked[k], axis=0)
            masky = np.any(psf_cube_masked[k], axis=1)
            if np.sum(maskx) > 0 and np.sum(masky) > 0:
                xmin, xmax = int(np.argmax(maskx)), int(len(maskx) - np.argmax(maskx[::-1]))
                ymin, ymax = int(np.argmax(masky)), int(len(masky) - np.argmax(masky[::-1]))
            else:
                xmin, xmax = -1, -1
                ymin, ymax = -1, -1
            boundaries["xmin"][k] = xmin
            boundaries["xmax"][k] = xmax
            boundaries["ymin"][k] = ymin
            boundaries["ymax"][k] = ymax
            psf_cube_masked[k, ymin:ymax, xmin:xmax] = True
        return boundaries, psf_cube_masked

    @staticmethod
    def get_sparse_indices(psf_cube_masked):
        """Methods that returns the indices to build sparse matrices from `psf_cube_masked`.

        Parameters
        ----------
        psf_cube_masked: np.ndarray
            Cube of boolean values where `psf_cube` cube is positive, eventually convolved.

        Returns
        -------
        psf_cube_sparse_indices: list
            List of sparse indices per wavelength.
        M_sparse_indices: np.ndarray
            Sparse indices for the integrated matrix model :math:`mathbf{M}`.

        Examples
        --------

        >>> s = ChromaticPSF(Moffat(), Nx=100, Ny=20, deg=2, saturation=20000)
        >>> profile_params = s.from_poly_params_to_profile_params(s.generate_test_poly_params(), apply_bounds=True)
        >>> profile_params[:, 1] = np.arange(s.Nx)
        >>> psf_cube_masked = s.build_psf_cube_masked(s.set_pixels(mode="2D"), profile_params)
        >>> psf_cube_masked = s.convolve_psf_cube_masked(psf_cube_masked)
        >>> psf_cube_sparse_indices, M_sparse_indices = s.get_sparse_indices(psf_cube_masked)
        >>> M_sparse_indices.shape
        (72000,)
        >>> len(psf_cube_sparse_indices)
        100
        """
        wl_size = psf_cube_masked.shape[0]
        psf_cube_sparse_indices = [np.where(psf_cube_masked[k].ravel() > 0)[0] for k in range(wl_size)]
        M_sparse_indices = np.concatenate(psf_cube_sparse_indices)
        return psf_cube_sparse_indices, M_sparse_indices

    def build_sparse_M(self, pixels, profile_params, M_sparse_indices, boundaries, dtype="float32"):
        r"""
        Compute the sparse model matrix :math:`\mathbf{M}`.
        Given a vector of amplitudes :math:`\mathbf{A}`, spectrogram model is:
        .. math::

            \mathbf{I} = \mathbf{M} \cdot \mathbf{A}.

        Parameters
        ----------
        pixels: np.ndarray
            Array of pixels to evaluate ChromaticPSF.
        profile_params: array_like
            ChromaticPSF profile parameters.
        M_sparse_indices: array_like
            Array of indices where each element gives the sparse indices for a slice of the ChromaticPSF cube.
        boundaries: dict
            Dictionary of boundaries for fast evaluation with keys ymin, ymax, xmin, xmax .
        dtype: str, optional
            Type of the output array (default: 'float32').

        Returns
        -------
        M: np.ndarray
            The model matrix :math:`\mathbf{M}`.

        Examples
        --------

        >>> s = ChromaticPSF(Moffat(), Nx=100, Ny=20, deg=2, saturation=20000)
        >>> profile_params = s.from_poly_params_to_profile_params(s.generate_test_poly_params(), apply_bounds=True)
        >>> profile_params[:, 0] = np.ones(s.Nx)  # normalized PSF
        >>> profile_params[:, 1] = np.arange(s.Nx)  # PSF x_c positions

        2D case

        >>> psf_cube_masked = s.build_psf_cube_masked(s.set_pixels(mode="2D"), profile_params)
        >>> psf_cube_masked = s.convolve_psf_cube_masked(psf_cube_masked)
        >>> boundaries, psf_cube_masked = s.get_boundaries(psf_cube_masked)
        >>> psf_cube_sparse_indices, M_sparse_indices = s.get_sparse_indices(psf_cube_masked)
        >>> M = s.build_sparse_M(s.set_pixels(mode="2D"), profile_params, M_sparse_indices, boundaries, dtype="float32")
        >>> M.shape
        (2000, 100)
        >>> M.dtype
        dtype('float32')
        >>> plt.imshow((M @ np.ones(s.Nx)).reshape((s.Ny, s.Nx)), origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>

        .. doctest::
            :hide:

            >>> assert M.shape == (s.Ny * s.Nx, s.Nx)

        1D case

        >>> psf_cube_masked = s.build_psf_cube_masked(s.set_pixels(mode="1D"), profile_params)
        >>> psf_cube_masked = s.convolve_psf_cube_masked(psf_cube_masked)
        >>> boundaries, psf_cube_masked = s.get_boundaries(psf_cube_masked)
        >>> psf_cube_sparse_indices, M_sparse_indices = s.get_sparse_indices(psf_cube_masked)
        >>> M = s.build_sparse_M(s.set_pixels(mode="1D"), profile_params, M_sparse_indices, boundaries, dtype="float32")
        >>> M.shape
        (2000, 100)
        >>> M.dtype
        dtype('float32')
        >>> plt.imshow((M @ np.ones(s.Nx)).reshape((s.Ny, s.Nx)), origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>

        .. doctest::
            :hide:

            >>> assert M.shape == (s.Ny * s.Nx, s.Nx)

        """
        if pixels.ndim == 3:
            mode = "2D"
            Ny, Nx = pixels[0].shape
        elif pixels.ndim == 1:
            mode = "1D"
            Ny, Nx = pixels.size, profile_params.shape[0]
        else:
            raise ValueError(f"pixels argument must have shape (2, Nx, Ny) or (Ny). Got {pixels.shape=}.")
        if Nx != profile_params.shape[0]:
            raise ValueError(f"Number of pixels along x axis must be same as profile_params table length. "
                             f"Got {Nx=} and {profile_params.shape}.")
        sparse_psf_cube = np.zeros(M_sparse_indices.size, dtype=dtype)
        indptr = np.zeros(len(profile_params)+1, dtype=int)
        for x in range(len(profile_params)):
            if mode == "2D":
                indptr[x + 1] = (boundaries["xmax"][x] - boundaries["xmin"][x]) * (boundaries["ymax"][x] - boundaries["ymin"][x]) + indptr[x]
                if boundaries["xmin"][x] < 0:
                    continue
                sparse_psf_cube[indptr[x]:indptr[x+1]] = self.psf.evaluate(pixels[:, boundaries["ymin"][x]:boundaries["ymax"][x],
                                                                                 boundaries["xmin"][x]:boundaries["xmax"][x]],
                                                                           values=profile_params[x, :]).ravel()
            else:
                indptr[x + 1] = boundaries["ymax"][x] - boundaries["ymin"][x] + indptr[x]
                sparse_psf_cube[indptr[x]:indptr[x+1]] = self.psf.evaluate(pixels[boundaries["ymin"][x]:boundaries["ymax"][x]],
                                                                           values=profile_params[x, :])
        return sparse.csr_matrix((sparse_psf_cube, M_sparse_indices, indptr), shape=(len(profile_params), Ny*Nx), dtype=dtype).T

    def build_psf_jacobian(self, pixels, profile_params, psf_cube_sparse_indices, boundaries, dtype="float32"):
        r"""
        Compute the Jacobian matrix :math:`\mathbf{J}` of a ChromaticPSF model, with analytical derivatives.
        Amplitude parameters :math:`\mathbf{A}` are excluded, only PSF shape and position parameters :math:`\theta` are included.

        .. math::

            \mathbf{J} = \frac{\partial \mathbf{M}}{\partial \theta} \cdot \mathbf{A}.

        Parameters
        ----------
        pixels: np.ndarray
            Array of pixels to evaluate ChromaticPSF.
        profile_params: array_like
            ChromaticPSF profile parameters.
        psf_cube_sparse_indices: array_like
            Array of indices where each element gives the sparse indices for a slice of the ChromaticPSF cube.
        boundaries: dict
            Dictionary of boundaries for fast evaluation with keys ymin, ymax, xmin, xmax .
        dtype: str, optional
            Type of the output array (default: 'float32').

        Returns
        -------
        J: np.ndarray
            The Jacobian matrix math:`\mathbf{J}`.

        Examples
        --------

        >>> s = ChromaticPSF(Moffat(), Nx=100, Ny=20, deg=2, saturation=20000)
        >>> profile_params = s.from_poly_params_to_profile_params(s.generate_test_poly_params(), apply_bounds=True)
        >>> profile_params[:, 0] = np.ones(s.Nx)  # normalized PSF
        >>> profile_params[:, 1] = np.arange(s.Nx)  # PSF x_c positions
        >>> psf_cube_masked = s.build_psf_cube_masked(s.set_pixels(mode="2D"), profile_params)
        >>> psf_cube_masked = s.convolve_psf_cube_masked(psf_cube_masked)
        >>> boundaries, psf_cube_masked = s.get_boundaries(psf_cube_masked)
        >>> psf_cube_sparse_indices, M_sparse_indices = s.get_sparse_indices(psf_cube_masked)
        >>> s.params.fixed[s.Nx:s.Nx+s.deg+1] = [True] * (s.deg+1)  # fix all x_c parameters
        >>> J = s.build_psf_jacobian(s.set_pixels(mode="2D"), profile_params, psf_cube_sparse_indices, boundaries, dtype="float32")
        >>> J.shape
        (13, 2000)
        >>> J.dtype
        dtype('float32')
        >>> plt.imshow(J[s.params.get_index("y_c_0")-s.Nx].reshape((s.Ny, s.Nx)), origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>

        .. doctest::
            :hide:

            >>> assert J.shape == (len(s.params) - s.Nx, s.Ny * s.Nx)

        """
        if pixels.ndim == 3:
            mode = "2D"
            Ny, Nx = pixels[0].shape
            J = np.zeros((self.n_poly_params - self.Nx, Ny * Nx), dtype=dtype)
        elif pixels.ndim == 1:
            mode = "1D"
            Ny, Nx = pixels.size, profile_params.shape[0]
            J = np.zeros((self.n_poly_params - self.Nx, Ny * Nx), dtype=dtype)
        else:
            raise ValueError(f"pixels argument must have shape (2, Nx, Ny) or (Ny). Got {pixels.shape=}.")
        if Nx != profile_params.shape[0]:
            raise ValueError(f"Number of pixels along x axis must be same as profile_params table length. "
                             f"Got {Nx=} and {profile_params.shape}.")
        leg_pixels = np.linspace(-1, 1, Nx)
        legs = np.zeros((self.n_poly_params-Nx, Nx), dtype=dtype)
        ip = 0
        repeats = []
        for ipsf, label in enumerate(self.psf.params.labels):
            if label == "amplitude": continue
            nparams = self.degrees[label]+1
            repeats.append(nparams)
            for k in range(nparams):
                coeffs = np.eye(1, nparams, k)[0]
                legs[ip] = np.polynomial.legendre.legval(leg_pixels, coeffs).astype(dtype)
                ip += 1
        for ip, label in enumerate(self.psf.params.labels):
            if "amplitude" in label:  # skip computation of ChromaticPSF jacobian for amplitude parameters
                self.psf.params.fixed[ip] = True
            else:  # check if all ChromaticPSF parameters related to PSF parameter ip are fixed
                indices = np.array([k for k in range(len(self.params)) if label in self.params.labels[k]])
                self.psf.params.fixed[ip] = np.all(np.array(self.params.fixed)[indices])
        for x in range(Nx):
            if mode == "2D":
                Jpsf = self.psf.jacobian(pixels[:, boundaries["ymin"][x]:boundaries["ymax"][x],
                                                   boundaries["xmin"][x]:boundaries["xmax"][x]],
                                         profile_params[x, :], analytical=True)
            else:
                Jpsf = self.psf.jacobian(pixels[boundaries["ymin"][x]:boundaries["ymax"][x]], profile_params[x, :], analytical=True)

            J[:, psf_cube_sparse_indices[x]] += np.repeat(Jpsf[1:], repeats, axis=0) * legs[:, x, None]  # Jpsf[1:] excludes amplitude
        return J

    def build_sparse_dM(self, pixels, profile_params, M_sparse_indices, boundaries, dtype="float32"):
        r"""
        Compute the partial derivatives of the model matrix :math:`\mathbf{M}`, with analytical derivatives.
        Amplitude parameters :math:`\mathbf{A}` are excluded, only PSF shape and position parameters :math:`\theta` are included.

        Parameters
        ----------
        pixels: np.ndarray
            Array of pixels to evaluate ChromaticPSF.
        profile_params: array_like
            ChromaticPSF profile parameters.
        M_sparse_indices: array_like
            Sparse indices of the model matrix :math:`\mathbf{M}`.
        boundaries: dict
            Dictionary of boundaries for fast evaluation with keys ymin, ymax, xmin, xmax .
        dtype: str, optional
            Type of the output array (default: 'float32').

        Returns
        -------
        dM: list
            List of sparse matrices :math:`\partial \mathbf{M}/\partial \theta`

        Examples
        --------

        >>> s = ChromaticPSF(Moffat(), Nx=100, Ny=20, deg=2, saturation=20000)
        >>> profile_params = s.from_poly_params_to_profile_params(s.generate_test_poly_params(), apply_bounds=True)
        >>> profile_params[:, 0] = np.ones(s.Nx)  # normalized PSF
        >>> profile_params[:, 1] = np.arange(s.Nx)  # PSF x_c positions
        >>> psf_cube_masked = s.build_psf_cube_masked(s.set_pixels(mode="2D"), profile_params)
        >>> psf_cube_masked = s.convolve_psf_cube_masked(psf_cube_masked)
        >>> boundaries, psf_cube_masked = s.get_boundaries(psf_cube_masked)
        >>> psf_cube_sparse_indices, M_sparse_indices = s.get_sparse_indices(psf_cube_masked)
        >>> s.params.fixed[s.Nx:s.Nx+s.deg+1] = [True] * (s.deg+1)  # fix all x_c parameters
        >>> dM = s.build_sparse_dM(s.set_pixels(mode="2D"), profile_params, M_sparse_indices, boundaries, dtype="float32")
        >>> len(dM), dM[0].shape
        (13, (2000, 100))
        >>> dM[0].dtype
        dtype('float32')
        >>> plt.imshow((dM[s.params.get_index("y_c_0")-s.Nx] @ np.ones(s.Nx)).reshape((s.Ny, s.Nx)), origin="lower"); plt.show()  # doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>

        .. doctest::
            :hide:

            >>> assert len(dM) == len(s.params) - s.Nx
            >>> assert dM[0].shape == (s.Ny * s.Nx, s.Nx)
            >>> J = s.build_psf_jacobian(s.set_pixels(mode="2D"), profile_params, psf_cube_sparse_indices, boundaries, dtype="float32")
            >>> assert np.allclose(dM[s.params.get_index("y_c_0")-s.Nx] @ np.ones(s.Nx), J[s.params.get_index("y_c_0")-s.Nx])

        """
        if pixels.ndim == 3:
            mode = "2D"
            Ny, Nx = pixels[0].shape
        elif pixels.ndim == 1:
            mode = "1D"
            Ny, Nx = pixels.size, profile_params.shape[0]
        else:
            raise ValueError(f"pixels argument must have shape (2, Nx, Ny) or (Ny). Got {pixels.shape=}.")
        if Nx != profile_params.shape[0]:
            raise ValueError(f"Number of pixels along x axis must be same as profile_params table length. "
                             f"Got {Nx=} and {profile_params.shape}.")
        leg_pixels = np.linspace(-1, 1, Nx)
        legs = np.zeros((self.n_poly_params-Nx, Nx), dtype=dtype)
        ip = 0
        repeats = []
        for ipsf, label in enumerate(self.psf.params.labels):
            if label == "amplitude": continue
            nparams = self.degrees[label]+1
            repeats.append(nparams)
            for k in range(nparams):
                # psf_index.append(ipsf)
                coeffs = np.eye(1, nparams, k)[0]
                legs[ip] = np.polynomial.legendre.legval(leg_pixels, coeffs).astype(dtype)
                ip += 1
        sparse_J = np.zeros((self.n_poly_params - self.Nx, M_sparse_indices.size), dtype=dtype)
        indptr = np.zeros(Nx+1, dtype=int)
        for x in range(Nx):
            if mode == "2D":
                Jpsf = self.psf.jacobian(pixels[:, boundaries["ymin"][x]:boundaries["ymax"][x],
                                                   boundaries["xmin"][x]:boundaries["xmax"][x]],
                                         profile_params[x, :], analytical=True)
            else:
                Jpsf = self.psf.jacobian(pixels[boundaries["ymin"][x]:boundaries["ymax"][x]], profile_params[x, :], analytical=True)
            indptr[x+1] = (boundaries["xmax"][x]-boundaries["xmin"][x])*(boundaries["ymax"][x]-boundaries["ymin"][x]) + indptr[x]
            if boundaries["xmin"][x] < 0:
                continue
            sparse_J[:, indptr[x]:indptr[x+1]] += np.repeat(Jpsf[1:], repeats, axis=0) * legs[:, x, None]
        dM = [sparse.csr_matrix((sparse_J[ip], M_sparse_indices, indptr), shape=(len(profile_params), pixels[0].size), dtype=dtype).T for ip in range(sparse_J.shape[0])]
        return dM

    def fill_table_with_profile_params(self, profile_params):
        """
        Fill the table with the profile parameters.

        Parameters
        ----------
        profile_params: np.ndarray
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
        for k, name in enumerate(self.psf.params.labels):
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

    def from_profile_params_to_poly_params(self, profile_params, indices=None):
        """
        Transform the profile_params array into a set of parameters for the chromatic PSF parameterisation.
        Fit Legendre polynomial functions across the pixels for each PSF parameters.
        The order of the polynomial functions is given by the self.degrees array.

        Parameters
        ----------
        profile_params: array
            a Nx * len(self.psf.param_names) numpy array containing the PSF parameters as a function of pixels.
        indices: array_like, optional
            Array of integer indices or boolean values that selects values in profile_params for the polynomial fits.
            If None every profile_params rows are used (default: None)

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
        >>> data = s.evaluate(s.set_pixels(mode="1D"), poly_params_test)
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
        if indices is None:
            indices = np.full(len(self.table), True)
        poly_params = np.array([])
        amplitude = None
        for k, name in enumerate(self.psf.params.labels):
            if name == 'amplitude':
                amplitude = profile_params[:, k]
                poly_params = np.concatenate([poly_params, amplitude])
        if amplitude is None:
            self.my_logger.warning('\n\tAmplitude array not initialized. '
                                   'Polynomial fit for shape parameters will be unweighted.')
            
        pixels = np.linspace(-1, 1, len(self.table))[indices]
        for k, name in enumerate(self.psf.params.labels):
            delta = 0
            if name != 'amplitude':
                weights = np.copy(amplitude)[indices]
                if name == 'x_c':
                    delta = self.x0
                if name == 'y_c':
                    delta = self.y0
                fit = np.polynomial.legendre.legfit(pixels, profile_params[indices, k] - delta,
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

            >>> assert(profile_params.shape == (s.chromatic_psf.Nx, len(s.chromatic_psf.psf.params.labels)))
            >>> assert not np.all(np.isclose(profile_params, np.zeros_like(profile_params)))
        """
        profile_params = np.zeros((len(self.table), len(self.psf.params.labels)))
        for k, name in enumerate(self.psf.params.labels):
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
        >>> data = s.evaluate(s.set_pixels(mode="1D"), poly_params_test)
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        From the polynomial parameters to the profile parameters:

        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test, apply_bounds=True)

        ..  doctest::
            :hide:

            >>> assert np.allclose(profile_params[0], [10, 0, 50, 5, 2, -5e-3, 1, 8e3], rtol=1e-3, atol=1e-3)

        From the profile parameters to the polynomial parameters:

        >>> profile_params = s.from_profile_params_to_poly_params(profile_params)

        ..  doctest::
            :hide:

            >>> assert np.allclose(profile_params, poly_params_test)

        From the polynomial parameters to the profile parameters without Moffat amplitudes:

        >>> profile_params = s.from_poly_params_to_profile_params(poly_params_test[100:])

        ..  doctest::
            :hide:

            >>> assert np.allclose(profile_params[0], [1, 0, 50, 5, 2, 0, 1, 8e3])

        """
        length = len(self.table)
        pixels = np.linspace(-1, 1, length)
        profile_params = np.zeros((length, len(self.psf.params.labels)))
        shift = 0
        for k, name in enumerate(self.psf.params.labels):
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
            for k, name in enumerate(self.psf.params.labels):
                profile_params[profile_params[:, k] <= self.psf.params.bounds[k][0], k] = self.psf.params.bounds[k][0]
                profile_params[profile_params[:, k] > self.psf.params.bounds[k][1], k] = self.psf.params.bounds[k][1]
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
        pixel_x = np.arange(0, self.Nx, parameters.PSF_PIXEL_STEP_TRANSVERSE_FIT, dtype=int)
        fwhms = np.zeros_like(pixel_x, dtype=float)
        # oversampling for precise computation of the PSF
        # pixels.shape = (2, Ny, Nx): self.pixels[1<-y, :, 0<-first pixel value column]
        # TODO: account for rotation ad projection effects is PSF is not round
        pixel_eval = np.arange(self.pixels[1, 0, 0], self.pixels[1, -1, 0], 0.5, dtype=np.float32)
        for ix, x in enumerate(pixel_x):
            p = profile_params[x, :]
            # compute FWHM transverse to dispersion axis (assuming revolution symmetry of the PSF)
            out = self.psf.evaluate(pixel_eval, values=p)
            fwhms[ix] = compute_fwhm(pixel_eval, out, center=p[2], minimum=0, epsilon=1e-2)
        # clean fwhm bad points
        mask = np.logical_and(fwhms > 1, fwhms < self.Ny // 2)  # more than 1 pixel or less than window
        self.table['fwhm'] = interp1d(pixel_x[mask], fwhms[mask], kind="linear",
                                      bounds_error=False, fill_value="extrapolate")(np.arange(self.Nx))
        self.table['flux_integral'] = profile_params[:, 0]  # if MoffatGauss1D normalized

    def set_bounds(self):
        """
        This function returns an array of bounds for PSF polynomial parameters (no amplitude ones).
        It is very touchy, change the values with caution !

        Returns
        -------
        bounds: list
            2D array containing the pair of bounds for each polynomial parameters.

        """
        bounds = [[], []]
        for k, name in enumerate(self.psf.params.labels):
            tmp_bounds = [[-np.inf] * (1 + self.degrees[name]), [np.inf] * (1 + self.degrees[name])]
            if name == "saturation":
                tmp_bounds = [[0], [2 * self.saturation]]
            elif name == "amplitude":
                continue
            bounds[0] += tmp_bounds[0]
            bounds[1] += tmp_bounds[1]
        return list(np.array(bounds).T)

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
        for k, name in enumerate(self.psf.params.labels):
            tmp_bounds = [[-np.inf] * (1 + self.degrees[name]), [np.inf] * (1 + self.degrees[name])]
            if name == "saturation":
                if data is not None:
                    tmp_bounds = [[0.1 * np.max(data)], [2 * self.saturation]]
                else:
                    tmp_bounds = [[0], [2 * self.saturation]]
            elif name == "amplitude":
                continue
            bounds[0] += tmp_bounds[0]
            bounds[1] += tmp_bounds[1]
        return np.array(bounds).T

    def get_algebraic_distance_along_dispersion_axis(self, shift_x=0, shift_y=0):
        return np.asarray(np.sign(self.table['Dx']) *
                          np.sqrt((self.table['Dx'] - shift_x) ** 2 + (self.table['Dy_disp_axis'] - shift_y) ** 2))

    def update(self, psf_poly_params, x0, y0, angle, plot=False, apply_bounds=True):
        profile_params = self.from_poly_params_to_profile_params(psf_poly_params, apply_bounds=apply_bounds)
        self.fill_table_with_profile_params(profile_params)
        Dx = np.arange(self.Nx) - x0  # distance in (x,y) spectrogram frame for column x
        self.table["Dx"] = Dx
        self.table['Dy_disp_axis'] = np.tan(angle * np.pi / 180) * self.table['Dx']
        self.table['Dy'] = np.copy(self.table['y_c']) - y0
        if plot:
            self.plot_summary()
        return profile_params

    def plot_summary(self, truth=None):
        fig, ax = plt.subplots(1, 1, sharex='all', figsize=(7, 4))
        PSF_models = []
        PSF_truth = []
        if truth is not None:
            truth.psf.apply_max_width_to_bounds(max_half_width=self.Ny)
            PSF_truth = truth.from_poly_params_to_profile_params(truth.params.values, apply_bounds=True)
            PSF_truth[:, 1] = np.arange(self.Nx)  # replace x_c
        all_pixels = np.arange(self.profile_params.shape[0])
        for i, name in enumerate(self.psf.params.labels):
            legs = [self.params.values[k] for k in range(self.params.ndim) if name in self.params.labels[k]]
            pval = np.polynomial.legendre.leg2poly(legs)[::-1]
            delta = 0
            if name == 'x_c':
                delta = self.x0
            if name == 'y_c':
                delta = self.y0
            PSF_models.append(np.polyval(pval, rescale_x_to_legendre(all_pixels)) + delta)
        for i, name in enumerate(self.psf.params.labels):
            p = ax.plot(all_pixels, self.profile_params[:, i], marker='+', linestyle='none')
            ax.plot(all_pixels[self.fitted_pixels], self.profile_params[self.fitted_pixels, i], label=name,
                       marker='o', linestyle='none', color=p[0].get_color())
            if i > 0:
                ax.plot(all_pixels, PSF_models[i], color=p[0].get_color())
            if truth is not None:
                ax.plot(all_pixels, PSF_truth[:, i], color=p[0].get_color(), linestyle='--')
        ax.set_xlabel('X pixels')
        ax.set_ylabel('PSF parameters')
        ax.grid()
        ax.set_yscale('symlog', linthresh=10)
        ax.legend()
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
        >>> s0.params.values = s0.generate_test_poly_params()
        >>> saturation = s0.params.values[-1]
        >>> data = s0.evaluate(s0.set_pixels(mode="1D"), s0.params.values)
        >>> bgd = 10*np.ones_like(data)
        >>> xx, yy = np.meshgrid(np.arange(s0.Nx), np.arange(s0.Ny))
        >>> bgd += 1000*np.exp(-((xx-20)**2+(yy-10)**2)/(2*2))
        >>> data += bgd
        >>> data_errors = np.sqrt(data+1)

        Extract the background:

        >>> bgd_model_func, _, _ = extract_spectrogram_background_sextractor(data, data_errors, ws=[30,50])

        Fit the transverse profile:

        >>> s = ChromaticPSF(psf, Nx=100, Ny=100, deg=4, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50], pixel_step=5,
        ... bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False, sigma_clip=5)
        >>> s.plot_summary(truth=s0)

        ..  doctest::
            :hide:

            >>> truth_profile_params = s0.from_poly_params_to_profile_params(s0.params.values)
            >>> truth_profile_params[:, 1] = np.arange(s0.Nx)  # replace x_c
            >>> for i in range(1, s.profile_params.shape[1]):
            ...     print(s.psf.params.labels[i], np.isclose(np.mean(s.profile_params[1:-1, i]), np.mean(truth_profile_params[:, i]), rtol=5e-2))
            x_c True
            y_c True
            gamma True
            alpha True
            eta_gauss True
            stddev True
            saturation True
            >>> assert(not np.any(np.isclose(s.table['flux_sum'][3:6], np.zeros(s.Nx)[3:6], rtol=1e-3)))

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
        guess = np.copy(psf.values_default).astype(float)
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
        initial_bounds = np.copy(psf.params.bounds)
        bounds = np.copy(psf.params.bounds)
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
        self.fitted_pixels = np.full(Nx, False)
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
            if signal_sum < 3 * np.nanstd(signal[bgd_index]):
                continue
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
            psf.params.values = guess
            psf.params.bounds = bounds
            w = PSFFitWorkspace(psf, signal, data_errors=err[:, x], bgd_model_func=None,
                                live_fit=False, verbose=False, jacobian_analytical=True)
            try:
                run_minimisation_sigma_clipping(w, method="newton", sigma_clip=sigma_clip, niter_clip=1, verbose=False)
            except:
                pass
            best_fit = w.params.values
            # It is better not to propagate the guess to further pixel columns
            # otherwise fit_chromatic_psf1D is more likely to get trapped in a local minimum
            # Randomness of the slice fit is better :
            # guess = best_fit
            self.profile_params[x, :] = best_fit
            # TODO: propagate amplitude uncertainties from Newton fit
            self.table['flux_err'][x] = np.sqrt(np.sum(err[:, x] ** 2))
            self.table['flux_sum'][x] = np.sum(signal)
            if live_fit and parameters.DISPLAY:  # pragma: no cover
                if not np.any(np.isnan(best_fit[0])):
                    w.live_fit = True
                    w.plot_fit()
            self.fitted_pixels[x] = True
        # interpolate the skipped pixels with splines
        all_pixels = np.arange(Nx)
        # xp = np.array(sorted(set(list(pixel_range))))
        xp = all_pixels[self.fitted_pixels]
        # self.fitted_pixels = xp
        for i in range(len(self.psf.params.labels)):
            yp = self.profile_params[xp, i]
            self.profile_params[:, i] = interp1d(xp, yp, kind='cubic', fill_value='extrapolate', bounds_error=False)(all_pixels)
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
        # keep only brightest transverse profiles
        selected_pixels = self.profile_params[:, 0] > 0.1 * np.max(self.profile_params[:, 0])
        # then keep only profiles with first shape parameter (index=3) are not too deviant from its median value
        selected_pixels = selected_pixels & (np.abs(self.profile_params[:, 3]) < 5 * np.median(self.profile_params[:, 3]))
        self.params.values = self.from_profile_params_to_poly_params(self.profile_params, indices=selected_pixels)
        self.from_profile_params_to_shape_params(self.profile_params)
        self.cov_matrix = np.diag(1 / np.array(self.table['flux_err']) ** 2)
        psf.params.bounds = initial_bounds

    def fit_chromatic_psf(self, data, bgd_model_func=None, data_errors=None, mode="1D", analytical=True,
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

        >>> psf = Moffat(clip=False)
        >>> s0 = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=10000000)
        >>> params = s0.generate_test_poly_params()
        >>> params[:s0.Nx] *= 1
        >>> s0.params.values = params
        >>> saturation = params[-1]
        >>> data = s0.evaluate(s0.set_pixels(mode="2D"), params)
        >>> bgd = 10*np.ones_like(data)
        >>> data += bgd
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(np.abs(data+1))

        Extract the background:

        >>> bgd_model_func, _, _ = extract_spectrogram_background_sextractor(data, data_errors, ws=[30,50])

        Propagate background uncertainties:

        >>> data_errors = np.sqrt(data_errors**2 + bgd_model_func(np.arange(s0.Nx), np.arange(s0.Ny)))

        Estimate the first guess values:

        >>> s = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
        ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)
        >>> s.plot_summary(truth=s0)
        >>> amplitude_residuals = [ [s0.params.values[:s0.Nx], np.array(s.table["amplitude"])-s0.params.values[:s0.Nx],
        ... np.array(s.table['amplitude'] * s.table['flux_err'] / s.table['flux_sum'])] ]

        Fit the data using the transverse 1D PSF model only:

        >>> w = s.fit_chromatic_psf(data, mode="1D", data_errors=data_errors, bgd_model_func=bgd_model_func,
        ... amplitude_priors_method="noprior", verbose=True)
        >>> s.plot_summary(truth=s0)
        >>> amplitude_residuals.append([s0.params.values[:s0.Nx], w.amplitude_params-s0.params.values[:s0.Nx],
        ... w.amplitude_params_err])

        ..  doctest::
            :hide:

            >>> residuals = [(w.data[x]-w.model[x])/w.err[x] for x in range(w.Nx)]
            >>> assert w.costs[-1] /(w.Nx*w.Ny) < 1.5
            >>> assert np.abs(np.mean(residuals)) < 0.15
            >>> assert np.std(residuals) < 1.2

        Fit the data using the full 2D PSF model

        >>> parameters.PSF_FIT_REG_PARAM = 0.002
        >>> w = s.fit_chromatic_psf(data, mode="2D", data_errors=data_errors, bgd_model_func=bgd_model_func,
        ... amplitude_priors_method="psf1d", verbose=True, analytical=True)
        >>> s.plot_summary(truth=s0)
        >>> amplitude_residuals.append([s0.params.values[:s0.Nx], w.amplitude_params-s0.params.values[:s0.Nx],
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

            >>> residuals = (w.data.flatten()-w.model.flatten())/w.err.flatten()
            >>> assert w.costs[-1] /(w.Nx*w.Ny) < 1.2
            >>> assert np.abs(np.mean(residuals)) < 0.15
            >>> assert np.std(residuals) < 1.2
            >>> assert np.abs(np.mean((w.amplitude_params - s0.params.values[:s0.Nx])/w.amplitude_params_err)) < 0.5

        """
        if mode == "1D":
            w = ChromaticPSFFitWorkspace(self, data, data_errors=data_errors, mode=mode, bgd_model_func=bgd_model_func,
                                        amplitude_priors_method=amplitude_priors_method, verbose=verbose,
                                        live_fit=live_fit, analytical=analytical)
            run_minimisation(w, method="newton", ftol=1 / (w.Nx * w.Ny), xtol=1e-6, niter=50, with_line_search=True)
        elif mode == "2D":
            # first shot to set the mask
            w = ChromaticPSFFitWorkspace(self, data, data_errors=data_errors, mode=mode, bgd_model_func=bgd_model_func,
                                         amplitude_priors_method=amplitude_priors_method, verbose=verbose,
                                         live_fit=live_fit, analytical=analytical)
            # first, fit the transverse position
            w.my_logger.info("\n\tFit y_c parameters...")
            fixed_default = np.copy(w.params.fixed)
            w.params.fixed = [True] * w.params.ndim
            for k in range(w.params.ndim):
                if "y_c" in w.params.labels[k]:
                    w.params.fixed[k] = False  # y_c_k
            run_minimisation(w, method="newton", ftol=100 / (w.Nx * w.Ny), xtol=1e-3, niter=10, verbose=verbose, with_line_search=False)
            # then fit all parameters together
            w.my_logger.info("\n\tFit all ChromaticPSF parameters...")
            w.params.fixed = fixed_default
            run_minimisation(w, method="newton", ftol=10 / (w.Nx * w.Ny), xtol=1e-4, niter=50, verbose=verbose, with_line_search=False)
        else:
            raise ValueError(f"mode argument must be '1D' or '2D'. Got {mode=}.")
        if amplitude_priors_method == "psf1d":
            w_reg = RegFitWorkspace(w, opt_reg=parameters.PSF_FIT_REG_PARAM, verbose=verbose)
            w_reg.run_regularisation(Ndof=w.trace_r)
            w.reg = np.copy(w_reg.opt_reg)
            w.trace_r = np.trace(w_reg.resolution)
            self.opt_reg = w_reg.opt_reg
            w.simulate(*w.params.values)
            if np.trace(w.amplitude_cov_matrix) < np.trace(w.amplitude_priors_cov_matrix):
                self.my_logger.warning(
                    f"\n\tTrace of final covariance matrix ({np.trace(w.amplitude_cov_matrix)}) is "
                    f"below the trace of the prior covariance matrix "
                    f"({np.trace(w.amplitude_priors_cov_matrix)}). This is probably due to a very "
                    f"high regularisation parameter in case of a bad fit. Therefore the final "
                    f"covariance matrix is mulitiplied by the ratio of the traces and "
                    f"the amplitude parameters are very close the amplitude priors.")
                r = np.trace(w.amplitude_priors_cov_matrix) / np.trace(w.amplitude_cov_matrix)
                w.amplitude_cov_matrix *= r
                w.amplitude_params_err = np.array([np.sqrt(w.amplitude_cov_matrix[x, x]) for x in range(self.Nx)])

            w.set_mask(poly_params=w.poly_params)
            # precise fit with sigma clipping
            run_minimisation_sigma_clipping(w, method="newton", ftol=1 / (w.Nx * w.Ny), xtol=1e-6, niter=50,
                                            sigma_clip=parameters.SPECTRACTOR_DECONVOLUTION_SIGMA_CLIP,
                                            niter_clip=3, verbose=verbose, with_line_search=False)

        # recompute and save params in class attributes
        w.simulate(*w.params.values)
        self.params.values = w.poly_params
        self.cov_matrix = np.copy(w.amplitude_cov_matrix)

        # add background crop to y_c
        self.params.values[w.Nx + w.y_c_0_index] += w.bgd_width

        # fill results
        self.psf.apply_max_width_to_bounds(max_half_width=w.Ny + 2 * w.bgd_width)
        self.profile_params = self.from_poly_params_to_profile_params(self.params.values, apply_bounds=True)
        self.fill_table_with_profile_params(self.profile_params)
        self.profile_params[:self.Nx, 0] = w.amplitude_params
        self.profile_params[:self.Nx, 1] = np.arange(self.Nx)
        self.from_profile_params_to_shape_params(self.profile_params)

        # save plots
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            w.plot_fit()
        return w


class ChromaticPSFFitWorkspace(FitWorkspace):

    def __init__(self, chromatic_psf, data, data_errors, mode, bgd_model_func=None, file_name="", analytical=True,
                 amplitude_priors_method="noprior", verbose=False, plot=False, live_fit=False, truth=None):
        if mode not in ["1D", "2D"]:
            raise ValueError(f"mode argument must be '1D' or '2D'. Got {mode=}.")
        length = len(chromatic_psf.table)
        params = FitParameters(np.copy(chromatic_psf.params.values[length:]),
                               labels=list(np.copy(chromatic_psf.params.labels[length:])),
                               axis_names=list(np.copy(chromatic_psf.params.axis_names[length:])), fixed=None,
                               truth=truth, filename=file_name)
        for k, par in enumerate(params.labels):
            if "x_c" in par or "saturation" in par:
                params.fixed[k] = True
        FitWorkspace.__init__(self, params, file_name=file_name, verbose=verbose, plot=plot, live_fit=live_fit)
        self.my_logger = set_logger(self.__class__.__name__)
        self.chromatic_psf = chromatic_psf
        self.data = data
        self.err = data_errors
        self.bgd_model_func = bgd_model_func
        self.analytical = analytical
        self.poly_params = np.copy(self.chromatic_psf.params.values)
        self.y_c_0_index = -1
        for k, par in enumerate(params.labels):
            if par == "y_c_0":
                self.y_c_0_index = k
                break

        # prepare the fit
        self.Ny, self.Nx = self.data.shape
        if self.Ny != self.chromatic_psf.Ny:
            raise AttributeError(f"Data y shape {self.Ny} different from "
                                 f"ChromaticPSF input Ny {self.chromatic_psf.Ny}.")
        if self.Nx != self.chromatic_psf.Nx:
            raise AttributeError(f"Data x shape {self.Nx} different from "
                                 f"ChromaticPSF input Nx {self.chromatic_psf.Nx}.")

        # prepare the background, data and errors
        self.bgd = np.zeros_like(self.data)
        if self.bgd_model_func is not None:
            self.bgd = self.bgd_model_func(np.arange(self.Nx), np.arange(self.Ny))
        self.data = self.data - self.bgd
        self.bgd_std = float(np.std(np.random.poisson(np.abs(self.bgd))))

        # crop spectrogram to fit faster
        self.bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        self.data = self.data[self.bgd_width:-self.bgd_width, :]
        self.err = self.err[self.bgd_width:-self.bgd_width, :]
        self.Ny, self.Nx = self.data.shape
        self.data = self.data.astype("float32").ravel()
        self.err = self.err.astype("float32").ravel()
        self.pixels = np.arange(self.data.shape[0])

        if mode == "1D":
            self.pixels = np.arange(self.Ny)
        else:
            yy, xx = np.mgrid[:self.Ny, :self.Nx]
            self.pixels = np.asarray([xx, yy])


        self.poly_params[self.Nx + self.y_c_0_index] -= self.bgd_width
        self.profile_params = self.chromatic_psf.from_poly_params_to_profile_params(self.poly_params)
        self.data_before_mask = np.copy(self.data)
        self.boundaries = None
        self.psf_cube_sparse_indices = None
        self.psf_cube_masked = None
        self.M_sparse_indices = None

        # update the bounds
        self.chromatic_psf.psf.apply_max_width_to_bounds(max_half_width=self.Ny)
        self.params.bounds = self.chromatic_psf.set_bounds()

        # error matrix
        # here image uncertainties are assumed to be uncorrelated
        # (which is not exactly true in rotated images)
        self.data_cov = sparse.diags(self.err * self.err, dtype="float32", format="dia")
        self.W = sparse.diags(1 / (self.err * self.err), dtype="float32", format="dia")
        self.sqrtW = self.W.sqrt()
        # create a mask
        self.W_before_mask = self.W.copy()

        # design matrix
        self.M = sparse.csr_matrix((self.Nx, self.data.size), dtype="float32")
        self.M_dot_W_dot_M = sparse.csr_matrix((self.Nx, self.Nx), dtype="float32")

        # prepare results
        self.amplitude_params = np.zeros(self.Nx)
        self.amplitude_params_err = np.zeros(self.Nx)
        self.amplitude_cov_matrix = np.zeros((self.Nx, self.Nx))

        # priors on amplitude parameters
        self.amplitude_priors_list = ['noprior', 'positive', 'smooth', 'psf1d', 'fixed', 'keep']
        self.amplitude_priors_method = amplitude_priors_method
        self.fwhm_priors = np.copy(self.chromatic_psf.table['fwhm'])
        self.reg = parameters.PSF_FIT_REG_PARAM
        self.trace_r = self.Nx / np.min(self.fwhm_priors)  # spectrophotometric uncertainty principle
        self.Q = np.zeros((self.Nx, self.Nx))
        self.Q_dot_A0 = np.zeros(self.Nx)
        if amplitude_priors_method not in self.amplitude_priors_list:
            raise ValueError(f"Unknown prior method for the amplitude fitting: {self.amplitude_priors_method}. "
                             f"Must be either {self.amplitude_priors_list}.")
        # regularisation matrices
        self.reg = parameters.PSF_FIT_REG_PARAM
        if self.amplitude_priors_method == "psf1d":
            self.amplitude_priors = np.copy(self.chromatic_psf.params.values[:self.Nx])
            self.amplitude_priors_cov_matrix = np.copy(self.chromatic_psf.cov_matrix)
            self.U = np.diag([1 / np.sqrt(self.amplitude_priors_cov_matrix[x, x]) for x in range(self.Nx)])
            L = np.diag(-2 * np.ones(self.Nx)) + np.diag(np.ones(self.Nx), -1)[:-1, :-1] + np.diag(np.ones(self.Nx), 1)[:-1, :-1]
            L.astype(float)
            L[0, 0] = -1
            L[-1, -1] = -1
            self.L = L
            self.Q = L.T @ self.U.T @ self.U @ L
            self.Q_dot_A0 = self.Q @ self.amplitude_priors
        if self.amplitude_priors_method == "fixed":
            self.amplitude_priors = np.copy(self.chromatic_psf.params.values[:self.Nx])
        self.set_mask()

    def plot_fit(self):
        cmap_bwr = copy.copy(mpl.colormaps["bwr"])
        cmap_bwr.set_bad(color='lightgrey')
        cmap_viridis = copy.copy(mpl.colormaps["viridis"])
        cmap_viridis.set_bad(color='lightgrey')

        data = np.copy(self.data_before_mask)
        model = np.copy(self.model)
        err = np.copy(self.err)
        if len(self.outliers) > 0 or len(self.mask) > 0:
            bad_indices = np.array(list(self.get_bad_indices()) + list(self.mask)).astype(int)
            data[bad_indices] = np.nan
        data = data.reshape((self.Ny, self.Nx))
        model = model.reshape((self.Ny, self.Nx))
        err = err.reshape((self.Ny, self.Nx))
        gs_kw = dict(width_ratios=[3, 0.15], height_ratios=[1, 1, 1, 1])

        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7, 7), gridspec_kw=gs_kw)

        norm = np.nanmax(data)
        plot_image_simple(ax[1, 0], data=model / norm, aspect='auto', cax=ax[1, 1], vmin=0, vmax=1,
                          units='1/max(data)', cmap=cmap_viridis)
        ax[1, 0].set_title("Model", fontsize=10, loc='center', color='white', y=0.8)
        plot_image_simple(ax[0, 0], data=data / norm, title='Data', aspect='auto',
                          cax=ax[0, 1], vmin=0, vmax=1, units='1/max(data)', cmap=cmap_viridis)
        ax[0, 0].set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
        residuals = (data - model)
        # residuals_err = self.spectrum.spectrogram_err / self.model
        norm = err
        residuals /= norm
        std = float(np.nanstd(residuals))
        plot_image_simple(ax[2, 0], data=residuals, vmin=-5 * std, vmax=5 * std, title='(Data-Model)/Err',
                          aspect='auto', cax=ax[2, 1], units='', cmap=cmap_bwr)

        ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, loc='center', color='black', y=0.8)
        ax[2, 0].text(0.05, 0.05, f'mean={np.nanmean(residuals):.3f}\nstd={np.nanstd(residuals):.3f}',
                      horizontalalignment='left', verticalalignment='bottom',
                      color='black', transform=ax[2, 0].transAxes)
        ax[0, 0].set_xticks(ax[2, 0].get_xticks()[1:-1])
        ax[0, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[1, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[2, 1].get_yaxis().set_label_coords(3.5, 0.5)
        ax[3, 1].remove()
        ax[3, 0].errorbar(np.arange(self.Nx), np.nansum(data, axis=0), yerr=np.sqrt(np.nansum(err ** 2, axis=0)),
                          label='Data', fmt='k.', markersize=0.1)
        model[np.isnan(data)] = np.nan  # mask background values outside fitted region
        ax[3, 0].plot(np.arange(self.Nx), np.nansum(model, axis=0), label='Model')
        ax[3, 0].set_ylabel('Transverse sum')
        ax[3, 0].set_xlabel(parameters.PLOT_XLABEL)
        ax[3, 0].legend(fontsize=7)
        ax[3, 0].set_xlim((0, data.shape[1]))
        ax[3, 0].grid(True)
        fig.tight_layout()
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

    def set_mask(self, poly_params=None):
        if poly_params is None:
            poly_params = self.poly_params
        profile_params = self.chromatic_psf.from_poly_params_to_profile_params(poly_params, apply_bounds=True)
        self.chromatic_psf.from_profile_params_to_shape_params(profile_params)
        if self.pixels.ndim == 1:
            psf_cube_masked = np.zeros((len(profile_params), self.Ny, self.Nx), dtype=bool)
            for x in range(psf_cube_masked.shape[0]):
                psf_cube_masked[x, :, x] = 1
        else:
            psf_cube_masked = self.chromatic_psf.build_psf_cube_masked(self.pixels, profile_params,
                                                                       fwhmx_clip=3 * parameters.PSF_FWHM_CLIP,
                                                                       fwhmy_clip=parameters.PSF_FWHM_CLIP)
        self.psf_cube_masked = self.chromatic_psf.convolve_psf_cube_masked(psf_cube_masked)
        self.boundaries, self.psf_cube_masked = self.chromatic_psf.get_boundaries(self.psf_cube_masked)
        self.psf_cube_sparse_indices, self.M_sparse_indices = self.chromatic_psf.get_sparse_indices(self.psf_cube_masked)
        mask = np.sum(self.psf_cube_masked.reshape(psf_cube_masked.shape[0], psf_cube_masked[0].size), axis=0) == 0
        W = np.copy(self.W_before_mask.data.ravel())
        W[mask] = 0
        self.W = sparse.diags(W, dtype="float32", format="dia")
        self.sqrtW = self.W.sqrt()
        self.mask = list(np.where(mask)[0])

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
        the set of amplitude parameters :math:`\hat{\mathbf{A}}` given by

        .. math::

            \hat{\mathbf{A}} =  (\mathbf{M}^T \mathbf{W} \mathbf{M})^{-1} \mathbf{M}^T \mathbf{W} \mathbf{y}

        The error matrix on the :math:`\hat{\mathbf{A}}` coefficient is simply
        :math:`(\mathbf{M}^T \mathbf{W} \mathbf{M})^{-1}`.

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

        Build a mock spectrogram without random Poisson noise:

        >>> psf = Moffat(clip=False)
        >>> s0 = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=100000)
        >>> params = s0.generate_test_poly_params()
        >>> params[:s0.Nx] *= 10
        >>> s0.params.values = params
        >>> saturation = params[-1]
        >>> data = s0.evaluate(s0.set_pixels(mode="2D"), params)
        >>> bgd = 10*np.ones_like(data)
        >>> data += bgd
        >>> data_errors = np.sqrt(data+1)

        Extract the background:

        >>> bgd_model_func, _, _ = extract_spectrogram_background_sextractor(data, data_errors, ws=[30,50])

        Estimate the first guess values:

        >>> s = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
        ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)
        >>> s.plot_summary(truth=s0)

        1D case.

        Simulate the data with fixed amplitude priors:

        >>> w = ChromaticPSFFitWorkspace(s, data, data_errors, "1D", bgd_model_func=bgd_model_func,
        ... amplitude_priors_method="fixed", verbose=True)
        >>> y, mod, mod_err = w.simulate(*s.params.values[s.Nx:])
        >>> w.plot_fit()

        .. doctest::
            :hide:

            >>> assert mod is not None

        Fit the amplitude of data without any prior:

        >>> w = ChromaticPSFFitWorkspace(s, data, data_errors, "1D", bgd_model_func=bgd_model_func, verbose=True,
        ... amplitude_priors_method="noprior")
        >>> y, mod, mod_err = w.simulate(*s.params.values[s.Nx:])
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert mod is not None
            >>> assert np.mean(np.abs(mod-w.data)/w.err) < 1

        Fit the amplitude of data smoothing the result with a window of size 10 pixels:

        >>> w = ChromaticPSFFitWorkspace(s, data, data_errors, "1D", bgd_model_func=bgd_model_func, verbose=True,
        ... amplitude_priors_method="smooth")
        >>> y, mod, mod_err = w.simulate(*s.params.values[s.Nx:])
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert mod is not None
            >>> assert np.mean(np.abs(mod-w.data)/w.err) < 1

        Fit the amplitude of data using the transverse PSF1D fit as a prior and with a
        Tikhonov regularisation parameter set by parameters.PSF_FIT_REG_PARAM:

        >>> w = ChromaticPSFFitWorkspace(s, data, data_errors, "1D", bgd_model_func=bgd_model_func, verbose=True,
        ... amplitude_priors_method="psf1d")
        >>> y, mod, mod_err = w.simulate(*s.params.values[s.Nx:])
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert mod is not None
            >>> assert np.mean(np.abs(mod-w.data)/w.err) < 1

        2D case

        Simulate the data with fixed amplitude priors:

        >>> w = ChromaticPSFFitWorkspace(s, data, data_errors, "2D", bgd_model_func=bgd_model_func,
        ... amplitude_priors_method="fixed", verbose=True)
        >>> y, mod, mod_err = w.simulate(*s.params.values[s.Nx:])
        >>> w.plot_fit()

        .. doctest::
            :hide:

            >>> assert mod is not None

        Simulate the data with a Tikhonov prior on amplitude parameters:

        >>> parameters.PSF_FIT_REG_PARAM = 0.002
        >>> w = ChromaticPSFFitWorkspace(s, data, data_errors, "2D", bgd_model_func=bgd_model_func,
        ... amplitude_priors_method="psf1d", verbose=True)
        >>> y, mod, mod_err = w.simulate(*s.params.values[s.Nx:])
        >>> w.plot_fit()

        .. doctest::
            :hide:

            >>> assert mod is not None

        """
        # linear regression for the amplitude parameters
        # prepare the vectors
        poly_params = np.concatenate([np.ones(self.Nx), shape_params])
        poly_params[self.Nx + self.y_c_0_index] -= self.bgd_width
        profile_params = self.chromatic_psf.from_poly_params_to_profile_params(poly_params, apply_bounds=True)
        profile_params[:self.Nx, 0] = 1
        profile_params[:self.Nx, 1] = np.arange(self.Nx)
        # profile_params[:self.Nx, 2] -= self.bgd_width
        if self.amplitude_priors_method != "fixed":
            M = self.chromatic_psf.build_sparse_M(self.pixels, profile_params, dtype="float32",
                                                  M_sparse_indices=self.M_sparse_indices, boundaries=self.boundaries)

            M_dot_W = M.T @ self.sqrtW
            W_dot_data = self.W @ self.data
            # Compute the minimizing amplitudes
            if sparse_dot_mkl is None:
                M_dot_W_dot_M = M_dot_W @ M_dot_W.T
            else:
                tri = sparse_dot_mkl.gram_matrix_mkl(M_dot_W, transpose=True)
                dia = sparse.csr_matrix((tri.diagonal(), (np.arange(tri.shape[0]), np.arange(tri.shape[0]))),
                                        shape=tri.shape, dtype="float32")
                M_dot_W_dot_M = tri + tri.T - dia
            if self.amplitude_priors_method != "psf1d":
                if self.amplitude_priors_method == "keep":
                    amplitude_params = np.copy(self.amplitude_params)
                    cov_matrix = np.copy(self.amplitude_cov_matrix)
                else:
                    # try:
                    #     L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M))
                    #     cov_matrix = L.T @ L
                    # except np.linalg.LinAlgError:
                    cov_matrix = np.linalg.inv(M_dot_W_dot_M.toarray())
                    amplitude_params = cov_matrix @ (M.T @ W_dot_data)
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
                # try:
                #     L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M_plus_Q))
                #     cov_matrix = L.T @ L
                # except np.linalg.LinAlgError:
                cov_matrix = np.linalg.inv(M_dot_W_dot_M_plus_Q)
                amplitude_params = cov_matrix @ (M.T @ W_dot_data + np.float32(self.reg) * self.Q_dot_A0)
            amplitude_params = np.asarray(amplitude_params).reshape(-1)
            self.M = M
            self.M_dot_W_dot_M = M_dot_W_dot_M
            self.model = M @ amplitude_params
        else:
            amplitude_params = np.copy(self.amplitude_priors)
            err2 = np.copy(amplitude_params)
            err2[err2 <= 0] = np.min(np.abs(err2[err2 > 0]))
            cov_matrix = np.diag(err2)
        self.amplitude_params = np.copy(amplitude_params)
        # TODO: propagate and marginalize over the shape parameter uncertainties ?
        self.amplitude_params_err = np.array([np.sqrt(cov_matrix[x, x]) for x in range(self.Nx)])
        self.amplitude_cov_matrix = np.copy(cov_matrix)
        poly_params[:self.Nx] = amplitude_params

        if self.amplitude_priors_method == "fixed":
            self.model = self.chromatic_psf.evaluate(self.pixels, poly_params)
            # self.chromatic_psf.params.values is updated in evaluate(): reset to original values
            self.chromatic_psf.params.values[self.Nx + self.y_c_0_index] += self.bgd_width
        self.poly_params = np.copy(poly_params)
        self.profile_params = np.copy(profile_params)
        self.model_err = np.zeros_like(self.model)
        return self.pixels, self.model, self.model_err

    def amplitude_derivatives(self):
        r"""
        Compute analytically the amplitude vector \hat{\mathbf{A}} derivatives with respect to the PSF parameters.
        With

        .. math::

            \hat{\mathbf{A}} =  \hat{\mathbf{C}} \cdot \mathbf{M}^T \mathbf{W} \mathbf{y}

            \hat{\mathbf{C}} = (\mathbf{M}^T \mathbf{W} \mathbf{M})^{-1}

        derivatives are

        .. math::

            \frac{\partial \hat{\mathbf{A}}}{\partial \theta} =  \frac{\partial \hat{\mathbf{C}}}{\partial \theta} \cdot \mathbf{M}^T \mathbf{W} \mathbf{y} + \hat{\mathbf{C}} \cdot \frac{\partial \mathbf{M}^T \mathbf{W} \mathbf{y}}{\partial \theta}

            \frac{\partial \hat{\mathbf{C}}}{\partial \theta} = - \hat{\mathbf{C}} \cdot \frac{\partial \mathbf{M}^T \mathbf{W} \mathbf{M}}{\partial \theta}  \cdot  \hat{\mathbf{C}}

            \frac{\partial \mathbf{M}^T \mathbf{W} \mathbf{M}}{\partial \theta} = 2 \frac{\partial \mathbf{M}^T}{\partial \theta} \mathbf{W} \mathbf{M}

            \frac{\partial \mathbf{M}^T \mathbf{W} \mathbf{y}}{\partial \theta} = \frac{\partial \mathbf{M}^T}{\partial \theta} \mathbf{W} \mathbf{y}

        If amplitude vector is regularized via Tikhonov regularisation, regularisation term is added appropriately.

        Returns
        -------
        dA_dtheta: list
            List of amplitude vector derivatives.

        Examples
        --------

        Set the parameters:

        >>> parameters.PIXDIST_BACKGROUND = 40
        >>> parameters.PIXWIDTH_BACKGROUND = 10
        >>> parameters.PIXWIDTH_SIGNAL = 30

        Build a mock spectrogram without random Poisson noise:

        >>> psf = Moffat(clip=False)
        >>> s0 = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=100000)
        >>> params = s0.generate_test_poly_params()
        >>> params[:s0.Nx] *= 10
        >>> s0.params.values = params
        >>> saturation = params[-1]
        >>> data = s0.evaluate(s0.set_pixels(mode="2D"), params)
        >>> bgd = 10*np.ones_like(data)
        >>> data += bgd
        >>> data_errors = np.sqrt(data+1)

        Extract the background:

        >>> bgd_model_func, _, _ = extract_spectrogram_background_sextractor(data, data_errors, ws=[30,50])

        Estimate the first guess values:

        >>> s = ChromaticPSF(psf, Nx=120, Ny=100, deg=2, saturation=saturation)
        >>> s.fit_transverse_PSF1D_profile(data, data_errors, w=20, ws=[30,50],
        ... pixel_step=1, bgd_model_func=bgd_model_func, saturation=saturation, live_fit=False)
        >>> s.plot_summary(truth=s0)

        Simulate the data with a Tikhonov prior on amplitude parameters:

        >>> parameters.PSF_FIT_REG_PARAM = 0.002
        >>> s.params.values = s.from_table_to_poly_params()
        >>> w = ChromaticPSFFitWorkspace(s, data, data_errors, "2D", bgd_model_func=bgd_model_func,
        ... amplitude_priors_method="psf1d", verbose=True)
        >>> y, mod, mod_err = w.simulate(*s.params.values[s.Nx:])
        >>> w.plot_fit()

        .. doctest::
            :hide:

            >>> assert mod is not None

        Get the derivatives:

        >>> dA_dtheta = w.amplitude_derivatives()
        >>> print(np.array(dA_dtheta).shape, w.amplitude_params.shape)
        (13, 120) (120,)

        """
        # compute matrices without derivatives
        WM =  self.W @ self.M
        WD = self.W @ self.data
        MWD = self.M.T @ WD
        if self.amplitude_priors_method == "psf1d":
            MWD += np.float32(self.reg) * self.Q_dot_A0
        # compute partial derivatives of model matrix M
        dM_dtheta = self.chromatic_psf.build_sparse_dM(self.pixels, profile_params=self.profile_params,
                                                       M_sparse_indices=self.M_sparse_indices,
                                                       boundaries=self.boundaries, dtype="float32")
        # compute partial derivatives of amplitude vector A
        nparams = len(dM_dtheta)
        dMWD_dtheta = [dM_dtheta[ip].T @ WD for ip in range(nparams)]
        dMWM_dtheta = [2 * dM_dtheta[ip].T @ WM for ip in range(nparams)]
        dcov_dtheta = [-self.amplitude_cov_matrix @ (dMWM_dtheta[ip] @ self.amplitude_cov_matrix) for ip in range(nparams)]
        dA_dtheta = [self.amplitude_cov_matrix @ dMWD_dtheta[ip] + dcov_dtheta[ip] @ MWD for ip in range(nparams)]
        return dA_dtheta

    def jacobian(self, params, epsilon, model_input=None):
        r"""Generic function to compute the Jacobian matrix of a model, linear parameters being fixed (see Notes),
        with analytical or numerical derivatives. Analytical derivatives are performed if `self.analytical` is True.
        Let's write :math:`\theta` the non-linear model parameters. If the model is written as:

        .. math::

            \mathbf{I} =  \mathbf{M}(\theta) \cdot \hat{\mathbf{A}}(\theta),

        this jacobian function returns:

        .. math::

            \frac{\partial \mathbf{I}}{\partial \theta} =   \frac{\partial \mathbf{M}}{\partial \theta} \cdot \hat{\mathbf{A}}.

        Notes
        -----
            The gradient descent is performed on the non-linear parameters :math:`\theta` (PSF shape and position). Linear parameters :math:`\mathbf{A}` (amplitudes) are computed on the fly.
            Therefore, :math:`\chi^2` is a function of :math:`\theta` only

            .. math ::

                \chi^2(\theta) = \chi'^2(\theta, \hat{\mathbf{A}}

            whose partial derivatives on :math:`\theta` for gradient descent are:

            .. math ::

                \frac{\partial \chi^2}{\partial \theta} = \left.\left(\frac{\partial \chi'^2}{\partial \theta}  + \frac{\partial \chi'^2}{\partial \mathbf{A}} \frac{\partial \mathbf{A}}{\partial \theta}\right)\right\vert_{\mathbf{A} = \hat{\mathbf{A}}}

            By definition, :math:`\left.\partial \chi'^2/\partial \mathbf{A}\right\vert_{\mathbf{A} = \hat{\mathbf{A}}}=0` then :math:`\chi^2`  partial derivatives must be performed with fixed :math:`\mathbf{A} = \hat{\mathbf{A}}`
            for gradient descent. `self.amplitude_priors_method` is temporarily switched to "keep" in `self.jacobian()` to use previously computed :math:`\hat{\mathbf{A}}` solution.


        Parameters
        ----------
        params: array_like
            The array of model parameters.
        epsilon: array_like
            The array of small steps to compute the partial derivatives of the model.
        model_input: array_like, optional
            A model input as a list with (x, model, model_err) to avoid an additional call to simulate().

        Returns
        -------
        J: np.array
            The Jacobian matrix.

        """
        method = copy.copy(self.amplitude_priors_method)
        self.amplitude_priors_method = "keep"
        if not self.analytical:
            J = super().jacobian(params, epsilon=epsilon, model_input=model_input)
        else:
            profile_params = np.copy(self.profile_params)
            amplitude_params = np.copy(self.amplitude_params)
            profile_params[:, 0] *= amplitude_params  # np.ones_like(amplitude_params)
            J = self.chromatic_psf.build_psf_jacobian(self.pixels, profile_params=profile_params,
                                                      psf_cube_sparse_indices=self.psf_cube_sparse_indices,
                                                      boundaries=self.boundaries, dtype="float32")
        self.amplitude_priors_method = method
        return J


if __name__ == "__main__":
    import doctest

    doctest.testmod()

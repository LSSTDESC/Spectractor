import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import sparse
from scipy import interpolate
from scipy.signal import convolve2d
import time

from spectractor import parameters
from spectractor.config import set_logger, load_config
from spectractor.extractor.images import Image, find_target, turn_image
from spectractor.extractor.spectrum import Spectrum, calibrate_spectrum
from spectractor.extractor.background import extract_spectrogram_background_sextractor
from spectractor.extractor.chromaticpsf import ChromaticPSF
from spectractor.extractor.psf import load_PSF
from spectractor.tools import ensure_dir, plot_image_simple, from_lambda_to_colormap, plot_spectrum_simple
from spectractor.fit.fitter import (run_minimisation, run_minimisation_sigma_clipping, write_fitparameter_json,
                                    RegFitWorkspace, FitWorkspace, FitParameters)

try:
    import sparse_dot_mkl
except ModuleNotFoundError:
    sparse_dot_mkl = None


def dumpParameters():
    for item in dir(parameters):
        if not item.startswith("__"):
            print(item, getattr(parameters, item))


class FullForwardModelFitWorkspace(FitWorkspace):

    def __init__(self, spectrum, amplitude_priors_method="noprior", verbose=False, plot=False, live_fit=False, truth=None):
        """Class to fit a full forward model on data to extract a spectrum, with ADR prediction and order 2 subtraction.

        Parameters
        ----------
        spectrum: Spectrum
        amplitude_priors_method
        verbose
        plot
        live_fit
        truth

        Examples
        --------
        >>> spec = Spectrum("./tests/data/sim_20170530_134_spectrum.fits")
        >>> w = FullForwardModelFitWorkspace(spectrum=spec, amplitude_priors_method="spectrum")
        """
        self.my_logger = set_logger(self.__class__.__name__)
        # prepare parameters to fit
        length = len(spectrum.chromatic_psf.table)
        self.diffraction_orders = np.arange(spectrum.order, spectrum.order + 3 * np.sign(spectrum.order), np.sign(spectrum.order))
        if len(self.diffraction_orders) == 0:
            raise ValueError(f"At least one diffraction order must be given for spectrogram simulation.")
        self.psf_poly_params = np.copy(spectrum.chromatic_psf.from_table_to_poly_params())
        self.psf_poly_params = self.psf_poly_params[length:]
        psf_poly_params_labels = np.copy(spectrum.chromatic_psf.params.labels[length:])
        psf_poly_params_names = np.copy(spectrum.chromatic_psf.params.axis_names[length:])
        spectrum.chromatic_psf.psf.apply_max_width_to_bounds(max_half_width=spectrum.spectrogram_Ny)
        psf_poly_params_bounds = spectrum.chromatic_psf.set_bounds()
        D2CCD = np.copy(spectrum.header['D2CCD'])
        p = np.array([1, 1, 1, D2CCD, np.copy(spectrum.header['PIXSHIFT']), 0,
                      np.copy(spectrum.rotation_angle), 1, parameters.OBS_CAMERA_ROTATION,
                      np.copy(spectrum.pressure),  np.copy(spectrum.temperature),  np.copy(spectrum.airmass)])
        self.psf_params_start_index = np.array([12 + len(self.psf_poly_params) * k for k in range(len(self.diffraction_orders))])
        self.saturation = spectrum.spectrogram_saturation
        p = np.concatenate([p] + [self.psf_poly_params] * len(self.diffraction_orders))
        input_labels = [f"A{order}" for order in self.diffraction_orders]
        input_labels += [r"D_CCD [mm]", r"shift_x [pix]", r"shift_y [pix]", r"angle [deg]", "B", "R", "P [hPa]", "T [Celsius]", "z"]
        for order in self.diffraction_orders:
            input_labels += [label+f"_{order}" for label in psf_poly_params_labels]
        axis_names = [f"$A_{order}$" for order in self.diffraction_orders]
        axis_names += [r"$D_{CCD}$ [mm]", r"$\delta_{\mathrm{x}}^{(\mathrm{fit})}$ [pix]",
                       r"$\delta_{\mathrm{y}}^{(\mathrm{fit})}$ [pix]", r"$\alpha$ [deg]", "$B$", "R",
                       r"$P_{\mathrm{atm}}$ [hPa]", r"$T_{\mathrm{atm}}$ [Celcius]", "$z$"]
        for order in self.diffraction_orders:
            axis_names += [label+rf"$\!_{order}$" for label in psf_poly_params_names]
        bounds = [[0, 2], [0, 2], [0, 2],
                  [D2CCD - 3 * parameters.DISTANCE2CCD_ERR, D2CCD + 3 * parameters.DISTANCE2CCD_ERR],
                  [-parameters.PIXSHIFT_PRIOR, parameters.PIXSHIFT_PRIOR],
                  [-10 * parameters.PIXSHIFT_PRIOR, 10 * parameters.PIXSHIFT_PRIOR],
                  [-90, 90], [0.2, 5], [-360, 360], [300, 1100], [-100, 100], [1.001, 3]]
        bounds += list(psf_poly_params_bounds) * len(self.diffraction_orders)
        fixed = [False] * p.size
        for k, par in enumerate(input_labels):
            if "x_c" in par or "saturation" in par:  # or "y_c" in par:
                fixed[k] = True
        for k, par in enumerate(input_labels):
            if "y_c" in par:
                fixed[k] = False
                p[k] = 0

        params = FitParameters(p, labels=input_labels, axis_names=axis_names, fixed=fixed, bounds=bounds,
                               truth=truth, filename=spectrum.filename)
        # This set of fixed parameters was determined so that the reconstructed spectrum has a ZERO bias
        # with respect to the true spectrum injected in the simulation
        # A2 is free only if spectrogram is a simulation or if the order 2/1 ratio is not known and flat
        params.fixed[params.get_index("A1")] = True  # A1
        params.fixed[params.get_index("A2")] = (not spectrum.disperser.flat_ratio_order_2over1) and (not ("A2_T" in spectrum.header))
        params.fixed[params.get_index("A3")] = True  # A3
        params.fixed[params.get_index("D_CCD [mm]")] = True  # D2CCD: spectrogram can not tell something on this parameter: rely on calibrate_spectrum
        params.fixed[params.get_index("shift_x [pix]")] = True  # delta x: if False, extracted spectrum is biased compared with truth
        params.fixed[params.get_index("shift_y [pix]")] = True  # delta y
        params.fixed[params.get_index("angle [deg]")] = True  # angle
        params.fixed[params.get_index("B")] = True  # B: not needed in simulations, to check with data
        params.fixed[params.get_index("R")] = True  # camera rot
        params.fixed[params.get_index("P [hPa]")] = True  # pressure
        params.fixed[params.get_index("T [Celsius]")] = True  # temperature
        params.fixed[params.get_index("z")] = True  # airmass

        FitWorkspace.__init__(self, params, spectrum.filename, verbose, plot, live_fit, truth=truth)
        self.spectrum = spectrum

        # crop data to fit faster
        self.lambdas = self.spectrum.lambdas
        self.bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        self.data = spectrum.spectrogram[self.bgd_width:-self.bgd_width, :]
        self.err = spectrum.spectrogram_err[self.bgd_width:-self.bgd_width, :]
        self.bgd = spectrum.spectrogram_bgd[self.bgd_width:-self.bgd_width, :]
        self.bgd_flat = self.bgd.flatten()
        self.Ny, self.Nx = self.data.shape
        yy, xx = np.mgrid[:self.Ny, :self.Nx]
        self.pixels = np.asarray([xx, yy], dtype=int)

        # adapt the ChromaticPSF table shape
        if self.Nx != self.spectrum.chromatic_psf.Nx:
            self.spectrum.chromatic_psf.resize_table(new_Nx=self.Nx)

        # load the disperser relative transmissions
        self.tr_ratio = interpolate.interp1d(parameters.LAMBDAS, np.ones_like(parameters.LAMBDAS), bounds_error=False, fill_value=1.)
        if abs(self.spectrum.order) == 1:
            self.tr_ratio_next_order = self.spectrum.disperser.ratio_order_2over1
            self.tr_ratio_next_next_order = self.spectrum.disperser.ratio_order_3over1
        elif abs(self.spectrum.order) == 2:
            self.tr_ratio_next_order = self.spectrum.disperser.ratio_order_3over2
            self.tr_ratio_next_next_order = None
        elif abs(self.spectrum.order) == 3:
            self.tr_ratio_next_order = None
            self.tr_ratio_next_next_order = None
        else:
            raise ValueError(f"{abs(self.spectrum.order)=}: must be 1, 2 or 3. "
                             f"Higher diffraction orders not implemented yet in full forward model.")
        self.tr = [self.tr_ratio, self.tr_ratio_next_order, self.tr_ratio_next_next_order]

        # PSF cube computation
        self.psf_cubes_masked = {}
        self.psf_profile_params = {}
        self.boundaries = {}
        for order in self.diffraction_orders:
            self.psf_cubes_masked[order] = np.empty(1)
            self.boundaries[order] = {}
            self.psf_profile_params[order] = None
        self.fix_psf_cube = False

        # prepare the background, data and errors
        self.bgd_std = float(np.std(np.random.poisson(np.abs(self.bgd))))

        # error matrix
        # here image uncertainties are assumed to be uncorrelated
        # (which is not exactly true in rotated images)
        self.W = 1. / (self.err * self.err)
        self.W = self.W.flatten()

        # flat data for fitworkspace
        self.data = self.data.flatten() - self.bgd_flat
        self.err = self.err.flatten()
        self.data_before_mask = np.copy(self.data)
        self.W_before_mask = np.copy(self.W)

        # create mask
        self.sqrtW = sparse.diags(np.sqrt(self.W), format="dia", dtype="float32")

        # design matrix
        self.M = None
        self.psf_cube_sparse_indices = {}
        self.M_sparse_indices = {}
        for order in self.diffraction_orders:
            self.M_sparse_indices[order] = None
            self.psf_cube_sparse_indices[order] = None
        self.set_mask(fwhmx_clip=3*parameters.PSF_FWHM_CLIP, fwhmy_clip=2*parameters.PSF_FWHM_CLIP)  # not a narrow mask for first fit
        self.M_dot_W_dot_M = None

        # prepare results
        self.amplitude_params = np.zeros(self.Nx)
        self.amplitude_params_err = np.zeros(self.Nx)
        self.amplitude_cov_matrix = np.zeros((self.Nx, self.Nx))

        # priors on amplitude parameters
        self.amplitude_priors_list = ['noprior', 'positive', 'smooth', 'spectrum', 'fixed', 'keep']
        self.amplitude_priors_method = amplitude_priors_method
        self.fwhm_priors = np.copy(spectrum.chromatic_psf.table['fwhm'])
        self.reg = spectrum.chromatic_psf.opt_reg
        if 'PSF_REG' in spectrum.header and float(spectrum.header["PSF_REG"]) > 0:
            self.reg = float(spectrum.header['PSF_REG'])
        if self.reg < 0:
            self.reg = parameters.PSF_FIT_REG_PARAM
        self.my_logger.info(f"\n\tFull forward model fitting with regularisation parameter r={self.reg}.")
        self.Q = np.zeros((self.Nx, self.Nx), dtype="float32")
        self.Q_dot_A0 = np.zeros(self.Nx, dtype="float32")
        if amplitude_priors_method not in self.amplitude_priors_list:
            raise ValueError(f"Unknown prior method for the amplitude fitting: {self.amplitude_priors_method}. "
                             f"Must be either {self.amplitude_priors_list}.")
        self.spectrum.convert_from_flam_to_ADUrate()
        self.amplitude_priors = np.copy(self.spectrum.data)
        if self.amplitude_priors_method == "spectrum":
            self.amplitude_priors_cov_matrix = np.copy(self.spectrum.cov_matrix)
        if self.spectrum.data.size != self.Nx:  # must rebin the priors
            old_x = np.linspace(0, 1, self.spectrum.data.size)
            new_x = np.linspace(0, 1, self.Nx)
            self.spectrum.lambdas = np.interp(new_x, old_x, self.spectrum.lambdas)
            self.amplitude_priors = np.interp(new_x, old_x, self.amplitude_priors)
            if self.amplitude_priors_method == "spectrum":
                # rebin prior cov matrix with monte-carlo
                niter = 10000
                samples = np.random.multivariate_normal(np.zeros_like(old_x), cov=self.amplitude_priors_cov_matrix, size=niter)
                new_samples = np.zeros((niter, new_x.size))
                for i in range(niter):
                    new_samples[i] = np.interp(new_x, old_x, samples[i])
                self.amplitude_priors_cov_matrix = np.cov(new_samples.T)
        # regularisation matrices
        if amplitude_priors_method == "spectrum":
            # U = np.diag([1 / np.sqrt(np.sum(self.err[:, x]**2)) for x in range(self.Nx)])
            # self.U = np.diag([1 / np.sqrt(self.amplitude_priors_cov_matrix[x, x]) for x in range(self.Nx)])
            self.UTU = np.linalg.inv(self.amplitude_priors_cov_matrix)
            L = np.diag(-2 * np.ones(self.Nx)) + np.diag(np.ones(self.Nx), -1)[:-1, :-1] \
                + np.diag(np.ones(self.Nx), 1)[:-1, :-1]
            L.astype(float)
            L[0, 0] = -1
            L[-1, -1] = -1
            self.L = L
            self.Q = (L.T @ self.UTU @ L).astype("float32")  # Q is dense do not sparsify it (leads to errors otherwise)
            self.Q_dot_A0 = self.Q @ self.amplitude_priors.astype("float32")

    def set_mask(self, params=None, fwhmx_clip=3*parameters.PSF_FWHM_CLIP, fwhmy_clip=parameters.PSF_FWHM_CLIP):
        """

        Parameters
        ----------
        params
        fwhmx_clip
        fwhmy_clip

        Returns
        -------

        Examples
        --------
        >>> spec = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> w = FullForwardModelFitWorkspace(spectrum=spec, amplitude_priors_method="fixed", verbose=True)
        >>> _ = w.simulate(*w.params.values)
        >>> w.plot_fit()

        """
        self.my_logger.info("\n\tReset spectrogram mask with current parameters.")
        if params is None:
            params = self.params.values
        A1, A2, A3, D2CCD, dx0, dy0, angle, B, rot, pressure, temperature, airmass, *psf_poly_params_all = params
        poly_params = np.array(psf_poly_params_all).reshape((len(self.diffraction_orders), -1))

        lambdas = self.spectrum.compute_lambdas_in_spectrogram(D2CCD, dx0, dy0, angle, niter=5, with_adr=True,
                                                               order=self.spectrum.order)
        self.psf_cubes_masked = {}
        self.M_sparse_indices = {}
        self.psf_cube_sparse_indices = {}
        for k, order in enumerate(self.diffraction_orders):
            profile_params = self.spectrum.chromatic_psf.from_poly_params_to_profile_params(poly_params[k],
                                                                                            apply_bounds=True)
            if order == self.diffraction_orders[0]:  # only first diffraction order
                self.spectrum.chromatic_psf.from_profile_params_to_shape_params(profile_params)
            dispersion_law = self.spectrum.compute_dispersion_in_spectrogram(lambdas, dx0, dy0, angle,
                                                                             niter=5, with_adr=True,
                                                                             order=order)
            profile_params[:, 0] = 1
            profile_params[:, 1] = dispersion_law.real + self.spectrum.spectrogram_x0
            profile_params[:, 2] += dispersion_law.imag - self.bgd_width
            psf_cube = self.spectrum.chromatic_psf.build_psf_cube(self.pixels, profile_params,
                                                                  fwhmx_clip=fwhmx_clip,
                                                                  fwhmy_clip=fwhmy_clip, dtype="float32")

            self.psf_cubes_masked[order] = self.spectrum.chromatic_psf.get_psf_cube_masked(psf_cube, convolve=True)
            # make rectangular mask per wavelength
            self.boundaries[order], self.psf_cubes_masked[order] = self.spectrum.chromatic_psf.get_boundaries(self.psf_cubes_masked[order])
            self.psf_cube_sparse_indices[order], self.M_sparse_indices[order] = self.spectrum.chromatic_psf.get_sparse_indices(self.psf_cubes_masked[order])
        mask = np.sum(self.psf_cubes_masked[self.diffraction_orders[0]].reshape(psf_cube.shape[0], psf_cube[0].size), axis=0) == 0
        self.W = np.copy(self.W_before_mask)
        self.W[mask] = 0
        self.sqrtW = sparse.diags(np.sqrt(self.W), format="dia", dtype="float32")
        self.mask = list(np.where(mask)[0])

    def simulate(self, *params):
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
        a set of amplitude parameters :math:`\hat{\mathbf{A}}` given by

        .. math::

            \hat{\mathbf{A}} =  (\mathbf{M}^T \mathbf{W} \mathbf{M})^{-1} \mathbf{M}^T \mathbf{W} \mathbf{y}

        The error matrix on the :math:`\hat{\mathbf{A}}` coefficient is simply
        :math:`(\mathbf{M}^T \mathbf{W} \mathbf{M})^{-1}`.

        See Also
        --------
        ChromaticPSF2DFitWorkspace.simulate

        Parameters
        ----------
        params: array_like
            Full forward model parameter array.

        Examples
        --------

        Load data:

        >>> spec = Spectrum("./tests/data/sim_20170530_134_spectrum.fits")

        Simulate the data with fixed amplitude priors:

        .. doctest::

            >>> w = FullForwardModelFitWorkspace(spectrum=spec, amplitude_priors_method="fixed", verbose=True)
            >>> y, mod, mod_err = w.simulate(*w.params.values)
            >>> w.plot_fit()

        .. doctest::
            :hide:

            >>> assert mod is not None

        Simulate the data with a Tikhonov prior on amplitude parameters:

        .. doctest::

            >>> spec = Spectrum("./tests/data/sim_20170530_134_spectrum.fits")
            >>> w = FullForwardModelFitWorkspace(spectrum=spec, amplitude_priors_method="spectrum", verbose=True)
            >>> y, mod, mod_err = w.simulate(*w.params.values)
            >>> w.plot_fit()

        """
        # linear regression for the amplitude parameters
        # prepare the vectors
        self.params.values = np.asarray(params)
        A1, A2, A3, D2CCD, dx0, dy0, angle, B, rot, pressure, temperature, airmass, *poly_params_all = params
        poly_params = np.array(poly_params_all).reshape((len(self.diffraction_orders), -1))
        self.spectrum.adr_params[2] = temperature
        self.spectrum.adr_params[3] = pressure
        self.spectrum.adr_params[-1] = airmass

        parameters.OBS_CAMERA_ROTATION = rot
        W_dot_data = (self.W * (self.data + (1 - B) * self.bgd_flat)).astype("float32")

        # Evaluate ADR and compute wavelength arrays
        self.lambdas = self.spectrum.compute_lambdas_in_spectrogram(D2CCD, dx0, dy0, angle, niter=5, with_adr=True,
                                                                    order=self.diffraction_orders[0])
        M = None
        for k, order in enumerate(self.diffraction_orders):
            if self.tr[k] is None or self.params[f"A{order}"] == 0:  # diffraction order undefined
                self.psf_profile_params[order] = None
                continue
            # Evaluate PSF profile
            if k == 0:
                self.psf_profile_params[order] = self.spectrum.chromatic_psf.update(poly_params[k], self.spectrum.spectrogram_x0 + dx0,
                                                                                    self.spectrum.spectrogram_y0 + dy0, angle, plot=False, apply_bounds=True)
            else:
                self.psf_profile_params[order] = self.spectrum.chromatic_psf.from_poly_params_to_profile_params(poly_params[k], apply_bounds=True)

            # Dispersion law
            dispersion_law = self.spectrum.compute_dispersion_in_spectrogram(self.lambdas, dx0, dy0, angle,
                                                                             niter=5, with_adr=True, order=order)

            # Fill spectrogram trace as a function of the pixel column x
            self.psf_profile_params[order][:, 0] = self.params[f"A{order}"] * self.tr[k](self.lambdas)
            self.psf_profile_params[order][:, 1] = dispersion_law.real + self.spectrum.spectrogram_x0
            self.psf_profile_params[order][:, 2] += dispersion_law.imag - self.bgd_width

            # Matrix filling
            # Older piece of code, using full matrices (non sparse). Keep here for temporary archive.
            # psf_cube_order = self.spectrum.chromatic_psf.build_psf_cube(self.pixels, profile_params[-1], fwhmx_clip=3 * parameters.PSF_FWHM_CLIP, fwhmy_clip=parameters.PSF_FWHM_CLIP, dtype="float32", mask=self.psf_cubes_masked[order], boundaries=self.boundaries[order])
            # if self.sparse_indices is None:
            #    self.sparse_indices = np.concatenate([np.where(self.psf_cube_masked[k].ravel() > 0)[0] for k in range(len(profile_params))])
            # if psf_cube is None:
            #     psf_cube = psf_cube_order
            # else:
            #     psf_cube += psf_cube_order
            M_order = self.spectrum.chromatic_psf.build_sparse_M(self.pixels, self.psf_profile_params[order],
                                                                 dtype="float32", M_sparse_indices=self.M_sparse_indices[order], boundaries=self.boundaries[order])
            if M is None:
                M = M_order
            else:
                M += M_order

        # M = psf_cube.reshape(len(profile_params[0]), self.pixels[0].size).T  # flattening
        # if self.sparse_indices is None:
        #     self.sparse_indices = np.where(M > 0)
        # M = sparse.csc_matrix((M[self.sparse_indices].ravel(), self.sparse_indices), shape=M.shape, dtype="float32")
        # Algebra to compute amplitude parameters
        if self.amplitude_priors_method != "fixed":
            M_dot_W = M.T @ self.sqrtW
            if sparse_dot_mkl is None:
                M_dot_W_dot_M = M_dot_W @ M_dot_W.T
            else:
                tri = sparse_dot_mkl.gram_matrix_mkl(M_dot_W, transpose=True)
                dia = sparse.csr_matrix((tri.diagonal(), (np.arange(tri.shape[0]), np.arange(tri.shape[0]))), shape=tri.shape, dtype="float32")
                M_dot_W_dot_M = (tri + tri.T - dia).toarray()
            if self.amplitude_priors_method != "spectrum":
                if self.amplitude_priors_method == "keep":
                    amplitude_params = np.copy(self.amplitude_params)
                    cov_matrix = np.copy(self.amplitude_cov_matrix)
                else:
                    # try:  # slower
                    #     L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M))
                    #     cov_matrix = L.T @ L
                    # except np.linalg.LinAlgError:
                    cov_matrix = np.linalg.inv(M_dot_W_dot_M)
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
                M_dot_W_dot_M_plus_Q = M_dot_W_dot_M + np.float32(self.reg) * self.Q
                # try:  # slower
                #     L = sparse.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M_plus_Q))
                #     cov_matrix = L.T @ L
                # except np.linalg.LinAlgError:
                cov_matrix = np.linalg.inv(M_dot_W_dot_M_plus_Q)  # M_dot_W_dot_M_plus_Q is not so sparse
                amplitude_params = cov_matrix @ (M.T @ W_dot_data + self.reg * self.Q_dot_A0)
            self.M_dot_W_dot_M = M_dot_W_dot_M
            amplitude_params = np.asarray(amplitude_params).reshape(-1)
        else:
            amplitude_params = np.copy(self.amplitude_priors)
            err2 = np.copy(amplitude_params)
            err2[err2 <= 0] = np.min(np.abs(err2[err2 > 0]))
            cov_matrix = np.diag(err2)

        # Save results
        self.M = M
        self.psf_poly_params = np.copy(poly_params[0])
        self.amplitude_params = np.copy(amplitude_params)
        self.amplitude_params_err = np.array([np.sqrt(np.abs(cov_matrix[x, x])) for x in range(self.Nx)])
        self.amplitude_cov_matrix = np.copy(cov_matrix)

        # Compute the model
        self.model = M @ amplitude_params
        self.model_err = np.zeros_like(self.model)

        return self.pixels, self.model, self.model_err

    def jacobian(self, params, epsilon, model_input=None):
        if model_input is not None:
            lambdas, model, model_err = model_input
        else:
            lambdas, model, model_err = self.simulate(*params)
        model = model.flatten()
        J = np.zeros((params.size, model.size))
        method = copy.copy(self.amplitude_priors_method)
        self.amplitude_priors_method = "keep"
        for ip, p in enumerate(params):
            if self.params.fixed[ip]:
                continue
            if ip >= self.psf_params_start_index[0]:
                continue
            tmp_p = np.copy(params)
            if tmp_p[ip] + epsilon[ip] < self.params.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.params.bounds[ip][1]:
                epsilon[ip] = - epsilon[ip]
            tmp_p[ip] += epsilon[ip]
            tmp_lambdas, tmp_model, tmp_model_err = self.simulate(*tmp_p)
            J[ip] = (tmp_model.flatten() - model) / epsilon[ip]
        self.amplitude_priors_method = method
        for k, order in enumerate(self.diffraction_orders):
            if self.psf_profile_params[order] is None:
                continue
            start = self.psf_params_start_index[k]
            profile_params = np.copy(self.psf_profile_params[order])
            amplitude_params = np.copy(self.amplitude_params)
            profile_params[:, 0] *= amplitude_params
            J[start:start+len(self.psf_poly_params)] = self.spectrum.chromatic_psf.build_psf_jacobian(self.pixels, profile_params=profile_params,
                                                                                                      psf_cube_sparse_indices=self.psf_cube_sparse_indices[order],
                                                                                                      boundaries=self.boundaries[order], dtype="float32")
        return J

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
        >>> spec = Spectrum("./tests/data/sim_20170530_134_spectrum.fits")
        >>> w = FullForwardModelFitWorkspace(spectrum=spec, amplitude_priors_method="spectrum", verbose=True)
        >>> y, mod, mod_err = w.simulate(*w.params.values)
        >>> dA_dtheta = w.amplitude_derivatives()
        >>> print(np.array(dA_dtheta).shape, w.amplitude_params.shape)
        (26, 669) (669,)

        """
        # compute matrices without derivatives
        WM = sparse.dia_matrix((self.W, 0), shape=(self.W.size, self.W.size), dtype="float32") @ self.M
        WD = (self.W * (self.data + (1 - self.params.values[self.params.get_index("B")]) * self.bgd_flat)).astype("float32")
        MWD = self.M.T @ WD
        if self.amplitude_priors_method == "spectrum":
              MWD += np.float32(self.reg) * self.Q_dot_A0
        # compute list of partial derivatives of model matrix M for all diffraction orders
        dM_dtheta = []
        Jpsf_indices = []
        for k, order in enumerate(self.diffraction_orders):
            if self.psf_profile_params[order] is None:
                continue
            profile_params = np.copy(self.psf_profile_params[order])
            dMsparse_order = self.spectrum.chromatic_psf.build_sparse_dM(self.pixels, profile_params=profile_params,
                                                                         M_sparse_indices=self.M_sparse_indices[order],
                                                                         boundaries=self.boundaries[order], dtype="float32")
            dM_dtheta += dMsparse_order
            start = self.psf_params_start_index[k]
            Jpsf_indices += list(range(start, start+len(dMsparse_order)))
        # compute partial derivatives of amplitude vector A
        nparams = len(dM_dtheta)
        dMWD_dtheta = [dM_dtheta[ip].T @ WD for ip in range(nparams)]
        dMWM_rQA_dtheta = [2 * dM_dtheta[ip].T @ WM for ip in range(nparams)]
        dcov_dtheta = [-self.amplitude_cov_matrix @ (dMWM_rQA_dtheta[ip] @ self.amplitude_cov_matrix) for ip in range(nparams)]
        dA_dtheta = [self.amplitude_cov_matrix @ dMWD_dtheta[ip] + dcov_dtheta[ip] @ MWD for ip in range(nparams)]
        return dA_dtheta

    def plot_spectrogram_comparison_simple(self, ax, title='', extent=None, dispersion=False):  # pragma: no cover
        """Method to plot a spectrogram issued from data and compare it with simulations.

        Parameters
        ----------
        ax: Axes
            Axes instance of shape (4, 2).
        title: str, optional
            Title for the simulation plot (default: '').
        extent: array_like, optional
            Extent argument for imshow to crop plots (default: None).
        dispersion: bool, optional
            If True, plot a colored bar to see the associated wavelength color along the x axis (default: False).
        """
        cmap_bwr = copy.copy(cm.get_cmap('bwr'))
        cmap_bwr.set_bad(color='lightgrey')
        cmap_viridis = copy.copy(cm.get_cmap('viridis'))
        cmap_viridis.set_bad(color='lightgrey')

        data = np.copy(self.data_before_mask)
        if len(self.outliers) > 0 or len(self.mask) > 0:
            bad_indices = np.array(list(self.get_bad_indices()) + list(self.mask)).astype(int)
            data[bad_indices] = np.nan

        lambdas = self.spectrum.lambdas
        sub = np.where((lambdas > parameters.LAMBDA_MIN) & (lambdas < parameters.LAMBDA_MAX))[0]
        sub = np.where(sub < self.spectrum.spectrogram.shape[1])[0]
        data = (data + self.bgd_flat).reshape((self.Ny, self.Nx))
        err = self.err.reshape((self.Ny, self.Nx))
        model = (self.model + self.params["B"] * self.bgd_flat).reshape((self.Ny, self.Nx))
        if extent is not None:
            sub = np.where((lambdas > extent[0]) & (lambdas < extent[1]))[0]
        if len(sub) > 0:
            norm = np.nanmax(data[:, sub])
            plot_image_simple(ax[0, 0], data=data[:, sub] / norm, title='Data', aspect='auto',
                              cax=ax[0, 1], vmin=0, vmax=1, units='1/max(data)', cmap=cmap_viridis)
            ax[0, 0].set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
            plot_image_simple(ax[1, 0], data=model[:, sub] / norm, aspect='auto', cax=ax[1, 1], vmin=0, vmax=1,
                              units='1/max(data)', cmap=cmap_viridis)
            if dispersion:
                x = self.spectrum.chromatic_psf.table['Dx'][sub[5:-5]] + self.spectrum.spectrogram_x0 - sub[0]
                y = np.ones_like(x)
                ax[1, 0].scatter(x, y, cmap=from_lambda_to_colormap(self.lambdas[sub[5:-5]]), edgecolors='None',
                                 c=self.lambdas[sub[5:-5]],
                                 label='', marker='o', s=10)
                ax[1, 0].set_xlim(0, model[:, sub].shape[1])
                ax[1, 0].set_ylim(0, model[:, sub].shape[0])
            # p0 = ax.plot(lambdas, self.model(lambdas), label='model')
            # # ax.plot(self.lambdas, self.model_noconv, label='before conv')
            if title != '':
                ax[1, 0].set_title(title, fontsize=10, loc='center', color='white', y=0.8)
            residuals = (data - model)
            # residuals_err = self.spectrum.spectrogram_err / self.model
            norm = err
            residuals /= norm
            std = float(np.nanstd(residuals[:, sub]))
            plot_image_simple(ax[2, 0], data=residuals[:, sub], vmin=-5 * std, vmax=5 * std, title='(Data-Model)/Err',
                              aspect='auto', cax=ax[2, 1], units='', cmap=cmap_bwr)
            ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, loc='center', color='black', y=0.8)
            ax[2, 0].text(0.05, 0.05,
                          f'mean={np.nanmean(residuals[:, sub]):.3f}\nstd={np.nanstd(residuals[:, sub]):.3f}',
                          horizontalalignment='left', verticalalignment='bottom',
                          color='black', transform=ax[2, 0].transAxes)
            ax[0, 0].set_xticks(ax[2, 0].get_xticks()[1:-1])
            ax[0, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[1, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[2, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[3, 1].remove()
            ax[3, 0].plot(self.lambdas[sub], np.nansum(data, axis=0)[sub], label='Data')
            model[np.isnan(data)] = np.nan  # mask background values outside fitted region
            ax[3, 0].plot(self.lambdas[sub], np.nansum(model, axis=0)[sub], label='Model')
            ax[3, 0].set_ylabel('Cross spectrum')
            ax[3, 0].set_xlabel(r'$\lambda$ [nm]')
            ax[3, 0].legend(fontsize=7)
            ax[3, 0].grid(True)

    def plot_fit(self):  # pragma: no cover
        """Plot the fit result.

        Examples
        --------

        >>> spec = Spectrum('tests/data/sim_20170530_134_spectrum.fits')
        >>> w = FullForwardModelFitWorkspace(spec, verbose=True, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.params.values)
        >>> w.plot_fit()

        .. plot::
            :include-source:

            from spectractor.fit.fit_spectrogram import SpectrogramFitWorkspace
            file_name = 'tests/data/reduc_20170530_134_spectrum.fits'
            atmgrid_file_name = file_name.replace('spectrum', 'atmsim')
            fit_workspace = SpectrogramFitWorkspace(file_name, atmgrid_file_name=atmgrid_file_name, verbose=True)
            lambdas, model, model_err = fit_workspace.simulation.simulate(*w.params.values)
            fit_workspace.lambdas = lambdas
            fit_workspace.model = model
            fit_workspace.model_err = model_err
            fit_workspace.plot_fit()

        """
        if np.max(self.lambdas) < 730:
            gs_kw = dict(width_ratios=[1, 0.15], height_ratios=[1, 1, 1, 1])
            fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(8, 10), gridspec_kw=gs_kw)
        elif 800 < np.max(self.lambdas) < 950:
            gs_kw = dict(width_ratios=[3, 0.15, 1, 0.15], height_ratios=[1, 1, 1, 1])
            fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(14, 10), gridspec_kw=gs_kw)
        else:
            gs_kw = dict(width_ratios=[3, 0.15, 1, 0.15, 1, 0.15], height_ratios=[1, 1, 1, 1])
            fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(18, 10), gridspec_kw=gs_kw)

        # main plot
        self.plot_spectrogram_comparison_simple(ax[:, 0:2], title='Spectrogram model', dispersion=True)
        # zoom O2
        if np.max(self.lambdas) > 800:
            self.plot_spectrogram_comparison_simple(ax[:, 2:4], extent=[730, 800], title='Zoom $O_2$', dispersion=True)
        # zoom H2O
        if np.max(self.lambdas) > 950:
            self.plot_spectrogram_comparison_simple(ax[:, 4:6], extent=[870, min(1000, int(np.max(self.lambdas)))], title='Zoom $H_2 O$', dispersion=True)
        # for i in range(ax.shape[0]-1):  # clear middle colorbars
        #     for j in range(ax.shape[1]//2-1):
        #         plt.delaxes(ax[i, 2 * j + 1])
        for i in range(ax.shape[0]):  # clear middle y axis labels
            for j in range(1, ax.shape[1]//2):
                ax[i, 2 * j].set_ylabel("")
        fig.tight_layout()
        if parameters.LSST_SAVEFIGPATH:
            figname = os.path.join(parameters.LSST_SAVEFIGPATH, f'ffm_bestfit.pdf')
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')
            gs_kw = dict(width_ratios=[3, 0.15], height_ratios=[1, 1, 1, 1])
            fig2, ax2 = plt.subplots(nrows=4, ncols=2, figsize=(10, 12), gridspec_kw=gs_kw)
            self.plot_spectrogram_comparison_simple(ax2, title='Spectrogram model', dispersion=True)
            # plt.delaxes(ax2[3, 1])
            fig2.tight_layout()
            figname = os.path.join(parameters.LSST_SAVEFIGPATH, f'ffm_bestfit_2.pdf')
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig2.savefig(figname, dpi=100, bbox_inches='tight')
        if self.live_fit:
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            if parameters.DISPLAY and self.verbose:
                plt.show()

    def adjust_spectrogram_position_parameters(self):
        # fit the spectrogram trace
        epsilon = 1e-4 * self.params.values
        epsilon[epsilon == 0] = 1e-4
        fixed_default = np.copy(self.params.fixed)
        self.params.fixed = [True] * len(self.params.values)
        self.params.fixed[self.params.get_index(r"shift_y [pix]")] = False  # shift y
        self.params.fixed[self.params.get_index(r"angle [deg]")] = False  # angle
        run_minimisation(self, "newton", epsilon, xtol=1e-2, ftol=0.01, with_line_search=False)  # 1000 / self.data.size)
        self.params.fixed = fixed_default
        self.set_mask(params=self.params.values, fwhmx_clip=3 * parameters.PSF_FWHM_CLIP, fwhmy_clip=parameters.PSF_FWHM_CLIP)


def run_ffm_minimisation(w, method="newton", niter=2):
    """Interface function to fit spectrogram simulation parameters to data.

    Parameters
    ----------
    w: FullForwardModelFitWorkspace
        An instance of the SpectrogramFitWorkspace class.
    method: str, optional
        Fitting method (default: 'newton').
    niter: int, optional
        Number of FFM iterations to final result (default: 2).

    Returns
    -------
    spectrum: Spectrum
        The extracted spectrum.

    Examples
    --------

    >>> spec = Spectrum("./tests/data/sim_20170530_134_spectrum.fits")
    >>> parameters.VERBOSE = True
    >>> w = FullForwardModelFitWorkspace(spec, verbose=True, plot=True, live_fit=True, amplitude_priors_method="spectrum")
    >>> spec = run_ffm_minimisation(w, method="newton")  # doctest: +ELLIPSIS
    >>> if 'LBDAS_T' in spec.header: plot_comparison_truth(spec, w)

    .. doctest:
        :hide:

        >>> assert w.costs[-1] / w.data.size < 1.22  # reduced chisq
        >>> assert np.isclose(w.params[r"angle [deg]"], -1.56, rtol=0.05)  # angle
        >>> assert np.isclose(w.params["B"], 1, rtol=0.05)  # B

    """
    my_logger = set_logger(__name__)
    my_logger.info(f"\n\tStart FFM with adjust_spectrogram_position_parameters.")
    w.adjust_spectrogram_position_parameters()

    if method != "newton":
        run_minimisation(w, method=method)
    else:
        costs = np.array([w.chisq(w.params.values)])
        if parameters.DISPLAY and (parameters.DEBUG or w.live_fit):
            w.plot_fit()
        start = time.time()
        my_logger.info(f"\tStart guess:\n\t" + '\n\t'.join([f'{w.params.labels[k]}: {w.params.values[k]} (fixed={w.params.fixed[k]})' for k in range(w.params.ndim)]))
        epsilon = 1e-4 * w.params.values
        epsilon[epsilon == 0] = 1e-4

        run_minimisation(w, method=method, xtol=1e-3, ftol=1e-2, with_line_search=False)  # 1000 / (w.data.size - len(w.mask)))
        if parameters.DEBUG and parameters.DISPLAY:
            w.plot_fit()

        weighted_mean_fwhm = np.average(w.spectrum.chromatic_psf.table['fwhm'], weights=w.spectrum.chromatic_psf.table['amplitude'])
        my_logger.info(f"\n\tMean FWHM: {weighted_mean_fwhm} pixels (weighted with spectrum amplitude)")
        if parameters.DEBUG:  # pragma: no cover
            fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex="all")
            ax.plot(w.spectrum.lambdas, np.array(w.spectrum.chromatic_psf.table['fwhm']), label=f"weighted mean={weighted_mean_fwhm} pix")
            ax.set_xlabel(r"$\lambda$ [nm]")
            ax.set_ylabel("Transverse FWHM [pixels]")
            ax.set_ylim((0.8 * np.min(w.spectrum.chromatic_psf.table['fwhm']), 1.2 * np.max(w.spectrum.chromatic_psf.table['fwhm'])))  # [-10:])))
            ax.grid()
            ax.legend()
            if parameters.DISPLAY:
                plt.show()
            if parameters.LSST_SAVEFIGPATH:
                fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'fwhm_2.pdf'))

        # Optimize the regularisation parameter only if it was not done before
        if w.amplitude_priors_method == "spectrum" and w.reg == parameters.PSF_FIT_REG_PARAM:  # pragma: no cover
            my_logger.info("\n\tStart regularization parameter estimation...")
            w_reg = RegFitWorkspace(w, opt_reg=parameters.PSF_FIT_REG_PARAM, verbose=True)
            w_reg.run_regularisation()
            w.opt_reg = w_reg.opt_reg
            w.reg = np.copy(w_reg.opt_reg)
            w.simulate(*w.params.values)
            if np.trace(w.amplitude_cov_matrix) < np.trace(w.amplitude_priors_cov_matrix):
                w.my_logger.warning(
                    f"\n\tTrace of final covariance matrix ({np.trace(w.amplitude_cov_matrix)}) is "
                    f"below the trace of the prior covariance matrix "
                    f"({np.trace(w.amplitude_priors_cov_matrix)}). This is probably due to a very "
                    f"high regularisation parameter in case of a bad fit. Therefore the final "
                    f"covariance matrix is multiplied by the ratio of the traces and "
                    f"the amplitude parameters are very close the amplitude priors.")
                r = np.trace(w.amplitude_priors_cov_matrix) / np.trace(w.amplitude_cov_matrix)
                w.amplitude_cov_matrix *= r
                w.amplitude_params_err = np.array(
                    [np.sqrt(w.amplitude_cov_matrix[x, x]) for x in range(w.Nx)])
            w.spectrum.header['PSF_REG'] = w.opt_reg
            w.spectrum.header['TRACE_R'] = np.trace(w_reg.resolution)

        if parameters.DEBUG and parameters.DISPLAY:
            w.plot_fit()

        my_logger.info(f"\n\tStart run_minimisation_sigma_clipping "
                       f"with sigma={parameters.SPECTRACTOR_DECONVOLUTION_SIGMA_CLIP}.")
        for i in range(niter):
            w.set_mask(params=w.params.values, fwhmx_clip=3 * parameters.PSF_FWHM_CLIP, fwhmy_clip=parameters.PSF_FWHM_CLIP)
            run_minimisation_sigma_clipping(w, "newton", epsilon, xtol=1e-5,
                                            ftol=1e-3, niter_clip=3,  # ftol=100 / (w.data.size - len(w.mask))
                                            sigma_clip=parameters.SPECTRACTOR_DECONVOLUTION_SIGMA_CLIP, verbose=True,
                                            with_line_search=False)
            my_logger.info(f"\n\t  niter = {i} : Newton: total computation time: {time.time() - start}s")
            if parameters.DEBUG and parameters.DISPLAY:
                w.plot_fit()

            # recompute angle and dy0 if fixed while y_c parameters are free
            # if w.fixed[3] and w.fixed[4] and not np.any([w.fixed[k] for k, par in enumerate(w.input_labels) if "y_c" in par]):
            #     pval_leg = [w.p[k] for k, par in enumerate(w.input_labels) if "y_c" in par][
            #                :w.spectrum.chromatic_psf.degrees["y_c"] + 1]
            #     pval_poly = np.polynomial.legendre.leg2poly(pval_leg)
            #     new_dy0, new_angle = w.p[2], w.p[4]
            #     from numpy.polynomial import Polynomial as P
            #     p = P(pval_poly)
            #     pX = P([0, 0.5 * (w.spectrum.spectrogram_Nx)])
            #     pfinal = p(pX)
            #     pval_poly = pfinal.coef
            #     for k in range(pval_poly.size):
            #         if k == 0:
            #             new_dy0 += pval_poly[k]
            #         if k == 1:
            #             new_angle += np.arctan(pval_poly[k]) * 180 / np.pi

            w.spectrum.lambdas = np.copy(w.lambdas)
            w.spectrum.chromatic_psf.table['lambdas'] = np.copy(w.lambdas)
            w.spectrum.data = np.copy(w.amplitude_params)
            w.spectrum.err = np.copy(w.amplitude_params_err)
            w.spectrum.cov_matrix = np.copy(w.amplitude_cov_matrix)
            w.spectrum.chromatic_psf.fill_table_with_profile_params(w.psf_profile_params[w.diffraction_orders[0]])
            w.spectrum.chromatic_psf.table["amplitude"] = np.copy(w.amplitude_params)
            w.spectrum.chromatic_psf.from_profile_params_to_shape_params(w.psf_profile_params[w.diffraction_orders[0]])
            w.spectrum.chromatic_psf.params.values = w.spectrum.chromatic_psf.from_table_to_poly_params()
            w.spectrum.spectrogram_fit = w.model
            w.spectrum.spectrogram_residuals = (w.data - w.spectrum.spectrogram_fit) / w.err
            w.spectrum.header['CHI2_FIT'] = w.costs[-1] / (w.data.size - len(w.mask))
            w.spectrum.header['PIXSHIFT'] = w.params[r"shift_x [pix]"]
            w.spectrum.header['D2CCD'] = w.params[r"D_CCD [mm]"]
            if len(w.diffraction_orders) >= 2:
                w.spectrum.header['A2_FIT'] = w.params.values[w.diffraction_orders[1]]
            w.spectrum.header["ROTANGLE"] = w.params[r"angle [deg]"]
            w.spectrum.header["AM_FIT"] = w.params["z"]
            # Compute next order contamination
            if len(w.diffraction_orders) >= 2 and w.tr[1]:
                w.spectrum.data_next_order = w.params.values[w.diffraction_orders[1]] * w.amplitude_params * w.tr[1](w.lambdas)
                w.spectrum.err_next_order = np.abs(w.params.values[w.diffraction_orders[1]] * w.amplitude_params_err * w.tr[1](w.lambdas))

            # Calibrate the spectrum
            calibrate_spectrum(w.spectrum, with_adr=True, grid_search=False)
            w.params.set(r"D_CCD [mm]", w.spectrum.disperser.D)
            w.params.set(r"shift_x [pix]", w.spectrum.header['PIXSHIFT'])
            w.spectrum.convert_from_flam_to_ADUrate()

        if w.filename != "":
            parameters.SAVE = True
            w.params.plot_correlation_matrix()
            write_fitparameter_json(w.params.json_filename, w.params,
                                    extra={"chi2": costs[-1] / (w.data.size - len(w.outliers) - len(w.mask)),
                                           "date-obs": w.spectrum.date_obs})
            w.plot_fit()
            parameters.SAVE = False

    # Propagate parameters
    A1, A2, A3, D2CCD, dx0, dy0, angle, B, rot, pressure, temperature, airmass, *poly_params_all = w.params.values
    w.spectrum.rotation_angle = angle
    w.spectrum.spectrogram_bgd *= B
    w.spectrum.spectrogram_bgd_rms *= B
    w.spectrum.spectrogram_x0 += dx0
    w.spectrum.spectrogram_y0 += dy0
    w.spectrum.x0[0] += dx0
    w.spectrum.x0[1] += dy0
    w.spectrum.header["TARGETX"] = w.spectrum.x0[0]
    w.spectrum.header["TARGETY"] = w.spectrum.x0[1]
    w.spectrum.header['MEANFWHM'] = np.mean(np.array(w.spectrum.chromatic_psf.table['fwhm']))

    # Convert to flam
    w.spectrum.convert_from_ADUrate_to_flam()

    # Compare with truth if available
    if 'LBDAS_T' in w.spectrum.header and parameters.DEBUG:
        plot_comparison_truth(w.spectrum, w)

    return w.spectrum


def SpectractorInit(file_name, target_label='', disperser_label="", config=''):
    """ Spectractor initialisation: load the config parameters and build the Image instance.

    Parameters
    ----------
    file_name: str
        Input file nam of the image to analyse.
    target_label: str, optional
        The name of the targeted object (default: "").
    disperser_label: str, optional
        The name of the disperser (default: "").
    config: str
        The config file name (default: "").

    Returns
    -------
    image: Image
        The prepared Image instance ready for spectrum extraction.

    Examples
    --------

    Extract the spectrogram and its characteristics from the image:

    .. doctest::

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> logbook = LogBook(logbook='./tests/data/ctiofulllogbook_jun2017_v5.csv')
        >>> file_names = ['./tests/data/reduc_20170530_134.fits']
        >>> for file_name in file_names:
        ...     tag = file_name.split('/')[-1]
        ...     disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        ...     if target_label is None or xpos is None or ypos is None:
        ...         continue
        ...     image = SpectractorInit(file_name, target_label=target_label,
        ...                             disperser_label=disperser_label, config='./config/ctio.ini')

    .. doctest::
        :hide:

        >>> assert image is not None

    """

    my_logger = set_logger(__name__)
    my_logger.info('\n\tSpectractor initialisation')
    # Load config file
    if config != "":
        load_config(config)
    if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
        ensure_dir(parameters.LSST_SAVEFIGPATH)

    # Load reduced image
    image = Image(file_name, target_label=target_label, disperser_label=disperser_label)
    return image


def SpectractorRun(image, output_directory, guess=None):
    """ Spectractor main function to extract a spectrum from an image

    Parameters
    ----------
    image: Image
        Input Image instance.
    output_directory: str
        Output directory.
    guess: [int,int], optional
        [x0,y0] list of the guessed pixel positions of the target in the image (must be integers). Mandatory if
        WCS solution is absent (default: None).

    Returns
    -------
    spectrum: Spectrum
        The extracted spectrum object.

    Examples
    --------

    Extract the spectrogram and its characteristics from the image:

    .. doctest::

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> logbook = LogBook(logbook='./tests/data/ctiofulllogbook_jun2017_v5.csv')
        >>> file_names = ['./tests/data/reduc_20170530_134.fits']
        >>> for file_name in file_names:
        ...     tag = file_name.split('/')[-1]
        ...     disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        ...     if target_label is None or xpos is None or ypos is None:
        ...         continue
        ...     image = SpectractorInit(file_name, target_label=target_label,
        ...                             disperser_label=disperser_label, config='./config/ctio.ini')
        ...     spectrum = SpectractorRun(image, './tests/data/', guess=[xpos, ypos])

    .. doctest::
        :hide:

        >>> assert spectrum is not None
        >>> assert os.path.isfile('tests/data/reduc_20170530_134_spectrum.fits')

    """

    my_logger = set_logger(__name__)
    my_logger.info('\n\tRun Spectractor')

    # Guess position of order 0
    if guess is not None and image.target_guess is None:
        image.target_guess = np.asarray(guess)
    if image.target_guess is None:
        from scipy.signal import medfilt2d
        data = medfilt2d(image.data.T, kernel_size=9)
        image.target_guess = np.unravel_index(np.argmax(data), data.shape)
        my_logger.info(f"\n\tNo guess position of order 0 has been given. Assuming the spectrum to extract comes "
                       f"from the brightest object, guess position is set as {image.target_guess}.")
    if parameters.DEBUG:
        image.plot_image(scale='symlog', title="before rebinning", target_pixcoords=image.target_guess, cmap='gray', vmax=1e3)

    # Use fast mode
    if parameters.CCD_REBIN > 1:
        my_logger.info('\n\t  ======================= REBIN =============================')
        image.rebin()
        if parameters.DEBUG:
            image.plot_image(scale='symlog', title="after rebinning ", target_pixcoords=image.target_guess)

    # Set output path
    ensure_dir(output_directory)
    output_filename = os.path.basename(image.file_name)
    output_filename = output_filename.replace('.fits', '_spectrum.fits')
    output_filename = output_filename.replace('.fz', '_spectrum.fits')
    output_filename = os.path.join(output_directory, output_filename)
    # Find the exact target position in the raw cut image: several methods
    my_logger.info(f'\n\tSearch for the target in the image with guess={image.target_guess}...')
    find_target(image, image.target_guess, widths=(parameters.XWINDOW, parameters.YWINDOW))
    # Rotate the image
    turn_image(image)
    # Find the exact target position in the rotated image: several methods
    my_logger.info('\n\tSearch for the target in the rotated image...')

    find_target(image, image.target_guess, rotated=True, widths=(parameters.XWINDOW_ROT,
                                                                 parameters.YWINDOW_ROT))
    # Create Spectrum object
    spectrum = Spectrum(image=image, order=parameters.SPEC_ORDER)
    # First 1D spectrum extraction and background extraction

    my_logger.info('\n\t ======================== PSF1D Extraction ====================================')
    w_psf1d, bgd_model_func = extract_spectrum_from_image(image, spectrum, signal_width=parameters.PIXWIDTH_SIGNAL,
                                                          ws=(parameters.PIXDIST_BACKGROUND,
                                                              parameters.PIXDIST_BACKGROUND
                                                              + parameters.PIXWIDTH_BACKGROUND),
                                                          right_edge=image.data.shape[1])

    # PSF2D deconvolution
    if parameters.SPECTRACTOR_DECONVOLUTION_PSF2D:
        my_logger.info('\n\t ========================== PSF2D DECONVOLUTION  ===============================')
        run_spectrogram_deconvolution_psf2d(spectrum, bgd_model_func=bgd_model_func)

    # Calibrate the spectrum
    my_logger.info(f'\n\tCalibrating order {spectrum.order:d} spectrum...')
    with_adr = True
    if parameters.OBS_OBJECT_TYPE != "STAR":
        with_adr = False
    calibrate_spectrum(spectrum, with_adr=with_adr, grid_search=True)
    spectrum.data_next_order = np.zeros_like(spectrum.lambdas)
    spectrum.err_next_order = np.zeros_like(spectrum.lambdas)

    # Full forward model extraction: add transverse ADR and order 2 subtraction
    if parameters.SPECTRACTOR_DECONVOLUTION_FFM:
        my_logger.info('\n\t  ======================= FFM DECONVOLUTION =============================')
        w = FullForwardModelFitWorkspace(spectrum, verbose=parameters.VERBOSE, plot=True, live_fit=False,
                                         amplitude_priors_method="spectrum")
        spectrum = run_ffm_minimisation(w, method="newton", niter=2)

    # Save the spectrum
    my_logger.info('\n\t  ======================= SAVE SPECTRUM =============================')
    spectrum.save_spectrum(output_filename, overwrite=True)
    spectrum.lines.table = spectrum.lines.build_detected_line_table(amplitude_units=spectrum.units)

    # Plot the spectrum
    if parameters.VERBOSE and parameters.DISPLAY:
        spectrum.plot_spectrum(xlim=None)

    return spectrum


def Spectractor(file_name, output_directory, target_label='', guess=None, disperser_label="", config=''):

    """ Spectractor main function to extract a spectrum from a FITS file.

    Parameters
    ----------
    file_name: str
        Input file nam of the image to analyse.
    output_directory: str
        Output directory.
    target_label: str, optional
        The name of the targeted object (default: "").
    guess: [int,int], optional
        [x0,y0] list of the guessed pixel positions of the target in the image (must be integers). Mandatory if
        WCS solution is absent (default: None).
    disperser_label: str, optional
        The name of the disperser (default: "").
    config: str
        The config file name (default: "").

    Returns
    -------
    spectrum: Spectrum
        The extracted spectrum object.

    Examples
    --------

    Extract the spectrogram and its characteristics from the image:

    .. doctest::

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> logbook = LogBook(logbook='./tests/data/ctiofulllogbook_jun2017_v5.csv')
        >>> file_names = ['./tests/data/reduc_20170530_134.fits']
        >>> for file_name in file_names:
        ...     tag = file_name.split('/')[-1]
        ...     disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        ...     if target_label is None or xpos is None or ypos is None:
        ...         continue
        ...     spectrum = Spectractor(file_name, './tests/data/', guess=[xpos, ypos], target_label=target_label,
        ...                            disperser_label=disperser_label, config='./config/ctio.ini')

    .. doctest::
        :hide:

        >>> assert spectrum is not None
        >>> assert os.path.isfile('tests/data/reduc_20170530_134_spectrum.fits')

    """
    image = SpectractorInit(file_name, target_label=target_label, disperser_label=disperser_label, config=config)
    spectrum = SpectractorRun(image, guess=guess, output_directory=output_directory)
    return spectrum


def extract_spectrum_from_image(image, spectrum, signal_width=10, ws=(20, 30), right_edge=parameters.CCD_IMSIZE):
    """Extract the 1D spectrum from the image.

    Method : remove a uniform background estimated from the rectangular lateral bands

    The spectrum amplitude is the sum of the pixels in the 2*w rectangular window
    centered on the order 0 y position.
    The up and down backgrounds are estimated as the median in rectangular regions
    above and below the spectrum, in the ws-defined rectangular regions; stars are filtered
    as nan values using an hessian analysis of the image to remove structures.
    The subtracted background is the mean of the two up and down backgrounds.
    Stars are filtered.

    Prerequisites: the target position must have been found before, and the
        image turned to have an horizontal dispersion line

    Parameters
    ----------
    image: Image
        Image object from which to extract the spectrum
    spectrum: Spectrum
        Spectrum object to store new wavelengths, data and error arrays
    signal_width: int
        Half width of central region where the spectrum is extracted and summed (default: 10)
    ws: list
        up/down region extension where the sky background is estimated with format [int, int] (default: [20,30])
    right_edge: int
        Right-hand pixel position above which no pixel should be used (default: parameters.CCD_IMSIZE)
    """

    my_logger = set_logger(__name__)
    if ws is None:
        ws = [signal_width + 20, signal_width + 30]

    my_logger.info('\n\t  ======================= extract_spectrum_from_image =============================')
    my_logger.info(
        f'\n\tExtracting spectrum from image: spectrum with width 2*{signal_width:.0f} pixels '
        f'and background from {ws[0]:.0f} to {ws[1]:.0f} pixels')

    # Make a data copy
    data = np.copy(image.data_rotated)#[:, 0:right_edge]
    err = np.copy(image.stat_errors_rotated)#[:, 0:right_edge]

    # Lateral bands to remove sky background
    Ny, Nx = data.shape
    y0 = int(image.target_pixcoords_rotated[1])
    ymax = min(Ny, y0 + ws[1])
    ymin = max(0, y0 - ws[1])

    # Roughly estimates the wavelengths and set start 0 nm before parameters.LAMBDA_MIN
    # and end 0 nm after parameters.LAMBDA_MAX
    if spectrum.order < 0:
        distance = np.sign(spectrum.order) * (np.arange(Nx) - image.target_pixcoords_rotated[0])
        lambdas = image.disperser.grating_pixel_to_lambda(distance, x0=image.target_pixcoords, order=spectrum.order)
        lambda_min_index = int(np.argmin(np.abs(lambdas[::np.sign(spectrum.order)] - parameters.LAMBDA_MIN)))
        lambda_max_index = int(np.argmin(np.abs(lambdas[::np.sign(spectrum.order)] - parameters.LAMBDA_MAX)))
        xmin = max(0, int(distance[lambda_min_index]))
        xmax = min(right_edge, int(distance[lambda_max_index]) + 1)  # +1 to  include edges
    else:
        lambdas = image.disperser.grating_pixel_to_lambda(np.arange(Nx) - image.target_pixcoords_rotated[0],
                                                          x0=image.target_pixcoords,
                                                          order=spectrum.order)
        xmin = int(np.argmin(np.abs(lambdas - parameters.LAMBDA_MIN)))
        xmax = int(np.argmin(np.abs(lambdas - parameters.LAMBDA_MAX)))

    # Create spectrogram
    data = data[ymin:ymax, xmin:xmax]
    err = err[ymin:ymax, xmin:xmax]
    Ny, Nx = data.shape
    my_logger.info(f'\n\tExtract spectrogram: crop rotated image [{xmin}:{xmax},{ymin}:{ymax}] (size ({Nx}, {Ny}))')

    # Position of the order 0 in the spectrogram coordinates
    target_pixcoords_spectrogram = [image.target_pixcoords_rotated[0] - xmin, image.target_pixcoords_rotated[1] - ymin]

    # Extract the background on the rotated image
    def bgd_model_func(x, y):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        return np.zeros((y.size, x.size))
    if parameters.SPECTRACTOR_BACKGROUND_SUBTRACTION:
        bgd_model_func, bgd_res, bgd_rms = extract_spectrogram_background_sextractor(data, err, ws=ws, mask_signal_region=True)
        # while np.nanmean(bgd_res)/np.nanstd(bgd_res) < -0.2 and parameters.PIXWIDTH_BOXSIZE >= 5:
        while (np.abs(np.nanmean(bgd_res)) > 0.5 or np.nanstd(bgd_res) > 1.3) and parameters.PIXWIDTH_BOXSIZE > 5:
            parameters.PIXWIDTH_BOXSIZE = max(5, parameters.PIXWIDTH_BOXSIZE // 2)
            my_logger.debug(f"\n\tPull distribution of background residuals differs too much from mean=0 and std=1. "
                            f"\n\t\tmean={np.nanmean(bgd_res):.3g}; std={np.nanstd(bgd_res):.3g}"
                            f"\n\tThese value should be smaller in absolute value than 0.5 and 1.3. "
                            f"\n\tTo do so, parameters.PIXWIDTH_BOXSIZE is divided "
                            f"by 2 from {parameters.PIXWIDTH_BOXSIZE * 2} -> {parameters.PIXWIDTH_BOXSIZE}.")
            bgd_model_func, bgd_res, bgd_rms = extract_spectrogram_background_sextractor(data, err, ws=ws, mask_signal_region=True)

        # Propagate background uncertainties
        err = np.sqrt(err * err + bgd_rms * bgd_rms)

    # Fit the transverse profile
    my_logger.info('\n\t  ======================= Fit the transverse profile =============================')

    my_logger.info(f'\n\tStart PSF1D transverse fit...')
    psf = load_PSF(psf_type=parameters.PSF_TYPE, target=image.target, clip=False)
    s = ChromaticPSF(psf, Nx=Nx, Ny=Ny, x0=target_pixcoords_spectrogram[0], y0=target_pixcoords_spectrogram[1],
                     deg=parameters.PSF_POLY_ORDER, saturation=image.saturation)
    verbose = copy.copy(parameters.VERBOSE)
    debug = copy.copy(parameters.DEBUG)
    parameters.VERBOSE = False
    parameters.DEBUG = False
    s.fit_transverse_PSF1D_profile(data, err, signal_width, ws, pixel_step=parameters.PSF_PIXEL_STEP_TRANSVERSE_FIT,
                                   sigma_clip=5, bgd_model_func=bgd_model_func, saturation=image.saturation,
                                   live_fit=False)
    parameters.VERBOSE = verbose
    parameters.DEBUG = debug

    # Fill spectrum object
    spectrum.pixels = np.arange(xmin, xmax, 1).astype(int)
    spectrum.data = np.copy(s.table['amplitude'])
    spectrum.err = np.copy(s.table['flux_err'])
    my_logger.debug(f'\n\tTransverse fit table:\n{s.table}')
    if parameters.DEBUG:
        s.plot_summary()

    # Fit the data:
    method = "noprior"
    mode = "1D"

    my_logger.info('\n\t  ======================= ChromaticPSF polynomial fit  =============================')

    my_logger.info(f'\n\tStart ChromaticPSF polynomial fit with '
                   f'mode={mode} and amplitude_priors_method={method}...')
    w = s.fit_chromatic_psf(data, bgd_model_func=bgd_model_func, data_errors=err,
                            amplitude_priors_method=method, mode=mode, verbose=parameters.VERBOSE, analytical=True)

    Dx_rot = spectrum.pixels.astype(float) - image.target_pixcoords_rotated[0]
    s.table['Dx'] = np.copy(Dx_rot)
    s.table['Dy'] = s.table['y_c'] - (image.target_pixcoords_rotated[1] - ymin)
    s.table['Dy_disp_axis'] = 0
    s.table['Dy_fwhm_inf'] = s.table['Dy'] - 0.5 * s.table['fwhm']
    s.table['Dy_fwhm_sup'] = s.table['Dy'] + 0.5 * s.table['fwhm']
    my_logger.debug(f"\n\tTransverse fit table before derotation:"
                    f"\n{s.table[['amplitude', 'x_c', 'y_c', 'Dx', 'Dy', 'Dy_disp_axis']]}")

    # Rotate, crop and save the table
    s.rotate_table(-image.rotation_angle)
    extra_pixels = int(np.max(s.table['Dx']) + image.target_pixcoords[0] - right_edge + 1)  # spectrum pixels outside CCD in rotated image
    new_Nx = len(s.table['Dx']) - extra_pixels
    if extra_pixels > 0:
        my_logger.info(f"\n\tCrop table of size {len(s.table)=} to {new_Nx=} first values "
                       f"to remove data fitted outside the CCD region in the rotated image.")
        s.crop_table(new_Nx=new_Nx)
    spectrum.data = np.copy(w.amplitude_params[:new_Nx])
    spectrum.err = np.copy(w.amplitude_params_err[:new_Nx])
    spectrum.cov_matrix = np.copy(w.amplitude_cov_matrix[:new_Nx, :new_Nx])
    spectrum.chromatic_psf = s

    # Extract the spectrogram edges
    data = np.copy(image.data)[:, 0:right_edge]
    err = np.copy(image.stat_errors)[:, 0:right_edge]
    Ny, Nx = data.shape
    x0 = int(image.target_pixcoords[0])
    y0 = int(image.target_pixcoords[1])
    ymax = min(Ny, y0 + int(s.table['Dy_disp_axis'].max()) + ws[1] + 1)  # +1 to  include edges
    ymin = max(0, y0 + int(s.table['Dy_disp_axis'].min()) - ws[1])
    distance = s.get_algebraic_distance_along_dispersion_axis()
    lambdas = image.disperser.grating_pixel_to_lambda(distance, x0=image.target_pixcoords, order=spectrum.order)
    lambda_min_index = int(np.argmin(np.abs(lambdas[::np.sign(spectrum.order)] - parameters.LAMBDA_MIN)))
    lambda_max_index = int(np.argmin(np.abs(lambdas[::np.sign(spectrum.order)] - parameters.LAMBDA_MAX)))
    xmin = max(0, int(s.table['Dx'][lambda_min_index] + x0))
    xmax = min(right_edge, int(s.table['Dx'][lambda_max_index] + x0) + 1)  # +1 to  include edges
    # Position of the order 0 in the spectrogram coordinates
    target_pixcoords_spectrogram = [image.target_pixcoords[0] - xmin, image.target_pixcoords[1] - ymin]
    s.y0 = target_pixcoords_spectrogram[1]
    s.x0 = target_pixcoords_spectrogram[0]

    # Update y_c and x_c after rotation
    s.table['y_c'] = s.table['Dy'] + target_pixcoords_spectrogram[1]
    s.table['x_c'] = s.table['Dx'] + target_pixcoords_spectrogram[0]
    my_logger.debug(f"\n\tTransverse fit table after derotation:"
                    f"\n{s.table[['amplitude', 'x_c', 'y_c', 'Dx', 'Dy', 'Dy_disp_axis']]}")

    # Create spectrogram
    data = data[ymin:ymax, xmin:xmax]
    err = err[ymin:ymax, xmin:xmax]
    Ny, Nx = data.shape
    my_logger.info(f'\n\tExtract spectrogram: crop raw image [{xmin}:{xmax},{ymin}:{ymax}] (size ({Nx}, {Ny}))')

    # Extract the non rotated background
    my_logger.info('\n\t  ======================= Extract the non rotated background  =============================')
    if parameters.SPECTRACTOR_BACKGROUND_SUBTRACTION:
        bgd_model_func, bgd_res, bgd_rms = extract_spectrogram_background_sextractor(data, err, ws=ws, Dy_disp_axis=s.table['y_c'])
        bgd = bgd_model_func(np.arange(Nx), np.arange(Ny))
        my_logger.info(f"\n\tBackground statistics: mean={np.nanmean(bgd):.3f} {image.units}, "
                   f"RMS={np.nanmean(bgd_rms):.3f} {image.units}.")

        # Propagate background uncertainties
        err = np.sqrt(err * err + bgd_rms * bgd_rms)
        spectrum.spectrogram_bgd = bgd
        spectrum.spectrogram_bgd_rms = bgd_rms

    # First guess for lambdas

    my_logger.info('\n\t  ======================= first guess for lambdas  =============================')

    first_guess_lambdas = image.disperser.grating_pixel_to_lambda(s.get_algebraic_distance_along_dispersion_axis(),
                                                                  x0=image.target_pixcoords, order=spectrum.order)
    s.table['lambdas'] = first_guess_lambdas
    spectrum.lambdas = np.array(first_guess_lambdas)

    # Position of the order 0 in the spectrogram coordinates
    my_logger.info(f'\n\tExtract spectrogram: crop image [{xmin}:{xmax},{ymin}:{ymax}] (size ({Nx}, {Ny}))'
                   f'\n\tNew target position in spectrogram frame: {target_pixcoords_spectrogram}')

    # Save results
    spectrum.spectrogram = data
    spectrum.spectrogram_err = err
    spectrum.spectrogram_x0 = target_pixcoords_spectrogram[0]
    spectrum.spectrogram_y0 = target_pixcoords_spectrogram[1]
    spectrum.spectrogram_xmin = xmin
    spectrum.spectrogram_xmax = xmax
    spectrum.spectrogram_ymin = ymin
    spectrum.spectrogram_ymax = ymax
    spectrum.spectrogram_Nx = Nx
    spectrum.spectrogram_Ny = Ny
    spectrum.spectrogram_deg = spectrum.chromatic_psf.deg
    spectrum.spectrogram_saturation = spectrum.chromatic_psf.saturation

    # Plot FHWM(lambda)
    if parameters.DEBUG:  # pragma: no cover
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex="all")
        ax[0].plot(spectrum.lambdas, np.array(s.table['fwhm']))
        ax[0].set_xlabel(r"$\lambda$ [nm]")
        ax[0].set_ylabel("Transverse FWHM [pixels]")
        ax[0].set_ylim((0.8 * np.min(s.table['fwhm']), 1.2 * np.max(s.table['fwhm'])))  # [-10:])))
        ax[0].grid()
        ax[1].plot(spectrum.lambdas, np.array(s.table['y_c']))
        ax[1].set_xlabel(r"$\lambda$ [nm]")
        ax[1].set_ylabel("Distance from mean dispersion axis [pixels]")
        # ax[1].set_ylim((0.8*np.min(s.table['Dy']), 1.2*np.max(s.table['fwhm'][-10:])))
        ax[1].grid()
        if parameters.DISPLAY:
            plt.show()
        if parameters.LSST_SAVEFIGPATH:
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'fwhm.pdf'))

    # Summary plot
    if parameters.DEBUG or parameters.LSST_SAVEFIGPATH:  # pragma: no cover
        gs_kw = dict(width_ratios=[3, 0.08], height_ratios=[1, 1])
        fig, ax = plt.subplots(2, 2, sharex='none', figsize=(16, 6), gridspec_kw=gs_kw)
        xx = np.arange(s.table['Dx'].size)
        plot_image_simple(ax[1, 0], data=data, scale="symlog", title='', units=image.units, aspect='auto', cax=ax[1, 1])
        ax[1, 0].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_disp_axis'], label='Dispersion axis', color="r")
        ax[1, 0].scatter(xx, target_pixcoords_spectrogram[1] + s.table['Dy'],
                         c=s.table['lambdas'], edgecolors='None', cmap=from_lambda_to_colormap(s.table['lambdas']),
                         label='Fitted spectrum centers', marker='o', s=10)
        ax[1, 0].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_fwhm_inf'], 'k-', label='Fitted FWHM')
        ax[1, 0].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_fwhm_sup'], 'k-', label='')
        # ax[1, 0].set_ylim(0.5 * Ny - signal_width, 0.5 * Ny + signal_width)
        ax[1, 0].set_xlim(0, xx.size)
        ax[1, 0].legend(loc='best')
        plot_spectrum_simple(ax[0, 0], spectrum.lambdas, spectrum.data, data_err=spectrum.err,
                             units=image.units, label='Fitted spectrum')
        ax[0, 0].plot(spectrum.lambdas, s.table['flux_sum'], 'k-', label='Cross spectrum')
        ax[1, 0].set_xlabel(r"$\lambda$ [nm]")
        ax[0, 0].legend(loc='best')
        fig.tight_layout()
        # fig.subplots_adjust(hspace=0)
        pos0 = ax[0, 0].get_position()
        pos1 = ax[1, 0].get_position()
        ax[0, 0].set_position([pos1.x0, pos0.y0, pos1.width, pos0.height])
        ax[0, 1].remove()
        if parameters.DISPLAY:
            plt.show()
        if parameters.LSST_SAVEFIGPATH:
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'intermediate_spectrum.pdf'))

    return w, bgd_model_func


def run_spectrogram_deconvolution_psf2d(spectrum, bgd_model_func):
    """Get the spectrum from a 2D PSF deconvolution of the spectrogram.

    Parameters
    ----------
    spectrum: Spectrum
    bgd_model_func: callable

    Returns
    -------
    w: ChromaticPSFFitWorkspace

    """
    my_logger = set_logger(__name__)
    s = spectrum.chromatic_psf
    Ny, Nx = spectrum.spectrogram.shape

    # build 1D priors
    Dx_rot = np.copy(s.table['Dx'])
    amplitude_priors = np.copy(s.table['amplitude'])
    amplitude_priors_err = np.copy(s.table['flux_err'])
    psf_poly_priors = s.from_table_to_poly_params()[s.Nx:]
    Dy_disp_axis = np.copy(s.table["Dy_disp_axis"])

    # initialize a new ChromaticPSF
    s = ChromaticPSF(s.psf, Nx=Nx, Ny=Ny, x0=spectrum.spectrogram_x0, y0=spectrum.spectrogram_y0,
                     deg=s.deg, saturation=s.saturation)

    # fill a first table with first guess
    s.table['Dx'] = (np.arange(spectrum.spectrogram_xmin, spectrum.spectrogram_xmax, 1)
                     - spectrum.x0[0])[:len(s.table['Dx'])]
    s.table["amplitude"] = np.interp(s.table['Dx'], Dx_rot, amplitude_priors)
    s.table["flux_sum"] = np.interp(s.table['Dx'], Dx_rot, amplitude_priors)
    s.table["flux_err"] = np.interp(s.table['Dx'], Dx_rot, amplitude_priors_err)
    s.table['Dy_disp_axis'] = np.interp(s.table['Dx'], Dx_rot, Dy_disp_axis)
    s.params.values = np.concatenate((s.table["amplitude"], psf_poly_priors))
    s.cov_matrix = np.copy(spectrum.cov_matrix)
    s.profile_params = s.from_poly_params_to_profile_params(s.params.values, apply_bounds=True)
    s.fill_table_with_profile_params(s.profile_params)
    s.from_profile_params_to_shape_params(s.profile_params)
    s.table['Dy'] = s.table['y_c'] - spectrum.spectrogram_y0

    # deconvolve and regularize with 1D priors
    method = "psf1d"
    mode = "2D"
    my_logger.debug(f"\n\tTransverse fit table before PSF_2D fit:"
                    f"\n{s.table[['amplitude', 'x_c', 'y_c', 'Dx', 'Dy', 'Dy_disp_axis']]}")
    my_logger.info(f'\n\tStart ChromaticPSF polynomial fit with '
                   f'mode={mode} and amplitude_priors_method={method}...')
    data = spectrum.spectrogram
    err = spectrum.spectrogram_err
    w = s.fit_chromatic_psf(data, bgd_model_func=bgd_model_func, data_errors=err, live_fit=False,
                            amplitude_priors_method=method, mode=mode, verbose=parameters.VERBOSE, analytical=True)

    # save results
    spectrum.spectrogram_fit = s.evaluate(s.set_pixels(mode=mode), poly_params=s.params.values)
    spectrum.spectrogram_residuals = (data - spectrum.spectrogram_fit - bgd_model_func(np.arange(Nx),
                                                                                       np.arange(Ny))) / err
    lambdas = spectrum.disperser.grating_pixel_to_lambda(s.get_algebraic_distance_along_dispersion_axis(),
                                                         x0=spectrum.x0, order=spectrum.order)
    s.table['lambdas'] = lambdas
    spectrum.lambdas = np.array(lambdas)
    spectrum.data = np.copy(w.amplitude_params)
    spectrum.err = np.copy(w.amplitude_params_err)
    spectrum.cov_matrix = np.copy(w.amplitude_cov_matrix)
    spectrum.pixels = np.copy(s.table['Dx'])
    s.table['Dy'] = s.table['y_c'] - spectrum.spectrogram_y0
    s.table['Dy_fwhm_inf'] = s.table['Dy'] - 0.5 * s.table['fwhm']
    s.table['Dy_fwhm_sup'] = s.table['Dy'] + 0.5 * s.table['fwhm']
    spectrum.chromatic_psf = s
    spectrum.header['PSF_REG'] = s.opt_reg
    spectrum.header['TRACE_R'] = w.trace_r
    spectrum.header['MEANFWHM'] = np.mean(np.array(s.table['fwhm']))

    # Plot FHWM(lambda)
    if parameters.DEBUG:  # pragma: no cover
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex="all")
        ax[0].plot(spectrum.lambdas, np.array(s.table['fwhm']))
        ax[0].set_xlabel(r"$\lambda$ [nm]")
        ax[0].set_ylabel("Transverse FWHM [pixels]")
        ax[0].set_ylim((0.8 * np.min(s.table['fwhm']), 1.2 * np.max(s.table['fwhm'])))  # [-10:])))
        ax[0].grid()
        ax[1].plot(spectrum.lambdas, np.array(s.table['y_c']))
        ax[1].set_xlabel(r"$\lambda$ [nm]")
        ax[1].set_ylabel("Distance from mean dispersion axis [pixels]")
        # ax[1].set_ylim((0.8*np.min(s.table['Dy']), 1.2*np.max(s.table['fwhm'][-10:])))
        ax[1].grid()
        if parameters.DISPLAY:
            plt.show()
        if parameters.LSST_SAVEFIGPATH:
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'fwhm_2.pdf'))

    return w


def plot_comparison_truth(spectrum, w):  # pragma: no cover
    s = spectrum.chromatic_psf
    lambdas_truth = np.fromstring(spectrum.header['LBDAS_T'][1:-1], sep=' ')
    psf_poly_truth = np.fromstring(spectrum.header['PSF_P_T'][1:-1], sep=' ', dtype=float)
    deg_truth = int(spectrum.header["PSF_DEG"])
    psf_poly_truth[-1] = spectrum.spectrogram_saturation
    amplitude_truth = np.fromstring(spectrum.header['AMPLIS_T'][1:-1], sep=' ', dtype=float)
    amplitude_truth *= parameters.FLAM_TO_ADURATE * lambdas_truth * np.gradient(lambdas_truth) * parameters.CCD_REBIN
    s0 = ChromaticPSF(s.psf, lambdas_truth.size, s.Ny, deg=deg_truth,
                      saturation=spectrum.spectrogram_saturation)
    s0.params.values = np.asarray(list(amplitude_truth) + list(psf_poly_truth))
    # s0.deg = (len(s0.poly_params[s0.Nx:]) - 1) // ((len(s0.psf.param_names) - 2) - 1) // 2
    # s0.set_polynomial_degrees(s0.deg)
    s0.profile_params = s0.from_poly_params_to_profile_params(s0.params.values)
    s0.from_profile_params_to_shape_params(s0.profile_params)

    gs_kw = dict(width_ratios=[2, 1], height_ratios=[2, 1])
    fig, ax = plt.subplots(2, 2, figsize=(11, 5), sharex="all", gridspec_kw=gs_kw)
    ax[0, 0].plot(lambdas_truth, amplitude_truth, label="truth")
    amplitude_priors_err = [np.sqrt(w.amplitude_priors_cov_matrix[x, x]) for x in range(w.Nx)]
    ax[0, 0].errorbar(spectrum.lambdas, w.amplitude_priors, yerr=amplitude_priors_err, label="prior")
    ax[0, 0].errorbar(spectrum.lambdas, w.amplitude_params, yerr=w.amplitude_params_err, label="2D")
    ax[0, 0].grid()
    ax[0, 0].legend()
    ax[0, 0].set_xlabel(r"$\lambda$ [nm]")
    ax[0, 0].set_ylabel(f"Amplitude $A$ [ADU/s]")
    ax[1, 0].plot(lambdas_truth, np.zeros_like(lambdas_truth), label="truth")
    amplitude_truth_interp = np.interp(spectrum.lambdas, lambdas_truth, amplitude_truth)
    res = (w.amplitude_priors - amplitude_truth_interp) / amplitude_priors_err
    ax[1, 0].errorbar(spectrum.lambdas, res, yerr=np.ones_like(res),
                      label=f"prior: mean={np.mean(res):.2f}, std={np.std(res):.2f}")
    res = (w.amplitude_params - amplitude_truth_interp) / w.amplitude_params_err
    ax[1, 0].errorbar(spectrum.lambdas, res, yerr=np.ones_like(res),
                      label=f"2D: mean={np.mean(res):.2f}, std={np.std(res):.2f}")
    ax[1, 0].grid()
    ax[1, 0].legend()
    ax[1, 0].set_xlabel(r"$\lambda$ [nm]")
    ax[1, 0].set_ylim(-5, 5)
    ax[1, 0].set_ylabel(r"$(A - A_{\rm truth})/\sigma_A$")

    fwhm_truth = np.interp(spectrum.lambdas, lambdas_truth, s0.table["fwhm"]) / parameters.CCD_REBIN
    # fwhm_prior = np.interp(spectrum.lambdas, np.arange(w.amplitude_priors.size), w.fwhm_priors)
    # fwhm_1d = np.interp(np.arange(len(s.table)), np.arange(w.fwhm_1d.size), w.fwhm_1d)
    # fwhm_2d = np.interp(np.arange(len(s.table)), np.arange(s.Nx), fwhm_1d)
    ax[0, 1].plot(lambdas_truth, s0.table["fwhm"], label="truth")
    ax[0, 1].plot(spectrum.lambdas, w.fwhm_priors, label="prior")
    ax[0, 1].plot(spectrum.lambdas, s.table["fwhm"], label="fit")
    ax[0, 1].grid()
    ax[0, 1].set_ylim(0, 10)
    ax[0, 1].legend()
    ax[0, 1].set_xlabel(r"$\lambda$ [nm]")
    ax[1, 1].set_xlabel(r"$\lambda$ [nm]")
    ax[0, 1].set_ylabel(f"FWHM [pix]")
    ax[1, 1].set_ylabel(r"FWHM - FWHM$_{\rm truth}$ [pix]")
    ax[1, 1].plot(lambdas_truth, np.zeros_like(lambdas_truth), label="truth")
    ax[1, 1].plot(spectrum.lambdas, w.fwhm_priors - fwhm_truth, label="prior")
    ax[1, 1].plot(spectrum.lambdas, s.table["fwhm"] - fwhm_truth, label="2D")
    ax[1, 1].grid()
    ax[1, 1].set_ylim(-0.5, 0.5)
    ax[1, 1].legend()
    plt.subplots_adjust(hspace=0)
    fig.tight_layout()
    if parameters.DISPLAY:
        plt.show()
    if parameters.LSST_SAVEFIGPATH:
        fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'deconvolution_truth.pdf'), transparent=True)
    plt.show()

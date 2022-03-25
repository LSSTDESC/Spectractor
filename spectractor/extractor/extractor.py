import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy import sparse
import time

from spectractor import parameters
from spectractor.config import set_logger, load_config
from spectractor.extractor.images import Image, find_target, turn_image
from spectractor.extractor.spectrum import Spectrum, calibrate_spectrum
from spectractor.extractor.background import extract_spectrogram_background_sextractor
from spectractor.extractor.chromaticpsf import ChromaticPSF
from spectractor.extractor.psf import load_PSF
from spectractor.tools import ensure_dir, plot_image_simple, from_lambda_to_colormap, plot_spectrum_simple
from spectractor.simulation.adr import adr_calib, flip_and_rotate_adr_to_image_xy_coordinates
from spectractor.fit.fitter import run_minimisation, run_minimisation_sigma_clipping, RegFitWorkspace, FitWorkspace

def dumpParameters():
    for item in dir(parameters):
        if not item.startswith("__"):
            print(item, getattr(parameters, item))

class FullForwardModelFitWorkspace(FitWorkspace):

    def __init__(self, spectrum, amplitude_priors_method="noprior", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        """Class to fit a full forward model on data to extract a spectrum, with ADR prediction and order 2 subtraction.

        Parameters
        ----------
        spectrum: Spectrum
        amplitude_priors_method
        nwalkers
        nsteps
        burnin
        nbins
        verbose
        plot
        live_fit
        truth

        Examples
        --------
        >>> spec = Spectrum("./tests/data/sim_20170530_134_spectrum.fits", config="./config/ctio.ini")
        >>> w = FullForwardModelFitWorkspace(spectrum=spec, amplitude_priors_method="spectrum")
        """
        FitWorkspace.__init__(self, spectrum.filename, nwalkers, nsteps, burnin, nbins, verbose, plot,
                              live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        self.spectrum = spectrum

        # check the shapes
        self.Ny, self.Nx = spectrum.spectrogram.shape
        # if self.Ny != self.spectrum.chromatic_psf.Ny:
        #     raise AttributeError(f"Data y shape {self.Ny} different from "
        #                          f"ChromaticPSF input Ny {spectrum.chromatic_psf.Ny}.")
        # if self.Nx != self.spectrum.chromatic_psf.Nx:
        #     raise AttributeError(f"Data x shape {self.Nx} different from "
        #                          f"ChromaticPSF input Nx {spectrum.chromatic_psf.Nx}.")

        # crop data to fit faster
        self.lambdas = self.spectrum.lambdas
        self.lambdas_order2 = self.spectrum.lambdas_order2
        self.bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        self.data = spectrum.spectrogram[self.bgd_width:-self.bgd_width, :]
        self.err = spectrum.spectrogram_err[self.bgd_width:-self.bgd_width, :]
        self.bgd = spectrum.spectrogram_bgd[self.bgd_width:-self.bgd_width, :]
        self.bgd_flat = self.bgd.flatten()
        self.Ny, self.Nx = self.data.shape
        yy, xx = np.mgrid[:self.Ny, :self.Nx]
        self.pixels = np.asarray([xx, yy])

        # prepare parameters to fit
        self.A2 = 1
        self.D = np.copy(self.spectrum.header['D2CCD'])
        self.shift_x = np.copy(self.spectrum.header['PIXSHIFT'])
        self.shift_y = 0.
        self.angle = np.copy(self.spectrum.rotation_angle)
        self.B = 1
        self.rotation = parameters.OBS_CAMERA_ROTATION
        self.psf_poly_params = np.copy(self.spectrum.chromatic_psf.from_table_to_poly_params())
        self.psf_profile_params = np.copy(self.spectrum.chromatic_psf.from_table_to_profile_params())
        length = len(self.spectrum.chromatic_psf.table)
        self.psf_poly_params = self.psf_poly_params[length:]
        self.psf_poly_params_labels = np.copy(self.spectrum.chromatic_psf.poly_params_labels[length:])
        self.psf_poly_params_names = np.copy(self.spectrum.chromatic_psf.poly_params_names[length:])
        self.psf_poly_params_bounds = self.spectrum.chromatic_psf.set_bounds_for_minuit(data=None)
        self.spectrum.chromatic_psf.psf.apply_max_width_to_bounds(max_half_width=self.spectrum.spectrogram_Ny)
        psf_poly_params_bounds = self.spectrum.chromatic_psf.set_bounds()
        self.saturation = self.spectrum.spectrogram_saturation
        self.p = np.array([self.A2, self.D, self.shift_x, self.shift_y, self.angle, self.B, self.rotation])
        self.psf_params_start_index = self.p.size
        self.p = np.concatenate([self.p, self.psf_poly_params])
        self.input_labels = ["A2", r"D_CCD [mm]", r"shift_x [pix]", r"shift_y [pix]",
                             r"angle [deg]", "B", "R"] + list(self.psf_poly_params_labels)
        self.axis_names = ["$A_2$", r"$D_{CCD}$ [mm]", r"$\Delta_{\mathrm{x}}$ [pix]", r"$\Delta_{\mathrm{y}}$ [pix]",
                           r"$\theta$ [deg]", "$B$", "R"] + list(self.psf_poly_params_names)
        bounds_D = (self.D - 5 * parameters.DISTANCE2CCD_ERR, self.D + 5 * parameters.DISTANCE2CCD_ERR)
        self.bounds = np.concatenate([np.array([(0, 2 / parameters.GRATING_ORDER_2OVER1), bounds_D,
                                                (-parameters.PIXSHIFT_PRIOR, parameters.PIXSHIFT_PRIOR),
                                                (-10 * parameters.CCD_PIXEL2ARCSEC, 10 * parameters.PIXSHIFT_PRIOR),
                                                (-90, 90), (0.2, 5), (-360, 360)]), psf_poly_params_bounds])
        self.fixed = [False] * self.p.size
        for k, par in enumerate(self.input_labels):
            if "x_c" in par or "saturation" in par or "y_c" in par:
                self.fixed[k] = True
        # This set of fixed parameters was determined so that the reconstructed spectrum has a ZERO bias
        # with respect to the true spectrum injected in the simulation
        # A2 is free only if spectrogram is a simulation or if the order 2/1 ratio is not known and flat
        self.fixed[0] = not self.spectrum.disperser.flat_ratio_order_2over1
        self.fixed[1] = True  # D2CCD: spectrogram can not tell something on this parameter: rely on calibrate_pectrum
        self.fixed[2] = True  # delta x: if False, extracted spectrum is biased compared with truth
        # self.fixed[3] = True  # delta y
        # self.fixed[4] = True  # angle
        self.fixed[5] = True  # B: not needed in simulations, to check with data
        self.fixed[6] = True  # camera rot
        self.fixed[-1] = True  # saturation
        self.nwalkers = max(2 * self.ndim, nwalkers)

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
        self.sqrtW = np.sqrt(sparse.diags(self.W))
        self.sparse_indices = None
        self.set_mask()

        # design matrix
        self.M = np.zeros((self.Nx, self.data.size))
        self.M_dot_W_dot_M = np.zeros((self.Nx, self.Nx))

        # prepare results
        self.amplitude_params = np.zeros(self.Nx)
        self.amplitude_params_err = np.zeros(self.Nx)
        self.amplitude_cov_matrix = np.zeros((self.Nx, self.Nx))

        # priors on amplitude parameters
        self.amplitude_priors_list = ['noprior', 'positive', 'smooth', 'spectrum', 'fixed']
        self.amplitude_priors_method = amplitude_priors_method
        self.fwhm_priors = np.copy(spectrum.chromatic_psf.table['fwhm'])
        self.reg = spectrum.chromatic_psf.opt_reg
        if 'PSF_REG' in spectrum.header and float(spectrum.header["PSF_REG"]) > 0:
            self.reg = float(spectrum.header['PSF_REG'])
        if self.reg < 0:
            self.reg = parameters.PSF_FIT_REG_PARAM
        self.my_logger.info(f"\n\tFull forward model fitting with regularisation parameter r={self.reg}.")
        self.Q = np.zeros((self.Nx, self.Nx))
        self.Q_dot_A0 = np.zeros(self.Nx)
        if amplitude_priors_method not in self.amplitude_priors_list:
            raise ValueError(f"Unknown prior method for the amplitude fitting: {self.amplitude_priors_method}. "
                             f"Must be either {self.amplitude_priors_list}.")
        self.spectrum.convert_from_flam_to_ADUrate()
        if self.amplitude_priors_method == "spectrum":
            self.amplitude_priors = np.copy(self.spectrum.data)
            self.amplitude_priors_cov_matrix = np.copy(self.spectrum.cov_matrix)
        if self.amplitude_priors_method == "fixed":
            self.amplitude_priors = np.copy(self.spectrum.data)

        # regularisation matrices
        if amplitude_priors_method == "spectrum":
            # U = np.diag([1 / np.sqrt(np.sum(self.err[:, x]**2)) for x in range(self.Nx)])
            self.U = np.diag([1 / np.sqrt(self.amplitude_priors_cov_matrix[x, x]) for x in range(self.Nx)])
            L = np.diag(-2 * np.ones(self.Nx)) + np.diag(np.ones(self.Nx), -1)[:-1, :-1] \
                + np.diag(np.ones(self.Nx), 1)[:-1, :-1]
            L.astype(float)
            L[0, 0] = -1
            L[-1, -1] = -1
            self.L = L
            self.Q = L.T @ np.linalg.inv(self.amplitude_priors_cov_matrix) @ L
            # self.Q = L.T @ U.T @ U @ L
            self.Q_dot_A0 = self.Q @ self.amplitude_priors

    def set_mask(self, psf_poly_params=None):
        if psf_poly_params is None:
            psf_poly_params = self.psf_poly_params
        psf_profile_params = self.spectrum.chromatic_psf.from_poly_params_to_profile_params(psf_poly_params,
                                                                                            apply_bounds=True)
        self.spectrum.chromatic_psf.from_profile_params_to_shape_params(psf_profile_params)
        psf_profile_params[:, 2] -= self.bgd_width
        psf_cube = self.spectrum.chromatic_psf.build_psf_cube(self.pixels, psf_profile_params,
                                                              fwhmx_clip=3 * parameters.PSF_FWHM_CLIP,
                                                              fwhmy_clip=parameters.PSF_FWHM_CLIP, dtype="float32")
        flat_spectrogram = np.sum(psf_cube.reshape(len(psf_profile_params), self.pixels[0].size), axis=0)
        mask = flat_spectrogram / np.max(flat_spectrogram) == 0
        self.data[mask] = 0
        self.W[mask] = 0
        self.sqrtW = np.sqrt(sparse.diags(self.W))
        self.sparse_indices = None
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
        a the set of amplitude parameters :math:`\hat{\mathbf{A}}` given by

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

        >>> spec = Spectrum("./tests/data/sim_20170530_134_spectrum.fits", config="./config/ctio.ini")

        Simulate the data with fixed amplitude priors:

        .. doctest::

            >>> w = FullForwardModelFitWorkspace(spectrum=spec, amplitude_priors_method="fixed", verbose=True)
            >>> y, mod, mod_err = w.simulate(*w.p)
            >>> w.plot_fit()

        .. doctest::
            :hide:

            >>> assert mod is not None

        Simulate the data with a Tikhonov prior on amplitude parameters:

        .. doctest::

            >>> spec = Spectrum("./tests/data/sim_20170530_134_spectrum.fits", config="./config/ctio.ini")
            >>> w = FullForwardModelFitWorkspace(spectrum=spec, amplitude_priors_method="spectrum", verbose=True)
            >>> y, mod, mod_err = w.simulate(*w.p)
            >>> w.plot_fit()

        """
        # linear regression for the amplitude parameters
        # prepare the vectors
        A2, D2CCD, dx0, dy0, angle, B, rot, *poly_params = params
        parameters.OBS_CAMERA_ROTATION = rot
        self.p = np.asarray(params)
        W_dot_data = self.W * (self.data + (1 - B) * self.bgd_flat)
        profile_params = self.spectrum.chromatic_psf.from_poly_params_to_profile_params(poly_params, apply_bounds=True)
        profile_params[:, 0] = 1
        profile_params[:, 1] = np.arange(self.Nx)
        self.spectrum.chromatic_psf.fill_table_with_profile_params(profile_params)

        # Distance in x and y with respect to the true order 0 position at lambda_ref
        Dx = np.arange(self.Nx) - self.spectrum.spectrogram_x0 - dx0  # distance in (x,y) spectrogram frame for column x
        Dy_disp_axis = np.tan(angle * np.pi / 180) * Dx  # disp axis height in spectrogram frame for x
        distance = np.sign(Dx) * np.sqrt(Dx * Dx + Dy_disp_axis * Dy_disp_axis)  # algebraic distance along dispersion axis
        self.spectrum.chromatic_psf.table["Dy_disp_axis"] = Dy_disp_axis
        self.spectrum.chromatic_psf.table["Dx"] = Dx

        # First guess of wavelengths
        self.spectrum.disperser.D = np.copy(D2CCD)
        self.lambdas = self.spectrum.disperser.grating_pixel_to_lambda(distance,
                                                                       self.spectrum.x0 + np.asarray([dx0, dy0]),
                                                                       order=self.spectrum.order)
        self.lambdas_order2 = self.spectrum.disperser.grating_pixel_to_lambda(distance,
                                                                              self.spectrum.x0 + np.asarray([dx0, dy0]),
                                                                              order=self.spectrum.order+np.sign(self.spectrum.order))

        # Evaluate ADR
        adr_ra, adr_dec = adr_calib(self.lambdas, self.spectrum.adr_params, parameters.OBS_LATITUDE,
                                    lambda_ref=self.spectrum.lambda_ref)
        adr_x, adr_y = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=0)
        adr_u, adr_v = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=angle)
        # Compute lambdas at pixel column x
        # self.lambdas = self.spectrum.disperser.grating_pixel_to_lambda(distance - adr_u,
        #                                                                self.spectrum.x0 + np.asarray([dx0, dy0]),
        #                                                                order=1)

        # Evaluate ADR for order 2
        adr_ra, adr_dec = adr_calib(self.lambdas_order2, self.spectrum.adr_params, parameters.OBS_LATITUDE,
                                    lambda_ref=self.spectrum.lambda_ref)
        adr_x_2, adr_y_2 = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=0)
        # adr_u_2, adr_v_2 = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=angle)
        # Compute lambdas at pixel column x for order 2
        # self.lambdas_order2 = self.spectrum.disperser.grating_pixel_to_lambda(distance - adr_u_2,
        #                                                                    self.spectrum.x0 + np.asarray([dx0, dy0]),
        #                                                                    order=2)

        # Fill spectrogram trace as a function of the pixel column x
        profile_params[:, 1] = Dx + self.spectrum.spectrogram_x0 + adr_x + dx0
        profile_params[:, 2] = Dy_disp_axis + (self.spectrum.spectrogram_y0 + adr_y + dy0) - self.bgd_width
        # Dy_disp_axis = np.copy(profile_params[:, 2])
        # profile_params[:, 2] += adr_y + dy0 - self.bgd_width

        # Prepare order 2 profile params indexed by the pixel column x
        profile_params_order2 = np.copy(profile_params)
        profile_params_order2[:, 0] = self.spectrum.disperser.ratio_order_2over1(self.lambdas)
        profile_params_order2[:, 1] = np.arange(self.Nx) + adr_x_2 + dx0
        profile_params_order2[:, 2] = Dy_disp_axis + (self.spectrum.spectrogram_y0 + adr_y_2 + dy0) - self.bgd_width
        # profile_params_order2[:, 2] = Dy_disp_axis + adr_y_2 + dy0 - self.bgd_width

        # For each A(lambda)=A_x, affect an order 2 PSF with correct position and
        # same PSF as for the order 1 but at the same position
        distance_order2 = self.spectrum.disperser.grating_lambda_to_pixel(self.lambdas - adr_u,
                                                                          self.spectrum.x0 + np.asarray([dx0, dy0]),
                                                                          order=self.spectrum.order+np.sign(self.spectrum.order))
        for k in range(1, profile_params.shape[1]):
            # profile_params_order2[:, k] = interp1d(self.lambdas_order2, profile_params_order2[:, k],
            #                                       kind="cubic", fill_value="extrapolate")(self.lambdas)
            profile_params_order2[:, k] = interp1d(distance, profile_params_order2[:, k],
                                                   kind="cubic", fill_value="extrapolate")(distance_order2)

        # if parameters.DEBUG:
        #     plt.imshow(self.data.reshape((self.Ny, self.Nx)), origin="lower")
        #     plt.scatter(profile_params[:, 1], profile_params[:, 2], label="profile",
        #                 cmap=from_lambda_to_colormap(self.lambdas), c=self.lambdas)
        #     plt.scatter(profile_params_order2[:, 1], profile_params_order2[:, 2], label="order 2",
        #                 cmap=from_lambda_to_colormap(self.lambdas), c=self.lambdas)
        #     plt.plot(profile_params[:, 1], profile_params[:, 2], label="profile")
        #     plt.plot(profile_params[:, 1], Dy_disp_axis + self.spectrum.spectrogram_y0 + dy0 - self.bgd_width, 'k-',
        #              label="disp_axis")
        #     plt.plot(self.spectrum.chromatic_psf.table['Dx'] + self.spectrum.spectrogram_x0 + dx0,
        #              self.spectrum.chromatic_psf.table['Dy'] + self.spectrum.spectrogram_y0 + dy0 - self.bgd_width,
        #              label="y_c")
        #     plt.legend()
        #     plt.title(f"D_CCD={D2CCD:.2f}, dx0={dx0:.2g}, dy0={dy0:.2g}")
        #     plt.xlim((0, self.Nx))
        #     plt.ylim((0, self.Ny))
        #     plt.grid()
        #     plt.gca().set_aspect("auto")
        #     plt.show()

        # Matrix filling
        psf_cube = self.spectrum.chromatic_psf.build_psf_cube(self.pixels, profile_params,
                                                              fwhmx_clip=3 * parameters.PSF_FWHM_CLIP,
                                                              fwhmy_clip=parameters.PSF_FWHM_CLIP, dtype="float32")
        if A2 > 0:
            # for x in range(self.Nx):
            # M[:, x] += A2 * self.spectrum.chromatic_psf.psf.evaluate(self.pixels,
            #                                                         p=profile_params_order2[x, :]).flatten()
            # if profile_params_order2[x, 1] > 1.2 * self.Nx:
            #    break
            psf_cube2 = A2 * self.spectrum.chromatic_psf.build_psf_cube(self.pixels, profile_params_order2,
                                                                        fwhmx_clip=3 * parameters.PSF_FWHM_CLIP,
                                                                        fwhmy_clip=parameters.PSF_FWHM_CLIP,
                                                                        dtype="float32")
            psf_cube += psf_cube2
        M = psf_cube.reshape(len(profile_params), self.pixels[0].size).T  # flattening
        if self.sparse_indices is None:
            self.sparse_indices = np.where(M > 0)
        M = sparse.csr_matrix((M[self.sparse_indices].ravel(), self.sparse_indices), shape=M.shape, dtype="float32")
        # Algebra to compute amplitude parameters
        if self.amplitude_priors_method != "fixed":
            M_dot_W = M.T * self.sqrtW
            M_dot_W_dot_M = M_dot_W @ M_dot_W.T
            if self.amplitude_priors_method != "spectrum":
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
                M_dot_W_dot_M_plus_Q = M_dot_W_dot_M + self.reg * self.Q
                # try:  # slower
                #     L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M_plus_Q))
                #     cov_matrix = L.T @ L
                # except np.linalg.LinAlgError:
                cov_matrix = np.linalg.inv(M_dot_W_dot_M_plus_Q)
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
        self.psf_profile_params = np.copy(profile_params)
        self.psf_poly_params = np.copy(poly_params)
        self.amplitude_params = np.copy(amplitude_params)
        self.amplitude_params_err = np.array([np.sqrt(cov_matrix[x, x]) for x in range(self.Nx)])
        self.amplitude_cov_matrix = np.copy(cov_matrix)

        # Compute the model
        self.model = M @ amplitude_params
        self.model_err = np.zeros_like(self.model)

        return self.pixels, self.model, self.model_err

    def plot_spectrogram_comparison_simple(self, ax, title='', extent=None, dispersion=False):
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
        model = (self.model + self.p[5] * self.bgd_flat).reshape((self.Ny, self.Nx))
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

    def plot_fit(self):
        """Plot the fit result.

        Examples
        --------

        >>> spec = Spectrum('tests/data/sim_20170530_134_spectrum.fits', config="config/ctio.ini")
        >>> w = FullForwardModelFitWorkspace(spec, verbose=1, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.p)
        >>> w.plot_fit()

        .. plot::
            :include-source:

            from spectractor.fit.fit_spectrogram import SpectrogramFitWorkspace
            file_name = 'tests/data/reduc_20170530_134_spectrum.fits'
            atmgrid_file_name = file_name.replace('spectrum', 'atmsim')
            fit_workspace = SpectrogramFitWorkspace(file_name, atmgrid_file_name=atmgrid_file_name, verbose=True)
            A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, angle, *psf = fit_workspace.p
            lambdas, model, model_err = fit_workspace.simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x,
                                                                          shift_y, angle, psf)
            fit_workspace.lambdas = lambdas
            fit_workspace.model = model
            fit_workspace.model_err = model_err
            fit_workspace.plot_fit()

        """
        gs_kw = dict(width_ratios=[3, 0.01, 1, 0.01, 1, 0.15], height_ratios=[1, 1, 1, 1])
        fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(10, 8), gridspec_kw=gs_kw)

        A2, D2CCD, dx0, dy0, angle, B, rot, *poly_params = self.p
        plt.suptitle(f'A2={A2:.3f}, D={D2CCD:.2f}mm, shift_x={dx0:.3f}pix, shift_y={dy0:.3f}pix, '
                     f'angle={angle:.2f}pix, B={B:.3f}', y=1)
        # main plot
        self.plot_spectrogram_comparison_simple(ax[:, 0:2], title='Spectrogram model', dispersion=True)
        # zoom O2
        self.plot_spectrogram_comparison_simple(ax[:, 2:4], extent=[730, 800], title='Zoom $O_2$', dispersion=True)
        # zoom H2O
        self.plot_spectrogram_comparison_simple(ax[:, 4:6], extent=[870, 1000], title='Zoom $H_2 O$', dispersion=True)
        for i in range(3):  # clear middle colorbars
            for j in range(2):
                plt.delaxes(ax[i, 2 * j + 1])
        for i in range(4):  # clear middle y axis labels
            for j in range(1, 3):
                ax[i, 2 * j].set_ylabel("")
        fig.tight_layout()
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            if parameters.DISPLAY and self.verbose:
                plt.show()
        if parameters.SAVE:  # pragma: no cover
            figname = os.path.splitext(self.filename)[0] + "_bestfit.pdf"
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')

    def adjust_spectrogram_position_parameters(self):
        # fit the spectrogram trace
        epsilon = 1e-4 * self.p
        epsilon[epsilon == 0] = 1e-4
        fixed = [True] * len(self.p)
        fixed[3:5] = [False, False]  # shift_y and angle
        self.sparse_indices = None
        run_minimisation(self, "newton", epsilon, fixed, xtol=1e-4, ftol=100 / self.data.size)
        self.sparse_indices = None


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

    >>> spec = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits", config="./config/ctio.ini")
    >>> parameters.VERBOSE = True
    >>> w = FullForwardModelFitWorkspace(spec, verbose=1, plot=True, live_fit=True, amplitude_priors_method="spectrum")
    >>> spec = run_ffm_minimisation(w, method="newton")  # doctest: +ELLIPSIS
    >>> if 'LBDAS_T' in spec.header: plot_comparison_truth(spec, w)
       Line   Tabulated  Detected ...

    .. doctest:
        :hide:

        >>> assert w.costs[-1] / w.data.size < 1.22  # reduced chisq
        >>> assert np.isclose(w.p[4], -1.56, rtol=0.05)  # angle
        >>> assert np.isclose(w.p[5], 1, rtol=0.05)  # B

    """
    my_logger = set_logger(__name__)
    w.adjust_spectrogram_position_parameters()

    if method != "newton":
        run_minimisation(w, method=method)
    else:
        costs = np.array([w.chisq(w.p)])
        if parameters.DISPLAY and (parameters.DEBUG or w.live_fit):
            w.plot_fit()
        start = time.time()
        my_logger.info(f"\n\tStart guess: {w.p}\n\twith {w.input_labels}")
        epsilon = 1e-4 * w.p
        epsilon[epsilon == 0] = 1e-4

        run_minimisation(w, method=method, fix=w.fixed, xtol=1e-4, ftol=10 / w.data.size)
        # Optimize the regularisation parameter only if it was not done before
        if w.amplitude_priors_method == "spectrum" and w.reg == parameters.PSF_FIT_REG_PARAM:  # pragma: no cover
            w_reg = RegFitWorkspace(w, opt_reg=parameters.PSF_FIT_REG_PARAM, verbose=True)
            w_reg.run_regularisation()
            w.opt_reg = w_reg.opt_reg
            w.reg = np.copy(w_reg.opt_reg)
            w.simulate(*w.p)
            if np.trace(w.amplitude_cov_matrix) < np.trace(w.amplitude_priors_cov_matrix):
                w.my_logger.warning(
                    f"\n\tTrace of final covariance matrix ({np.trace(w.amplitude_cov_matrix)}) is "
                    f"below the trace of the prior covariance matrix "
                    f"({np.trace(w.amplitude_priors_cov_matrix)}). This is probably due to a very "
                    f"high regularisation parameter in case of a bad fit. Therefore the final "
                    f"covariance matrix is mulitiplied by the ratio of the traces and "
                    f"the amplitude parameters are very close the amplitude priors.")
                r = np.trace(w.amplitude_priors_cov_matrix) / np.trace(w.amplitude_cov_matrix)
                w.amplitude_cov_matrix *= r
                w.amplitude_params_err = np.array(
                    [np.sqrt(w.amplitude_cov_matrix[x, x]) for x in range(w.Nx)])

        for i in range(niter):
            w.set_mask(psf_poly_params=w.p[w.psf_params_start_index:])
            run_minimisation_sigma_clipping(w, "newton", epsilon, w.fixed, xtol=1e-5,
                                            ftol=1 / w.data.size, sigma_clip=20, niter_clip=3)
            my_logger.info(f"\n\tNewton: total computation time: {time.time() - start}s")

            if parameters.DEBUG:
                w.plot_fit()
            w.spectrum.lambdas = np.copy(w.lambdas)
            w.spectrum.data = np.copy(w.amplitude_params)
            w.spectrum.err = np.copy(w.amplitude_params_err)
            w.spectrum.cov_matrix = np.copy(w.amplitude_cov_matrix)
            w.spectrum.chromatic_psf.fill_table_with_profile_params(w.psf_profile_params)
            w.spectrum.chromatic_psf.table["amplitude"] = np.copy(w.amplitude_params)
            w.spectrum.chromatic_psf.from_profile_params_to_shape_params(w.psf_profile_params)
            w.spectrum.chromatic_psf.poly_params = w.spectrum.chromatic_psf.from_table_to_poly_params()
            w.spectrum.spectrogram_fit = w.model
            w.spectrum.spectrogram_residuals = (w.data - w.spectrum.spectrogram_fit) / w.err
            w.spectrum.header['CHI2_FIT'] = w.costs[-1] / (w.data.size - len(w.mask))
            w.spectrum.header['PIXSHIFT'] = w.p[2]
            w.spectrum.header['D2CCD'] = w.p[1]
            w.spectrum.header['A2_FIT'] = w.p[0]
            w.spectrum.header["ROTANGLE"] = w.p[4]

            # Calibrate the spectrum
            calibrate_spectrum(w.spectrum, with_adr=False)
            w.p[1] = w.spectrum.disperser.D
            w.p[2] = w.spectrum.header['PIXSHIFT']
            w.spectrum.convert_from_flam_to_ADUrate()

        if w.filename != "":
            parameters.SAVE = True
            ipar = np.array(np.where(np.array(w.fixed).astype(int) == 0)[0])
            w.plot_correlation_matrix(ipar)
            w.save_parameters_summary(ipar, header=f"{w.spectrum.date_obs}\n"
                                                   f"chi2: {costs[-1] / w.data.size}")
            w.plot_fit()
            parameters.SAVE = False

    # Save results
    w.spectrum.convert_from_ADUrate_to_flam()
    x, model, model_err = w.simulate(*w.p)

    # Propagate uncertainties
    # from spectractor.tools import plot_correlation_matrix_simple, compute_correlation_matrix
    # M = np.copy(w.M)
    # amplitude_params = np.copy(w.amplitude_params)
    # ipar = np.array(np.where(np.array(w.fixed).astype(int) == 0)[0])
    # w.amplitude_priors_method = "fixed"
    # jac = w.jacobian(w.p, epsilon=epsilon, fixed_params=w.fixed, model_input=[x, model, model_err])
    # jac = jac[ipar]
    # start = jac.shape[0]
    # J = np.hstack([jac.T, M])
    # H = (J.T * w.W) @ J
    # H[start:, start:] += w.reg*w.Q
    # full_cov_matrix = np.linalg.inv(H)
    # amplitude_params_err = np.array([np.sqrt(full_cov_matrix[start+x, start+x]) for x in range(w.Nx)])
    # plt.errorbar(w.lambdas, amplitude_params, yerr=amplitude_params_err)
    # plt.grid()
    # plt.show()
    # plt.figure()
    # plot_correlation_matrix_simple(ax=plt.gca(), rho=compute_correlation_matrix(full_cov_matrix))
    # plt.show()
    # w.amplitude_params = amplitude_params
    # w.amplitude_params_err = amplitude_params_err
    # w.amplitude_cov_matrix = full_cov_matrix[start:, start:]

    # Propagate parameters
    A2, D2CCD, dx0, dy0, angle, B, *poly_params = w.p
    w.spectrum.rotation_angle = angle
    w.spectrum.spectrogram_bgd *= B
    w.spectrum.spectrogram_bgd_rms *= B
    w.spectrum.spectrogram_x0 += dx0
    w.spectrum.spectrogram_y0 += dy0
    w.spectrum.x0[0] += dx0
    w.spectrum.x0[1] += dy0
    w.spectrum.header["TARGETX"] = w.spectrum.x0[0]
    w.spectrum.header["TARGETY"] = w.spectrum.x0[1]

    # Compute order 2 contamination
    w.spectrum.lambdas_order2 = w.lambdas
    w.spectrum.data_order2 = A2 * w.amplitude_params * w.spectrum.disperser.ratio_order_2over1(w.lambdas)
    w.spectrum.err_order2 = A2 * w.amplitude_params_err * w.spectrum.disperser.ratio_order_2over1(w.lambdas)

    # Convert to flam
    w.spectrum.convert_from_ADUrate_to_flam()

    # Compare with truth if available
    if 'LBDAS_T' in w.spectrum.header and parameters.DEBUG:
        plot_comparison_truth(w.spectrum, w)

    return w.spectrum

def Spectractor(file_name, output_directory, target_label, guess=None, disperser_label="", config='./config/ctio.ini',
                atmospheric_lines=True, line_detection=True):
    """ Spectractor
    Main function to extract a spectrum from an image

    Parameters
    ----------
    file_name: str
        Input file nam of the image to analyse.
    output_directory: str
        Output directory.
    target_label: str
        The name of the targeted object.
    guess: [int,int], optional
        [x0,y0] list of the guessed pixel positions of the target in the image (must be integers). Mandatory if
        WCS solution is absent (default: None).
    disperser_label: str, optional
        The name of the disperser (default: "").
    config: str
        The config file name (default: "./config/ctio.ini").
    atmospheric_lines: bool, optional
        If True atmospheric lines are used in the calibration fit.

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
        >>> logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
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

    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTOR')
    # Load config file
    if config != "":
        load_config(config)
    if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
        ensure_dir(parameters.LSST_SAVEFIGPATH)

    # Load reduced image
    image = Image(file_name, target_label=target_label, disperser_label=disperser_label)
    if guess is not None and image.target_guess is None:
        image.target_guess = np.asarray(guess)
    if image.target_guess is None:
        from scipy.signal import medfilt2d
        data = medfilt2d(image.data.T, kernel_size=9)
        image.target_guess = np.unravel_index(np.argmax(data), data.shape)
        my_logger.info(f"\n\tNo guess position of order 0 has been given. Assuming the spectrum to extract comes "
                       f"from the brightest object, guess position is set as {image.target_guess}.")
    if parameters.DEBUG:
        image.plot_image(scale='symlog', target_pixcoords=image.target_guess)

    # Use fast mode
    if parameters.CCD_REBIN > 1:
        image.rebin()
        if parameters.DEBUG:
            image.plot_image(scale='symlog', target_pixcoords=image.target_guess)

    # Set output path
    ensure_dir(output_directory)
    output_filename = file_name.split('/')[-1]
    output_filename = output_filename.replace('.fits', '_spectrum.fits')
    output_filename = output_filename.replace('.fz', '_spectrum.fits')
    output_filename = os.path.join(output_directory, output_filename)
    output_filename_spectrogram = output_filename.replace('spectrum', 'spectrogram')
    output_filename_psf = output_filename.replace('spectrum.fits', 'table.csv')
    # Find the exact target position in the raw cut image: several methods
    my_logger.info(f'\n\tSearch for the target in the image with guess={image.target_guess}...')
    find_target(image, image.target_guess, widths=(parameters.XWINDOW, parameters.YWINDOW))
    # Rotate the image
    turn_image(image)
    # Find the exact target position in the rotated image: several methods
    my_logger.info('\n\tSearch for the target in the rotated image...')
    find_target(image, image.target_guess, rotated=True, widths=(parameters.XWINDOW_ROT, parameters.YWINDOW_ROT))
    # Create Spectrum object
    spectrum = Spectrum(image=image, order=parameters.SPEC_ORDER)
    # First 1D spectrum extraction and background extraction
    w_psf1d, bgd_model_func = extract_spectrum_from_image(image, spectrum, signal_width=parameters.PIXWIDTH_SIGNAL,
                                                          ws=(parameters.PIXDIST_BACKGROUND,
                                                              parameters.PIXDIST_BACKGROUND
                                                              + parameters.PIXWIDTH_BACKGROUND),
                                                          right_edge=parameters.CCD_IMSIZE)
    spectrum.atmospheric_lines = atmospheric_lines

    # PSF2D deconvolution
    if parameters.SPECTRACTOR_DECONVOLUTION_PSF2D:
        run_spectrogram_deconvolution_psf2d(spectrum, bgd_model_func=bgd_model_func)

    # Calibrate the spectrum
    my_logger.info(f'\n\tCalibrating order {spectrum.order:d} spectrum...')
    with_adr = True
    if parameters.OBS_OBJECT_TYPE != "STAR":
        with_adr = False
    calibrate_spectrum(spectrum, with_adr=with_adr)
    spectrum.data_order2 = np.zeros_like(spectrum.lambdas_order2)
    spectrum.err_order2 = np.zeros_like(spectrum.lambdas_order2)

    # Full forward model extraction: add transverse ADR and order 2 subtraction
    if parameters.SPECTRACTOR_DECONVOLUTION_FFM:
        w = FullForwardModelFitWorkspace(spectrum, verbose=parameters.VERBOSE, plot=True, live_fit=False,
                                         amplitude_priors_method="spectrum")
        spectrum = run_ffm_minimisation(w, method="newton", niter=2)

    # Save the spectrum
    spectrum.save_spectrum(output_filename, overwrite=True)
    spectrum.save_spectrogram(output_filename_spectrogram, overwrite=True)
    spectrum.lines.print_detected_lines(output_file_name=output_filename.replace('_spectrum.fits', '_lines.csv'),
                                        overwrite=True, amplitude_units=spectrum.units)

    # Plot the spectrum
    if parameters.VERBOSE and parameters.DISPLAY:
        spectrum.plot_spectrum(xlim=None)
    spectrum.chromatic_psf.table['lambdas'] = spectrum.lambdas
    spectrum.chromatic_psf.table.write(output_filename_psf, overwrite=True)

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
    my_logger.info(
        f'\n\tExtracting spectrum from image: spectrum with width 2*{signal_width:d} pixels '
        f'and background from {ws[0]:d} to {ws[1]:d} pixels')

    # Make a data copy
    data = np.copy(image.data_rotated)[:, 0:right_edge]
    err = np.copy(image.stat_errors_rotated)[:, 0:right_edge]

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
    bgd_model_func, bgd_res, bgd_rms = extract_spectrogram_background_sextractor(data, err, ws=ws)
    # while np.nanmean(bgd_res)/np.nanstd(bgd_res) < -0.2 and parameters.PIXWIDTH_BOXSIZE >= 5:
    while (np.abs(np.nanmean(bgd_res)) > 0.5 or np.nanstd(bgd_res) > 1.3) and parameters.PIXWIDTH_BOXSIZE > 5:
        parameters.PIXWIDTH_BOXSIZE = max(5, parameters.PIXWIDTH_BOXSIZE // 2)
        my_logger.debug(f"\n\tPull distribution of background residuals differs too much from mean=0 and std=1. "
                        f"\n\t\tmean={np.nanmean(bgd_res):.3g}; std={np.nanstd(bgd_res):.3g}"
                        f"\n\tThese value should be smaller in absolute value than 0.5 and 1.3. "
                        f"\n\tTo do so, parameters.PIXWIDTH_BOXSIZE is divided "
                        f"by 2 from {parameters.PIXWIDTH_BOXSIZE * 2} -> {parameters.PIXWIDTH_BOXSIZE}.")
        bgd_model_func, bgd_res, bgd_rms = extract_spectrogram_background_sextractor(data, err, ws=ws)

    # Propagate background uncertainties
    err = np.sqrt(err * err + bgd_rms * bgd_rms)

    # Fit the transverse profile
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
    my_logger.info(f'\n\tStart ChromaticPSF polynomial fit with '
                   f'mode={mode} and amplitude_priors_method={method}...')
    w = s.fit_chromatic_psf(data, bgd_model_func=bgd_model_func, data_errors=err,
                            amplitude_priors_method=method, mode=mode, verbose=parameters.VERBOSE)
    spectrum.data = np.copy(w.amplitude_params)
    spectrum.err = np.copy(w.amplitude_params_err)
    spectrum.cov_matrix = np.copy(w.amplitude_cov_matrix)
    spectrum.chromatic_psf = s

    Dx_rot = spectrum.pixels.astype(float) - image.target_pixcoords_rotated[0]
    s.table['Dx'] = np.copy(Dx_rot)
    s.table['Dy'] = s.table['y_c'] - (image.target_pixcoords_rotated[1] - ymin)
    s.table['Dy_disp_axis'] = 0
    s.table['Dy_fwhm_inf'] = s.table['Dy'] - 0.5 * s.table['fwhm']
    s.table['Dy_fwhm_sup'] = s.table['Dy'] + 0.5 * s.table['fwhm']
    my_logger.debug(f"\n\tTransverse fit table before derotation:"
                    f"\n{s.table[['amplitude', 'x_c', 'y_c', 'Dx', 'Dy', 'Dy_disp_axis']]}")

    # Rotate and save the table
    s.rotate_table(-image.rotation_angle)

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
    bgd_model_func, bgd_res, bgd_rms = extract_spectrogram_background_sextractor(data, err, ws=ws)
    bgd = bgd_model_func(np.arange(Nx), np.arange(Ny))
    my_logger.info(f"\n\tBackground statistics: mean={np.nanmean(bgd):.3f} {image.units}, "
                   f"RMS={np.nanmean(bgd_rms):.3f} {image.units}.")

    # Propagate background uncertainties
    err = np.sqrt(err * err + bgd_rms * bgd_rms)

    # First guess for lambdas
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
    spectrum.spectrogram_bgd = bgd
    spectrum.spectrogram_bgd_rms = bgd_rms
    spectrum.spectrogram_x0 = target_pixcoords_spectrogram[0]
    spectrum.spectrogram_y0 = target_pixcoords_spectrogram[1]
    spectrum.spectrogram_xmin = xmin
    spectrum.spectrogram_xmax = xmax
    spectrum.spectrogram_ymin = ymin
    spectrum.spectrogram_ymax = ymax
    spectrum.spectrogram_deg = spectrum.chromatic_psf.deg
    spectrum.spectrogram_saturation = spectrum.chromatic_psf.saturation

    # Plot FHWM(lambda)
    if parameters.DEBUG:
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
    if parameters.DEBUG or parameters.LSST_SAVEFIGPATH:
        gs_kw = dict(width_ratios=[3, 0.08], height_ratios=[1, 1, 1])
        fig, ax = plt.subplots(3, 2, sharex='none', figsize=(12, 6), gridspec_kw=gs_kw)
        xx = np.arange(s.table['Dx'].size)
        plot_image_simple(ax[2, 0], data=data, scale="symlog", title='', units=image.units, aspect='auto', cax=ax[2, 1])
        ax[2, 0].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_disp_axis'], label='Dispersion axis', color="r")
        ax[2, 0].scatter(xx, target_pixcoords_spectrogram[1] + s.table['Dy'],
                         c=s.table['lambdas'], edgecolors='None', cmap=from_lambda_to_colormap(s.table['lambdas']),
                         label='Fitted spectrum centers', marker='o', s=10)
        ax[2, 0].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_fwhm_inf'], 'k-', label='Fitted FWHM')
        ax[2, 0].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_fwhm_sup'], 'k-', label='')
        ax[2, 0].set_ylim(0.5 * Ny - signal_width, 0.5 * Ny + signal_width)
        ax[2, 0].set_xlim(0, xx.size)
        ax[2, 0].legend(loc='best')
        plot_spectrum_simple(ax[0, 0], spectrum.lambdas, spectrum.data, data_err=spectrum.err,
                             units=image.units, label='Fitted spectrum')
        ax[0, 0].plot(spectrum.lambdas, s.table['flux_sum'], 'k-', label='Cross spectrum')
        ax[1, 0].set_xlabel(r"$\lambda$ [nm]")
        ax[0, 0].legend(loc='best')
        ax[1, 0].plot(spectrum.lambdas, (s.table['flux_sum'] - s.table['flux_integral']) / s.table['flux_sum'],
                      label='(model_integral-cross_sum)/cross_sum')
        ax[1, 0].legend()
        ax[1, 0].grid(True)
        ax[1, 0].set_xlim(ax[0, 0].get_xlim())
        ax[1, 0].set_ylim(-1, 1)
        ax[1, 0].set_ylabel('Relative difference')
        fig.tight_layout()
        # fig.subplots_adjust(hspace=0)
        pos0 = ax[0, 0].get_position()
        pos1 = ax[1, 0].get_position()
        pos2 = ax[2, 0].get_position()
        ax[0, 0].set_position([pos2.x0, pos0.y0, pos2.width, pos0.height])
        ax[1, 0].set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])
        ax[0, 1].remove()
        ax[1, 1].remove()
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
    s.poly_params = np.concatenate((s.table["amplitude"], psf_poly_priors))
    s.cov_matrix = np.copy(spectrum.cov_matrix)
    s.profile_params = s.from_poly_params_to_profile_params(s.poly_params, apply_bounds=True)
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
                            amplitude_priors_method=method, mode=mode, verbose=parameters.VERBOSE)

    # save results
    spectrum.spectrogram_fit = s.build_spectrogram_image(s.poly_params, mode=mode)
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

    # Plot FHWM(lambda)
    if parameters.DEBUG:
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
    amplitude_truth *= parameters.FLAM_TO_ADURATE * lambdas_truth * np.gradient(lambdas_truth)
    s0 = ChromaticPSF(s.psf, lambdas_truth.size, s.Ny, deg=deg_truth,
                      saturation=spectrum.spectrogram_saturation)
    s0.poly_params = np.asarray(list(amplitude_truth) + list(psf_poly_truth))
    s0.deg = (len(s0.poly_params[s0.Nx:]) - 1) // (len(s0.psf.param_names) - 2) - 1
    s0.set_polynomial_degrees(s0.deg)
    s0.profile_params = s0.from_poly_params_to_profile_params(s0.poly_params)
    s0.from_profile_params_to_shape_params(s0.profile_params)
    gs_kw = dict(width_ratios=[2, 1], height_ratios=[2, 1])
    fig, ax = plt.subplots(2, 2, figsize=(9, 5), sharex="all", gridspec_kw=gs_kw)
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

    fwhm_truth = np.interp(spectrum.lambdas, lambdas_truth, s0.table["fwhm"])
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
        fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'deconvolution_truth.pdf'))
    plt.show()

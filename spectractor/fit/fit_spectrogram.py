import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.signal import convolve2d
import copy
import getCalspec

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.tools import plot_image_simple, from_lambda_to_colormap
from spectractor.extractor.spectrum import Spectrum
from spectractor.simulation.simulator import SpectrogramModel
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.fit.fitter import (FitWorkspace, FitParameters, run_minimisation, run_minimisation_sigma_clipping,
                                    write_fitparameter_json)
try:
    from gaiaspec import getGaia
except ModuleNotFoundError:
    getGaia = None

plot_counter = 0


class SpectrogramFitWorkspace(FitWorkspace):

    def __init__(self, spectrum, atmgrid_file_name="", fit_angstrom_exponent=False,
                 verbose=False, plot=False, live_fit=False, truth=None):
        """Class to fit a spectrogram extracted with Spectractor.

        First the spectrogram is cropped using the parameters.PIXWIDTH_SIGNAL parameter to increase speedness.
        The truth parameters are loaded from the file header if provided.
        If provided, the atmospheric grid is used for the atmospheric transmission simulations and interpolated
        with splines, otherwise Libradtran is called at each step (slower).

        Parameters
        ----------
        spectrum: Spectrum
            Spectrum object.
        atmgrid_file_name: str, optional
            Atmospheric grid file name (default: "").
        fit_angstrom_exponent: bool, optional
            If True, fit angstrom exponent (default: False).
        verbose: bool, optional
            Verbosity level (default: False).
        plot: bool, optional
            If True, many plots are produced (default: False).
        live_fit: bool, optional
            If True, many plots along the fitting procedure are produced to see convergence in live (default: False).
        truth: array_like, optional
            Array of truth parameters to compare with the best fit result (default: None).

        Examples
        --------
        >>> spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
        >>> parameters.SPECTRACTOR_ATMOSPHERE_SIM = "libradtran"
        >>> atmgrid_filename = spec.filename.replace('spectrum', 'atmsim')
        >>> w = SpectrogramFitWorkspace(spec, atmgrid_file_name=atmgrid_filename, verbose=True, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.params.values)
        >>> w.plot_fit()

        """
        if not getCalspec.is_calspec(spectrum.target.label):
            if getGaia is not None:
                is_gaiaspec = getGaia.is_gaiaspec(spectrum.target.label)
                is_gaia_full = False
                if is_gaiaspec == False:
                    is_gaia_full = getGaia.is_gaia_full(spectrum.target.label)
                if not is_gaiaspec:
                    if not is_gaia_full:
                        raise ValueError(f"{spectrum.target.label=} must be a CALSPEC or GAIA star.")
            else:
                raise ValueError(f"{spectrum.target.label=} must be a CALSPEC star according to getCalspec package.")
        self.spectrum = spectrum
        self.filename = spectrum.filename.replace("spectrum", "spectrogram")
        self.diffraction_orders = np.arange(spectrum.order, spectrum.order + 3 * np.sign(spectrum.order), np.sign(spectrum.order))
        if len(self.diffraction_orders) == 0:
            raise ValueError(f"At least one diffraction order must be given for spectrogram simulation.")
        length = len(self.spectrum.chromatic_psf.table)
        self.psf_poly_params = self.spectrum.chromatic_psf.from_table_to_poly_params()[length:]
        self.spectrum.chromatic_psf.psf.apply_max_width_to_bounds(max_half_width=self.spectrum.spectrogram_Ny)
        self.saturation = self.spectrum.spectrogram_saturation
        D2CCD = np.copy(spectrum.header['D2CCD'])
        p = np.array([1, 1, 0, 0.05, 1.2, 400, 5, 1, 1, D2CCD, self.spectrum.header['PIXSHIFT'],
                      0, self.spectrum.rotation_angle, self.spectrum.pressure])
        # parameter indices for which we don't need to recompute the PSF cube for model evaluation
        # warning: they must be contiguous to preserve psf_cube in jacobian function loop
        self.fixed_psf_params = np.arange(0, 9, dtype=int)
        self.psf_params_start_index = np.array([p.size + len(self.psf_poly_params) * k for k in range(len(self.diffraction_orders))])
        psf_poly_params_labels = np.copy(self.spectrum.chromatic_psf.params.labels[length:])
        psf_poly_params_names = np.copy(self.spectrum.chromatic_psf.params.axis_names[length:])
        psf_poly_params_bounds = self.spectrum.chromatic_psf.set_bounds()
        p = np.concatenate([p] + [self.psf_poly_params] * len(self.diffraction_orders))
        input_labels = [f"A{order}" for order in self.diffraction_orders]
        input_labels += ["VAOD", "angstrom_exp", "ozone [db]", "PWV [mm]", "B", "A_star",
                         r"D_CCD [mm]", r"shift_x [pix]", r"shift_y [pix]", r"angle [deg]", "P [hPa]"]
        for order in self.diffraction_orders:
            input_labels += [label + f"_{order}" for label in psf_poly_params_labels]
        axis_names = [f"$A_{order}$" for order in self.diffraction_orders]
        axis_names += ["VAOD", r'$\"a$', "ozone [db]", "PWV [mm]", "$B$", r"$A_{star}$", r"$D_{CCD}$ [mm]",
                       r"$\Delta_{\mathrm{x}}$ [pix]", r"$\Delta_{\mathrm{y}}$ [pix]", r"$\theta$ [deg]",
                       r"$P_{\mathrm{atm}}$ [hPa]"]
        for order in self.diffraction_orders:
            axis_names += [label+rf"$\!_{order}$" for label in psf_poly_params_names]
        bounds = [[0, 2], [0, 2], [0, 2], [0, 10], [0, 3], [100, 700], [0, 20], [0.8, 1.2], [0, np.inf],
                  [D2CCD - 5 * parameters.DISTANCE2CCD_ERR, D2CCD + 5 * parameters.DISTANCE2CCD_ERR], [-2, 2],
                  [-10, 10], [-90, 90], [0, np.inf]]
        bounds += list(psf_poly_params_bounds) * len(self.diffraction_orders)
        fixed = [False] * p.size
        for k, par in enumerate(input_labels):
            if "x_c" in par or "saturation" in par:
                fixed[k] = True
        for k, par in enumerate(input_labels):
            if "y_c" in par:
                fixed[k] = True
                p[k] = 0
        for k, par in enumerate(input_labels):
            if k >= self.psf_params_start_index[0] and "y_c" not in par and "x_c" not in par and par[-2:] != f"_{spectrum.order}" and "_0_" not in par:
                fixed[k] = True
                p[k] = 0
            if k >= self.psf_params_start_index[0] and "eta" in par and par[-2:] != f"_{spectrum.order}":
                fixed[k] = True
                p[k] = 0
        # for k, par in enumerate(input_labels):
        #     if k >= self.psf_params_start_index[0] and "y_c" not in par and "x_c" not in par and par[-2:] != f"_{spectrum.order}" and "_0_" not in par:
        #         fixed[k] = True
        #         p[k] = 0

        params = FitParameters(p, labels=input_labels, axis_names=axis_names, bounds=bounds, fixed=fixed,
                               truth=truth, filename=self.filename)
        params.fixed[params.get_index(f"A{self.diffraction_orders[0]}")] = False  # A1
        self.atm_params_indices = np.array([params.get_index(label) for label in ["VAOD", "angstrom_exp", "ozone [db]", "PWV [mm]"]])
        # A2 is free only if spectrogram is a simulation or if the order 2/1 ratio is not known and flat
        if "A2" in params.labels:
            params.fixed[params.get_index(f"A{self.diffraction_orders[1]}")] = False  #not getCalspec.is_calspec(spectrum.target.label) #"A2_T" not in self.spectrum.header
        if "A3" in params.labels:
            params.fixed[params.get_index(f"A{self.diffraction_orders[2]}")] = "A3_T" not in self.spectrum.header
        params.fixed[params.get_index(r"shift_x [pix]")] = False  # Delta x
        params.fixed[params.get_index(r"shift_y [pix]")] = False  # Delta y
        params.fixed[params.get_index(r"angle [deg]")] = False  # angle
        params.fixed[params.get_index("B")] = True  # B
        params.fixed[params.get_index("P [hPa]")] = False  # pressure for ADR

        if self.spectrum.spectrogram_Ny > 2 * parameters.PIXDIST_BACKGROUND:
            self.crop_spectrogram()
        FitWorkspace.__init__(self, params, data=self.spectrum.spectrogram_data.flatten(),
                              err=self.spectrum.spectrogram_err.flatten(), epsilon=1e-4,
                              verbose=verbose, plot=plot, live_fit=live_fit, file_name=self.filename)
        self.my_logger = set_logger(self.__class__.__name__)
        if atmgrid_file_name == "":
            self.atmosphere = Atmosphere(self.spectrum.airmass, self.spectrum.pressure, self.spectrum.temperature)
            if self.atmosphere.emulator is not None:
                self.params.bounds[self.params.get_index("ozone [db]")] = (self.atmosphere.emulator.OZMIN, self.atmosphere.emulator.OZMAX)
                self.params.bounds[self.params.get_index("PWV [mm]")] = (self.atmosphere.emulator.PWVMIN, self.atmosphere.emulator.PWVMAX)
        else:
            self.use_grid = True
            self.atmosphere = AtmosphereGrid(spectrum_filename=spectrum.filename, atmgrid_filename=atmgrid_file_name)
            self.params.bounds[self.params.get_index("VAOD")] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
            self.params.bounds[self.params.get_index("ozone [db]")] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.params.bounds[self.params.get_index("PWV [mm]")] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.params.fixed[self.params.get_index("angstrom_exp")] = True  # angstrom exponent
            self.my_logger.info(f'\n\tUse atmospheric grid models from file {atmgrid_file_name}. ')
        self.lambdas = self.spectrum.lambdas
        self.Ny, self.Nx = self.spectrum.spectrogram_data.shape
        self.bgd = self.spectrum.spectrogram_bgd.flatten()
        if self.spectrum.spectrogram_flat is not None:
            self.flat = self.spectrum.spectrogram_flat.flatten()
        else:
            self.flat = None
        if self.spectrum.spectrogram_starfield is not None:
            self.starfield = self.spectrum.spectrogram_starfield.flatten()
        else:
            self.starfield = None
        if self.spectrum.spectrogram_mask is not None:
            self.mask = list(np.where(spectrum.spectrogram_mask.astype(bool).ravel())[0])
        else:
            self.mask = []
        self.fit_angstrom_exponent = fit_angstrom_exponent
        if not fit_angstrom_exponent:
            self.params.fixed[self.params.get_index("angstrom_exp")] = True  # angstrom exponent
        self.params.values[self.params.get_index("angstrom_exp")] = self.atmosphere.angstrom_exponent_default
        self.spectrogram_simulation = SpectrogramModel(self.spectrum, atmosphere=self.atmosphere,
                                           diffraction_orders=self.diffraction_orders,
                                           fast_sim=False, with_adr=True)
        self.lambdas_truth = None
        self.amplitude_truth = None
        self.get_spectrogram_truth()

        # error matrix
        # here image uncertainties are assumed to be uncorrelated
        # (which is not exactly true in rotated images)
        self.W = 1. / (self.err * self.err)
        self.W = self.W.flatten()

        # flat data for fitworkspace
        self.data_before_mask = np.copy(self.data)
        self.W_before_mask = np.copy(self.W)
        self.mask_before_mask = list(np.copy(self.mask))
        # create mask
        self.set_mask()

    def crop_spectrogram(self):
        """Crop the spectrogram in the middle, keeping a vertical width of 2*parameters.PIXWIDTH_SIGNAL around
        the signal region.

        """
        bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        self.spectrum.spectrogram_ymax = self.spectrum.spectrogram_ymax - bgd_width
        self.spectrum.spectrogram_ymin += bgd_width
        self.spectrum.spectrogram_bgd = self.spectrum.spectrogram_bgd[bgd_width:-bgd_width, :]
        self.spectrum.spectrogram_data = self.spectrum.spectrogram_data[bgd_width:-bgd_width, :]
        self.spectrum.spectrogram_err = self.spectrum.spectrogram_err[bgd_width:-bgd_width, :]
        if self.spectrum.spectrogram_flat is not None:
            self.spectrum.spectrogram_flat = self.spectrum.spectrogram_flat[bgd_width:-bgd_width, :]
        if self.spectrum.spectrogram_starfield is not None:
            self.spectrum.spectrogram_starfield = self.spectrum.spectrogram_starfield[bgd_width:-bgd_width, :]
        if self.spectrum.spectrogram_mask is not None:
            self.spectrum.spectrogram_mask = self.spectrum.spectrogram_mask[bgd_width:-bgd_width, :]
        self.spectrum.spectrogram_y0 -= bgd_width
        self.spectrum.chromatic_psf.y0 -= bgd_width
        self.spectrum.spectrogram_Ny, self.spectrum.spectrogram_Nx = self.spectrum.spectrogram_data.shape
        self.spectrum.chromatic_psf.table["y_c"] -= bgd_width

    def set_mask(self, params=None):
        """

        Parameters
        ----------
        params

        Returns
        -------

        Examples
        --------
        >>> spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
        >>> parameters.SPECTRACTOR_ATMOSPHERE_SIM = "libradtran"
        >>> w = SpectrogramFitWorkspace(spec, verbose=True)
        >>> _ = w.simulate(*w.params.values)
        >>> w.plot_fit()

        """
        self.my_logger.info("\n\tReset spectrogram mask with current parameters.")
        if params is None:
            params = self.params.values
        A1, A2, A3, aerosols, angstrom_exponent, ozone, pwv, B, Astar, D, shift_x, shift_y, angle, pressure, *psf_poly_params_all = params
        poly_params = np.array(psf_poly_params_all).reshape((len(self.diffraction_orders), -1))
        self.spectrogram_simulation.psf_cubes_masked = {}
        self.spectrogram_simulation.M_sparse_indices = {}
        self.spectrogram_simulation.psf_cube_sparse_indices = {}
        self.spectrum.adr_params[3] = pressure
        for k, order in enumerate(self.diffraction_orders):
            profile_params = self.spectrum.chromatic_psf.from_poly_params_to_profile_params(poly_params[k],
                                                                                            apply_bounds=True)
            if order == self.diffraction_orders[0]:  # only first diffraction order
                self.spectrum.chromatic_psf.from_profile_params_to_shape_params(profile_params)
            dispersion_law = self.spectrum.compute_dispersion_in_spectrogram(self.lambdas, D, shift_x, shift_y, angle,
                                                                             with_adr=True, order=order)
            profile_params[:, 0] = 1
            profile_params[:, 1] = dispersion_law.real + self.spectrogram_simulation.r0.real
            profile_params[:, 2] += dispersion_law.imag # - self.bgd_width
            psf_cube_masked = self.spectrum.chromatic_psf.build_psf_cube_masked(self.spectrogram_simulation.pixels, profile_params,
                                                                                fwhmx_clip=3 * parameters.PSF_FWHM_CLIP,
                                                                                fwhmy_clip=parameters.PSF_FWHM_CLIP)
            psf_cube_masked = self.spectrum.chromatic_psf.convolve_psf_cube_masked(psf_cube_masked)
            # make rectangular mask per wavelength
            self.spectrogram_simulation.boundaries[order], self.spectrogram_simulation.psf_cubes_masked[order] = self.spectrum.chromatic_psf.set_rectangular_boundaries(psf_cube_masked)
            if k > 0:
                # spectrogram model must be accurate inside the k=0 order footprint: enlarge the next order footprints
                self.spectrogram_simulation.boundaries[order]["ymin"] = np.zeros_like(self.spectrogram_simulation.boundaries[order]["ymin"])
                self.spectrogram_simulation.boundaries[order]["ymax"] = self.Ny * np.ones_like(self.spectrogram_simulation.boundaries[order]["ymax"])
            self.spectrogram_simulation.psf_cube_sparse_indices[order], self.spectrogram_simulation.M_sparse_indices[order] = self.spectrum.chromatic_psf.get_sparse_indices(self.spectrogram_simulation.boundaries[order])
        # mask = np.sum(self.spectrogram_simulation.psf_cubes_masked[self.diffraction_orders[0]].reshape(psf_cube_masked.shape[0], self.spectrogram_simulation.pixels[0].size), axis=0) == 0
        # cumulate the boolean values as int
        weight_mask = np.sum(self.spectrogram_simulation.psf_cubes_masked[self.diffraction_orders[0]], axis=0)
        # look for indices with maximum weight per column (all sheets of the psf cube have contributed)
        res = np.max(weight_mask, axis=0)[np.newaxis,:] * np.ones((weight_mask.shape[0],1))
        # keep only the pixels where all psf_cube sheets have contributed per column
        mask = (weight_mask != res).ravel()
        self.mask = list(self.mask_before_mask) + list(np.where(mask)[0])
        self.mask = list(set(self.mask))
        self.W = np.copy(self.W_before_mask)
        self.W[self.mask] = 0

    def get_spectrogram_truth(self):
        """Load the truth parameters (if provided) from the file header.

        """
        if 'A1_T' in list(self.spectrum.header.keys()):
            A1_truth = self.spectrum.header['A1_T']
            A2_truth = self.spectrum.header['A2_T']
            if 'A3_T' in self.spectrum.header:
                A3_truth = self.spectrum.header['A3_T']
            else:
                A3_truth = 0
            ozone_truth = self.spectrum.header['OZONE_T']
            pwv_truth = self.spectrum.header['PWV_T']
            aerosols_truth = self.spectrum.header['VAOD_T']
            D_truth = self.spectrum.header['D2CCD_T']
            shiftx_truth = 0
            shifty_truth = 0
            rotation_angle = self.spectrum.header['ROT_T']
            B = 1
            Astar = 1
            pressure = self.spectrum.header["OUTPRESS"]
            poly_truth = np.fromstring(self.spectrum.header['PSF_P_T'][1:-1], sep=',', dtype=float)
            self.truth = (A1_truth, A2_truth, A3_truth, aerosols_truth, ozone_truth, pwv_truth,
                          D_truth, shiftx_truth, shifty_truth, rotation_angle, B, Astar, pressure, *poly_truth)
            self.lambdas_truth = np.fromstring(self.spectrum.header['LBDAS_T'][1:-1], sep=',', dtype=float)
            self.amplitude_truth = np.fromstring(self.spectrum.header['AMPLIS_T'][1:-1], sep=',', dtype=float)
        else:
            self.truth = None

    def plot_spectrogram_comparison_simple(self, ax, title='', extent=None, dispersion=False):
        """Method to plot a spectrogram issued from data and compare it with simulations.

        Parameters
        ----------
        ax: Axes
            Axes instance of shape (3, 2).
        title: str, optional
            Title for the simulation plot (default: '').
        extent: array_like, optional
            Extent argument for imshow to crop plots (default: None).
        dispersion: bool, optional
            If True, plot a colored bar to see the associated wavelength color along the x axis (default: False).
        """
        cmap_bwr = copy.copy(mpl.colormaps["bwr"])
        cmap_bwr.set_bad(color='lightgrey')
        cmap_viridis = copy.copy(mpl.colormaps["viridis"])
        cmap_viridis.set_bad(color='lightgrey')

        data = np.copy(self.data_before_mask)
        if len(self.outliers) > 0 or len(self.mask) > 0:
            bad_indices = np.array(list(self.get_bad_indices()) + list(self.mask)).astype(int)
            data[bad_indices] = np.nan

        lambdas = self.spectrum.lambdas
        sub = np.where((lambdas > parameters.LAMBDA_MIN) & (lambdas < parameters.LAMBDA_MAX))[0]
        sub = np.where(sub < self.spectrum.spectrogram_Nx)[0]
        data = data.reshape((self.Ny, self.Nx))
        model = self.model.reshape((self.Ny, self.Nx))
        err = self.err.reshape((self.Ny, self.Nx))
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
            norm = np.sqrt(err**2 + self.model_err.reshape((self.Ny, self.Nx))**2)
            residuals /= norm
            std = float(np.nanstd(residuals[:, sub]))
            plot_image_simple(ax[2, 0], data=residuals[:, sub], vmin=-5 * std, vmax=5 * std, title='(Data-Model)/Err',
                              aspect='auto', cax=ax[2, 1], units='', cmap=cmap_bwr)
            ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, loc='center', color='black', y=0.8)
            ax[2, 0].text(0.05, 0.05, f'mean={np.nanmean(residuals[:, sub]):.3f}\nstd={np.nanstd(residuals[:, sub]):.3f}',
                          horizontalalignment='left', verticalalignment='bottom',
                          color='black', transform=ax[2, 0].transAxes)
            ax[0, 0].set_xticks(ax[2, 0].get_xticks()[1:-1])
            ax[0, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[1, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[2, 1].get_yaxis().set_label_coords(3.5, 0.5)

    def simulate(self, *params):
        """Interface method to simulate a spectrogram.

        Parameters
        ----------
        params: array_like
            Simulation parameter array.

        Returns
        -------
        lambdas: array_like
            Array of wavelengths (1D).
        model: array_like
            Flat 1D array of the spectrogram simulation.
        model_err: array_like
            Flat 1D array of the spectrogram simulation uncertainty.

        Examples
        --------

        >>> spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
        >>> parameters.SPECTRACTOR_ATMOSPHERE_SIM = "libradtran"
        >>> w = SpectrogramFitWorkspace(spec, verbose=True)
        >>> lambdas, model, model_err = w.simulate(*w.params.values)
        >>> w.plot_fit()

        """
        A1, A2, A3, aerosols, angstrom_exponent, ozone, pwv, B, Astar, D, shift_x, shift_y, angle, pressure, *psf_poly_params = params
        self.params.values = np.asarray(params)
        self.spectrogram_simulation.adr_params[3] = pressure
        if not self.fit_angstrom_exponent:
            angstrom_exponent = None
        lambdas, model, model_err = self.spectrogram_simulation.simulate(A1, A2, A3, aerosols, angstrom_exponent, ozone, pwv, D, shift_x, shift_y, angle, psf_poly_params)
        self.lambdas = lambdas
        self.model = model.flatten()
        self.model_err = model_err.flatten()
        self.model += B * self.bgd
        if self.starfield is not None:
            self.model += Astar * self.starfield
        if self.flat is not None:
            # TODO: if flat array is a cube flat, needs to multiply directly in build_psf_cube
            self.model *= self.flat
        return self.lambdas, self.model, self.model_err

    def jacobian(self, params, model_input=None):
        start = time.time()
        if model_input is not None:
            lambdas, model, model_err = model_input
        else:
            lambdas, model, model_err = self.simulate(*params)
        model = model.flatten()
        J = np.zeros((params.size, model.size))
        strategy = copy.copy(self.spectrogram_simulation.fix_psf_cube)
        atmosphere = copy.copy(self.spectrogram_simulation.atmosphere_sim)
        for ip, p in enumerate(params):
            if self.params.fixed[ip]:
                continue
            if ip in self.fixed_psf_params:
                self.spectrogram_simulation.fix_psf_cube = True
            else:
                self.spectrogram_simulation.fix_psf_cube = False
            if ip in self.atm_params_indices:
                self.spectrogram_simulation.fix_atm_sim = False
            else:
                self.spectrogram_simulation.fix_atm_sim = True
            if ip >= self.psf_params_start_index[0]:
                continue
            tmp_p = np.copy(params)
            if tmp_p[ip] + self.epsilon[ip] < self.params.bounds[ip][0] or tmp_p[ip] + self.epsilon[ip] > self.params.bounds[ip][1]:
                self.epsilon[ip] = - self.epsilon[ip]
            tmp_p[ip] += self.epsilon[ip]
            tmp_lambdas, tmp_model, tmp_model_err = self.simulate(*tmp_p)
            if self.spectrogram_simulation.fix_atm_sim is False:
                self.spectrogram_simulation.atmosphere_sim = atmosphere
            J[ip] = (tmp_model.flatten() - model) / self.epsilon[ip]
        self.spectrogram_simulation.fix_atm_sim = True
        self.spectrogram_simulation.fix_psf_cube = False
        for k, order in enumerate(self.diffraction_orders):
            if self.spectrogram_simulation.profile_params[order] is None:
                continue
            start = self.psf_params_start_index[k]
            profile_params = np.copy(self.spectrogram_simulation.profile_params[order])
            J[start:start+len(self.psf_poly_params)] = self.spectrogram_simulation.chromatic_psf.build_psf_jacobian(self.spectrogram_simulation.pixels, profile_params=profile_params,
                                                                                                                    psf_cube_sparse_indices=self.spectrogram_simulation.psf_cube_sparse_indices[order],
                                                                                                                    boundaries=self.spectrogram_simulation.boundaries[order], dtype="float32")
        self.spectrogram_simulation.fix_psf_cube = strategy
        self.spectrogram_simulation.fix_atm_sim = False
        self.my_logger.debug(f"\n\tJacobian time computation = {time.time() - start:.1f}s")
        return J

    def plot_fit(self):
        """Plot the fit result.

        Examples
        --------

        >>> spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
        >>> parameters.SPECTRACTOR_ATMOSPHERE_SIM = "libradtran"
        >>> w = SpectrogramFitWorkspace(spec, verbose=True)
        >>> lambdas, model, model_err = w.simulate(*w.params.values)
        >>> w.plot_fit()

        .. plot::
            :include-source:

            from spectractor.fit.fit_spectrogram import SpectrogramFitWorkspace
            file_name = 'tests/data/reduc_20170530_134_spectrum.fits'
            atmgrid_file_name = file_name.replace('spectrum', 'atmsim')
            fit_workspace = SpectrogramFitWorkspace(file_name, atmgrid_file_name=atmgrid_file_name, verbose=True)
            A1, A2, aerosols, ozone, pwv, D, shift_x, shift_y, angle, *psf = fit_workspace.p
            lambdas, model, model_err = fit_workspace.simulation.simulate(A1, A2, aerosols, ozone, pwv, D, shift_x,
                                                                          shift_y, angle, psf)
            fit_workspace.lambdas = lambdas
            fit_workspace.model = model
            fit_workspace.model_err = model_err
            fit_workspace.plot_fit()

        """
        gs_kw = dict(width_ratios=[3, 0.01, 1, 0.01, 1, 0.15], height_ratios=[1, 1, 1])
        fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(10, 8), gridspec_kw=gs_kw)

        # A1, A2, aerosols, ozone, pwv, D, shift_x, shift_y, shift_t, B,  *psf = self.p
        # plt.suptitle(f'A1={A1:.3f}, A2={A2:.3f}, PWV={pwv:.3f}, OZ={ozone:.3g}, VAOD={aerosols:.3f}, '
        #              f'D={D:.2f}mm, shift_y={shift_y:.2f}pix, B={B:.3f}', y=1)
        # main plot
        self.plot_spectrogram_comparison_simple(ax[:, 0:2], title='Spectrogram model', dispersion=True)
        # zoom O2
        if np.max(self.spectrum.lambdas) > 800 and np.min(self.spectrum.lambdas) < 730:
            self.plot_spectrogram_comparison_simple(ax[:, 2:4], extent=[730, 800], title='Zoom $O_2$', dispersion=True)
        # zoom H2O
        if np.max(self.spectrum.lambdas) > 1000 and np.min(self.spectrum.lambdas) < 870:
            self.plot_spectrogram_comparison_simple(ax[:, 4:6], extent=[870, 1000], title='Zoom $H_2 O$', dispersion=True)
        for i in range(3):  # clear middle colorbars
            for j in range(2):
                plt.delaxes(ax[i, 2*j+1])
        for i in range(3):  # clear middle y axis labels
            for j in range(1, 3):
                ax[i, 2*j].set_ylabel("")
        fig.tight_layout()
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            if parameters.DISPLAY and self.verbose:
                plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()
        if parameters.SAVE:
            figname = os.path.splitext(self.filename)[0] + "_bestfit.pdf"
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight', transparent=True)


def lnprob_spectrogram(p):  # pragma: no cover
    """Logarithmic likelihood function to maximize in MCMC exploration.

    Parameters
    ----------
    p: array_like
        Array of SpectrogramFitWorkspace parameters.

    Returns
    -------
    lp: float
        Log of the likelihood function.

    """
    global fit_workspace
    lp = fit_workspace.lnprior(p)
    if not np.isfinite(lp):
        return -1e20
    return lp + fit_workspace.lnlike_spectrogram(p)


def run_spectrogram_minimisation(fit_workspace, method="newton", verbose=False):
    """Interface function to fit spectrogram simulation parameters to data.

    Parameters
    ----------
    fit_workspace: SpectrogramFitWorkspace
        An instance of the SpectrogramFitWorkspace class.
    method: str, optional
        Fitting method (default: 'newton').

    Examples
    --------
    >>> spec = Spectrum('tests/data/reduc_20170530_134_spectrum.fits')
    >>> parameters.SPECTRACTOR_ATMOSPHERE_SIM = "libradtran"
    >>> w = SpectrogramFitWorkspace(spec, verbose=True, atmgrid_file_name='tests/data/reduc_20170530_134_atmsim.fits')
    >>> parameters.VERBOSE = True
    >>> run_spectrogram_minimisation(w, method="newton")

    """
    my_logger = set_logger(__name__)
    guess = np.asarray(fit_workspace.params.values)
    fit_workspace.simulate(*guess)
    fit_workspace.plot_fit()
    if method != "newton":
        run_minimisation(fit_workspace, method=method)
    else:
        # costs = np.array([fit_workspace.chisq(guess)])
        # if parameters.DISPLAY and (parameters.DEBUG or fit_workspace.live_fit):
        #     fit_workspace.plot_fit()
        # params_table = np.array([guess])
        start = time.time()
        my_logger.info(f"\n\tStart guess: {guess}\n\twith {fit_workspace.params.labels}")
        fixed_default = np.copy(fit_workspace.params.fixed)

        # fit_workspace.simulation.fast_sim = True
        # fit_workspace.simulation.fix_psf_cube = False
        # fit_workspace.fixed = np.copy(fixed)
        # fit_workspace.fixed[:fit_workspace.psf_params_start_index] = True
        # params_table, costs = run_gradient_descent(fit_workspace, guess, epsilon, params_table, costs,
        #                                            fix=fit_workspace.fixed, xtol=1e-3, ftol=1e-2, niter=10)

        # fit_workspace.simulation.fast_sim = True
        # fit_workspace.simulation.fix_psf_cube = False
        # fit_workspace.fixed = np.copy(fixed)
        # for ip, label in enumerate(fit_workspace.input_labels):
        #     if "y_c_0" in label:
        #         fit_workspace.fixed[ip] = False
        #     else:
        #         fit_workspace.fixed[ip] = True
        # run_minimisation(fit_workspace, method="newton", epsilon=epsilon, fix=fit_workspace.fixed,
        #                  xtol=1e-2, ftol=10 / fit_workspace.data.size, verbose=False)

        fit_workspace.spectrogram_simulation.fast_sim = False
        fit_workspace.spectrogram_simulation.fix_psf_cube = False
        fit_workspace.params.fixed = [True] * len(fit_workspace.params.values)
        # fit_workspace.params.fixed[fit_workspace.params.get_index(r"A1")] = False  # A1
        fit_workspace.params.fixed[fit_workspace.params.get_index(r"shift_y [pix]")] = False  # shift y
        fit_workspace.params.fixed[fit_workspace.params.get_index(r"angle [deg]")] = False  # angle
        run_minimisation(fit_workspace, "newton", xtol=1e-2, ftol=0.01, with_line_search=False)
        fit_workspace.params.fixed = fixed_default

        fit_workspace.spectrogram_simulation.fast_sim = False
        fit_workspace.spectrogram_simulation.fix_psf_cube = False
        fit_workspace.params.fixed = np.copy(fixed_default)
        # guess = fit_workspace.p
        # params_table, costs = run_gradient_descent(fit_workspace, guess, epsilon, params_table, costs,
        #                                            fix=fit_workspace.fixed, xtol=1e-6, ftol=1 / fit_workspace.data.size,
        #                                            niter=40)
        run_minimisation_sigma_clipping(fit_workspace, method="newton", xtol=1e-6,
                                        ftol=1 / fit_workspace.data.size, sigma_clip=100, niter_clip=3, verbose=verbose,
                                        with_line_search=True)
        my_logger.info(f"\n\tNewton: total computation time: {time.time() - start}s")
        if fit_workspace.filename != "":
            fit_workspace.params.plot_correlation_matrix()
            write_fitparameter_json(fit_workspace.params.json_filename, fit_workspace.params,
                                    extra={"chi2": fit_workspace.costs[-1] / fit_workspace.data.size,
                                           "date-obs": fit_workspace.spectrum.date_obs})
            # save_gradient_descent(fit_workspace, costs, params_table)
            fit_workspace.plot_fit()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

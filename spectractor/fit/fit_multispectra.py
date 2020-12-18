import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import MaxNLocator

import copy
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.linalg import pinvh, solve_triangular

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.simulation.simulator import SimulatorInit
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.fit.fitter import FitWorkspace, run_minimisation_sigma_clipping
from spectractor.tools import plot_image_simple, plot_spectrum_simple
from spectractor.extractor.spectrum import Spectrum


class MultiSpectraFitWorkspace(FitWorkspace):

    def __init__(self, output_file_name, file_names, fixed_A1s=True, bin_width=10,
                 nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        """Class to fit jointly multiple spectra extracted with Spectractor.

        The spectrum is supposed to be the product of the star SED, a common instrumental throughput,
        a grey term (clouds) and a common atmospheric transmission, with the second order diffraction removed.
        The truth parameters are loaded from the file header if provided.
        If provided, the atmospheric grid files are used for the atmospheric transmission simulations and interpolated
        with splines, otherwise Libradtran is called at each step (slower). The files should have the same name as
        the spectrum files but with the atmsim suffix.

        Parameters
        ----------
        output_file_name: str
            Generic file name to output results.
        file_names: list
            List of spectrum file names.
        bin_width: float
            Size of the wavelength bins in nm.
        nwalkers: int, optional
            Number of walkers for MCMC fitting.
        nsteps: int, optional
            Number of steps for MCMC fitting.
        burnin: int, optional
            Number of burn-in steps for MCMC fitting.
        nbins: int, optional
            Number of bins for MCMC chains analysis.
        verbose: int, optional
            Verbosity level (default: 0).
        plot: bool, optional
            If True, many plots are produced (default: False).
        live_fit: bool, optional
            If True, many plots along the fitting procedure are produced to see convergence in live (default: False).
        truth: array_like, optional
            Array of truth parameters to compare with the best fit result (default: None).

        Examples
        --------

        >>> filename = 'tests/data/reduc_20170530_134_spectrum.fits'
        >>> atmgrid_filename = filename.replace('spectrum', 'atmsim')
        >>> load_config("config/ctio.ini")
        >>> w = SpectrumFitWorkspace(filename, atmgrid_file_name=atmgrid_filename, nsteps=1000,
        ... burnin=2, nbins=10, verbose=1, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.p)
        >>> w.plot_fit()

        """
        FitWorkspace.__init__(self, output_file_name, nwalkers, nsteps, burnin, nbins, verbose, plot, live_fit)
        for name in file_names:
            if "spectrum" not in name:
                raise ValueError(f"ALl file names must contain spectrum keyword and be an output from Spectractor. "
                                 f"I found {name} in file_names list.")
        self.my_logger = set_logger(self.__class__.__name__)
        self.output_file_name = output_file_name
        self.bin_widths = bin_width
        self.spectrum, self.telescope, self.disperser, self.target = SimulatorInit(file_names[0], fast_load=True)
        self.spectra = []
        self.atmospheres = []
        for name in file_names:
            try:
                spectrum = Spectrum(name, fast_load=True)
                self.spectra.append(spectrum)
                atmgrid_file_name = name.replace("sim", "reduc").replace("spectrum.fits", "atmsim.fits")
                if os.path.isfile(atmgrid_file_name):
                    self.atmospheres.append(AtmosphereGrid(name, atmgrid_file_name))
                else:
                    print(f"no grid {atmgrid_file_name}")
                    self.atmospheres.append(Atmosphere(spectrum.airmass, spectrum.pressure, spectrum.temperature))
            except:
                print(f"fail to open {name}")
        self.nspectra = len(self.spectra)
        self.spectrum_lambdas = [self.spectra[k].lambdas for k in range(self.nspectra)]
        self.spectrum_data = [self.spectra[k].data for k in range(self.nspectra)]
        self.spectrum_err = [self.spectra[k].err for k in range(self.nspectra)]
        self.spectrum_data_cov = [self.spectra[k].cov_matrix for k in range(self.nspectra)]
        self.lambdas = np.empty(1)
        self.lambdas_bin_edges = None
        self.data_invcov = None
        self.data_cube = []
        self.err_cube = []
        self.data_cov_cube = []
        self.data_invcov_cube = []
        self.ref_spectrum_cube = []
        self.prepare_data()
        self.ozone = 400.
        self.pwv = 3
        self.aerosols = 0.05
        self.A1s = np.ones(self.nspectra)
        self.p = np.array([self.ozone, self.pwv, self.aerosols, *self.A1s])
        self.fixed = [False] * self.p.size
        self.fixed[3] = True
        if fixed_A1s:
            for ip in range(3, len(self.fixed)):
                self.fixed[ip] = True
        self.input_labels = ["ozone", "PWV", "VAOD"] + [f"A1_{k}" for k in range(self.nspectra)]
        self.axis_names = ["ozone", "PWV", "VAOD"] + ["$A_1^{(" + str(k) + ")}$" for k in range(self.nspectra)]
        self.bounds = [(100, 700), (0, 10), (0, 0.01)] + [(1e-3, 1.2)] * self.nspectra
        for atmosphere in self.atmospheres:
            if isinstance(atmosphere, AtmosphereGrid):
                self.bounds[0] = (min(self.atmospheres[0].OZ_Points), max(self.atmospheres[0].OZ_Points))
                self.bounds[1] = (min(self.atmospheres[0].PWV_Points), max(self.atmospheres[0].PWV_Points))
                self.bounds[2] = (min(self.atmospheres[0].AER_Points), max(self.atmospheres[0].AER_Points))
                break
        self.nwalkers = max(2 * self.ndim, nwalkers)
        # self.simulation = SpectrumSimulation(self.spectrum, self.atmosphere, self.telescope, self.disperser)
        self.amplitude_truth = None
        self.lambdas_truth = None
        self.atmosphere = Atmosphere(airmass=1,
                                     pressure=float(np.mean([self.spectra[k].header["OUTPRESS"]
                                                             for k in range(self.nspectra)])),
                                     temperature=float(np.mean([self.spectra[k].header["OUTTEMP"]
                                                                for k in range(self.nspectra)])))
        self.true_instrumental_transmission = None
        self.true_atmospheric_transmission = None
        self.get_truth()
        # XXXX TEST XXXX
        # self.simulate(300, 5, 0.03)
        # self.plot_fit()

        # design matrix
        self.M = np.zeros((self.nspectra, self.lambdas.size, self.lambdas.size))
        self.M_dot_W_dot_M = np.zeros((self.lambdas.size, self.lambdas.size))

        # prepare results
        self.amplitude_params = np.ones(self.lambdas.size)
        self.amplitude_params_err = np.zeros(self.lambdas.size)
        self.amplitude_cov_matrix = np.zeros((self.lambdas.size, self.lambdas.size))

    def prepare_data(self):
        # rebin wavelengths
        self.lambdas = []
        if self.bin_widths > 0:
            lambdas_bin_edges = np.arange(int(np.min(np.concatenate(list(self.spectrum_lambdas)))),
                                          int(np.max(np.concatenate(list(self.spectrum_lambdas)))) + 1,
                                          self.bin_widths)
            self.lambdas_bin_edges = lambdas_bin_edges
            for i in range(1, lambdas_bin_edges.size):
                self.lambdas.append(0.5 * (lambdas_bin_edges[i] + lambdas_bin_edges[i - 1]))
            self.lambdas = np.asarray(self.lambdas)
        else:
            self.lambdas = np.copy(self.spectrum_lambdas)
        # mask
        lambdas_to_mask = np.asarray(
            [350, 355, 360, 365, 370, 375, 400, 405, 410, 415, 430, 435, 440, 510, 515, 520, 525, 530,
             560, 565, 570, 575, 650, 655, 660, 680, 685, 690, 695, 755, 760, 765, 770,
             980, 985, 990, 995, 1000, 1080, 1085, 1090, 1095, 1100, 1105, 1110, 1115])
        lambdas_to_mask = np.asarray([350, 355, 360, 365, 370, 375, 380])
        lambdas_to_mask_indices = np.asarray(
            [np.argmin(np.abs(self.lambdas - lambdas_to_mask[i])) for i in range(lambdas_to_mask.size)])
        # rebin atmosphere
        if isinstance(self.atmospheres[0], AtmosphereGrid):
            self.atmosphere_lambda_bins = []
            for i in range(0, lambdas_bin_edges.size):
                self.atmosphere_lambda_bins.append([])
                for j in range(0, self.atmospheres[0].lambdas.size):
                    if self.atmospheres[0].lambdas[j] >= lambdas_bin_edges[i]:
                        self.atmosphere_lambda_bins[-1].append(j)
                    if i < lambdas_bin_edges.size - 1 and self.atmospheres[0].lambdas[j] >= lambdas_bin_edges[i + 1]:
                        self.atmosphere_lambda_bins[-1] = np.array(self.atmosphere_lambda_bins[-1])
                        break
            self.atmosphere_lambda_bins = np.array(self.atmosphere_lambda_bins)
            self.atmosphere_lambda_step = np.gradient(self.atmospheres[0].lambdas)[0]
        # rebin data
        self.data_cube = []
        for k in range(self.nspectra):
            data_func = interp1d(self.spectra[k].lambdas, self.spectra[k].data,
                                 kind="cubic", fill_value="extrapolate", bounds_error=None)
            lambdas_truth = np.fromstring(self.spectra[k].header['LBDAS_T'][1:-1], sep=' ')
            amplitude_truth = np.fromstring(self.spectra[k].header['AMPLIS_T'][1:-1], sep=' ', dtype=float)
            data_func = interp1d(lambdas_truth, amplitude_truth,
                                 kind="cubic", fill_value="extrapolate", bounds_error=None)
            data = []
            for i in range(1, lambdas_bin_edges.size):
                data.append(quad(data_func, lambdas_bin_edges[i - 1], lambdas_bin_edges[i])[0] / self.bin_widths)
            self.data_cube.append(np.copy(data))
            # if parameters.DEBUG:
            #     if "LBDAS_T" in self.spectra[k].header:
            #         lambdas_truth = np.fromstring(self.spectra[k].header['LBDAS_T'][1:-1], sep=' ')
            #         amplitude_truth = np.fromstring(self.spectra[k].header['AMPLIS_T'][1:-1], sep=' ', dtype=float)
            #         plt.plot(lambdas_truth, amplitude_truth, label="truth")  # -amplitude_truth)
            #     plt.plot(self.lambdas, self.data_cube[-1], label="binned data")  # -amplitude_truth)
            #     plt.plot(self.spectra[k].lambdas, self.spectra[k].data, label="raw data")  # -amplitude_truth)
            #     # plt.title(self.spectra[k].filename)
            #     # plt.xlim(480,700)
            #     plt.grid()
            #     plt.legend()
            #     plt.show()
        self.data_cube = np.asarray(self.data_cube)
        self.data = np.hstack(self.data_cube)
        # rebin reference star
        self.ref_spectrum_cube = []
        for k in range(self.nspectra):
            data_func = interp1d(self.spectra[k].target.wavelengths[0], self.spectra[k].target.spectra[0],
                                 kind="cubic", fill_value="extrapolate", bounds_error=None)
            data = []
            for i in range(1, lambdas_bin_edges.size):
                data.append(quad(data_func, lambdas_bin_edges[i - 1],
                                 lambdas_bin_edges[i])[0] / self.bin_widths)
            self.ref_spectrum_cube.append(np.copy(data))
        self.ref_spectrum_cube = np.asarray(self.ref_spectrum_cube)
        # rebin errors
        self.err_cube = []
        for k in range(self.nspectra):
            err_func = interp1d(self.spectra[k].lambdas, self.spectra[k].err ** 2,
                                kind="cubic", fill_value="extrapolate", bounds_error=False)
            err = []
            for i in range(1, lambdas_bin_edges.size):
                if i in lambdas_to_mask_indices:
                    err.append(np.nan)
                else:
                    err.append(np.sqrt(np.abs(quad(err_func, lambdas_bin_edges[i - 1], lambdas_bin_edges[i])[0])
                                       / self.bin_widths))
            self.err_cube.append(np.copy(err))
        self.err_cube = np.asarray(self.err_cube)
        self.err = np.hstack(self.err_cube)
        if parameters.DEBUG:
            for k in range(self.nspectra):
                plt.errorbar(self.lambdas, self.data_cube[k], self.err_cube[k], label=f"spectrum {k}")
                # plt.plot(self.lambdas, self.ref_spectrum_cube[k] * np.max(self.data_cube[k]) / np.max(self.ref_spectrum_cube[k]), "k-")
            plt.ylim(0, 1.1 * np.max(self.data))
            plt.grid()
            plt.legend()
            plt.show()
        # rebin covariance matrices
        import time
        start = time.time()
        self.data_cov_cube = []
        lmins = []
        lmaxs = []
        for k in range(self.nspectra):
            lmins.append([])
            lmaxs.append([])
            for i in range(self.lambdas.size):
                lmins[-1].append(max(0, int(np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[i])))))
                lmaxs[-1].append(min(self.spectrum_data_cov[k].shape[0] - 1,
                           np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[i + 1]))))
        for k in range(self.nspectra):
            cov = np.zeros((self.lambdas.size, self.lambdas.size))
            for i in range(cov.shape[0]):
                #imin = max(0, int(np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[i]))))
                #imax = min(self.spectrum_data_cov[k].shape[0] - 1,
                #           np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[i + 1])))
                imin = lmins[k][i]
                imax = lmaxs[k][i]
                if imin == imax:
                    cov[i, i] = (i + 1) * 1e10
                    continue
                if i in lambdas_to_mask_indices:
                    cov[i, i] = (i + 1e10)
                    continue
                for j in range(i, cov.shape[1]):
                    #jmin = max(0, int(np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[j]))))
                    #jmax = min(self.spectrum_data_cov[k].shape[0] - 1,
                    #           np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[j + 1])))
                    jmin = lmins[k][j]
                    jmax = lmaxs[k][j]
                    # if imin == imax:
                    #     cov[i, i] = (i + 1) * 1e10
                    # elif jmin == jmax:
                    #     cov[j, j] = (j + 1) * 1e10
                    # else:
                    if jmin == jmax:
                        cov[j, j] = (j + 1) * 1e10
                    else:
                        if j in lambdas_to_mask_indices:
                            cov[j, j] = (j + 1e10)
                        else:
                            mean = np.mean(self.spectrum_data_cov[k][imin:imax, jmin:jmax])
                            cov[i, j] = mean
                            cov[j, i] = mean
            self.data_cov_cube.append(np.copy(cov))
        self.data_cov_cube = np.asarray(self.data_cov_cube)
        self.data_cov = np.zeros(self.nspectra * np.array(self.data_cov_cube[0].shape))
        for k in range(self.nspectra):
            self.data_cov[k * self.lambdas.size:(k + 1) * self.lambdas.size,
            k * self.lambdas.size:(k + 1) * self.lambdas.size] = \
                self.data_cov_cube[k]
        print("fill data_cov_cube", time.time()-start)
        start = time.time()
        self.data_invcov_cube = np.zeros_like(self.data_cov_cube)
        for k in range(self.nspectra):
            try:
                L = np.linalg.inv(np.linalg.cholesky(self.data_cov_cube[k]))
                invcov_matrix = L.T @ L
            except np.linalg.LinAlgError:
                invcov_matrix = np.linalg.inv(self.data_cov_cube[k])
            self.data_invcov_cube[k] = invcov_matrix
        self.data_invcov = np.zeros(self.nspectra * np.array(self.data_cov_cube[0].shape))
        for k in range(self.nspectra):
            self.data_invcov[k * self.lambdas.size:(k + 1) * self.lambdas.size,
            k * self.lambdas.size:(k + 1) * self.lambdas.size] = \
                self.data_invcov_cube[k]
        print("inv data_cov_cube", time.time()-start)
        start = time.time()

    def get_truth(self):
        """Load the truth parameters (if provided) from the file header.

        """
        if 'A1_T' in list(self.spectra[0].header.keys()):
            ozone_truth = self.spectrum.header['OZONE_T']
            pwv_truth = self.spectrum.header['PWV_T']
            aerosols_truth = self.spectrum.header['VAOD_T']
            self.truth = (ozone_truth, pwv_truth, aerosols_truth)
            self.true_atmospheric_transmission = []
            tatm = self.atmosphere.simulate(ozone=ozone_truth, pwv=pwv_truth, aerosols=aerosols_truth)
            for i in range(1, self.lambdas_bin_edges.size):
                self.true_atmospheric_transmission.append(quad(tatm, self.lambdas_bin_edges[i - 1],
                                                               self.lambdas_bin_edges[i])[0] / self.bin_widths)
            self.true_atmospheric_transmission = np.array(self.true_atmospheric_transmission)
        else:
            self.truth = None
        self.true_instrumental_transmission = []
        tinst = lambda lbda: self.disperser.transmission(lbda) * self.telescope.transmission(lbda)
        for i in range(1, self.lambdas_bin_edges.size):
            self.true_instrumental_transmission.append(quad(tinst, self.lambdas_bin_edges[i - 1],
                                                            self.lambdas_bin_edges[i])[0] / self.bin_widths)
        self.true_instrumental_transmission = np.array(self.true_instrumental_transmission)

    def simulate(self, ozone, pwv, aerosols, *A1s):
        """Interface method to simulate multiple spectra with a single atmosphere.

        Parameters
        ----------
        ozone: float
            Ozone parameter for Libradtran (in db).
        pwv: float
            Precipitable Water Vapor quantity for Libradtran (in mm).
        aerosols: float
            Vertical Aerosols Optical Depth quantity for Libradtran (no units).

        Returns
        -------
        lambdas: array_like
            Array of wavelengths (1D).
        model: array_like
            2D array of the spectrogram simulation.
        model_err: array_like
            2D array of the spectrogram simulation uncertainty.

        Examples
        --------

        >>> filename = 'tests/data/sim_20170530_134_spectrum.fits'
        >>> atmgrid_filename = filename.replace('spectrum', 'atmsim')
        >>> load_config("config/ctio.ini")
        >>> w = SpectrumFitWorkspace(filename, atmgrid_filename, verbose=1, plot=True, live_fit=False)
        >>> lambdas, model, model_err = w.simulate(*w.p)
        >>> w.plot_fit()

        """
        # linear regression for the instrumental transmission parameters T
        # first: force the grey terms to have an average of 1
        A1s = np.array(A1s)
        if A1s.size > 1:
            m = 1
            A1s[0] = m * A1s.size - np.sum(A1s[1:])
            self.p[3] = A1s[0]
        # Matrix M filling: hereafter a fast integration is used
        # M = []
        # for k in range(self.nspectra):
        #     atm = []
        #     a = self.atmospheres[k].simulate(ozone, pwv, aerosols)
        #     lbdas = self.atmospheres[k].lambdas
        #     for i in range(1, self.lambdas_bin_edges.size):
        #         atm.append(
        #             np.trapz(a(lbdas[self.atmosphere_lambda_bins[i]]), dx=self.atmosphere_lambda_step)
        #             / self.bin_widths)
        #     M.append(A1s[k] * np.diag(self.ref_spectrum_cube[k] * np.array(atm)))
        # hereafter: no binning but gives unbiased result on extracted spectra from simulations and truth spectra
        import time
        start = time.time()
        M = np.array([A1s[k] * np.diag(self.ref_spectrum_cube[k] *
                                       self.atmospheres[k].simulate(ozone, pwv, aerosols)(self.lambdas))
                      for k in range(self.nspectra)])
        print("compute M", time.time()-start)
        start = time.time()
        # Matrix W filling: if spectra are not independent, use these lines with einstein summations:
        # W = np.zeros((self.nspectra, self.nspectra, self.lambdas.size, self.lambdas.size))
        # for k in range(self.nspectra):
        #     W[k, k, ...] = self.data_invcov[k]
        # W_dot_M = np.einsum('lkji,kjh->lih', W, M)
        # M_dot_W_dot_M = np.einsum('lkj,lki->ij', M, W_dot_M)
        # M_dot_W_dot_M = np.zeros_like(M_dot_W_dot_M)
        # otherwise, this is much faster:
        M_dot_W_dot_M = np.sum([M[k].T @ self.data_invcov_cube[k] @ M[k] for k in range(self.nspectra)], axis=0)
        for i in range(self.lambdas.size):
            if np.sum(M_dot_W_dot_M[i]) == 0:
                M_dot_W_dot_M[i, i] = 1e-10 * np.mean(M_dot_W_dot_M)
        #try:
        # L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M))
        #cov_matrix = L.T @ L
        #cov_matrix = pinvh(M_dot_W_dot_M, check_finite=False)
        L = np.linalg.cholesky(M_dot_W_dot_M)
        inv_L = solve_triangular(L, np.eye(L.shape[0]), check_finite=False, overwrite_b=True)
        cov_matrix = inv_L.T @ inv_L
        print("inv", time.time()-start)
        start = time.time()
        #except np.linalg.LinAlgError:
         #   cov_matrix = np.linalg.inv(M_dot_W_dot_M)
        amplitude_params = cov_matrix @ (np.sum([M[k].T @ self.data_invcov_cube[k] @ self.data_cube[k]
                                                 for k in range(self.nspectra)], axis=0))
        self.M = M
        self.M_dot_W_dot_M = M_dot_W_dot_M
        model_cube = np.zeros_like(self.data_cube)
        for k in range(self.nspectra):
            model_cube[k, :] = M[k] @ amplitude_params
        self.model = np.hstack(model_cube)
        self.amplitude_params = np.copy(amplitude_params)
        self.amplitude_params_err = np.array([np.sqrt(cov_matrix[i, i])
                                              if cov_matrix[i, i] > 0 else 0 for i in range(self.lambdas.size)])
        self.amplitude_cov_matrix = np.copy(cov_matrix)
        self.model_err = np.zeros_like(self.model)
        print("algebra", time.time()-start)
        start = time.time()
        return self.lambdas, self.model, self.model_err

    def plot_fit(self):
        """Plot the fit result.

        """
        cmap_bwr = copy.copy(cm.get_cmap('bwr'))
        cmap_bwr.set_bad(color='lightgrey')
        cmap_viridis = copy.copy(cm.get_cmap('viridis'))
        cmap_viridis.set_bad(color='lightgrey')

        data_good = np.copy(self.data)
        data_good[self.outliers] = np.nan
        data = data_good.reshape(self.nspectra, self.lambdas.size)
        model = self.model.reshape(self.nspectra, self.lambdas.size)
        err = self.err.reshape(self.nspectra, self.lambdas.size)
        gs_kw = dict(width_ratios=[3, 0.15], height_ratios=[1, 1, 1])
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 6), gridspec_kw=gs_kw)
        ozone, pwv, aerosols, *A1s = self.p
        plt.suptitle(f'VAOD={aerosols:.3f}, ozone={ozone:.0f}db, PWV={pwv:.2f}mm')
        norm = np.nanmax(self.data)
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
        for i in range(3):
            ax[i, 0].set_ylabel("Spectrum index")
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
                                     f'T_inst_best_fit.pdf'),
                        dpi=100, bbox_inches='tight')

    def plot_transmissions(self):
        """Plot the fit result.

        Examples
        --------
        >>> file_names = ["./tests/data/sim_20170530_134_spectrum.fits"]
        >>> w = MultiSpectraFitWorkspace("./outputs/test", file_names, bin_width=5, verbose=True)
        >>> w.plot_transmissions()
        """
        gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 0.15])
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), gridspec_kw=gs_kw, sharex="all")
        ozone, pwv, aerosols, *A1s = self.p
        plt.suptitle(f'VAOD={aerosols:.3f}, ozone={ozone:.0f}db, PWV={pwv:.2f}mm', y=1)
        masked = self.amplitude_params_err > 1e6
        transmission = np.copy(self.amplitude_params)
        transmission_err = np.copy(self.amplitude_params_err)
        transmission[masked] = np.nan
        transmission_err[masked] = np.nan
        ax[0, 0].errorbar(self.lambdas, transmission, yerr=transmission_err,
                          label=r'$T_{\mathrm{inst}}$', fmt='k.')  # , markersize=0.1)
        ax[0, 0].set_ylabel(r'Instrumental transmission')
        ax[0, 0].set_xlim(self.lambdas[0], self.lambdas[-1])
        ax[0, 0].set_ylim(0, 1.1 * np.nanmax(transmission))
        ax[0, 0].grid(True)
        ax[0, 0].set_xlabel(r'$\lambda$ [nm]')
        if self.true_instrumental_transmission is not None:
            ax[0, 0].plot(self.lambdas, self.true_instrumental_transmission, "g-", label=r'true $T_{\mathrm{inst}}$')
            ax[1, 0].set_xlabel(r'$\lambda$ [nm]')
            ax[1, 0].grid(True)
            ax[1, 0].set_ylabel(r'Data-Truth')
            residuals = self.amplitude_params - self.true_instrumental_transmission
            residuals[masked] = np.nan
            ax[1, 0].errorbar(self.lambdas, residuals, yerr=transmission_err,
                              label=r'$T_{\mathrm{inst}}$', fmt='k.')  # , markersize=0.1)
            ax[1, 0].set_ylim(-1.1 * np.nanmax(np.abs(residuals)), 1.1 * np.nanmax(np.abs(residuals)))
        else:
            ax[1, 0].remove()
        ax[0, 0].legend()

        tatm = self.atmosphere.simulate(ozone=ozone, pwv=pwv, aerosols=aerosols)
        tatm_binned = []
        for i in range(1, self.lambdas_bin_edges.size):
            tatm_binned.append(quad(tatm, self.lambdas_bin_edges[i - 1],
                                    self.lambdas_bin_edges[i])[0] / self.bin_widths)

        ax[0, 1].errorbar(self.lambdas, tatm_binned,
                          label=r'$T_{\mathrm{atm}}$', fmt='k.')  # , markersize=0.1)
        ax[0, 1].set_ylabel(r'Atmospheric transmission')
        ax[0, 1].set_xlabel(r'$\lambda$ [nm]')
        ax[0, 1].set_xlim(self.lambdas[0], self.lambdas[-1])
        ax[0, 1].grid(True)
        if self.truth is not None:
            ax[0, 1].plot(self.lambdas, self.true_atmospheric_transmission, "b-", label=r'true $T_{\mathrm{atm}}$')
            ax[1, 1].set_xlabel(r'$\lambda$ [nm]')
            ax[1, 1].set_ylabel(r'Data-Truth')
            ax[1, 1].grid(True)
            residuals = np.asarray(tatm_binned) - self.true_atmospheric_transmission
            ax[1, 1].errorbar(self.lambdas, residuals, label=r'$T_{\mathrm{inst}}$', fmt='k.')  # , markersize=0.1)
            ax[1, 1].set_ylim(-1.1 * np.max(np.abs(residuals)), 1.1 * np.max(np.abs(residuals)))
        else:
            ax[1, 1].remove()
        ax[0, 1].legend()
        fig.tight_layout()
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:  # pragma: no cover
            if parameters.DISPLAY and self.verbose:
                plt.show()
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, f'T_inst_best_fit.pdf'), dpi=100, bbox_inches='tight')

    def plot_A1s(self):
        """
        Examples
        --------
        >>> file_names = ["./tests/data/sim_20170530_134_spectrum.fits"]
        >>> w = MultiSpectraFitWorkspace("./outputs/test", file_names, bin_width=5, verbose=True)
        >>> w.cov = np.eye(3 + w.nspectra - 1)
        >>> w.plot_A1s()

        """
        ozone, pwv, aerosols, *A1s = self.p
        zs = [self.spectra[k].airmass for k in range(self.nspectra)]
        err = np.sqrt([0] + [self.cov[ip, ip] for ip in range(3, self.cov.shape[0])])
        spectra_index = np.arange(self.nspectra)
        sc = plt.scatter(spectra_index, A1s, c=zs, s=0)
        clb = plt.colorbar(sc, label="Airmass")

        # convert time to a color tuple using the colormap used for scatter
        norm = colors.Normalize(vmin=np.min(zs), vmax=np.max(zs), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        z_color = np.array([(mapper.to_rgba(z)) for z in zs])

        # loop over each data point to plot
        for k, A1, e, color in zip(spectra_index, A1s, err, z_color):
            plt.plot(k, A1, 'o', color=color)
            plt.errorbar(k, A1, e, lw=1, capsize=3, color=color)

        plt.axhline(1, color="k", linestyle="--")
        plt.axhline(np.mean(A1s), color="b", linestyle="--",
                    label=rf"$\left\langle A_1\right\rangle = {np.mean(A1s)} (std={np.std(A1s)})$")
        plt.grid()
        plt.ylabel("Grey transmission relative to first spectrum")
        plt.xlabel("Spectrum index")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.show()

    # def jacobian(self, params, epsilon, fixed_params=None):
    #     start = time.time()
    #     lambdas, model, model_err = self.simulate(*params)
    #     model = model.flatten()[self.not_outliers]
    #     J = np.zeros((params.size, model.size))
    #     strategy = copy.copy(self.simulation.fix_psf_cube)
    #     for ip, p in enumerate(params):
    #         if fixed_params[ip]:
    #             continue
    #         if ip in self.fixed_psf_params:
    #             self.simulation.fix_psf_cube = True
    #         else:
    #             self.simulation.fix_psf_cube = False
    #         tmp_p = np.copy(params)
    #         if tmp_p[ip] + epsilon[ip] < self.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.bounds[ip][1]:
    #             epsilon[ip] = - epsilon[ip]
    #         tmp_p[ip] += epsilon[ip]
    #         tmp_lambdas, tmp_model, tmp_model_err = self.simulate(*tmp_p)
    #         J[ip] = (tmp_model.flatten()[self.not_outliers] - model) / epsilon[ip]
    #     self.simulation.fix_psf_cube = strategy
    #     self.my_logger.debug(f"\n\tJacobian time computation = {time.time() - start:.1f}s")
    #     return J


def lnprob_spectrum(p):
    """Logarithmic likelihood function to maximize in MCMC exploration.

    Parameters
    ----------
    p: array_like
        Array of SpectrumFitWorkspace parameters.

    Returns
    -------
    lp: float
        Log of the likelihood function.

    """
    global w
    lp = w.lnprior(p)
    if not np.isfinite(lp):
        return -1e20
    return lp + w.lnlike(p)


def run_multispectra_minimisation(fit_workspace, method="newton"):
    """Interface function to fit spectrum simulation parameters to data.

    Parameters
    ----------
    fit_workspace: MultiSpectraFitWorkspace
        An instance of the SpectrogramFitWorkspace class.
    method: str, optional
        Fitting method (default: 'newton').

    Examples
    --------

    >>> filename = 'tests/data/sim_20170530_134_spectrum.fits'
    >>> atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')
    >>> load_config("config/ctio.ini")
    >>> w = SpectrumFitWorkspace(filename, atmgrid_file_name=atmgrid_filename, verbose=1, plot=True, live_fit=False)
    >>> parameters.VERBOSE = True
    >>> run_spectrum_minimisation(w, method="newton")

    """
    my_logger = set_logger(__name__)
    guess = np.asarray(fit_workspace.p)
    if method != "newton":
        run_minimisation(fit_workspace, method=method)
    else:
        # fit_workspace.simulation.fast_sim = True
        # costs = np.array([fit_workspace.chisq(guess)])
        # if parameters.DISPLAY and (parameters.DEBUG or fit_workspace.live_fit):
        #     fit_workspace.plot_fit()
        # params_table = np.array([guess])
        my_logger.info(f"\n\tStart guess: {guess}\n\twith {fit_workspace.input_labels}")
        epsilon = 1e-2 * guess
        epsilon[epsilon == 0] = 1e-2
        epsilon = np.array([np.gradient(fit_workspace.atmospheres[0].OZ_Points)[0],
                            np.gradient(fit_workspace.atmospheres[0].PWV_Points)[0],
                            np.gradient(fit_workspace.atmospheres[0].AER_Points)[0]]) / 2
        epsilon = np.array(list(epsilon) + [1e-4] * w.A1s.size)
        # epsilon = np.array([100, 1e-2, 0.5])
        # epsilon[-1] = 0.001 * np.max(fit_workspace.data)

        # fit_workspace.simulation.fast_sim = True
        # fit_workspace.simulation.fix_psf_cube = False
        # run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=epsilon, fix=fit_workspace.fixed,
        #                                 xtol=1e-4, ftol=1 / fit_workspace.data.size, sigma_clip=10, niter_clip=3,
        #                                 verbose=False)

        # fit_workspace.simulation.fast_sim = False
        run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=epsilon, fix=fit_workspace.fixed,
                                        xtol=1e-6, ftol=1 / fit_workspace.data.size, sigma_clip=3, niter_clip=3,
                                        verbose=False)
        if fit_workspace.filename != "":
            parameters.SAVE = True
            ipar = np.array(np.where(np.array(fit_workspace.fixed).astype(int) == 0)[0])
            fit_workspace.plot_correlation_matrix(ipar)
            header = f"{fit_workspace.spectrum.date_obs}\nchi2: {fit_workspace.costs[-1] / fit_workspace.data.size}"
            fit_workspace.save_parameters_summary(ipar, header=header)
            # save_gradient_descent(fit_workspace, costs, params_table)
            fit_workspace.plot_fit()
            fit_workspace.plot_transmissions()
            fit_workspace.plot_A1s()
            parameters.SAVE = False


def filter_data(file_names):
    from scipy.stats import median_absolute_deviation
    new_file_names = []
    D = []
    chi2 = []
    dx = []
    amplitude = []
    for name in file_names:
        try:
            spectrum = Spectrum(name, fast_load=True)
            D.append(spectrum.header["D2CCD"])
            dx.append(spectrum.header["PIXSHIFT"])
            amplitude.append(np.sum(spectrum.data[300:]))
            if "CHI2_FIT" in spectrum.header:
                chi2.append(spectrum.header["A2_FIT"])
        except:
            print(f"fail to open {name}")
    D = np.array(D)
    dx = np.array(dx)
    chi2 = np.array(chi2)
    k = np.arange(len(D))
    plt.plot(k, amplitude)
    plt.show()
    plt.plot(k, D)
    # plt.plot(k, np.polyval(np.polyfit(k, reg, deg=1), k))
    plt.axhline(np.median(D))
    plt.axhline(np.median(D) + 3 * median_absolute_deviation(D))
    plt.axhline(np.median(D) - 3 * median_absolute_deviation(D))
    plt.grid()
    plt.show()
    filter_indices = np.logical_and(D > np.median(D) - 3 * median_absolute_deviation(D),
                                    D < np.median(D) + 3 * median_absolute_deviation(D))
    if len(chi2) > 0:
        filter_indices *= np.logical_and(chi2 > np.median(chi2) - 3 * median_absolute_deviation(chi2),
                                         chi2 < np.median(chi2) + 3 * median_absolute_deviation(chi2))
    filter_indices *= np.logical_and(dx > np.median(dx) - 3 * median_absolute_deviation(dx),
                                     dx < np.median(dx) + 3 * median_absolute_deviation(dx))
    plt.plot(k, D)
    plt.plot(k[filter_indices], D[filter_indices], "ko")
    plt.show()
    plt.plot(k, dx)
    plt.plot(k[filter_indices], dx[filter_indices], "ko")
    plt.show()
    if len(chi2) > 0:
        plt.title("chi2")
        plt.plot(k, chi2)
        plt.plot(k[filter_indices], chi2[filter_indices], "ko")
        plt.show()
    return np.array(file_names)[filter_indices]


if __name__ == "__main__":
    from argparse import ArgumentParser
    from spectractor.config import load_config
    from spectractor.fit.fitter import run_minimisation
    import os

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path', default=["tests/data/reduc_20170530_134_spectrum.fits"],
                        help="Input fits file name. It can be a list separated by spaces, or it can use * as wildcard.",
                        nargs='*')
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=True)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                        help="Write results in given output directory (default: ./outputs/).")
    parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    parser.add_argument("-c", "--config", dest="config", default="config/ctio.ini",
                        help="INI config file. (default: config.ctio.ini).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    load_config(args.config)

    file_names = []
    disperser_label = "Thor300"
    target_label = "HD111980"
    input_directory = "../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/"
    tag = "reduc"
    extension = ".fits"

    all_files = os.listdir(input_directory)
    parameters.VERBOSE = False
    parameters.DEBUG = False
    # for file_name in sorted(all_files):
    #     if tag not in file_name or extension not in file_name or "spectrum" not in file_name:
    #         continue
    #     try:
    #         s = Spectrum(os.path.join(input_directory, file_name), fast_load=True)
    #         if s.disperser_label == disperser_label and s.target.label == target_label:
    #             file_names.append(os.path.join(input_directory, file_name))
    #     except:
    #         print(f"File {file_name} buggy.")
    # HoloAmAg
    file_names = ['../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_064_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_069_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_074_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_079_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_084_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_089_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_094_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_099_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_104_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_109_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_114_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_119_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_124_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_129_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_134_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_139_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_144_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_149_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_154_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_159_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_164_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_169_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_174_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_179_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_184_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_189_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_194_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_199_spectrum.fits']
    # Thor300
    file_names = ['../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_058_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_061_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_066_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_071_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_076_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_081_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_086_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_091_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_096_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_101_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_106_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_111_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_116_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_121_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_126_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_131_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_136_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_141_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_146_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_151_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_156_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_161_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_166_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_171_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_176_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_181_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_186_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_191_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/sim_20170530_196_spectrum.fits']
    # file_names = ['../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_058_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_061_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_066_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_071_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_076_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_081_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_086_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_091_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_096_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_101_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_106_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_111_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_116_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_121_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_126_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_131_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_136_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_141_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_146_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_151_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_156_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_161_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_166_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_171_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_176_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_181_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_186_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_191_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.0/data_30may17_A2=0.1/reduc_20170530_196_spectrum.fits']

    print(file_names)

    file_names = filter_data(file_names)

    parameters.VERBOSE = True
    parameters.DEBUG = True
    output_filename = f"sim_20170530_{disperser_label}"
    w = MultiSpectraFitWorkspace(output_filename, file_names, bin_width=1, nsteps=1000, fixed_A1s=False,
                                 burnin=200, nbins=10, verbose=1, plot=True, live_fit=True)
    run_multispectra_minimisation(w, method="newton")

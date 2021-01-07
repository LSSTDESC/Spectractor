import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from astropy.io import ascii

import copy
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.simulation.simulator import SimulatorInit
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.fit.fitter import FitWorkspace, run_minimisation_sigma_clipping, run_minimisation, RegFitWorkspace
from spectractor.tools import from_lambda_to_colormap, fftconvolve_gaussian
from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.spectroscopy import HALPHA, HBETA, HGAMMA, HDELTA, O2_1, O2_2, O2B


class MultiSpectraFitWorkspace(FitWorkspace):

    def __init__(self, output_file_name, file_names, fixed_A1s=True, inject_random_A1s=False, bin_width=-1,
                 nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False):
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
            Size of the wavelength bins in nm. If negative, no binning.
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

        Examples
        --------
        >>> file_names = ["./tests/data/reduc_20170530_134_spectrum.fits"]
        >>> w = MultiSpectraFitWorkspace("./outputs/test", file_names, bin_width=5, verbose=True)
        >>> w.output_file_name
        './outputs/test'
        >>> w.spectra  #doctest: +ELLIPSIS
        [<spectractor.extractor.spectrum.Spectrum object at ...>]
        >>> w.lambdas  #doctest: +ELLIPSIS
        array([[ ...
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
        self.file_names = file_names
        for name in file_names:
            spectrum = Spectrum(name, fast_load=True)
            self.spectra.append(spectrum)
            atmgrid_file_name = name.replace("sim", "reduc").replace("spectrum.fits", "atmsim.fits")
            if os.path.isfile(atmgrid_file_name):
                self.atmospheres.append(AtmosphereGrid(name, atmgrid_file_name))
            else:
                self.my_logger.warning(f"\n\tNo atmosphere grid {atmgrid_file_name}, the fit will be slower...")
                self.atmospheres.append(Atmosphere(spectrum.airmass, spectrum.pressure, spectrum.temperature))
        self.nspectra = len(self.spectra)
        self.spectrum_lambdas = [self.spectra[k].lambdas for k in range(self.nspectra)]
        self.spectrum_data = [self.spectra[k].data for k in range(self.nspectra)]
        self.spectrum_err = [self.spectra[k].err for k in range(self.nspectra)]
        self.spectrum_data_cov = [self.spectra[k].cov_matrix for k in range(self.nspectra)]
        self.lambdas = np.empty(1)
        self.lambdas_bin_edges = None
        self.ref_spectrum_cube = []
        self.random_A1s = None
        self._prepare_data()
        self.ozone = 260.
        self.pwv = 3
        self.aerosols = 0.015
        self.reso = -1
        self.A1s = np.ones(self.nspectra)
        self.p = np.array([self.ozone, self.pwv, self.aerosols, self.reso, *self.A1s])
        self.A1_first_index = 4
        self.fixed = [False] * self.p.size
        # self.fixed[0] = True
        self.fixed[3] = True
        self.fixed[self.A1_first_index] = True
        if fixed_A1s:
            for ip in range(self.A1_first_index, len(self.fixed)):
                self.fixed[ip] = True
        self.input_labels = ["ozone", "PWV", "VAOD", "reso"] + [f"A1_{k}" for k in range(self.nspectra)]
        self.axis_names = ["ozone", "PWV", "VAOD", "reso"] + ["$A_1^{(" + str(k) + ")}$" for k in range(self.nspectra)]
        self.bounds = [(100, 700), (0, 10), (0, 0.01), (0.1, 100)] + [(1e-3, 2)] * self.nspectra
        for atmosphere in self.atmospheres:
            if isinstance(atmosphere, AtmosphereGrid):
                self.bounds[0] = (min(self.atmospheres[0].OZ_Points), max(self.atmospheres[0].OZ_Points))
                self.bounds[1] = (min(self.atmospheres[0].PWV_Points), max(self.atmospheres[0].PWV_Points))
                self.bounds[2] = (min(self.atmospheres[0].AER_Points), max(self.atmospheres[0].AER_Points))
                break
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.amplitude_truth = None
        self.lambdas_truth = None
        self.atmosphere = Atmosphere(airmass=1,
                                     pressure=float(np.mean([self.spectra[k].header["OUTPRESS"]
                                                             for k in range(self.nspectra)])),
                                     temperature=float(np.mean([self.spectra[k].header["OUTTEMP"]
                                                                for k in range(self.nspectra)])))
        self.true_instrumental_transmission = None
        self.true_atmospheric_transmission = None
        self.true_A1s = None
        self.get_truth()
        if inject_random_A1s:
            self.inject_random_A1s()

        # design matrix
        self.M = np.zeros((self.nspectra, self.lambdas.size, self.lambdas.size))
        self.M_dot_W_dot_M = np.zeros((self.lambdas.size, self.lambdas.size))

        # prepare results
        self.amplitude_params = np.ones(self.lambdas.size)
        self.amplitude_params_err = np.zeros(self.lambdas.size)
        self.amplitude_cov_matrix = np.zeros((self.lambdas.size, self.lambdas.size))

        # regularisation
        self.amplitude_priors_method = "noprior"
        self.reg = parameters.PSF_FIT_REG_PARAM * self.bin_widths
        if self.amplitude_priors_method == "spectrum":
            self.amplitude_priors = np.copy(self.true_instrumental_transmission)
            self.amplitude_priors_cov_matrix = np.eye(self.lambdas[0].size)  # np.diag(np.ones_like(self.lambdas))
            self.U = np.diag([1 / np.sqrt(self.amplitude_priors_cov_matrix[i, i]) for i in range(self.lambdas[0].size)])
            L = np.diag(-2 * np.ones(self.lambdas[0].size)) + np.diag(np.ones(self.lambdas[0].size), -1)[:-1, :-1] \
                + np.diag(np.ones(self.lambdas[0].size), 1)[:-1, :-1]
            L[0, 0] = -1
            L[-1, -1] = -1
            self.L = L.astype(float)
            self.Q = L.T @ np.linalg.inv(self.amplitude_priors_cov_matrix) @ L
            self.Q_dot_A0 = self.Q @ self.amplitude_priors

    def _prepare_data(self):
        # rebin wavelengths
        if self.bin_widths > 0:
            lambdas_bin_edges = np.arange(int(np.min(np.concatenate(list(self.spectrum_lambdas)))),
                                          int(np.max(np.concatenate(list(self.spectrum_lambdas)))) + 1,
                                          self.bin_widths)
            self.lambdas_bin_edges = lambdas_bin_edges
            lbdas = []
            for i in range(1, lambdas_bin_edges.size):
                lbdas.append(0.5 * (0*lambdas_bin_edges[i] + 2*lambdas_bin_edges[i - 1]))  # lambda bin value on left
            self.lambdas = []
            for k in range(self.nspectra):
                self.lambdas.append(np.asarray(lbdas))
            self.lambdas = np.asarray(self.lambdas)
        else:
            for k in range(1, len(self.spectrum_lambdas)):
                if self.spectrum_lambdas[k].size != self.spectrum_lambdas[0].size or \
                        not np.all(np.isclose(self.spectrum_lambdas[k], self.spectrum_lambdas[0])):
                    raise ValueError("\nIf you don't rebin your spectra, "
                                     "they must share the same wavelength arrays (in length and values).")
            self.lambdas = np.copy(self.spectrum_lambdas)
            dlbda = self.lambdas[0, -1] - self.lambdas[0, -2]
            lambdas_bin_edges = list(self.lambdas[0]) + [self.lambdas[0, -1] + dlbda]
        # mask
        lambdas_to_mask = [np.arange(300, 355, self.bin_widths)]
        for line in [HALPHA, HBETA, HGAMMA, HDELTA, O2_1, O2_2, O2B]:
            width = line.width_bounds[1]
            lambdas_to_mask += [np.arange(line.wavelength - width, line.wavelength + width, self.bin_widths)]
        lambdas_to_mask = np.concatenate(lambdas_to_mask).ravel()
        lambdas_to_mask_indices = []
        for k in range(self.nspectra):
            lambdas_to_mask_indices.append(np.asarray([np.argmin(np.abs(self.lambdas[k] - lambdas_to_mask[i]))
                                                       for i in range(lambdas_to_mask.size)]))
        # rebin atmosphere
        if self.bin_widths > 0 and isinstance(self.atmospheres[0], AtmosphereGrid):
            self.atmosphere_lambda_bins = []
            for i in range(0, lambdas_bin_edges.size):
                self.atmosphere_lambda_bins.append([])
                for j in range(0, self.atmospheres[0].lambdas.size):
                    if self.atmospheres[0].lambdas[j] >= lambdas_bin_edges[i]:
                        self.atmosphere_lambda_bins[-1].append(j)
                    if i < lambdas_bin_edges.size - 1 and self.atmospheres[0].lambdas[j] >= lambdas_bin_edges[i + 1]:
                        self.atmosphere_lambda_bins[-1] = np.array(self.atmosphere_lambda_bins[-1])
                        break
            self.atmosphere_lambda_bins = np.array(self.atmosphere_lambda_bins, dtype=object)
            self.atmosphere_lambda_step = np.gradient(self.atmospheres[0].lambdas)[0]
        # rescale data lambdas
        # D2CCD = np.median([self.spectra[k].header["D2CCD"] for k in range(self.nspectra)])
        # for k in range(self.nspectra):
        #     self.spectra[k].disperser.D = self.spectra[k].header["D2CCD"]
        #     dist = self.spectra[k].disperser.grating_lambda_to_pixel(self.spectra[k].lambdas, x0=self.spectra[k].x0)
        #     self.spectra[k].disperser.D = D2CCD
        #     self.spectra[k].lambdas = self.spectra[k].disperser.grating_pixel_to_lambda(dist, x0=self.spectra[k].x0)
        # rebin data
        self.data = np.empty(self.nspectra, dtype=np.object)
        if self.bin_widths > 0:
            for k in range(self.nspectra):
                data_func = interp1d(self.spectra[k].lambdas, self.spectra[k].data,
                                     kind="cubic", fill_value="extrapolate", bounds_error=None)
                # lambdas_truth = np.fromstring(self.spectra[k].header['LBDAS_T'][1:-1], sep=' ')
                # amplitude_truth = np.fromstring(self.spectra[k].header['AMPLIS_T'][1:-1], sep=' ', dtype=float)
                # data_func = interp1d(lambdas_truth, amplitude_truth,
                #                      kind="cubic", fill_value="extrapolate", bounds_error=None)
                data = []
                for i in range(1, lambdas_bin_edges.size):
                    data.append(quad(data_func, lambdas_bin_edges[i - 1], lambdas_bin_edges[i])[0] / self.bin_widths)
                self.data[k] = np.copy(data)
                # if parameters.DEBUG:
                #     if "LBDAS_T" in self.spectra[k].header:
                #         lambdas_truth = np.fromstring(self.spectra[k].header['LBDAS_T'][1:-1], sep=' ')
                #         amplitude_truth = np.fromstring(self.spectra[k].header['AMPLIS_T'][1:-1],sep=' ',dtype=float)
                #         plt.plot(lambdas_truth, amplitude_truth, label="truth")  # -amplitude_truth)
                #     plt.plot(self.lambdas, self.data_cube[-1], label="binned data")  # -amplitude_truth)
                #     plt.plot(self.spectra[k].lambdas, self.spectra[k].data, label="raw data")  # -amplitude_truth)
                #     # plt.title(self.spectra[k].filename)
                #     # plt.xlim(480,700)
                #     plt.grid()
                #     plt.legend()
                #     plt.show()
        else:
            for k in range(self.nspectra):
                self.data[k] = np.copy(self.spectrum_data[k])
        # rebin reference star
        self.ref_spectrum_cube = []
        if self.bin_widths > 0:
            for k in range(self.nspectra):
                data_func = interp1d(self.spectra[k].target.wavelengths[0], self.spectra[k].target.spectra[0],
                                     kind="cubic", fill_value="extrapolate", bounds_error=None)
                data = []
                for i in range(1, lambdas_bin_edges.size):
                    data.append(quad(data_func, lambdas_bin_edges[i - 1], lambdas_bin_edges[i])[0] / self.bin_widths)
                self.ref_spectrum_cube.append(np.copy(data))
        else:
            for k in range(self.nspectra):
                ref = interp1d(self.spectra[k].target.wavelengths[0], self.spectra[k].target.spectra[0],
                               kind="cubic", fill_value="extrapolate", bounds_error=None)(self.lambdas[k])
                self.ref_spectrum_cube.append(np.copy(ref))
        self.ref_spectrum_cube = np.asarray(self.ref_spectrum_cube)
        # rebin errors
        self.err = np.empty(self.nspectra, dtype=np.object)
        if self.bin_widths > 0:
            for k in range(self.nspectra):
                err_func = interp1d(self.spectra[k].lambdas, self.spectra[k].err ** 2,
                                    kind="cubic", fill_value="extrapolate", bounds_error=False)
                err = []
                for i in range(1, lambdas_bin_edges.size):
                    if i in lambdas_to_mask_indices[k]:
                        err.append(np.nan)
                    else:
                        err.append(np.sqrt(np.abs(quad(err_func, lambdas_bin_edges[i - 1], lambdas_bin_edges[i])[0])
                                           / self.bin_widths))
                self.err[k] = np.copy(err)
        else:
            for k in range(self.nspectra):
                self.err[k] = np.copy(self.spectrum_err[k])
        if parameters.DEBUG:
            for k in range(self.nspectra):
                plt.errorbar(self.lambdas[k], self.data[k], self.err[k], label=f"spectrum {k}")
                plt.ylim(0, 1.2 * np.max(self.data[k]))
            plt.grid()
            # plt.legend()
            plt.show()
        # rebin W matrices
        # import time
        # start = time.time()
        self.data_cov = np.empty(self.nspectra, dtype=np.object)
        self.W = np.empty(self.nspectra, dtype=np.object)
        if self.bin_widths > 0:
            lmins = []
            lmaxs = []
            for k in range(self.nspectra):
                lmins.append([])
                lmaxs.append([])
                for i in range(self.lambdas[k].size):
                    lmins[-1].append(max(0, int(np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[i])))))
                    lmaxs[-1].append(min(self.spectrum_data_cov[k].shape[0] - 1,
                                         np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[i + 1]))))
            for k in range(self.nspectra):
                cov = np.zeros((self.lambdas[k].size, self.lambdas[k].size))
                for i in range(cov.shape[0]):
                    # imin = max(0, int(np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[i]))))
                    # imax = min(self.spectrum_data_cov[k].shape[0] - 1,
                    #           np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[i + 1])))
                    imin = lmins[k][i]
                    imax = lmaxs[k][i]
                    if imin == imax:
                        cov[i, i] = (i + 1) * 1e10
                        continue
                    if i in lambdas_to_mask_indices[k]:
                        cov[i, i] = (i + 1e10)
                        continue
                    for j in range(i, cov.shape[1]):
                        # jmin = max(0, int(np.argmin(np.abs(self.spectrum_lambdas[k] - lambdas_bin_edges[j]))))
                        # jmax = min(self.spectrum_data_cov[k].shape[0] - 1,
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
                            if j in lambdas_to_mask_indices[k]:
                                cov[j, j] = (j + 1e10)
                            else:
                                mean = np.mean(self.spectrum_data_cov[k][imin:imax, jmin:jmax])
                                cov[i, j] = mean
                                cov[j, i] = mean
                self.data_cov[k] = np.copy(cov)
            # self.data_cov = np.zeros(self.nspectra * np.array(self.data_cov_cube[0].shape))
            # for k in range(self.nspectra):
            #     self.data_cov[k * self.lambdas[k].size:(k + 1) * self.lambdas[k].size,
            #     k * self.lambdas[k].size:(k + 1) * self.lambdas[k].size] = \
            #         self.data_cov_cube[k]
            # self.data_cov = self.data_cov_cube
            # print("fill data_cov_cube", time.time() - start)
            # start = time.time()
            for k in range(self.nspectra):
                try:
                    L = np.linalg.inv(np.linalg.cholesky(self.data_cov[k]))
                    invcov_matrix = L.T @ L
                except np.linalg.LinAlgError:
                    invcov_matrix = np.linalg.inv(self.data_cov[k])
                self.W[k] = invcov_matrix
            # self.data_invcov = np.zeros(self.nspectra * np.array(self.data_cov_cube[0].shape))
            # for k in range(self.nspectra):
            #     self.data_invcov[k * self.lambdas[k].size:(k + 1) * self.lambdas[k].size,
            #     k * self.lambdas[k].size:(k + 1) * self.lambdas[k].size] = \
            #         self.data_invcov_cube[k]
            # self.data_invcov = self.data_invcov_cube
            # print("inv data_cov_cube", time.time() - start)
            # start = time.time()
        else:
            self.W = np.empty(self.nspectra, dtype=np.object)
            for k in range(self.nspectra):
                try:
                    L = np.linalg.inv(np.linalg.cholesky(self.spectrum_data_cov[k]))
                    invcov_matrix = L.T @ L
                except np.linalg.LinAlgError:
                    invcov_matrix = np.linalg.inv(self.spectrum_data_cov[k])
                invcov_matrix[lambdas_to_mask_indices[k], :] = 0
                invcov_matrix[:, lambdas_to_mask_indices[k]] = 0
                self.W[k] = invcov_matrix

    def inject_random_A1s(self):
        random_A1s = np.random.uniform(0.5, 1, size=self.nspectra)
        for k in range(self.nspectra):
            self.data[k] *= random_A1s[k]
            self.err[k] *= random_A1s[k]
            self.data_cov[k] *= random_A1s[k] ** 2
            self.W[k] /= random_A1s[k] ** 2
        if self.true_A1s is not None:
            self.true_A1s *= random_A1s

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
            if self.bin_widths > 0:
                for i in range(1, self.lambdas_bin_edges.size):
                    self.true_atmospheric_transmission.append(quad(tatm, self.lambdas_bin_edges[i - 1],
                                                                   self.lambdas_bin_edges[i])[0] / self.bin_widths)
            else:
                self.true_atmospheric_transmission = tatm(self.lambdas[0])
            self.true_atmospheric_transmission = np.array(self.true_atmospheric_transmission)
            self.true_A1s = np.array([self.spectra[k].header["A1_T"] for k in range(self.nspectra)], dtype=float)
        else:
            self.truth = None
        self.true_instrumental_transmission = []
        tinst = lambda lbda: self.disperser.transmission(lbda) * self.telescope.transmission(lbda)
        if self.bin_widths > 0:
            for i in range(1, self.lambdas_bin_edges.size):
                self.true_instrumental_transmission.append(quad(tinst, self.lambdas_bin_edges[i - 1],
                                                                self.lambdas_bin_edges[i])[0] / self.bin_widths)
        else:
            self.true_instrumental_transmission = tinst(self.lambdas[0])
        self.true_instrumental_transmission = np.array(self.true_instrumental_transmission)

    def simulate(self, ozone, pwv, aerosols, reso, *A1s):
        """Interface method to simulate multiple spectra with a single atmosphere.

        Parameters
        ----------
        ozone: float
            Ozone parameter for Libradtran (in db).
        pwv: float
            Precipitable Water Vapor quantity for Libradtran (in mm).
        aerosols: float
            Vertical Aerosols Optical Depth quantity for Libradtran (no units).
        reso: float
            Width of the gaussian kernel to smooth the spectra (if <0: no convolution).

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
        >>> file_names = ["./tests/data/reduc_20170530_134_spectrum.fits"]
        >>> w = MultiSpectraFitWorkspace("./outputs/test", file_names, bin_width=5, verbose=True)
        >>> lambdas, model, model_err = w.simulate(*w.p)
        >>> assert np.sum(model) > 0
        >>> assert np.all(lambdas == w.lambdas)
        >>> assert np.sum(w.amplitude_params) > 0

        """
        # linear regression for the instrumental transmission parameters T
        # first: force the grey terms to have an average of 1
        A1s = np.array(A1s)
        if A1s.size > 1:
            m = 1
            A1s[0] = m * A1s.size - np.sum(A1s[1:])
            self.p[self.A1_first_index] = A1s[0]
        # Matrix M filling: hereafter a fast integration is used
        M = []
        for k in range(self.nspectra):
            atm = []
            a = self.atmospheres[k].simulate(ozone, pwv, aerosols)
            lbdas = self.atmospheres[k].lambdas
            for i in range(1, self.lambdas_bin_edges.size):
                delta = self.atmosphere_lambda_bins[i][-1] - self.atmosphere_lambda_bins[i][0]
                if delta > 0:
                    atm.append(
                        np.trapz(a(lbdas[self.atmosphere_lambda_bins[i]]), dx=self.atmosphere_lambda_step) / delta)
                else:
                    atm.append(1)
            if reso > 0:
                M.append(A1s[k] * np.diag(fftconvolve_gaussian(self.ref_spectrum_cube[k] * np.array(atm), reso)))
            else:
                M.append(A1s[k] * np.diag(self.ref_spectrum_cube[k] * np.array(atm)))
        # hereafter: no binning but gives unbiased result on extracted spectra from simulations and truth spectra
        # if self.reso > 0:
        #     M = np.array([A1s[k] * np.diag(fftconvolve_gaussian(self.ref_spectrum_cube[k] *
        #                                    self.atmospheres[k].simulate(ozone, pwv, aerosols)(self.lambdas[k]), reso))
        #                   for k in range(self.nspectra)])
        # else:
        #     M = np.array([A1s[k] * np.diag(self.ref_spectrum_cube[k] *
        #                                    self.atmospheres[k].simulate(ozone, pwv, aerosols)(self.lambdas[k]))
        #                   for k in range(self.nspectra)])
        # print("compute M", time.time() - start)
        # start = time.time()
        # for k in range(self.nspectra):
        #     plt.plot(self.atmospheres[k].lambdas, [M[k][i,i] for i in range(self.atmospheres[k].lambdas.size)])
        #     # plt.plot(self.lambdas, self.ref_spectrum_cube[k], linestyle="--")
        # plt.grid()
        # plt.title(f"reso={reso:.3f}")
        # plt.show()
        # Matrix W filling: if spectra are not independent, use these lines with einstein summations:
        # W = np.zeros((self.nspectra, self.nspectra, self.lambdas.size, self.lambdas.size))
        # for k in range(self.nspectra):
        #     W[k, k, ...] = self.data_invcov[k]
        # W_dot_M = np.einsum('lkji,kjh->lih', W, M)
        # M_dot_W_dot_M = np.einsum('lkj,lki->ij', M, W_dot_M)
        # M_dot_W_dot_M = np.zeros_like(M_dot_W_dot_M)
        # otherwise, this is much faster:
        M_dot_W_dot_M = np.sum([M[k].T @ self.W[k] @ M[k] for k in range(self.nspectra)], axis=0)
        M_dot_W_dot_D = np.sum([M[k].T @ self.W[k] @ self.data[k] for k in range(self.nspectra)], axis=0)
        if self.amplitude_priors_method != "spectrum":
            for i in range(self.lambdas[0].size):
                if np.sum(M_dot_W_dot_M[i]) == 0:
                    M_dot_W_dot_M[i, i] = 1e-10 * np.mean(M_dot_W_dot_M) * np.random.random()
            try:
                L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M))
                cov_matrix = L.T @ L
            except np.linalg.LinAlgError:
                cov_matrix = np.linalg.inv(M_dot_W_dot_M)
            amplitude_params = cov_matrix @ M_dot_W_dot_D
        else:
            M_dot_W_dot_M_plus_Q = M_dot_W_dot_M + self.reg * self.Q
            try:
                L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M_plus_Q))
                cov_matrix = L.T @ L
            except np.linalg.LinAlgError:
                cov_matrix = np.linalg.inv(M_dot_W_dot_M_plus_Q)
            amplitude_params = cov_matrix @ (M_dot_W_dot_D + self.reg * self.Q_dot_A0)

        self.M = M
        self.M_dot_W_dot_M = M_dot_W_dot_M
        self.M_dot_W_dot_D = M_dot_W_dot_D
        model_cube = []
        model_err_cube = []
        for k in range(self.nspectra):
            model_cube.append(M[k] @ amplitude_params)
            model_err_cube.append(np.zeros_like(model_cube[-1]))
        self.model = np.asarray(model_cube)
        self.model_err = np.asarray(model_err_cube)
        self.amplitude_params = np.copy(amplitude_params)
        self.amplitude_params_err = np.array([np.sqrt(cov_matrix[i, i])
                                              if cov_matrix[i, i] > 0 else 0 for i in range(self.lambdas[0].size)])
        self.amplitude_cov_matrix = np.copy(cov_matrix)
        # print("algebra", time.time() - start)
        # start = time.time()
        return self.lambdas, self.model, self.model_err

    def plot_fit(self):
        """Plot the fit result.

        Examples
        --------
        >>> file_names = 3 * ["./tests/data/reduc_20170530_134_spectrum.fits"]
        >>> w = MultiSpectraFitWorkspace("./outputs/test", file_names, bin_width=5, verbose=True)
        >>> w.simulate(*w.p)  #doctest: +ELLIPSIS
        (array(...
        >>> w.plot_fit()
        """
        cmap_bwr = copy.copy(cm.get_cmap('bwr'))
        cmap_bwr.set_bad(color='lightgrey')
        cmap_viridis = copy.copy(cm.get_cmap('viridis'))
        cmap_viridis.set_bad(color='lightgrey')

        data = copy.deepcopy(self.data)
        for k in range(self.nspectra):
            data[k][np.isnan(data[k]/self.err[k])] = np.nan
        if len(self.outliers) > 0:
            bad_indices = self.get_bad_indices()
            for k in range(self.nspectra):
                data[k][bad_indices[k]] = np.nan
                data[k] = np.ma.masked_invalid(data[k])
        data = np.array([data[k] for k in range(self.nspectra)], dtype=float)
        model = np.array([self.model[k] for k in range(self.nspectra)], dtype=float)
        err = np.array([self.err[k] for k in range(self.nspectra)], dtype=float)
        gs_kw = dict(width_ratios=[3, 0.13], height_ratios=[1, 1, 1])
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 6), gridspec_kw=gs_kw)
        ozone, pwv, aerosols, reso, *A1s = self.p
        plt.suptitle(f'VAOD={aerosols:.3f}, ozone={ozone:.0f}db, PWV={pwv:.2f}mm, reso={reso:.2f}', y=0.995)
        norm = np.nanmax(data)
        y = np.arange(0, self.nspectra+1).astype(int) - 0.5
        xx, yy = np.meshgrid(self.lambdas[0], y)
        ylbda = -0.45 * np.ones_like(self.lambdas[0][1:-1])
        # model
        im = ax[1, 0].pcolormesh(xx, yy, model / norm, vmin=0, vmax=1, cmap=cmap_viridis)
        plt.colorbar(im, cax=ax[1, 1], label='1/max(data)', format="%.1f")
        ax[1, 0].set_title("Model", fontsize=12, color='white', x=0.91, y=0.76)
        ax[1, 0].grid(color='silver', ls='solid')
        ax[1, 0].scatter(self.lambdas[0][1:-1], ylbda, cmap=from_lambda_to_colormap(self.lambdas[0][1:-1]),
                         edgecolors='None', c=self.lambdas[0][1:-1], label='', marker='o', s=20)
        # data
        im = ax[0, 0].pcolormesh(xx, yy, data / norm, vmin=0, vmax=1, cmap=cmap_viridis)
        plt.colorbar(im, cax=ax[0, 1], label='1/max(data)', format="%.1f")
        ax[0, 0].set_title("Data", fontsize=12, color='white', x=0.91, y=0.76)
        ax[0, 0].grid(color='silver', ls='solid')
        ax[0, 0].scatter(self.lambdas[0][1:-1], ylbda, cmap=from_lambda_to_colormap(self.lambdas[0][1:-1]),
                         edgecolors='None', c=self.lambdas[0][1:-1], label='', marker='o', s=20)
        # residuals
        residuals = (data - model)
        norm = err
        residuals /= norm
        std = float(np.nanstd(residuals))
        im = ax[2, 0].pcolormesh(xx, yy, residuals, vmin=-3 * std, vmax=3 * std, cmap=cmap_bwr)
        plt.colorbar(im, cax=ax[2, 1], label='(Data-Model)/Err', format="%.0f")
        # ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, color='black', x=0.84, y=0.76)
        ax[2, 0].grid(color='silver', ls='solid')
        ax[2, 0].scatter(self.lambdas[0][1:-1], ylbda, cmap=from_lambda_to_colormap(self.lambdas[0][1:-1]),
                         edgecolors='None', c=self.lambdas[0][1:-1], label='', marker='o', s=10*self.nspectra)
        ax[2, 0].text(0.05, 0.8, f'mean={np.nanmean(residuals):.3f}\nstd={np.nanstd(residuals):.3f}',
                      horizontalalignment='left', verticalalignment='bottom',
                      color='black', transform=ax[2, 0].transAxes)
        ax[2, 0].set_xlabel(r"$\lambda$ [nm]")
        for i in range(3):
            ax[i, 0].set_xlim(self.lambdas[0, 0], self.lambdas[0, -1])
            ax[i, 0].set_ylim(-0.5, self.nspectra-0.5)
            ax[i, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i, 0].set_ylabel("Spectrum index")
            ax[i, 1].get_yaxis().set_label_coords(2.6, 0.5)
            ax[i, 0].get_yaxis().set_label_coords(-0.06, 0.5)
        fig.tight_layout()
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:  # pragma: no cover
            if parameters.DISPLAY and self.verbose:
                plt.show()
        if parameters.SAVE:
            fig.savefig(self.output_file_name + '_bestfit.pdf', dpi=100, bbox_inches='tight')

    def plot_transmissions(self):
        """Plot the fit result for transmissions.

        Examples
        --------
        >>> file_names = ["./tests/data/sim_20170530_134_spectrum.fits"]
        >>> w = MultiSpectraFitWorkspace("./outputs/test", file_names, bin_width=5, verbose=True)
        >>> w.plot_transmissions()
        """
        gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 0.15])
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), gridspec_kw=gs_kw, sharex="all")
        ozone, pwv, aerosols, reso, *A1s = self.p
        plt.suptitle(f'VAOD={aerosols:.3f}, ozone={ozone:.0f}db, PWV={pwv:.2f}mm', y=1)
        masked = self.amplitude_params_err > 1e6
        transmission = np.copy(self.amplitude_params)
        transmission_err = np.copy(self.amplitude_params_err)
        transmission[masked] = np.nan
        transmission_err[masked] = np.nan
        ax[0, 0].errorbar(self.lambdas[0], transmission, yerr=transmission_err,
                          label=r'$T_{\mathrm{inst}} * \left\langle A_1 \right\rangle$', fmt='k.')  # , markersize=0.1)
        ax[0, 0].set_ylabel(r'Instrumental transmission')
        ax[0, 0].set_xlim(self.lambdas[0][0], self.lambdas[0][-1])
        ax[0, 0].set_ylim(0, 1.1 * np.nanmax(transmission))
        ax[0, 0].grid(True)
        ax[0, 0].set_xlabel(r'$\lambda$ [nm]')
        if self.true_instrumental_transmission is not None:
            ax[0, 0].plot(self.lambdas[0], self.true_instrumental_transmission, "g-",
                          label=r'true $T_{\mathrm{inst}}* \left\langle A_1 \right\rangle$')
            ax[1, 0].set_xlabel(r'$\lambda$ [nm]')
            ax[1, 0].grid(True)
            ax[1, 0].set_ylabel(r'(Data-Truth)/Err')
            norm = transmission_err
            residuals = (self.amplitude_params - self.true_instrumental_transmission) / norm
            residuals[masked] = np.nan
            ax[1, 0].errorbar(self.lambdas[0], residuals, yerr=transmission_err / norm,
                              label=r'$T_{\mathrm{inst}}$', fmt='k.')  # , markersize=0.1)
            ax[1, 0].set_ylim(-1.1 * np.nanmax(np.abs(residuals)), 1.1 * np.nanmax(np.abs(residuals)))
        else:
            ax[1, 0].remove()
        ax[0, 0].legend()

        tatm = self.atmosphere.simulate(ozone=ozone, pwv=pwv, aerosols=aerosols)
        tatm_binned = []
        for i in range(1, self.lambdas_bin_edges.size):
            tatm_binned.append(quad(tatm, self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i])[0] /
                               (self.lambdas_bin_edges[i] - self.lambdas_bin_edges[i - 1]))

        ax[0, 1].errorbar(self.lambdas[0], tatm_binned,
                          label=r'$T_{\mathrm{atm}}$', fmt='k.')  # , markersize=0.1)
        ax[0, 1].set_ylabel(r'Atmospheric transmission')
        ax[0, 1].set_xlabel(r'$\lambda$ [nm]')
        ax[0, 1].set_xlim(self.lambdas[0][0], self.lambdas[0][-1])
        ax[0, 1].grid(True)
        if self.truth is not None:
            ax[0, 1].plot(self.lambdas[0], self.true_atmospheric_transmission, "b-", label=r'true $T_{\mathrm{atm}}$')
            ax[1, 1].set_xlabel(r'$\lambda$ [nm]')
            ax[1, 1].set_ylabel(r'Data-Truth')
            ax[1, 1].grid(True)
            residuals = np.asarray(tatm_binned) - self.true_atmospheric_transmission
            ax[1, 1].errorbar(self.lambdas[0], residuals, label=r'$T_{\mathrm{inst}}$', fmt='k.')  # , markersize=0.1)
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
        if parameters.SAVE:
            fig.savefig(self.output_file_name + '_Tinst_best_fit.pdf', dpi=100, bbox_inches='tight')

    def plot_A1s(self):
        """
        Examples
        --------
        >>> file_names = ["./tests/data/sim_20170530_134_spectrum.fits"]
        >>> w = MultiSpectraFitWorkspace("./outputs/test", file_names, bin_width=5, verbose=True)
        >>> w.cov = np.eye(3 + w.nspectra - 1)
        >>> w.plot_A1s()

        """
        ozone, pwv, aerosols, reso, *A1s = self.p
        zs = [self.spectra[k].header["AIRMASS"] for k in range(self.nspectra)]
        err = np.sqrt([0] + [self.cov[ip, ip] for ip in range(self.A1_first_index, self.cov.shape[0])])
        spectra_index = np.arange(self.nspectra)
        sc = plt.scatter(spectra_index, A1s, c=zs, s=0)
        plt.colorbar(sc, label="Airmass")

        # convert time to a color tuple using the colormap used for scatter
        norm = colors.Normalize(vmin=np.min(zs), vmax=np.max(zs), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        z_color = np.array([(mapper.to_rgba(z)) for z in zs])

        # loop over each data point to plot
        for k, A1, e, color in zip(spectra_index, A1s, err, z_color):
            plt.plot(k, A1, 'o', color=color)
            plt.errorbar(k, A1, e, lw=1, capsize=3, color=color)

        if self.true_A1s is not None:
            plt.plot(spectra_index, self.true_A1s, 'b-', label="true relative $A_1$'s")

        plt.axhline(1, color="k", linestyle="--")
        plt.axhline(np.mean(A1s), color="b", linestyle="--",
                    label=rf"$\left\langle A_1\right\rangle = {np.mean(A1s):.3f}$ (std={np.std(A1s):.3f})")
        plt.grid()
        plt.ylabel("Relative grey transmissions")
        plt.xlabel("Spectrum index")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        if parameters.SAVE:
            plt.gcf().savefig(self.output_file_name + '_A1s.pdf', dpi=100, bbox_inches='tight')
        plt.show()

    def save_transmissions(self):
        ozone, pwv, aerosols, reso, *A1s = self.p
        tatm = self.atmosphere.simulate(ozone=ozone, pwv=pwv, aerosols=aerosols)
        tatm_binned = []
        for i in range(1, self.lambdas_bin_edges.size):
            tatm_binned.append(quad(tatm, self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i])[0] /
                               (self.lambdas_bin_edges[i] - self.lambdas_bin_edges[i - 1]))

        throughput = self.amplitude_params / self.disperser.transmission(self.lambdas[0])
        throughput_err = self.amplitude_params_err / self.disperser.transmission(self.lambdas[0])
        # mask_good = throughput_err < 10 * np.nanmedian(throughput_err)
        # throughput_err[~mask_good] = np.interp(self.lambdas[0][~mask_good],
        #                                        self.lambdas[0][mask_good], throughput_err[mask_good])
        # from scipy.signal import savgol_filter
        # throughput = savgol_filter(throughput, 17, 3)
        # throughput_err = savgol_filter(throughput_err, 17, 3)
        if "sim" in self.file_names[0]:
            file_name = self.output_file_name + f"_sim_transmissions.txt"
        else:
            file_name = self.output_file_name + f"_transmissions.txt"
        ascii.write([self.lambdas[0], self.amplitude_params, self.amplitude_params_err,
                     throughput, throughput_err, tatm_binned], file_name,
                    names=["wl", "Tinst", "Tinst_err", "Ttel", "Ttel_err", "Tatm"], overwrite=True)

    def jacobian(self, params, epsilon, fixed_params=None, model_input=None):
        """Generic function to compute the Jacobian matrix of a model, with numerical derivatives.

        Parameters
        ----------
        params: array_like
            The array of model parameters.
        epsilon: array_like
            The array of small steps to compute the partial derivatives of the model.
        fixed_params: array_like
            List of boolean values. If True, the parameter is considered fixed and no derivative are computed.
        model_input: array_like, optional
            A model input as a list with (x, model, model_err) to avoid an additional call to simulate().

        Returns
        -------
        J: np.array
            The Jacobian matrix.

        """
        if model_input:
            x, model, model_err = model_input
        else:
            x, model, model_err = self.simulate(*params)
        # M = np.copy(self.M)
        # inv_M_dot_W_dot_M = np.copy(self.amplitude_cov_matrix)
        # M_dot_W_dot_D = np.copy(self.M_dot_W_dot_D)
        # Tinst = np.copy(self.amplitude_params)
        if self.W.dtype == np.object and self.W[0].ndim == 2:
            J = [[] for _ in range(params.size)]
        else:
            model = model.flatten()
            J = np.zeros((params.size, model.size))
        for ip, p in enumerate(params):
            if fixed_params[ip]:
                continue
            tmp_p = np.copy(params)
            if tmp_p[ip] + epsilon[ip] < self.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.bounds[ip][1]:
                epsilon[ip] = - epsilon[ip]
            tmp_p[ip] += epsilon[ip]
            # if "A1_" not in self.input_labels[ip]:
            tmp_x, tmp_model, tmp_model_err = self.simulate(*tmp_p)
            if self.W.dtype == np.object and self.W[0].ndim == 2:
                for k in range(model.shape[0]):
                    J[ip].append((tmp_model[k] - model[k]) / epsilon[ip])
            else:
                J[ip] = (tmp_model.flatten() - model) / epsilon[ip]
            # else:
            #     import time
            #     start = time.time()
            #     k = int(self.input_labels[ip].split("_")[-1])
            #     for k in range(self.nspectra):
            #         dcov_dA1k = - 2 * inv_M_dot_W_dot_M @ (M[k].T @ self.W[k] @ M[k]) @ inv_M_dot_W_dot_M
            #         dTinst_dA1k = dcov_dA1k @ M_dot_W_dot_D + inv_M_dot_W_dot_M @ (M[k].T @ self.W[k] @ self.data[k])
            #         J[ip].append((M[k] @ Tinst + 0*M[k] @ dTinst_dA1k) / p)
            #     print("JA1", time.time()-start)
        return np.asarray(J)


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
    >>> file_names = 4 * ["./tests/data/reduc_20170530_134_spectrum.fits"]
    >>> w = MultiSpectraFitWorkspace("./outputs/test", file_names, bin_width=5, verbose=True, fixed_A1s=False)
    >>> parameters.VERBOSE = True
    >>> run_multispectra_minimisation(w, method="newton")
    >>> assert np.all(np.isclose(w.A1s, 1))

    """
    my_logger = set_logger(__name__)
    guess = np.asarray(fit_workspace.p)
    if method != "newton":
        run_minimisation(fit_workspace, method=method)
    else:
        my_logger.info(f"\n\tStart guess: {guess}\n\twith {fit_workspace.input_labels}")
        epsilon = 1e-2 * guess
        epsilon[epsilon == 0] = 1e-2
        epsilon = np.array([np.gradient(fit_workspace.atmospheres[0].OZ_Points)[0],
                            np.gradient(fit_workspace.atmospheres[0].PWV_Points)[0],
                            np.gradient(fit_workspace.atmospheres[0].AER_Points)[0], 0.04]) / 2
        epsilon = np.array(list(epsilon) + [1e-4] * fit_workspace.A1s.size)

        run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=epsilon, fix=fit_workspace.fixed,
                                        xtol=1e-6, ftol=1 / fit_workspace.data.size, sigma_clip=5, niter_clip=3,
                                        verbose=False)

        # w_reg = RegFitWorkspace(fit_workspace, opt_reg=parameters.PSF_FIT_REG_PARAM, verbose=parameters.VERBOSE)
        # run_minimisation(w_reg, method="minimize", ftol=1e-4, xtol=1e-2, verbose=parameters.VERBOSE, epsilon=[1e-1],
        #                  minimizer_method="Nelder-Mead")
        # w_reg.opt_reg = 10 ** w_reg.p[0]
        # w_reg.my_logger.info(f"\n\tOptimal regularisation parameter: {w_reg.opt_reg}")
        # fit_workspace.reg = np.copy(w_reg.opt_reg)
        # fit_workspace.opt_reg = w_reg.opt_reg
        # Recompute and save params in class attributes
        fit_workspace.simulate(*fit_workspace.p)

        # Renormalize A1s and instrumental transmission
        ozone, pwv, aerosols, reso, *A1s = fit_workspace.p
        mean_A1 = np.mean(A1s)
        fit_workspace.amplitude_params /= mean_A1
        fit_workspace.amplitude_params_err /= mean_A1
        if fit_workspace.true_A1s is not None:
            fit_workspace.true_instrumental_transmission *= np.mean(fit_workspace.true_A1s)
            fit_workspace.true_A1s /= np.mean(fit_workspace.true_A1s)

        tinst = np.array(fit_workspace.amplitude_params)
        for k in range(fit_workspace.nspectra):
            plt.plot(fit_workspace.lambdas[k],
                     tinst * np.array([fit_workspace.M[k][i, i] for i in range(fit_workspace.lambdas[k].size)]))
            plt.ylim(0, 1.2 * np.max(fit_workspace.data[k]))
        plt.grid()
        plt.title(f"reso={reso:.3f}")
        plt.show()
        if fit_workspace.filename != "":
            parameters.SAVE = True
            ipar = np.array(np.where(np.array(fit_workspace.fixed).astype(int) == 0)[0])
            fit_workspace.plot_correlation_matrix(ipar)
            header = f"{fit_workspace.spectrum.date_obs}\nchi2: {fit_workspace.costs[-1] / fit_workspace.data.size}"
            fit_workspace.save_parameters_summary(ipar, header=header)
            fit_workspace.plot_fit()
            fit_workspace.plot_transmissions()
            fit_workspace.plot_A1s()
            fit_workspace.save_transmissions()
            parameters.SAVE = False


if __name__ == "__main__":
    import doctest

    doctest.testmod()

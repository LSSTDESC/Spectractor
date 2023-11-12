import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from astropy.io import ascii

import copy
import numpy as np
from scipy.interpolate import interp1d

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.simulation.simulator import SpectrumSimulation
from spectractor.fit.fitter import (FitWorkspace, run_minimisation_sigma_clipping, run_minimisation, FitParameters)
from spectractor.tools import from_lambda_to_colormap, fftconvolve_gaussian
from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.spectroscopy import HALPHA, HBETA, HGAMMA, HDELTA, O2_1, O2_2, O2B
from spectractor.extractor.targets import load_target


def _build_sim_sample(spectra, aerosols=0.05, ozone=300, pwv=5, angstrom_exponent=None):
    """

    Parameters
    ----------
    spectra: list[Spectrum]
        List of spectra to copy as simulations.
    aerosols: float
        VAOD Vertical Aerosols Optical Depth
    angstrom_exponent: float, optional
        Angstrom exponent for aerosols.
        If None, the Atmosphere.angstrom_exponent_default value is used (default: None).
    ozone: float
        Ozone quantity in Dobson
    pwv: float
        Precipitable Water Vapor quantity in mm

    Examples
    --------
    >>> spectra = _build_sim_sample([Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")])
    >>> len(spectra)
    1
    """
    sim_spectra = []
    for spec in spectra:
        atm = Atmosphere(airmass=spec.airmass, pressure=spec.pressure, temperature=spec.temperature,
                         lambda_min=np.min(spec.lambdas), lambda_max=np.max(spec.lambdas))
        # fast_sim must be True to avoid biases (the rebinning is done after in _prepare_data())
        s = SpectrumSimulation(spec, atmosphere=atm, fast_sim=True, with_adr=True)
        s.simulate(A1=1, A2=0, aerosols=aerosols, angstrom_exponent=angstrom_exponent, ozone=ozone, pwv=pwv,
                   reso=-1, D=parameters.DISTANCE2CCD, shift_x=0, B=0)
        sim_spectra.append(s)
    return sim_spectra


def _build_test_sample(nspectra=3, aerosols=0.05, ozone=300, pwv=5, angstrom_exponent=None):
    """

    Parameters
    ----------
    nspectra: int
        Number of spectra to simulate.
    aerosols: float
        VAOD Vertical Aerosols Optical Depth
    angstrom_exponent: float, optional
        Angstrom exponent for aerosols.
        If None, the Atmosphere.angstrom_exponent_default value is used (default: None).
    ozone: float
        Ozone quantity in Dobson
    pwv: float
        Precipitable Water Vapor quantity in mm

    Examples
    --------
    >>> _build_test_sample(3)
    """
    spec = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")
    targets = ["HD111980", "HD111980"] #, "MU COL", "ETADOR", "HD205905", "HD142331", "HD160617"]
    zs = np.linspace(1, 2, len(targets))
    pressure, temperature = 800, 10
    spectra = []
    for k in range(nspectra):
        target = load_target(targets[k % len(targets)])
        # add +0.03 to airmass after a loop on targets list
        airmass = zs[k % len(targets)] + 0.03 * (k // len(targets))
        atm = Atmosphere(airmass=airmass, pressure=pressure, temperature=temperature,
                         lambda_min=np.min(spec.lambdas), lambda_max=np.max(spec.lambdas))
        # fast_sim must be True to avoid biases (the rebinning is done after in _prepare_data())
        s = SpectrumSimulation(spec, atmosphere=atm, target=target, fast_sim=True, with_adr=True)
        s.airmass = airmass
        s.pressure = pressure
        s.temperature = temperature
        s.adr_params = [s.dec, s.hour_angle, temperature, pressure, s.humidity, airmass]
        s.simulate(A1=1, A2=0, aerosols=aerosols, angstrom_exponent=angstrom_exponent, ozone=ozone, pwv=pwv,
                   reso=-1, D=parameters.DISTANCE2CCD, shift_x=0, B=0)
        spectra.append(s)
    return spectra


class MultiSpectraFitWorkspace(FitWorkspace):

    def __init__(self, output_file_name, spectra, fixed_A1s=True, fixed_deltas=True, inject_random_A1s=False,
                 bin_width=-1, verbose=False, plot=False, live_fit=False, fit_angstrom_exponent=False,
                 amplitude_priors_method="noprior"):
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
        spectra: list[Spectrum]
            List of Spectrum objects.
        bin_width: float
            Size of the wavelength bins in nm. If negative, no binning.
        fit_angstrom_exponent: bool, optional
            If True, fit angstrom exponent (default: False).
        verbose: bool, optional
            Verbosity level (default: False).
        plot: bool, optional
            If True, many plots are produced (default: False).
        live_fit: bool, optional
            If True, many plots along the fitting procedure are produced to see convergence in live (default: False).

        Examples
        --------
        >>> spectra = _build_test_sample(3)
        >>> w = MultiSpectraFitWorkspace("./outputs/test", spectra, bin_width=5, verbose=True)
        >>> w.output_file_name
        './outputs/test'
        >>> w.nspectra
        3
        >>> w.lambdas  #doctest: +ELLIPSIS
        array([[ ...
        """
        self.spectra = spectra
        self.nspectra = len(spectra)
        self.atmospheres = []
        for spectrum in self.spectra:
            self.atmospheres.append(Atmosphere(spectrum.airmass, spectrum.pressure, spectrum.temperature,
                                               lambda_min=np.min(spectrum.lambdas),
                                               lambda_max=np.max(spectrum.lambdas) + 1))
        self.fix_atm_sim = False
        self.atmospheres_curr = []
        p = np.array([0.05, self.atmospheres[0].angstrom_exponent_default, 400, 5,
                      self.atmospheres[0].angstrom_exponent_default, *np.zeros(self.nspectra), *np.ones(self.nspectra)])
        self.deltas_first_index = 5
        self.A1_first_index = self.deltas_first_index + self.nspectra
        fixed = [False] * p.size
        fixed[0] = False  # aerosols
        fixed[2] = False  # ozone
        fixed[3] = False  # pwv
        fixed[4] = True  # reso
        if not fit_angstrom_exponent:
            fixed[1] = True  # angstrom exponent
        fixed[self.A1_first_index] = True
        fixed[self.deltas_first_index] = True
        if fixed_A1s:
            for ip in range(self.A1_first_index, len(fixed)):
                fixed[ip] = True
        if fixed_deltas:
            for ip in range(self.deltas_first_index, self.nspectra + self.deltas_first_index):
                fixed[ip] = True
        labels = ["VAOD", "angstrom_exp", "ozone [db]", "PWV [mm]", "reso"] + [f"delta_{k}" for k in range(self.nspectra)] + [f"A1_{k}" for k in range(self.nspectra)]
        axis_names = ["VAOD", r'$\"a$', "ozone [db]", "PWV [mm]", "reso"] + [f"$\delta_{k}$" for k in range(self.nspectra)] + ["$A_1^{(" + str(k) + ")}$" for k in range(self.nspectra)]
        bounds = [(0, 1), (0, 3), (100, 700), (0, 20), (0.1, 100)] + [(-20, 20)] * self.nspectra + [(1e-3, 2)] * self.nspectra
        if fixed[4]:  # reso
            bounds[4] = (-1, 0)
        params = FitParameters(p, labels=labels, axis_names=axis_names, fixed=fixed, bounds=bounds)
        self.atm_params_indices = [0, 1, 2, 3]
        self.fit_angstrom_exponent = fit_angstrom_exponent
        FitWorkspace.__init__(self, params, output_file_name, verbose, plot, live_fit)
        self.my_logger = set_logger(self.__class__.__name__)
        self.output_file_name = output_file_name
        self.bin_widths = bin_width
        self.spectrum_lambdas = [self.spectra[k].lambdas for k in range(self.nspectra)]
        self.spectrum_data = [self.spectra[k].data for k in range(self.nspectra)]
        self.spectrum_err = [self.spectra[k].err for k in range(self.nspectra)]
        self.spectrum_data_cov = [self.spectra[k].cov_matrix for k in range(self.nspectra)]
        self.lambdas = np.empty(1)
        self.lambdas_bin_edges = None
        self.ref_spectrum_cube = []
        self.random_A1s = None
        self._prepare_data()
        self.amplitude_truth = None
        self.lambdas_truth = None
        self.atmosphere = Atmosphere(airmass=1,
                                     pressure=float(np.mean([self.spectra[k].pressure
                                                             for k in range(self.nspectra)])),
                                     temperature=float(np.mean([self.spectra[k].temperature
                                                                for k in range(self.nspectra)])))
        if self.atmospheres[0].emulator is not None:
            self.params.bounds[self.params.get_index("ozone [db]")] = (self.atmospheres[0].emulator.OZMIN, self.atmospheres[0].emulator.OZMAX)
            self.params.bounds[self.params.get_index("PWV [mm]")] = (self.atmospheres[0].emulator.PWVMIN, self.atmospheres[0].emulator.PWVMAX)
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
        self.amplitude_params = np.ones(self.lambdas[0].size)
        self.amplitude_params_err = np.zeros(self.lambdas[0].size)
        self.amplitude_cov_matrix = np.zeros((self.lambdas[0].size, self.lambdas[0].size))

        # regularisation
        self.amplitude_priors_method = amplitude_priors_method
        self.reg = parameters.PSF_FIT_REG_PARAM * self.bin_widths
        L = np.diag(-2 * np.ones(self.lambdas[0].size)) + np.diag(np.ones(self.lambdas[0].size), -1)[:-1, :-1] \
            + np.diag(np.ones(self.lambdas[0].size), 1)[:-1, :-1]
        L[0, 0] = -1
        L[-1, -1] = -1
        self.L = L.astype(float)
        if self.amplitude_priors_method == "spectrum" and self.true_instrumental_transmission is not None:
            self.amplitude_priors = np.copy(self.true_instrumental_transmission)
            # self.amplitude_priors = np.ones_like(self.data[0])
            self.amplitude_priors_cov_matrix = np.eye(self.lambdas[0].size)  # np.diag(np.ones_like(self.lambdas))
            self.U = np.diag([1 / np.sqrt(self.amplitude_priors_cov_matrix[i, i]) for i in range(self.lambdas[0].size)])
            self.Q = L.T @ np.linalg.inv(self.amplitude_priors_cov_matrix) @ L
            self.Q_dot_A0 = self.Q @ self.amplitude_priors

    @property
    def A1s(self):
        return self.params.values[self.A1_first_index:self.nspectra]

    def _prepare_data(self):
        # rebin wavelengths
        if self.bin_widths > 0:
            lambdas_bin_edges = np.arange(int(np.min(np.concatenate(list(self.spectrum_lambdas)))),
                                          int(np.max(np.concatenate(list(self.spectrum_lambdas)))) + 1,
                                          self.bin_widths)
            lbdas = []
            for i in range(1, lambdas_bin_edges.size):
                lbdas.append(0.5 * (0 * lambdas_bin_edges[i] + 2 * lambdas_bin_edges[i - 1]))  # lambda bin value on left
            self.lambdas = []
            for k in range(self.nspectra):
                self.lambdas.append(np.asarray(lbdas))
            self.lambdas = np.asarray(self.lambdas)
        else:
            self.my_logger.warning(f'\n\tMultispectra fit code works without rebinning '
                                   f'but must be tested on a simulation to trust outputs.')
            self.lambdas = np.copy(self.spectrum_lambdas)
            dlbda = self.lambdas[0, -1] - self.lambdas[0, -2]
            lambdas_bin_edges = list(self.lambdas[0]) + [self.lambdas[0, -1] + dlbda]
        self.lambdas_bin_edges = np.asarray(lambdas_bin_edges)
        # mask
        lambdas_to_mask = [np.arange(300, 335, 1)]
        for line in [HALPHA, HBETA, HGAMMA, HDELTA, O2_1, O2_2, O2B]:
            width = line.width_bounds[1]
            lambdas_to_mask += [np.arange(line.wavelength - width, line.wavelength + width, 1)]
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
        self.data = np.empty(self.nspectra, dtype=object)
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
                    lbdas = np.arange(self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i] + 1, 1)
                    data.append(np.trapz(data_func(lbdas), x=lbdas) / self.bin_widths)
                    # data.append(quad(data_func, lambdas_bin_edges[i - 1], lambdas_bin_edges[i])[0] / self.bin_widths)
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
                    lbdas = np.arange(self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i] + 1, 1)
                    data.append(np.trapz(data_func(lbdas), x=lbdas) / self.bin_widths)
                    # data.append(quad(data_func, lambdas_bin_edges[i - 1], lambdas_bin_edges[i])[0] / self.bin_widths)
                self.ref_spectrum_cube.append(np.copy(data))
        else:
            for k in range(self.nspectra):
                ref = interp1d(self.spectra[k].target.wavelengths[0], self.spectra[k].target.spectra[0],
                               kind="cubic", fill_value="extrapolate", bounds_error=None)(self.spectrum_lambdas[k])
                self.ref_spectrum_cube.append(np.copy(ref))
        self.ref_spectrum_cube = np.asarray(self.ref_spectrum_cube)
        # rebin errors
        self.err = np.empty(self.nspectra, dtype=object)
        if self.bin_widths > 0:
            for k in range(self.nspectra):
                err_func = interp1d(self.spectra[k].lambdas, self.spectra[k].err ** 2,
                                    kind="cubic", fill_value="extrapolate", bounds_error=False)
                err = []
                for i in range(1, lambdas_bin_edges.size):
                    if i in lambdas_to_mask_indices[k]:
                        err.append(np.nan)
                    else:
                        lbdas = np.arange(self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i] + 1, 1)
                        err.append(np.sqrt(np.abs(np.trapz(err_func(lbdas), x=lbdas) / self.bin_widths)))
                        # err.append(np.sqrt(np.abs(quad(err_func, lambdas_bin_edges[i - 1], lambdas_bin_edges[i])[0])  / self.bin_widths))
                self.err[k] = np.copy(err)
        else:
            for k in range(self.nspectra):
                self.err[k] = np.copy(self.spectrum_err[k])
        # rebin W matrices
        # import time
        # start = time.time()
        self.data_cov = np.empty(self.nspectra, dtype=object)
        self.W = np.empty(self.nspectra, dtype=object)
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
                                if i == j:
                                    mean = np.mean(self.spectrum_data_cov[k][imin:imax, jmin:jmax])
                                    cov[i, j] = mean
                                    cov[j, i] = mean
                                else:
                                    cov[i, j] = 0
                                    cov[j, i] = 0

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
            self.W = np.empty(self.nspectra, dtype=object)
            for k in range(self.nspectra):
                try:
                    L = np.linalg.inv(np.linalg.cholesky(self.spectrum_data_cov[k]))
                    invcov_matrix = L.T @ L
                except np.linalg.LinAlgError:
                    invcov_matrix = np.linalg.inv(self.spectrum_data_cov[k])
                invcov_matrix[lambdas_to_mask_indices[k], :] = 0
                invcov_matrix[:, lambdas_to_mask_indices[k]] = 0
                self.W[k] = invcov_matrix
                self.data_cov[k] = np.copy(self.spectrum_data_cov[k])

    def inject_random_A1s(self):  # pragma: no cover
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
            ozone_truth = self.spectra[0].header['OZONE_T']
            pwv_truth = self.spectra[0].header['PWV_T']
            aerosols_truth = self.spectra[0].header['VAOD_T']
            if "ANGEXP_T" in self.spectra[0].header and self.spectra[0].header["ANGEXP_T"] is not None:
                angstrom_exponent_truth = self.spectra[0].header["ANGEXP_T"]
                self.truth = (aerosols_truth, angstrom_exponent_truth, ozone_truth, pwv_truth)
            else:
                angstrom_exponent_truth = None
                self.truth = (aerosols_truth, None, ozone_truth, pwv_truth)
            self.true_atmospheric_transmission = []
            tatm = self.atmosphere.simulate(ozone=ozone_truth, pwv=pwv_truth, aerosols=aerosols_truth, 
                                            angstrom_exponent=angstrom_exponent_truth)
            if self.bin_widths > 0:
                for i in range(1, self.lambdas_bin_edges.size):
                    lbdas = np.arange(self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i] + 1, 1)
                    self.true_atmospheric_transmission.append(np.trapz(tatm(lbdas), x=lbdas) / self.bin_widths)
                    # self.true_atmospheric_transmission.append(quad(tatm, self.lambdas_bin_edges[i - 1],
                    #                                                self.lambdas_bin_edges[i])[0] / self.bin_widths)
            else:
                self.true_atmospheric_transmission = tatm(self.lambdas[0])
            self.true_atmospheric_transmission = np.array(self.true_atmospheric_transmission)
            self.true_A1s = np.array([self.spectra[k].header["A1_T"] for k in range(self.nspectra)], dtype=float)
        else:
            self.truth = None
        self.true_instrumental_transmission = []
        tinst = lambda lbda: self.spectra[0].disperser.transmission(lbda) * self.spectra[0].throughput.transmission(lbda)
        if self.bin_widths > 0:
            for i in range(1, self.lambdas_bin_edges.size):
                lbdas = np.arange(self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i] + 1, 1)
                self.true_instrumental_transmission.append(np.trapz(tinst(lbdas), x=lbdas) / self.bin_widths)
                # self.true_instrumental_transmission.append(quad(tinst, self.lambdas_bin_edges[i - 1],
                #                                                 self.lambdas_bin_edges[i])[0] / self.bin_widths)
        else:
            self.true_instrumental_transmission = tinst(self.lambdas[0])
        self.true_instrumental_transmission = np.array(self.true_instrumental_transmission)

    def simulate(self, aerosols, angstrom_exponent, ozone, pwv, reso, *A1s):
        """Interface method to simulate multiple spectra with a single atmosphere.

        Parameters
        ----------
        aerosols: float
            Vertical Aerosols Optical Depth quantity for Libradtran (no units).
        angstrom_exponent: float
            Angstrom exponent for aerosols.
        ozone: float
            Ozone parameter for Libradtran (in db).
        pwv: float
            Precipitable Water Vapor quantity for Libradtran (in mm).
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
        >>> spectra = _build_test_sample(3)
        >>> w = MultiSpectraFitWorkspace("./outputs/test", spectra, bin_width=5, verbose=True)
        >>> lambdas, model, model_err = w.simulate(*w.params.values)
        >>> assert np.sum(model) > 0
        >>> assert np.all(lambdas == w.lambdas)
        >>> assert np.sum(w.amplitude_params) > 0

        # Test without rebinning
        >>> w = MultiSpectraFitWorkspace("./outputs/test", spectra, bin_width=-1, verbose=True)
        >>> lambdas, model, model_err = w.simulate(*w.params.values)
        >>> assert np.sum(model) > 0
        >>> assert np.all(lambdas == w.lambdas)
        >>> assert np.sum(w.amplitude_params) > 0

        """
        if not self.fit_angstrom_exponent:
            angstrom_exponent = None
        # linear regression for the instrumental transmission parameters T
        # first: force the grey terms to have an average of 1
        deltas = np.array(A1s)[:self.nspectra]
        A1s = np.array(A1s)[self.nspectra:]
        # if A1s.size > 1:
        #     m = 1
        #     A1s[0] = m * A1s.size - np.sum(A1s[1:])
        #     self.params.values[self.A1_first_index] = A1s[0]
        A1s[0] = 1
        # second: force the delta lambda terms to have an average of 0
        if deltas.size > 1:
            m = 0
            deltas[0] = m * deltas.size - np.sum(deltas[1:])
            self.params.values[self.deltas_first_index] = deltas[0]
        # Matrix M filling: hereafter a fast integration is used
        M = []
        if self.fix_atm_sim is False:
            self.atmospheres_curr = []
            for k in range(self.nspectra):
                atm = []
                a = self.atmospheres[k].simulate(aerosols=aerosols, ozone=ozone, pwv=pwv, angstrom_exponent=angstrom_exponent)
                if self.bin_widths > 0:
                    for i in range(1, self.lambdas_bin_edges.size):
                        delta = self.lambdas_bin_edges[i] - self.lambdas_bin_edges[i-1]
                        if delta > 0:
                            # atm.append(quad(a, self.lambdas_bin_edges[i-1] + deltas[k], self.lambdas_bin_edges[i] + deltas[k])[0] / delta)
                            lbdas = np.arange(self.lambdas_bin_edges[i-1] + deltas[k], self.lambdas_bin_edges[i] + deltas[k] + 1, 1)
                            atm.append(np.trapz(a(lbdas), x=lbdas)/delta)
                        else:
                            atm.append(1)
                else:
                    atm = a(self.spectrum_lambdas[k])
                self.atmospheres_curr.append(np.asarray(atm))
                # fig = plt.figure()
                # lbdas = np.arange(300, 1100, 1)
                # plt.plot(self.lambdas_bin_edges[:-1], a(self.lambdas_bin_edges[:-1])-np.array(atm))
                # # plt.plot(self.lambdas_bin_edges[:-1], atm)
                # plt.title(f"{k} {aerosols=}")
                # plt.show()
                # self.my_logger.warning(f"{k=} {atm=}")
        for k in range(self.nspectra):
            if reso > 0:
                M.append(A1s[k] * np.diag(fftconvolve_gaussian(self.ref_spectrum_cube[k] * self.atmospheres_curr[k], reso)))
            else:
                M.append(A1s[k] * np.diag(self.ref_spectrum_cube[k] * self.atmospheres_curr[k]))
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
        M_dot_W = [M[k].T @ self.W[k] for k in range(self.nspectra)]
        M_dot_W_dot_M = np.sum([M_dot_W[k] @ M[k] for k in range(self.nspectra)], axis=0)
        M_dot_W_dot_D = np.sum([M_dot_W[k] @ self.data[k] for k in range(self.nspectra)], axis=0)
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
            model = M[k] @ amplitude_params
            model[model < 0] = 0
            model_cube.append(model)
            model_err_cube.append(np.zeros_like(model))
        self.model = np.asarray(model_cube)
        self.model_err = np.asarray(model_err_cube)
        self.amplitude_params = np.copy(amplitude_params)
        self.amplitude_params_err = np.array([np.sqrt(cov_matrix[i, i])
                                              if cov_matrix[i, i] > 0 else 0 for i in range(self.lambdas[0].size)])
        self.amplitude_cov_matrix = np.copy(cov_matrix)
        # print("algebra", time.time() - start)
        # start = time.time()
        return self.lambdas, self.model, self.model_err

    def plot_fit(self):  # pragma: no cover
        """Plot the fit result.

        Examples
        --------
        >>> spectra = _build_test_sample(3)
        >>> w = MultiSpectraFitWorkspace("./outputs/test", spectra, bin_width=5, verbose=True)
        >>> w.simulate(*w.params.values)  #doctest: +ELLIPSIS
        (array(...
        >>> w.plot_fit()
        """
        cmap_bwr = copy.copy(colormaps.get_cmap('bwr'))
        cmap_bwr.set_bad(color='lightgrey')
        cmap_viridis = copy.copy(colormaps.get_cmap('viridis'))
        cmap_viridis.set_bad(color='lightgrey')

        data = copy.deepcopy(self.data)
        for k in range(self.nspectra):
            data[k][np.isnan(data[k] / self.err[k])] = np.nan
        if len(self.outliers) > 0:
            bad_indices = self.get_bad_indices()
            for k in range(self.nspectra):
                data[k][bad_indices[k]] = np.nan
                data[k] = np.ma.masked_invalid(data[k])
        data = np.array([data[k] for k in range(self.nspectra)], dtype=float)
        model = np.array([self.model[k] for k in range(self.nspectra)], dtype=float)
        residuals = np.array([(self.model[k] - data[k]) / self.err[k] for k in range(self.nspectra)], dtype=float)
        gs_kw = dict(width_ratios=[3, 0.13], height_ratios=[1, 1, 1])
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 6), gridspec_kw=gs_kw)
        # plt.suptitle(f'VAOD={aerosols:.3f}, ozone={ozone:.0f}db, PWV={pwv:.2f}mm, reso={reso:.2f}', y=0.995)
        norm = np.nanmax(data)
        y = np.arange(0, self.nspectra).astype(int) + 0 * 0.5
        xx, yy = np.meshgrid(self.lambdas[0], y)
        ylbda = -0.45 * np.ones_like(self.lambdas[0][1:-1])
        # model
        im = ax[1, 0].pcolormesh(xx, yy, model / norm, vmin=0, vmax=1, cmap=cmap_viridis, shading="auto")
        plt.colorbar(im, cax=ax[1, 1], label='1/max(data)', format="%.1f")
        ax[1, 0].set_title("Model", fontsize=12, color='white', x=0.91, y=0.76)
        ax[1, 0].grid(color='silver', ls='solid')
        ax[1, 0].scatter(self.lambdas[0][1:-1], ylbda, cmap=from_lambda_to_colormap(self.lambdas[0][1:-1]),
                         edgecolors='None', c=self.lambdas[0][1:-1], label='', marker='o', s=20)
        # data
        im = ax[0, 0].pcolormesh(xx, yy, data / norm, vmin=0, vmax=1, cmap=cmap_viridis, shading="auto")
        plt.colorbar(im, cax=ax[0, 1], label='1/max(data)', format="%.1f")
        ax[0, 0].set_title("Data", fontsize=12, color='white', x=0.91, y=0.76)
        ax[0, 0].grid(color='silver', ls='solid')
        ax[0, 0].scatter(self.lambdas[0][1:-1], ylbda, cmap=from_lambda_to_colormap(self.lambdas[0][1:-1]),
                         edgecolors='None', c=self.lambdas[0][1:-1], label='', marker='o', s=20)
        # residuals
        res_mean = float(np.max([np.nanmean(list(res)) for res in residuals]))
        res_std = max(1., float(np.min([np.nanstd(list(res)) for res in residuals])))
        im = ax[2, 0].pcolormesh(xx, yy, residuals, vmin=-3 * res_std, vmax=3 * res_std, cmap=cmap_bwr, shading="auto")
        plt.colorbar(im, cax=ax[2, 1], label='(Data-Model)/Err', format="%.0f")
        # ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, color='black', x=0.84, y=0.76)
        ax[2, 0].grid(color='silver', ls='solid')
        ax[2, 0].scatter(self.lambdas[0][1:-1], ylbda, cmap=from_lambda_to_colormap(self.lambdas[0][1:-1]),
                         edgecolors='None', c=self.lambdas[0][1:-1], label='', marker='o', s=10 * self.nspectra)
        ax[2, 0].text(0.05, 0.8, f'mean={res_mean:.3f}\nstd={res_std:.3f}',
                      horizontalalignment='left', verticalalignment='bottom',
                      color='black', transform=ax[2, 0].transAxes)
        ax[2, 0].set_xlabel(r"$\lambda$ [nm]")
        for i in range(3):
            ax[i, 0].set_xlim(self.lambdas[0, 0], self.lambdas[0, -1])
            ax[i, 0].set_ylim(-0.5, self.nspectra - 0.5)
            ax[i, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i, 0].set_ylabel("Spectrum index")
            ax[i, 1].get_yaxis().set_label_coords(2.6, 0.5)
            ax[i, 0].get_yaxis().set_label_coords(-0.06, 0.5)
        fig.tight_layout()
        if parameters.SAVE:
            fig.savefig(self.output_file_name + '_bestfit.pdf', dpi=100, bbox_inches='tight')
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:  # pragma: no cover
            if parameters.DISPLAY and self.verbose:
                plt.show()

    def plot_transmissions(self):  # pragma: no cover
        """Plot the fit result for transmissions.

        Examples
        --------
        >>> spectra = _build_test_sample(3)
        >>> w = MultiSpectraFitWorkspace("./outputs/test", spectra, bin_width=5, verbose=True)
        >>> _ = w.simulate(*w.params.values)
        >>> w.plot_transmissions()
        """
        gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 0.15])
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), gridspec_kw=gs_kw, sharex="all")
        aerosols, angstrom_exponent, ozone, pwv, reso, *A1s = self.params.values
        plt.suptitle(f'VAOD={aerosols:.3f}, ang_exp={angstrom_exponent}, ozone={ozone:.0f}db, PWV={pwv:.2f}mm', y=1)
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

        tatm = self.atmosphere.simulate(ozone=ozone, pwv=pwv, aerosols=aerosols, angstrom_exponent=angstrom_exponent)
        tatm_binned = []
        for i in range(1, self.lambdas_bin_edges.size):
            lbdas = np.arange(self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i] + 1, 1)
            tatm_binned.append(np.trapz(tatm(lbdas), x=lbdas) / (self.lambdas_bin_edges[i] - self.lambdas_bin_edges[i - 1]))
            # tatm_binned.append(quad(tatm, self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i])[0] /
            #                    (self.lambdas_bin_edges[i] - self.lambdas_bin_edges[i - 1]))

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
        if parameters.SAVE:
            fig.savefig(self.output_file_name + '_Tinst_best_fit.pdf', dpi=100, bbox_inches='tight')
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:  # pragma: no cover
            if parameters.DISPLAY and self.verbose:
                plt.show()

    def plot_A1s(self):  # pragma: no cover
        """
        Examples
        --------
        >>> spectra = _build_test_sample(3)
        >>> w = MultiSpectraFitWorkspace("./outputs/test", spectra, bin_width=5, verbose=True)
        >>> w.cov = np.eye(3 + w.nspectra - 1)
        >>> w.plot_A1s()

        """
        A1s = [self.params.get_parameter(key).value for key in self.params.labels if "A1_" in key]
        zs = [self.spectra[k].airmass for k in range(self.nspectra)]
        err = [self.params.get_parameter(key).err for key in self.params.labels if "A1_" in key]
        spectra_index = np.arange(len(A1s))

        fig = plt.figure()
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
            fig.savefig(self.output_file_name + '_A1s.pdf', dpi=100, bbox_inches='tight')
        plt.show()

    def save_transmissions(self):
        aerosols, angstrom_exponent, ozone, pwv, reso, *A1s = self.params.values
        tatm = self.atmosphere.simulate(ozone=ozone, pwv=pwv, aerosols=aerosols, angstrom_exponent=angstrom_exponent)
        tatm_binned = []
        for i in range(1, self.lambdas_bin_edges.size):
            lbdas = np.arange(self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i] + 1, 1)
            tatm_binned.append(np.trapz(tatm(lbdas), x=lbdas) / (self.lambdas_bin_edges[i] - self.lambdas_bin_edges[i - 1]))
            # tatm_binned.append(quad(tatm, self.lambdas_bin_edges[i - 1], self.lambdas_bin_edges[i])[0] /
            #                    (self.lambdas_bin_edges[i] - self.lambdas_bin_edges[i - 1]))

        throughput = self.amplitude_params / self.spectra[0].disperser.transmission(self.lambdas[0])
        throughput_err = self.amplitude_params_err / self.spectra[0].disperser.transmission(self.lambdas[0])
        # mask_good = throughput_err < 10 * np.nanmedian(throughput_err)
        # throughput_err[~mask_good] = np.interp(self.lambdas[0][~mask_good],
        #                                        self.lambdas[0][mask_good], throughput_err[mask_good])
        # from scipy.signal import savgol_filter
        # throughput = savgol_filter(throughput, 17, 3)
        # throughput_err = savgol_filter(throughput_err, 17, 3)
        # if "sim" in self.file_names[0]:
        # file_name = self.output_file_name + f"_sim_transmissions.txt"
        # else:
        file_name = self.output_file_name + f"_transmissions.txt"
        ascii.write([self.lambdas[0], self.amplitude_params, self.amplitude_params_err,
                     throughput, throughput_err, tatm_binned], file_name,
                    names=["wl", "Tinst", "Tinst_err", "Ttel", "Ttel_err", "Tatm"], overwrite=True)

    def jacobian(self, params, epsilon, model_input=None):
        """Generic function to compute the Jacobian matrix of a model, with numerical derivatives.

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
        if model_input:
            x, model, model_err = model_input
        else:
            x, model, model_err = self.simulate(*params)
        # M = np.copy(self.M)
        # inv_M_dot_W_dot_M = np.copy(self.amplitude_cov_matrix)
        # M_dot_W_dot_D = np.copy(self.M_dot_W_dot_D)
        # Tinst = np.copy(self.amplitude_params)
        J = [[] for _ in range(params.size)]
        for ip, p in enumerate(params):
            if self.params.fixed[ip]:
                continue
            if ip in self.atm_params_indices or "delta_" in self.params.labels[ip]:
                self.fix_atm_sim = False
            else:
                self.fix_atm_sim = True
            tmp_p = np.copy(params)
            if tmp_p[ip] + epsilon[ip] < self.params.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.params.bounds[ip][1]:
                epsilon[ip] = - epsilon[ip]
            tmp_p[ip] += epsilon[ip]
            # if "XXXX_" not in self.params.labels[ip]:
            tmp_x, tmp_model, tmp_model_err = self.simulate(*tmp_p)
            for s in range(model.shape[0]):
                J[ip].append((tmp_model[s] - model[s]) / epsilon[ip])
            # else:  # don't work
            #     k = int(self.params.labels[ip].split("_")[-1]) - 1
            #     dM_dA1k = np.zeros_like(M)
            #     dM_dA1k[k] = np.copy(M[k]) / p
            #     for s in range(self.nspectra):
            #         dcov_dA1s = - 2 * inv_M_dot_W_dot_M @ (dM_dA1k[s].T @ self.W[s] @ M[s]) @ inv_M_dot_W_dot_M
            #         dTinst_dA1s = dcov_dA1s @ M_dot_W_dot_D + inv_M_dot_W_dot_M @ (dM_dA1k[s].T @ self.W[s] @ self.data[s])
            #         J[ip].append((M[s] @ dTinst_dA1s) + dM_dA1k[s] @ Tinst)
        self.fix_atm_sim = False
        return np.asarray(J, dtype=object)

    def chisq(self, p, model_output=False):
        # aerosols, ozone, pwv, reso, *A1s = p
        # tatm = self.atmosphere.simulate(ozone=ozone, pwv=pwv, aerosols=aerosols)(self.lambdas[0])
        # penalty = 10000*np.linalg.norm((self.L.T @ self.amplitude_params)*savgol_filter(tatm, 5, 2, deriv=2)/self.amplitude_params_err)**2
        # penalty = np.linalg.norm((self.L.T @ self.amplitude_params))**2 #/self.amplitude_params_err)**2
        # from scipy.signal import savgol_filter
        penalty = 0  # 1e5 * np.linalg.norm(savgol_filter(self.amplitude_params, 5, 2, deriv=2)[5:-5]) ** 2
        if model_output:
            chisq, x, model, model_err = super().chisq(p, model_output=model_output)
            return chisq + penalty, x, model, model_err
        else:
            chisq = super().chisq(p, model_output=model_output)
            return chisq + penalty


def run_multispectra_minimisation(fit_workspace, method="newton", verbose=False, sigma_clip=5, niter_clip=3):
    """Interface function to fit spectrum simulation parameters to data.

    Parameters
    ----------
    fit_workspace: MultiSpectraFitWorkspace
        An instance of the SpectrogramFitWorkspace class.
    method: str, optional
        Fitting method (default: 'newton').

    Examples
    --------
    >>> spectra = _build_test_sample(20, aerosols=0.5, angstrom_exponent=1.5, ozone=300, pwv=3)
    >>> parameters.VERBOSE = True
    >>> parameters.DEBUG = True
    >>> w = MultiSpectraFitWorkspace("./outputs/test", spectra, bin_width=5, verbose=True,
    ... fixed_deltas=True, fixed_A1s=False, fit_angstrom_exponent=True)
    >>> run_multispectra_minimisation(w, method="newton", verbose=True, sigma_clip=10)
    >>> print(w.params.print_parameters_summary())  #doctest: +ELLIPSIS
    VAOD:...
    >>> w.plot_fit()
    >>> assert np.all(np.isclose(w.A1s, 1, atol=5e-3))

    """
    my_logger = set_logger(__name__)
    guess = np.asarray(fit_workspace.params.values)
    if method != "newton":
        run_minimisation(fit_workspace, method=method)
    else:
        my_logger.info(f"\n\tStart guess: {guess}\n\twith {fit_workspace.params.labels}")
        epsilon = 1e-4 * guess
        epsilon[epsilon == 0] = 1e-4

        # very touchy to avoid biases when fitting simulations with A1s
        # good epsilons: epsilon = np.array([0.0001, 1e-4, 5, 0.05, 0.01]) + [1e-2] * fit_workspace.nspectra + [1e-4] * fit_workspace.nspectra)
        #epsilon = np.array([0.0001, 1e-3, 5, 0.01, 0.01])
        #epsilon = np.array(list(epsilon) + [1e-3] * fit_workspace.nspectra + [1e-4] * fit_workspace.nspectra)

        run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=epsilon,
                                        xtol=1e-6, ftol=1e-4 / fit_workspace.data.size,
                                        sigma_clip=sigma_clip, niter_clip=niter_clip,
                                        verbose=verbose, with_line_search=True)

        # w_reg = RegFitWorkspace(fit_workspace, opt_reg=parameters.PSF_FIT_REG_PARAM, verbose=parameters.VERBOSE)
        # run_minimisation(w_reg, method="minimize", ftol=1e-4, xtol=1e-2, verbose=parameters.VERBOSE, epsilon=[1e-1],
        #                  minimizer_method="Nelder-Mead")
        # w_reg.opt_reg = 10 ** w_reg.p[0]
        # w_reg.my_logger.info(f"\n\tOptimal regularisation parameter: {w_reg.opt_reg}")
        # fit_workspace.reg = np.copy(w_reg.opt_reg)
        # fit_workspace.opt_reg = w_reg.opt_reg
        # Recompute and save params in class attributes
        fit_workspace.simulate(*fit_workspace.params.values)

        # Renormalize A1s and instrumental transmission
        aerosols, angstrom_exponent, ozone, pwv,reso, *A1s = fit_workspace.params.values
        A1s = np.array(A1s)[fit_workspace.nspectra:]
        mean_A1 = np.mean(A1s)
        fit_workspace.amplitude_params /= mean_A1
        fit_workspace.amplitude_params_err /= mean_A1
        if fit_workspace.true_A1s is not None:
            fit_workspace.true_instrumental_transmission *= np.mean(fit_workspace.true_A1s)
            fit_workspace.true_A1s /= np.mean(fit_workspace.true_A1s)

        if fit_workspace.filename != "":
            parameters.SAVE = True
            fit_workspace.params.plot_correlation_matrix()
            fit_workspace.plot_fit()
            fit_workspace.plot_transmissions()
            fit_workspace.plot_A1s()
            fit_workspace.save_transmissions()
            parameters.SAVE = False


if __name__ == "__main__":
    import doctest

    doctest.testmod()

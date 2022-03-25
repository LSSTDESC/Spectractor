import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.interpolate import interp1d

from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.dispersers import Grating, Hologram
from spectractor.extractor.targets import Target
from spectractor.extractor.psf import load_PSF
from spectractor.tools import fftconvolve_gaussian, ensure_dir
from spectractor.config import set_logger
from spectractor.simulation.throughput import TelescopeTransmission
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
import spectractor.parameters as parameters
from spectractor.simulation.adr import adr_calib, flip_and_rotate_adr_to_image_xy_coordinates


class SpectrumSimulation(Spectrum):

    def __init__(self, spectrum, atmosphere, telescope, disperser, fast_sim=True):
        """Class to simulate cross spectrum.

        Parameters
        ----------
        spectrum: Spectrum
            Spectrum instance to load main properties before simulation.
        atmosphere: Atmosphere
            Atmosphere or AtmosphereGrid instance to make the atmospheric simulation.
        telescope: TelescopeTransmission
            Telescope transmission.
        disperser: Grating
            Disperser instance.
        fast_sim: bool, optional
            If True, do a fast simulation without integrating within the wavelength bins (default: True).

        Examples
        --------
        >>> spectrum, telescope, disperser, target = SimulatorInit("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> atmosphere = Atmosphere(airmass=1.2, pressure=800, temperature=10)
        >>> sim = SpectrumSimulation(spectrum, atmosphere, telescope, disperser, fast_sim=True)

        """
        Spectrum.__init__(self)
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        self.my_logger = set_logger(self.__class__.__name__)
        self.disperser = disperser
        self.telescope = telescope
        self.atmosphere = atmosphere
        self.fast_sim = fast_sim
        # save original pixel distances to zero order
        # self.disperser.grating_lambda_to_pixel(self.lambdas, x0=self.x0, order=1)
        # now reset data
        self.lambdas = None
        self.lambdas_order2 = None
        self.err = None
        self.model = lambda x: np.zeros_like(x)
        self.model_err = lambda x: np.zeros_like(x)
        lbdas_sed = self.target.wavelengths[0]
        sub = np.where((lbdas_sed > parameters.LAMBDA_MIN) & (lbdas_sed < parameters.LAMBDA_MAX))
        self.lambdas_step = min(parameters.LAMBDA_STEP, np.min(lbdas_sed[sub]))

    def simulate_without_atmosphere(self, lambdas):
        """Compute the spectrum of an object and its uncertainties
        after its transmission throught the instrument except the atmosphere.
        The units remains the ones of the Target instance.

        Parameters
        ----------
        lambdas: array_like
            The wavelength array in nm

        Returns
        -------
        spectrum: array_like
            The spectrum in Target units.
        spectrum_err: array_like
            The spectrum uncertainties in Target units.
        """
        self.lambdas = lambdas
        self.lambdas_binwidths = np.gradient(lambdas)
        self.data = self.disperser.transmission(lambdas)
        self.data *= self.telescope.transmission(lambdas)
        self.data *= self.target.sed(lambdas)
        self.err = np.zeros_like(self.data)
        idx = np.where(self.telescope.transmission(lambdas) > 0)[0]
        self.err[idx] = self.telescope.transmission_err(lambdas)[idx] / self.telescope.transmission(lambdas)[idx]
        self.err[idx] *= self.data[idx]
        idx = np.where(self.telescope.transmission(lambdas) <= 0)[0]
        self.err[idx] = 1e6 * np.max(self.err)
        return self.data, self.err

    def simulate(self, A1=1.0, A2=0., ozone=300, pwv=5, aerosols=0.05, reso=0.,
                 D=parameters.DISTANCE2CCD, shift_x=0., B=0.):
        """Simulate the cross spectrum of an object and its uncertainties
        after its transmission throught the instrument and the atmosphere.

        Parameters
        ----------
        A1: float
            Global amplitude of the spectrum (default: 1).
        A2: float
            Relative amplitude of the order 2 spectrum contamination (default: 0).
        ozone: float
            Ozone quantity in Dobson
        pwv: float
            Precipitable Water Vapor quantity in mm
        aerosols: float
            VAOD Vertical Aerosols Optical Depth
        reso: float
            Gaussian kernel size for the convolution (default: 0).
        D: float
            Distance between the CCD and the disperser in mm (default: parameters.DISTANCE2CCD)
        shift_x: float
            Shift in pixels of the order 0 position estimate (default: 0).
        B: float
            Amplitude level for the background (default: 0).

        Returns
        -------
        lambdas: array_like
            The wavelength array in nm used for the interpolation.
        spectrum: array_like
            The spectrum interpolated function in Target units.
        spectrum_err: array_like
            The spectrum uncertainties interpolated function in Target units.

        Examples
        --------
        >>> spectrum, telescope, disperser, target = SimulatorInit("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> atmosphere = AtmosphereGrid(filename="./tests/data/reduc_20170530_134_atmsim.fits")
        >>> sim = SpectrumSimulation(spectrum, atmosphere, telescope, disperser, fast_sim=True)
        >>> lambdas, model, model_err = sim.simulate(A1=1, A2=1, ozone=300, pwv=5, aerosols=0.05, reso=0.,
        ... D=parameters.DISTANCE2CCD, shift_x=0., B=0.)
        >>> sim.plot_spectrum()

        .. doctest::
            :hide:

            >>> assert np.sum(lambdas) > 0
            >>> assert np.sum(model) > 0
            >>> assert np.sum(model) < 1e-10
            >>> assert np.sum(sim.data_order2) > 0
            >>> assert np.sum(sim.data_order2) < 1e-11

        """
        # find lambdas including ADR effect
        new_x0 = [self.x0[0] - shift_x, self.x0[1]]
        self.disperser.D = D
        distance = self.chromatic_psf.get_algebraic_distance_along_dispersion_axis(shift_x=shift_x)
        lambdas = self.disperser.grating_pixel_to_lambda(distance, x0=new_x0, order=1)
        lambdas_order2 = self.disperser.grating_pixel_to_lambda(distance, x0=new_x0, order=2)
        # lambda_ref = self.lambda_ref
        # adr_ra, adr_dec = adr_calib(lambdas, self.adr_params, parameters.OBS_LATITUDE, lambda_ref=lambda_ref)
        # adr_u, _ = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec,
        #                                                        dispersion_axis_angle=self.rotation_angle)
        # distance_order1 = distance - adr_u
        # adr_ra, adr_dec = adr_calib(lambdas_order2, self.adr_params, parameters.OBS_LATITUDE, lambda_ref=lambda_ref)
        # adr_u, _ = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec,
        #                                                        dispersion_axis_angle=self.rotation_angle)
        # distance_order2 = distance - adr_u
        # lambdas = self.disperser.grating_pixel_to_lambda(distance_order1, x0=new_x0, order=1)
        # lambdas_order2 = self.disperser.grating_pixel_to_lambda(distance_order2, x0=new_x0, order=2)
        self.lambdas = lambdas
        self.lambdas_order2 = lambdas_order2
        atmospheric_transmission = self.atmosphere.simulate(ozone, pwv, aerosols)
        if self.fast_sim:
            self.data, self.err = self.simulate_without_atmosphere(lambdas)
            self.data *= A1 * atmospheric_transmission(lambdas)
            self.err *= A1 * atmospheric_transmission(lambdas)
        else:
            def integrand(lbda):
                return self.target.sed(lbda) * self.telescope.transmission(lbda) \
                       * self.disperser.transmission(lbda) * atmospheric_transmission(lbda)

            self.data = np.zeros_like(lambdas)
            self.err = np.zeros_like(lambdas)
            for i in range(len(lambdas) - 1):
                lbdas = np.arange(lambdas[i], lambdas[i + 1], self.lambdas_step)
                self.data[i] = A1 * np.mean(integrand(lbdas))
                # self.data[i] = A1 * quad(integrand, lambdas[i], lambdas[i + 1])[0]
            self.data[-1] = self.data[-2]
            # self.data /= np.gradient(lambdas)
            telescope_transmission = self.telescope.transmission(lambdas)
            idx = telescope_transmission > 0
            self.err[idx] = self.data[idx] * self.telescope.transmission_err(lambdas)[idx] / telescope_transmission[idx]
            idx = telescope_transmission <= 0
            self.err[idx] = 1e6 * np.max(self.err)
        # Now add the systematics
        if reso > 0.1:
            self.data = fftconvolve_gaussian(self.data, reso)
            # self.err = np.sqrt(np.abs(fftconvolve_gaussian(self.err ** 2, reso)))
        if A2 > 0:
            lambdas_binwidths_order2 = np.gradient(lambdas_order2)
            sim_conv = interp1d(lambdas, self.data * lambdas, kind="linear", bounds_error=False, fill_value=(0, 0))
            err_conv = interp1d(lambdas, self.err * lambdas, kind="linear", bounds_error=False, fill_value=(0, 0))
            spectrum_order2 = self.disperser.ratio_order_2over1(lambdas_order2) * sim_conv(lambdas_order2) \
                              * lambdas_binwidths_order2 / self.lambdas_binwidths
            err_order2 = err_conv(lambdas_order2) * lambdas_binwidths_order2 / self.lambdas_binwidths
            self.data = (sim_conv(lambdas) + A2 * spectrum_order2) / lambdas
            self.data_order2 = A2 * spectrum_order2 / lambdas
            self.err = (err_conv(lambdas) + A2 * err_order2) / lambdas
        if B != 0:
            self.data += B / (lambdas * np.gradient(lambdas))
        if np.any(self.err <= 0):
            min_positive = np.min(self.err[self.err > 0])
            self.err[np.isclose(self.err, 0., atol=0.01 * min_positive)] = min_positive
        return self.lambdas, self.data, self.err


class SpectrogramModel(Spectrum):

    def __init__(self, spectrum, atmosphere, telescope, disperser, with_background=True, fast_sim=True,
                 full_image=False):
        """Class to simulate a spectrogram.

        Parameters
        ----------
        spectrum: Spectrum
            Spectrum instance to load main properties before simulation.
        atmosphere: Atmosphere
            Atmosphere or AtmosphereGrid instance to make the atmospheric simulation.
        telescope: TelescopeTransmission
            Telescope transmission.
        disperser: Grating
            Disperser instance.
        with_background: bool, optional
            If True, add the background model to the simulated spectrogram (default: True).
        fast_sim: bool, optional
            If True, perform a fast simulation of the spectrum without integrated the spectrum
            in pixel bins (default: True).
        full_image: bool, optional
            If True, simulate the spectrogram on the full CCD size,
            otherwise only the cropped spectrogram (default: False).

        Examples
        --------
        >>> spectrum, telescope, disperser, target = SimulatorInit("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> atmosphere = Atmosphere(airmass=1.2, pressure=800, temperature=10)
        >>> sim = SpectrogramModel(spectrum, atmosphere, telescope, disperser, with_background=True, fast_sim=True)
        """
        Spectrum.__init__(self)
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        self.disperser = disperser
        self.telescope = telescope
        self.atmosphere = atmosphere
        self.true_lambdas = None
        self.true_spectrum = None
        self.lambdas = None
        self.err = None
        self.model = lambda x, y: np.zeros((x.size, y.size))
        self.psf = load_PSF(psf_type=parameters.PSF_TYPE)
        self.profile_params = None
        self.psf_cube = None
        self.psf_cube_order2 = None
        self.fix_psf_cube = False
        self.fix_atm_sim = False
        self.atmosphere_sim = None
        self.fast_sim = fast_sim
        self.with_background = with_background
        self.full_image = full_image
        if self.full_image:
            self.Nx = parameters.CCD_IMSIZE
            self.Ny = self.spectrogram_Ny  # too long if =parameters.CCD_IMSIZE
            self.r0 = self.x0[0] + 1j * self.spectrogram_y0
        else:
            self.Nx = self.spectrogram_Nx
            self.Ny = self.spectrogram_Ny
            self.r0 = self.spectrogram_x0 + 1j * self.spectrogram_y0
        lbdas_sed = self.target.wavelengths[0]
        sub = np.where((lbdas_sed > parameters.LAMBDA_MIN) & (lbdas_sed < parameters.LAMBDA_MAX))
        self.lambdas_step = min(parameters.LAMBDA_STEP, np.min(lbdas_sed[sub]))
        self.yy, self.xx = np.mgrid[:self.Ny, :self.Nx]
        self.pixels = np.asarray([self.xx, self.yy])

    def set_true_spectrum(self, lambdas, ozone, pwv, aerosols, shift_t=0.):
        atmosphere = self.atmosphere.simulate(ozone, pwv, aerosols)
        spectrum = self.target.sed(lambdas)
        spectrum *= self.disperser.transmission(lambdas - shift_t)
        spectrum *= self.telescope.transmission(lambdas - shift_t)
        spectrum *= atmosphere(lambdas)
        self.true_spectrum = spectrum
        self.true_lambdas = lambdas
        return spectrum

    def simulate_spectrum(self, lambdas, atmosphere, shift_t=0.):
        """
        Simulate the spectrum of the object and return the result in Target units.

        Parameters
        ----------
        lambdas: array_like
            The wavelength array in nm.
        atmosphere: callable
            A callable function of the atmospheric transmission.
        shift_t: float
            Shift of the transmission in nm (default: 0).

        Returns
        -------
        spectrum: array_like
            The spectrum array in ADU/s units.
        spectrum_err: array_like
            The spectrum uncertainty array in ADU/s units.

        """
        spectrum = np.zeros_like(lambdas)
        telescope_transmission = self.telescope.transmission(lambdas - shift_t)
        if self.fast_sim:
            spectrum = self.target.sed(lambdas)
            spectrum *= self.disperser.transmission(lambdas - shift_t)
            spectrum *= telescope_transmission
            spectrum *= atmosphere(lambdas)
            spectrum *= parameters.FLAM_TO_ADURATE * lambdas * np.gradient(lambdas)
        else:
            def integrand(lbda):
                return lbda * self.target.sed(lbda) * self.telescope.transmission(lbda - shift_t) \
                       * self.disperser.transmission(lbda - shift_t) * atmosphere(lbda)

            for i in range(len(lambdas) - 1):
                # spectrum[i] = parameters.FLAM_TO_ADURATE * quad(integrand, lambdas[i], lambdas[i + 1])[0]
                lbdas = np.arange(lambdas[i], lambdas[i + 1], self.lambdas_step)
                spectrum[i] = parameters.FLAM_TO_ADURATE * np.mean(integrand(lbdas)) * (lambdas[i + 1] - lambdas[i])
            spectrum[-1] = spectrum[-2]
        spectrum_err = np.zeros_like(spectrum)
        idx = telescope_transmission > 0
        spectrum_err[idx] = self.telescope.transmission_err(lambdas)[idx] / telescope_transmission[idx] * spectrum[idx]
        # idx = telescope_transmission <= 0: not ready yet to be implemented
        # spectrum_err[idx] = 1e6 * np.max(spectrum_err)
        return spectrum, spectrum_err

    def simulate_psf(self, psf_poly_params):
        profile_params = self.chromatic_psf.from_poly_params_to_profile_params(psf_poly_params, apply_bounds=True)
        self.chromatic_psf.fill_table_with_profile_params(profile_params)
        self.chromatic_psf.table['Dy_disp_axis'] = \
            np.tan(self.rotation_angle * np.pi / 180) * self.chromatic_psf.table['Dx']
        self.chromatic_psf.table['Dy'] = np.copy(self.chromatic_psf.table['y_c']) - self.r0.imag
        self.chromatic_psf.profile_params = self.chromatic_psf.from_table_to_profile_params()
        if False:
            self.chromatic_psf.plot_summary()
        return self.chromatic_psf.profile_params

    def simulate_dispersion(self, D, shift_x, shift_y):
        # Distance in x and y with respect to the true order 0 position at lambda_ref
        Dx = np.arange(self.spectrogram_Nx) - self.spectrogram_x0 - shift_x  # distance in (x,y) spectrogram frame for column x
        Dy_disp_axis = np.tan(self.rotation_angle * np.pi / 180) * Dx  # disp axis height in spectrogram frame for x
        distance = np.sqrt(Dx**2 + Dy_disp_axis**2)

        # Wavelengths using the order 0 shifts (ADR has no impact as it shifts order 0 and order p equally)
        new_x0 = [self.x0[0] + shift_x, self.x0[1] + shift_y]
        self.disperser.D = D
        self.lambdas = self.disperser.grating_pixel_to_lambda(distance, x0=new_x0, order=1)
        distance_order2 = self.disperser.grating_lambda_to_pixel(self.lambdas, x0=new_x0, order=2)
        distance_order2 = distance_order2[distance_order2 + self.r0.real < self.Nx + 100]  # little margin of 100 pixels
        Dx_order2 = distance_order2 * np.cos(self.rotation_angle * np.pi / 180)
        Dy_disp_axis_order2 = distance_order2 * np.sin(self.rotation_angle * np.pi / 180)
        self.lambdas_order2 = self.disperser.grating_pixel_to_lambda(distance_order2, x0=new_x0, order=2)
        self.lambdas_binwidths = np.gradient(self.lambdas)

        # ADR for order 1
        adr_ra, adr_dec = adr_calib(self.lambdas, self.adr_params, parameters.OBS_LATITUDE, lambda_ref=self.lambda_ref)
        adr_x, adr_y = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=0)
        # adr_u, adr_v = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec,
        #                                                            dispersion_axis_angle=self.rotation_angle)

        # Compute lambdas at pixel column x
        # lambdas = self.disperser.grating_pixel_to_lambda(distance - 0*adr_u, new_x0, order=1)
        # Position (not distance) in pixel of wavelength lambda order 1 centroid in the (x,y) spectrogram frame
        dispersion_law = self.r0 + (Dx + shift_x + adr_x) + 1j * (Dy_disp_axis + adr_y + shift_y)

        # ADR for order 2
        adr_ra, adr_dec = adr_calib(self.lambdas_order2, self.adr_params, parameters.OBS_LATITUDE,
                                    lambda_ref=self.lambda_ref)
        adr_x, adr_y = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=0)
        # adr_u, adr_v = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec,
        #                                                            dispersion_axis_angle=self.rotation_angle)

        # Compute lambdas at pixel column x
        # lambdas_order2 = self.disperser.grating_pixel_to_lambda(distance - 0*adr_u, new_x0, order=2)
        # Position (not distance) in pixel of wavelength lambda order 2 centroid in the (x,y) spectrogram frame
        dispersion_law_order2 = self.r0 + (Dx_order2 + shift_x + adr_x) + 1j * (Dy_disp_axis_order2 + adr_y + shift_y)

        # Dx_func = interp1d(lambdas / 2, self.chromatic_psf.table['Dx'], bounds_error=False, fill_value=(0, 0))
        # Dy_mean_func=interp1d(lambdas/2,self.chromatic_psf.table['Dy_disp_axis'],bounds_error=False,fill_value=(0,0))
        # dy_func = interp1d(lambdas / 2, self.chromatic_psf.table['Dy'] - self.chromatic_psf.table['Dy_disp_axis'],
        #                    bounds_error=False, fill_value=(0, 0))
        # dispersion_law = r0 + (self.chromatic_psf.table['Dx'] - shift_x) + 1j * (
        #             self.chromatic_psf.table['Dy'] - shift_y)
        # dispersion_law_order2 = r0 + (Dx_func(lambdas_order2) - shift_x) + 1j * (
        #             Dy_mean_func(lambdas_order2) + dy_func(lambdas_order2) - shift_y)
        # Dx_func = interp1d(lambdas, self.chromatic_psf.table['Dx'], bounds_error=False, fill_value=(0, 0))
        # Dy_mean_func=interp1d(lambdas,self.chromatic_psf.table['Dy_dips_axis'],bounds_error=False, fill_value=(0, 0))

        # dispersion laws from the PSF table
        # dy_func = interp1d(lambdas, self.chromatic_psf.table['Dy'] - self.chromatic_psf.table['Dy_disp_axis'],
        #                    bounds_error=False, fill_value="extrapolate")
        # dispersion_law = r0 + (self.chromatic_psf.table['Dx'] - shift_x) \
        #                  + 1j * (self.chromatic_psf.table['Dy'] - shift_y)
        # dispersion_law_order2 = dispersion_law + 1j * (dy_func(lambdas_order2) - self.chromatic_psf.table['Dy']
        #                                                + self.chromatic_psf.table['Dy_disp_axis'])

        if False:
            from spectractor.tools import from_lambda_to_colormap
            plt.plot(self.chromatic_psf.table['Dx'], self.chromatic_psf.table['Dy_disp_axis'], 'k-', label="mean")
            plt.scatter(dispersion_law.real-self.r0.real, -self.r0.imag + dispersion_law.imag, label="dispersion_law",
                        cmap=from_lambda_to_colormap(self.lambdas), c=self.lambdas)
            plt.scatter(dispersion_law_order2.real-self.r0.real, -self.r0.imag + dispersion_law_order2.imag,
                        label="dispersion_law_order2",
                        cmap=from_lambda_to_colormap(self.lambdas_order2), c=self.lambdas_order2)
            plt.title(f"x0={new_x0}")
            plt.legend()
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()

        return self.lambdas, self.lambdas_order2, dispersion_law, dispersion_law_order2

    # @profile
    def simulate(self, A1=1.0, A2=0., ozone=300, pwv=5, aerosols=0.05, D=parameters.DISTANCE2CCD,
                 shift_x=0., shift_y=0., angle=0., B=1., psf_poly_params=None):
        """

        Parameters
        ----------
        A1: float
            Global amplitude of the spectrum (default: 1).
        A2: float
            Relative amplitude of the order 2 spectrum contamination (default: 0).
        ozone: float
            Ozone quantity in Dobson.
        pwv: float
            Precipitable Water Vapor quantity in mm.
        aerosols: float
            VAOD Vertical Aerosols Optical Depth.
        D: float
            Distance between the CCD and the disperser in mm (default: parameters.DISTANCE2CCD)
        shift_x: float
            Shift in pixels along x axis of the order 0 position estimate (default: 0).
        shift_y: float
            Shift in pixels along y axis of the order 0 position estimate (default: 0).
        angle: float
            Angle of the dispersion axis in degree (default: 0).
        B: float
            Amplitude level for the background (default: 0).
        psf_poly_params: array_like
            Polynomial parameters describing the PSF dependence in wavelength (default: None).

        Returns
        -------
        lambdas: array_like
            The wavelength array in nm used for the interpolation.
        spectrogram: array_like
            The spectrogram array in ADU/s units.
        spectrogram_err: array_like
            The spectrogram uncertainty array in ADU/s units.

        Example
        -------

        >>> spectrum, telescope, disperser, target = SimulatorInit('./tests/data/reduc_20170530_134_spectrum.fits')
        >>> atmosphere = AtmosphereGrid(filename="./tests/data/reduc_20170530_134_atmsim.fits")
        >>> psf_poly_params = spectrum.chromatic_psf.from_table_to_poly_params()
        >>> sim = SpectrogramModel(spectrum, atmosphere, telescope, disperser, with_background=True, fast_sim=True)
        >>> lambdas, model, model_err = sim.simulate(A2=1, angle=-1.5, psf_poly_params=psf_poly_params)
        >>> sim.plot_spectrogram()

        .. doctest::
            :hide:

            >>> assert np.sum(lambdas) > 0
            >>> assert np.sum(model) > 20
        """
        import time
        start = time.time()
        self.rotation_angle = angle
        if self.profile_params is None or not self.fix_psf_cube:
            self.profile_params = self.simulate_psf(psf_poly_params)
        self.my_logger.debug(f'\n\tAfter psf params: {time.time() - start}')
        start = time.time()
        lambdas, lambdas_order2, dispersion_law, dispersion_law_order2 = self.simulate_dispersion(D, shift_x, shift_y)
        self.chromatic_psf.table["Dx"] = dispersion_law.real - self.r0.real
        self.chromatic_psf.table["Dy"] = dispersion_law.imag - self.r0.imag
        self.my_logger.debug(f'\n\tAfter dispersion: {time.time() - start}')
        start = time.time()
        if self.atmosphere_sim is None or not self.fix_atm_sim:
            self.atmosphere_sim = self.atmosphere.simulate(ozone, pwv, aerosols)
        spectrum, spectrum_err = self.simulate_spectrum(lambdas, self.atmosphere_sim)
        self.my_logger.debug(f'\n\tAfter spectrum: {time.time() - start}')
        # Fill the order 1 cube
        nlbda = dispersion_law.size
        if self.psf_cube is None or not self.fix_psf_cube:
            start = time.time()
            self.psf_cube = np.zeros((nlbda, self.Ny, self.Nx))
            for i in range(0, nlbda, 1):
                p = np.array([1, dispersion_law[i].real, dispersion_law[i].imag] + list(self.profile_params[i, 3:]))
                self.psf_cube[i] = self.psf.evaluate(self.pixels, p=p)
            self.my_logger.debug(f'\n\tAfter psf cube: {time.time() - start}')
        start = time.time()
        ima1 = np.zeros((self.Ny, self.Nx))
        ima1_err2 = np.zeros((self.Ny, self.Nx))
        for i in range(0, nlbda, 1):
            # here spectrum[i] is in ADU/s
            ima1 += spectrum[i] * self.psf_cube[i]
            ima1_err2 += (spectrum_err[i] * self.psf_cube[i]) ** 2
        self.my_logger.debug(f'\n\tAfter build ima1: {time.time() - start}')

        # Add order 2
        if A2 > 0.:
            spectrum_order2, spectrum_order2_err = self.disperser.ratio_order_2over1(lambdas_order2) * \
                                                   self.simulate_spectrum(lambdas_order2, self.atmosphere_sim)
            if np.any(np.isnan(spectrum_order2)):
                spectrum_order2[np.isnan(spectrum_order2)] = 0.
            nlbda2 = dispersion_law_order2.size
            if self.psf_cube_order2 is None or not self.fix_psf_cube:
                start = time.time()
                self.psf_cube_order2 = np.zeros((nlbda2, self.Ny, self.Nx))
                # For each A(lambda)=A_x, affect an order 2 PSF with correct position and
                # same PSF as for the order 1 but at the same position
                profile_params_order2 = np.copy(self.profile_params)
                profile_params_order2[:, 0] = 1
                profile_params_order2[:nlbda2, 1] = dispersion_law_order2.real
                profile_params_order2[:nlbda2, 2] = dispersion_law_order2.imag
                distance = np.abs(dispersion_law)
                distance_order2 = np.abs(dispersion_law_order2)
                for k in range(3, self.profile_params.shape[1]):
                    profile_params_order2[:nlbda2, k] = interp1d(distance, profile_params_order2[:, k], kind="cubic",
                                                                 fill_value="extrapolate")(distance_order2)
                for i in range(0, nlbda2, 1):
                    self.psf_cube_order2[i] = self.psf.evaluate(self.pixels, p=profile_params_order2[i, :])
                self.my_logger.debug(f'\n\tAfter psf cube order 2: {time.time() - start}')
            start = time.time()
            ima2 = np.zeros_like(ima1)
            ima2_err2 = np.zeros_like(ima1)
            for i in range(0, nlbda2, 1):
                # here spectrum[i] is in ADU/s
                ima2 += spectrum_order2[i] * self.psf_cube_order2[i]
                ima2_err2 += (spectrum_order2_err[i] * self.psf_cube_order2[i]) ** 2
            # self.data is in ADU/s units here
            self.data = A1 * (ima1 + A2 * ima2)
            self.err = np.sqrt(A1*A1*(ima1_err2 + A2*A2*ima2_err2))
            self.my_logger.debug(f'\n\tAfter build ima2: {time.time() - start}')
        else:
            self.data = A1 * ima1
            self.err = A1 * np.sqrt(ima1_err2)
        start = time.time()
        if self.with_background:
            self.data += B * self.spectrogram_bgd
        self.my_logger.debug(f'\n\tAfter bgd: {time.time() - start}')
        if False:
            fig, ax = plt.subplots(2, 1, sharex="all", figsize=(12, 9))
            im = ax[0].imshow(self.data, origin='lower')
            plt.colorbar(im, ax=ax[0], label=self.units)
            ax[0].set_title('Model')
            im = ax[1].imshow(self.err, origin='lower')
            plt.colorbar(im, ax=ax[1], label=self.units)
            ax[1].set_title('Err')
            ax[1].set_xlabel('X [pixels]')
            ax[0].set_ylabel('Y [pixels]')
            ax[1].set_ylabel('Y [pixels]')
            ax[0].grid()
            ax[1].grid()
            # if self.with_background:
            #     ax[2].plot(np.sum(self.data, axis=0), label="model")
            # else:
            #     ax[2].plot(np.sum(self.data + self.spectrogram_bgd, axis=0), label="model")
            # ax[2].plot(np.sum(self.spectrogram, axis=0), label="data")
            # ax[2].grid()
            # ax[2].legend()
            fig.tight_layout()
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()

        return self.lambdas, self.data, self.err


class SpectrumSimGrid:
    """ SpectrumSim class used to store information and methods
    relative to spectrum simulation.
    NEED TO ADAPT THIS CLASS TO THE FULL SIMULATION WITH SYSTEMATICS.
    """

    # ---------------------------------------------------------------------------
    def __init__(self, spectrum, atmgrid, telescope, disperser, target, filename=""):
        """
        Args:
            filename (:obj:`str`): path to the image
        """
        self.my_logger = set_logger(self.__class__.__name__)

        self.spectrum = spectrum
        self.header = spectrum.header
        self.disperser = disperser
        self.target = target
        self.telescope = telescope

        self.atmgrid = atmgrid
        self.lambdas = self.atmgrid.atmgrid[0, self.atmgrid.index_atm_data:]
        self.lambdas_binwidths = np.gradient(self.lambdas)
        self.spectragrid = None

        self.filename = ""
        if filename != "":
            self.filename = filename
            self.spectrum.load_spectrum(filename)

    def compute(self):
        sim = SpectrumSimulation(self.spectrum, self.atmgrid.atmgrid, self.telescope, self.disperser)
        # product of all sed and transmissions except atmosphere
        all_transm, all_transm_err = sim.simulate_without_atmosphere(self.lambdas)
        # copy atmospheric grid parameters into spectra grid
        self.spectragrid = np.zeros_like(self.atmgrid.atmgrid)
        self.spectragrid[0, self.atmgrid.index_atm_data:] = self.lambdas
        self.spectragrid[:, self.atmgrid.index_atm_count:self.atmgrid.index_atm_data] = \
            self.atmgrid.atmgrid[:, self.atmgrid.index_atm_count:self.atmgrid.index_atm_data]
        # Is broadcasting working OK ?
        self.spectragrid[1:, self.atmgrid.index_atm_data:] = self.atmgrid.atmgrid[1:, self.atmgrid.index_atm_data:]
        self.spectragrid[1:, self.atmgrid.index_atm_data:] *= all_transm
        return self.spectragrid

    def plot_spectra(self):
        plt.figure()
        counts = self.spectragrid[1:, self.atmgrid.index_atm_count]
        for count in counts:
            plt.plot(self.lambdas, self.spectragrid[int(count), self.atmgrid.index_atm_data:])
        plt.grid()
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel("Flux [ADU/s]")
        plt.title("Spectra for Atmospheric variations")
        if parameters.DISPLAY:
            plt.show()
        else:
            plt.close('all')

    def plot_spectra_img(self):
        plt.figure()
        img = plt.imshow(self.spectragrid[1:, self.atmgrid.index_atm_data:], origin='lower', cmap='jet')
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel("Simulation number")
        plt.title("Spectra for atmospheric variations")
        cbar = plt.colorbar(img)
        cbar.set_label("Flux [ADU/s]")
        plt.grid(True)
        if parameters.DISPLAY:
            plt.show()
        else:
            plt.close('all')

    def save_spectra(self, filename):

        if filename != "":
            self.filename = filename

        if self.filename == "":
            return
        else:

            hdu = fits.PrimaryHDU(self.spectragrid, header=self.header)
            hdu.writeto(self.filename, overwrite=True)
            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info(f'\n\tSPECTRA.save atm-file={self.filename}')


def SimulatorInit(filename, fast_load=False):
    """ SimulatorInit
    Main function to evaluate several spectra
    A grid of spectra will be produced for a given target, airmass and pressure
    """
    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SIMULATOR initialisation')
    # Load data spectrum
    spectrum = Spectrum(filename, fast_load=fast_load)

    # TELESCOPE TRANSMISSION
    # ------------------------
    telescope = TelescopeTransmission(spectrum.filter_label)

    # DISPERSER TRANSMISSION
    # ------------------------
    if not isinstance(spectrum.disperser, str):
        disperser = spectrum.disperser
    else:
        disperser = Hologram(spectrum.disperser)

    # STAR SPECTRUM
    # ------------------------
    if not isinstance(spectrum.target, str):
        target = spectrum.target
    else:
        target = Target(spectrum.target)

    return spectrum, telescope, disperser, target


def SpectrumSimulatorCore(spectrum, telescope, disperser, airmass=1.0, pressure=800, temperature=10,
                          pwv=5, ozone=300, aerosols=0.05, A1=1.0, A2=0., reso=0, D=parameters.DISTANCE2CCD, shift=0.):
    """ SimulatorCore
    Main function to evaluate several spectra
    A grid of spectra will be produced for a given target, airmass and pressure
    """
    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SPECTRUMSIMULATOR core program')
    # SIMULATE ATMOSPHERE
    # -------------------
    atmosphere = Atmosphere(airmass, pressure, temperature)

    # SPECTRUM SIMULATION
    # --------------------
    spectrum_simulation = SpectrumSimulation(spectrum, atmosphere, telescope, disperser)
    spectrum_simulation.simulate(A1, A2, ozone, pwv, aerosols, reso, D, shift)
    if parameters.DEBUG:
        spectrum_simulation.plot_spectrum(force_lines=True)
    return spectrum_simulation


def SpectrogramSimulatorCore(spectrum, telescope, disperser, airmass=1.0, pressure=800, temperature=10,
                             pwv=5, ozone=300, aerosols=0.05, A1=1.0, A2=0.,
                             D=parameters.DISTANCE2CCD, shift_x=0., shift_y=0., shift_t=0., angle=0.,
                             B=1., psf_poly_params=None, with_background=True, fast_sim=False, full_image=False):
    """ SimulatorCore
    Main function to evaluate several spectra
    A grid of spectra will be produced for a given target, airmass and pressure
    """
    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SPECTROGRAMSIMULATOR core program')
    # SIMULATE ATMOSPHERE
    # -------------------

    atmosphere = Atmosphere(airmass, pressure, temperature)
    spectrum.rotation_angle = angle

    # SPECTRUM SIMULATION
    # --------------------
    spectrogram_simulation = SpectrogramModel(spectrum, atmosphere, telescope, disperser,
                                              with_background=with_background, fast_sim=fast_sim, full_image=full_image)
    spectrogram_simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, angle, B, psf_poly_params)
    return spectrogram_simulation


def SpectrumSimulatorSimGrid(filename, outputdir, pwv_grid=[0, 10, 5], ozone_grid=[100, 700, 7],
                             aerosol_grid=[0, 0.1, 5]):
    """ SimulatorSimGrid
    Main function to evaluate several spectra
    A grid of spectra will be produced for a given target, airmass and pressure

    """
    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SIMULATORGRID')
    # Initialisation
    spectrum, telescope, disperser, target = SimulatorInit(filename)
    # Set output path
    ensure_dir(outputdir)
    # extract the basename : simimar as os.path.basename(file)
    base_filename = filename.split('/')[-1]
    output_filename = os.path.join(outputdir, base_filename.replace('spectrum', 'spectrasim'))
    output_atmfilename = os.path.join(outputdir, base_filename.replace('spectrum', 'atmsim'))

    # SIMULATE ATMOSPHERE GRID
    # ------------------------
    airmass = spectrum.header['AIRMASS']
    pressure = spectrum.header['OUTPRESS']
    temperature = spectrum.header['OUTTEMP']
    atm = AtmosphereGrid(filename, airmass=airmass, pressure=pressure, temperature=temperature)
    atm.set_grid(pwv_grid, ozone_grid, aerosol_grid)

    # test if file already exists
    if os.path.exists(output_atmfilename) and os.path.getsize(output_atmfilename) > 20000:
        filesize = os.path.getsize(output_atmfilename)
        infostring = " atmospheric simulation file %s of size %d already exists, thus load_image it ..." % (
            output_atmfilename, filesize)
        my_logger.info(infostring)
        atm.load_file(output_atmfilename)
    else:
        my_logger.info(f"\n\tFile {output_atmfilename} does not exist yet. Compute it...")
        atm.compute()
        atm.save_file(filename=output_atmfilename)
        # libradtran.clean_simulation_directory()
    if parameters.DEBUG:
        infostring = '\n\t ========= Atmospheric simulation :  ==============='
        my_logger.info(infostring)
        atm.plot_transmission()  # plot all atm transp profiles
        atm.plot_transmission_image()  # plot 2D image summary of atm simulations

    # SPECTRA-GRID
    # -------------
    # in any case we re-calculate the spectra in case of change of spectrum function
    spectra = SpectrumSimGrid(spectrum, atm, telescope, disperser, target)
    spectra.compute()
    spectra.save_spectra(output_filename)
    if parameters.DEBUG:
        spectra.plot_spectra()
        spectra.plot_spectra_img()
    # ---------------------------------------------------------------------------


def SpectrumSimulator(filename, outputdir="", pwv=5, ozone=300, aerosols=0.05, A1=1., A2=0.,
                      reso=None, D=parameters.DISTANCE2CCD, shift=0.):
    """ Simulator
    Main function to evaluate several spectra
    A grid of spectra will be produced for a given target, airmass and pressure
    """
    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTORSIM')
    # Initialisation
    spectrum, telescope, disperser, target = SimulatorInit(filename)

    # SIMULATE SPECTRUM
    # -------------------
    airmass = spectrum.header['AIRMASS']
    pressure = spectrum.header['OUTPRESS']
    temperature = spectrum.header['OUTTEMP']

    spectrum_simulation = SpectrumSimulatorCore(spectrum, telescope, disperser, airmass, pressure,
                                                temperature, pwv, ozone, aerosols, A1=A1, A2=A2, reso=reso, D=D,
                                                shift=shift)

    # Save the spectrum
    spectrum_simulation.header['OZONE_T'] = ozone
    spectrum_simulation.header['PWV_T'] = pwv
    spectrum_simulation.header['VAOD_T'] = aerosols
    spectrum_simulation.header['A1_T'] = A1
    spectrum_simulation.header['A2_T'] = A2
    spectrum_simulation.header['RESO_T'] = reso
    spectrum_simulation.header['D2CCD_T'] = D
    spectrum_simulation.header['X0_T'] = shift
    output_filename = filename.replace('spectrum', 'sim')
    if outputdir != "":
        base_filename = filename.split('/')[-1]
        output_filename = os.path.join(outputdir, base_filename.replace('spectrum', 'sim'))
    spectrum_simulation.save_spectrum(output_filename, overwrite=True)

    return spectrum_simulation


def SpectrogramSimulator(filename, outputdir="", pwv=5, ozone=300, aerosols=0.05, A1=1., A2=0.,
                         D=parameters.DISTANCE2CCD, shift_x=0., shift_y=0., shift_t=0., angle=0., B=1.,
                         psf_poly_params=None):
    """ Simulator
    Main function to evaluate several spectra
    A grid of spectra will be produced for a given target, airmass and pressure
    """
    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTORSIM')
    # Initialisation
    spectrum, telescope, disperser, target = SimulatorInit(filename)

    if psf_poly_params is None:
        my_logger.info('\n\tUse PSF parameters from _table.csv file.')
        psf_poly_params = spectrum.chromatic_psf.from_table_to_poly_params()

    # SIMULATE SPECTRUM
    # -------------------
    airmass = spectrum.header['AIRMASS']
    pressure = spectrum.header['OUTPRESS']
    temperature = spectrum.header['OUTTEMP']

    spectrogram_simulation = SpectrogramSimulatorCore(spectrum, telescope, disperser, airmass, pressure,
                                                      temperature, pwv, ozone, aerosols,
                                                      D=D, shift_x=shift_x,
                                                      shift_y=shift_y, shift_t=shift_t, angle=angle, B=B,
                                                      psf_poly_params=psf_poly_params)

    # Save the spectrum
    spectrogram_simulation.header['OZONE_T'] = ozone
    spectrogram_simulation.header['PWV_T'] = pwv
    spectrogram_simulation.header['VAOD_T'] = aerosols
    spectrogram_simulation.header['A1_T'] = A1
    spectrogram_simulation.header['A2_T'] = A2
    spectrogram_simulation.header['D2CCD_T'] = D
    spectrogram_simulation.header['X0_T'] = shift_x
    spectrogram_simulation.header['Y0_T'] = shift_y
    spectrogram_simulation.header['TSHIFT_T'] = shift_t
    spectrogram_simulation.header['ROTANGLE'] = angle
    output_filename = filename.replace('spectrum', 'sim')
    if outputdir != "":
        base_filename = filename.split('/')[-1]
        output_filename = os.path.join(outputdir, base_filename.replace('spectrum', 'sim'))
    # spectrogram_simulation.save_spectrum(output_filename, overwrite=True)

    return spectrogram_simulation


if __name__ == "__main__":
    import doctest

    doctest.testmod()

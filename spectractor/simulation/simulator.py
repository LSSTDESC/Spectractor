import copy
import numpy as np

from scipy.interpolate import interp1d

from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.dispersers import Hologram
from spectractor.extractor.targets import Target
from spectractor.extractor.psf import load_PSF
from spectractor.tools import fftconvolve_gaussian
from spectractor.config import set_logger
from spectractor.simulation.throughput import TelescopeTransmission
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
import spectractor.parameters as parameters


class SpectrumSimulation(Spectrum):

    def __init__(self, spectrum, target=None, disperser=None, throughput=None, atmosphere=None, fast_sim=True):
        """Class to simulate cross spectrum.

        Parameters
        ----------
        spectrum: Spectrum
            Spectrum instance.
        target: Target
            Target instance.
        throughput: TelescopeTransmission, optional
            Telescope throughput (default: None).
        disperser: Grating, optional
            Disperser instance (default: None).
        atmosphere: Atmosphere, optional
            Atmosphere or AtmosphereGrid instance to make the atmospheric simulation (default: None).
        fast_sim: bool, optional
            If True, do a fast simulation without integrating within the wavelength bins (default: True).

        Examples
        --------
        >>> spectrum = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> atmosphere = Atmosphere(airmass=1.2, pressure=800, temperature=10)
        >>> sim = SpectrumSimulation(spectrum, atmosphere=atmosphere, fast_sim=True)

        """
        Spectrum.__init__(self)
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        self.my_logger = set_logger(self.__class__.__name__)
        # if parameters.CCD_REBIN > 1:
        #     self.chromatic_psf.table['Dx'] *= parameters.CCD_REBIN
        #     apply_rebinning_to_parameters(reverse=True)
        if target is not None:
            self.target = target
        if disperser is not None:
            self.disperser = disperser
        if throughput is not None:
            self.throughput = throughput
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
        The units remain the ones of the Target instance.

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
        self.data *= self.throughput.transmission(lambdas)
        self.data *= self.target.sed(lambdas)
        self.err = np.zeros_like(self.data)
        idx = np.where(self.throughput.transmission(lambdas) > 0)[0]
        self.err[idx] = self.throughput.transmission_err(lambdas)[idx] / self.throughput.transmission(lambdas)[idx]
        self.err[idx] *= self.data[idx]
        idx = np.where(self.throughput.transmission(lambdas) <= 0)[0]
        self.err[idx] = 1e6 * np.max(self.err)
        return self.data, self.err

    def simulate(self, A1=1.0, A2=0., aerosols=0.05, angstrom_exponent=None, ozone=300, pwv=5, reso=0.,
                 D=parameters.DISTANCE2CCD, shift_x=0., B=0.):
        """Simulate the cross spectrum of an object and its uncertainties
        after its transmission throught the instrument and the atmosphere.

        Parameters
        ----------
        A1: float
            Global amplitude of the spectrum (default: 1).
        A2: float
            Relative amplitude of the order 2 spectrum contamination (default: 0).
        aerosols: float
            VAOD Vertical Aerosols Optical Depth
        angstrom_exponent: float, optional
            Angstrom exponent for aerosols. If negative or None, default aerosol model from Libradtran is used.
            If value is 0.0192, the atmospheric transmission is very close to the case with angstrom_exponent=None (default: None).
        ozone: float
            Ozone quantity in Dobson
        pwv: float
            Precipitable Water Vapor quantity in mm
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
        >>> spectrum = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> atmosphere = AtmosphereGrid(atmgrid_filename="./tests/data/reduc_20170530_134_atmsim.fits")
        >>> sim = SpectrumSimulation(spectrum, atmosphere=atmosphere, fast_sim=True)
        >>> lambdas, model, model_err = sim.simulate(A1=1, A2=1, ozone=300, pwv=5, aerosols=0.05, reso=0.,
        ... D=parameters.DISTANCE2CCD, shift_x=0., B=0.)
        >>> sim.plot_spectrum()

        .. doctest::
            :hide:

            >>> assert np.sum(lambdas) > 0
            >>> assert np.sum(model) > 0
            >>> assert np.sum(model) < 1e-10
            >>> assert np.sum(sim.data_next_order) > 0
            >>> assert np.sum(sim.data_next_order) < 1e-11

        """
        # find lambdas including ADR effect
        # must add ADR to get perfect result on atmospheric fit in full chain test with SpectrumSimulation()
        lambdas = self.compute_lambdas_in_spectrogram(D, shift_x=shift_x, shift_y=0, angle=self.rotation_angle,
                                                      order=1, with_adr=True, niter=5)
        lambdas_order2 = self.compute_lambdas_in_spectrogram(D, shift_x=shift_x, shift_y=0, angle=self.rotation_angle,
                                                             order=2, with_adr=True, niter=5)
        self.lambdas = lambdas
        if self.atmosphere is not None:
            self.atmosphere.set_lambda_range(lambdas)
            atmospheric_transmission = self.atmosphere.simulate(aerosols=aerosols, ozone=ozone, pwv=pwv, angstrom_exponent=angstrom_exponent)
        else:
            def atmospheric_transmission(lbda):
                return 1
        if self.fast_sim:
            self.data, self.err = self.simulate_without_atmosphere(lambdas)
            self.data *= A1 * atmospheric_transmission(lambdas)
            self.err *= A1 * atmospheric_transmission(lambdas)
        else:
            def integrand(lbda):
                return self.target.sed(lbda) * self.throughput.transmission(lbda) \
                       * self.disperser.transmission(lbda) * atmospheric_transmission(lbda)

            self.data = np.zeros_like(lambdas)
            self.err = np.zeros_like(lambdas)
            for i in range(len(lambdas) - 1):
                lbdas = np.arange(lambdas[i], lambdas[i + 1], self.lambdas_step)
                self.data[i] = A1 * np.mean(integrand(lbdas))
                # self.data[i] = A1 * quad(integrand, lambdas[i], lambdas[i + 1])[0]
            self.data[-1] = self.data[-2]
            # self.data /= np.gradient(lambdas)
            telescope_transmission = self.throughput.transmission(lambdas)
            idx = telescope_transmission > 0
            self.err[idx] = self.data[idx] * self.throughput.transmission_err(lambdas)[idx] / telescope_transmission[idx]
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
        if np.any(self.err <= 0) and not np.all(self.err<=0):
            min_positive = np.min(self.err[self.err > 0])
            self.err[np.isclose(self.err, 0., atol=0.01 * min_positive)] = min_positive
        # Save the truth parameters
        self.header['OZONE_T'] = ozone
        self.header['PWV_T'] = pwv
        self.header['VAOD_T'] = aerosols
        self.header['A1_T'] = A1
        self.header['A2_T'] = A2
        self.header['RESO_T'] = reso
        self.header['D2CCD_T'] = D
        self.header['X0_T'] = shift_x
        return self.lambdas, self.data, self.err


class SpectrogramModel(Spectrum):

    def __init__(self, spectrum, target=None, disperser=None, throughput=None, atmosphere=None, with_background=True,
                 fast_sim=True, full_image=False, with_adr=True):
        """Class to simulate a spectrogram.

        Parameters
        ----------
        spectrum: Spectrum
            Spectrum instance to load main properties before simulation.
        target: Target
            Target instance.
        throughput: TelescopeTransmission, optional
            Telescope throughput (default: None).
        disperser: Grating, optional
            Disperser instance (default: None).
        atmosphere: Atmosphere, optional
            Atmosphere or AtmosphereGrid instance to make the atmospheric simulation (default: None).
        with_background: bool, optional
            If True, add the background model to the simulated spectrogram (default: True).
        fast_sim: bool, optional
            If True, perform a fast simulation of the spectrum without integrated the spectrum
            in pixel bins (default: True).
        full_image: bool, optional
            If True, simulate the spectrogram on the full CCD size,
            otherwise only the cropped spectrogram (default: False).
        with_adr: bool, optional
            If True, simulate the spectrogram with ADR effect (default: True).

        Examples
        --------
        >>> spectrum = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> atmosphere = Atmosphere(airmass=1.2, pressure=800, temperature=10)
        >>> sim = SpectrogramModel(spectrum, atmosphere=atmosphere, with_background=True, fast_sim=True)
        """
        Spectrum.__init__(self)
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        if target is not None:
            self.target = target
        if disperser is not None:
            self.disperser = disperser
        if throughput is not None:
            self.throughput = throughput
        self.atmosphere = atmosphere
        self.true_lambdas = None
        self.true_spectrum = None
        self.lambdas = None
        self.model = lambda x, y: np.zeros((x.size, y.size))
        self.psf = load_PSF(psf_type=parameters.PSF_TYPE)
        self.profile_params = None
        self.psf_cube = None
        self.psf_cube_order2 = None
        self.psf_cube_order3 = None
        self.psf_cube_masked = None
        self.fix_psf_cube = False
        self.fix_atm_sim = False
        self.atmosphere_sim = None
        self.fast_sim = fast_sim
        self.with_background = with_background
        self.full_image = full_image
        self.with_adr = with_adr
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

    def set_true_spectrum(self, lambdas, aerosols, ozone, pwv, shift_t=0.):
        atmosphere = self.atmosphere.simulate(aerosols=aerosols, ozone=ozone, pwv=pwv)
        spectrum = self.target.sed(lambdas)
        spectrum *= self.disperser.transmission(lambdas - shift_t)
        spectrum *= self.throughput.transmission(lambdas - shift_t)
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
        telescope_transmission = self.throughput.transmission(lambdas - shift_t)
        if self.fast_sim:
            spectrum = self.target.sed(lambdas)
            spectrum *= self.disperser.transmission(lambdas - shift_t)
            spectrum *= telescope_transmission
            spectrum *= atmosphere(lambdas)
            spectrum *= parameters.FLAM_TO_ADURATE * lambdas * np.gradient(lambdas)
        else:
            def integrand(lbda):
                return lbda * self.target.sed(lbda) * self.throughput.transmission(lbda - shift_t) \
                       * self.disperser.transmission(lbda - shift_t) * atmosphere(lbda)

            for i in range(len(lambdas) - 1):
                # spectrum[i] = parameters.FLAM_TO_ADURATE * quad(integrand, lambdas[i], lambdas[i + 1])[0]
                lbdas = np.arange(lambdas[i], lambdas[i + 1], self.lambdas_step)
                spectrum[i] = parameters.FLAM_TO_ADURATE * np.mean(integrand(lbdas)) * (lambdas[i + 1] - lambdas[i])
            spectrum[-1] = spectrum[-2]
        spectrum_err = np.zeros_like(spectrum)
        idx = telescope_transmission > 0
        spectrum_err[idx] = self.throughput.transmission_err(lambdas)[idx] / telescope_transmission[idx] * spectrum[idx]
        # idx = telescope_transmission <= 0: not ready yet to be implemented
        # spectrum_err[idx] = 1e6 * np.max(spectrum_err)
        return spectrum, spectrum_err

    # @profile
    def simulate(self, A1=1.0, A2=0., aerosols=0.05, angstrom_exponent=None, ozone=300, pwv=5, D=parameters.DISTANCE2CCD,
                 shift_x=0., shift_y=0., angle=0., B=1., psf_poly_params=None):
        """

        Parameters
        ----------
        A1: float
            Global amplitude of the spectrum (default: 1).
        A2: float
            Relative amplitude of the order 2 spectrum contamination (default: 0).
        aerosols: float
            VAOD Vertical Aerosols Optical Depth.
        angstrom_exponent: float, optional
            Angstrom exponent for aerosols. If negative or None, default aerosol model from Libradtran is used.
            If value is 0.0192, the atmospheric transmission is very close to the case with angstrom_exponent=None (default: None).
        ozone: float
            Ozone quantity in Dobson.
        pwv: float
            Precipitable Water Vapor quantity in mm.
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
        >>> spec = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")
        >>> spec.disperser.ratio_ratio_order_3over2 = lambda lbda: 0.1
        >>> psf_poly_params = spec.chromatic_psf.from_table_to_poly_params()
        >>> atmosphere = Atmosphere(airmass=1.2, pressure=800, temperature=10)
        >>> sim = SpectrogramModel(spec, atmosphere=atmosphere, with_background=True, fast_sim=True)
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
        self.lambdas = self.compute_lambdas_in_spectrogram(D, shift_x, shift_y, angle, niter=5, with_adr=True,
                                                           order=self.order)
        dispersion_law = self.compute_dispersion_in_spectrogram(self.lambdas, shift_x, shift_y, angle,
                                                                niter=5, with_adr=True, order=self.order)
        self.lambdas_binwidths = np.gradient(self.lambdas)
        self.my_logger.debug(f'\n\tAfter dispersion: {time.time() - start}')
        start = time.time()
        if len(psf_poly_params) % 2 != 0:
            raise ValueError(f"Argument psf_poly_params must be even size, to be split in parameters"
                             f"for order 1 and order 2 spectrograms. Got {len(psf_poly_params)=}.")
        psf_poly_params_order1 = psf_poly_params[:len(psf_poly_params)//2]
        psf_poly_params_order2 = psf_poly_params[len(psf_poly_params)//2:]
        if self.profile_params is None or not self.fix_psf_cube:
            self.profile_params = self.chromatic_psf.update(psf_poly_params_order1, x0=self.r0.real + shift_x,
                                                            y0=self.r0.imag + shift_y, angle=angle, plot=False)
            self.profile_params[:, 1] = dispersion_law.real + self.r0.real
            self.profile_params[:, 2] += dispersion_law.imag
        self.chromatic_psf.table["Dx"] = self.profile_params[:, 1] - self.r0.real
        self.chromatic_psf.table["Dy"] = self.profile_params[:, 2] - self.r0.imag
        self.my_logger.debug(f'\n\tAfter psf params: {time.time() - start}')
        start = time.time()
        if self.atmosphere_sim is None or not self.fix_atm_sim:
            self.atmosphere.set_lambda_range(self.lambdas)
            self.atmosphere_sim = self.atmosphere.simulate(aerosols=aerosols, ozone=ozone, pwv=pwv,
                                                           angstrom_exponent=angstrom_exponent)
        spectrum, spectrum_err = self.simulate_spectrum(self.lambdas, self.atmosphere_sim)
        self.my_logger.debug(f'\n\tAfter spectrum: {time.time() - start}')
        # Fill the order 1 cube
        nlbda = dispersion_law.size
        if self.psf_cube is None or not self.fix_psf_cube:
            start = time.time()
            self.psf_cube = self.chromatic_psf.build_psf_cube(self.pixels, self.profile_params,
                                                              fwhmx_clip=3 * parameters.PSF_FWHM_CLIP,
                                                              fwhmy_clip=parameters.PSF_FWHM_CLIP, dtype="float32",
                                                              mask=self.psf_cube_masked)
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
        if A2 > 0. and self.disperser.ratio_order_2over1 is not None:
            spectrum_order2 = self.disperser.ratio_order_2over1(self.lambdas) * spectrum
            spectrum_order2_err = self.disperser.ratio_order_2over1(self.lambdas) * spectrum_err
            if np.any(np.isnan(spectrum_order2)):
                spectrum_order2[np.isnan(spectrum_order2)] = 0.
            dispersion_law_order2 = self.compute_dispersion_in_spectrogram(self.lambdas, shift_x, shift_y, angle,
                                                                           niter=5, with_adr=True,
                                                                           order=self.order + np.sign(self.order))
            nlbda2 = dispersion_law_order2.size
            if self.psf_cube_order2 is None or not self.fix_psf_cube:
                start = time.time()
                profile_params_order2 = self.chromatic_psf.from_poly_params_to_profile_params(psf_poly_params_order2,
                                                                                              apply_bounds=True)
                profile_params_order2[:, 0] = 1
                profile_params_order2[:nlbda2, 1] = dispersion_law_order2.real + self.r0.real
                profile_params_order2[:nlbda2, 2] += dispersion_law_order2.imag
                self.psf_cube_order2 = self.chromatic_psf.build_psf_cube(self.pixels, profile_params_order2,
                                                                         fwhmx_clip=3 * parameters.PSF_FWHM_CLIP,
                                                                         fwhmy_clip=parameters.PSF_FWHM_CLIP,
                                                                         dtype="float32", mask=None)
                self.my_logger.debug(f'\n\tAfter psf cube order 2: {time.time() - start}')
            start = time.time()
            ima2 = np.zeros_like(ima1)
            ima2_err2 = np.zeros_like(ima1)
            for i in range(0, nlbda2, 1):
                # here spectrum[i] is in ADU/s
                ima2 += spectrum_order2[i] * self.psf_cube_order2[i]
                ima2_err2 += (spectrum_order2_err[i] * self.psf_cube_order2[i]) ** 2
            if self.disperser.ratio_order_3over2 is not None:
                spectrum_order3 = self.disperser.ratio_order_3over2(self.lambdas) * spectrum_order2
                spectrum_order3_err = self.disperser.ratio_order_3over2(self.lambdas) * spectrum_order2_err
                if np.any(np.isnan(spectrum_order3)):
                    spectrum_order3[np.isnan(spectrum_order3)] = 0.
                dispersion_law_order3 = self.compute_dispersion_in_spectrogram(self.lambdas, shift_x, shift_y, angle,
                                                                               niter=5, with_adr=True,
                                                                               order=self.order + 2*np.sign(self.order))
                nlbda3 = dispersion_law_order3.size
                if self.psf_cube_order3 is None or not self.fix_psf_cube:
                    start3 = time.time()
                    profile_params_order3 = self.chromatic_psf.from_poly_params_to_profile_params(psf_poly_params_order2, apply_bounds=True)
                    profile_params_order3[:, 0] = 1
                    profile_params_order3[:nlbda3, 1] = dispersion_law_order3.real + self.r0.real
                    profile_params_order3[:nlbda3, 2] += dispersion_law_order3.imag
                    self.psf_cube_order3 = self.chromatic_psf.build_psf_cube(self.pixels, profile_params_order3,
                                                                             fwhmx_clip=3 * parameters.PSF_FWHM_CLIP,
                                                                             fwhmy_clip=parameters.PSF_FWHM_CLIP,
                                                                             dtype="float32", mask=None)
                    self.my_logger.debug(f'\n\tAfter psf cube order 3: {time.time() - start3}')
                for i in range(0, nlbda2, 1):
                    # here spectrum[i] is in ADU/s
                    ima2 += spectrum_order3[i] * self.psf_cube_order3[i]
                    ima2_err2 += (spectrum_order3_err[i] * self.psf_cube_order3[i]) ** 2

            # Assemble all diffraction orders in simulation
            # self.data is in ADU/s units here
            self.spectrogram = A1 * (ima1 + A2 * ima2)
            self.spectrogram_err = np.sqrt(A1*A1*(ima1_err2 + A2*A2*ima2_err2))
            self.my_logger.debug(f'\n\tAfter build ima2: {time.time() - start}')
        else:
            self.spectrogram = A1 * ima1
            self.spectrogram_err = A1 * np.sqrt(ima1_err2)
        start = time.time()
        if self.with_background:
            self.spectrogram += B * self.spectrogram_bgd
        self.my_logger.debug(f'\n\tAfter bgd: {time.time() - start}')
        # Save the simulation parameters
        self.header['OZONE_T'] = ozone
        self.header['PWV_T'] = pwv
        self.header['VAOD_T'] = aerosols
        self.header['A1_T'] = A1
        self.header['A2_T'] = A2
        self.header['D2CCD_T'] = D
        self.header['X0_T'] = shift_x
        self.header['Y0_T'] = shift_y
        self.header['ROTANGLE'] = angle

        return self.lambdas, self.spectrogram, self.spectrogram_err


if __name__ == "__main__":
    import doctest

    doctest.testmod()

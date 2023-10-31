import copy
import numpy as np

from scipy.interpolate import interp1d

from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.targets import Target
from spectractor.extractor.psf import load_PSF
from spectractor.tools import fftconvolve_gaussian
from spectractor.config import set_logger
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
import spectractor.parameters as parameters


class SpectrumSimulation(Spectrum):

    def __init__(self, spectrum, target=None, disperser=None, throughput=None, atmosphere=None, fast_sim=True, with_adr=True):
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
        with_adr: bool, optional
            If True, use ADR model to build lambda array (default: False).

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
        self.with_adr = with_adr
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
                                                      order=1, with_adr=self.with_adr, niter=5)
        lambdas_order2 = self.compute_lambdas_in_spectrogram(D, shift_x=shift_x, shift_y=0, angle=self.rotation_angle,
                                                             order=2, with_adr=self.with_adr, niter=5)
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
        self.header['LOG10A_T'] = angstrom_exponent
        self.header['PWV_T'] = pwv
        self.header['VAOD_T'] = aerosols
        self.header['A1_T'] = A1
        self.header['A2_T'] = A2
        self.header['RESO_T'] = reso
        self.header['D2CCD_T'] = D
        self.header['X0_T'] = shift_x
        return self.lambdas, self.data, self.err


class SpectrogramModel(Spectrum):

    def __init__(self, spectrum, target=None, disperser=None, throughput=None, atmosphere=None, diffraction_orders=None,
                 with_background=True, fast_sim=True, full_image=False, with_adr=True):
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
        diffraction_orders: array_like, optional
            List of diffraction orders to simulate. If None, takes first three (default: None).
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
        if diffraction_orders is None:
            self.diffraction_orders = np.arange(spectrum.order, spectrum.order + 3 * np.sign(spectrum.order), np.sign(spectrum.order))
        else:
            self.diffraction_orders = diffraction_orders
        if self.diffraction_orders[0] != 1:
            raise NotImplementedError("Spectrogram simulations are only implemented for 1st diffraction order and followings.")
        if len(self.diffraction_orders) == 0:
            raise ValueError(f"At least one diffraction order must be given for spectrogram simulation.")
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        if target is not None:
            self.target = target
        if disperser is not None:
            self.disperser = disperser
        if throughput is not None:
            self.throughput = throughput
        self.atmosphere = atmosphere

        # load the disperser relative transmissions
        self.tr_ratio = interp1d(parameters.LAMBDAS, np.ones_like(parameters.LAMBDAS), bounds_error=False, fill_value=1.)
        if abs(self.order) == 1:
            self.tr_ratio_next_order = self.disperser.ratio_order_2over1
            self.tr_ratio_next_next_order = self.disperser.ratio_order_3over1
        elif abs(self.order) == 2:
            self.tr_ratio_next_order = self.disperser.ratio_order_3over2
            self.tr_ratio_next_next_order = None
        elif abs(self.order) == 3:
            self.tr_ratio_next_order = None
            self.tr_ratio_next_next_order = None
        else:
            raise ValueError(f"{abs(self.order)=}: must be 1, 2 or 3. "
                             f"Higher diffraction orders not implemented yet in full forward model.")
        self.tr = [self.tr_ratio, self.tr_ratio_next_order, self.tr_ratio_next_next_order]

        self.true_lambdas = None
        self.true_spectrum = None
        self.lambdas = None
        self.model = lambda x, y: np.zeros((x.size, y.size))
        self.psf = load_PSF(psf_type=parameters.PSF_TYPE)
        self.fix_psf_cube = False

        # PSF cube computation
        self.psf_cubes = {}
        self.psf_cubes_masked = {}
        self.boundaries = {}
        self.profile_params = {}
        self.M_sparse_indices = {}
        self.psf_cube_sparse_indices = {}
        for order in self.diffraction_orders:
            self.psf_cubes[order] = None
            self.psf_cubes_masked[order] = None
            self.boundaries[order] = {}
            self.profile_params[order] = None
            self.M_sparse_indices[order] = None
            self.psf_cube_sparse_indices[order] = None
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

    def simulate(self, A1=1.0, A2=0., A3=0., aerosols=0.05, angstrom_exponent=None, ozone=300, pwv=5,
                 D=parameters.DISTANCE2CCD, shift_x=0., shift_y=0., angle=0., B=1., psf_poly_params=None):
        """

        Parameters
        ----------
        A1: float
            Global amplitude of the spectrum (default: 1).
        A2: float
            Relative amplitude of the order 2 spectrum contamination (default: 0).
        A3: float
            Relative amplitude of the order 3 spectrum contamination (default: 0).
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
        >>> psf_poly_params = list(spec.chromatic_psf.from_table_to_poly_params()) * 3
        >>> atmosphere = Atmosphere(airmass=1.2, pressure=800, temperature=10)
        >>> sim = SpectrogramModel(spec, atmosphere=atmosphere, with_background=True, fast_sim=True)
        >>> lambdas, model, model_err = sim.simulate(A2=1, angle=-1.5, psf_poly_params=psf_poly_params)
        >>> sim.plot_spectrogram()

        .. doctest::
            :hide:

            >>> assert np.sum(lambdas) > 0
            >>> assert np.sum(model) > 20
        """
        poly_params = np.array(psf_poly_params).reshape((len(self.diffraction_orders), -1))
        self.rotation_angle = angle
        self.lambdas = self.compute_lambdas_in_spectrogram(D, shift_x, shift_y, angle, niter=5, with_adr=True,
                                                           order=self.diffraction_orders[0])
        self.lambdas_binwidths = np.gradient(self.lambdas)
        if self.atmosphere_sim is None or not self.fix_atm_sim:
            self.atmosphere.set_lambda_range(self.lambdas)
            self.atmosphere_sim = self.atmosphere.simulate(aerosols=aerosols, ozone=ozone, pwv=pwv,
                                                           angstrom_exponent=angstrom_exponent)
        spectrum, spectrum_err = self.simulate_spectrum(self.lambdas, self.atmosphere_sim)

        As = [1, A2, A3]
        ima = np.zeros((self.Ny, self.Nx), dtype="float32")
        ima_err2 = np.zeros((self.Ny, self.Nx), dtype="float32")
        for k, order in enumerate(self.diffraction_orders):
            if self.tr[k] is None or As[k] == 0:  # diffraction order undefined
                continue
            # Dispersion law
            dispersion_law = self.compute_dispersion_in_spectrogram(self.lambdas, shift_x, shift_y, angle,
                                                                    niter=5, with_adr=True, order=order)

            # Spectrum amplitude is in ADU/s
            spec = As[k] * self.tr[k](self.lambdas) * spectrum
            spec_err = As[k] * self.tr[k](self.lambdas) * spectrum_err
            if np.any(np.isnan(spec)):
                spec[np.isnan(spec)] = 0.

            # Evaluate PSF profile
            if self.profile_params[order] is None or not self.fix_psf_cube:
                if k==0:
                    self.profile_params[order] = self.chromatic_psf.update(poly_params[k], x0=self.r0.real + shift_x,
                                                                           y0=self.r0.imag + shift_y, angle=angle, plot=False, apply_bounds=True)
                else:
                    self.profile_params[order] = self.chromatic_psf.from_poly_params_to_profile_params(poly_params[k], apply_bounds=True)
                self.profile_params[order][:, 0] = 1
                self.profile_params[order][:, 1] = dispersion_law.real + self.r0.real
                self.profile_params[order][:, 2] += dispersion_law.imag
            if k == 0:
                self.chromatic_psf.table["Dx"] = self.profile_params[order][:, 1] - self.r0.real
                self.chromatic_psf.table["Dy"] = self.profile_params[order][:, 2] - self.r0.imag

            # Fill the PSF cube for each diffraction order
            argmin = max(0, int(np.argmin(np.abs(self.profile_params[order][:, 1]))))
            argmax = min(self.Nx, np.argmin(np.abs(self.profile_params[order][:, 1]-self.Nx)) + 1)
            if self.psf_cubes[order] is None or not self.fix_psf_cube:
                self.psf_cubes[order] = self.chromatic_psf.build_psf_cube(self.pixels, self.profile_params[order],
                                                                          fwhmx_clip=3 * parameters.PSF_FWHM_CLIP,
                                                                          fwhmy_clip=parameters.PSF_FWHM_CLIP,
                                                                          dtype="float32",
                                                                          mask=self.psf_cubes_masked[order],
                                                                          boundaries=self.boundaries[order])

            # Assemble all diffraction orders in simulation
            for x in range(argmin, argmax):
                if self.boundaries[order]:
                    xmin = self.boundaries[order]["xmin"][x]
                    xmax = self.boundaries[order]["xmax"][x]
                    ymin = self.boundaries[order]["ymin"][x]
                    ymax = self.boundaries[order]["ymax"][x]
                else:
                    xmin, xmax = 0, self.Nx
                    ymin, ymax = 0, self.Ny
                ima[ymin:ymax, xmin:xmax] += spec[x] * self.psf_cubes[order][x, ymin:ymax, xmin:xmax]
                ima_err2[ymin:ymax, xmin:xmax] += (spec_err[x] * self.psf_cubes[order][x, ymin:ymax, xmin:xmax])**2
            if np.allclose(self.profile_params[order][:, 0] , 1):
                self.profile_params[order][:, 0] = spec

        # self.spectrogram is in ADU/s units here
        self.spectrogram = A1 * ima
        self.spectrogram_err = A1 * np.sqrt(ima_err2)

        if self.with_background:
            self.spectrogram += B * self.spectrogram_bgd
        # Save the simulation parameters
        self.psf_poly_params = np.copy(poly_params[0])
        self.header['OZONE_T'] = ozone
        self.header['PWV_T'] = pwv
        self.header['VAOD_T'] = aerosols
        self.header['A1_T'] = A1
        self.header['A2_T'] = A2
        self.header['A3_T'] = A3
        self.header['D2CCD_T'] = D
        self.header['X0_T'] = shift_x
        self.header['Y0_T'] = shift_y
        self.header['ROTANGLE'] = angle

        return self.lambdas, self.spectrogram, self.spectrogram_err


if __name__ == "__main__":
    import doctest

    doctest.testmod()

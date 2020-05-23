"""
simulator
=============----

author : Sylvie Dagoret-Campagne, Jérémy Neveu
affiliation : LAL/CNRS/IN2P3/FRANCE
Collaboration : DESC-LSST

Purpose : Simulate a series of spectra for each experimental spectra measured by auxiliary telescope.
Structure in parallel to Spectractor.
For each experimental spectra a fits file image is generated which holds all possible auxiliary telescope spectra
corresponding to different conditions in aerosols, pwv, and ozone.

creation date : April 18th
Last update : July 2018

"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.interpolate import interp1d
from scipy.integrate import quad

from spectractor.extractor.images import Image
from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.dispersers import Hologram
from spectractor.extractor.targets import Target
from spectractor.extractor.psf import load_PSF
from spectractor.tools import fftconvolve_gaussian, ensure_dir, rebin
from spectractor.config import set_logger
from spectractor.simulation.throughput import TelescopeTransmission
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
import spectractor.parameters as parameters
from spectractor.simulation.adr import adr_calib

class SpectrumSimulation(Spectrum):

    def __init__(self, spectrum, atmosphere, telescope, disperser):
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

        """
        Spectrum.__init__(self)
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        self.my_logger = set_logger(self.__class__.__name__)
        self.disperser = disperser
        self.telescope = telescope
        self.atmosphere = atmosphere
        # save original pixel distances to zero order
        # self.disperser.grating_lambda_to_pixel(self.lambdas, x0=self.x0, order=1)
        # now reset data
        self.lambdas = None
        self.err = None
        self.model = lambda x: np.zeros_like(x)
        self.model_err = lambda x: np.zeros_like(x)

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
        # self.data *= self.lambdas*self.lambdas_binwidths
        return self.data, self.err

    def simulate(self, A1=1.0, A2=0., ozone=300, pwv=5, aerosols=0.05, reso=0.,
                 D=parameters.DISTANCE2CCD, shift_x=0.):
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

        Returns
        -------
        lambdas: array_like
            The wavelength array in nm used for the interpolation.
        spectrum: callable
            The spectrum interpolated function in Target units.
        spectrum_err: array_like
            The spectrum uncertainties interpolated function in Target units.

        """
        # find lambdas including ADR effect
        new_x0 = [self.x0[0] - shift_x, self.x0[1]]
        self.disperser.D = D
        distance = self.chromatic_psf.table['Dx_rot']
        lambdas = self.disperser.grating_pixel_to_lambda(distance - shift_x, x0=new_x0, order=1)
        lambda_ref = np.sum(lambdas * self.data) / np.sum(self.data)
        distance += adr_calib(lambdas, self.adr_params, parameters.OBS_LATITUDE, lambda_ref = lambda_ref)
        lambdas = self.disperser.grating_pixel_to_lambda(distance - shift_x, x0=new_x0, order=1)
        lambdas_order2 = self.disperser.grating_pixel_to_lambda(distance - shift_x, x0=new_x0, order=2)
        # simulate order 1 spectrum amplitude
        self.simulate_without_atmosphere(lambdas)
        atmospheric_transmission = self.atmosphere.simulate(ozone, pwv, aerosols)(lambdas)
        # np.savetxt('atmospheric_trans_20170530_130.txt',np.array([lambdas,atmospheric_transmission(lambdas)]).T)
        self.data *= A1 * atmospheric_transmission
        self.err *= A1 * atmospheric_transmission

        # Now add the systematics
        if reso > 0.1:
            self.data = fftconvolve_gaussian(self.data, reso)
            self.err = np.sqrt(np.abs(fftconvolve_gaussian(self.err ** 2, reso)))
        if A2 > 0.:
            sim_conv = interp1d(lambdas, self.data, kind="linear", bounds_error=False, fill_value=(0, 0))
            err_conv = interp1d(lambdas, self.err, kind="linear", bounds_error=False, fill_value=(0, 0))
            self.data = sim_conv(lambdas) + A2 * sim_conv(lambdas_order2)
            self.err = err_conv(lambdas) + A2 * err_conv(lambdas_order2)

        # now we include effects related to the wrong extraction of the spectrum:
        # wrong estimation of the order 0 position and wrong DISTANCE2CCD
        # pixels = np.arange(0, parameters.CCD_IMSIZE) - self.x0[0]
        # self.disperser.D = parameters.DISTANCE2CCD
        # self.lambdas = self.disperser.grating_pixel_to_lambda(pixels, self.x0, order=1)
        # self.model = interp1d(self.lambdas, self.data, kind="linear", bounds_error=False, fill_value=(0, 0))
        # self.model_err = interp1d(self.lambdas, self.err, kind="linear", bounds_error=False, fill_value=(0, 0))
        min_positive = np.min(self.err[self.err > 0])
        self.err[np.isclose(self.err, 0., atol=0.01 * min_positive)] = min_positive
        return self.lambdas, self.data, self.err


class SpectrogramModel(Spectrum):

    def __init__(self, spectrum, atmosphere, telescope, disperser, with_background=True, fast_sim=False):
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

        """
        Spectrum.__init__(self)
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        self.disperser = disperser
        self.telescope = telescope
        self.atmosphere = atmosphere
        # self.pixels_x = np.arange(self.chromatic_psf.Nx).astype(int)
        # self.pixels_y = np.arange(self.chromatic_psf.Ny).astype(int)
        self.true_lambdas = None
        self.true_spectrum = None
        self.lambdas = None
        self.lambdas_order2 = None
        self.lambdas_binwidths_order2 = None
        self.err = None
        self.model = lambda x, y: np.zeros((x.size, y.size))
        self.psf = load_PSF(psf_type=parameters.PSF_TYPE)
        self.psf_cube = None
        self.fhcube = None
        self.fix_psf_cube = False
        self.fast_sim = fast_sim
        self.with_background = with_background

    def set_true_spectrum(self, lambdas, ozone, pwv, aerosols, shift_t=0.):
        atmosphere = self.atmosphere.simulate(ozone, pwv, aerosols)
        spectrum = self.target.sed(lambdas)
        spectrum *= self.disperser.transmission(lambdas - shift_t)
        spectrum *= self.telescope.transmission(lambdas - shift_t)
        spectrum *= atmosphere(lambdas)
        self.true_spectrum = spectrum
        self.true_lambdas = lambdas
        return spectrum

    def simulate_spectrum(self, lambdas, ozone, pwv, aerosols, shift_t=0.):
        """
        Simulate the spectrum of the object and return the result in Target units.

        Parameters
        ----------
        lambdas: array_like
            The wavelength array in nm.
        ozone: float
            Ozone quantity in Dobson
        pwv: float
            Precipitable Water Vapor quantity in mm
        aerosols: float
            VAOD Vertical Aerosols Optical Depth
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
        atmosphere = self.atmosphere.simulate(ozone, pwv, aerosols)
        if self.fast_sim:
            # TODO: evaluate impact of fast-sim on rapidity/quality of the fit
            spectrum = self.target.sed(lambdas)
            spectrum *= self.disperser.transmission(lambdas - shift_t)
            telescope_transmission = self.telescope.transmission(lambdas - shift_t)
            spectrum *= telescope_transmission
            spectrum *= atmosphere(lambdas)
            spectrum_err = np.zeros_like(spectrum)
            idx = np.where(telescope_transmission > 0)[0]
            spectrum *= parameters.FLAM_TO_ADURATE * lambdas * np.gradient(lambdas)
            spectrum_err[idx] = self.telescope.transmission_err(lambdas)[idx] / telescope_transmission[idx] * spectrum[
                idx]
        else:
            def integrand(lbda):
                return lbda * self.target.sed(lbda) * self.telescope.transmission(lbda - shift_t) \
                       * self.disperser.transmission(lbda - shift_t) * atmosphere(lbda)

            for i in range(len(lambdas) - 1):
                spectrum[i] = parameters.FLAM_TO_ADURATE * quad(integrand, lambdas[i], lambdas[i + 1])[0]

        return spectrum, np.zeros_like(spectrum)

    def simulate_psf(self, psf_poly_params):
        psf_poly_params_with_saturation = np.concatenate([psf_poly_params, [self.spectrogram_saturation]])
        profile_params = self.chromatic_psf.from_poly_params_to_profile_params(psf_poly_params_with_saturation,
                                                                               apply_bounds=True)
        self.chromatic_psf.fill_table_with_profile_params(profile_params)
        # self.chromatic_psf.from_profile_params_to_shape_params(profile_params)
        # self.chromatic_psf.table['Dx'] = np.arange(self.spectrogram_Nx) - self.spec
        self.chromatic_psf.table['Dy_mean'] = 0
        self.chromatic_psf.table['Dy'] = np.copy(self.chromatic_psf.table['y_mean'])
        # derotate
        # self.my_logger.warning(f"\n\tbefore\n {self.chromatic_psf.table[['Dx_rot','Dx','Dy','Dy_mean']][:5]} {angle}")
        self.chromatic_psf.rotate_table(-self.rotation_angle)
        self.chromatic_psf.profile_params = self.chromatic_psf.from_table_to_profile_params()
        if parameters.DEBUG:
            self.chromatic_psf.plot_summary()
        # self.my_logger.warning(f"\n\tafter\n {self.chromatic_psf.table[['Dx_rot','Dx','Dy','Dy_mean']][:5]} {angle}")
        return self.chromatic_psf.profile_params

    def simulate_dispersion(self, D, shift_x, shift_y, r0):
        new_x0 = [self.x0[0] - shift_x, self.x0[1] - shift_y]
        distance = np.array(self.chromatic_psf.get_distance_along_dispersion_axis(shift_x=shift_x, shift_y=shift_y))
        # must have odd size
        if distance.size % 2 == 0:
            distance = distance[:-1]

        # convert pixels into lambdas with ADR for spectrum amplitude evaluation
        self.disperser.D = D
        lambdas = self.disperser.grating_pixel_to_lambda(distance, x0=new_x0, order=1)
        lambda_ref = np.sum(lambdas * self.data) / np.sum(self.data)
        distance += adr_calib(lambdas, self.adr_params, parameters.OBS_LATITUDE, lambda_ref = lambda_ref)
        lambdas = self.disperser.grating_pixel_to_lambda(distance, x0=new_x0, order=1)
        lambdas_order2 = self.disperser.grating_pixel_to_lambda(distance, x0=new_x0, order=2)
        lambdas_order2 = lambdas_order2[lambdas_order2 > np.min(lambdas)]
        distances_order2 = self.disperser.grating_lambda_to_pixel(lambdas_order2, x0=new_x0, order=2)
        # Dx_func = interp1d(lambdas / 2, self.chromatic_psf.table['Dx'], bounds_error=False, fill_value=(0, 0))
        # Dy_mean_func = interp1d(lambdas / 2, self.chromatic_psf.table['Dy_mean'],bounds_error=False, fill_value=(0,0))
        # dy_func = interp1d(lambdas / 2, self.chromatic_psf.table['Dy'] - self.chromatic_psf.table['Dy_mean'],
        #                    bounds_error=False, fill_value=(0, 0))
        # dispersion_law = r0 + (self.chromatic_psf.table['Dx'] - shift_x) + 1j * (
        #             self.chromatic_psf.table['Dy'] - shift_y)
        # dispersion_law_order2 = r0 + (Dx_func(lambdas_order2) - shift_x) + 1j * (
        #             Dy_mean_func(lambdas_order2) + dy_func(lambdas_order2) - shift_y)
        # Dx_func = interp1d(lambdas, self.chromatic_psf.table['Dx'], bounds_error=False, fill_value=(0, 0))
        # Dy_mean_func = interp1d(lambdas, self.chromatic_psf.table['Dy_mean'], bounds_error=False, fill_value=(0, 0))

        # dispersion laws from the PSF table
        dy_func = interp1d(lambdas,
                           self.chromatic_psf.table['Dy'][:distance.size] - self.chromatic_psf.table['Dy_mean'][
                                                                            :distance.size],
                           bounds_error=False, fill_value=(0, 0))

        dispersion_law = r0 + (self.chromatic_psf.table['Dx'][:distance.size] - shift_x) + 1j * (
                self.chromatic_psf.table['Dy'][:distance.size] - shift_y)
        dispersion_law_order2 = r0 + (distances_order2 * np.cos(np.pi * self.rotation_angle / 180) - shift_x) + 1j * (
                distances_order2 * np.sin(np.pi * self.rotation_angle / 180) + dy_func(lambdas_order2) - shift_y)

        self.lambdas_order2 = lambdas_order2
        self.lambdas = lambdas
        self.lambdas_binwidths = np.gradient(lambdas)
        self.lambdas_binwidths_order2 = np.gradient(lambdas_order2)

        if parameters.DEBUG:
            from spectractor.tools import from_lambda_to_colormap
            plt.plot(self.chromatic_psf.table['Dx'], self.chromatic_psf.table['Dy_mean'], 'k-', label="mean")
            plt.scatter(-r0.real + dispersion_law.real, -r0.imag + dispersion_law.imag, label="dispersion_law",
                        cmap=from_lambda_to_colormap(lambdas), c=lambdas)
            plt.scatter(-r0.real + dispersion_law_order2.real, -r0.imag + dispersion_law_order2.imag,
                        label="dispersion_law_order2", cmap=from_lambda_to_colormap(lambdas_order2), c=lambdas_order2)
            plt.title(f"{new_x0}")
            plt.legend()
            plt.show()

        return lambdas, lambdas_order2, dispersion_law, dispersion_law_order2

    def build_spectrogram(self, profile_params, spectrum, dispersion_law):
        # Spectrum units must in ADU/s
        pixels = np.mgrid[:self.spectrogram_Nx, :self.spectrogram_Ny]
        simul = np.zeros((self.spectrogram_Ny, self.spectrogram_Nx))
        nlbda = dispersion_law.size
        # cannot use directly ChromaticPSF2D class because it does not include the rotation of the spectrogram
        # TODO: increase rapidity (multithreading, optimisation...)
        # pixels = np.arange(self.spectrogram_Ny)
        # for i in range(0, nlbda, 1):
        #     p = np.array([spectrum[i], dispersion_law[i].real, dispersion_law[i].imag] + list(profile_params[i, 3:]))
        #     simul[:, i] = self.psf.evaluate(pixels, p=p)
        for i in range(0, nlbda, 1):
            # here spectrum[i] is in ADU/s
            p = np.array([spectrum[i], dispersion_law[i].real, dispersion_law[i].imag] + list(profile_params[i, 3:]))
            psf_lambda = self.psf.evaluate(pixels, p=p)
            simul += psf_lambda
        return simul

    # @profile
    def simulate(self, A1=1.0, A2=0., ozone=300, pwv=5, aerosols=0.05, D=parameters.DISTANCE2CCD,
                 shift_x=0., shift_y=0., angle=0., psf_poly_params=None):
        """

        Parameters
        ----------
        A1
        A2
        ozone
        pwv
        aerosols
        psf_poly_params
        D
        shift_x
        shift_y
        angle

        Returns
        -------

        Example
        -------
        >>> from spectractor import parameters
        >>> parameters.DEBUG = True
        >>> spectrum, telescope, disperser, target = SimulatorInit('outputs/reduc_20170530_134_spectrum.fits')
        >>> airmass = spectrum.header['AIRMASS']
        >>> pressure = spectrum.header['OUTPRESS']
        >>> temperature = spectrum.header['OUTTEMP']
        >>> atmosphere = Atmosphere(airmass, pressure, temperature)
        >>> psf_poly_params = spectrum.chromatic_psf.from_table_to_poly_params()
        >>> spec = SpectrogramModel(spectrum, atmosphere, telescope, disperser, with_background=True)
        >>> lambdas, data, err = spec.simulate(A2=0, angle=spectrum.rotation_angle, psf_poly_params=psf_poly_params)
        """
        import time
        start = time.time()
        self.rotation_angle = angle
        profile_params = self.simulate_psf(psf_poly_params)
        self.my_logger.debug(f'\n\tAfter psf: {time.time() - start}')
        start = time.time()
        r0 = self.spectrogram_x0 + 1j * self.spectrogram_y0
        lambdas, lambdas_order2, dispersion_law, dispersion_law_order2 = self.simulate_dispersion(D, shift_x, shift_y, r0)
        self.my_logger.debug(f'\n\tAfter dispersion: {time.time() - start}')
        start = time.time()
        spectrum, spectrum_err = self.simulate_spectrum(lambdas, ozone, pwv, aerosols)
        self.true_spectrum = spectrum
        self.my_logger.debug(f'\n\tAfter spectrum: {time.time() - start}')
        start = time.time()
        ima1 = self.build_spectrogram(profile_params, spectrum, dispersion_law)
        self.my_logger.debug(f'\n\tAfter build ima1: {time.time() - start}')
        start = time.time()

        # Add order 2
        ima2 = np.zeros_like(ima1)
        if A2 > 0.:
            nlbda_order2 = dispersion_law_order2.size
            spectrum_order1 = interp1d(lambdas,spectrum,bounds_error=False, fill_value=(0, 0))
            lambdas_binwidths_O1_for_O2 = np.zeros(nlbda_order2)
            for i in range(nlbda_order2):
                jmin = max(np.argmin(abs(lambdas - lambdas_order2[i])),0)
                if lambdas[jmin] > lambdas_order2[i]:
                    jmax = jmin
                    jmin -=1
                else:
                    jmax = jmin + 1
                lambdas_binwidths_O1_for_O2[i] = lambdas[jmax] - lambdas[jmin]

            spectrum_order2 = spectrum_order1(lambdas_order2) * self.lambdas_binwidths_order2 / lambdas_binwidths_O1_for_O2
            ima2 = self.build_spectrogram(profile_params[-nlbda_order2:], spectrum_order2,
                                          dispersion_law_order2)

        self.my_logger.debug(f'\n\tAfter build ima2: {time.time() - start}')
        start = time.time()
        # self.data is in ADU/s units here
        self.data = A1 * (ima1 + A2 * ima2)
        if self.with_background:
            self.data += self.spectrogram_bgd
        self.err = np.zeros_like(self.data)
        self.my_logger.debug(f'\n\tAfter all: {time.time() - start}')
        if parameters.DEBUG:
            fig, ax = plt.subplots(3, 1, sharex="all", figsize=(12, 9))
            ax[0].imshow(self.data, origin='lower')
            ax[0].set_title('Model')
            ax[1].imshow(self.spectrogram, origin='lower')
            ax[1].set_title('Data')
            ax[1].set_xlabel('X [pixels]')
            ax[0].set_ylabel('Y [pixels]')
            ax[1].set_ylabel('Y [pixels]')
            ax[0].grid()
            ax[1].grid()
            if self.with_background:
                ax[2].plot(np.sum(self.data, axis=0), label="model")
            else:
                ax[2].plot(np.sum(self.data + self.spectrogram_bgd, axis=0), label="model")
            ax[2].plot(np.sum(self.spectrogram, axis=0), label="data")
            ax[2].grid()
            ax[2].legend()
            fig.tight_layout()
            plt.show()
        return self.lambdas, self.data, self.err

    def simulate_FFT(self, A1=1.0, A2=0., ozone=300, pwv=5, aerosols=0.05, D=parameters.DISTANCE2CCD,
                     shift_x=0., shift_y=0., angle=0., psf_poly_params=None):  # pragma: nocover
        """
        DEPRECATED

        Parameters
        ----------
        A1
        A2
        ozone
        pwv
        aerosols
        psf_poly_params
        D
        shift_x
        shift_y

        Returns
        -------

        Example
        -------
        # >>> from spectractor.extractor.psf import  ChromaticPSF1D
        # >>> spectrum, telescope, disperser, target = SimulatorInit('outputs/reduc_20170530_134_spectrum.fits')
        # >>> airmass = spectrum.header['AIRMASS']
        # >>> pressure = spectrum.header['OUTPRESS']
        # >>> temperature = spectrum.header['OUTTEMP']
        # >>> atmosphere = Atmosphere(airmass, pressure, temperature)
        # >>> psf_poly_params = spectrum.chromatic_psf.from_table_to_poly_params()
        # >>> spec = SpectrogramModel(spectrum, atmosphere, telescope, disperser)
        # >>> lambdas, data, err = spec.simulate_FFT(psf_poly_params=psf_poly_params)
        """
        import slitless.fourier.arrays as FA
        import slitless.fourier.fourier as F
        import time
        start = time.time()
        self.rotation_angle = angle
        self.simulate_psf(psf_poly_params)
        self.my_logger.debug(f'\n\tTime after simulate PSF: {time.time() - start}')
        start = time.time()
        # print(self.spectrogram_x0, self.spectrogram_Nx, self.spectrogram_y0, self.spectrogram_Ny,
        # self.spectrogram_xmin,self.spectrogram_ymin)
        # If nofftshift,use this r0:
        r0 = (self.spectrogram_x0 - self.spectrogram_Nx / 2) + 1j * (self.spectrogram_y0 - self.spectrogram_Ny / 2)
        # Else, this one:
        # r0 = (self.spectrogram_x0 ) + 1j * (self.spectrogram_y0 )
        lambdas, dispersion_law, dispersion_law_order2 = self.simulate_dispersion(D, shift_x, shift_y, r0)
        self.my_logger.debug(f'\n\tTime after simulate disp: {time.time() - start}')
        start = time.time()
        spectrum, spectrum_err = self.simulate_spectrum(lambdas, ozone, pwv, aerosols)
        self.true_spectrum = spectrum
        self.my_logger.debug(f'\n\tTime after simulate spec: {time.time() - start}')
        start = time.time()
        nlbda = lambdas.size

        # oversampling of the PSF to avoid Gibbs ringings
        # rescale_factor must be odd
        rescale_factor = 3
        if self.fast_sim:
            rescale_factor = 1
        nima = rescale_factor * self.spectrogram_Ny
        y, x = FA.create_coords((nima, nima), starts='auto', steps=1 / rescale_factor)  # (ny, nx)

        # PSF cube
        if self.psf_cube is None or not self.fix_psf_cube:
            cube = pyfftw.zeros_aligned((nlbda, nima, nima), dtype='float32')
            shape_params = np.array([self.chromatic_psf.table[name] for name in MoffatGauss.param_names[3:]]).T
            for lbda in range(nlbda):
                ima = self.psf.evaluate(x, y, 1, 0, 0, *shape_params[lbda])
                # norm = MoffatGauss.normalisation(1, *shape_params[l])
                # if norm != 0.:
                #     ima /= norm  # Flux normalization: the flux should go in the spectrum
                cube[lbda] = np.copy(ima)
            self.psf_cube = cube
        else:
            cube = self.psf_cube

        # Extended image (nima × nlbda) has to be perfecty centered
        ny, nx = (nima, rescale_factor * self.spectrogram_Nx)  # Rectangular minimal embedding
        hcube = FA.embed_array(cube, (nlbda, ny, nx))
        self.my_logger.debug(f'\n\tTime after filling the cube: {time.time() - start}')
        start = time.time()

        # Generate slitless-spectroscopy image from Fourier analysis
        if self.fhcube is None or not self.fix_psf_cube:
            uh, vh, fhcube = F.fft_cube(hcube)  # same shape as hima
            uh *= rescale_factor
            vh *= rescale_factor
            self.uh = uh
            self.vh = vh
            self.fhcube = fhcube
            # print(uh.shape, vh.shape, np.min(self.uh), np.max(self.uh), fhcube.shape,
            # dispersion_law.reshape(-1, 1, 1).shape)
        else:
            uh = self.uh
            vh = self.vh
            fhcube = self.fhcube
        self.my_logger.debug(f'\n\tTime after fourier cube: {time.time() - start}')
        # r0 = (self.spectrogram_x0 - self.spectrogram_Nx / 2 - shift_x) \
        #      + 1j * (self.spectrogram_y0 - self.spectrogram_Ny / 2 - shift_y)
        fdima0 = F.disperse_fcube(uh, vh, fhcube, spectrum, dispersion_law, method="numexpr")  # FT
        self.my_logger.debug(f'\n\tTime after simulate after fourier: {time.time() - start}')
        start = time.time()

        # Dispersed image (noiseless)
        dima0 = F.ifft_image(fdima0)
        self.my_logger.debug(f'\n\tTime after simulate inverse fourier: {time.time() - start}')
        start = time.time()

        # Now add the systematics
        # if reso > 1:
        #     self.data = fftconvolve_gaussian(self.data, reso)
        #     self.err = np.sqrt(np.abs(fftconvolve_gaussian(self.err ** 2, reso)))
        dima0_2 = np.zeros_like(dima0)
        if A2 > 0.:
            nlbda_order2 = dispersion_law_order2.size
            fdima0_2 = F.disperse_fcube(uh, vh, fhcube[-nlbda_order2:], spectrum[:nlbda_order2], dispersion_law_order2,
                                        method="numexpr")  # FT
            self.my_logger.debug(f'\n\tTime after simulate after fourier order 2: {time.time() - start}')
            start = time.time()
            dima0_2 = F.ifft_image(fdima0_2)
            self.my_logger.debug(f'\n\tTime after simulate inverse fourier order 2: {time.time() - start}')

            # sim_conv = interp1d(lambdas, self.data, kind="linear", bounds_error=False, fill_value=(0, 0))
            # err_conv = interp1d(lambdas, self.err, kind="linear", bounds_error=False, fill_value=(0, 0))
            # self.model = lambda x: sim_conv(x) + A2 * sim_conv(x / 2)
            # self.model_err = lambda x: np.sqrt(np.abs((err_conv(x)) ** 2 + (0.5 * A2 * err_conv(x / 2)) ** 2))
            # self.data = self.model(lambdas)
            # self.err = self.model_err(lambdas)
        # Going to observable spectrum: must convert units (ie multiply by dlambda)
        self.data = A1 * (dima0 + A2 * dima0_2)
        # print(self.data.shape)
        # toto = np.copy(self.data)
        if rescale_factor != 1:
            self.data = rebin(self.data, (self.spectrogram_Ny, self.spectrogram_Nx))
        # plt.plot(np.sum(self.data,axis=0))
        # plt.plot(np.sum(A1*dima0,axis=0))
        # plt.plot(np.sum(A1*A2*dima0_2,axis=0))
        # plt.plot(spectrum)
        # plt.show()
        middle = self.data.shape[0] // 2
        width = int(self.spectrogram_ymax - self.spectrogram_ymin) // 2
        # self.data = self.data[middle-width:middle+width+1, :]
        self.lambdas = lambdas
        self.lambdas_binwidths = np.gradient(lambdas)
        # self.convert_from_flam_to_ADUrate()
        # print(self.data.shape, (self.spectrogram_Ny,self.spectrogram_Nx))
        # fig, ax = plt.subplots(2,1)
        # ax[0].imshow(self.data, origin="lower", aspect="auto")
        # ax[1].imshow(toto, origin="lower", aspect="auto")
        # plt.show()
        if self.with_background:
            self.data += self.spectrogram_bgd
        self.err = np.zeros_like(self.data)
        if parameters.DEBUG:
            fig, ax = plt.subplots(2, 1, sharex="all", figsize=(12, 6))
            ax[0].imshow(self.data, origin='lower')
            ax[0].set_title('Model')
            ax[1].imshow(self.spectrogram, origin='lower')
            ax[1].set_title('Data')
            ax[1].set_xlabel('X [pixels]')
            ax[0].set_ylabel('Y [pixels]')
            ax[1].set_ylabel('Y [pixels]')
            fig.tight_layout()
            plt.show()
        return self.lambdas, self.data, self.err


class SpectrumSimGrid:
    """ SpectrumSim class used to store information and methods
    relative to spectrum simulation.
    NEED TO ADAPT THIS CLASS TO THE FULL SIMULATION WITH SYSTEMATICS.
    """

    # ---------------------------------------------------------------------------
    def __init__(self, spectrum, atmgrid, telescope, disperser, target, header, filename=""):
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


def SimulatorInit(filename):
    """ SimulatorInit
    Main function to evaluate several spectra
    A grid of spectra will be produced for a given target, airmass and pressure
    """
    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SIMULATOR initialisation')
    # Load data spectrum
    try:
        spectrum = Spectrum(filename)
    except:
        spectrum = Image(filename)

    # TELESCOPE TRANSMISSION
    # ------------------------
    telescope = TelescopeTransmission(spectrum.filter)

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
        spectrum_simulation.plot_spectrum()
    return spectrum_simulation


def SpectrogramSimulatorCore(spectrum, telescope, disperser, airmass=1.0, pressure=800, temperature=10,
                             pwv=5, ozone=300, aerosols=0.05, A1=1.0, A2=0.,
                             D=parameters.DISTANCE2CCD, shift_x=0., shift_y=0., shift_t=0., angle=0.,
                             psf_poly_params=None, with_background=True, fast_sim=False):
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
                                              with_background=with_background, fast_sim=fast_sim)
    spectrogram_simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, angle, psf_poly_params)
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
    spectra = SpectrumSimGrid(spectrum, atm, telescope, disperser, target, atm.header)
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
                         D=parameters.DISTANCE2CCD, shift_x=0., shift_y=0., shift_t=0., angle=0., psf_poly_params=None):
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
                                                      shift_y=shift_y, shift_t=shift_t, angle=angle,
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


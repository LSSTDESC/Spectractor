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

from scipy.interpolate import interp1d, RegularGridInterpolator

from spectractor.extractor.images import Image
from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.dispersers import Hologram
from spectractor.extractor.targets import Target
from spectractor.extractor.psf import PSF2D
from spectractor.tools import fftconvolve_gaussian, ensure_dir
from spectractor.config import set_logger
import spectractor.parameters as parameters

import spectractor.simulation.libradtran as libradtran
from spectractor.simulation.throughput import Throughput

import slitless.fourier.arrays as FA
# import slitless.fourier.plots as FP
import slitless.fourier.fourier as F
import slitless.fourier.models as FM


class Atmosphere(object):
    """
    Atmosphere():
        class to evaluate an atmospheric transmission calling libradtran
    Args:
        airmass (:obj:`float`): airmass of the target
        pressure (:obj:`float`): pressure of the atmosphere
        temperature (:obj:`float`): temperature of the atmosphere
    """

    def __init__(self, airmass, pressure, temperature):
        self.my_logger = set_logger(self.__class__.__name__)
        self.airmass = airmass
        self.pressure = pressure
        self.temperature = temperature
        self.transmission = lambda x: np.ones_like(x).astype(float)

    def simulate(self, ozone, pwv, aerosols):
        """
        Args:
            ozone (:obj:`float`): ozone quantity
            pwv (:obj:`float`): pressure water vapor
            aerosols (:obj:`float`): VAOD Vertical Aerosols Optical Depth
        """
        # first determine the length
        if parameters.VERBOSE:
            self.my_logger.info(
                '\n\tAtmospheric simulation with z=%4.2f, P=%4.2f, T=%4.2f, PWV=%4.2f, OZ=%4.2f, VAOD=%4.2f ' % (
                    self.airmass, self.pressure, self.temperature, pwv, ozone, aerosols))

        lib = libradtran.Libradtran()
        path = lib.simulate(self.airmass, pwv, ozone, aerosols, self.pressure)
        data = np.loadtxt(path)
        wl = data[:, 0]
        atm = data[:, 1]
        self.transmission = interp1d(wl, atm, kind='linear', bounds_error=False, fill_value=(0, 0))
        self.pwv = pwv
        self.ozone = ozone
        self.aerosols = aerosols
        return self.transmission

    def plot_transmission(self):
        plt.figure()
        plt.plot(parameters.LAMBDAS, self.transmission(parameters.LAMBDAS),
                 label='z=%4.2f, P=%4.2f, T=%4.2f' % (self.airmass, self.pressure, self.temperature))
        plt.grid()
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Atmospheric transparency")
        plt.legend()
        if parameters.DISPLAY: plt.show()


# ----------------------------------------------------------------------------------
class AtmosphereGrid(Atmosphere):

    def __init__(self, data_filename, filename="", airmass=1., pressure=800., temperature=10.):
        Atmosphere.__init__(self, airmass, pressure, temperature)
        self.my_logger = set_logger(self.__class__.__name__)
        self.data_filename = data_filename
        # ------------------------------------------------------------------------
        # Definition of data format for the atmospheric grid
        # -----------------------------------------------------------------------------

        #  column 0 : count number
        #  column 1 : aerosol value
        #  column 2 : pwv value
        #  column 3 : ozone value
        #  column 4 : data start
        self.index_atm_count = 0
        self.index_atm_aer = 1
        self.index_atm_pwv = 2
        self.index_atm_oz = 3
        self.index_atm_data = 4

        # specify parameters for the atmospheric grid
        self.set_grid(pwv_grid=[0, 10, 5], ozone_grid=[100, 700, 7], aerosol_grid=[0, 0.1, 5])

        self.header = fits.Header()
        if filename != "":
            self.load_file(filename)

    def set_grid(self, pwv_grid=[0, 10, 5], ozone_grid=[100, 700, 7], aerosol_grid=[0, 0.1, 5]):
        # aerosols
        # NB_AER_POINTS=20
        self.NB_AER_POINTS = int(aerosol_grid[2])
        self.AER_MIN = float(aerosol_grid[0])
        self.AER_MAX = float(aerosol_grid[1])

        # ozone
        # NB_OZ_POINTS=5
        self.NB_OZ_POINTS = int(ozone_grid[2])
        self.OZ_MIN = float(ozone_grid[0])
        self.OZ_MAX = float(ozone_grid[1])

        # pwv
        # NB_PWV_POINTS=11
        self.NB_PWV_POINTS = int(pwv_grid[2])
        self.PWV_MIN = float(pwv_grid[0])
        self.PWV_MAX = float(pwv_grid[1])

        # definition of the grid
        self.AER_Points = np.linspace(self.AER_MIN, self.AER_MAX, self.NB_AER_POINTS)
        self.OZ_Points = np.linspace(self.OZ_MIN, self.OZ_MAX, self.NB_OZ_POINTS)
        self.PWV_Points = np.linspace(self.PWV_MIN, self.PWV_MAX, self.NB_PWV_POINTS)

        # total number of points
        self.NB_ATM_POINTS = self.NB_AER_POINTS * self.NB_OZ_POINTS * self.NB_PWV_POINTS

        self.NB_atm_HEADER = 5
        self.NB_atm_DATA = len(parameters.LAMBDAS) - 1

        # create the numpy array that will contains the atmospheric grid
        self.atmgrid = np.zeros((self.NB_ATM_POINTS + 1, self.NB_atm_HEADER + self.NB_atm_DATA))
        self.atmgrid[0, self.index_atm_data:] = parameters.LAMBDAS

    def compute(self):
        # first determine the length
        if parameters.VERBOSE or parameters.DEBUG:
            self.my_logger.info('\n\tAtmosphere simulations for z=%4.2f, P=%4.2f, T=%4.2f, for data-file=%s ' % (
                self.airmass, self.pressure, self.temperature, self.data_filename))
        count = 0
        for aer in self.AER_Points:
            for pwv in self.PWV_Points:
                for oz in self.OZ_Points:
                    count += 1
                    # fills headers info in the numpy array
                    self.atmgrid[count, self.index_atm_count] = count
                    self.atmgrid[count, self.index_atm_aer] = aer
                    self.atmgrid[count, self.index_atm_pwv] = pwv
                    self.atmgrid[count, self.index_atm_oz] = oz
                    transmission = super(AtmosphereGrid, self).simulate(oz, pwv, aer)
                    transm = transmission(parameters.LAMBDAS)
                    self.atmgrid[count, self.index_atm_data:] = transm  # each of atmospheric spectrum
        return self.atmgrid

    def plot_transmission(self):
        plt.figure()
        counts = self.atmgrid[1:, self.index_atm_count]
        for count in counts:
            plt.plot(parameters.LAMBDAS, self.atmgrid[int(count), self.index_atm_data:])
        plt.grid()
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Atmospheric transmission")
        plt.title("Atmospheric variations")
        if parameters.DISPLAY: plt.show()

    def plot_transm_img(self):
        plt.figure()
        img = plt.imshow(self.atmgrid[1:, self.index_atm_data:], origin='lower', cmap='jet')
        plt.grid(True)
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Simulation number")
        plt.title(" Atmospheric variations")
        cbar = plt.colorbar(img)
        cbar.set_label('Atmospheric transmission')
        if parameters.DISPLAY:
            plt.show()

    def save_file(self, filename=""):

        hdr = fits.Header()

        if filename != "":
            self.filename = filename

        if self.filename == "":
            infostring = '\n\t Atmosphere:save_file no settings file given ...'
            self.my_logger.info(infostring)
            return
        else:
            hdr['ATMSIM'] = "libradtran"
            hdr['SIMVERS'] = "2.0.1"
            hdr['DATAFILE'] = self.data_filename
            hdr['SIMUFILE'] = os.path.basename(self.filename)

            hdr['AIRMASS'] = self.airmass
            hdr['PRESSURE'] = self.pressure
            hdr['TEMPERAT'] = self.temperature
            hdr['NBATMPTS'] = self.NB_ATM_POINTS

            hdr['NBAERPTS'] = self.NB_AER_POINTS
            hdr['AERMIN'] = self.AER_MIN
            hdr['AERMAX'] = self.AER_MAX

            hdr['NBPWVPTS'] = self.NB_PWV_POINTS
            hdr['PWVMIN'] = self.PWV_MIN
            hdr['PWVMAX'] = self.PWV_MAX

            hdr['NBOZPTS'] = self.NB_OZ_POINTS
            hdr['OZMIN'] = self.OZ_MIN
            hdr['OZMAX'] = self.OZ_MAX

            hdr['AER_PTS'] = np.array_str(self.AER_Points)
            hdr['PWV_PTS'] = np.array_str(self.PWV_Points)
            hdr['OZ_PTS'] = np.array_str(self.OZ_Points)
            hdr['NBWLBIN'] = parameters.LAMBDAS.size
            hdr['WLMIN'] = parameters.LAMBDA_MIN
            hdr['WLMAX'] = parameters.LAMBDA_MAX

            hdr['IDX_CNT'] = self.index_atm_count
            hdr['IDX_AER'] = self.index_atm_aer
            hdr['IDX_PWV'] = self.index_atm_pwv
            hdr['IDX_OZ'] = self.index_atm_oz
            hdr['IDX_DATA'] = self.index_atm_data

            hdu = fits.PrimaryHDU(self.atmgrid, header=hdr)
            hdu.writeto(self.filename, overwrite=True)
            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info('\n\tAtmosphere.save atm-file=%s' % (self.filename))

            return hdr

    def load_file(self, filename):

        if filename != "":
            self.filename = filename

        if self.filename == "":
            infostring = '\n\t Atmosphere:load_file no settings file given ...'
            self.my_logger.info(infostring)

            return
        else:

            hdu = fits.open(self.filename)
            hdr = hdu[0].header
            self.header = hdr

            # hdr['ATMSIM'] = "libradtran"
            # hdr['SIMVERS'] = "2.0.1"
            self.data_filename = hdr['DATAFILE']
            # hdr['SIMUFILE']=os.path.basename(self.file_name)

            self.airmass = hdr['AIRMASS']
            self.pressure = hdr['PRESSURE']
            self.temperature = hdr['TEMPERAT']

            # hope those are the same parameters : TBD !!!!
            self.NB_ATM_POINTS = hdr['NBATMPTS']

            self.NB_AER_POINTS = hdr['NBAERPTS']
            self.AER_MIN = hdr['AERMIN']
            self.AER_MAX = hdr['AERMAX']

            self.NB_PWV_POINTS = hdr['NBPWVPTS']
            self.PWV_MIN = hdr['PWVMIN']
            self.PWV_MAX = hdr['PWVMAX']

            self.NB_OZ_POINTS = hdr['NBOZPTS']
            self.OZ_MIN = hdr['OZMIN']
            self.OZ_MAX = hdr['OZMAX']

            self.AER_Points = np.linspace(self.AER_MIN, self.AER_MAX, self.NB_AER_POINTS)
            self.OZ_Points = np.linspace(self.OZ_MIN, self.OZ_MAX, self.NB_OZ_POINTS)
            self.PWV_Points = np.linspace(self.PWV_MIN, self.PWV_MAX, self.NB_PWV_POINTS)

            self.NBWLBINS = hdr['NBWLBIN']
            self.WLMIN = hdr['WLMIN']
            self.WLMAX = hdr['WLMAX']

            self.index_atm_count = hdr['IDX_CNT']
            self.index_atm_aer = hdr['IDX_AER']
            self.index_atm_pwv = hdr['IDX_PWV']
            self.index_atm_oz = hdr['IDX_OZ']
            self.index_atm_data = hdr['IDX_DATA']

            self.atmgrid = np.zeros((self.NB_ATM_POINTS + 1, self.NB_atm_HEADER + self.NB_atm_DATA))

            self.atmgrid[:, :] = hdu[0].data[:, :]

            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info('\n\tAtmosphere.load_image atm-file=%s' % (self.filename))

            # interpolate the grid
            self.lambdas = self.atmgrid[0, self.index_atm_data:]
            self.model = RegularGridInterpolator((self.lambdas, self.OZ_Points, self.PWV_Points, self.AER_Points), (
                self.atmgrid[1:, self.index_atm_data:].reshape(self.NB_AER_POINTS, self.NB_PWV_POINTS,
                                                               self.NB_OZ_POINTS,
                                                               len(self.lambdas))).T, bounds_error=False, fill_value=0)

            return self.atmgrid, self.header

    def simulate(self, ozone, pwv, aerosols):
        """ first ozone, second pwv, last aerosols, to respect order of loops when generating the grid"""
        ones = np.ones_like(self.lambdas)
        points = np.array([self.lambdas, ozone * ones, pwv * ones, aerosols * ones]).T
        atm = self.model(points)
        self.transmission = interp1d(self.lambdas, atm, kind='linear', bounds_error=False, fill_value=(0, 0))
        self.pwv = pwv
        self.ozone = ozone
        self.aerosols = aerosols
        return self.transmission


# ----------------------------------------------------------------------------------
class TelescopeTransmission():
    """
    TelescopeTransmission : Transmission of the telescope
    - mirrors
    - throughput
    - QE
    - Filter

    """

    # ---------------------------------------------------------------------------
    def __init__(self, filtername=""):
        """
        Args:
        file_name (:obj:`str`): path to the data file_name (for info only)
        """

        self.my_logger = set_logger(self.__class__.__name__)
        self.filtername = filtername
        self.load_transmission()

    # ---------------------------------------------------------------------------
    def load_transmission(self):
        """
        load_transmission(self) :
            load_image the telescope transmission
            return the total telescope transmission, disperser excluded,
                as a fnction of the wavelength in Angstrom
        """

        '''
        # QE
        wl,qe=ctio.get_quantum_efficiency(datapath)
        self.qe=interp1d(wl,qe,kind='linear',bounds_error=False,fill_value=0.)

        #  Throughput
        wl,trt=ctio.get_telescope_throughput(datapath)
        self.to=interp1d(wl,trt,kind='linear',bounds_error=False,fill_value=0.)

        # Mirrors
        wl,trm=ctio.get_mirror_reflectivity(datapath)
        self.tm=interp1d(wl,trm,kind='linear',bounds_error=False,fill_value=0.)
        '''
        throughput = Throughput()
        wl, trm, err = throughput.get_total_throughput()
        self.to = interp1d(wl, trm, kind='linear', bounds_error=False, fill_value=0.)
        err = np.sqrt(err ** 2 + parameters.OBS_TRANSMISSION_SYSTEMATICS ** 2)
        self.to_err = interp1d(wl, err, kind='linear', bounds_error=False, fill_value=0.)

        # Filter RG715
        wl, trg = throughput.get_RG715()
        self.tfr = interp1d(wl, trg, kind='linear', bounds_error=False, fill_value=0.)

        # Filter FGB37
        wl, trb = throughput.get_FGB37()
        self.tfb = interp1d(wl, trb, kind='linear', bounds_error=False, fill_value=0.)

        if self.filtername == "RG715":
            TF = self.tfr
        elif self.filtername == "FGB37":
            TF = self.tfb
        else:
            TF = lambda x: np.ones_like(x).astype(float)

        self.tf = TF

        # self.transmission=lambda x: self.qe(x)*self.to(x)*(self.tm(x)**2)*self.tf(x)
        self.transmission = lambda x: self.to(x) * self.tf(x)
        self.transmission_err = lambda x: self.to_err(x)
        return self.transmission

    # ---------------------------------------------------------------------------
    def plot_transmission(self, xlim=None):
        """
        plot_transmission()
            plot the various transmissions of the instrument
        """
        plt.figure()
        # plt.plot(parameters.LAMBDAS,self.qe(parameters.LAMBDAS),'b-',label='qe')
        plt.plot(parameters.LAMBDAS, self.to(parameters.LAMBDAS), 'g-', label='othr')
        # plt.plot(parameters.LAMBDAS,self.tm(parameters.LAMBDAS),'y-',label='mirr')
        plt.plot(parameters.LAMBDAS, self.tf(parameters.LAMBDAS), 'k-', label='filt')
        plt.plot(parameters.LAMBDAS, self.tfr(parameters.LAMBDAS), 'k:', label='RG715')
        plt.plot(parameters.LAMBDAS, self.tfb(parameters.LAMBDAS), 'k--', label='FGB37')
        plt.errorbar(parameters.LAMBDAS, self.transmission(parameters.LAMBDAS),
                     yerr=self.transmission_err(parameters.LAMBDAS), color='r', linestyle='-', lw=2, label='tot')
        plt.legend()
        plt.grid()
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Transmission")
        plt.title("Telescope transmissions")


# ----------------------------------------------------------------------------------
class SpectrumSimulation(Spectrum):
    """ SpectrumSim class used to store information and methods
    relative to spectrum simulation.
    """

    # ---------------------------------------------------------------------------
    def __init__(self, spectrum, atmosphere, telescope, disperser):
        Spectrum.__init__(self)
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        self.my_logger = set_logger(self.__class__.__name__)
        self.disperser = disperser
        self.telescope = telescope
        self.atmosphere = atmosphere
        self.lambdas = None
        self.data = None
        self.err = None
        self.model = lambda x: np.zeros_like(x)

    def simulate_without_atmosphere(self, lambdas):
        self.lambdas = lambdas
        self.err = np.zeros_like(lambdas)
        self.lambdas_binwidths = np.gradient(lambdas)
        self.data = self.disperser.transmission(lambdas)
        self.data *= self.telescope.transmission(lambdas)
        self.data *= self.target.sed(lambdas)
        self.err = np.zeros_like(self.data)
        idx = np.where(self.telescope.transmission(lambdas) > 0)[0]
        self.err[idx] = self.telescope.transmission_err(lambdas)[idx] / self.telescope.transmission(lambdas)[idx] * \
                        self.data[idx]
        # self.data *= self.lambdas*self.lambdas_binwidths
        return self.data, self.err

    def simulate(self, A1=1.0, A2=0., ozone=300, pwv=5, aerosols=0.05, reso=0., D=parameters.DISTANCE2CCD, shift=0.):
        pixels = np.arange(0, parameters.CCD_IMSIZE) - self.x0[0] - shift
        new_x0 = [self.x0[0] - shift, self.x0[1]]
        self.disperser.D = D
        lambdas = self.disperser.grating_pixel_to_lambda(pixels, x0=new_x0, order=1)
        self.simulate_without_atmosphere(lambdas)
        atmospheric_transmission = self.atmosphere.simulate(ozone, pwv, aerosols)(lambdas)
        # np.savetxt('atmospheric_transmission_20170530_130.txt', np.array([lambdas, atmospheric_transmission(lambdas)]).T)
        self.data *= A1 * atmospheric_transmission
        self.err *= A1 * atmospheric_transmission
        # Now add the systematics
        if reso > 1:
            self.data = fftconvolve_gaussian(self.data, reso)
            self.err = np.sqrt(np.abs(fftconvolve_gaussian(self.err ** 2, reso)))
        if A2 > 0.:
            sim_conv = interp1d(lambdas, self.data, kind="linear", bounds_error=False, fill_value=(0, 0))
            err_conv = interp1d(lambdas, self.err, kind="linear", bounds_error=False, fill_value=(0, 0))
            self.model = lambda x: sim_conv(x) + A2 * sim_conv(x / 2)
            self.model_err = lambda x: np.sqrt(np.abs((err_conv(x)) ** 2 + (0.5 * A2 * err_conv(x / 2)) ** 2))
            self.data = self.model(lambdas)
            self.err = self.model_err(lambdas)
        # now we include effects related to the wrong extraction of the spectrum:
        # wrong estimation of the order 0 position and wrong DISTANCE2CCD
        pixels = np.arange(0, parameters.CCD_IMSIZE) - self.x0[0]
        self.disperser.D = parameters.DISTANCE2CCD
        self.lambdas = self.disperser.grating_pixel_to_lambda(pixels, self.x0, order=1)
        self.model = interp1d(self.lambdas, self.data, kind="linear", bounds_error=False, fill_value=(0, 0))
        self.model_err = interp1d(self.lambdas, self.err, kind="linear", bounds_error=False, fill_value=(0, 0))
        return self.lambdas, self.model, self.model_err


# ----------------------------------------------------------------------------------
class SpectrogramSimulation(Spectrum):
    """ SpectrumSim class used to store information and methods
    relative to spectrum simulation.
    """

    # ---------------------------------------------------------------------------
    def __init__(self, spectrum, atmosphere, telescope, disperser):
        Spectrum.__init__(self)
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        self.disperser = disperser
        self.telescope = telescope
        self.atmosphere = atmosphere
        self.pixels_x = np.arange(self.chromatic_psf.Nx).astype(int)
        self.pixels_y = np.arange(self.chromatic_psf.Ny).astype(int)
        self.lambdas = None
        self.data = None
        self.err = None
        self.model = lambda x, y: np.zeros((x.size, y.size))
        self.psf_cube = None
        self.fhcube = None
        self.fix_psf_cube = False

    def simulate_spectrum(self, lambdas, ozone, pwv, aerosols):
        """
        Simulate the spectrum of the object and return the result in flam units.

        Parameters
        ----------
        lambdas: array_like
            The wavelength array in nm.
        ozone: float
            Ozone parameter for Libradtran.
        pwv: float
            Precipitable Water Vapor parameter for Libradtran.
        aerosols: float
            Aerosols parameter for Libradtran.

        Returns
        -------
        spectrum: array_like
            The spectrum array in flam units.
        spectrum_err: array_like
            The spectrum uncertainty array in flam units.

        """
        spectrum_err = np.zeros_like(lambdas)
        spectrum = self.disperser.transmission(lambdas)
        telescope_transmission = self.telescope.transmission(lambdas)
        spectrum *= telescope_transmission
        spectrum *= self.target.sed(lambdas)
        spectrum *= self.atmosphere.simulate(ozone, pwv, aerosols)(lambdas)
        spectrum_err = np.zeros_like(spectrum)
        idx = np.where(telescope_transmission > 0)[0]
        spectrum_err[idx] = self.telescope.transmission_err(lambdas)[idx] / telescope_transmission[idx] * spectrum[idx]
        return spectrum, spectrum_err

    def simulate_psf(self, psf_poly_params, angle):
        profile_params = self.chromatic_psf.from_poly_params_to_profile_params(psf_poly_params, force_positive=True)
        self.chromatic_psf.fill_table_with_profile_params(profile_params)
        # self.chromatic_psf.from_profile_params_to_shape_params(profile_params)
        # self.chromatic_psf.table['Dx'] = np.arange(self.spectrogram_Nx) - self.spec
        self.chromatic_psf.table['Dy_mean'] = 0
        self.chromatic_psf.table['Dy'] = np.copy(self.chromatic_psf.table['x_mean'])
        # derotate
        # self.my_logger.warning(f"\n\tbefore\n {self.chromatic_psf.table[['Dx_rot', 'Dx', 'Dy', 'Dy_mean']][:5]} {angle}")
        self.chromatic_psf.rotate_table(-angle)
        # self.my_logger.warning(f"\n\tafter\n {self.chromatic_psf.table[['Dx_rot', 'Dx', 'Dy', 'Dy_mean']][:5]}  {angle}")

    def simulate_dispersion(self, D, shift_x, shift_y):
        new_x0 = [self.x0[0] - shift_x, self.x0[1] - shift_y]
        distance = np.array(self.chromatic_psf.get_distance_along_dispersion_axis(shift_x=shift_x, shift_y=shift_y))
        self.disperser.D = D
        lambdas = self.disperser.grating_pixel_to_lambda(distance, x0=new_x0)
        dispersion_law = (self.chromatic_psf.table['Dx'] - shift_x) + 1j * (self.chromatic_psf.table['Dy'] - shift_y)
        return lambdas, dispersion_law

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
        >>> from spectractor.extractor.psf import  ChromaticPSF1D
        >>> spectrum, telescope, disperser, target = SimulatorInit('outputs/reduc_20170530_130_spectrum.fits')
        >>> airmass = spectrum.header['AIRMASS']
        >>> pressure = spectrum.header['OUTPRESS']
        >>> temperature = spectrum.header['OUTTEMP']
        >>> atmosphere = Atmosphere(airmass, pressure, temperature)
        >>> psf_poly_params = spectrum.chromatic_psf.from_table_to_poly_params()
        >>> spec = SpectrogramSimulation(spectrum, atmosphere, telescope, disperser)
        >>> lambdas, data, err = spec.simulate(psf_poly_params=psf_poly_params, angle=spec.rotation_angle)
        """
        import time
        start = time.time()
        self.simulate_psf(psf_poly_params, angle)
        self.my_logger.debug(f'\n\tTime after simulate PSF: {time.time()-start}')
        start = time.time()
        lambdas, dispersion_law = self.simulate_dispersion(D, shift_x, shift_y)
        self.my_logger.debug(f'\n\tTime after simulate disp: {time.time()-start}')
        start = time.time()
        spectrum, spectrum_err = self.simulate_spectrum(lambdas, ozone, pwv, aerosols)
        self.my_logger.debug(f'\n\tTime after simulate spec: {time.time()-start}')
        start = time.time()
        nlbda = lambdas.size

        # PSF cube
        nima = self.spectrogram_Ny
        y, x = FA.create_coords((nima, nima), starts='auto')  # (ny, nx)
        if self.psf_cube is None or not self.fix_psf_cube:
            cube = np.zeros((nlbda, nima, nima))
            shape_params = np.array([self.chromatic_psf.table[name] for name in PSF2D.param_names[3:]]).T
            for l in range(nlbda):
                ima = PSF2D.evaluate(x, y, 1, 0, 0, *shape_params[l])
                ima /= ima.sum()  # Flux normalization: the flux should go in the spectrum
                cube[l] = np.copy(ima)
            self.psf_cube = cube
        else:
            cube = self.psf_cube
        self.my_logger.debug(f'\n\tTime after cube: {time.time()-start}')
        start = time.time()

        # Extended image (nima × nlbda) has to be perfecty centered
        ny, nx = (nima, nlbda)  # Rectangular minimal embedding
        hcube = FA.embed_array(cube, (nlbda, ny, nx))

        # Generate slitless-spectroscopy image from Fourier analysis
        if self.fhcube is None or not self.fix_psf_cube:
            uh, vh, fhcube = F.fft_cube(hcube)  # same shape as hima
            self.uh = uh
            self.vh = vh
            self.fhcube = fhcube
        else:
            uh = self.uh
            vh = self.vh
            fhcube = self.fhcube
        self.my_logger.debug(f'\n\tTime after fourier cube: {time.time()-start}')
        start = time.time()
        r0 = (self.spectrogram_x0 - self.spectrogram_Nx / 2 - shift_x) \
             + 1j * (self.spectrogram_y0 - self.spectrogram_Ny / 2 - shift_y)
        fdima0 = F.disperse_fcube(uh, vh, fhcube, spectrum, dispersion_law, r0)  # FT
        self.my_logger.debug(f'\n\tTime after simulate after fourier: {time.time()-start}')
        start = time.time()

        # Dispersed image (noiseless)
        dima0 = F.ifft_image(fdima0)
        self.my_logger.debug(f'\n\tTime after simulate inverse fourier: {time.time()-start}')
        start = time.time()

        # Going to observable spectrum: must convert units (ie multiply by dlambda)
        self.data = A1 * dima0
        self.lambdas = lambdas
        self.lambdas_binwidths = np.gradient(lambdas)
        self.convert_from_flam_to_ADUrate()
        self.data += self.spectrogram_bgd
        self.err = np.zeros_like(self.data)
        # Now add the systematics
        # if reso > 1:
        #     self.data = fftconvolve_gaussian(self.data, reso)
        #     self.err = np.sqrt(np.abs(fftconvolve_gaussian(self.err ** 2, reso)))
        if A2 > 0.:
            pass
            # sim_conv = interp1d(lambdas, self.data, kind="linear", bounds_error=False, fill_value=(0, 0))
            # err_conv = interp1d(lambdas, self.err, kind="linear", bounds_error=False, fill_value=(0, 0))
            # self.model = lambda x: sim_conv(x) + A2 * sim_conv(x / 2)
            # self.model_err = lambda x: np.sqrt(np.abs((err_conv(x)) ** 2 + (0.5 * A2 * err_conv(x / 2)) ** 2))
            # self.data = self.model(lambdas)
            # self.err = self.model_err(lambdas)
        # now we include effects related to the wrong extraction of the spectrum:
        # wrong estimation of the order 0 position and wrong DISTANCE2CCD
        distance = self.chromatic_psf.get_distance_along_dispersion_axis()
        self.disperser.D = parameters.DISTANCE2CCD
        self.lambdas = self.disperser.grating_pixel_to_lambda(distance, self.x0, order=1)
        # self.model = interp1d(self.lambdas, self.data, kind="linear", bounds_error=False, fill_value=(0, 0))
        # self.model_err = interp1d(self.lambdas, self.err, kind="linear", bounds_error=False, fill_value=(0, 0))
        self.my_logger.debug(f'\n\tTime after conclusions: {time.time()-start}')
        start = time.time()
        if parameters.DEBUG:
            fig, ax = plt.subplots(2, 1, sharex="all", figsize=(12, 4))
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


# ----------------------------------------------------------------------------------
class SpectrumSimGrid():
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

    # ----------------------------------------------------------------------------
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
        self.spectragrid[1:, self.atmgrid.index_atm_data:] = self.atmgrid.atmgrid[1:,
                                                             self.atmgrid.index_atm_data:] * all_transm
        return self.spectragrid

    # ---------------------------------------------------------------------------
    def plot_spectra(self):
        plt.figure()
        counts = self.spectragrid[1:, self.atmgrid.index_atm_count]
        for count in counts:
            plt.plot(parameters.LAMBDAS, self.spectragrid[int(count), self.atmgrid.index_atm_data:])
        plt.grid()
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Flux [ADU/s]")
        plt.title("Spectra for Atmospheric variations")
        if parameters.DISPLAY: plt.show()

    # ---------------------------------------------------------------------------
    def plot_spectra_img(self):
        plt.figure()
        img = plt.imshow(self.spectragrid[1:, self.atmgrid.index_atm_data:], origin='lower', cmap='jet')
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Simulation number")
        plt.title("Spectra for atmospheric variations")
        cbar = plt.colorbar(img)
        cbar.set_label("Flux [ADU/s]")
        plt.grid(True)
        if parameters.DISPLAY: plt.show()

    def save_spectra(self, filename):

        if filename != "":
            self.filename = filename

        if self.filename == "":
            return
        else:

            hdu = fits.PrimaryHDU(self.spectragrid, header=self.header)
            hdu.writeto(self.filename, overwrite=True)
            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info('\n\tSPECTRA.save atm-file=%s' % (self.filename))


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
    if parameters.DEBUG:
        infostring = '\n\t ========= Telescope transmission :  ==============='
        my_logger.info(infostring)
        telescope.plot_transmission()

    # DISPERSER TRANSMISSION
    # ------------------------
    if not isinstance(spectrum.disperser, str):
        disperser = spectrum.disperser
    else:
        disperser = Hologram(spectrum.disperser)
    if parameters.DEBUG:
        infostring = '\n\t ========= Disperser transmission :  ==============='
        my_logger.info(infostring)
        disperser.plot_transmission()

    # STAR SPECTRUM
    # ------------------------
    target = spectrum.target
    if not isinstance(spectrum.target, str):
        target = spectrum.target
    else:
        target = Target(spectrum.target)
    if parameters.DEBUG:
        infostring = '\n\t ========= SED : %s  ===============' % target.label
        my_logger.info(infostring)
        target.plot_spectra()

    return spectrum, telescope, disperser, target


# ----------------------------------------------------------------------------------
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
        infostring = '\n\t ========= Spectra simulation :  ==============='
        spectrum_simulation.plot_spectrum()
    return spectrum_simulation


# ----------------------------------------------------------------------------------
def SpectrogramSimulatorCore(spectrum, telescope, disperser, airmass=1.0, pressure=800, temperature=10,
                             pwv=5, ozone=300, aerosols=0.05, A1=1.0, A2=0.,
                             D=parameters.DISTANCE2CCD, shift_x=0., shift_y=0., angle=0., psf_poly_params=None):
    """ SimulatorCore
    Main function to evaluate several spectra
    A grid of spectra will be produced for a given target, airmass and pressure
    """
    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SPECTROGRAMSIMULATOR core program')
    # SIMULATE ATMOSPHERE
    # -------------------
    atmosphere = Atmosphere(airmass, pressure, temperature)

    # SPECTRUM SIMULATION
    # --------------------
    spectrogram_simulation = SpectrogramSimulation(spectrum, atmosphere, telescope, disperser)
    spectrogram_simulation.simulate(A1, A2, ozone, pwv, aerosols, psf_poly_params, D, shift_x, shift_y, angle)
    return spectrogram_simulation


# ----------------------------------------------------------------------------------
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
        atmgrid, header = atm.load_file(output_atmfilename)
    else:
        atmgrid = atm.compute()
        header = atm.save_file(filename=output_atmfilename)
        # libradtran.clean_simulation_directory()
    if parameters.VERBOSE:
        infostring = '\n\t ========= Atmospheric simulation :  ==============='
        my_logger.info(infostring)
        atm.plot_transmission()  # plot all atm transp profiles
        atm.plot_transm_img()  # plot 2D image summary of atm simulations

    # SPECTRA-GRID
    # -------------
    # in any case we re-calculate the spectra in case of change of spectrum function
    spectra = SpectrumSimGrid(spectrum, atm, telescope, disperser, target, header)
    spectragrid = spectra.compute()
    spectra.save_spectra(output_filename)
    if parameters.VERBOSE:
        infostring = '\n\t ========= Spectra simulation :  ==============='
        spectra.plot_spectra()
        spectra.plot_spectra_img()
    # ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
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
    spectrum_simulation.header['OZONE'] = ozone
    spectrum_simulation.header['PWV'] = pwv
    spectrum_simulation.header['VAOD'] = aerosols
    spectrum_simulation.header['A1'] = A1
    spectrum_simulation.header['A2'] = A2
    spectrum_simulation.header['RESO'] = reso
    spectrum_simulation.header['D2CCD'] = D
    spectrum_simulation.header['X0SHIFT'] = shift
    output_filename = filename.replace('spectrum', 'sim')
    if outputdir != "":
        base_filename = filename.split('/')[-1]
        output_filename = os.path.join(outputdir, base_filename.replace('spectrum', 'sim'))
    spectrum_simulation.save_spectrum(output_filename, overwrite=True)

    return spectrum_simulation


# ----------------------------------------------------------------------------------
def SpectrogramSimulator(filename, outputdir="", pwv=5, ozone=300, aerosols=0.05, A1=1., A2=0.,
                         D=parameters.DISTANCE2CCD, shift_x=0., shift_y=0., angle=0., psf_poly_params=None):
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

    spectrogram_simulation = SpectrogramSimulatorCore(spectrum, telescope, disperser, target, airmass, pressure,
                                                      temperature, pwv, ozone, aerosols,
                                                      D=D, shift_x=shift_x,
                                                      shift_y=shift_y, angle=angle, psf_poly_params=psf_poly_params)

    # Save the spectrum
    spectrogram_simulation.header['OZONE'] = ozone
    spectrogram_simulation.header['PWV'] = pwv
    spectrogram_simulation.header['VAOD'] = aerosols
    spectrogram_simulation.header['A1'] = A1
    spectrogram_simulation.header['A2'] = A2
    spectrogram_simulation.header['D2CCD'] = D
    spectrogram_simulation.header['X0SHIFT'] = shift_x
    spectrogram_simulation.header['Y0SHIFT'] = shift_y
    spectrogram_simulation.header['ROTANGLE'] = angle
    output_filename = filename.replace('spectrum', 'sim')
    if outputdir != "":
        base_filename = filename.split('/')[-1]
        output_filename = os.path.join(outputdir, base_filename.replace('spectrum', 'sim'))
    # spectrogram_simulation.save_spectrum(output_filename, overwrite=True)

    return spectrogram_simulation

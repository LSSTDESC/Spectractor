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

from spectractor.pipeline.images import Image
from spectractor.pipeline.spectroscopy import Spectrum
from spectractor.pipeline.dispersers import Hologram
from spectractor.pipeline.targets import Target
from spectractor.tools import fftconvolve_gaussian, ensure_dir
import spectractor.parameters as parameters

import spectractor.simulation.libradtran as libradtran
from spectractor.simulation.throughput import Throughput


class Atmosphere(object):
    """
    Atmosphere(): 
        class to simulate an atmospheric transmission calling libradtran
    Args:
        airmass (:obj:`float`): airmass of the target
        pressure (:obj:`float`): pressure of the atmosphere 
        temperature (:obj:`float`): temperature of the atmosphere 
    """

    def __init__(self, airmass, pressure, temperature):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
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

        return self.transmission

    def plot_transmission(self):
        plt.figure()
        plt.plot(parameters.LAMBDAS, self.transmission(parameters.LAMBDAS),
                 label='z=%4.2f, P=%4.2f, T=%4.2f' % (self.airmass, self.pressure, self.temperature))
        plt.grid()
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Atmospheric transparency")
        plt.legend()
        plt.show()


# ----------------------------------------------------------------------------------
class AtmosphereGrid(Atmosphere):
    """
    Atmosphere(): 
        class to simulate series of atmospheres calling libradtran
    Args:
        airmass (:obj:`float`): airmass of the target
        pressure (:obj:`float`): pressure of the atmosphere 
        temperature (:obj:`float`): temperature of the atmosphere 
        filenamedata (:obj:`strt`): XXXXXXXXXX    
        filename (:obj:`strt`): atmospheric grid file name to load   
    """

    def __init__(self, data_filename, filename="", airmass=1., pressure=800., temperature=10.):
        Atmosphere.__init__(self, airmass, pressure, temperature)
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.data_filename = data_filename
        # ------------------------------------------------------------------------
        # Definition of data format for the atmospheric grid
        # -----------------------------------------------------------------------------

        # specify parameters for the atmospheric grid

        # aerosols
        # NB_AER_POINTS=20
        self.NB_AER_POINTS = 5
        self.AER_MIN = 0.
        self.AER_MAX = 0.1

        # ozone
        # NB_OZ_POINTS=5
        self.NB_OZ_POINTS = 5
        self.OZ_MIN = 200
        self.OZ_MAX = 400

        # pwv
        # NB_PWV_POINTS=11
        self.NB_PWV_POINTS = 5
        self.PWV_MIN = 0.
        self.PWV_MAX = 10.

        # definition of the grid
        self.AER_Points = np.linspace(self.AER_MIN, self.AER_MAX, self.NB_AER_POINTS)
        self.OZ_Points = np.linspace(self.OZ_MIN, self.OZ_MAX, self.NB_OZ_POINTS)
        self.PWV_Points = np.linspace(self.PWV_MIN, self.PWV_MAX, self.NB_PWV_POINTS)

        # total number of points
        self.NB_ATM_POINTS = self.NB_AER_POINTS * self.NB_OZ_POINTS * self.NB_PWV_POINTS

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

        self.NB_atm_HEADER = 5
        self.NB_atm_DATA = len(parameters.LAMBDAS) - 1

        # create the numpy array that will contains the atmospheric grid
        self.atmgrid = np.zeros((self.NB_ATM_POINTS + 1, self.NB_atm_HEADER + self.NB_atm_DATA))
        self.atmgrid[0, self.index_atm_data:] = parameters.LAMBDAS
        self.header = fits.Header()
        if filename != "":
            self.loadfile(filename)

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
                    self.atmgrid[count, self.index_atm_data:] = transm  # each of atmospheric transmission
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
        plt.show()

    def plot_transm_img(self):
        plt.figure()
        img = plt.imshow(self.atmgrid[1:, self.index_atm_data:], origin='lower', cmap='jet')
        plt.grid(True)
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Simulation number")
        plt.title(" Atmospheric variations")
        cbar = plt.colorbar(img)
        cbar.set_label('Atmospheric transmission')
        plt.show()

    def savefile(self, filename=""):

        hdr = fits.Header()

        if filename != "":
            self.filename = filename

        if self.filename == "":
            infostring = '\n\t Atmosphere:savefile no settings file given ...'
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

            if parameters.VERBOSE:
                print(hdr)

            hdu = fits.PrimaryHDU(self.atmgrid, header=hdr)
            hdu.writeto(self.filename, overwrite=True)
            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info('\n\tAtmosphere.save atm-file=%s' % (self.filename))

            return hdr

    def loadfile(self, filename):

        if filename != "":
            self.filename = filename

        if self.filename == "":
            infostring = '\n\t Atmosphere:loadfile no settings file given ...'
            self.my_logger.info(infostring)

            return
        else:

            hdu = fits.open(self.filename)
            hdr = hdu[0].header
            self.header = hdr

            # hdr['ATMSIM'] = "libradtran"
            # hdr['SIMVERS'] = "2.0.1"
            self.data_filename = hdr['DATAFILE']
            # hdr['SIMUFILE']=os.path.basename(self.filename)

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
                self.my_logger.info('\n\tAtmosphere.load atm-file=%s' % (self.filename))

            # interpolate the grid
            self.lambdas = self.atmgrid[0, self.index_atm_data:]
            self.model = RegularGridInterpolator((self.lambdas, self.OZ_Points, self.PWV_Points, self.AER_Points), (
                self.atmgrid[1:, self.index_atm_data:].reshape(self.NB_AER_POINTS, self.NB_PWV_POINTS, self.NB_OZ_POINTS,
                                                          len(self.lambdas))).T, bounds_error=False, fill_value=0)

            return self.atmgrid, self.header

    def simulate(self, ozone, pwv, aerosols):
        """ first ozone, second pwv, last aerosols, to respect order of loops when generating the grid"""
        ones = np.ones_like(self.lambdas)
        points = np.array([self.lambdas, ozone * ones, pwv * ones, aerosols * ones]).T
        atm = self.model(points)
        self.transmission = interp1d(self.lambdas, atm, kind='linear', bounds_error=False, fill_value=(0, 0))
        return self.transmission(self.lambdas)


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
        filename (:obj:`str`): path to the data filename (for info only)
        """

        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.filtername = filtername
        self.load_transmission()

    # ---------------------------------------------------------------------------
    def load_transmission(self):
        """
        load_transmission(self) :
            load the telescope transmission
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
    def __init__(self, spectrum, atmosphere, telescope, disperser, reso=None):
        """
        Args:
            filename (:obj:`str`): path to the image
        """
        Spectrum.__init__(self)
        for k, v in list(spectrum.__dict__.items()):
            self.__dict__[k] = copy.copy(v)
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.disperser = disperser
        self.telescope = telescope
        self.atmosphere = atmosphere
        self.lambdas = None
        self.data = None
        self.err = None
        self.reso = reso
        self.model = lambda x: np.zeros_like(x)

    # ----------------------------------------------------------------------------

    def simulate_without_atmosphere(self, lambdas):
        self.lambdas = lambdas
        self.err = np.zeros_like(lambdas)
        self.lambda_binwidths = np.gradient(lambdas)
        self.data = self.disperser.transmission(lambdas)
        self.data *= self.telescope.transmission(lambdas)
        self.data *= self.target.sed(lambdas)
        self.err = np.zeros_like(self.data)
        idx = np.where(self.telescope.transmission(lambdas) > 0)[0]
        self.err[idx] = self.telescope.transmission_err(lambdas)[idx] / self.telescope.transmission(lambdas)[idx] * \
                        self.data[idx]
        # self.data *= self.lambdas*self.lambda_binwidths
        return self.data, self.err

    # ----------------------------------------------------------------------------
    def simulate(self, lambdas):
        self.simulate_without_atmosphere(lambdas)
        self.data *= self.atmosphere.transmission(lambdas)
        self.err *= self.atmosphere.transmission(lambdas)
        # self.data = all_transm*Factor
        if self.reso is not None:
            self.data = fftconvolve_gaussian(self.data, self.reso)
            self.err = np.sqrt(fftconvolve_gaussian(self.err ** 2, self.reso))
        self.model = interp1d(lambdas, self.data, kind="linear", bounds_error=False, fill_value=(0, 0))
        return self.data, self.err
    # ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
class SpectrumSimGrid():
    """ SpectrumSim class used to store information and methods
    relative to spectrum simulation.
    """

    # ---------------------------------------------------------------------------
    def __init__(self, spectrum, atmgrid, telescope, disperser, target, header, filename=""):
        """
        Args:
            filename (:obj:`str`): path to the image
        """
        self.my_logger = parameters.set_logger(self.__class__.__name__)

        self.spectrum = spectrum
        self.header = spectrum.header
        self.disperser = disperser
        self.target = target
        self.telescope = telescope

        self.atmgrid = atmgrid
        self.lambdas = self.atmgrid.atmgrid[0, self.atmgrid.index_atm_data:]
        self.lambda_binwidths = np.gradient(self.lambdas)
        self.spectragrid = None

        self.filename = ""
        if filename != "":
            self.filename = filename
            self.load_spectrum(filename)

        if parameters.VERBOSE:
            print(self.header)

    # ----------------------------------------------------------------------------
    def compute(self):
        sim = SpectrumSimulation(self.spectrum, self.atmgrid.atmgrid, self.telescope, self.disperser)
        # product of all sed and transmission except atmosphere
        all_transm, all_transm_err = sim.simulate_without_atmosphere(self.lambdas)
        # copy atmospheric grid parameters into spectra grid
        self.spectragrid = np.zeros_like(self.atmgrid.atmgrid)
        self.spectragrid[0, self.atmgrid.index_atm_data:] = self.lambdas
        self.spectragrid[:, self.atmgrid.index_atm_count:self.atmgrid.index_atm_data] = \
            self.atmgrid.atmgrid[:, self.atmgrid.index_atm_count:self.atmgrid.index_atm_data]
        # Is broadcasting working OK ?
        self.spectragrid[1:, self.atmgrid.index_atm_data:] = self.atmgrid.atmgrid[1:, self.atmgrid.index_atm_data:] * all_transm
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
        plt.show()

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
        plt.show()

    # ---------------------------------------------------------------------------
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
    # ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
def SimulatorInit(filename):
    """ SimulatorInit
    Main function to simulate several spectra 
    A grid of spectra will be produced for a given target, airmass and pressure

    Args:
        filename (:obj:`str`): filename of the image (data)
    """
    my_logger = parameters.set_logger(__name__)
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
def SimulatorCore(spectrum, telescope, disperser, target, lambdas, airmass=1.0, pressure=800, temperature=10,
                  pwv=5, ozone=300, aerosols=0.05, reso=None):
    """ SimulatorCore
    Main function to simulate several spectra 
    A grid of spectra will be produced for a given target, airmass and pressure

    Args:
        spectrum (:obj:`Spectrum`): data spectrum object
        telescope (:obj:`TelescopeTransmission`): telescope transmission
        disperer (:obj:`Hologram`): disperser object
        target (:obj:`Target`): target object
        lambdas (:obj:`float`): wavelength array (in nm)
        airmass (:obj:`float`): airmass of the target
        pressure (:obj:`float`): pressure in hPa
        temperature (:obj:`float`): temperature in celsius
        pwv (:obj:`float`): pressure water vapor
        ozone (:obj:`float`): ozone quantity
        aerosols (:obj:`float`): VAOD Vertical Aerosols Optical Depth        
        reso (:obj:`float`): width of gaussian in nm to convolve with spectrum
    """
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTOR core program')
    # SIMULATE ATMOSPHERE
    # -------------------
    atmosphere = Atmosphere(airmass, pressure, temperature)
    atmosphere.simulate(ozone, pwv, aerosols)
    if parameters.DEBUG:
        infostring = '\n\t ========= Atmospheric simulation :  ==============='
        my_logger.info(infostring)
        atmosphere.plot_transmission()  # plot all atm transp profiles

    # SPECTRUM SIMULATION  
    # --------------------
    spectrum_simulation = SpectrumSimulation(spectrum, atmosphere, telescope, disperser, reso=reso)
    spectrum_simulation.simulate(lambdas)
    if parameters.DEBUG:
        infostring = '\n\t ========= Spectra simulation :  ==============='
        spectrum_simulation.plot_spectrum(nofit=True)

    return spectrum_simulation


# ----------------------------------------------------------------------------------
def SimulatorSimGrid(filename, outputdir):
    """ SimulatorSimGrid
    Main function to simulate several spectra 
    A grid of spectra will be produced for a given target, airmass and pressure

    Args:
        filename (:obj:`str`): filename of the image (data)
        outputdir (:obj:`str`): path to the output directory
        
    """
    my_logger = parameters.set_logger(__name__)
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

    # test if file already exists
    if os.path.exists(output_atmfilename) and os.path.getsize(output_atmfilename) > 20000:
        filesize = os.path.getsize(output_atmfilename)
        infostring = " atmospheric simulation file %s of size %d already exists, thus load it ..." % (
            output_atmfilename, filesize)
        my_logger.info(infostring)
        atmgrid, header = atm.loadfile(output_atmfilename)
    else:
        atmgrid = atm.compute()
        header = atm.savefile(filename=output_atmfilename)
        libradtran.clean_simulation_directory()
    if parameters.VERBOSE:
        infostring = '\n\t ========= Atmospheric simulation :  ==============='
        my_logger.info(infostring)
        atm.plot_transmission()  # plot all atm transp profiles
        atm.plot_transm_img()  # plot 2D image summary of atm simulations

    # SPECTRA-GRID  
    # -------------
    # in any case we re-calculate the spectra in case of change of transmission function
    spectra = SpectrumSimGrid(spectrum, atm, telescope, disperser, target, header)
    spectragrid = spectra.compute()
    spectra.save_spectra(output_filename)
    if parameters.VERBOSE:
        infostring = '\n\t ========= Spectra simulation :  ==============='
        spectra.plot_spectra()
        spectra.plot_spectra_img()
    # ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
def Simulator(filename, lambdas, outputdir="", pwv=5, ozone=300, aerosols=0.05, reso=None):
    """ Simulator
    Main function to simulate several spectra 
    A grid of spectra will be produced for a given target, airmass and pressure

    Args:
        filename (:obj:`str`): filename of the image (data)
        outputdir (:obj:`str`): path to the output directory
        pwv (:obj:`float`): pressure water vapor
        ozone (:obj:`float`): ozone quantity
        aerosols (:obj:`float`): VAOD Vertical Aerosols Optical Depth        
        reso (:obj:`float`): width of gaussian in nm to convolve with spectrum
    """
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTORSIM')
    # Initialisation
    spectrum, telescope, disperser, target = SimulatorInit(filename)

    # SIMULATE SPECTRUM
    # -------------------
    airmass = spectrum.header['AIRMASS']
    pressure = spectrum.header['OUTPRESS']
    temperature = spectrum.header['OUTTEMP']

    spectrum_simulation = SimulatorCore(spectrum, telescope, disperser, target, lambdas, airmass, pressure,
                                        temperature, pwv, ozone, aerosols, reso=reso)

    # Save the spectrum
    if outputdir != "":
        base_filename = filename.split('/')[-1]
        output_filename = os.path.join(outputdir, base_filename.replace('spectrum', 'sim'))
        spectrum_simulation.header['OZONE'] = ozone
        spectrum_simulation.header['PWV'] = pwv
        spectrum_simulation.header['VAOD'] = aerosols
        spectrum_simulation.header['reso'] = reso
        spectrum_simulation.save_spectrum(output_filename, overwrite=True)

    return spectrum_simulation


if __name__ == "__main__":
    # import commands, string,  time
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--debug", dest="debug", action="store_true",
                      help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="Enter verbose (print more stuff).", default=False)
    parser.add_option("-o", "--output_directory", dest="output_directory", default="test/",
                      help="Write results in given output directory (default: ./tests/).")
    (opts, args) = parser.parse_args()

    parameters.VERBOSE = opts.verbose

    if opts.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    filename = "../Spectractor/outputs/data_30may17/reduc_20170530_134_spectrum.fits"

    # spectrum_simulation = Simulator(filename,lambdas=parameters.LAMBDAS,pwv=5,ozone=300,aerosols=0.05)
    SimulatorSimGrid(filename, opts.output_directory)

    atmgrid = AtmosphereGrid(filename, "../Spectractor/outputs/data_30may17/reduc_20170530_134_atmsim.fits")
    atm = Atmosphere(atmgrid.airmass, atmgrid.pressure, atmgrid.temperature)

    fig = plt.figure()
    for i in range(5):
        aerosols = i * 0.01 + 0.005
        a = atmgrid.simulate(300, 5, aerosols )
        plt.plot(atmgrid.lambdas, a, label=f"aerosols = {aerosols}")
    plt.legend()
    plt.xlabel('$\lambda$ [nm]')
    plt.ylabel('Transmission')
    plt.show()

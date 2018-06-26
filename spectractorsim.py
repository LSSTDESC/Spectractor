"""
spectractorsim
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

import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import sys, os
import copy
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

from scipy.interpolate import interp1d, griddata, RegularGridInterpolator

from spectractor.tools import *
from spectractor.dispersers import *
from spectractor.targets import *
from spectractor.images import *
from spectractor.spectroscopy import *
import spectractor.parameters as parameters

# ----------------------------------------------------------------------------
# where is spectractorsim
# ----------------------------------------------------------------------------
spectractorsim_path = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Libraries to interface LibRadTran and CTIO 0.9m telescope transparencies
# -------------------------------------------------------------------------

import libradtran
import libCTIOTransm as ctio

# ------------------------------------------------------------------------
# Definition of data format for the atmospheric grid
# -----------------------------------------------------------------------------
WLMIN = parameters.LAMBDA_MIN  # Minimum wavelength : PySynPhot works with Angstrom
WLMAX = parameters.LAMBDA_MAX  # Minimum wavelength : PySynPhot works with Angstrom
WL = np.arange(WLMIN, WLMAX, 1)  # Array of wavelength in Angstrom

# specify parameters for the atmospheric grid

# aerosols
# NB_AER_POINTS=20
NB_AER_POINTS = 5
AER_MIN = 0.
AER_MAX = 0.1

# ozone
# NB_OZ_POINTS=5
NB_OZ_POINTS = 5
OZ_MIN = 200
OZ_MAX = 400

# pwv
# NB_PWV_POINTS=11
NB_PWV_POINTS = 5
PWV_MIN = 0.
PWV_MAX = 10.

# definition of the grid
AER_Points = np.linspace(AER_MIN, AER_MAX, NB_AER_POINTS)
OZ_Points = np.linspace(OZ_MIN, OZ_MAX, NB_OZ_POINTS)
PWV_Points = np.linspace(PWV_MIN, PWV_MAX, NB_PWV_POINTS)

# total number of points
NB_ATM_POINTS = NB_AER_POINTS * NB_OZ_POINTS * NB_PWV_POINTS

#  column 0 : count number
#  column 1 : aerosol value
#  column 2 : pwv value
#  column 3 : ozone value
#  column 4 : data start 
#
index_atm_count = 0
index_atm_aer = 1
index_atm_pwv = 2
index_atm_oz = 3
index_atm_data = 4

NB_atm_HEADER = 5
NB_atm_DATA = len(WL) - 1

MINFILESIZE = 20000


# ----------------------------------------------------------------------------------
class Atmosphere(object):
    """
    Atmosphere(): 
        class to simulate an atmospheric transmission calling libradtran
    Args:
        airmass (:obj:`float`): airmass of the target
        pressure (:obj:`float`): pressure of the atmosphere 
        temperature (:obj:`float`): temperature of the atmosphere 
    """

    # ---------------------------------------------------------------------------
    def __init__(self, airmass, pressure, temperature):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.airmass = airmass
        self.pressure = pressure
        self.temperature = temperature
        self.transmission = lambda x: np.ones_like(x).astype(float)

    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    def plot_transmission(self):
        plt.figure()
        plt.plot(WL, self.transmission(WL),
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

    # ---------------------------------------------------------------------------
    def __init__(self, filenamedata, filename="", airmass=1., pressure=800., temperature=10.):
        Atmosphere.__init__(self, airmass, pressure, temperature)
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.filenamedata = filenamedata
        # create the numpy array that will contains the atmospheric grid    
        self.atmgrid = np.zeros((NB_ATM_POINTS + 1, NB_atm_HEADER + NB_atm_DATA))
        self.atmgrid[0, index_atm_data:] = WL
        self.header = fits.Header()
        if filename != "":
            self.loadfile(filename)

    # ---------------------------------------------------------------------------
    def compute(self):
        # first determine the length
        if parameters.VERBOSE or parameters.DEBUG:
            self.my_logger.info('\n\tAtmosphere simulations for z=%4.2f, P=%4.2f, T=%4.2f, for data-file=%s ' % (
            self.airmass, self.pressure, self.temperature, self.filenamedata))

        count = 0
        for aer in AER_Points:
            for pwv in PWV_Points:
                for oz in OZ_Points:
                    count += 1
                    # fills headers info in the numpy array
                    self.atmgrid[count, index_atm_count] = count
                    self.atmgrid[count, index_atm_aer] = aer
                    self.atmgrid[count, index_atm_pwv] = pwv
                    self.atmgrid[count, index_atm_oz] = oz
                    transmission = super(AtmosphereGrid, self).simulate(oz, pwv, aer)
                    transm = transmission(WL)
                    self.atmgrid[count, index_atm_data:] = transm  # each of atmospheric transmission

        return self.atmgrid

    # ---------------------------------------------------------------------------
    def plot_transmission(self):
        plt.figure()
        counts = self.atmgrid[1:, index_atm_count]
        for count in counts:
            plt.plot(WL, self.atmgrid[int(count), index_atm_data:])
        plt.grid()
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Atmospheric transmission")
        plt.title("Atmospheric variations")
        plt.show()

    # ---------------------------------------------------------------------------
    def plot_transm_img(self):
        plt.figure()
        img = plt.imshow(self.atmgrid[1:, index_atm_data:], origin='lower', cmap='jet')
        plt.grid(True)
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Simulation number")
        plt.title(" Atmospheric variations")
        cbar = plt.colorbar(img)
        cbar.set_label('Atmospheric transmission')
        plt.show()

    # ---------------------------------------------------------------------------
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
            hdr['DATAFILE'] = self.filenamedata
            hdr['SIMUFILE'] = os.path.basename(self.filename)

            hdr['AIRMASS'] = self.airmass
            hdr['PRESSURE'] = self.pressure
            hdr['TEMPERAT'] = self.temperature
            hdr['NBATMPTS'] = NB_ATM_POINTS

            hdr['NBAERPTS'] = NB_AER_POINTS
            hdr['AERMIN'] = AER_MIN
            hdr['AERMAX'] = AER_MAX

            hdr['NBPWVPTS'] = NB_PWV_POINTS
            hdr['PWVMIN'] = PWV_MIN
            hdr['PWVMAX'] = PWV_MAX

            hdr['NBOZPTS'] = NB_OZ_POINTS
            hdr['OZMIN'] = OZ_MIN
            hdr['OZMAX'] = OZ_MAX

            hdr['AER_PTS'] = np.array_str(AER_Points)
            hdr['PWV_PTS'] = np.array_str(PWV_Points)
            hdr['OZ_PTS'] = np.array_str(OZ_Points)
            hdr['NBWLBIN'] = WL.size
            hdr['WLMIN'] = WLMIN
            hdr['WLMAX'] = WLMAX

            hdr['IDX_CNT'] = index_atm_count
            hdr['IDX_AER'] = index_atm_aer
            hdr['IDX_PWV'] = index_atm_pwv
            hdr['IDX_OZ'] = index_atm_oz
            hdr['IDX_DATA'] = index_atm_data

            if parameters.VERBOSE:
                print(hdr)

            hdu = fits.PrimaryHDU(self.atmgrid, header=hdr)
            hdu.writeto(self.filename, overwrite=True)
            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info('\n\tAtmosphere.save atm-file=%s' % (self.filename))

            return hdr

    # ---------------------------------------------------------------------------
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
            self.filenamedata = hdr['DATAFILE']
            # hdr['SIMUFILE']=os.path.basename(self.filename)

            self.airmass = hdr['AIRMASS']
            self.pressure = hdr['PRESSURE']
            self.temperature = hdr['TEMPERAT']

            # hope those are the same parameters : TBD !!!!
            NB_ATM_POINTS = hdr['NBATMPTS']

            NB_AER_POINTS = hdr['NBAERPTS']
            AER_MIN = hdr['AERMIN']
            AER_MAX = hdr['AERMAX']

            NB_PWV_POINTS = hdr['NBPWVPTS']
            PWV_MIN = hdr['PWVMIN']
            PWV_MAX = hdr['PWVMAX']

            NB_OZ_POINTS = hdr['NBOZPTS']
            OZ_MIN = hdr['OZMIN']
            OZ_MAX = hdr['OZMAX']

            AER_Points = np.linspace(AER_MIN, AER_MAX, NB_AER_POINTS)
            OZ_Points = np.linspace(OZ_MIN, OZ_MAX, NB_OZ_POINTS)
            PWV_Points = np.linspace(PWV_MIN, PWV_MAX, NB_PWV_POINTS)

            NBWLBINS = hdr['NBWLBIN']
            WLMIN = hdr['WLMIN']
            WLMAX = hdr['WLMAX']

            index_atm_count = hdr['IDX_CNT']
            index_atm_aer = hdr['IDX_AER']
            index_atm_pwv = hdr['IDX_PWV']
            index_atm_oz = hdr['IDX_OZ']
            index_atm_data = hdr['IDX_DATA']

            self.atmgrid = np.zeros((NB_ATM_POINTS + 1, NB_atm_HEADER + NB_atm_DATA))

            self.atmgrid[:, :] = hdu[0].data[:, :]

            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info('\n\tAtmosphere.load atm-file=%s' % (self.filename))

            # interpolate the grid
            self.lambdas = self.atmgrid[0, index_atm_data:]
            self.model = RegularGridInterpolator((self.lambdas, OZ_Points, PWV_Points, AER_Points), (
                self.atmgrid[1:, index_atm_data:].reshape(NB_AER_POINTS, NB_PWV_POINTS, NB_OZ_POINTS,
                                                          len(self.lambdas))).T, bounds_error=False, fill_value=0)

            return self.atmgrid, self.header
        # ---------------------------------------------------------------------------

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

        # defines the datapath relative to the Spectractor sim path
        datapath = os.path.join(spectractorsim_path, "CTIOThroughput")

        '''
        # QE
        wl,qe=ctio.Get_QE(datapath)
        self.qe=interp1d(wl,qe,kind='linear',bounds_error=False,fill_value=0.) 
        
        #  Throughput
        wl,trt=ctio.Get_Throughput(datapath)
        self.to=interp1d(wl,trt,kind='linear',bounds_error=False,fill_value=0.)
        
        # Mirrors 
        wl,trm=ctio.Get_Mirror(datapath)
        self.tm=interp1d(wl,trm,kind='linear',bounds_error=False,fill_value=0.) 
        '''
        wl, trm, err = ctio.Get_Total_Throughput(datapath)
        self.to = interp1d(wl, trm, kind='linear', bounds_error=False, fill_value=0.)
        self.to_err = interp1d(wl, err, kind='linear', bounds_error=False, fill_value=0.)

        # Filter RG715
        wl, trg = ctio.Get_RG715(datapath)
        self.tfr = interp1d(wl, trg, kind='linear', bounds_error=False, fill_value=0.)

        # Filter FGB37
        wl, trb = ctio.Get_FGB37(datapath)
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
        # plt.plot(WL,self.qe(WL),'b-',label='qe')
        plt.plot(WL, self.to(WL), 'g-', label='othr')
        # plt.plot(WL,self.tm(WL),'y-',label='mirr')
        plt.plot(WL, self.tf(WL), 'k-', label='filt')
        plt.plot(WL, self.tfr(WL), 'k:', label='RG715')
        plt.plot(WL, self.tfb(WL), 'k--', label='FGB37')
        plt.errorbar(WL, self.transmission(WL), yerr=self.transmission_err(WL), color='r', linestyle='-', lw=2,
                     label='tot')
        plt.legend()
        plt.grid()
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Transmission")
        plt.title("Telescope transmissions")


# ----------------------------------------------------------------------------------


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
        self.lambdas = atmgrid[0, index_atm_data:]
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
        sim = SpectrumSimulation(self.spectrum, self.atmgrid, self.telescope, self.disperser)
        # product of all sed and transmission except atmosphere
        all_transm, all_transm_err = sim.simulate_without_atmosphere(self.lambdas)
        # copy atmospheric grid parameters into spectra grid
        self.spectragrid = np.zeros_like(self.atmgrid)
        self.spectragrid[0, index_atm_data:] = self.lambdas
        self.spectragrid[:, index_atm_count:index_atm_data] = self.atmgrid[:, index_atm_count:index_atm_data]
        # Is broadcasting working OK ?
        self.spectragrid[1:, index_atm_data:] = self.atmgrid[1:, index_atm_data:] * all_transm  # *Factor

        return self.spectragrid

    # ---------------------------------------------------------------------------
    def plot_spectra(self):
        plt.figure()
        counts = self.spectragrid[1:, index_atm_count]
        for count in counts:
            plt.plot(WL, self.spectragrid[int(count), index_atm_data:])
        plt.grid()
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Flux [ADU/s]")
        plt.title("Spectra for Atmospheric variations")
        plt.show()

    # ---------------------------------------------------------------------------
    def plot_spectra_img(self):
        plt.figure()
        img = plt.imshow(self.spectragrid[1:, index_atm_data:], origin='lower', cmap='jet')
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
def SpectractorSimInit(filename):
    """ SpectractorInit
    Main function to simulate several spectra 
    A grid of spectra will be produced for a given target, airmass and pressure

    Args:
        filename (:obj:`str`): filename of the image (data)
    """
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTORSIM initialisation')
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
def SpectractorSimCore(spectrum, telescope, disperser, target, lambdas, airmass=1.0, pressure=800, temperature=10,
                       pwv=5, ozone=300, aerosols=0.05, reso=None):
    """ SpectractorCore
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
    # ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
def SpectractorSimGrid(filename, outputdir):
    """ SpectractorSimGrid
    Main function to simulate several spectra 
    A grid of spectra will be produced for a given target, airmass and pressure

    Args:
        filename (:obj:`str`): filename of the image (data)
        outputdir (:obj:`str`): path to the output directory
        
    """
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTORSIMGRID')
    # Initialisation
    spectrum, telescope, disperser, target = SpectractorSimInit(filename)
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
    if os.path.exists(output_atmfilename) and os.path.getsize(output_atmfilename) > MINFILESIZE:
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
    spectra = SpectrumSimGrid(spectrum, atmgrid, telescope, disperser, target, header)
    spectragrid = spectra.compute()
    spectra.save_spectra(output_filename)
    if parameters.VERBOSE:
        infostring = '\n\t ========= Spectra simulation :  ==============='
        spectra.plot_spectra()
        spectra.plot_spectra_img()
    # ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
def SpectractorSim(filename, lambdas, outputdir="", pwv=5, ozone=300, aerosols=0.05, reso=None):
    """ SpectractorSim
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
    spectrum, telescope, disperser, target = SpectractorSimInit(filename)

    # SIMULATE SPECTRUM
    # -------------------
    airmass = spectrum.header['AIRMASS']
    pressure = spectrum.header['OUTPRESS']
    temperature = spectrum.header['OUTTEMP']

    spectrum_simulation = SpectractorSimCore(spectrum, telescope, disperser, target, lambdas, airmass, pressure,
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
    # ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
#  START SPECTRACTORSIM HERE !
# ----------------------------------------------------------------------------------


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

    # spectrum_simulation = SpectractorSim(filename,lambdas=WL,pwv=5,ozone=300,aerosols=0.05)
    SpectractorSimGrid(filename, opts.output_directory)

    atmgrid = AtmosphereGrid(filename, "../Spectractor/outputs/data_30may17/reduc_20170530_134_atmsim.fits")
    atm = Atmosphere(atmgrid.airmass, atmgrid.pressure, atmgrid.temperature)

    fig = plt.figure()
    for i in range(5):
        print(i)
        a = atmgrid.simulate(300, 5, i * 0.01 + 0.005)
        plt.plot(atmgrid.lambdas, a / atm.simulate(300, 5, i * 0.01 + 0.005)(atmgrid.lambdas), label="%d" % i)
    plt.legend()
    plt.show()

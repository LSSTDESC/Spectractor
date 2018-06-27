import coloredlogs
import logging
import numpy as np
import astropy.units as units
from astropy import constants as const
import os

# Paths
mypath = os.path.dirname(__file__)
HOLO_DIR = os.path.join(mypath, "dispersers/")

# CCD characteristics
IMSIZE = 2048  # size of the image in pixel
PIXEL2MM = 24e-3  # pixel size in mm
PIXEL2ARCSEC = 0.401  # pixel size in arcsec
ARCSEC2RADIANS = np.pi / (180. * 3600.)  # conversion factor from arcsec to radians
MAXADU = 60000  # approximate maximum ADU output of the CCD
GAIN = 3.  # electronic gain : elec/ADU

# Observatory characteristics
OBS_NAME = 'CTIO'
OBS_ALTITUDE = 2.200 # CTIO altitude in k meters from astropy package (Cerro Pachon)
# LSST_Altitude = 2.750  # in k meters from astropy package (Cerro Pachon)
OBS_LATITUDE = '-30 10 07.90'  # CTIO latitude
OBS_DIAMETER = 0.9 * units.m  # Diameter of the telescope
OBS_SURFACE = np.pi * OBS_DIAMETER ** 2 / 4.  # Surface of telescope
EPOCH = "J2000.0"

# Filters
HALPHA_CENTER = 655.9e-6  # center of the filter in mm
HALPHA_WIDTH = 6.4e-6  # width of the filter in mm
FGB37 = {'label': 'FGB37', 'min': 300, 'max': 800}
RG715 = {'label': 'RG715', 'min': 690, 'max': 1100}
HALPHA_FILTER = {'label': 'Halfa', 'min': HALPHA_CENTER - 2 * HALPHA_WIDTH, 'max': HALPHA_CENTER + 2 * HALPHA_WIDTH}
ZGUNN = {'label': 'Z-Gunn', 'min': 800, 'max': 1100}
FILTERS = [RG715, FGB37, HALPHA_FILTER, ZGUNN]

# Conversion factor
# Units of SEDs in flam (erg/s/cm2/nm) :
SED_UNIT = 1 * units.erg / units.s / units.cm ** 2 / units.nanometer
TIME_UNIT = 1 * units.s  # flux for 1 second
hc = const.h * const.c  # h.c product of fontamental constants c and h
wl_dwl_unit = units.nanometer ** 2  # lambda.dlambda  in wavelength in nm
g_disperser_ronchi = 0.2  # theoretical gain for order+1 : 20%
FLAM_TO_ADURATE = (
    (OBS_SURFACE * SED_UNIT * TIME_UNIT * wl_dwl_unit / hc / GAIN * g_disperser_ronchi).decompose()).value

# Search windows in images
XWINDOW = 100  # window x size to search for the targetted object
YWINDOW = 100  # window y size to search for the targetted object
XWINDOW_ROT = 50  # window x size to search for the targetted object
YWINDOW_ROT = 50  # window y size to search for the targetted object

# Rotation parameters
ROT_PREFILTER = True  # must be set to true, otherwise create residuals and correlated noise
ROT_ORDER = 5  # must be above 3

# Range for spectrum
LAMBDA_MIN = 350  # minimum wavelength for spectrum extraction (in nm)
LAMBDA_MAX = 1100  # maxnimum wavelength for spectrum extraction (in nm)
LAMBDAS = np.arange(LAMBDA_MIN, LAMBDA_MAX, 1)

# Detection line algorithm
BGD_ORDER = 3  # order of the background polynome to fit
BGD_NPARAMS = BGD_ORDER + 1  # number of unknown parameters for background

# Plotting
PAPER = False
LINEWIDTH = 2
PLOT_DIR = 'plots'
SAVE = False

# Verbosity
VERBOSE = False
DEBUG = False
MY_FORMAT = "%(asctime)-20s %(name)-10s %(funcName)-20s %(levelname)-6s %(message)s"
logging.basicConfig(format=MY_FORMAT, level=logging.WARNING)


def set_logger(logger):
    my_logger = logging.getLogger(logger)
    if VERBOSE > 0:
        my_logger.setLevel(logging.INFO)
        coloredlogs.install(fmt=MY_FORMAT, level=logging.INFO)
    else:
        my_logger.setLevel(logging.WARNING)
        coloredlogs.install(fmt=MY_FORMAT, level=logging.WARNING)
    if DEBUG:
        my_logger.setLevel(logging.DEBUG)
        coloredlogs.install(fmt=MY_FORMAT, level=logging.DEBUG)
    return my_logger

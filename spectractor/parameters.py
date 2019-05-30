import numpy as np
import os
import astropy.units as units
from astropy import constants as const
import matplotlib as mpl

# These parameters are the default values adapted to CTIO
# To modify them, please create a new config file and load it.

# Paths
mypath = os.path.dirname(__file__)
HOLO_DIR = os.path.join(mypath, "extractor/dispersers/")
THROUGHPUT_DIR = os.path.join(mypath, "simulation/CTIOThroughput/")
LOGBOOK = "None" # external logbook

# CCD characteristics
CCD_IMSIZE = 2048  # size of the image in pixel
CCD_PIXEL2MM = 24e-3  # pixel size in mm
CCD_PIXEL2ARCSEC = 0.401  # pixel size in arcsec
CCD_ARCSEC2RADIANS = np.pi / (180. * 3600.)  # conversion factor from arcsec to radians
CCD_MAXADU = 60000  # approximate maximum ADU output of the CCD
CCD_GAIN = 3.  # electronic gain : elec/ADU

# Instrument characteristics
OBS_NAME = 'CTIO'
OBS_ALTITUDE = 2.200  # CTIO altitude in k meters from astropy package (Cerro Pachon)
OBS_LATITUDE = '-30 10 07.90'  # CTIO latitude
OBS_DIAMETER = 0.9 * units.m   # Diameter of the telescope
OBS_SURFACE = np.pi * OBS_DIAMETER ** 2 / 4.  # Surface of telescope
OBS_EPOCH = "J2000.0"
OBS_TRANSMISSION_SYSTEMATICS = 0.005
OBS_OBJECT_TYPE = 'STAR'  # To choose between STAR, HG-AR, MONOCHROMATOR
OBS_TELESCOPE_TRANSMISSION = 'ctio_throughput.txt' # telescope transmission file
OBS_FULL_INSTRUMENT_TRANSMISSON = 'ctio_throughput_300517_v1.txt' # full instrument transmission file
OBS_QUANTUM_EFFICIENCY = "qecurve.txt"  # quantum efficiency of the detector file

# Filters
HALPHA_CENTER = 655.9e-6  # center of the filter in mm
HALPHA_WIDTH = 6.4e-6  # width of the filter in mm
FGB37 = {'label': 'FGB37', 'min': 300, 'max': 800}
RG715 = {'label': 'RG715', 'min': 690, 'max': 1100}
HALPHA_FILTER = {'label': 'Halfa', 'min': HALPHA_CENTER - 2 * HALPHA_WIDTH, 'max': HALPHA_CENTER + 2 * HALPHA_WIDTH}
ZGUNN = {'label': 'Z-Gunn', 'min': 800, 'max': 1100}
FILTERS = [RG715, FGB37, HALPHA_FILTER, ZGUNN]

# Making of the holograms
DISPERSER_DEFAULT = "" # default disperser label
DISTANCE2CCD = 55.45  # distance between hologram and CCD in mm
DISTANCE2CCD_ERR = 0.19  # uncertainty on distance between hologram and CCD in mm
LAMBDA_CONSTRUCTOR = 639e-6  # constructor wavelength to make holograms in mm
GROOVES_PER_MM = 350  # approximate effective number of lines per millimeter of the hologram
PLATE_CENTER_SHIFT_X = -6.  # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y = -8.  # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_X_ERR = 2.  # estimate uncertainty on plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y_ERR = 2.  # estimate uncertainty on plate center shift on x in mm in filter frame

# Search windows in images
IS_ORDER0 = True # is order 0 present on image
ORDER0_X = 0 # estimated position of the order 0 in pixels on x axis
ORDER0_Y = 0 # estimated position of the order 0 in pixels on y axis
XWINDOW = 100  # window x size to search for the targeted object
YWINDOW = 100  # window y size to search for the targeted object
XWINDOW_ROT = 50   # window x size to search for the targeted object
YWINDOW_ROT = 50   # window y size to search for the targeted object
PIXSHIFT_PRIOR = 2 # prior on the reliability of the centroid estimate in pixels

# Rotation parameters
ROT_PREFILTER = True  # must be set to true, otherwise create residuals and correlated noise
ROT_ORDER = 5  # must be above 3

# Range for spectrum
LAMBDA_MIN = 300  # minimum wavelength for spectrum extraction (in nm)
LAMBDA_MAX = 1100  # maximum wavelength for spectrum extraction (in nm)
LAMBDAS = np.arange(LAMBDA_MIN, LAMBDA_MAX, 1)

# Background subtraction parameters
PIXWIDTH_SIGNAL = 10 # half transverse width of the signal rectangular window in pixels
PIXDIST_BACKGROUND = 20 # distance from dispersion axis to analyse the background in pixels
PIXWIDTH_BACKGROUND = 10 # transverse width of the background rectangular window in pixels
BGD_ORDER = 1 # the order of the polynomial background to fit transversaly

# Detection line algorithm
CALIB_BGD_ORDER = 3  # order of the background polynome to fit
CALIB_BGD_NPARAMS = CALIB_BGD_ORDER + 1  # number of unknown parameters for background
CALIB_PEAK_WIDTH = 7  # half range to look for local extrema in pixels around tabulated line values
CALIB_BGD_WIDTH = 10  # size of the peak sides to use to fit spectrum base line

# Conversion factor
# Units of SEDs in flam (erg/s/cm2/nm) :
SED_UNIT = 1 * units.erg / units.s / units.cm ** 2 / units.nanometer
TIME_UNIT = 1 * units.s  # flux for 1 second
hc = const.h * const.c  # h.c product of fontamental constants c and h
wl_dwl_unit = units.nanometer ** 2  # lambda.dlambda  in wavelength in nm
g_disperser_ronchi = 0.2  # theoretical gain for order+1 : 20%
FLAM_TO_ADURATE = (
    (OBS_SURFACE * SED_UNIT * TIME_UNIT * wl_dwl_unit / hc / CCD_GAIN * g_disperser_ronchi).decompose()).value

# fit workspace
FIT_WORKSPACE = None

# Plotting
PAPER = False
LINEWIDTH = 2
PLOT_DIR = 'plots'
SAVE = False

# Verbosity
VERBOSE = False
DEBUG = False
MY_FORMAT = "%(asctime)-20s %(name)-10s %(funcName)-20s %(levelname)-6s %(message)s"

# Plots
DISPLAY = True
if os.environ.get('DISPLAY', '') == '':
    mpl.use('agg')
    DISPLAY = False

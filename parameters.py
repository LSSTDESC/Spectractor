import os, sys
mypath = os.path.dirname(__file__)
import coloredlogs, logging


# Paths
HOLO_DIR = os.path.join(mypath,"dispersers/")

# Search windows in images
XWINDOW = 100 # window x size to search for the targetted object 
YWINDOW = 100  # window y size to search for the targetted object
XWINDOW_ROT = 50 # window x size to search for the targetted object 
YWINDOW_ROT = 50  # window y size to search for the targetted object

# Rotation parameters
ROT_PREFILTER = True # must be set to true, otherwise create residuals and correlated noise
ROT_ORDER = 5 # must be above 3

# Range for spectrum
LAMBDA_MIN = 350 # minimum wavelength for spectrum extraction (in nm)
LAMBDA_MAX = 1100 # maxnimum wavelength for spectrum extraction (in nm)

# Detection line algorithm
BGD_ORDER = 1 # order of the background polynome to fit
BGD_NPARAMS = BGD_ORDER + 1 # number of unknown parameters for background

# Verbosity
VERBOSE = False
DEBUG = False
MY_FORMAT = "%(asctime)-20s %(name)-10s %(funcName)-20s %(levelname)-6s %(message)s"  
logging.basicConfig(format=MY_FORMAT, level=logging.WARNING)
def set_logger(logger):
    my_logger = logging.getLogger(logger)
    if VERBOSE > 0:
        my_logger.setLevel(logging.INFO)
        coloredlogs.install(fmt=MY_FORMAT,level=logging.INFO)
    else:
        my_logger.setLevel(logging.WARNING)
        coloredlogs.install(fmt=MY_FORMAT,level=logging.WARNING)
    if DEBUG:
        my_logger.setLevel(logging.DEBUG)
        coloredlogs.install(fmt=MY_FORMAT,level=logging.DEBUG)
    return my_logger


import os
import matplotlib as mpl
import numpy as np

# These parameters are the default values adapted to CTIO
# To modify them, please create a new config file and load it.


def __getattr__(name):
    """Method to allow querying items without them existing.

    NB: This breaks hasattr(parameters, name), as that is implemented by
    calling __getattr__() and checking it raises an AttributeError, so
    hasattr() for parameters will now always return True.

    If necessary this can be worked around by instead doing:
        `if name in dir(parameters):`

    Examples
    --------
    >>> from spectractor import parameters
    >>> print(parameters.CCD_IMSIZE)
    2048
    >>> print(parameters.DUMMY)
    False
    """
    if name in locals():
        return locals()[name]
    else:
        return False


# Pipeline
SPECTRACTOR_FIT_TARGET_CENTROID = "fit"  # method to get target centroid, choose among: guess, fit, WCS
SPECTRACTOR_COMPUTE_ROTATION_ANGLE = "hessian"  # method to get image rotation angle: False, disperser, hessian
SPECTRACTOR_DECONVOLUTION_PSF2D = True  # deconvolve spectrogram with simple 2D PSF analysis: False, True
SPECTRACTOR_DECONVOLUTION_FFM = True  # deconvolve spectrogram with full forward model: False, True
SPECTRACTOR_DECONVOLUTION_SIGMA_CLIP = 20  # value of sigma clip parameter for the spectractor deconvolution process PSF2D and FFM
SPECTRACTOR_BACKGROUND_SUBTRACTION = True #if True the background is estimated and subtracted 
SPECTRACTOR_FIT_TIMEOUT_PER_ITER = 600  # maximum time per gradient descent iteration before TimeoutError in seconds
SPECTRACTOR_FIT_TIMEOUT = 3600  # maximum time per gradient descent before TimeoutError in seconds

# Paths
DISPERSER_DIR = "./extractor/dispersers/"
CONFIG_DIR = "../config/"
THROUGHPUT_DIR = "./simulation/CTIOThroughput/"
if 'ASTROMETRYNET_DIR' in os.environ:
    ASTROMETRYNET_DIR = os.getenv('ASTROMETRYNET_DIR') + '/'
else:
    ASTROMETRYNET_DIR = ''
if 'LIBRADTRAN_DIR' in os.environ:
    LIBRADTRAN_DIR = os.getenv('LIBRADTRAN_DIR') + '/'
else:
    LIBRADTRAN_DIR = ''

# CCD characteristics
CCD_IMSIZE = 2048  # size of the image in pixel
CCD_PIXEL2MM = 24e-3  # pixel size in mm
CCD_PIXEL2ARCSEC = 0.401  # pixel size in arcsec
CCD_MAXADU = 60000  # approximate maximum ADU output of the CCD
CCD_GAIN = 3.  # electronic gain : elec/ADU
CCD_REBIN = 1  # rebinning of the image in pixel

# Instrument characteristics
OBS_NAME = 'CTIO'
OBS_ALTITUDE = 2.200  # CTIO altitude in k meters from astropy package (Cerro Pachon)
OBS_LATITUDE = '-30 10 07.90'  # CTIO latitude
OBS_SURFACE = 6361  # Effective surface of the telescope in cm**2 accounting for obscuration
OBS_EPOCH = "J2000.0"
OBS_OBJECT_TYPE = 'STAR'  # To choose between STAR, HG-AR, MONOCHROMATOR
OBS_FULL_INSTRUMENT_TRANSMISSON = 'ctio_throughput_300517_v1.txt'  # Full instrument transmission file
OBS_TRANSMISSION_SYSTEMATICS = 0.005
OBS_CAMERA_ROTATION = 0  # Camera (x,y) rotation angle with respect to (north-up, east-left) system in degrees
OBS_CAMERA_DEC_FLIP_SIGN = 1  # Camera (x,y) flip signs with respect to (north-up, east-left) system
OBS_CAMERA_RA_FLIP_SIGN = 1  # Camera (x,y) flip signs with respect to (north-up, east-left) system

# Spectrograph
DISTANCE2CCD = 55.45  # distance between hologram and CCD in mm
DISTANCE2CCD_ERR = 0.19  # uncertainty on distance between hologram and CCD in mm
GRATING_ORDER_2OVER1 = 0.1  # default value for order 2 over order 1 transmission ratio

# Search windows in images
XWINDOW = 100  # window x size to search for the targeted object
YWINDOW = 100  # window y size to search for the targeted object
XWINDOW_ROT = 50  # window x size to search for the targeted object
YWINDOW_ROT = 50  # window y size to search for the targeted object
PIXSHIFT_PRIOR = 1  # prior on the reliability of the centroid estimate in pixels

# Rotation parameters
ROT_PREFILTER = True  # must be set to true, otherwise create residuals and correlated noise
ROT_ORDER = 5  # must be above 3
ROT_ANGLE_MIN = -10
ROT_ANGLE_MAX = 10  # in the Hessian analysis to compute rotation angle, cut all angles outside this range [degrees]

# Range for spectrum
LAMBDA_MIN = 300  # minimum wavelength for spectrum extraction (in nm)
LAMBDA_MAX = 1100  # maximum wavelength for spectrum extraction (in nm)
LAMBDA_STEP = 1  # step size for the wavelength array (in nm)
SPEC_ORDER = 1  # spectrum order to extract
LAMBDAS = np.arange(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP)

# Background subtraction parameters
PIXWIDTH_SIGNAL = 10  # half transverse width of the signal rectangular window in pixels
PIXDIST_BACKGROUND = 20  # distance from dispersion axis to analyse the background in pixels
PIXWIDTH_BACKGROUND = 10  # transverse width of the background rectangular window in pixels
PIXWIDTH_BOXSIZE = 20  # box size for sextractor evaluation of the background
BGD_ORDER = 1  # the order of the polynomial background to fit in the transverse direction

# PSF
PSF_TYPE = "Moffat"  # the PSF model: Gauss, Moffat or MoffatGauss
PSF_POLY_ORDER = 2  # the order of the polynomials to model wavelength dependence of the PSF shape parameters
PSF_FIT_REG_PARAM = 0.01  # regularisation parameter for the chisq minimisation to extract the spectrum
PSF_PIXEL_STEP_TRANSVERSE_FIT = 10  # step size in pixels for the first transverse PSF1D fit
PSF_FWHM_CLIP = 2  # PSF is not evaluated outside a region larger than max(PIXWIDTH_SIGNAL, PSF_FWHM_CLIP*fwhm) pixels

# Detection line algorithm
CALIB_BGD_ORDER = 3  # order of the background polynome to fit
CALIB_PEAK_WIDTH = 7  # half range to look for local extrema in pixels around tabulated line values
CALIB_BGD_WIDTH = 10  # size of the peak sides to use to fit spectrum base line
CALIB_SAVGOL_WINDOW = 5  # window size for the savgol filter in pixels
CALIB_SAVGOL_ORDER = 2  # polynom order for the savgol filter

# Plotting
PAPER = False
LINEWIDTH = 2
PLOT_DIR = 'plots'
SAVE = False

# Verbosity
VERBOSE = False
DEBUG = False
DEBUG_LOGGING = False

# Plots
DISPLAY = True
if os.environ.get('DISPLAY', '') == '':
    mpl.use('agg')
    DISPLAY = False
PLOT_XLABEL = r"$x$ [pixels]"
PLOT_YLABEL = r"$y$ [pixels]"
PLOT_ROT_LABEL = r"$\varphi_d$ [degrees]"

STYLE_PARAMETERS = ["VERBOSE", "DEBUG", "PAPER", "LINEWIDTH", "PLOT_DIR", "SAVE", "DEBUG_LOGGING", "DISPLAY",
                    "PLOT_XLABEL", "PLOT_YLABEL", "PLOT_ROT_LABEL"]

import numpy as np
import os
import coloredlogs
import logging
import astropy.units as units
from astropy import constants as const
from spectractor.tools import Line
import matplotlib as mpl
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# Paths
mypath = os.path.dirname(__file__)
HOLO_DIR = os.path.join(mypath, "extractor/dispersers/")
THROUGHPUT_DIR = os.path.join(mypath, "simulation/CTIOThroughput/")

# Plots
DISPLAY = True
if os.environ.get('DISPLAY', '') == '':
    mpl.use('agg')
    DISPLAY = False

# CCD characteristics
IMSIZE = 2048  # size of the image in pixel
PIXEL2MM = 24e-3  # pixel size in mm
PIXEL2ARCSEC = 0.401  # pixel size in arcsec
ARCSEC2RADIANS = np.pi / (180. * 3600.)  # conversion factor from arcsec to radians
MAXADU = 60000  # approximate maximum ADU output of the CCD
GAIN = 3.  # electronic gain : elec/ADU

# Observatory characteristics
OBS_NAME = 'CTIO'
OBS_ALTITUDE = 2.200  # CTIO altitude in k meters from astropy package (Cerro Pachon)
# LSST_Altitude = 2.750  # in k meters from astropy package (Cerro Pachon)
OBS_LATITUDE = '-30 10 07.90'  # CTIO latitude
OBS_DIAMETER = 0.9 * units.m  # Diameter of the telescope
OBS_SURFACE = np.pi * OBS_DIAMETER ** 2 / 4.  # Surface of telescope
EPOCH = "J2000.0"
TELESCOPE_TRANSMISSION_SYSTEMATICS = 0.005

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

# Making of the holograms
DISTANCE2CCD = 55.45  # distance between hologram and CCD in mm
DISTANCE2CCD_ERR = 0.19  # uncertainty on distance between hologram and CCD in mm
LAMBDA_CONSTRUCTOR = 639e-6  # constructor wavelength to make holograms in mm
GROOVES_PER_MM = 350  # approximate effective number of lines per millimeter of the hologram
PLATE_CENTER_SHIFT_X = -6.  # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y = -8.  # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_X_ERR = 2.  # estimate uncertainty on plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y_ERR = 2.  # estimate uncertainty on plate center shift on x in mm in filter frame

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

# Lines
HALPHA = Line(656.3, atmospheric=False, label='$H\\alpha$', label_pos=[-0.016, 0.02], use_for_calibration=True)
HBETA = Line(486.3, atmospheric=False, label='$H\\beta$', label_pos=[0.007, 0.02], use_for_calibration=True)
HGAMMA = Line(434.0, atmospheric=False, label='$H\\gamma$', label_pos=[0.007, 0.02], use_for_calibration=True)
HDELTA = Line(410.2, atmospheric=False, label='$H\\delta$', label_pos=[0.007, 0.02])
OIII = Line(500.7, atmospheric=False, label=r'$O_{III}$', label_pos=[0.007, 0.02])
CII1 = Line(723.5, atmospheric=False, label=r'$C_{II}$', label_pos=[0.005, 0.88])
CII2 = Line(711.0, atmospheric=False, label=r'$C_{II}$', label_pos=[0.005, 0.02])
CIV = Line(706.0, atmospheric=False, label=r'$C_{IV}$', label_pos=[-0.016, 0.88])
CII3 = Line(679.0, atmospheric=False, label=r'$C_{II}$', label_pos=[0.005, 0.02])
CIII1 = Line(673.0, atmospheric=False, label=r'$C_{III}$', label_pos=[-0.016, 0.88])
CIII2 = Line(570.0, atmospheric=False, label=r'$C_{III}$', label_pos=[0.007, 0.02])
CIII3 = Line(970.5, atmospheric=False, label=r'$C_{III}$', label_pos=[0.007, 0.02])
FEII1 = Line(463.8, atmospheric=False, label=r'$Fe_{II}$', label_pos=[-0.016, 0.02])
FEII2 = Line(515.8, atmospheric=False, label=r'$Fe_{II}$', label_pos=[0.007, 0.02])
FEII3 = Line(527.3, atmospheric=False, label=r'$Fe_{II}$', label_pos=[0.007, 0.02])
FEII4 = Line(534.9, atmospheric=False, label=r'$Fe_{II}$', label_pos=[0.007, 0.02])
HEI1 = Line(388.8, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI2 = Line(447.1, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI3 = Line(587.5, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI4 = Line(750.0, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI5 = Line(776.0, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI6 = Line(781.6, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI7 = Line(848.2, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI8 = Line(861.7, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI9 = Line(906.5, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI10 = Line(923.5, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI11 = Line(951.9, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI12 = Line(1023.5, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
HEI13 = Line(353.1, atmospheric=False, label=r'$He_{I}$', label_pos=[0.007, 0.02])
OI = Line(630.0, atmospheric=False, label=r'$O_{II}$', label_pos=[0.007, 0.02])
OII = Line(732.5, atmospheric=False, label=r'$O_{II}$', label_pos=[0.007, 0.02])
HEII1 = Line(468.6, atmospheric=False, label=r'$He_{II}$', label_pos=[0.007, 0.02])
HEII2 = Line(611.8, atmospheric=False, label=r'$He_{II}$', label_pos=[0.007, 0.02])
HEII3 = Line(617.1, atmospheric=False, label=r'$He_{II}$', label_pos=[0.007, 0.02])
HEII4 = Line(856.7, atmospheric=False, label=r'$He_{II}$', label_pos=[0.007, 0.02])
HI = Line(833.9, atmospheric=False, label=r'$H_{I}$', label_pos=[0.007, 0.02])
FE1 = Line(382.044, atmospheric=True, label=r'$Fe$',
           label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
FE2 = Line(430.790, atmospheric=True, label=r'$Fe$',
           label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
FE3 = Line(438.355, atmospheric=True, label=r'$Fe$',
           label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
CAII1 = Line(393.366, atmospheric=True, label=r'$Ca_{II}$',
             label_pos=[0.007, 0.02],
             use_for_calibration=True)  # https://en.wikipedia.org/wiki/Fraunhofer_lines
CAII2 = Line(396.847, atmospheric=True, label=r'$Ca_{II}$',
             label_pos=[0.007, 0.02],
             use_for_calibration=True)  # https://en.wikipedia.org/wiki/Fraunhofer_lines
O2 = Line(762.1, atmospheric=True, label=r'$O_2$',
          label_pos=[0.007, 0.02],
          use_for_calibration=True)  # http://onlinelibrary.wiley.com/doi/10.1029/98JD02799/pdf
# O2_1 = Line( 760.6,atmospheric=True,label='',label_pos=[0.007,0.02]) # libradtran paper fig.3
# O2_2 = Line( 763.2,atmospheric=True,label='$O_2$',label_pos=[0.007,0.02])  # libradtran paper fig.3
O2B = Line(686.719, atmospheric=True, label=r'$O_2(B)$',
           label_pos=[0.007, 0.02], use_for_calibration=True)  # https://en.wikipedia.org/wiki/Fraunhofer_lines
O2Y = Line(898.765, atmospheric=True, label=r'$O_2(Y)$',
           label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
O2Z = Line(822.696, atmospheric=True, label=r'$O_2(Z)$',
           label_pos=[0.007, 0.02])  # https://en.wikipedia.org/wiki/Fraunhofer_lines
# H2O = Line( 960,atmospheric=True,label='$H_2 O$',label_pos=[0.007,0.02],width_bounds=(1,50))  #
H2O_1 = Line(935, atmospheric=True, label=r'$H_2 O$', label_pos=[0.007, 0.02],
             width_bounds=[5, 30])  # libradtran paper fig.3, broad line
H2O_2 = Line(960, atmospheric=True, label=r'$H_2 O$', label_pos=[0.007, 0.02],
             width_bounds=[5, 30])  # libradtran paper fig.3, broad line
ATMOSPHERIC_LINES = [O2, O2B, O2Y, O2Z, H2O_1, H2O_2, CAII1, CAII2, FE1, FE2, FE3]
HYDROGEN_LINES = [HALPHA, HBETA, HGAMMA, HDELTA]
ISM_LINES = [OIII, CII1, CII2, CIV, CII3, CIII1, CIII2, CIII3, HEI1, HEI2, HEI3, HEI4, HEI5, HEI6, HEI7, HEI8,
             HEI9, HEI10, HEI11, HEI12, HEI13, OI, OII, HEII1, HEII2, HEII3, HEII4,  HI, FEII1, FEII2, FEII3, FEII4]



def set_logger(logger):
    """Logger function for all classes.

    Parameters
    ----------
    logger: str
        Name of the class, usually self.__class__.__name__

    Returns
    -------
    my_logger: logging
        Logging object

    Examples
    --------
    >>> class Test:
    ...     def __init__(self):
    ...         self.my_logger = set_logger(self.__class__.__name__)
    ...     def log(self):
    ...         self.my_logger.info('This info test function works.')
    ...         self.my_logger.debug('This debug test function works.')
    ...         self.my_logger.warning('This warning test function works.')
    >>> test = Test()
    >>> test.log()
    """
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()

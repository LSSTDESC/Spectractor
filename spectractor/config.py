import configparser
import os
import re
import shutil

import astropy.units.quantity
import numpy as np
import logging
import astropy.units as units
from astropy import constants as const

from spectractor import parameters

if not parameters.CALLING_CODE:
    import coloredlogs

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.ERROR)


def from_config_to_dict(path):
    """Convert config file keywords into dictionnary.

    Parameters
    ----------
    path: str
        The path to the config file.

    Examples
    --------
    >>> mypath = os.path.dirname(__file__)
    >>> out = from_config_to_dict(os.path.join(mypath, "../config/", "default.ini"))
    >>> assert type(out) == dict

    """
    # List all contents
    config = configparser.ConfigParser()
    config.read(path)
    out = {}
    for section in config.sections():
        out[section] = {}
        for options in config.options(section):
            value = config.get(section, options)
            if re.match(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", value):
                if ' ' in value:
                    value = str(value)
                elif '.' in value or 'e' in value:
                    value = float(value)
                else:
                    value = int(value)
            elif value == 'True' or value == 'False':
                value = config.getboolean(section, options)
            else:
                value = str(value)
            out[section][options] = value
    return out


def from_config_to_parameters(path):
    """Convert config file keywords into spectractor.parameters parameters.

    Parameters
    ----------
    path: str
        The path to the config file.

    Examples
    --------

    >>> mypath = os.path.dirname(__file__)
    >>> from_config_to_parameters(os.path.join(mypath, parameters.CONFIG_DIR, "default.ini"))
    >>> assert parameters.OBS_NAME == "DEFAULT"

    """
    # List all contents
    d = from_config_to_dict(path)
    for section in d.keys():
        for options in d[section].keys():
            setattr(parameters, options.upper(), d[section][options])


def load_config(config_filename, rebin=True):
    """Load configuration parameters from a .ini config file.

    Parameters
    ----------
    config_filename: str
        The path to the config file.
    rebin: bool, optional
        If True, the parameters.REBIN parameter is used and every parameters are changed to comply with the REBIN value.
        If False, the parameters.REBIN parameter is skipped and set to 1.

    Examples
    --------
    >>> parameters.VERBOSE = True
    >>> load_config("./config/ctio.ini")
    >>> assert parameters.OBS_NAME == "CTIO"

    .. doctest:
        :hide:

        >>> load_config("./config/ctio.ini")
        >>> load_config("ctio.ini")
        >>> load_config("./config/unknown_file.ini")  #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: Config file ./config/unknown_file.ini does not exist.

    """
    my_logger = set_logger(__name__)
    mypath = os.path.dirname(__file__)
    if not os.path.isfile(os.path.join(mypath, parameters.CONFIG_DIR, "default.ini")):
        raise FileNotFoundError('Config file default.ini does not exist.')
    # Load the configuration file
    from_config_to_parameters(os.path.join(mypath, parameters.CONFIG_DIR, "default.ini"))

    if not os.path.isfile(config_filename):
        if not os.path.isfile(os.path.join(mypath, parameters.CONFIG_DIR, config_filename)):
            raise FileNotFoundError(f'Config file {config_filename} does not exist.')
        else:
            config_filename = os.path.join(mypath, parameters.CONFIG_DIR, config_filename)
    # Load the configuration file
    my_logger.info(f"\n\tLoading {config_filename} with {parameters.VERBOSE=}...")
    from_config_to_parameters(config_filename)

    # Derive other parameters
    update_derived_parameters()

    # Apply rebinning
    if parameters.CCD_REBIN > 1 and rebin:
        apply_rebinning_to_parameters()
    else:
        parameters.CCD_REBIN = 1

    # check consistency
    if parameters.PIXWIDTH_BOXSIZE > parameters.PIXWIDTH_BACKGROUND:
        raise ValueError(f'parameters.PIXWIDTH_BOXSIZE must be smaller than parameters.PIXWIDTH_BACKGROUND (or equal).')

    # check consistency
    if parameters.PSF_POLY_TYPE not in ["polynomial", "legendre"]:
        raise ValueError(f'parameters.PSF_POLY_TYPE must be either "polynomial" or "legendre". Got {parameters.PSF_POLY_TYPE=}')

    # check presence of atmospheric simulation packages
    if parameters.SPECTRACTOR_ATMOSPHERE_SIM.lower() not in ["none", "libradtran", "getobsatmo"]:
        raise ValueError(f'parameters.SPECTRACTOR_ATMOSPHERE_SIM must be either ["none", "libradtran", "getobsatmo"]. '
                         f'Got {parameters.SPECTRACTOR_ATMOSPHERE_SIM}.')
    if parameters.SPECTRACTOR_ATMOSPHERE_SIM.lower() == "libradtran":
        if not shutil.which("uvspec") and not os.path.isfile(os.path.join(parameters.LIBRADTRAN_DIR, 'bin/uvspec')):
            raise OSError(f"{parameters.SPECTRACTOR_ATMOSPHERE_SIM=} but uvspec executable not found in $PATH "
                          f"or {os.path.join(parameters.LIBRADTRAN_DIR, 'bin/')}")
    if parameters.SPECTRACTOR_ATMOSPHERE_SIM.lower() == "getobsatmo":
        try:
            import getObsAtmo
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"{parameters.SPECTRACTOR_ATMOSPHERE_SIM=} but getObsAtmo module not found.")

    # verbosity
    if parameters.VERBOSE or parameters.DEBUG:
        txt = ""
        # default.ini should be the config file with the most parameters
        config = configparser.ConfigParser()
        config.read(os.path.join(mypath, parameters.CONFIG_DIR, "default.ini"))
        for section in config.sections():
            txt += f"Section: {section}\n"
            for options in config.options(section):
                value = config.get(section, options)
                par = getattr(parameters, options.upper())
                txt += f"x {options}: {value}\t=> parameters.{options.upper()}: {par}\t {type(par)}\n"
        my_logger.info(f"Loaded {config_filename} with\n{txt}")

def update_derived_parameters():
    # Derive other parameters
    parameters.CALIB_BGD_NPARAMS = parameters.CALIB_BGD_ORDER + 1
    parameters.LAMBDAS = np.arange(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX, parameters.LAMBDA_STEP)
    if not isinstance(parameters.OBS_SURFACE, astropy.units.quantity.Quantity):
        parameters.OBS_SURFACE *= units.cm ** 2  # Surface of telescope
    # Units of SEDs in flam (erg/s/cm2/nm) :
    SED_UNIT = 1 * units.erg / units.s / units.cm ** 2 / units.nanometer
    parameters.FLAM_TO_ADURATE = ((parameters.OBS_SURFACE * SED_UNIT * units.s
                                   * units.nanometer ** 2 / (const.h * const.c) / parameters.CCD_GAIN).decompose()).value


def apply_rebinning_to_parameters():
    """Divide or multiply original parameters by parameters.CCD_REBIN to set them correctly
    in the case of an image rebinning.

    Examples
    --------
    >>> parameters.PIXWIDTH_SIGNAL = 40
    >>> parameters.CCD_PIXEL2MM = 10
    >>> parameters.CCD_REBIN = 2
    >>> apply_rebinning_to_parameters()
    >>> parameters.PIXWIDTH_SIGNAL
    20
    >>> parameters.CCD_PIXEL2MM
    20
    """
    # Apply rebinning
    parameters.PIXDIST_BACKGROUND = int(parameters.PIXDIST_BACKGROUND // parameters.CCD_REBIN)
    parameters.PIXWIDTH_BOXSIZE = int(max(10, parameters.PIXWIDTH_BOXSIZE // parameters.CCD_REBIN))
    parameters.PIXWIDTH_BACKGROUND = int(parameters.PIXWIDTH_BACKGROUND // parameters.CCD_REBIN)
    parameters.PIXWIDTH_SIGNAL = int(parameters.PIXWIDTH_SIGNAL // parameters.CCD_REBIN)
    parameters.CCD_IMSIZE = int(parameters.CCD_IMSIZE // parameters.CCD_REBIN)
    parameters.CCD_PIXEL2MM *= parameters.CCD_REBIN
    parameters.CCD_PIXEL2ARCSEC *= parameters.CCD_REBIN
    parameters.XWINDOW = int(parameters.XWINDOW // parameters.CCD_REBIN)
    parameters.YWINDOW = int(parameters.YWINDOW // parameters.CCD_REBIN)
    parameters.XWINDOW_ROT = int(parameters.XWINDOW_ROT // parameters.CCD_REBIN)
    parameters.YWINDOW_ROT = int(parameters.YWINDOW_ROT // parameters.CCD_REBIN)
    parameters.PSF_PIXEL_STEP_TRANSVERSE_FIT = int(parameters.PSF_PIXEL_STEP_TRANSVERSE_FIT // parameters.CCD_REBIN)
    update_derived_parameters()


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
    >>> from spectractor import parameters
    >>> parameters.VERBOSE = True
    >>> parameters.DEBUG = True
    >>> test = Test()
    >>> test.log()
    
    """
    my_logger = logging.getLogger(logger)
    my_format = "%(asctime)-20s %(name)-10s %(funcName)-20s %(levelname)-6s %(message)s"
    logging.basicConfig(format=my_format, level=logging.WARNING)
    if not parameters.CALLING_CODE:
        coloredlogs.DEFAULT_LEVEL_STYLES['warn'] = {'color': 'yellow'}
        coloredlogs.DEFAULT_FIELD_STYLES['levelname'] = {'color': 'white', 'bold': True}
    if parameters.VERBOSE > 0:
        my_logger.setLevel(logging.INFO)
        if not parameters.CALLING_CODE:
            coloredlogs.install(fmt=my_format, level=logging.INFO)
    else:
        my_logger.setLevel(logging.WARNING)
        if not parameters.CALLING_CODE:
            coloredlogs.install(fmt=my_format, level=logging.WARNING)
    if parameters.DEBUG_LOGGING:
        my_logger.setLevel(logging.DEBUG)
        if not parameters.CALLING_CODE:
            coloredlogs.install(fmt=my_format, level=logging.DEBUG)
    return my_logger


if __name__ == "__main__":
    import doctest

    doctest.testmod()

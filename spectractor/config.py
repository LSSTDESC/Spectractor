import configparser
import os
import sys
import re
import numpy as np
import coloredlogs
import logging
import astropy.units as units

from spectractor import parameters

logging.getLogger("matplotlib").setLevel(logging.ERROR)


def from_config_to_parameters(config):
    """Convert config file keywords into spectractor.parameters parameters.

    Parameters
    ----------
    config: ConfigParser
        The ConfigParser instance to convert

    Examples
    --------

    >>> config = configparser.ConfigParser()
    >>> config.read(os.path.join(parameters.CONFIG_DIR,"default.ini"))  # doctest: +ELLIPSIS
    ['/.../config/default.ini']
    >>> from_config_to_parameters(config)
    >>> assert parameters.OBS_NAME == "DEFAULT"

    """
    # List all contents
    for section in config.sections():
        for options in config.options(section):
            value = config.get(section, options)
            if re.match("[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", value):
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
            setattr(parameters, options.upper(), value)


def load_config(config_filename):
    """Load configuration parameters from a .ini config file.

    Parameters
    ----------
    config_filename: str
        The path to the config file.

    Examples
    --------

    >>> load_config("./config/ctio.ini")
    >>> assert parameters.OBS_NAME == "CTIO"

    .. doctest:
        :hide:

        >>> load_config("./config/unknown_file.ini")
        Traceback (most recent call last):
        ...
        SystemExit: Config file ./config/unknown_file.ini does not exist.
        >>> load_config("./config/ctio.ini")

    """
    if not os.path.isfile(os.path.join(parameters.CONFIG_DIR, "default.ini")):
        raise FileNotFoundError('Config file default.ini does not exist.')
    # Load the configuration file
    config = configparser.ConfigParser()
    config.read(os.path.join(parameters.CONFIG_DIR, "default.ini"))
    from_config_to_parameters(config)

    if not os.path.isfile(config_filename):
        raise FileNotFoundError(f'Config file {config_filename} does not exist.')
    # Load the configuration file
    config = configparser.ConfigParser()
    config.read(config_filename)
    from_config_to_parameters(config)

    # Derive other parameters
    parameters.MY_FORMAT = "%(asctime)-20s %(name)-10s %(funcName)-20s %(levelname)-6s %(message)s"
    logging.basicConfig(format=parameters.MY_FORMAT, level=logging.WARNING)
    mypath = os.path.dirname(__file__)
    parameters.DISPERSER_DIR = os.path.join(mypath, parameters.DISPERSER_DIR)
    parameters.THROUGHPUT_DIR = os.path.join(mypath, parameters.THROUGHPUT_DIR)
    parameters.CCD_ARCSEC2RADIANS = np.pi / (180. * 3600.)  # conversion factor from arcsec to radians
    parameters.OBS_DIAMETER = parameters.OBS_DIAMETER * units.m  # Diameter of the telescope
    parameters.OBS_SURFACE = np.pi * parameters.OBS_DIAMETER ** 2 / 4.  # Surface of telescope
    parameters.LAMBDAS = np.arange(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX, 1)
    parameters.FLAM_TO_ADURATE = ((parameters.OBS_SURFACE * parameters.SED_UNIT * parameters.TIME_UNIT
                                   * parameters.wl_dwl_unit / parameters.hc / parameters.CCD_GAIN).decompose()).value
    parameters.CALIB_BGD_NPARAMS = parameters.CALIB_BGD_ORDER + 1

    # check consistency
    if parameters.PIXWIDTH_BOXSIZE > parameters.PIXWIDTH_BACKGROUND:
        sys.exit(f'parameters.PIXWIDTH_BOXSIZE must be smaller than parameters.PIXWIDTH_BACKGROUND (or equal).')

    if parameters.VERBOSE:
        for section in config.sections():
            print(f"Section: {section}")
            for options in config.options(section):
                value = config.get(section, options)
                par = getattr(parameters, options.upper())
                print(f"x {options}: {value}\t=> parameters.{options.upper()}: {par}\t {type(par)}")


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
    coloredlogs.DEFAULT_LEVEL_STYLES['warn'] = {'color': 'yellow'}
    coloredlogs.DEFAULT_FIELD_STYLES['levelname'] = {'color': 'white', 'bold': True}
    if parameters.VERBOSE > 0:
        my_logger.setLevel(logging.INFO)
        coloredlogs.install(fmt=parameters.MY_FORMAT, level=logging.INFO)
    else:
        my_logger.setLevel(logging.WARNING)
        coloredlogs.install(fmt=parameters.MY_FORMAT, level=logging.WARNING)
    if parameters.DEBUG:
        my_logger.setLevel(logging.DEBUG)
        coloredlogs.install(fmt=parameters.MY_FORMAT, level=logging.DEBUG)
    return my_logger


if __name__ == "__main__":
    import doctest

    doctest.testmod()


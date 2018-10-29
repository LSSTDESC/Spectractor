import configparser
import os
import re
import numpy as np
import coloredlogs
import logging
import astropy.units as units

from spectractor import parameters

logging.getLogger("matplotlib").setLevel(logging.ERROR)


def load_config(config_filename):
    # Load the configuration file
    config = configparser.ConfigParser()
    config.read(config_filename)

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

    # Derive other parameters
    parameters.MY_FORMAT = "%(asctime)-20s %(name)-10s %(funcName)-20s %(levelname)-6s %(message)s"
    logging.basicConfig(format=parameters.MY_FORMAT, level=logging.WARNING)
    mypath = os.path.dirname(__file__)
    parameters.HOLO_DIR = os.path.join(mypath, parameters.HOLO_DIR)
    parameters.THROUGHPUT_DIR = os.path.join(mypath, parameters.THROUGHPUT_DIR)
    parameters.CCD_ARCSEC2RADIANS = np.pi / (180. * 3600.)  # conversion factor from arcsec to radians
    parameters.OBS_DIAMETER = parameters.OBS_DIAMETER * units.m  # Diameter of the telescope
    parameters.OBS_SURFACE = np.pi * parameters.OBS_DIAMETER ** 2 / 4.  # Surface of telescope
    parameters.LAMBDAS = np.arange(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX, 1)
    parameters.FLAM_TO_ADURATE = ((parameters.OBS_SURFACE * parameters.SED_UNIT * parameters.TIME_UNIT
                                   * parameters.wl_dwl_unit / parameters.hc / parameters.CCD_GAIN
                                   * parameters.g_disperser_ronchi).decompose()).value
    parameters.BGD_NPARAMS = parameters.BGD_ORDER + 1

    if parameters.DEBUG:
        for section in config.sections():
            print("Section: %s" % section)
            for options in config.options(section):
                value = config.get(section, options)
                par = getattr(parameters, options.upper())
                print(f"x {options}: {value}\t => parameters.{options.upper()}: {par}\t {type(par)}")


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

    load_config('../config/ctio.ini')

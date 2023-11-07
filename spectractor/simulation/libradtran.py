################################################################
#
# Script to evaluate air transparency with LibRadTran
# With a pure absorbing atmosphere
# author: sylvielsstfr, jeremy.neveu
# creation date : November 2nd 2016
# update : July 2018
#
#################################################################
import os
import io
import sys
import shutil
import numpy as np

import subprocess

from spectractor.tools import ensure_dir
import spectractor.parameters as parameters
from spectractor.config import set_logger


class Libradtran:

    def __init__(self, libradtran_path="", atm_standard="afglus", absorption_model="reptran"):
        """Initialize the Libradtran settings for Spectractor.

        Parameters
        ----------
        libradtran_path: str, optional
            The path to the directory where libradtran directory is (default: '').
        atm_standard: str, optional
            Short name of atmospheric sky (default: afglus, US standard).
        absorption_model: str, optional
            Name of model for absorption bands: can be reptran, lowtran, kato, kato2, fu, crs (default: reptran).

        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.settings = {}
        if absorption_model not in ["reptran", "lowtran", "kato"," kato2", "fu", "crs"]:
            raise ValueError(f"absorption_model={absorption_model}: value must be either "
                             f"reptran, lowtran, kato, kato2, fu or crs.")
        if libradtran_path != "":
            self.libradtran_path = ""
        elif os.getenv("CONDA_PREFIX") != "" and os.path.isdir(
            os.path.join(os.getenv("CONDA_PREFIX"), "share/libRadtran/data")):
            self.libradtran_path = os.path.join(os.getenv("CONDA_PREFIX"), "share/libRadtran/")
        elif parameters.LIBRADTRAN_DIR != "":
            if not os.path.isdir(parameters.LIBRADTRAN_DIR):
                # reset libradtran path
                if 'LIBRADTRAN_DIR' in os.environ:
                    self.my_logger.warning(f"Reset parameters.LIBRADTRAN_DIR={parameters.LIBRADTRAN_DIR} (not found) "
                                           f"to {os.getenv('LIBRADTRAN_DIR')}.")
                    parameters.LIBRADTRAN_DIR = os.getenv('LIBRADTRAN_DIR') + '/'
                else:
                    self.my_logger.error(f"parameters.LIBRADTRAN_DIR={parameters.LIBRADTRAN_DIR} but directory does "
                                         f"not exist and LIBRADTRAN_DIR is not in OS environment.")
                    raise OSError("No Libtradtran library found with parameters.LIBRADTRAN_DIR or LIBRADTRAN_DIR environment.")
            self.libradtran_path = parameters.LIBRADTRAN_DIR
        else:
            self.my_logger.warning(f"\n\tYou should set a LIBRADTRAN_DIR environment variable (={parameters.LIBRADTRAN_DIR})"
                                   f" or give a path to the Libradtran class (={libradtran_path}) or install rubin-libradtran package.")
        self.proc = 'as'  # Absorption + Rayleigh + aerosols special
        self.equation_solver = 'pp'  # pp for parallel plane or ps for pseudo-spherical
        self.atmosphere_standard = atm_standard  # short name of atmospheric sky
        self.absorption_model = absorption_model  # absorption model

    def run(self, path=''):
        """Run the Libradtran command uvspec.

        Parameters
        ----------
        path: str, optional
            Path to bin/uvspec if necessary, otherwise use $PATH (default: "")

        Returns
        -------
        lambdas: array_like
            Wavelength array.
        atmosphere: array_like
            Atmospheric transmission array.
        """
        if shutil.which("uvspec"):
            cmd = shutil.which("uvspec")
        elif path != '':
            cmd = os.path.join(path, 'bin/uvspec')
        else:
            raise OSError(f"uvspec executable not found in $PATH or {os.path.join(path, 'bin/uvspec')}")

        inputstr = '\n'.join([f'{name} {self.settings[name]}' for name in self.settings.keys()])
        try:
            process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     input=inputstr, encoding='ascii')
            return np.genfromtxt(io.StringIO(process.stdout)).T
        except subprocess.CalledProcessError as e:  # pragma: nocover
            self.my_logger.warning(f"\n\tLibradtran input command:\n{inputstr}")
            self.my_logger.error(f"\n\t{e.stderr}")
            sys.exit()

    def simulate(self, airmass, aerosol, ozone, pwv, pressure, angstrom_exponent=None, lambda_min=250, lambda_max=1200,
                 altitude=parameters.OBS_ALTITUDE):
        """Simulate the atmosphere transmission with Libratran.

        Parameters
        ----------
        airmass: float
            The airmass of the atmosphere.
        aerosol: float
            VAOD of the aerosols.
        ozone: float
            Ozone quantity in Dobson units.
        pwv: float
            Precipitable Water Vapor amount in mm.
        pressure: float
            Pressure of the atmosphere at observatory altitude in hPa.
        angstrom_exponent: float, optional
            Angstrom exponent for aerosols. If negative or None, default aerosol model from Libradtran is used.
            If value is 0.0192, the atmospheric transmission is very close to the case with angstrom_exponent=None (default: None).
        lambda_min: float
            Minimum wavelength for simulation in nm.
        lambda_max: float
            Maximum wavelength for simulation in nm.
        altitude: float
            Observatory altitude in km (default: parameters.OBS_ALTITUDE).

        Returns
        -------
        output_file_name: str
            The output file name relative to the current directory.

        Examples
        --------
        >>> parameters.DEBUG = True
        >>> lib = Libradtran()
        >>> lambdas, atmosphere = lib.simulate(1.2, 0.07, 400, 2, 800, angstrom_exponent=None, lambda_max=1200)
        >>> print(lambdas[-5:])
        [1196. 1197. 1198. 1199. 1200.]
        >>> print(atmosphere[-5:])
        [0.9617202 0.9617202 0.9529933 0.9529933 0.9512588]
        >>> lambdas2, atmosphere2 = lib.simulate(1.2, 0.07, 400, 2, 800, angstrom_exponent=-0.02, lambda_max=1200)
        >>> print(lambdas2[-5:])
        [1196. 1197. 1198. 1199. 1200.]
        >>> print(atmosphere2[-5:])
        [0.9659722 0.9659722 0.9571998 0.9571998 0.9554523]
        """

        self.my_logger.debug(
            f'\n\t--------------------------------------------'
            f'\n\tevaluate'
            f'\n\t 1) airmass = {airmass}'
            f'\n\t 2) pwv = {pwv}'
            f'\n\t 3) ozone = {ozone}'
            f'\n\t 4) aer = {aerosol}'
            f'\n\t 5) pressure =  {pressure}'
            f'\n\t--------------------------------------------')

        # Set up type of run
        if self.proc == 'sc':
            runtype = 'no_absorption'
        elif self.proc == 'ab':
            runtype = 'no_scattering'
        elif self.proc == 'ae':
            runtype = 'aerosol_default'
        elif self.proc == 'as':
            runtype = 'aerosol_special'
        else:
            runtype = 'clearsky'

        #   Selection of RTE equation solver
        if self.equation_solver == 'pp':  # parallel plan
            equation_solver_equations = 'twostr'
        elif self.equation_solver == 'ps':  # pseudo spherical
            equation_solver_equations = 'sdisort'
        else:
            self.my_logger.error(f'Unknown RTE equation solver {self.equation_solver}.')
            sys.exit()

        # loop on molecular model resolution
        # molecular_resolution = np.array(['coarse','medium','fine'])
        # select only COARSE Model
        molecular_resolution = 'coarse'

        self.settings["data_files_path"] = os.path.join(self.libradtran_path, 'data')
        self.settings["atmosphere_file"] = os.path.join(self.libradtran_path, 'data/atmmod/', self.atmosphere_standard + '.dat')
        self.settings["albedo"] = '0.2'
        self.settings["rte_solver"] = equation_solver_equations

        if self.absorption_model == 'reptran':
            self.settings["mol_abs_param"] = self.absorption_model + ' ' + molecular_resolution
        else:
            self.settings["mol_abs_param"] = self.absorption_model

        # Convert airmass into zenith angle
        sza = np.arccos(1. / airmass) * 180. / np.pi

        # Should be no_absorption
        if runtype == 'aerosol_default':
            self.settings["aerosol_default"] = ''
        elif runtype == 'aerosol_special':
            self.settings["aerosol_default"] = ''
            if angstrom_exponent is None or angstrom_exponent > 0:
                self.settings["aerosol_set_tau_at_wvl"] = f'500 {aerosol:.20f}'
            else:
                # below formula recover default aerosols models with angstrom_exponent=0.0192
                tau = aerosol / 0.04 * (0.5 ** -angstrom_exponent)
                self.settings["aerosol_angstrom"] = f"{tau:.10f} {-angstrom_exponent:.10f}"

        if runtype == 'no_scattering':
            self.settings["no_scattering"] = ''
        if runtype == 'no_absorption':
            self.settings["no_absorption"] = ''

        # water vapor
        self.settings["mol_modify H2O"] = f'{pwv:.20f} MM'

        # Ozone
        self.settings["mol_modify O3"] = f'{ozone:.20f} DU'

        # rescale pressure if reasonable pressure values are provided
        if 600. < pressure < 1015.:
            self.settings["pressure"] = pressure
        else:
            self.my_logger.error(f'\n\tcrazy pressure p={pressure} hPa')
        # only for mie executable from libradtran to compute mie diffusion
        # self.settings["temperature"] = temperature + 273.15

        self.settings["altitude"] = str(altitude)  # observatory altitude
        self.settings["source"] = 'solar ' + os.path.join(self.libradtran_path, 'data/solar_flux/kurudz_1.0nm.dat')
        self.settings["sza"] = str(sza)
        self.settings["phi0"] = '0'
        self.settings["wavelength"] = f'{int(lambda_min)} {int(np.ceil(lambda_max))}'
        self.settings["output_user"] = 'lambda edir'
        self.settings["output_quantity"] = 'reflectivity'  # transmittance
        self.settings["quiet"] = ''

        wl, atm = self.run(path=self.libradtran_path)
        return wl, atm


def clean_simulation_directory():
    """Remove the libradtran directory.

    Examples
    --------
    >>> ensure_dir('libradtran')
    >>> clean_simulation_directory()
    >>> assert not os.path.isfile('libradtran')
    """
    os.system("rm -rf libradtran")


if __name__ == "__main__":
    import doctest

    doctest.testmod()

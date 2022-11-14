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
import numpy as np

import subprocess

from spectractor.tools import ensure_dir
import spectractor.parameters as parameters
from spectractor.config import set_logger


class Libradtran:

    def __init__(self, home=''):
        """Initialize the Libradtran settings for Spectractor.

        Parameters
        ----------
        home: str, optional
            The path to the directory where libradtran directory is. If not specified $HOME is taken (default: '').
        """
        self.my_logger = set_logger(self.__class__.__name__)
        if home == '':
            self.home = os.environ['HOME']
        else:
            self.home = home
        self.settings = {}

        # Definitions and configuration
        # -------------------------------------

        # LibRadTran installation directory
        self.simulation_directory = 'libradtran'
        ensure_dir(self.simulation_directory)
        self.libradtran_path = parameters.LIBRADTRAN_DIR

        # Filename : RT_LS_pp_us_sa_rt_z15_wv030_oz30.txt
        #          : Prog_Obs_Rte_Atm_proc_Mod_zXX_wv_XX_oz_XX
        self.Prog = 'RT'  # definition the simulation program is libRadTran
        self.proc = 'as'  # Absoprtion + Rayleigh + aerosols special
        self.equation_solver = 'pp'  # pp for parallel plane or ps for pseudo-spherical
        self.Atm = 'afglus'  # short name of atmospheric sky here US standard
        self.Proc = 'sa'  # light interaction processes : sc for pure scattering,ab for pure absorption
        # sa for scattering and absorption, ae with aerosols default, as with aerosol special
        self.Mod = 'rt'  # Models for absorption bands : rt for REPTRAN, lt for LOWTRAN, k2 for Kato2

    def run(self, path=''):
        """Run the libratran command uvpsec.

        Parameters
        ----------
        path: str, optional
            Path to bin/uvpsec if necessary, otherwise use  self.home (default: "")

        Returns
        -------
        lambdas: array_like
            Wavelength array.
        atmosphere: array_like
            Atmospheric transmission array.
        """
        if path != '':
            cmd = os.path.join(path, 'bin/uvspec')
        else:
            cmd = os.path.join(self.home, '/libRadtran/bin/uvspec')

        inputstr = '\n'.join(['{} {}'.format(name, self.settings[name])
                              for name in self.settings.keys()])

        process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 input=inputstr, encoding='ascii')
        return np.genfromtxt(io.StringIO(process.stdout)).T

    def simulate(self, airmass, pwv, ozone, aerosol, pressure, lambda_min=250, lambda_max=1200):
        """Simulate the atmosphere transmission with Libratran.

        Parameters
        ----------
        airmass: float
            The airmass of the atmosphere.
        pwv: float
            Precipitable Water Vapor amount in mm.
        ozone: float
            Ozone quantity in Dobson units.
        aerosol: float
            VAOD of the aerosols.
        pressure: float
            Pressure of the atmosphere in hPa.
        lambda_min: float
            Minimum wavelength for simulation in nm.
        lambda_max: float
            Maximum wavelength for simulation in nm.

        Returns
        -------
        output_file_name: str
            The output file name relative to the current directory.

        Examples
        --------
        >>> parameters.DEBUG = True
        >>> lib = Libradtran()
        >>> lambdas, atmosphere = lib.simulate(1.2, 2, 400, 0.07, 800, lambda_max=1200)
        >>> print(lambdas[-5:])
        [1196. 1197. 1198. 1199. 1200.]
        >>> print(atmosphere[-5:])
        [0.9617202 0.9617202 0.9529933 0.9529933 0.9512588]
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
            equation_solver_equations = 'disort'
        elif self.equation_solver == 'ps':  # pseudo spherical
            equation_solver_equations = 'sdisort'
        else:
            self.my_logger.error(f'Unknown RTE equation solver {self.equation_solver}.')
            sys.exit()

        #   Selection of absorption model
        absorption_model = 'reptran'
        if self.Mod == 'rt':
            absorption_model = 'reptran'
        if self.Mod == 'lt':
            absorption_model = 'lowtran'
        if self.Mod == 'kt':
            absorption_model = 'kato'
        if self.Mod == 'k2':
            absorption_model = 'kato2'
        if self.Mod == 'fu':
            absorption_model = 'fu'
        if self.Mod == 'cr':
            absorption_model = 'crs'

        # loop on molecular model resolution
        # molecular_resolution = np.array(['coarse','medium','fine'])
        # select only COARSE Model
        molecular_resolution = 'coarse'

        self.settings["data_files_path"] = self.libradtran_path + 'data'

        self.settings["atmosphere_file"] = os.path.join(self.libradtran_path, 'data/atmmod/', self.Atm+'.dat')
        self.settings["albedo"] = '0.2'

        self.settings["rte_solver"] = equation_solver_equations

        if self.Mod == 'rt':
            self.settings["mol_abs_param"] = absorption_model + ' ' + molecular_resolution
        else:
            self.settings["mol_abs_param"] = absorption_model

        # Convert airmass into zenith angle
        sza = np.arccos(1. / airmass) * 180. / np.pi

        # Should be no_absorption
        aerosol_string = f'500 {aerosol:.20f}'
        if runtype == 'aerosol_default':
            self.settings["aerosol_default"] = ''
        elif runtype == 'aerosol_special':
            self.settings["aerosol_default"] = ''
            self.settings["aerosol_set_tau_at_wvl"] = aerosol_string

        if runtype == 'no_scattering':
            self.settings["no_scattering"] = ''
        if runtype == 'no_absorption':
            self.settings["no_absorption"] = ''

        # water vapor
        self.settings["mol_modify H2O"] = f'{pwv:.20f} MM'

        # Ozone
        self.settings["mol_modify O3"] = f'{ozone:.20f} DU'

        # rescale pressure   if reasonable pressure values are provided
        if 600. < pressure < 1015.:
            self.settings["pressure"] = pressure
        else:
            self.my_logger.error(f'\n\tcrazy pressure p={pressure} hPa')

        self.settings["output_user"] = 'lambda edir'
        self.settings["altitude"] = str(parameters.OBS_ALTITUDE)  # Altitude LSST observatory
        self.settings["source"] = 'solar ' + os.path.join(self.libradtran_path, 'data/solar_flux/kurudz_1.0nm.dat')
        self.settings["sza"] = str(sza)
        self.settings["phi0"] = '0'
        self.settings["wavelength"] = f'{lambda_min} {lambda_max}'
        self.settings["output_quantity"] = 'reflectivity'  # 'transmittance' #
        self.settings["quiet"] = ''

        # airmass
        # airmass_index = int(airmass * 10)
        # pwv_index = int(10 * pwv)
        # ozone_index = int(ozone / 10.)
        # aerosol_index = int(aerosol * 100.)

        # base_filename_part1 = self.Prog + '_' + parameters.OBS_NAME + '_' + self.equation_solver + '_'
        # base_filename = f'{base_filename_part1}{self.Atm}_{self.proc}_{self.Mod}_z{airmass_index}' \
        #                 f'_pwv{pwv_index}_oz{ozone_index}_aer{aerosol_index}'

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

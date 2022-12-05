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
import re
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
        self.Atm = ['us']  # short name of atmospheric sky here US standard and  Subarctic winter
        self.Proc = 'sa'  # light interaction processes : sc for pure scattering,ab for pure absorption
        # sa for scattering and absorption, ae with aerosols default, as with aerosol special
        self.Mod = 'rt'  # Models for absorption bands : rt for REPTRAN, lt for LOWTRAN, k2 for Kato2

    def write_input(self, filename):
        f = open(filename, 'w')
        for key in sorted(self.settings):
            if key == "mol_modify2":
                f.write("mol_modify" + ' ' + str(self.settings[key]) + '\n')
            else:
                f.write(key + ' ' + str(self.settings[key]) + '\n')
        f.close()

    def run(self, inp, out, path=''):
        """Run the libratran command uvpsec.

        Parameters
        ----------
        inp: str
            Input file.
        out: str
            Output file.
        path: str, optional
            Path to bin/uvpsec if necessary, otherwise use  self.home (default: "")
        """
        if path != '':
            cmd = os.path.join(path, 'bin/uvspec') + ' < ' + inp + ' > ' + out
        else:
            cmd = os.path.join(self.home, '/libRadtran/bin/uvspec') + ' < ' + inp + ' > ' + out
        subprocess.run(cmd, shell=True, check=True)

    def simulate(self, airmass, pwv, ozone, aerosol, pressure):
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

        Returns
        -------
        output_file_name: str
            The output file name relative to the current directory.

        Examples
        --------
        >>> parameters.DEBUG = True
        >>> lib = Libradtran()
        >>> output = lib.simulate(1.2, 2, 400, 0.07, 800)
        >>> print(output)
        libradtran/pp/us/as/rt/in/RT_CTIO_pp_us_as_rt_z12_pwv20_oz40_aer7.OUT
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

        # build the part 1 of file_name
        base_filename_part1 = self.Prog + '_' + parameters.OBS_NAME + '_' + self.equation_solver + '_'

        aerosol_string = f'500 {aerosol:.20f}'
        # aerosol_str=str(wl0_num)+ ' '+str(tau0_num)
        aerosol_index = int(aerosol * 100.)

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

        # for simulation select only two atmosphere
        # atmospheres = np.array(['afglus','afglms','afglmw','afglt','afglss','afglsw'])
        atmosphere_map = dict()  # map atmospheric names to short names
        atmosphere_map['afglus'] = 'us'
        atmosphere_map['afglms'] = 'ms'
        atmosphere_map['afglmw'] = 'mw'
        atmosphere_map['afglt'] = 'tp'
        atmosphere_map['afglss'] = 'ss'
        atmosphere_map['afglsw'] = 'sw'

        atmospheres = []
        for skyindex in self.Atm:
            if re.search('us', skyindex):
                atmospheres.append('afglus')
            if re.search('sw', skyindex):
                atmospheres.append('afglsw')

        output_directory, output_filename = None, None
        # 1) LOOP ON ATMOSPHERE
        for atmosphere in atmospheres:
            # if atmosphere != 'afglus':  # just take us standard sky
            #    break
            atmkey = atmosphere_map[atmosphere]

            # manage settings and output directories
            topdir = f'{self.simulation_directory}/{self.equation_solver}/{atmkey}/{self.proc}/{self.Mod}'
            ensure_dir(topdir)
            input_directory = topdir + '/' + 'in'
            ensure_dir(input_directory)
            output_directory = topdir + '/' + 'out'
            ensure_dir(output_directory)

            # loop on molecular model resolution
            # molecular_resolution = np.array(['coarse','medium','fine'])
            # select only COARSE Model
            molecular_resolution = 'coarse'

            # water vapor
            pwv_str = f'H2O {pwv:.20f} MM'
            pwv_index = int(10 * pwv)

            # airmass
            airmass_index = int(airmass * 10)

            # Ozone
            oz_str = f'O3 {ozone:.20f} DU'
            ozone_index = int(ozone / 10.)

            base_filename = f'{base_filename_part1}{atmkey}_{self.proc}_{self.Mod}_z{airmass_index}' \
                            f'_pwv{pwv_index}_oz{ozone_index}_aer{aerosol_index}'

            self.settings["data_files_path"] = self.libradtran_path + 'data'

            self.settings["atmosphere_file"] = self.libradtran_path + 'data/atmmod/' + atmosphere + '.dat'
            self.settings["albedo"] = '0.2'

            self.settings["rte_solver"] = equation_solver_equations

            if self.Mod == 'rt':
                self.settings["mol_abs_param"] = absorption_model + ' ' + molecular_resolution
            else:
                self.settings["mol_abs_param"] = absorption_model

            # Convert airmass into zenith angle
            sza = np.arccos(1. / airmass) * 180. / np.pi

            # Should be no_absorption
            if runtype == 'aerosol_default':
                self.settings["aerosol_default"] = ''
            elif runtype == 'aerosol_special':
                self.settings["aerosol_default"] = ''
                self.settings["aerosol_set_tau_at_wvl"] = aerosol_string

            if runtype == 'no_scattering':
                self.settings["no_scattering"] = ''
            if runtype == 'no_absorption':
                self.settings["no_absorption"] = ''

            # set up the ozone value
            self.settings["mol_modify"] = pwv_str
            self.settings["mol_modify2"] = oz_str

            # rescale pressure   if reasonable pressure values are provided
            if 600. < pressure < 1015.:
                self.settings["pressure"] = pressure
            else:
                self.my_logger.error(f'\n\tcrazy pressure p={pressure} hPa')

            self.settings["output_user"] = 'lambda edir'
            self.settings["altitude"] = str(parameters.OBS_ALTITUDE)  # Altitude LSST observatory
            self.settings["source"] = 'solar ' + self.libradtran_path + 'data/solar_flux/kurudz_1.0nm.dat'
            self.settings["sza"] = str(sza)
            self.settings["phi0"] = '0'
            self.settings["wavelength"] = '250.0 1200.0'
            self.settings["output_quantity"] = 'reflectivity'  # 'transmittance' #
            self.settings["quiet"] = ''

            input_filename = os.path.join(input_directory, base_filename + '.INP')
            output_filename = os.path.join(input_directory, base_filename + '.OUT')

            self.write_input(input_filename)
            self.run(input_filename, output_filename, path=self.libradtran_path)

        return output_filename


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

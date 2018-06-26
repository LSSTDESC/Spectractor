################################################################
#
# Script to simulate air transparency with LibRadTran
# With a pure absorbing atmosphere
# Here we vary PWV
# author: sylvielsstfr, jeremy.neveu
# creation date : November 2nd 2016
# update : July 2018
#
#################################################################
import os
import re
import sys
import numpy as np

from subprocess import Popen, PIPE

from spectractor.tools import ensure_dir
import spectractor.parameters as parameters


class Libradtran:

    def __init__(self, home=''):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        if home == '':
            self.home = os.environ['HOME']
        else:
            self.home = home
        self.settings = {}

        # Definitions and configuration
        # -------------------------------------

        # LibRadTran installation directory
        self.simulation_directory = 'simulations'
        ensure_dir(self.simulation_directory)
        self.libradtran_path = os.getenv('LIBRADTRANDIR') + '/'

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
        if parameters.DEBUG:
            self.my_logger.info("Running uvspec with settings file: ", inp)
            self.my_logger.info("Output to file                : ", out)
        if path != '':
            cmd = path + 'bin/uvspec ' + ' < ' + inp + ' > ' + out
        else:
            cmd = self.home + '/libRadtran/bin/uvspec ' + ' < ' + inp + ' > ' + out
        if parameters.DEBUG:
            self.my_logger.info("uvspec cmd: ", cmd)
        #        p   = call(cmd,shell=True,stdin=PIPE,stdout=PIPE)
        p = Popen(cmd, shell=True, stdout=PIPE)
        p.wait()

    def simulate(self, airmass, pwv, ozone, aerosol, pressure):
        """
        ProcessSimulationaer(airmass,pwv,ozone,aerosol,pressure)
        with aerosol simulation is performed
        default profile
        """

        if parameters.DEBUG:
            print('--------------------------------------------')
            print('simulate')
            print(' 1) airmass = ', airmass)
            print(' 2) pwv = ', pwv)
            print(' 3) ozone = ', ozone)
            print(' 4) aer = ', aerosol)
            print(' 5) pressure =', pressure)
            print('--------------------------------------------')

        # build the part 1 of filename
        base_filename_part1 = self.Prog + '_' + parameters.OBS_NAME + '_' + self.equation_solver + '_'

        aerosol_string = '500 ' + str(aerosol)
        # aerosol_str=str(wl0_num)+ ' '+str(tau0_num)
        aerosol_index = int(aerosol * 100.)

        # Set up type of run
        runtype = 'aerosol_special'  # 'no_scattering' #aerosol_special #aerosol_default# #'clearsky'#

        if self.proc == 'sc':
            runtype = 'no_absorption'
            outtext = 'no_absorption'
        elif self.proc == 'ab':
            runtype = 'no_scattering'
            outtext = 'no_scattering'
        elif self.proc == 'ae':
            runtype = 'aerosol_default'
            outtext = 'aerosol_default'
        elif self.proc == 'as':
            runtype = 'aerosol_special'
            outtext = 'aerosol_special'
        else:
            runtype == 'clearsky'
            outtext = 'clearsky'

        #   Selection of RTE equation solver
        if self.equation_solver == 'pp':  # parallel plan
            equation_solver_equations = 'disort'
        elif self.equation_solver == 'ps':  # pseudo spherical
            equation_solver_equations = 'sdisort'
        else:
            sys.exit(f'Unknown RTE equation solver {self.equation_solver}.')

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
            topdir = self.simulation_directory + '/' + self.equation_solver + '/' + atmkey + '/' + self.proc + '/' + self.Mod
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
            pwv_str = 'H2O ' + str(pwv) + ' MM'
            pwv_index = int(10 * pwv)

            # airmass
            airmass_index = int(airmass * 10)

            # Ozone
            oz_str = 'O3 ' + str(ozone) + ' DU'
            ozone_index = int(ozone / 10.)

            base_filename = base_filename_part1 + atmkey + '_' + self.proc + '_' + self.Mod + '_z' + \
                            str(airmass_index) + '_pwv' + str(pwv_index) + '_oz' + str(ozone_index) + \
                            '_aer' + str(aerosol_index)

            verbose = parameters.DEBUG

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
                print("crazy pressure p=", pressure, ' hPa')

            self.settings["output_user"] = 'lambda edir'
            self.settings["altitude"] = str(parameters.OBS_ALTITUDE)  # Altitude LSST observatory
            self.settings["source"] = 'solar ' + self.libradtran_path + 'data/solar_flux/kurudz_1.0nm.dat'
            self.settings["sza"] = str(sza)
            self.settings["phi0"] = '0'
            self.settings["wavelength"] = '250.0 1200.0'
            self.settings["output_quantity"] = 'reflectivity'  # 'transmittance' #
            #       self.settings["verbose"] = ''
            self.settings["quiet"] = ''

            if "output_quantity" in list(self.settings.keys()):
                outtextfinal = outtext + '_' + self.settings["output_quantity"]

            input_filename = os.path.join(input_directory, base_filename + '.INP')
            output_filename = os.path.join(input_directory, base_filename + '.OUT')

            self.write_input(input_filename)
            self.run(input_filename, output_filename, path=self.libradtran_path)

        return output_filename


def clean_simulation_directory():
    os.system("rm -rf simulations")

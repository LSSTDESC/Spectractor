from numpy.testing import run_module_suite

from spectractor import parameters
from spectractor.simulation.simulator import Simulator, SimulatorSimGrid, Atmosphere, AtmosphereGrid
import os


def test_simulator():
    file_names = ['tests/data/reduc_20170530_134_spectrum.fits']

    parameters.VERBOSE = True

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        spectrum_simulation = Simulator(file_name, pwv=3, ozone=350, aerosols=0.02,
                                        A1=1.1, A2=0.1, reso=2, D=56, shift=-3)
        SimulatorSimGrid(file_name, './tests/data/', pwv_grid=[0,10,2], ozone_grid=[200,400,2], aerosol_grid=[0,0.1,2])
        atmgrid = AtmosphereGrid(file_name, file_name.replace('spectrum', 'atmsim'))
        atm = Atmosphere(atmgrid.airmass, atmgrid.pressure, atmgrid.temperature)
        assert spectrum_simulation.data is not None
        #assert os.path.isfile('./outputs/' + tag.replace('.fits', '_sim.fits')) is True


if __name__ == "__main__":
    run_module_suite()

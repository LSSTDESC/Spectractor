import matplotlib as mpl
mpl.use('Agg')  # must be run first! But therefore requires noqa E402 on all other imports

import numpy as np  # noqa: E402

from spectractor import parameters  # noqa: E402
from spectractor.extractor.spectrum import Spectrum  # noqa: E402
from spectractor.simulation.simulator import (SpectrumSimulation, Atmosphere, AtmosphereGrid)  # noqa: E402
from spectractor.tools import uvspec_available  # noqa: E402
from spectractor.simulation.simulator import (Atmosphere, AtmosphereGrid)  # noqa: E402
from spectractor.config import load_config  # noqa: E402
import os  # noqa: E402
import unittest  # noqa: E402


# TODO: DM-33441 Fix broken spectractor tests
@unittest.skipIf(uvspec_available() is False, 'Skipping to avoid libradtran dependency')
def test_atmosphere():
    a = Atmosphere(airmass=1.2, pressure=800, temperature=5)
    transmission = a.simulate(ozone=400, pwv=5, aerosols=0.05)
    assert transmission is not None
    assert a.transmission(500) > 0
    assert a.ozone == 400
    assert a.pwv == 5
    assert a.aerosols == 0.05
    a.plot_transmission()

    a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits', pwv_grid=[5, 5, 1],
                       ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
    atmospheric_grid = a.compute()
    assert np.sum(atmospheric_grid) > 0
    assert np.all(np.isclose(a.atmgrid[0, a.index_atm_data:], parameters.LAMBDAS))
    assert not np.any(np.isclose(a.atmgrid[1, a.index_atm_data:], np.zeros_like(parameters.LAMBDAS), rtol=1e-6))
    assert a.atmgrid.shape == (3, a.index_atm_data + len(parameters.LAMBDAS))
    a.save_file(a.image_filename.replace('.fits', '_atmsim.fits'))
    assert os.path.isfile('tests/data/reduc_20170605_028_atmsim.fits')
    a.load_file(a.image_filename.replace('.fits', '_atmsim.fits'))
    assert np.all(a.AER_Points == np.array([0., 0.1]))
    assert np.all(a.PWV_Points == np.array([5.]))
    assert np.all(a.OZ_Points == np.array([400.]))

    a.plot_transmission()
    a.plot_transmission_image()

    a = AtmosphereGrid(atmgrid_filename='tests/data/reduc_20170530_134_atmsim.fits')
    lambdas = np.arange(200, 1200)
    transmission = a.simulate(ozone=400, pwv=5, aerosols=0.05)
    assert np.max(transmission(lambdas)) < 1 and np.min(transmission(lambdas)) >= 0


@unittest.skipIf(uvspec_available() is False, 'Skipping to avoid libradtran dependency')
def test_simulator():
    file_names = ['tests/data/reduc_20170530_134_spectrum.fits']

    parameters.VERBOSE = True
    parameters.DEBUG = False
    load_config('config/ctio.ini')

    for file_name in file_names:
        spectrum = Spectrum(file_name)
        atmosphere = AtmosphereGrid(atmgrid_filename="./tests/data/reduc_20170530_134_atmsim.fits")
        spectrum_simulation = SpectrumSimulation(spectrum, atmosphere=atmosphere, fast_sim=True)
        spectrum_simulation.simulate(A1=1, A2=1, ozone=300, pwv=5, aerosols=0.05, angstrom_exponent=None,
                                     reso=0., D=56, shift_x=0., B=0.)

        assert np.sum(spectrum_simulation.data) > 0
        assert np.sum(spectrum_simulation.data) < 1e-10
        assert np.sum(spectrum_simulation.data_next_order) < 1e-10


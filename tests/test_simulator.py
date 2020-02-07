from numpy.testing import run_module_suite
import numpy as np

from spectractor import parameters
from spectractor.simulation.simulator import (SpectrumSimulatorSimGrid,
                                              Atmosphere, AtmosphereGrid, SpectrogramSimulator)
from spectractor.simulation.image_simulation import ImageSim
from spectractor.config import load_config
import os


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
    assert np.all(a.aerosols_grid == np.array([0., 0.1]))
    assert np.all(a.pwv_grid == np.array([5.]))
    assert np.all(a.ozone_grid == np.array([400.]))

    a.plot_transmission()
    a.plot_transmission_image()

    a = AtmosphereGrid(filename='tests/data/reduc_20170530_134_atmsim.fits')
    lambdas = np.arange(200, 1200)
    transmission = a.simulate(ozone=400, pwv=5, aerosols=0.05)
    assert np.max(transmission(lambdas)) < 1 and np.min(transmission(lambdas)) >= 0


def test_simulator():
    file_names = ['tests/data/reduc_20170530_134_spectrum.fits']

    parameters.VERBOSE = True
    parameters.DEBUG = True
    load_config('config/ctio.ini')

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        # spectrum_simulation = SpectrumSimulator(file_name, pwv=3, ozone=350, aerosols=0.02,
        #                                        A1=1.1, A2=0.1, reso=2, D=56, shift=-3)
        spectrogram_simulation = SpectrogramSimulator(file_name, pwv=3, ozone=350, aerosols=0.02,
                                                      A1=1.1, A2=0.1, D=56, shift_x=-3, shift_y=1, angle=-1)
        psf_poly_params = spectrogram_simulation.chromatic_psf.from_table_to_poly_params()
        image_simulation = ImageSim(file_name.replace('_spectrum.fits', '.fits'), file_name, './tests/data/', A2=0.01,
                                    psf_poly_params=psf_poly_params, with_stars=True)
        SpectrumSimulatorSimGrid(file_name, './tests/data/', pwv_grid=[0, 10, 2], ozone_grid=[200, 400, 2],
                                 aerosol_grid=[0, 0.1, 2])
        atmgrid = AtmosphereGrid(file_name, file_name.replace('spectrum', 'atmsim'))
        atm = Atmosphere(atmgrid.airmass, atmgrid.pressure, atmgrid.temperature)
        assert os.path.isfile('./tests/data/' + tag.replace('_spectrum.fits', '_atmsim.fits')) is True
        assert image_simulation.data is not None
        # assert spectrum_simulation.data is not None
        assert spectrogram_simulation.data is not None
        assert os.path.isfile('./tests/data/' + tag.replace('_spectrum.fits', '_sim.fits')) is True
        assert atm.transmission is not None


if __name__ == "__main__":
    run_module_suite()

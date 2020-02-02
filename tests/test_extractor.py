from numpy.testing import run_module_suite

from spectractor import parameters
from spectractor.extractor.extractor import Spectractor
from spectractor.logbook import LogBook
import os
import numpy as np


def test_logbook():
    logbook = LogBook('./ctiofulllogbook_jun2017_v5.csv')
    # target, xpos, ypos = logbook.search_for_image('reduc_20170529_085.fits')
    # assert xpos is None
    disperser_label, target, xpos, ypos = logbook.search_for_image('reduc_20170603_020.fits')
    assert target == "PKS1510-089"
    assert xpos == 830
    assert ypos == 590
    # logbook = LogBook('./ctiofulllogbook_jun2017_v5.csv')
    # logbook.plot_columns_vs_date(['T', 'seeing', 'W'])


def test_extractor():
    file_names = ['tests/data/reduc_20170530_134.fits']

    logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
    parameters.VERBOSE = True
    parameters.DEBUG = True

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue
        spectrum = Spectractor(file_name, './outputs/', target, [xpos, ypos], disperser_label,
                               config='./config/ctio.ini', line_detection=True, atmospheric_lines=True)
        assert spectrum.data is not None
        assert np.sum(spectrum.data) > 1e-10
        spectrum.my_logger.warning(f"\n\tQuantities to test:"
                                   f"\n\t\tspectrum.lambdas[0]={spectrum.lambdas[0]}"
                                   f"\n\t\tspectrum.lambdas[-1]={spectrum.lambdas[-1]}"
                                   f"\n\t\tspectrum.x0={spectrum.x0}"
                                   f"\n\t\tspectrum.spectrogram_x0={spectrum.spectrogram_x0}"
                                   f"\n\t\tnp.mean(spectrum.chromatic_psf.table['gamma']="
                                   f"{np.mean(spectrum.chromatic_psf.table['gamma'])}")
        assert np.isclose(spectrum.lambdas[0], 296, atol=1)
        assert np.isclose(spectrum.lambdas[-1], 1083.5, atol=1)
        assert np.isclose(spectrum.x0[0], 743.6651370068676, atol=0.5)
        assert np.isclose(spectrum.x0[1], 683.0577836601408, atol=1)
        assert np.isclose(spectrum.spectrogram_x0, -240, atol=1)
        assert 2 < np.mean(spectrum.chromatic_psf.table['gamma']) < 3
        assert os.path.isfile('./outputs/' + tag.replace('.fits', '_spectrum.fits')) is True
        assert os.path.isfile('./outputs/' + tag.replace('.fits', '_spectrogram.fits')) is True


def test_extractor_auxtel():
    file_names = ['tests/data/auxtel_first_light-1.fits']

    # logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
    parameters.VERBOSE = True
    parameters.DEBUG = True
    xpos = 1600
    ypos = 2293
    target_label = "HD107696"

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        # disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        spectrum = Spectractor(file_name, './outputs/', target_label=target_label, guess=[xpos, ypos],
                               config='./config/auxtel.ini', line_detection=True, atmospheric_lines=True)
        assert spectrum.data is not None
        assert np.sum(spectrum.data) > 1e-10
        # spectrum.my_logger.warning(f"\n\tQuantities to test:"
        #                            f"\n\t\tspectrum.lambdas[0]={spectrum.lambdas[0]}"
        #                            f"\n\t\tspectrum.lambdas[-1]={spectrum.lambdas[-1]}"
        #                            f"\n\t\tspectrum.x0={spectrum.x0}"
        #                            f"\n\t\tspectrum.spectrogram_x0={spectrum.spectrogram_x0}"
        #                            f"\n\t\tnp.mean(spectrum.chromatic_psf.table['gamma']="
        #                            f"{np.mean(spectrum.chromatic_psf.table['gamma'])}")
        # assert np.isclose(spectrum.lambdas[0], 296, atol=1)
        # assert np.isclose(spectrum.lambdas[-1], 1083.5, atol=1)
        # assert np.isclose(spectrum.x0[0], 743.6651370068676, atol=0.5)
        # assert np.isclose(spectrum.x0[1], 683.0577836601408, atol=1)
        # assert np.isclose(spectrum.spectrogram_x0, -240, atol=1)
        # assert 2 < np.mean(spectrum.chromatic_psf.table['gamma']) < 3
        # assert os.path.isfile('./outputs/' + tag.replace('.fits', '_spectrum.fits')) is True
        # assert os.path.isfile('./outputs/' + tag.replace('.fits', '_spectrogram.fits')) is True


if __name__ == "__main__":
    run_module_suite()

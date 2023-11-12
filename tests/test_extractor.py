import matplotlib as mpl
mpl.use('Agg')  # must be run first! But therefore requires noqa E402 on all other imports

from spectractor import parameters  # noqa: E402
from spectractor.extractor.extractor import Spectractor  # noqa: E402
from spectractor.logbook import LogBook  # noqa: E402
from spectractor.config import load_config, apply_rebinning_to_parameters  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import numpy as np  # noqa: E402
import unittest  # noqa: E402
import astropy.config  # noqa: E402


def test_logbook():
    logbook = LogBook('./tests/data/ctiofulllogbook_jun2017_v5.csv')
    # target, xpos, ypos = logbook.search_for_image('reduc_20170529_085.fits')
    # assert xpos is None
    disperser_label, target, xpos, ypos = logbook.search_for_image('reduc_20170603_020.fits')
    assert target == "PKS1510-089"
    assert xpos == 830
    assert ypos == 590
    # logbook = LogBook('./ctiofulllogbook_jun2017_v5.csv')
    # logbook.plot_columns_vs_date(['T', 'seeing', 'W'])


@astropy.config.set_temp_cache(os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "cache"))
def test_extractor_ctio():
    file_names = ['tests/data/reduc_20170530_134.fits']
    output_directory = "./outputs"

    logbook = LogBook(logbook='./tests/data/ctiofulllogbook_jun2017_v5.csv')
    parameters.VERBOSE = True
    load_config("ctio.ini")
    parameters.VERBOSE = True
    parameters.DEBUG = False
    parameters.CCD_REBIN = 2  # rebin=1 to build tests/data spectrum
    apply_rebinning_to_parameters()

    for file_name in file_names:
        tag = file_name.split('/')[-1].replace("sim", "reduc")
        disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        if target_label is None or xpos is None or ypos is None:
            continue
        spectrum = Spectractor(file_name, output_directory, target_label, [xpos, ypos], disperser_label, config="")
        assert spectrum.data is not None
        spectrum.my_logger.warning(f"\n\tQuantities to test:"
                                   f"\n\t\tspectrum.lambdas[0]={spectrum.lambdas[0]}"
                                   f"\n\t\tspectrum.lambdas[-1]={spectrum.lambdas[-1]}"
                                   f"\n\t\tspectrum.x0={spectrum.x0}"
                                   f"\n\t\tspectrum.spectrogram_x0={spectrum.spectrogram_x0}"
                                   f"\n\t\tspectrum total flux={np.sum(spectrum.data) * parameters.CCD_REBIN ** 2}"
                                   f"\n\t\tnp.mean(spectrum.chromatic_psf.table['gamma']="
                                   f"{np.mean(spectrum.chromatic_psf.table['gamma'])}")
        assert np.sum(spectrum.data) * parameters.CCD_REBIN**2 > 2e-11 / parameters.CCD_REBIN
        if parameters.CCD_REBIN == 1:
            if parameters.SPECTRACTOR_DECONVOLUTION_PSF2D or parameters.SPECTRACTOR_DECONVOLUTION_FFM:
                assert np.isclose(spectrum.lambdas[0], 345, atol=1)
                assert np.isclose(spectrum.lambdas[-1], 1083.0, atol=1)
            else:
                assert np.isclose(spectrum.lambdas[0], 347, atol=1)
                assert np.isclose(spectrum.lambdas[-1], 1085.0, atol=1)
            assert np.isclose(spectrum.spectrogram_x0, -280, atol=1)
        assert np.isclose(spectrum.x0[0] * parameters.CCD_REBIN, 743.6651370068676, atol=0.5 * parameters.CCD_REBIN)
        assert np.isclose(spectrum.x0[1] * parameters.CCD_REBIN, 683.0577836601408, atol=1 * parameters.CCD_REBIN)
        assert 2 < np.mean(spectrum.chromatic_psf.table['gamma']) * parameters.CCD_REBIN < 3.5
        assert os.path.isfile(os.path.join(output_directory, tag.replace('.fits', '_spectrum.fits'))) is True


# TODO: DM-33441 Fix broken spectractor tests
@unittest.skip('Skipping due broken test (TypeError: argument of type bool is not iterable)')
def test_extractor_ctio_planetary_nebula():
    file_names = ['tests/data/reduc_20170605_028.fits']
    output_directory = "./outputs"

    logbook = LogBook(logbook='./tests/data/ctiofulllogbook_jun2017_v5.csv')
    load_config("./config/ctio.ini")
    parameters.VERBOSE = True
    parameters.DEBUG = False
    parameters.CCD_REBIN = 1  # do not work with other values
    parameters.LAMBDA_MIN = 450
    parameters.LAMBDA_MAX = 1000

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        if target_label is None or xpos is None or ypos is None:
            continue
        spectrum = Spectractor(file_name, output_directory, target_label, [xpos, ypos], disperser_label)
        assert spectrum.data is not None
        spectrum.my_logger.warning(f"\n\tQuantities to test:"
                                   f"\n\t\tspectrum.lambdas[0]={spectrum.lambdas[0]}"
                                   f"\n\t\tspectrum.lambdas[-1]={spectrum.lambdas[-1]}"
                                   f"\n\t\tspectrum.x0={spectrum.x0}"
                                   f"\n\t\tspectrum.spectrogram_x0={spectrum.spectrogram_x0}"
                                   f"\n\t\tspectrum total flux={np.sum(spectrum.data) * parameters.CCD_REBIN ** 2}"
                                   f"\n\t\tnp.mean(spectrum.chromatic_psf.table['gamma']="
                                   f"{np.mean(spectrum.chromatic_psf.table['gamma'])}")
        if parameters.SPECTRACTOR_DECONVOLUTION_PSF2D or parameters.SPECTRACTOR_DECONVOLUTION_FFM:
            assert np.isclose(spectrum.lambdas[0], 449, atol=1)
            assert np.isclose(spectrum.lambdas[-1], 996.5, atol=1)
        else:
            assert np.isclose(spectrum.lambdas[0], 443, atol=1)
            assert np.isclose(spectrum.lambdas[-1], 981, atol=1)
        assert np.isclose(spectrum.spectrogram_x0, -368, atol=1)
        assert np.sum(spectrum.data) * parameters.CCD_REBIN ** 2 > 1e-11 / parameters.CCD_REBIN
        assert np.isclose(spectrum.x0[0] * parameters.CCD_REBIN, 816.75, atol=0.5 * parameters.CCD_REBIN)
        assert np.isclose(spectrum.x0[1] * parameters.CCD_REBIN, 587.67, atol=1 * parameters.CCD_REBIN)
        assert 1 < np.mean(spectrum.chromatic_psf.table['gamma']) * parameters.CCD_REBIN < 2.5
        assert os.path.isfile(os.path.join(output_directory, tag.replace('.fits', '_spectrum.fits'))) is True


def extractor_auxtel():
    file_names = ['tests/data/calexp_2020031500162-EMPTY_ronchi90lpmm-det000.fits']  # image 1
    # file_names = ['tests/data/calexp_2020031200313-EMPTY_ronchi90lpmm-det000.fits']  # image 2
    # file_names = ['tests/data/calexp_2020022100767-EMPTY_ronchi90lpmm-det000.fits']  # image 3
    # file_names = ['tests/data//calexp_2020021800154-EMPTY_ronchi90lpmm-det000.fits']  # image 4
    file_names = ['tests/data/Cor_holo4_003_empty_HD60753_2021-02-18_585.fits']  # ronchi170lpmm
    # tests/data/auxtel_first_light-1.fits']

    # logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
    parameters.VERBOSE = True
    parameters.DEBUG = True
    #xpos = 1600
    #ypos = 2293
    #target_label = "HD107696"

    for config in ['./config/auxtel.ini']:  #'./config/auxtel_quicklook.ini',
        for file_name in file_names:
            # tag = file_name.split('/')[-1]
            # disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
            #spectrum = Spectractor(file_name, './outputs/', target_label=target_label, guess=[xpos, ypos],
            #                       config=config)
            spectrum = Spectractor(file_name, './outputs/', target_label="", guess=None, config=config)
            assert spectrum.data is not None
            assert np.sum(spectrum.data) > 1e-14
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


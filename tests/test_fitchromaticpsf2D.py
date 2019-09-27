from numpy.testing import run_module_suite

from spectractor import parameters
from spectractor.extractor.images import Image
from spectractor.extractor.extractor import Spectractor
from spectractor.logbook import LogBook
from spectractor.simulation.image_simulation import ImageSim
import os
import numpy as np
import matplotlib.pyplot as plt

parameters.PSF_POLY_ORDER = 2
PSF_POLY_PARAMS_TRUTH = [0, 0, 0,
                         3, 0, 0,
                         2, 0, 0,
                         0, -0.5, 0,
                         1, 0, 0,
                         500]


def make_test_image():
    spectrum_filename = "tests/data/reduc_20170530_134_spectrum.fits"
    image_filename = spectrum_filename.replace("_spectrum.fits", ".fits")
    ImageSim(image_filename, spectrum_filename, "./tests/data/", A1=1, A2=0.05,
             psf_poly_params=PSF_POLY_PARAMS_TRUTH, with_stars=False, with_rotation=False)


def test_fitchromaticpsf2d():
    parameters.VERBOSE = True
    # parameters.DEBUG = True
    sim_image = "./tests/data/sim_20170530_134.fits"
    if not os.path.isfile(sim_image):
        make_test_image()
    image = Image(sim_image)
    lambdas_truth = np.fromstring(image.header['LAMBDAS'][1:-1], sep=' ')
    amplitude_truth = np.fromstring(image.header['PSF_POLY'][1:-1], sep=' ', dtype=float)[:lambdas_truth.size]
    parameters.PSF_POLY_ORDER = int(image.header['PSF_DEG'])

    tag = sim_image.split('/')[-1]
    tag = tag.replace('sim_', 'reduc_')
    logbook = LogBook(logbook="./ctiofulllogbook_jun2017_v5.csv")
    disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
    spectrum = Spectractor(sim_image, "./tests/data", [xpos, ypos], target, disperser_label, "./config/ctio.ini")
    plt.plot(amplitude_truth)
    plt.plot(spectrum.data)
    plt.show()

    assert np.isclose(float(image.header['X0_T']), spectrum.target_pixcoords[0], atol=0.01)
    assert np.isclose(float(image.header['Y0_T']), spectrum.target_pixcoords[1], atol=0.01)
    assert np.isclose(float(image.header['ROTANGLE']), spectrum.rotation_angle, atol=180/np.pi*1/parameters.CCD_IMSIZE)
    assert np.isclose(float(image.header['BKGD_LEV']), np.mean(spectrum.spectrogram_bgd), atol=2e-3)
    assert np.isclose(float(image.header['D2CCD_T']), spectrum.disperser.D, atol=0.05)
    print(spectrum.chromatic_psf.poly_params[spectrum.lambdas.size+3:]-np.array(PSF_POLY_PARAMS_TRUTH)[3:])
    print(np.std((amplitude_truth-spectrum.data)/spectrum.err))


if __name__ == "__main__":
    run_module_suite()

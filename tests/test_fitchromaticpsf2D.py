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
    parameters.DEBUG = True
    sim_image = "tests/data/sim_20170530_134.fits"
    if not os.path.isfile(sim_image):
        make_test_image()
    image = Image(sim_image)
    lambdas_truth = np.fromstring(image.header['LAMBDAS'][1:-1], sep=' ')
    amplitude_truth = np.fromstring(image.header['PSF_POLY'][1:-1], sep=' ', dtype=float)[:lambdas_truth.size]
    parameters.PSF_POLY_ORDER = int(image.header['PSF_DEG'])
    image.plot_image(scale="log")

    tag = sim_image.split('/')[-1]
    tag = tag.replace('sim_', 'reduc_')
    logbook = LogBook(logbook="./ctiofulllogbook_jun2017_v5.csv")
    disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
    spectrum = Spectractor(sim_image, "./tests/data", [xpos, ypos], target, disperser_label, "./config/ctio.ini")
    plt.plot(amplitude_truth)
    plt.plot(spectrum.data * parameters.FLAM_TO_ADURATE)
    plt.show()


if __name__ == "__main__":
    run_module_suite()

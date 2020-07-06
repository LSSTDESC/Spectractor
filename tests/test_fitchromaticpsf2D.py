from numpy.testing import run_module_suite
from scipy.interpolate import interp1d

from spectractor import parameters
from spectractor.extractor.images import Image
from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.extractor import Spectractor
from spectractor.logbook import LogBook
from spectractor.config import load_config
from spectractor.simulation.image_simulation import ImageSim
from spectractor.tools import plot_spectrum_simple
import os
import numpy as np
import matplotlib.pyplot as plt

parameters.PSF_POLY_ORDER = 2
PSF_POLY_PARAMS_TRUTH = [0, 0, 0,
                         3, 2, 0,
                         2, 0, 0,
                         0, -0.5, 0,
                         1, 0, 0,
                         500]


def make_test_image(config="./config/ctio.ini"):
    load_config(config)
    spectrum_filename = "tests/data/reduc_20170530_134_spectrum.fits"
    image_filename = spectrum_filename.replace("_spectrum.fits", ".fits")
    ImageSim(image_filename, spectrum_filename, "./tests/data/", A1=1, A2=1,
             psf_poly_params=PSF_POLY_PARAMS_TRUTH, with_stars=True, with_rotation=True)


def plot_residuals(spectrum, lambdas_truth, amplitude_truth):
    """

    Parameters
    ----------
    spectrum
    lambdas_truth
    amplitude_truth

    Examples
    --------

    >>> from spectractor.extractor.spectrum import Spectrum
    >>> image = Image("./tests/data/sim_20170530_134.fits")
    >>> spectrum = Spectrum("./tests/data/sim_20170530_134_spectrum.fits")
    >>> lambdas_truth = np.fromstring(image.header['LBDAS_T'][1:-1], sep=' ')
    >>> amplitude_truth = np.fromstring(image.header['AMPLIS_T'][1:-1], sep=' ', dtype=float)[:lambdas_truth.size]
    >>> plot_residuals(spectrum, lambdas_truth, amplitude_truth)
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex="all", gridspec_kw={'height_ratios': [3, 1]})
    plot_spectrum_simple(ax[0], spectrum.lambdas, spectrum.data, data_err=spectrum.err, label="Fit",
                         units=spectrum.units)
    # spectrum.lines.plot_atomic_lines(ax[0], fontsize=12, force=True)
    ax[0].plot(lambdas_truth, amplitude_truth, label="Truth")
    # transverse_sum = np.mean((spectrum.spectrogram-spectrum.spectrogram_bgd)*spectrum.expo
    # /(parameters.FLAM_TO_ADURATE*spectrum.lambdas*spectrum.lambdas_binwidths), axis=0)
    # transverse_sum *= np.max(spectrum.data)/np.max(transverse_sum)
    # ax[0].plot(spectrum.lambdas, transverse_sum, label="Transverse sum")
    ax[0].set_ylabel(f"Spectrum [{spectrum.units}]")
    ax[0].legend()
    amplitude_truth_interp = interp1d(lambdas_truth, amplitude_truth, kind='cubic', fill_value=0, bounds_error=False)(spectrum.lambdas)
    residuals = (spectrum.data - amplitude_truth_interp)/spectrum.err
    ax[1].errorbar(spectrum.lambdas, residuals, yerr=np.ones_like(spectrum.data), label="Fit", fmt="r.")
    ax[1].set_ylabel(f"Residuals")
    ax[1].set_xlabel(r"$\lambda$ [nm]")
    ax[1].grid()
    ax[1].legend()
    ax[1].text(0.05, 0.05, f'mean={np.mean(residuals):.3g}\nstd={np.std(residuals):.3g}',
               horizontalalignment='left', verticalalignment='bottom',
               color='black', transform=ax[1].transAxes)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def load_test(sim_image, config="./config/ctio.ini"):
    parameters.VERBOSE = True
    parameters.DEBUG = True
    if not os.path.isfile(sim_image):
        make_test_image(config=config)
    image = Image(sim_image, config=config)
    lambdas_truth = np.fromstring(image.header['LBDAS_T'][1:-1], sep=' ')
    amplitude_truth = np.fromstring(image.header['AMPLIS_T'][1:-1], sep=' ', dtype=float)
    parameters.AMPLITUDE_TRUTH = np.copy(amplitude_truth)
    parameters.LAMBDA_TRUTH = np.copy(lambdas_truth)
    parameters.PSF_POLY_ORDER = int(image.header['PSF_DEG'])
    parameters.PSF_POLY_ORDER = 2
    return image, lambdas_truth, amplitude_truth


def test_fitchromaticpsf2d_run(sim_image="./tests/data/sim_20170530_134.fits", config="./config/ctio.ini"):
    load_test(sim_image, config=config)
    tag = sim_image.split('/')[-1]
    tag = tag.replace('sim_', 'reduc_')
    logbook = LogBook(logbook="./ctiofulllogbook_jun2017_v5.csv")
    disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
    spectrum = Spectractor(sim_image, "./tests/data", guess=[xpos, ypos], target_label=target,
                           disperser_label=disperser_label, config="./config/ctio.ini")
    return spectrum


def test_fitchromaticpsf2d_test(sim_image="./tests/data/sim_20170530_134.fits", config="./config/ctio.ini"):
    image, lambdas_truth, amplitude_truth = load_test(sim_image, config=config)
    spectrum = Spectrum(sim_image.replace(".fits", "_spectrum.fits"), config=config)

    plot_residuals(spectrum, lambdas_truth, amplitude_truth)

    assert np.isclose(float(image.header['X0_T']), spectrum.target_pixcoords[0], atol=0.01)
    assert np.isclose(float(image.header['Y0_T']), spectrum.target_pixcoords[1], atol=0.01)
    assert np.isclose(float(image.header['ROTANGLE']), spectrum.rotation_angle,
                      atol=180 / np.pi * 1 / parameters.CCD_IMSIZE)
    assert np.isclose(float(image.header['BKGD_LEV']), np.mean(spectrum.spectrogram_bgd), atol=2e-3)
    assert np.isclose(float(image.header['D2CCD_T']), spectrum.disperser.D, atol=0.05)
    print(spectrum.chromatic_psf.poly_params[spectrum.lambdas.size + 6:] - np.array(PSF_POLY_PARAMS_TRUTH)[3:])
    print(np.std((amplitude_truth - spectrum.data) / spectrum.err))


if __name__ == "__main__":

    test_fitchromaticpsf2d_test("outputs/sim_20170530_134.fits")
    #run_module_suite()

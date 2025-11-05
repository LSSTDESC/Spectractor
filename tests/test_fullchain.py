import matplotlib as mpl
mpl.use('Agg')  # must be run first! But therefore requires noqa E402 on all other imports

from scipy.interpolate import interp1d  # noqa: E402

from spectractor import parameters  # noqa: E402
from spectractor.extractor.images import Image  # noqa: E402
from spectractor.extractor.spectrum import Spectrum  # noqa: E402
from spectractor.extractor.extractor import SpectractorRun, SpectractorInit  # noqa: E402
from spectractor.logbook import LogBook  # noqa: E402
from spectractor.config import load_config, apply_rebinning_to_parameters  # noqa: E402
from spectractor.simulation.image_simulation import ImageSim  # noqa: E402
from spectractor.tools import (plot_spectrum_simple, uvspec_available)  # noqa: E402
from spectractor.fit.fit_spectrum import SpectrumFitWorkspace, run_spectrum_minimisation  # noqa: E402
from spectractor.fit.fit_spectrogram import (SpectrogramFitWorkspace,  # noqa: E402
                                             run_spectrogram_minimisation)  # noqa: E402
import os  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import unittest  # noqa: E402
import astropy.config  # noqa: E402


# original parameters
N_DIFF_ORDERS = 3
PSF_POLY_ORDER = 2
PSF_POLY_PARAMS_TRUTH = [1, 0, 0,
                         0, 0, 0,
                         3, 0, 0,
                         3, 0, 0,
                         1e6] * N_DIFF_ORDERS

PSF_POLY_PARAMS_AUXTEL_TRUTH = [1, 0, 0,
                         0, 0, 0,
                         8, 0, 0,
                         3, 0, 0,
                         1e6] * N_DIFF_ORDERS

A1_T = 1
A2_T = 1
A3_T = 0

#
# PSF_POLY_PARAMS_TRUTH = [1, 0, 0,
#                          0, 0, 0,
#                          10, 2, 5,
#                          2, 0, 0,
#                          1e6]
# A1_T = 1
# A2_T = 0.


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
    >>> lambdas_truth = np.fromstring(image.header['LBDAS_T'][1:-1], sep=',')
    >>> amplitude_truth = np.fromstring(image.header['AMPLIS_T'][1:-1], sep=',', dtype=float)[:lambdas_truth.size]
    >>> plot_residuals(spectrum, lambdas_truth, amplitude_truth)  #doctest: +ELLIPSIS
    array([...
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
    amplitude_truth_interp = interp1d(lambdas_truth, amplitude_truth, kind='cubic',
                                      fill_value=0, bounds_error=False)(spectrum.lambdas)
    residuals = (spectrum.data - amplitude_truth_interp) / spectrum.err
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
    return residuals


def make_image():
    spectrum_filename = "./tests/data/reduc_20170530_134_spectrum.fits"
    image_filename = "./tests/data/reduc_20170530_134.fits"
    sim = ImageSim(image_filename, spectrum_filename, "./tests/data/", A1=A1_T, A2=A2_T, A3=A3_T,
                   psf_poly_params=PSF_POLY_PARAMS_TRUTH, with_starfield=True, with_rotation=True, with_noise=False,
                   with_flat=True)
    return sim


def make_auxtel_image():
    spectrum_filename = "./tests/data/test_auxtel_spectrum.fits"
    image_filename = "./tests/data/exposure_2023092800266_dmpostisrccd.fits"
    sim = ImageSim(image_filename, spectrum_filename, "./tests/data/", A1=A1_T, A2=A2_T, A3=A3_T,
                   psf_poly_params=PSF_POLY_PARAMS_AUXTEL_TRUTH, with_starfield=False, with_rotation=True,
                   with_noise=False, with_flat=False)
    return sim


@unittest.skipIf(uvspec_available() is False, 'Skipping to avoid libradtran dependency')
@astropy.config.set_temp_cache(os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "cache"))
def test_ctio_fullchain():
    parameters.VERBOSE = True
    parameters.DEBUG = False
    parameters.SPECTRACTOR_ATMOSPHERE_SIM = "libradtran"
    sim_image_filename = "./tests/data/sim_20170530_134.fits"

    # load test and make image simulation
    # if not os.path.isfile(sim_image_filename):
    sim = make_image()
    image = Image(sim_image_filename, config="ctio.ini")
    lambdas_truth = np.fromstring(image.header['LBDAS_T'][1:-1], sep=',')
    amplitude_truth = np.fromstring(image.header['AMPLIS_T'][1:-1], sep=',', dtype=float)
    parameters.AMPLITUDE_TRUTH = np.copy(amplitude_truth)
    parameters.LAMBDA_TRUTH = np.copy(lambdas_truth)

    # extractor
    tag = os.path.basename(sim_image_filename)
    tag = tag.replace('sim_', 'reduc_')
    logbook = LogBook(logbook="./tests/data/ctiofulllogbook_jun2017_v5.csv")
    disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
    load_config("ctio.ini")
    parameters.SPECTRACTOR_ATMOSPHERE_SIM = "libradtran"
    parameters.PSF_POLY_ORDER = PSF_POLY_ORDER
    parameters.CCD_REBIN = 1
    #  JN: > 1 not working well for now: I guess CTIO spectra are too narrow
    #  and under-sampled to extract unbiased rebinned spectrum, but pipeline is ok.
    apply_rebinning_to_parameters()
    if parameters.CCD_REBIN > 1:
        for k in range(2 * (PSF_POLY_ORDER + 1), 3 * (PSF_POLY_ORDER +1)):
            PSF_POLY_PARAMS_TRUTH[k] /= parameters.CCD_REBIN

    image = SpectractorInit(sim_image_filename, target_label=target_label,
                            disperser_label=disperser_label, config="")  # config already loaded, do not overwrite PSF_POLY_ORDER
    image.flat = sim.flat
    image.starfield = sim.starfield
    image.mask = np.zeros_like(image.data).astype(bool)
    image.mask[680:685, 1255:1260] = True  # test masking of some pixels like cosmic rays
    parameters.SPECTRACTOR_SIMULATE_STARFIELD = False  # here we want to test fit with an exact starfield simulation
    spectrum = SpectractorRun(image, guess=[xpos, ypos], output_directory="./tests/data")

    # tests
    residuals = plot_residuals(spectrum, lambdas_truth, amplitude_truth)

    spectrum.my_logger.warning(f"\n\tQuantities to test with {parameters.CCD_REBIN=}:"
                               f"\n\t\tspectrum.header['X0_T']={spectrum.header['X0_T'] / parameters.CCD_REBIN:.5g} vs {spectrum.x0[0]:.5g}"
                               f"\n\t\tspectrum.header['Y0_T']={spectrum.header['Y0_T'] / parameters.CCD_REBIN:.5g} vs {spectrum.x0[1]:.5g}"
                               f"\n\t\tspectrum.header['ROT_T']={spectrum.header['ROT_T']:.5g} "
                               f"vs {spectrum.rotation_angle:.5g}"
                               f"\n\t\tspectrum.header['BKGD_LEV']={spectrum.header['BKGD_LEV'] * parameters.CCD_REBIN**2:.5g} "
                               f"vs {np.mean(spectrum.spectrogram_bgd):.5g}"
                               f"\n\t\tspectrum.header['D2CCD_T']={spectrum.header['D2CCD_T']:.5g} "
                               f"vs {spectrum.header['D2CCD']:.5g}"
                               f"\n\t\tspectrum.header['A2_FIT']={spectrum.header['A2_FIT']:.5g} vs {A2_T:.5g}"
                               f"\n\t\tspectrum.header['CHI2_FIT']={spectrum.header['CHI2_FIT']:.4g}"
                               f"\n\t\tspectrum.chromatic_psf.poly_params="
                               f"{spectrum.chromatic_psf.params.values[spectrum.chromatic_psf.Nx + 2 * (PSF_POLY_ORDER + 1):-1]}"
                               f" vs {PSF_POLY_PARAMS_TRUTH[2 * (PSF_POLY_ORDER + 1):len(PSF_POLY_PARAMS_TRUTH)//N_DIFF_ORDERS - 1]}"
                               f"\n\t\tresiduals wrt truth: mean={np.mean(residuals[100:-100]):.5g}, "
                               f"std={np.std(residuals[100:-100]):.5g}")
    assert np.isclose(float(spectrum.header['X0_T'] / parameters.CCD_REBIN), spectrum.x0[0], atol=0.2 * parameters.CCD_REBIN)
    assert np.isclose(float(spectrum.header['Y0_T'] / parameters.CCD_REBIN), spectrum.x0[1], atol=0.2 * parameters.CCD_REBIN)
    assert np.isclose(float(spectrum.header['ROT_T']), spectrum.rotation_angle, atol=1e-3)
    assert np.isclose(float(spectrum.header['BKGD_LEV'] * parameters.CCD_REBIN**2), np.mean(spectrum.spectrogram_bgd), rtol=1e-3)
    assert np.isclose(float(spectrum.header['D2CCD_T']), spectrum.header["D2CCD"], atol=0.1)
    if parameters.CCD_REBIN == 1:
        assert float(spectrum.header['CHI2_FIT']) < 1.5e-3
    else:
        assert float(spectrum.header['CHI2_FIT']) < 1.5e-1
    assert np.all(
        np.isclose(spectrum.chromatic_psf.params.values[spectrum.chromatic_psf.Nx + 2 * (PSF_POLY_ORDER + 1):-1],
                   np.array(PSF_POLY_PARAMS_TRUTH)[2 * (PSF_POLY_ORDER + 1):len(PSF_POLY_PARAMS_TRUTH)//N_DIFF_ORDERS - 1], rtol=0.01, atol=0.01))
    assert np.abs(np.mean(residuals[100:-100])) < 0.25
    assert np.std(residuals[100:-100]) < 3

    spectrum_file_name = "./tests/data/sim_20170530_134_spectrum.fits"
    assert os.path.isfile(spectrum_file_name)
    atmgrid_filename = sim_image_filename.replace('sim', 'reduc').replace('.fits', '_atmsim.fits')
    assert os.path.isfile(atmgrid_filename)
    spectrum = Spectrum(spectrum_file_name)
    w = SpectrumFitWorkspace(spectrum, atmgrid_file_name=atmgrid_filename, fit_angstrom_exponent=False,
                             verbose=True, plot=True, live_fit=False)
    run_spectrum_minimisation(w, method="newton")
    nsigma = 3
    labels = ["VAOD_T", "OZONE_T", "PWV_T"]
    indices = [2, 4, 5]
    ipar = w.params.get_free_parameters()  # non fixed param indices
    cov_indices = [list(ipar).index(k) for k in indices]  # non fixed param indices in cov matrix
    assert w.costs[-1] / w.data.size < 0.5
    k = 0
    for i, l in zip(indices, labels):
        icov = cov_indices[k]
        spectrum.my_logger.info(f"Test {l} best-fit {w.params.values[i]:.3f}+/-{np.sqrt(w.params.cov[icov, icov]):.3f} "
                                f"vs {spectrum.header[l]:.3f} at {nsigma}sigma level: "
                                f"{np.abs(w.params.values[i] - spectrum.header[l]) / np.sqrt(w.params.cov[icov, icov]) < nsigma}")
        assert np.abs(w.params.values[i] - spectrum.header[l]) / np.sqrt(w.params.cov[icov, icov]) < nsigma
        k += 1
    A2id = w.params.get_index("A2")
    assert np.abs(w.params.values[A2id] / w.params.err[A2id]) < 3 * nsigma  # A2
    assert np.isclose(np.abs(w.params.values[w.params.get_index("alpha_pix [pix]")]), 0, atol=parameters.PIXSHIFT_PRIOR)  # pixshift
    assert np.isclose(np.abs(w.params.values[w.params.get_index("B")]), 0, atol=1e-3)  # B

    parameters.DEBUG = False
    parameters.SPECTRACTOR_ATMOSPHERE_SIM = "libradtran"
    w = SpectrogramFitWorkspace(spectrum, atmgrid_file_name=atmgrid_filename, fit_angstrom_exponent=False,
                                verbose=True, plot=True, live_fit=False)
    run_spectrogram_minimisation(w, method="newton")
    nsigma = 2
    labels = ["A2_T", "VAOD_T", "OZONE_T", "PWV_T"]  # "A1_T" fixed
    indices = [1, 3, 5, 6]
    A1, A2, A3, aerosols, angstrom_exponent, ozone, pwv, B, Astar, D, shift_x, shift_y, shift_t, pressure, *psf_poly_params = w.params.values
    ipar = w.params.get_free_parameters()  # non fixed param indices
    cov_indices = [list(ipar).index(k) for k in indices]  # non fixed param indices in cov matrix
    assert w.costs[-1] / w.data.size < 1e-3
    k = 0
    for i, l in zip(indices, labels):
        icov = cov_indices[k]
        spectrum.my_logger.info(f"Test {l} best-fit {w.params.values[i]:.3f}+/-{np.sqrt(w.params.cov[icov, icov]):.3f} "
                                f"vs {spectrum.header[l]:.3f} at {nsigma}sigma level: "
                                f"{np.abs(w.params.values[i] - spectrum.header[l]) / np.sqrt(w.params.cov[icov, icov]) < nsigma}")
        assert np.abs(w.params.values[i] - spectrum.header[l]) / np.sqrt(w.params.cov[icov, icov]) < nsigma
        k += 1
    A2id = w.params.get_index("A2")
    assert np.abs((A2 - 1) / w.params.err[A2id]) < 2 * nsigma  # A2
    assert np.isclose(shift_y, 0, atol=parameters.PIXSHIFT_PRIOR)  # shift_y
    assert np.isclose(D, spectrum.header["D2CCD_T"], atol=0.1)  # D2CCD
    assert np.isclose(B, 1, atol=1e-3)  # B
    assert np.isclose(Astar, 1, atol=1e-2)  # Astar
    assert np.all(np.isclose(psf_poly_params[(PSF_POLY_ORDER + 1):len(PSF_POLY_PARAMS_TRUTH)//N_DIFF_ORDERS - 1],
                             np.array(PSF_POLY_PARAMS_TRUTH)[(PSF_POLY_ORDER + 1):len(PSF_POLY_PARAMS_TRUTH)//N_DIFF_ORDERS - 1],
                             rtol=0.01, atol=0.01))

@unittest.skipIf(uvspec_available() is False, 'Skipping to avoid libradtran dependency')
@astropy.config.set_temp_cache(os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "cache"))
def auxtel_fullchain():
    """

    Returns
    -------

    Examples
    --------
    >>> test_auxtel_fullchain()
    """
    parameters.VERBOSE = True
    parameters.DEBUG = True
    parameters.SPECTRACTOR_ATMOSPHERE_SIM = "getObsAtmo"
    sim_image_filename = "./tests/data/sim_2023092800266_dmpostisrccd.fits"

    # load test and make image simulation
    # if not os.path.isfile(sim_image_filename):
    sim = make_auxtel_image()
    image = Image(sim_image_filename, config="./config/auxtel.ini")
    lambdas_truth = np.fromstring(image.header['LBDAS_T'][1:-1], sep=',')
    amplitude_truth = np.fromstring(image.header['AMPLIS_T'][1:-1], sep=',', dtype=float)
    parameters.AMPLITUDE_TRUTH = np.copy(amplitude_truth)
    parameters.LAMBDA_TRUTH = np.copy(lambdas_truth)

    # extractor
    load_config("./config/auxtel.ini")
    parameters.PSF_POLY_ORDER = PSF_POLY_ORDER
    parameters.CCD_REBIN = 1
    #  JN: > 1 not working well for now: I guess CTIO spectra are too narrow
    #  and under-sampled to extract unbiased rebinned spectrum, but pipeline is ok.
    apply_rebinning_to_parameters()
    if parameters.CCD_REBIN > 1:
        for k in range(2 * (PSF_POLY_ORDER + 1), 3 * (PSF_POLY_ORDER +1)):
            PSF_POLY_PARAMS_TRUTH[k] /= parameters.CCD_REBIN

    image = SpectractorInit(sim_image_filename, config="")  # config already loaded, do not overwrite PSF_POLY_ORDER
    xpos, ypos = np.array([image.header["TARGETX"], image.header["TARGETY"]]) / 2
    parameters.SPECTRACTOR_SIMULATE_STARFIELD = False  # here we want to test fit with an exact starfield simulation
    spectrum = SpectractorRun(image, guess=[xpos, ypos], output_directory="./tests/data")

    # tests
    residuals = plot_residuals(spectrum, lambdas_truth, amplitude_truth)

    spectrum.my_logger.warning(f"\n\tQuantities to test with {parameters.CCD_REBIN=}:"
                               f"\n\t\tspectrum.header['X0_T']={spectrum.header['X0_T'] / parameters.CCD_REBIN:.5g} vs {spectrum.x0[0]:.5g}"
                               f"\n\t\tspectrum.header['Y0_T']={spectrum.header['Y0_T'] / parameters.CCD_REBIN:.5g} vs {spectrum.x0[1]:.5g}"
                               f"\n\t\tspectrum.header['ROT_T']={spectrum.header['ROT_T']:.5g} "
                               f"vs {spectrum.rotation_angle:.5g}"
                               f"\n\t\tspectrum.header['BKGD_LEV']={spectrum.header['BKGD_LEV'] * parameters.CCD_REBIN**2:.5g} "
                               f"vs {np.mean(spectrum.spectrogram_bgd):.5g}"
                               f"\n\t\tspectrum.header['D2CCD_T']={spectrum.header['D2CCD_T']:.5g} "
                               f"vs {spectrum.header['D2CCD']:.5g}"
                               f"\n\t\tspectrum.header['A2_FIT']={spectrum.header['A2_FIT']:.5g} vs {A2_T:.5g}"
                               f"\n\t\tspectrum.header['CHI2_FIT']={spectrum.header['CHI2_FIT']:.4g}"
                               f"\n\t\tspectrum.chromatic_psf.poly_params="
                               f"{spectrum.chromatic_psf.params.values[spectrum.chromatic_psf.Nx + 2 * (PSF_POLY_ORDER + 1):-1]}"
                               f" vs {PSF_POLY_PARAMS_TRUTH[2 * (PSF_POLY_ORDER + 1):len(PSF_POLY_PARAMS_TRUTH)//N_DIFF_ORDERS - 1]}"
                               f"\n\t\tresiduals wrt truth: mean={np.mean(residuals[100:-100]):.5g}, "
                               f"std={np.std(residuals[100:-100]):.5g}")
    assert np.isclose(float(spectrum.header['X0_T'] / parameters.CCD_REBIN), spectrum.x0[0], atol=0.2 * parameters.CCD_REBIN)
    assert np.isclose(float(spectrum.header['Y0_T'] / parameters.CCD_REBIN), spectrum.x0[1], atol=0.2 * parameters.CCD_REBIN)
    assert np.isclose(float(spectrum.header['ROT_T']), spectrum.rotation_angle, atol=1e-3)
    assert np.isclose(float(spectrum.header['BKGD_LEV'] * parameters.CCD_REBIN**2), np.mean(spectrum.spectrogram_bgd), rtol=1e-3)
    assert np.isclose(float(spectrum.header['D2CCD_T']), spectrum.header['D2CCD'], atol=0.1)
    if parameters.CCD_REBIN == 1:
        assert float(spectrum.header['CHI2_FIT']) < 1.5e-3
    else:
        assert float(spectrum.header['CHI2_FIT']) < 1.5e-1
    assert np.all(
        np.isclose(spectrum.chromatic_psf.params.values[spectrum.chromatic_psf.Nx + 2 * (PSF_POLY_ORDER + 1):-1],
                   np.array(PSF_POLY_PARAMS_TRUTH)[2 * (PSF_POLY_ORDER + 1):len(PSF_POLY_PARAMS_TRUTH)//N_DIFF_ORDERS - 1], rtol=0.01, atol=0.01))
    assert np.abs(np.mean(residuals[100:-100])) < 0.25
    assert np.std(residuals[100:-100]) < 3

    spectrum_file_name = "./tests/data/sim_2023092800266_dmpostisrccd_spectrum.fits"
    assert os.path.isfile(spectrum_file_name)
    spectrum = Spectrum(spectrum_file_name)
    parameters.SPECTRACTOR_ATMOSPHERE_SIM = "getObsAtmo"
    w = SpectrumFitWorkspace(spectrum, fit_angstrom_exponent=False, verbose=True, plot=True, live_fit=False)
    run_spectrum_minimisation(w, method="newton")
    nsigma = 2
    labels = ["VAOD_T", "OZONE_T", "PWV_T"]
    indices = [2, 4, 5]
    ipar = w.params.get_free_parameters()  # non fixed param indices
    cov_indices = [list(ipar).index(k) for k in indices]  # non fixed param indices in cov matrix
    assert w.costs[-1] / w.data.size < 0.5
    k = 0
    for i, l in zip(indices, labels):
        icov = cov_indices[k]
        spectrum.my_logger.info(f"Test {l} best-fit {w.params.values[i]:.3f}+/-{np.sqrt(w.params.cov[icov, icov]):.3f} "
                                f"vs {spectrum.header[l]:.3f} at {nsigma}sigma level: "
                                f"{np.abs(w.params.values[i] - spectrum.header[l]) / np.sqrt(w.params.cov[icov, icov]) < nsigma}")
        assert np.abs(w.params.values[i] - spectrum.header[l]) / np.sqrt(w.params.cov[icov, icov]) < nsigma
        k += 1
    assert np.abs(w.params.values[1]) / np.sqrt(w.params.cov[1, 1]) < 2 * nsigma  # A2
    assert np.isclose(np.abs(w.params.values[8]), 0, atol=parameters.PIXSHIFT_PRIOR)  # pixshift
    assert np.isclose(np.abs(w.params.values[9]), 0, atol=1e-3)  # B

    parameters.DEBUG = False
    parameters.SPECTRACTOR_ATMOSPHERE_SIM = "getObsAtmo"
    w = SpectrogramFitWorkspace(spectrum, fit_angstrom_exponent=False, verbose=True, plot=True, live_fit=False)
    run_spectrogram_minimisation(w, method="newton")
    nsigma = 2
    labels = ["A1_T", "A2_T", "VAOD_T", "OZONE_T", "PWV_T"]
    indices = [0, 1, 3, 5, 6]
    A1, A2, A3, aerosols, angstrom_exponent, ozone, pwv, B, Astar, D, shift_x, shift_y, shift_t, *psf_poly_params = w.params.values
    ipar = w.params.get_free_parameters()  # non fixed param indices
    cov_indices = [list(ipar).index(k) for k in indices]  # non fixed param indices in cov matrix
    assert w.costs[-1] / w.data.size < 1e-3
    k = 0
    for i, l in zip(indices, labels):
        icov = cov_indices[k]
        spectrum.my_logger.info(f"Test {l} best-fit {w.params.values[i]:.3f}+/-{np.sqrt(w.params.cov[icov, icov]):.3f} "
                                f"vs {spectrum.header[l]:.3f} at {nsigma}sigma level: "
                                f"{np.abs(w.params.values[i] - spectrum.header[l]) / np.sqrt(w.params.cov[icov, icov]) < nsigma}")
        assert np.abs(w.params.values[i] - spectrum.header[l]) / np.sqrt(w.params.cov[icov, icov]) < nsigma
        k += 1
    assert np.isclose(shift_y, 0, atol=parameters.PIXSHIFT_PRIOR)  # shift_y
    assert np.isclose(D, spectrum.header["D2CCD_T"], atol=0.1)  # D2CCD
    assert np.isclose(B, 1, atol=1e-3)  # B
    assert np.isclose(Astar, 1, atol=1e-2)  # Astar
    assert np.all(np.isclose(psf_poly_params[(PSF_POLY_ORDER + 1):len(PSF_POLY_PARAMS_TRUTH)//N_DIFF_ORDERS - 1],
                             np.array(PSF_POLY_PARAMS_TRUTH)[(PSF_POLY_ORDER + 1):len(PSF_POLY_PARAMS_TRUTH)//N_DIFF_ORDERS - 1],
                             rtol=0.01, atol=0.01))


test_ctio_fullchain()
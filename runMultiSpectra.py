import numpy as np
import matplotlib.pyplot as plt

from spectractor import parameters
from spectractor.extractor.spectrum import Spectrum
from spectractor.fit.fit_multispectra import run_multispectra_minimisation, MultiSpectraFitWorkspace


def filter_data(file_names):  # pragma: no cover
    from scipy.stats import median_absolute_deviation
    D = []
    chi2 = []
    dx = []
    amplitude = []
    regs = []
    for name in file_names:
        # try:
        spectrum = Spectrum(name, fast_load=True)
        D.append(spectrum.header["D2CCD"])
        dx.append(spectrum.header["PIXSHIFT"])
        regs.append(np.log10(spectrum.header["PSF_REG"]))
        amplitude.append(np.sum(spectrum.data[300:]))
        if "CHI2_FIT" in spectrum.header:
            chi2.append(spectrum.header["CHI2_FIT"])
        # except:
        #    print(f"fail to open {name}")
    D = np.array(D)
    dx = np.array(dx)
    regs = np.array(regs)
    chi2 = np.array(chi2)
    k = np.arange(len(D))
    plt.plot(k, amplitude)
    plt.show()
    plt.plot(k, D)
    # plt.plot(k, np.polyval(np.polyfit(k, reg, deg=1), k))
    plt.axhline(np.median(D))
    plt.axhline(np.median(D) + 3 * median_absolute_deviation(D))
    plt.axhline(np.median(D) - 3 * median_absolute_deviation(D))
    plt.grid()
    plt.title("D2CCD")
    plt.show()
    filter_indices = np.logical_and(D > np.median(D) - 3 * median_absolute_deviation(D),
                                    D < np.median(D) + 3 * median_absolute_deviation(D))
    if len(chi2) > 0:
        filter_indices *= np.logical_and(chi2 > np.median(chi2) - 3 * median_absolute_deviation(chi2),
                                         chi2 < np.median(chi2) + 3 * median_absolute_deviation(chi2))
    filter_indices *= np.logical_and(dx > np.median(dx) - 3 * median_absolute_deviation(dx),
                                     dx < np.median(dx) + 3 * median_absolute_deviation(dx))
    filter_indices *= np.logical_and(regs > np.median(regs) - 3 * median_absolute_deviation(regs),
                                     regs < np.median(regs) + 3 * median_absolute_deviation(regs))
    plt.plot(k, D)
    plt.title("D2CCD")
    plt.plot(k[filter_indices], D[filter_indices], "ko")
    plt.show()
    plt.plot(k, dx)
    plt.title("dx")
    plt.plot(k[filter_indices], dx[filter_indices], "ko")
    plt.show()
    plt.plot(k, regs)
    plt.title("regs")
    plt.plot(k[filter_indices], regs[filter_indices], "ko")
    plt.show()
    if len(chi2) > 0:
        plt.title("chi2")
        plt.plot(k, chi2)
        plt.plot(k[filter_indices], chi2[filter_indices], "ko")
        plt.show()
    return np.array(file_names)[filter_indices]


if __name__ == "__main__":
    from argparse import ArgumentParser
    from spectractor.config import load_config
    from spectractor.fit.fitter import run_minimisation
    import os

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path', default=["tests/data/reduc_20170530_134_spectrum.fits"],
                        help="Input fits file name. It can be a list separated by spaces, or it can use * as wildcard.",
                        nargs='*')
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=True)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                        help="Write results in given output directory (default: ./outputs/).")
    parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    parser.add_argument("-c", "--config", dest="config", default="config/ctio.ini",
                        help="INI config file. (default: config.ctio.ini).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    load_config(args.config)

    file_names = []
    disperser_label = "Thor300"
    target_label = "HD111980"
    input_directory = "../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/"
    tag = "reduc"
    extension = ".fits"

    all_files = os.listdir(input_directory)
    parameters.VERBOSE = False
    parameters.DEBUG = False
    # for file_name in sorted(all_files):
    #     if tag not in file_name or extension not in file_name or "spectrum" not in file_name:
    #         continue
    #     try:
    #         s = Spectrum(os.path.join(input_directory, file_name), fast_load=True)
    #         if s.disperser_label == disperser_label and s.target.label == target_label:
    #             file_names.append(os.path.join(input_directory, file_name))
    #     except:
    #         print(f"File {file_name} buggy.")
    # HoloAmAg
    # file_names = ['../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_064_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_069_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_074_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_079_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_084_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_089_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_094_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_099_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_104_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_109_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_114_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_119_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_124_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_129_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_134_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_139_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_144_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_149_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_154_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_159_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_164_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_169_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_174_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_179_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_184_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_189_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_194_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_199_spectrum.fits']
    file_names = ['../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_064_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_069_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_074_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_079_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_084_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_089_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_094_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_099_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_104_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_109_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_114_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_119_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_124_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_129_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_134_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_139_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_144_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_149_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_154_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_159_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_164_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_169_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_174_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_179_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_184_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_189_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_194_spectrum.fits',
                  '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_199_spectrum.fits']
    # Thor300
    # file_names = ['../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_066_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_071_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_076_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_081_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_086_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_091_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_096_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_101_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_106_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_111_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_116_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_121_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_126_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_131_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_136_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_141_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_146_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_151_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_156_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_161_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_166_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_171_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_176_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_181_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_186_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_191_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/sim_20170530_196_spectrum.fits']
    # file_names = ['../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_061_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_066_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_071_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_076_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_081_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_086_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_091_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_096_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_101_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_106_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_111_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_116_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_121_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_126_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_131_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_136_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_141_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_146_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_151_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_156_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_161_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_166_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_171_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_176_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_181_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_186_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_191_spectrum.fits',
    #               '../CTIODataJune2017_reduced_RG715_v2_prod7.5/data_30may17_A2=0.1/reduc_20170530_196_spectrum.fits']

    print(file_names)

    file_names = filter_data(file_names)

    parameters.VERBOSE = True
    parameters.DEBUG = True
    output_filename = f"outputs/test_multispectra_{disperser_label}_{target_label}"
    w = MultiSpectraFitWorkspace(output_filename, file_names, bin_width=3, nsteps=1000, fixed_A1s=False,
                                 burnin=200, nbins=10, verbose=1, plot=True, live_fit=True, inject_random_A1s=False)
    run_multispectra_minimisation(w, method="newton")

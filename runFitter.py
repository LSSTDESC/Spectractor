from spectractor import parameters
from spectractor.fit.fit_spectrogram import SpectrogramFitWorkspace, run_spectrogram_minimisation
from spectractor.fit.fit_spectrum import SpectrumFitWorkspace, run_spectrum_minimisation
from spectractor.config import load_config

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path', default=["tests/data/reduc_20170530_134_spectrum.fits"],
                        help="Input fits file name. It can be a list separated by spaces, or it can use * as wildcard.",
                        nargs='*')
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                        help="Write results in given output directory (default: ./outputs/).")
    parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    parser.add_argument("-c", "--config", dest="config", default="config/ctio.ini",
                        help="INI config file. (default: config/ctio.ini).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    load_config(args.config)

    for file_name in file_names:
        atmgrid_filename = file_name.replace('sim', 'reduc').replace('spectrum', 'atmsim')
        w = SpectrumFitWorkspace(file_name, atmgrid_file_name=atmgrid_filename, nsteps=1000,
                                 burnin=200, nbins=10, verbose=1, plot=True, live_fit=False)
        run_spectrum_minimisation(w, method="newton")
        w = SpectrogramFitWorkspace(file_name, atmgrid_file_name=atmgrid_filename, nsteps=2000,
                                    burnin=1000, nbins=10, verbose=1, plot=True, live_fit=False)
        run_spectrogram_minimisation(w, method="newton")

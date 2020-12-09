from spectractor import parameters
from spectractor.simulation.simulator import AtmosphereGrid, SpectrumSimulatorSimGrid
from spectractor.config import load_config
from spectractor.simulation.image_simulation import ImageSim
from spectractor.logbook import LogBook
from spectractor.extractor.extractor import Spectractor

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path', default=["tests/data/reduc_20170530_134.fits"],
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
                        help="INI config file. (default: config.ctio.ini).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    load_config(args.config)
    logbook = LogBook(logbook=args.logbook)

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue
        spectrum_file_name = args.output_directory + '/' + tag.replace('.fits', '_spectrum.fits')
        atmgrid = AtmosphereGrid(file_name)
        SpectrumSimulatorSimGrid(spectrum_file_name, args.output_directory)
        image = ImageSim(file_name, spectrum_file_name, args.output_directory, A1=1, A2=1,
                         pwv=5, ozone=300, aerosols=0.03,
                         psf_poly_params=None, with_stars=True)
        sim_file_name = args.output_directory + tag.replace('reduc_', 'sim_')
        Spectractor(sim_file_name, args.output_directory, target, [xpos, ypos], disperser_label, args.config)

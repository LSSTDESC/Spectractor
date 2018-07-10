from spectractor import parameters
from spectractor.simulation.simulator import Simulator, SimulatorSimGrid, Atmosphere, AtmosphereGrid


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
    parser.add_argument("-c", "--csv", dest="csv", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    for file_name in file_names:
        spectrum_simulation = Simulator(file_name, pwv=3, ozone=350, aerosols=0.02,
                                        A1=1.1, A2=0.1, reso=2, D=56, shift=-3)
        atmgrid = AtmosphereGrid(file_name, args.output_directory)
        atm = Atmosphere(atmgrid.airmass, atmgrid.pressure, atmgrid.temperature)
        SimulatorSimGrid(file_name, args.output_directory)

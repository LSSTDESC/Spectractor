from spectractor import parameters
from spectractor.simulation.atmosphere import FullAtmosphereGrid
import numpy as np


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(dest="output", metavar='OUTPUT_PATH', default=["./full_atmospheric_grid.h5"],
                        help="Output file name", nargs='*')
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-z", "--airmass", dest="airmass",
                        help="Three numbers min,max,nsteps to set the airmass grid.", default="1,3,3")
    parser.add_argument("-p", "--pressure", dest="pressure",
                        help="Three numbers min,max,nsteps to set the pressure grid (in hPa).", default="800,1000,3")
    parser.add_argument("-t", "--temperature", dest="temperature",
                        help="Three numbers min,max,nsteps to set the temperature grid (in Celsius degree).", default="0,20,3")
    parser.add_argument("-w", "--pwv", dest="pwv",
                        help="Three numbers min,max,nsteps to set the precipitable water vapor (PWV) grid (in mm).",
                        default="0,10,3")
    parser.add_argument("-o", "--ozone", dest="ozone",
                        help="Three numbers min,max,nsteps to set the ozone concentration grid (in dobson).",
                        default="300,500,3")
    parser.add_argument("-a", "--aerosols", dest="aerosols",
                        help="Three numbers min,max,nsteps to set the aerosol concentration grid.",
                        default="0,0.1,3")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_name = args.output[0]

    airmass_grid = np.linspace(*[float(x) for x in args.airmass.split(',')[:-1]] + [int(args.airmass.split(',')[-1])])
    pressure_grid = np.linspace(*[float(x) for x in args.pressure.split(',')[:-1]]
                                 + [int(args.pressure.split(',')[-1])])
    temperature_grid = np.linspace(*[float(x) for x in args.temperature.split(',')[:-1]]
                                    + [int(args.temperature.split(',')[-1])])
    pwv_grid = np.linspace(*[float(x) for x in args.pwv.split(',')[:-1]] + [int(args.pwv.split(',')[-1])])
    ozone_grid = np.linspace(*[float(x) for x in args.ozone.split(',')[:-1]] + [int(args.ozone.split(',')[-1])])
    aerosol_grid = np.linspace(*[float(x) for x in args.aerosols.split(',')[:-1]] + [int(args.aerosols.split(',')[-1])])

    g = FullAtmosphereGrid(file_name=file_name, airmass_grid=airmass_grid, pressure_grid=pressure_grid,
                           temperature_grid=temperature_grid, pwv_grid=pwv_grid, ozone_grid=ozone_grid,
                           aerosol_grid=aerosol_grid)
    # g.compute()
    g.load(file_name)
    g.plot_transmission_image()

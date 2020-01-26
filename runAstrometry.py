from spectractor import parameters
from spectractor.astrometry import Astrometry, plot_shifts_histograms
from spectractor.logbook import LogBook
from spectractor.config import load_config

import numpy as np
import astropy.units as u

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
    parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    parser.add_argument("-c", "--config", dest="config", default="config/ctio.ini",
                        help="INI config file. (default: config.ctio.ini).")
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="",
                        help="Write results in given output directory (default: '' if same directory as input file).")
    parser.add_argument("-r", "--radius", dest="radius", default=parameters.CCD_IMSIZE,
                        help="Radius in pixel around the guessed target position to detect sources "
                             "and set the new WCS solution (default: parameters.CCD_IMSIZE).")
    parser.add_argument("-m", "--maxiter", dest="maxiter", default=10,
                        help="Maximum iterations before WCS solution convergence below 1 mas.")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    radius = int(args.radius)
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    load_config(args.config)

    logbook = LogBook(logbook=args.logbook)
    for file_name in file_names:
        tag = file_name.split('/')[-1]
        tag = tag.replace('sim_', 'reduc_')
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue
        a = Astrometry(file_name, target, disperser_label, output_directory=args.output_directory)
        extent = ((max(0, xpos - radius), min(xpos + radius, parameters.CCD_IMSIZE)),
                  (max(0, ypos - radius), min(ypos + radius, parameters.CCD_IMSIZE)))
        a.run_simple_astrometry(extent=extent)
        # iterate process until shift is below 1 mas on RA and DEC directions
        # or maximum iterations is reached
        dra, ddec = 0, 0
        for i in range(int(args.maxiter)):
            dra, ddec = a.run_gaia_astrometry()
            dra_median = np.median(dra.to(u.arcsec).value)
            ddec_median = np.median(ddec.to(u.arcsec).value)
            if np.abs(dra_median) < 1e-3 and np.abs(ddec_median) < 1e-3:
                break
        if parameters.DEBUG:
            plot_shifts_histograms(dra, ddec)
            a.plot_sources_and_gaia_catalog(sources=a.sources, gaia_coord=a.gaia_matches, label=target,
                                            quad=a.quad_stars_pixel_positions, margin=200)
            a.plot_astrometry_shifts(vmax=3)
        # overwrite input file
        # if args.overwrite:
        #     a.my_logger.warning(f"Overwrite option is True: {a.file_name} replaced by {a.new_file_name}")
        #     subprocess.check_output(f"mv {a.new_file_name} {a.file_name}", shell=True)

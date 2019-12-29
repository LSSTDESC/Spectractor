from spectractor import parameters
from spectractor.astrometry import Astrometry
from spectractor.logbook import LogBook
from spectractor.config import load_config

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
        tag = tag.replace('sim_', 'reduc_')
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue
        a = Astrometry(file_name, target, disperser_label)
        margin = 500
        a.run_simple_astrometry(extent=((xpos-margin, xpos+margin), (ypos-margin, ypos+margin)))
        for i in range(10):
            dra, ddec = a.run_gaia_astrometry()
            print(dra, ddec)
            if dra < 1e-3 and ddec < 1e-3:
                break
        if parameters.DEBUG or True:
            a.plot_sources_and_gaia_catalog(sources=a.sources, gaia_coord=a.gaia_matches, margin=200)
            a.plot_astrometry_shifts(vmax=3)

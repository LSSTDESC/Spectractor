from spectractor import parameters
from spectractor.astrometry import Astrometry
from spectractor.logbook import LogBook
from spectractor.config import load_config


if __name__ == "__main__":
    import sys
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
    parser.add_argument("-m", "--maxiter", dest="maxiter", default=20,
                        help="Maximum iterations to find best WCS (with the lowest residuals with the Gaia catalog).")
    parser.add_argument("-x", "--xy", dest="target_xy", default="0,0",
                        help="X,Y guessed position of the order 0, separated by a comma (default: 0,0).")
    parser.add_argument("-t", "--target", dest="target_label", default="",
                        help="Target label (default: '').")
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
        disperser_label = ""
        if args.target_xy == "0,0" and args.target_label == "":
            tag = file_name.split('/')[-1]
            tag = tag.replace('sim_', 'reduc_')
            disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
            if target_label is None or xpos is None or ypos is None:
                continue
        else:
            xpos, ypos = args.target_xy.split(",")
            target_label = args.target_label
            xpos = float(xpos)
            ypos = float(ypos)
            if target_label == "" or (xpos == 0 and ypos == 0):
                sys.exit("Options --xy and --target must be used together, one of these seems not set.")
        a = Astrometry(file_name, target_label=target_label, disperser_label=disperser_label,
                       output_directory=args.output_directory)
        extent = ((int(max(0, xpos - radius)), int(min(xpos + radius, parameters.CCD_IMSIZE))),
                  (int(max(0, ypos - radius)), int(min(ypos + radius, parameters.CCD_IMSIZE))))
        gaia_min_residuals = a.run_full_astrometry(extent=extent, maxiter=int(args.maxiter))
        # overwrite input file
        # if args.overwrite:
        #     a.my_logger.warning(f"Overwrite option is True: {a.file_name} replaced by {a.new_file_name}")
        #     subprocess.check_output(f"mv {a.new_file_name} {a.file_name}", shell=True)

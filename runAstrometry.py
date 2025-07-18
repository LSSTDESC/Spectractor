from spectractor import parameters
from spectractor.astrometry import Astrometry
from spectractor.logbook import LogBook
from spectractor.config import load_config, apply_rebinning_to_parameters
from spectractor.extractor.images import Image


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
    parser.add_argument("-r", "--radius", dest="radius", default=-1,
                        help="Radius in pixel around the guessed target position to detect sources "
                             "and set the new WCS solution (default: parameters.CCD_IMSIZE).")
    parser.add_argument("-m", "--maxiter", dest="maxiter", default=20,
                        help="Maximum iterations to find best WCS (with the lowest residuals with the Gaia catalog).")
    parser.add_argument("-x", "--xy", dest="target_xy", default="0,0",
                        help="X,Y guessed position of the order 0, separated by a comma (default: 0,0).")
    parser.add_argument("-t", "--target", dest="target_label", default="",
                        help="Target label (default: '').")
    parser.add_argument("-g", "--grating", dest="disperser_label", default="",
                        help="Disperser label (default: '').")
    args = parser.parse_args()

    load_config(args.config, rebin=False)

    radius = int(args.radius)
    if radius < 0:
        radius = parameters.CCD_IMSIZE

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    logbook = LogBook(logbook=args.logbook)
    for file_name in file_names:
        disperser_label = args.disperser_label
        if parameters.OBS_NAME == "CTIO":
            tag = file_name.split('/')[-1]
            tag = tag.replace('sim_', 'reduc_')
            disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
            guess = [xpos, ypos]
            if target_label is None or xpos is None or ypos is None:
                continue
        else:
            guess = None
            if args.target_xy != "0,0":
                xpos, ypos = args.target_xy.split(",")
                xpos = float(xpos)
                ypos = float(ypos)
                guess = [xpos, ypos]
            target_label = args.target_label
        im = Image(file_name, target_label=target_label, guess=guess,
                  disperser_label=disperser_label, config=args.config)
        a = Astrometry(im, output_directory=args.output_directory)
        extent = ((int(max(0, xpos - radius)), int(min(xpos + radius, parameters.CCD_IMSIZE))),
                  (int(max(0, ypos - radius)), int(min(ypos + radius, parameters.CCD_IMSIZE))))
        gaia_min_residuals = a.run_full_astrometry(extent=extent, maxiter=int(args.maxiter))
        # overwrite input file
        # if args.overwrite:
        #     a.my_logger.warning(f"Overwrite option is True: {a.file_name} replaced by {a.new_file_name}")
        #     subprocess.check_output(f"mv {a.new_file_name} {a.file_name}", shell=True)

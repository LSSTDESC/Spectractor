from spectractor import parameters
from spectractor.extractor.extractor import Spectractor
from spectractor.logbook import LogBook
from spectractor.config import load_config

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path', default=["tests/data/reduc_20170605_028.fits"],
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

    if parameters.LOGBOOK != "None":
        print(parameters.LOGBOOK, type(parameters.LOGBOOK))
        logbook = LogBook(logbook=args.logbook)

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        if parameters.LOGBOOK != "None":
            print(parameters.LOGBOOK)
            disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        else:
            disperser_label = parameters.DISPERSER_DEFAULT
            target = ""
            xpos = parameters.ORDER0_X
            ypos = parameters.ORDER0_Y
        if target is None or xpos is None or ypos is None:
            continue
        # file_name = "outputs/sim_20170530_134.fits"
        Spectractor(file_name, args.output_directory, [xpos, ypos], target, disperser_label, args.config)

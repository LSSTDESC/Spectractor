import spectractor.parameters as parameters
from spectractor.simulation.image_simulation import ImageSim


if __name__ == "__main__":

    from spectractor.logbook import LogBook
    from spectractor.config import load_config
    from argparse import ArgumentParser

    parser = ArgumentParser()
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

    load_config(args.config)

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = ['CTIODataJune2017_reduced_RG715_v2/data_30may17/reduc_20170530_134.fits']
    spectrum_file_name = 'outputs/reduc_20170530_134_spectrum.fits'
    # guess = [720, 670]
    # hologramme HoloAmAg
    # psf_poly_params = [0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955,
    #                    -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901,
    #                    0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673,
    #                    528.3594585697788, 628.4966480821147, 12.438043546369354, 499.99999999999835]
    psf_poly_params = None
    # file_name="../CTIOAnaJun2017/ana_31may17/OverScanRemove/trim_images/trim_20170531_150.fits"
    # guess = [840, 530]
    # target = "HD205905"

    logbook = LogBook(logbook=args.logbook)
    for file_name in file_names:
        tag = file_name.split('/')[-1]
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue

        image = ImageSim(file_name, spectrum_file_name, args.output_directory, A2=1,
                         psf_poly_params=psf_poly_params, with_stars=False)

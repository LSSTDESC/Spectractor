#----------------------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------------------

from spectractor import parameters
from spectractor.extractor.extractor import Spectractor
from spectractor.logbook import LogBook


import os
import sys
import pandas as pd





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

    if not 'workbookDir' in globals():
        workbookDir = os.getcwd()
    print('workbookDir: ' + workbookDir)
    # os.chdir(workbookDir)  # If you changed the current working dir, this will take you back to the workbook dir.



    sys.path.append(workbookDir)
    sys.path.append(os.path.dirname(workbookDir))


    # choose the date
    thedate = "20190214"
    #thedate = "20190215"

    #This defines the logbbok to be used
    logbookfilename = "simple_logbook_PicDuMidi_" + thedate + ".csv"

    #Read the logbook
    df = pd.read_csv(logbookfilename, sep=",", decimal=".", encoding='latin-1', header='infer')

    # Get the first filename
    print("The first filename in the logbook :",df.file[0])

    # Get the directory
    if thedate == "20190215":
        INPUTDIR = "/Users/dagoret/DATA/PicDuMidiFev2019/prod_20190215"
    else:
        INPUTDIR = "/Users/dagoret/DATA/PicDuMidiFev2019/prod_20190214"

    #select the filename to search for
    if thedate == "20190215":
        file_name = "T1M_20190215_225550_730_HD116405_Filtre_None_bin1x1.1_red.fit"
    else:
        file_name = "T1M_20190214_234122_495_HD116405-Ronchi_Filtre_None_bin1x1.1_red.fit"

    #defines the output
    output_directory = "output"

    #define the configuration file for Pic Du Midi
    config = "config/picdumidi.ini"

    # Run logbook
    logbook = LogBook(logbookfilename)


    #extract info from logbook to run Spectractor on the selected input file
    tag_file = file_name
    disperser_label, target, xpos, ypos = logbook.search_for_image(tag_file)


    print("logbook : filename ..........= ",tag_file)
    print("logbook : disperser_label .. = ", disperser_label)
    print("logbook : xpos ..............= ", xpos)
    print("logbook : ypos ..............= ", ypos)

    # full input filename
    fullfilename = os.path.join(INPUTDIR, file_name)


    #### RUN Spectractor

    Spectractor(fullfilename, output_directory, [xpos, ypos], target, disperser_label, config, logbook=logbookfilename)


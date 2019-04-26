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
    parser.add_argument("-n", "--obsnum", dest="linenum", default="0",
                       help="line number in logbook")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose

    linenum_str=args.linenum
    linenum=int(linenum_str)

    
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


    ########################################################################################################################
    #
    # 1) Configuration
    #
    ########################################################################################################################


    # Select the reduced rotated images
    #----------------------------------
    FLAG_ROTATION=False   # should not be used from now on


    # choose the date
    #-----------------
    #thedate = "20190214"
    thedate = "20190215"


    #This above defines the logbbok to be used
    #------------------------------------
    if FLAG_ROTATION:
        logbookfilename = "simple_logbook_PicDuMidi_" + thedate + "_rot_v2.csv"
    else:
        #logbookfilename = "simple_logbook_PicDuMidi_" + thedate + "_v2.csv"
        #logbookfilename = "simple_logbook_PicDuMidi_" + thedate + "_v4.csv"
        logbookfilename = "allobs_logbook_PicDuMidi_" + thedate + "_v4_filter_None.csv"


    #Read the logbook for testing
    #------------------------------
    df = pd.read_csv(logbookfilename, sep=",", decimal=".", encoding='latin-1', header='infer')


    #Select the index of the file in range 0..30
    #--------------------------

    idx_sel=linenum

    # Get the first filename
    #--------------------------
    filename_sel=df.file[idx_sel]

    # Defines the directory containing the reduced images
    #-------------------------------------------------------

    if thedate == "20190215":
        if FLAG_ROTATION:
            INPUTDIR = "/Users/dagoret/DATA/PicDuMidiFev2019/prod_20190215_v3"
        else:
            #INPUTDIR = "/Users/dagoret/DATA/PicDuMidiFev2019/prod_20190215_v2"
            INPUTDIR = "/Users/dagoret/DATA/PicDuMidiFev2019/prod_20190215_v4"
    else:
        if FLAG_ROTATION:
            INPUTDIR = "/Users/dagoret/DATA/PicDuMidiFev2019/prod_20190214_v3"
        else:
            #INPUTDIR = "/Users/dagoret/DATA/PicDuMidiFev2019/prod_20190214_v2"
            INPUTDIR = "/Users/dagoret/DATA/PicDuMidiFev2019/prod_20190214_v4"

    # Defines the input image filename required
    # -------------------------------------------------------


    #if thedate == "20190215":
    #    if FLAG_ROTATION:
    #        file_name = "T1M_20190215_225550_730_HD116405_Filtre_None_bin1x1.1_red_rot.fit"
    #    else:
    #        file_name = "T1M_20190215_225550_730_HD116405_Filtre_None_bin1x1.1_red.fit"
    #else:
    #    if FLAG_ROTATION:
    #        file_name = "T1M_20190214_234122_495_HD116405-Ronchi_Filtre_None_bin1x1.1_red_rot.fit"
    #    else:
    #        file_name = "T1M_20190214_234122_495_HD116405-Ronchi_Filtre_None_bin1x1.1_red.fit"
    file_name=filename_sel



    # Defines the output directory
    #------------------------------------
    output_directory = "output_test/"+thedate


    # Define the configuration file for Pic Du Midi
    #-----------------------------------------------

    if FLAG_ROTATION:
        config = "config/picdumidirot.ini"
    else:
        config = "config/picdumidi.ini"



    # Run logbook
    #--------------
    logbook = LogBook(logbookfilename)


    #extract info from logbook to run Spectractor on the selected input file
    tag_file = file_name
    disperser_label, target, xpos, ypos = logbook.search_for_image(tag_file)

    print("-----------------------------------------------------------------------------------------------------------")
    print("logbook : linenumber ........= ", linenum)
    print("logbook : filename ..........= ",tag_file)
    print("logbook : disperser_label .. = ", disperser_label)
    print("logbook : xpos ..............= ", xpos)
    print("logbook : ypos ..............= ", ypos)

    # Build the full input filename
    #----------------------
    fullfilename_input = os.path.join(INPUTDIR, file_name)


    #workout the outputfilename to test if it exists
    filetype = file_name.split('.')[-1]

    if filetype == "fits":
        # output_filename = file_name.split('/')[-1]
        output_filename = os.path.basename(file_name)
        output_filename = output_filename.replace('.fits', '_spectrum.fits')
        output_filename = output_filename.replace('.fz', '_spectrum.fits')
        output_filename = os.path.join(output_directory, output_filename)
        output_filename_spectrogram = output_filename.replace('spectrum', 'spectrogram')
        output_filename_psf = output_filename.replace('spectrum.fits', 'table.csv')
    elif filetype == "fit":
        output_filename = os.path.basename(file_name)
        output_filename = output_filename.replace('.fit', '_spectrum.fits')
        output_filename = output_filename.replace('.fz', '_spectrum.fits')
        output_filename = os.path.join(output_directory, output_filename)
        output_filename_spectrogram = output_filename.replace('spectrum', 'spectrogram')
        output_filename_psf = output_filename.replace('spectrum.fits', 'table.csv')
    else:
        output_filename = os.path.basename(file_name)
        output_filename = output_filename.replace('.fits', '_spectrum.fits')
        output_filename = output_filename.replace('.fz', '_spectrum.fits')
        output_filename = os.path.join(output_directory, output_filename)
        output_filename_spectrogram = output_filename.replace('spectrum', 'spectrogram')
        output_filename_psf = output_filename.replace('spectrum.fits', 'table.csv')



    # test if outputfilename exists
    fullfilename_output=output_filename
    fullfilename_output_spectrogram=output_filename_spectrogram
    fullfilename_output_psf = output_filename_psf

    print("fullfilename_output = {}".format(fullfilename_output))


    if os.path.isfile(fullfilename_output) and os.path.isfile(fullfilename_output_spectrogram) and os.path.isfile(fullfilename_output_psf):
        filesize=os.stat(fullfilename_output).st_size
        print(">>>>> output filename : {} already exists with size {} ! Skip Spectractor".format(fullfilename_output,filesize) )

        filesize = os.stat(fullfilename_output_spectrogram).st_size
        print(">>>>> output filename : {} already exists with size {} ! Skip Spectractor".format(fullfilename_output_spectrogram,
                                                                                                 filesize))

        filesize = os.stat(fullfilename_output_psf).st_size
        print(">>>>> output filename : {} already exists with size {} ! Skip Spectractor".format(
            fullfilename_output_psf,
            filesize))

        exit(0)


    #############################
    #### RUN Spectractor
    ############################

    parameters.DISPLAY=True

    Spectractor(fullfilename_input, output_directory, [xpos, ypos], target, disperser_label, config, logbook=logbookfilename)


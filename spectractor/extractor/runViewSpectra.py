#########################################################################################################
# View spectra produced in Spectractor
#########################################################################################################

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re

import matplotlib.pyplot as plt

if not 'workbookDir' in globals():
    workbookDir = os.getcwd()
print('workbookDir: ' + workbookDir)


import sys
sys.path.append(workbookDir)
sys.path.append(os.path.dirname(workbookDir))

from spectractor import parameters
from spectractor.extractor.extractor import Spectractor
from spectractor.logbook import LogBook
from spectractor.extractor.dispersers import *
from spectractor.extractor.spectrum import *



if __name__ == "__main__":

    #####################
    # 1) Configuration
    ######################

    parameters.VERBOSE = True
    parameters.DEBUG = True

    # thedate="20190214"
    thedate = "20190215"

    output_directory = "output/" + thedate

    parameters.VERBOSE = True
    parameters.DISPLAY = True



    ############################
    # 2) Get Spectra filelist
    #########################


    # get all files
    onlyfiles = [f for f in listdir(output_directory) if isfile(join(output_directory, f))]
    onlyfiles = np.array(onlyfiles)

    # sort files
    sortedindexes = np.argsort(onlyfiles)
    onlyfiles = onlyfiles[sortedindexes]

    # get only _spectrum.fits file
    onlyfilesspectrum = []
    for f in onlyfiles:
        if re.search("^T.*_spectrum.fits$", f):
            onlyfilesspectrum.append(re.findall("(^T.*_spectrum.fits$)", f)[0])

    onlyfilesspectrum = np.array(onlyfilesspectrum)
    sortedindexes = np.argsort(onlyfilesspectrum)
    onlyfilesspectrum = onlyfilesspectrum[sortedindexes]


    #get basnemae of files for later use (to check if _table.csv and _spectrogram.fits exists
    onlyfilesbasename=[]
    for f in onlyfilesspectrum:
        onlyfilesbasename.append(re.findall("(^T.*)_spectrum.fits$",f)[0])



    #############################################
    # 3) Plot Spectra
    ##########################################

    NBSPEC = len(sortedindexes)

    for idx in np.arange(0, NBSPEC):
        # if idx in [0,1,4]:
        #    continue

        print( .format("{}) : {}",idx,onlyfilespectrum[idx]))

        fullfilename = os.path.join(output_directory, onlyfiles[idx])
        s = Spectrum()
        s.load_spectrum(fullfilename)
        s.plot_spectrum(xlim=None, force_lines=True)















import sys,os

from spectroscopy import *
from images import *
import parameters 

            
def Spectractor(filename,outputdir,guess,target,atmospheric_lines=True):
    """ Spectractor
    Main function to extract a spectrum from an image

    Args:
        filename (:obj:`str`): path to the image
        outputdir (:obj:`str`): path to the output directory
    """
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTOR')
    # Load reduced image
    image = Image(filename,target=target)
    if parameters.DEBUG:
        image.plot_image(scale='log10',target_pixcoords=guess)
    # Set output path
    ensure_dir(outputdir)
    output_filename = filename.split('/')[-1]
    output_filename = output_filename.replace('.fits','_spectrum.fits')
    output_filename = os.path.join(outputdir,output_filename)
    # Find the exact target position in the raw cut image: several methods
    my_logger.info('\n\tSearch for the target in the image...')
    if parameters.DEBUG:
        target_pixcoords = image.find_target_1Dprofile(guess)
    target_pixcoords = image.find_target_2Dprofile(guess)
    # Rotate the image: several methods
    image.turn_image()
    # Find the exact target position in the rotated image: several methods
    my_logger.info('\n\tSearch for the target in the rotated image...')
    target_pixcoords_rotated = image.find_target_2Dprofile(guess,rotated=True)
    # Subtract background and bad pixels
    spectrum = image.extract_spectrum_from_image()
    spectrum.atmospheric_lines = atmospheric_lines
    # Calibrate the spectrum
    spectrum.calibrate_spectrum()
    spectrum.calibrate_spectrum_with_lines()
    # Subtract second order

    # Save the spectra
    spectrum.save_spectrum(output_filename,overwrite=True)

    

if __name__ == "__main__":
    import commands, string, re, time, os
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--debug", dest="debug",action="store_true",
                      help="Enter debug mode (more verbose and plots).",default=False)
    parser.add_option("-v", "--verbose", dest="verbose",action="store_true",
                      help="Enter verbose (print more stuff).",default=False)
    parser.add_option("-o", "--output_directory", dest="output_directory", default="test/",
                      help="Write results in given output directory (default: ./tests/).")
    (opts, args) = parser.parse_args()

    parameters.VERBOSE = opts.verbose
    if opts.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True
        
        
    #filename = "../../CTIODataJune2017_reducedRed/data_05jun17/reduc_20170605_00.fits"
    #filename = "notebooks/fits/trim_20170605_007.fits"
    #guess = [745,643]
    #target = "3C273"

    filename="../CTIOAnaJun2017/ana_05jun17/OverScanRemove/trim_images/trim_20170605_028.fits"
    guess = [814, 585]
    target = "PNG321.0+3.9"
    #filename="../CTIOAnaJun2017/ana_05jun17/OverScanRemove/trim_images/trim_20170605_026.fits"
    #guess = [735, 645]
    #target = "PNG321.0+3.9"
    filename="../CTIOAnaJun2017/ana_29may17/OverScanRemove/trim_images/trim_20170529_150.fits"
    guess = [720, 670]
    target = "HD185975"
    #filename="../CTIOAnaJun2017/ana_31may17/OverScanRemove/trim_images/trim_20170531_150.fits"
    #guess = [840, 530]
    #target = "HD205905"

    Spectractor(filename,opts.output_directory,guess,target)

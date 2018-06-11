from images import *
from spectroscopy import *


def Spectractor(filename, outputdir, guess, target, atmospheric_lines=True, line_detection=False):
    """ Spectractor
    Main function to extract a spectrum from an image

    Args:
        filename (str):
        outputdir: 
        guess: 
        target: 
        atmospheric_lines: 
        line_detection: 

    Returns:
        spectrum:
    """
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTOR')
    # Load reduced image
    image = Image(filename, target=target)
    if parameters.DEBUG:
        image.plot_image(scale='log10', target_pixcoords=guess)
    # Set output path
    ensure_dir(outputdir)
    output_filename = filename.split('/')[-1]
    output_filename = output_filename.replace('.fits', '_spectrum.fits')
    output_filename = os.path.join(outputdir, output_filename)

    # Test if file already exists
    # if os.path.exists(output_filename) and os.path.getsize(output_filename)>20000:
    #    filesize= os.path.getsize(output_filename)
    #    infostring=" !!!!!! Spectrum file file %s of size %d already exists, thus SKIP the reconstruction ..." % (output_filename,filesize)
    #    my_logger.info(infostring)
    #    return

    # Find the exact target position in the raw cut image: several methods
    my_logger.info('\n\tSearch for the target in the image...')
    target_pixcoords = image.find_target(guess)
    # Rotate the image: several methods
    image.turn_image()
    # Find the exact target position in the rotated image: several methods
    my_logger.info('\n\tSearch for the target in the rotated image...')
    target_pixcoords_rotated = image.find_target(guess, rotated=True)
    # Create Spectrum object
    spectrum = Spectrum(Image=image)
    # Subtract background and bad pixels
    image.extract_spectrum_from_image(spectrum)
    spectrum.atmospheric_lines = atmospheric_lines
    # Calibrate the spectrum
    spectrum.calibrate_spectrum()
    if line_detection:
        try:
            spectrum.calibrate_spectrum_with_lines()
        except:
            my_logger.warning('\n\tCalibration procedure with spectral features failed.')
            spectrum.header['WARNINGS'] = 'Calibration procedure with spectral features failed.'
    else:
        spectrum.header['WARNINGS'] = 'No calibration procedure with spectral features.'
    # Save the spectrum
    spectrum.save_spectrum(output_filename, overwrite=True)
    # Plot the spectrum
    if parameters.VERBOSE:
        if os.getenv("DISPLAY"): spectrum.plot_spectrum(xlim=None, nofit=False)
    return spectrum


if __name__ == "__main__":
    import os
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--debug", dest="debug", action="store_true",
                      help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="Enter verbose (print more stuff).", default=False)
    parser.add_option("-o", "--output_directory", dest="output_directory", default="test/",
                      help="Write results in given output directory (default: ./tests/).")
    (opts, args) = parser.parse_args()

    parameters.VERBOSE = opts.verbose
    if opts.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    # filename = "../../CTIODataJune2017_reducedRed/data_05jun17/reduc_20170605_005.fits"
    # filename = "notebooks/fits/trim_20170605_005.fits"
    filename = "../CTIOAnaJun2017/ana_05jun17/OverScanRemove/trim_images/trim_20170605_005.fits"
    guess = [745, 643]
    target = "3C273"

    # filename="../CTIOAnaJun2017/ana_05jun17/OverScanRemove/trim_images/trim_20170605_028.fits"
    # guess = [814, 585]
    # target = "PNG321.0+3.9"
    # filename="../CTIOAnaJun2017/ana_05jun17/OverScanRemove/trim_images/trim_20170605_026.fits"
    # guess = [735, 645]
    # target = "PNG321.0+3.9"
    # filename="../CTIOAnaJun2017/ana_29may17/OverScanRemove/trim_images/trim_20170529_150.fits"
    # guess = [720, 670]
    # target = "HD185975"
    # filename="../CTIOAnaJun2017/ana_31may17/OverScanRemove/trim_images/trim_20170531_150.fits"
    # guess = [840, 530]
    # target = "HD205905"

    Spectractor(filename, opts.output_directory, guess, target)

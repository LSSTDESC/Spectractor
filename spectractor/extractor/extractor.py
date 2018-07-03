from spectractor.extractor.images import *
from spectractor.extractor.spectroscopy import *
from spectractor import parameters
from spectractor.tools import ensure_dir


def Spectractor(file_name, output_directory, guess, target, atmospheric_lines=True, line_detection=False):
    """ Spectractor
    Main function to extract a spectrum from an image

    Args:
        file_name (str): input file nam of the image to analyse
        output_directory (str): output directory
        guess: [x0,y0] list of the guessed pixel positions of the target in the image (must be integers)
        target: (str) the name of the targeted object
        atmospheric_lines: if True atmospheric lines are used in the calibration fit
        line_detection: if True the absorption or emission lines are
            used to calibrate the pixel to wavelength relationship

    Returns:
        spectrum (Spectrum): the extracted spectrum
    """
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTOR')
    # Load reduced image
    image = Image(file_name, target=target)
    if parameters.DEBUG:
        image.plot_image(scale='log10', target_pixcoords=guess)
    # Set output path
    ensure_dir(output_directory)
    output_filename = file_name.split('/')[-1]
    output_filename = output_filename.replace('.fits', '_spectrum.fits')
    output_filename = os.path.join(output_directory, output_filename)
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
            spectrum.header['LSHIFT'] = 0.
            spectrum.header['D2CCD'] = DISTANCE2CCD
            spectrum.header['WARNINGS'] = 'Calibration procedure with spectral features failed.'
    else:
        spectrum.header['LSHIFT'] = 0.
        spectrum.header['D2CCD'] = DISTANCE2CCD
        spectrum.header['WARNINGS'] = 'No calibration procedure with spectral features.'
    # Save the spectrum
    spectrum.save_spectrum(output_filename, overwrite=True)
    # Plot the spectrum
    if parameters.VERBOSE:
        if os.getenv("DISPLAY"):
            spectrum.plot_spectrum(xlim=None, fit=True)
    return spectrum


if __name__ == "__main__":
    import os
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--debug", dest="debug", action="store_true",
                      help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="Enter verbose (print more stuff).", default=False)
    parser.add_option("-o", "--output_directory", dest="output_directory", default="outputs/",
                      help="Write results in given output directory (default: ./outputs/).")
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

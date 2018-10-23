from spectractor.extractor.spectroscopy import *
from spectractor import parameters
from spectractor.tools import ensure_dir


def Spectractor(file_name, output_directory, guess, target, atmospheric_lines=True, line_detection=True):
    """ Spectractor
    Main function to extract a spectrum from an image

    Parameters
    ----------
    file_name: str
        Input file nam of the image to analyse
    output_directory: str
        Output directory
    guess: [int,int]
        [x0,y0] list of the guessed pixel positions of the target in the image (must be integers)
    target: str
        The name of the targeted object
    atmospheric_lines: bool
        If True atmospheric lines are used in the calibration fit
    line_detection: bool
        If True the absorption or emission lines are
            used to calibrate the pixel to wavelength relationship

    Returns
    -------
    spectrum: Spectrum
        The extracted spectrum object

    Examples
    --------
    Look for the image charactristics:
    >>> import os
    >>> from spectractor.logbook import LogBook
    >>> logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
    >>> file_names = ['./tests/data/reduc_20170605_028.fits']
    >>> for file_name in file_names:
    ...     tag = file_name.split('/')[-1]
    ...     target, xpos, ypos = logbook.search_for_image(tag)
    ...     if target is None or xpos is None or ypos is None:
    ...         continue
    ...     spectrum = Spectractor(file_name, './tests/data/', [xpos, ypos], target, line_detection=False, atmospheric_lines=True)
    ...     assert spectrum is not None
    ...     assert os.path.isfile('tests/data/reduc_20170605_028_spectrum.fits')
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
    target_pixcoords = find_target(image, guess)
    # Rotate the image: several methods
    turn_image(image)
    # Find the exact target position in the rotated image: several methods
    my_logger.info('\n\tSearch for the target in the rotated image...')
    target_pixcoords_rotated = find_target(image, guess, rotated=True)
    # Create Spectrum object
    spectrum = Spectrum(image=image)
    # Subtract background and bad pixels
    extract_spectrum_from_image(image, spectrum)
    spectrum.atmospheric_lines = atmospheric_lines
    # Calibrate the spectrum
    calibrate_spectrum(spectrum)
    if line_detection:
        my_logger.info('\n\tCalibrating order %d spectrum...' % spectrum.order)
        #try:
        calibrate_spectrum_with_lines(spectrum)
        #except:
        #    my_logger.warning('\n\tCalibration procedure with spectral features failed.')
        #    spectrum.header['WARNINGS'] = 'Calibration procedure with spectral features failed.'
    else:
        spectrum.header['WARNINGS'] = 'No calibration procedure with spectral features.'
    # Save the spectrum
    spectrum.save_spectrum(output_filename, overwrite=True)
    # Plot the spectrum
    if parameters.VERBOSE and parameters.DISPLAY:
        spectrum.plot_spectrum(xlim=None, fit=True)
    return spectrum

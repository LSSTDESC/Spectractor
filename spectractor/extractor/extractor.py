from spectractor.extractor.spectrum import *
from spectractor import parameters
from spectractor.tools import ensure_dir


def Spectractor(file_name, output_directory, guess, target, disperser_label="", config='./config/ctio.ini',
                atmospheric_lines=True, line_detection=True,logbook=None):
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
    disperser_label: str
        The name of the disperser
    config: str
        The config file name
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
    Extract the spectrogram and its characteristics from the image:
    >>> import os
    >>> from spectractor.logbook import LogBook
    >>> logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
    >>> file_names = ['./tests/data/reduc_20170605_028.fits']
    >>> for file_name in file_names:
    ...     tag = file_name.split('/')[-1]
    ...     disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
    ...     if target is None or xpos is None or ypos is None:
    ...         continue
    ...     spectrum = Spectractor(file_name, './tests/data/', [xpos, ypos], target, disperser_label, './config/ctio.ini')
    ...     assert spectrum is not None
    ...     assert os.path.isfile('tests/data/reduc_20170605_028_spectrum.fits')
    """

    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTOR')
    # Load config file
    load_config(config)
    # Load reduced image
    image = Image(file_name, target=target, disperser_label=disperser_label,logbook=logbook)
    if parameters.DEBUG:
        image.plot_image(scale='log10', target_pixcoords=guess)
    # Set output path
    ensure_dir(output_directory)
    output_filename = file_name.split('/')[-1]
    output_filename = output_filename.replace('.fits', '_spectrum.fits')
    output_filename = output_filename.replace('.fz', '_spectrum.fits')
    output_filename = os.path.join(output_directory, output_filename)
    output_filename_spectrogram = output_filename.replace('spectrum','spectrogram')
    output_filename_psf = output_filename.replace('spectrum.fits','table.csv')
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
    extract_spectrum_from_image(image, spectrum, w=parameters.PIXWIDTH_SIGNAL,
                                ws = (parameters.PIXDIST_BACKGROUND,
                                      parameters.PIXDIST_BACKGROUND+parameters.PIXWIDTH_BACKGROUND),
                                right_edge=parameters.CCD_IMSIZE-200)
    spectrum.atmospheric_lines = atmospheric_lines
    # Calibrate the spectrum
    calibrate_spectrum(spectrum)
    if line_detection:
        my_logger.info('\n\tCalibrating order %d spectrum...' % spectrum.order)
        calibrate_spectrum_with_lines(spectrum)
    else:
        spectrum.header['WARNINGS'] = 'No calibration procedure with spectral features.'
    # Save the spectrum
    spectrum.save_spectrum(output_filename, overwrite=True)
    spectrum.save_spectrogram(output_filename_spectrogram, overwrite=True)
    # Plot the spectrum
    if parameters.VERBOSE and parameters.DISPLAY:
        spectrum.plot_spectrum(xlim=None)
    distance = spectrum.chromatic_psf.get_distance_along_dispersion_axis()
    spectrum.lambdas = np.interp(distance, spectrum.pixels, spectrum.lambdas)
    spectrum.chromatic_psf.table['lambdas'] = spectrum.lambdas
    spectrum.chromatic_psf.table.write(output_filename_psf, overwrite=True)
    return spectrum


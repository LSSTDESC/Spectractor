import os
import numpy as np
import matplotlib.pyplot as plt

from spectractor import parameters
from spectractor.config import set_logger, load_config
from spectractor.extractor.images import Image, find_target, turn_image
from spectractor.extractor.spectrum import (Spectrum, calibrate_spectrum,
                                            calibrate_spectrum_with_lines)
from spectractor.extractor.background import extract_spectrogram_background_sextractor
from spectractor.extractor.chromaticpsf import ChromaticPSF
from spectractor.extractor.psf import load_PSF
from spectractor.tools import ensure_dir, plot_image_simple, from_lambda_to_colormap, plot_spectrum_simple


def Spectractor(file_name, output_directory, target_label, guess=None, disperser_label="", config='./config/ctio.ini',
                atmospheric_lines=True, line_detection=True):
    """ Spectractor
    Main function to extract a spectrum from an image

    Parameters
    ----------
    file_name: str
        Input file nam of the image to analyse.
    output_directory: str
        Output directory.
    target_label: str
        The name of the targeted object.
    guess: [int,int], optional
        [x0,y0] list of the guessed pixel positions of the target in the image (must be integers). Mandatory if
        WCS solution is absent (default: None).
    disperser_label: str, optional
        The name of the disperser (default: "").
    config: str
        The config file name (default: "./config/ctio.ini").
    atmospheric_lines: bool, optional
        If True atmospheric lines are used in the calibration fit.
    line_detection: bool, optional
        If True the absorption or emission lines are
        used to calibrate the pixel to wavelength relationship.

    Returns
    -------
    spectrum: Spectrum
        The extracted spectrum object.

    Examples
    --------

    Extract the spectrogram and its characteristics from the image:

    .. doctest::

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
        >>> file_names = ['./tests/data/reduc_20170530_134.fits']
        >>> for file_name in file_names:
        ...     tag = file_name.split('/')[-1]
        ...     disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        ...     if target_label is None or xpos is None or ypos is None:
        ...         continue
        ...     spectrum = Spectractor(file_name, './tests/data/', guess=[xpos, ypos], target_label=target_label,
        ...                            disperser_label=disperser_label, config='./config/ctio.ini')

    .. doctest::
        :hide:

        >>> assert spectrum is not None
        >>> assert os.path.isfile('tests/data/educ_20170530_134_spectrum.fits')

    """

    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTOR')
    # Load config file
    load_config(config)

    # Load reduced image
    image = Image(file_name, target_label=target_label, disperser_label=disperser_label)
    if parameters.DEBUG:
        image.plot_image(scale='symlog', target_pixcoords=guess)
    # Set output path
    ensure_dir(output_directory)
    output_filename = file_name.split('/')[-1]
    output_filename = output_filename.replace('.fits', '_spectrum.fits')
    output_filename = output_filename.replace('.fz', '_spectrum.fits')
    output_filename = os.path.join(output_directory, output_filename)
    output_filename_spectrogram = output_filename.replace('spectrum', 'spectrogram')
    output_filename_psf = output_filename.replace('spectrum.fits', 'table.csv')
    # Find the exact target position in the raw cut image: several methods
    my_logger.info('\n\tSearch for the target in the image...')
    find_target(image, guess, use_wcs=True)
    # Rotate the image
    turn_image(image)
    # Find the exact target position in the rotated image: several methods
    my_logger.info('\n\tSearch for the target in the rotated image...')
    find_target(image, guess, rotated=True, use_wcs=True)
    # Create Spectrum object
    spectrum = Spectrum(image=image)
    # Subtract background and bad pixels
    extract_spectrum_from_image(image, spectrum, w=parameters.PIXWIDTH_SIGNAL,
                                ws=(parameters.PIXDIST_BACKGROUND,
                                    parameters.PIXDIST_BACKGROUND + parameters.PIXWIDTH_BACKGROUND),
                                right_edge=parameters.CCD_IMSIZE - 200)
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
    spectrum.lines.print_detected_lines(output_file_name=output_filename.replace('_spectrum.fits', '_lines.csv'),
                                        overwrite=True, amplitude_units=spectrum.units)
    # Plot the spectrum
    if parameters.VERBOSE and parameters.DISPLAY:
        spectrum.plot_spectrum(xlim=None)
    distance = spectrum.chromatic_psf.get_distance_along_dispersion_axis()
    lambdas = np.interp(distance, spectrum.pixels, spectrum.lambdas)
    spectrum.chromatic_psf.table['lambdas'] = lambdas
    spectrum.chromatic_psf.table.write(output_filename_psf, overwrite=True)
    return spectrum


def extract_spectrum_from_image(image, spectrum, w=10, ws=(20, 30), right_edge=parameters.CCD_IMSIZE - 200):
    """Extract the 1D spectrum from the image.

    Method : remove a uniform background estimated from the rectangular lateral bands

    The spectrum amplitude is the sum of the pixels in the 2*w rectangular window
    centered on the order 0 y position.
    The up and down backgrounds are estimated as the median in rectangular regions
    above and below the spectrum, in the ws-defined rectangular regions; stars are filtered
    as nan values using an hessian analysis of the image to remove structures.
    The subtracted background is the mean of the two up and down backgrounds.
    Stars are filtered.

    Prerequisites: the target position must have been found before, and the
        image turned to have an horizontal dispersion line

    Parameters
    ----------
    image: Image
        Image object from which to extract the spectrum
    spectrum: Spectrum
        Spectrum object to store new wavelengths, data and error arrays
    w: int
        Half width of central region where the spectrum is extracted and summed (default: 10)
    ws: list
        up/down region extension where the sky background is estimated with format [int, int] (default: [20,30])
    right_edge: int
        Right-hand pixel position above which no pixel should be used (default: 1800)
    """

    my_logger = set_logger(__name__)
    if ws is None:
        ws = [20, 30]
    my_logger.info(
        f'\n\tExtracting spectrum from image: spectrum with width 2*{w:d} pixels '
        f'and background from {ws[0]:d} to {ws[1]:d} pixels')

    # Make a data copy
    data = np.copy(image.data_rotated)[:, 0:right_edge]
    err = np.copy(image.stat_errors_rotated)[:, 0:right_edge]

    # Lateral bands to remove sky background
    Ny, Nx = data.shape
    y0 = int(image.target_pixcoords_rotated[1])
    ymax = min(Ny, y0 + ws[1])
    ymin = max(0, y0 - ws[1])

    # Roughly estimates the wavelengths and set start 0 nm before parameters.LAMBDA_MIN
    # and end 0 nm after parameters.LAMBDA_MAX
    lambdas = image.disperser.grating_pixel_to_lambda(np.arange(Nx) - image.target_pixcoords_rotated[0],
                                                      x0=image.target_pixcoords)
    pixel_start = int(np.argmin(np.abs(lambdas - (parameters.LAMBDA_MIN - 0))))
    pixel_end = min(right_edge, int(np.argmin(np.abs(lambdas - (parameters.LAMBDA_MAX + 0)))))
    if (pixel_end - pixel_start) % 2 == 0:  # spectrogram table must have odd size in x for the fourier simulation
        pixel_end -= 1

    # Create spectrogram
    data = data[ymin:ymax, pixel_start:pixel_end]
    err = err[ymin:ymax, pixel_start:pixel_end]
    Ny, Nx = data.shape
    my_logger.info(
        f'\n\tExtract spectrogram: crop rotated image [{pixel_start}:{pixel_end},{ymin}:{ymax}] (size ({Nx}, {Ny}))')

    # Extract the background on the rotated image
    bgd_index = np.concatenate((np.arange(0, Ny//2 - ws[0]), np.arange(Ny//2 + ws[0], Ny))).astype(int)
    bgd_model_func = extract_spectrogram_background_sextractor(data, err, ws=ws)
    bgd_res = ((data - bgd_model_func(np.arange(Nx), np.arange(Ny)))/err)[bgd_index]
    while np.nanmean(bgd_res)/np.nanstd(bgd_res) < -0.2 and parameters.PIXWIDTH_BOXSIZE >= 5:
        parameters.PIXWIDTH_BOXSIZE = max(5, parameters.PIXWIDTH_BOXSIZE // 2)
        my_logger.warning(f"\n\tPull distribution of background residuals has a negative mean which may lead to "
                          f"background over-subtraction: mean(pull)/RMS(pull)={np.nanmean(bgd_res)/np.nanstd(bgd_res)}."
                          f"This value should be greater than -0.5. To do so, parameters.PIXWIDTH_BOXSIZE is divided "
                          f"by 2 from {parameters.PIXWIDTH_BOXSIZE*2} -> {parameters.PIXWIDTH_BOXSIZE}.")
        bgd_model_func = extract_spectrogram_background_sextractor(data, err, ws=ws)
        bgd_res = ((data - bgd_model_func(np.arange(Nx), np.arange(Ny)))/err)[bgd_index]

    # bgd_model_func = extract_spectrogram_background_poly2D(data, ws=ws)

    # Fit the transverse profile
    my_logger.info(f'\n\tStart PSF1D transverse fit...')
    psf = load_PSF(psf_type=parameters.PSF_TYPE)
    s = ChromaticPSF(psf, Nx=Nx, Ny=Ny, deg=parameters.PSF_POLY_ORDER, saturation=image.saturation)
    s.fit_transverse_PSF1D_profile(data, err, w, ws, pixel_step=10, sigma_clip=5, bgd_model_func=bgd_model_func,
                                   saturation=image.saturation, live_fit=False)

    # Fill spectrum object
    spectrum.pixels = np.arange(pixel_start, pixel_end, 1).astype(int)
    spectrum.data = np.copy(s.table['flux_sum'])
    spectrum.err = np.copy(s.table['flux_err'])
    my_logger.debug(f'\n\tTransverse fit table:\n{s.table}')
    if parameters.DEBUG:
        s.plot_summary()

    # Fit the data:
    method = "noprior"
    mode = "1D"
    my_logger.info(f'\n\tStart ChromaticPSF polynomial fit with '
                   f'mode={mode} and amplitude_priors_method={method}...')
    w = s.fit_chromatic_psf(data, bgd_model_func=bgd_model_func, data_errors=err,
                            amplitude_priors_method=method, mode=mode)
    # w = s.fit_chromatic_psf(data, bgd_model_func=bgd_model_func, data_errors=err,
    #                         amplitude_priors_method="psf1d", mode="2D", verbose=True)
    if parameters.DEBUG:
        s.plot_summary()
        w.plot_fit()
    spectrum.spectrogram_fit = s.evaluate(s.poly_params, mode=mode)
    spectrum.spectrogram_residuals = (data - spectrum.spectrogram_fit - bgd_model_func(np.arange(Nx),
                                                                                       np.arange(Ny))) / err
    spectrum.chromatic_psf = s
    # spectrum.data = np.copy(s.table['amplitude'])
    spectrum.data = np.copy(w.amplitude_params)
    spectrum.err = np.copy(w.amplitude_params_err)

    # fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex="all")
    # x = np.arange(spectrum.data.size)
    # ax[0].errorbar(x, s.table['flux_sum'], yerr=s.table['flux_err'], fmt="k.", label="flux_sum")
    # ax[0].errorbar(x, w.amplitude_params, yerr=w.amplitude_params_err, fmt="r.", label="amplitudes")
    # truth = parameters.AMPLITUDE_TRUTH*parameters.LAMBDA_TRUTH*np.gradient(parameters.LAMBDA_TRUTH)
    # truth *= parameters.FLAM_TO_ADURATE
    # ax[0].plot(np.arange(parameters.AMPLITUDE_TRUTH.size), truth, label="Truth")
    # ax[0].grid()
    # ax[0].legend()
    # ax[0].set_xlabel("X [pixels]")
    # ax[0].set_title(f"lambda={parameters.PSF_FIT_REG_PARAM}")
    # ax[0].set_ylabel('Spectrum amplitudes')
    # ax[1].errorbar(x, s.table['flux_sum']-truth, yerr=s.table['flux_err'], fmt="k.", label="flux_sum")
    # ax[1].axhline(0, color="k")
    # ax[1].grid()
    # ax[1].legend()
    # ax[1].set_xlabel("X [pixels]")
    # ax[1].set_ylabel('Sum-Truth')
    # ax[2].errorbar(x, w.amplitude_params-truth, yerr=w.amplitude_params_err, fmt="r.", label="amplitudes")
    # ax[2].axhline(0, color="k")
    # ax[2].grid()
    # ax[2].legend()
    # ax[2].set_xlabel("X [pixels]")
    # ax[2].set_ylabel('Amplitudes-Truth')
    # fig.tight_layout()
    # plt.show()

    s.table['Dx_rot'] = spectrum.pixels.astype(float) - image.target_pixcoords_rotated[0]
    s.table['Dx'] = np.copy(s.table['Dx_rot'])
    s.table['Dy'] = s.table['y_mean'] - (image.target_pixcoords_rotated[1] - ymin)
    s.table['Dy_fwhm_inf'] = s.table['Dy'] - 0.5 * s.table['fwhm']
    s.table['Dy_fwhm_sup'] = s.table['Dy'] + 0.5 * s.table['fwhm']
    s.table['y_mean'] = s.table['y_mean'] - (image.target_pixcoords_rotated[1] - ymin)
    my_logger.debug(f"\n\tTransverse fit table before derotation:\n{s.table[['Dx_rot', 'y_mean', 'Dx', 'Dy']]}")

    # rotate and save the table
    s.rotate_table(-image.rotation_angle)
    my_logger.debug(f"\n\tTransverse fit table after derotation:\n{s.table[['Dx_rot', 'y_mean', 'Dx', 'Dy']]}")

    # Extract the spectrogram edges
    data = np.copy(image.data)[:, 0:right_edge]
    err = np.copy(image.stat_errors)[:, 0:right_edge]
    Ny, Nx = data.shape
    x0 = int(image.target_pixcoords[0])
    y0 = int(image.target_pixcoords[1])
    ymax = min(Ny, y0 + int(s.table['Dy_mean'].max()) + ws[1] + 1)  # +1 to  include edges
    ymin = max(0, y0 + int(s.table['Dy_mean'].min()) - ws[1])
    distance = s.get_distance_along_dispersion_axis()
    lambdas = image.disperser.grating_pixel_to_lambda(distance, x0=image.target_pixcoords)
    lambda_min_index = int(np.argmin(np.abs(lambdas - (parameters.LAMBDA_MIN - 0))))
    lambda_max_index = int(np.argmin(np.abs(lambdas - (parameters.LAMBDA_MAX + 0))))
    xmin = int(s.table['Dx'][lambda_min_index] + x0)
    xmax = min(right_edge, int(s.table['Dx'][lambda_max_index] + x0) + 1)  # +1 to  include edges
    if (xmax - xmin) % 2 == 0:  # spectrogram must have odd size in x for the fourier simulation
        xmax -= 1
        s.table.remove_row(-1)
    # Position of the order 0 in the spectrogram coordinates
    target_pixcoords_spectrogram = [image.target_pixcoords[0] - xmin, image.target_pixcoords[1] - ymin]

    # Create spectrogram
    data = data[ymin:ymax, xmin:xmax]
    err = err[ymin:ymax, xmin:xmax]
    Ny, Nx = data.shape

    # Extract the non rotated background
    bgd_model_func = extract_spectrogram_background_sextractor(data, err, ws=ws)
    bgd = bgd_model_func(np.arange(Nx), np.arange(Ny))

    # Crop the background lateral regions
    # bgd_width = ws[1] - w
    # yeven = 0
    # if (Ny - 2 * bgd_width) % 2 == 0:  # spectrogram must have odd size in y for the fourier simulation
    #     yeven = 1
    # ymax = ymax - bgd_width + yeven
    # ymin += bgd_width
    # bgd = bgd[bgd_width:-bgd_width + yeven, :]
    # data = data[bgd_width:-bgd_width + yeven, :]
    # err = err[bgd_width:-bgd_width + yeven, :]
    # Ny, Nx = data.shape
    # target_pixcoords_spectrogram[1] -= bgd_width

    # Spectrogram must have odd size in y for the fourier simulation
    if Ny % 2 == 0:
        ymax = ymax - 1
        bgd = bgd[:-1, :]
        data = data[:-1, :]
        err = err[:-1, :]
        Ny, Nx = data.shape

    # First guess for lambdas
    first_guess_lambdas = image.disperser.grating_pixel_to_lambda(s.get_distance_along_dispersion_axis(),
                                                                  x0=image.target_pixcoords)
    s.table['lambdas'] = first_guess_lambdas
    spectrum.lambdas = np.array(first_guess_lambdas)
    my_logger.debug(f"\n\tTransverse fit table after derotation:\n{s.table[['lambdas', 'Dx_rot', 'Dx', 'Dy']]}")

    # Position of the order 0 in the spectrogram coordinates
    my_logger.info(f'\n\tExtract spectrogram: crop image [{xmin}:{xmax},{ymin}:{ymax}] (size ({Nx}, {Ny}))'
                   f'\n\tNew target position in spectrogram frame: {target_pixcoords_spectrogram}')

    # Save results
    spectrum.spectrogram = data
    spectrum.spectrogram_err = err
    spectrum.spectrogram_bgd = bgd
    spectrum.spectrogram_x0 = target_pixcoords_spectrogram[0]
    spectrum.spectrogram_y0 = target_pixcoords_spectrogram[1]
    spectrum.spectrogram_xmin = xmin
    spectrum.spectrogram_xmax = xmax
    spectrum.spectrogram_ymin = ymin
    spectrum.spectrogram_ymax = ymax
    spectrum.spectrogram_deg = spectrum.chromatic_psf.deg
    spectrum.spectrogram_saturation = spectrum.chromatic_psf.saturation

    # Plot FHWM(lambda)
    if parameters.DEBUG:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex="all")
        ax[0].plot(spectrum.lambdas, np.array(s.table['fwhm']))
        ax[0].set_xlabel(r"$\lambda$ [nm]")
        ax[0].set_ylabel("Transverse FWHM [pixels]")
        ax[0].set_ylim((0.8 * np.min(s.table['fwhm']), 1.2 * np.max(s.table['fwhm'])))  # [-10:])))
        ax[0].grid()
        ax[1].plot(spectrum.lambdas, np.array(s.table['y_mean']))
        ax[1].set_xlabel(r"$\lambda$ [nm]")
        ax[1].set_ylabel("Distance from mean dispersion axis [pixels]")
        # ax[1].set_ylim((0.8*np.min(s.table['Dy']), 1.2*np.max(s.table['fwhm'][-10:])))
        ax[1].grid()
        if parameters.DISPLAY:
            plt.show()
        if parameters.LSST_SAVEFIGPATH:
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'fwhm.pdf'))

    # Summary plot
    if parameters.DEBUG or parameters.LSST_SAVEFIGPATH:
        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(12, 6))
        xx = np.arange(s.table['Dx_rot'].size)
        plot_image_simple(ax[2], data=data, scale="symlog", title='', units=image.units, aspect='auto')
        ax[2].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_mean'], label='Dispersion axis')
        ax[2].scatter(xx, target_pixcoords_spectrogram[1] + s.table['Dy'],
                      c=s.table['lambdas'], edgecolors='None', cmap=from_lambda_to_colormap(s.table['lambdas']),
                      label='Fitted spectrum centers', marker='o', s=10)
        ax[2].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_fwhm_inf'], 'k-', label='Fitted FWHM')
        ax[2].plot(xx, target_pixcoords_spectrogram[1] + s.table['Dy_fwhm_sup'], 'k-', label='')
        ax[2].set_ylim(0, Ny)
        ax[2].set_xlim(0, xx.size)
        ax[2].legend(loc='best')
        plot_spectrum_simple(ax[0], np.arange(spectrum.data.size), spectrum.data, data_err=spectrum.err,
                             units=image.units, label='Fitted spectrum', xlim=[0, spectrum.data.size])
        ax[0].plot(xx, s.table['flux_sum'], 'k-', label='Cross spectrum')
        ax[0].set_xlim(0, xx.size)
        ax[0].legend(loc='best')
        ax[1].plot(xx, (s.table['flux_sum'] - s.table['flux_integral']) / s.table['flux_sum'],
                   label='(model_integral-cross_sum)/cross_sum')
        ax[1].legend()
        ax[1].grid(True)
        ax[1].set_ylim(-1, 1)
        ax[1].set_ylabel('Relative difference')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        pos0 = ax[0].get_position()
        pos1 = ax[1].get_position()
        pos2 = ax[2].get_position()
        ax[0].set_position([pos2.x0, pos0.y0, pos2.width, pos0.height])
        ax[1].set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])
        if parameters.DISPLAY:
            plt.show()
        if parameters.LSST_SAVEFIGPATH:
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'spectrum.pdf'))
    return spectrum

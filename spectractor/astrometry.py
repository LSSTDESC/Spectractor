import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.io import fits

from photutils import Background2D, SExtractorBackground, DAOStarFinder
from spectractor import parameters
from spectractor.tools import plot_image_simple, ensure_dir
from spectractor.config import set_logger
from spectractor.extractor.images import Image


def remove_background(data, sigma=3.0, box_size=(50, 50), filter_size=(3, 3)):
    sigma_clip = SigmaClip(sigma=sigma)
    bkg_estimator = SExtractorBackground()
    bkg = Background2D(data, box_size, filter_size=filter_size,
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data_wo_bkg = data - bkg.background
    if parameters.DEBUG:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(bkg.background, origin='lower')
        ax[1].imshow(np.log10(1 + data_wo_bkg), origin='lower')
        plt.show()
    return data_wo_bkg


def source_detection(data_wo_bkg, sigma=3.0, fwhm=3.0, threshold_std_factor=3.5):
    mean, median, std = sigma_clipped_stats(data_wo_bkg, sigma=sigma)
    mask = np.zeros(data_wo_bkg.shape, dtype=bool)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_std_factor * std)
    sources = daofind(data_wo_bkg - median, mask=mask)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
    sources.sort('mag')
    if parameters.DEBUG:
        print(sources)
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        plt.figure(figsize=(10, 10))
        plot_image_simple(plt.gca(), data_wo_bkg, scale="log10", target_pixcoords=positions)
        plt.show()
    return sources


def run_astrometry(file_name, guess, target, disperser_label="", extent=None):
    """Build a World Coordinate System (WCS) using astrometry.net library given an exposure as a FITS file.

    The name of the target must be given to get its RA,DEC coordinates via a Simbad query.
    First the background of the exposure is removed using the astropy SExtractorBackground() method.
    Then photutils source_detection() is used to get the positions in pixels en flux of the objects in the field.
    The results are saved in the {file_name}_sources.fits file and used by the solve_field command from the
    astrometry.net library. The solve_field path must be set using the spectractor.parameters.ASTROMETRYNET_BINDIR
    variable. A new WCS is created and merged with the given exposure.
    The intermediate results are saved in a new directory named as the FITS file name with a _wcs suffix.

    Parameters
    ----------
    file_name: str
        Input file nam of the image to analyse.
    guess: [int,int]
        [x0,y0] list of the guessed pixel positions of the target in the image (must be integers).
    target: str
        The name of the targeted object.
    disperser_label: str
        The name of the disperser (default: "").
    extent: 2-tuple
        ((ymin,ymax),(xmin,xmax)) 2 dimensional typle to crop the exposure before any operation (default: None).

    Examples
    --------

    >>> import os
    >>> from spectractor.logbook import LogBook
    >>> from spectractor import parameters
    >>> parameters.VERBOSE = True
    >>> logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
    >>> file_names = ['./tests/data/reduc_20170530_134.fits']
    >>> for file_name in file_names:
    ...     tag = file_name.split('/')[-1]
    ...     disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
    ...     if target is None or xpos is None or ypos is None:
    ...         continue
    ...     run_astrometry(file_name, [xpos, ypos], target, disperser_label)
    ...     assert os.path.isdir('./tests/data/reduc_20170530_134_wcs')


    """
    my_logger = set_logger(__name__)
    # load image
    im = Image(file_name, target=target, disperser_label=disperser_label)
    # prepare outputs
    output_directory = os.path.join(os.path.dirname(file_name),
                                    os.path.splitext(os.path.basename(file_name))[0]) + "_wcs"
    my_logger.info(f"\n\tIntermediate outputs will be stored in {output_directory}")
    ensure_dir(output_directory)
    tag = os.path.splitext(os.path.basename(file_name))[0]
    # crop data
    if extent is not None:
        data = im.data[extent[0][0]:extent[0][1], extent[1][0]:extent[1][1]]
    else:
        data = np.copy(im.data)
    if parameters.DEBUG:
        im.plot_image(scale="log10")
    # remove background
    my_logger.info('\n\tRemove background using astropy SExtractorBackground()...')
    data_wo_bkg = remove_background(data)
    # extract source positions and fluxes
    my_logger.info('\n\tDetect sources using photutils source_detection()...')
    sources = source_detection(data_wo_bkg)
    # write results in fits file
    colx = fits.Column(name='X', format='D', array=sources['xcentroid'])
    coly = fits.Column(name='Y', format='D', array=sources['ycentroid'])
    colflux = fits.Column(name='FLUX', format='D', array=sources['flux'])
    coldefs = fits.ColDefs([colx, coly, colflux])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.header['IMAGEW'] = data.shape[1]
    hdu.header['IMAGEH'] = data.shape[0]
    output_sources_fitsfile = f"{output_directory}/{tag}_sources.fits"
    hdu.writeto(output_sources_fitsfile, overwrite=True)
    my_logger.info(f'\n\tSources positions saved in {output_sources_fitsfile}')
    # run astrometry.net
    command = f"{os.path.join(parameters.ASTROMETRYNET_BINDIR, 'solve-field')} --scale-unit arcsecperpix " \
              f"--scale-low {0.9 * parameters.CCD_PIXEL2ARCSEC} " \
              f"--scale-high {1.1 * parameters.CCD_PIXEL2ARCSEC} " \
              f"--ra {im.target.coord.ra.value} --dec {im.target.coord.dec.value} " \
              f"--radius {2 * parameters.CCD_IMSIZE * parameters.CCD_PIXEL2ARCSEC / 3600.} " \
              f"--dir {output_directory} --out {tag} " \
              f"--overwrite --x-column X --y-column Y {output_sources_fitsfile}"
    my_logger.info(f'\n\tRun astrometry.net solve_field command:\n\t{command}')
    log = subprocess.check_output(command, shell=True)
    log_file = open(f"{output_directory}/{tag}.log", "w+")
    log_file.write(command + "\n")
    log_file.write(log.decode("utf-8") + "\n")
    # save new WCS in original fits file
    new_file_name = file_name.replace('.fits', '_new.fits')
    command = f"{os.path.join(parameters.ASTROMETRYNET_BINDIR, 'new-wcs')} -v -d -i {file_name} " \
              f"-w {os.path.join(output_directory, tag + '.wcs')} -o {new_file_name}\n"
    # f"mv {new_file_name} {file_name}"
    my_logger.info(f'\n\tSave WCS in original file:\n\t{command}')
    log = subprocess.check_output(command, shell=True)
    log_file.write(command + "\n")
    log_file.write(log.decode("utf-8") + "\n")
    log_file.close()

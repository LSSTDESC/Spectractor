import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.io import fits

from photutils import Background2D, SExtractorBackground, DAOStarFinder
from spectractor import parameters
from spectractor.tools import plot_image_simple
from spectractor.config import set_logger
from spectractor.extractor.images import Image

def remove_background(data, sigma=3.0, box_size=(50,50), filter_size=(3,3)):
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


def run_astrometry(file_name, extent=None):
    my_logger = set_logger(__name__)
    # load image
    im = Image(file_name, target=target, disperser_label=disperser_label)
    if extent is not None:
        data = im.data[extent[0][0]:extent[0][1], extent[1][0]:extent[1][1]]
    else:
        data = np.copy(im.data)
    if parameters.DEBUG:
        im.plot_image(scale="log10")
    if parameters.VERBOSE:
        my_logger.info('\n\tRemove background using astropy SExtractorBackground()...')
    data_wo_bkg = remove_background(data)
    if parameters.VERBOSE:
        my_logger.info('\n\tDetect sources using photutils...')
    sources = source_detection(data_wo_bkg)
    colx = fits.Column(name='X', format='D', array=sources['xcentroid'])
    coly = fits.Column(name='Y', format='D', array=sources['ycentroid'])
    colflux = fits.Column(name='FLUX', format='D', array=sources['flux'])
    coldefs = fits.ColDefs([colx, coly, colflux])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.header['IMAGEW'] = data.shape[1]
    hdu.header['IMAGEH'] = data.shape[0]
    hdu.writeto(f"astrometry/{tag}_sources.fits", overwrite=True)
    if parameters.VERBOSE:
        my_logger.info(f'\n\tSources positions saved in {oihohi}')
    command = f"{parameters.ASTROMETRYNET_BINDIR}/solve-field --scale-unit arcsecperpix " \
              f"--scale-low {0.9*parameters.CCD_PIXEL2ARCSEC} " \
              f"--scale-high {1.1*parameters.CCD_PIXEL2ARCSEC} " \
              f"--ra {im.target.coord.ra.value} --dec {im.target.coord.dec.value} --radius 0.5 " \
              f"--dir astrometry --out {tag} " \
              f"--overwrite --x-column X --y-column Y astrometry/{tag}_sources.fits"
    if parameters.VERBOSE:
        my_logger.info(f'\n\tRun astrometry.net solve_field:\n\t{command}')



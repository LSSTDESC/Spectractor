import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.io import fits
import astropy.units as u
from astropy import wcs as WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, Distance

from photutils import Background2D, SExtractorBackground, DAOStarFinder

from spectractor import parameters
from spectractor.tools import plot_image_simple, ensure_dir
from spectractor.config import set_logger
from spectractor.extractor.images import Image


def set_wcs_output_directory(file_name):
    output_directory = os.path.join(os.path.dirname(file_name),
                                    os.path.splitext(os.path.basename(file_name))[0]) + "_wcs"
    ensure_dir(output_directory)
    return output_directory


def set_file_tag(file_name):
    tag = os.path.splitext(os.path.basename(file_name))[0]
    return tag


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


def load_gaia_catalog(target):
    from astroquery.gaia import Gaia

    job = Gaia.cone_search_async(target.coord, radius=5 * u.arcmin)
    gaia_catalog = job.get_results()
    gaia_catalog.fill_value = 0
    gaia_catalog['parallax'].fill_value = np.min(gaia_catalog['parallax'])
    return gaia_catalog


def load_wcs_from_file(filename):
    # Load the FITS hdulist using astropy.io.fits
    hdulist = fits.open(filename)
    # Parse the WCS keywords in the primary HDU
    wcs = WCS.WCS(hdulist[0].header)
    return wcs


def update_target_coord_with_proper_motion(target, date_obs):
    target_pmra = target.simbad[0]['PMRA'] * u.mas / u.yr
    if np.isnan(target_pmra):
        target_pmra = 0 * u.mas / u.yr
    target_pmdec = target.simbad[0]['PMDEC'] * u.mas / u.yr
    if np.isnan(target_pmdec):
        target_pmdec = 0 * u.mas / u.yr
    target_parallax = target.simbad[0]['PLX_VALUE'] * u.mas
    if target_parallax == 0 * u.mas:
        target_parallax = 1e-4 * u.mas
    target_coord = SkyCoord(ra=target.coord.ra, dec=target.coord.dec, distance=Distance(parallax=target_parallax),
                            pm_ra_cosdec=target_pmra, pm_dec=target_pmdec, frame='icrs', equinox="J2000",
                            obstime="J2000")
    target_coord_after_proper_motion = target_coord.apply_space_motion(new_obstime=Time(date_obs))
    # c_list = target_coord.apply_space_motion(
    # new_obstime=Time(np.arange(2000,Time(date_obs.byear, 1), format='decimalyear)) * u.yr)
    return target_coord_after_proper_motion


def update_gaia_catalog_with_proper_motion(gaia_catalog, date_obs):
    parallax = np.array(gaia_catalog['parallax'].filled(np.nanmin(np.abs(gaia_catalog['parallax']))))
    parallax[parallax < 0] = np.min(parallax[parallax > 0])
    gaia_stars = SkyCoord(ra=gaia_catalog['ra'], dec=gaia_catalog['dec'], frame='icrs', equinox="J2000",
                          obstime=Time(gaia_catalog['ref_epoch'], format='decimalyear'),
                          pm_ra_cosdec=gaia_catalog['pmra'].filled(0)
                                       * np.cos(np.array(gaia_catalog['dec']) * np.pi / 180),
                          pm_dec=gaia_catalog['pmdec'].filled(0),
                          distance=Distance(parallax=parallax * u.mas, allow_negative=True))
    gaia_stars_after_proper_motion = gaia_stars.apply_space_motion(new_obstime=Time(date_obs))
    return gaia_stars_after_proper_motion


def plot_sources_and_gaia_catalog(im, wcs, sources, target_coord, gaia_coord):
    target_x, target_y = wcs.all_world2pix(target_coord.ra, target_coord.dec, 0)
    gaia_x, gaia_y = wcs.all_world2pix(gaia_coord.ra, gaia_coord.dec, 0)

    fig = plt.figure(figsize=(10, 10))

    fig.add_subplot(111, projection=wcs)

    plot_image_simple(plt.gca(), im.data, scale="log10")
    plt.xlabel('RA')
    plt.ylabel('Dec')
    # plt.grid(color='white', ls='solid')
    plt.scatter(sources[0], sources[1], s=300, lw=3,
                edgecolor='yellow', facecolor='none', label="all detected sources")
    plt.scatter(target_x, target_y, s=300,
                edgecolor='cyan', facecolor='none', label=f"the target {target} after motion", lw=3)
    # plt.scatter(c_list.ra, c_list.dec, transform=plt.gca().get_transform('icrs'), s=30,
    #             edgecolor='red', facecolor='none', label=f"motion", lw=3)
    # plt.scatter(x_field, y_field, s=300,
    #             edgecolor='green', facecolor='none', label=f"gaia stars", lw=3)
    plt.scatter(gaia_x, gaia_y, s=300,
                edgecolor='blue', facecolor='none', label=f"gaia stars after motion", lw=3)
    # plt.scatter(x_list, y_list, s=30,
    #             edgecolor='cyan', facecolor='none', label=f"motion", lw=3)
    plt.legend()
    margin = 100
    plt.xlim(target_x - margin + 0, target_x + margin + 0)
    plt.ylim(target_y - margin + 0, target_y + margin + 0)
    plt.show()


def run_simple_astrometry(file_name, target, disperser_label="", extent=None):
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
    ...     run_simple_astrometry(file_name, target, disperser_label)
    ...     assert os.path.isdir('./tests/data/reduc_20170530_134_wcs')


    """
    my_logger = set_logger(__name__)
    # load image
    im = Image(file_name, target=target, disperser_label=disperser_label)
    # prepare outputs
    output_directory = set_wcs_output_directory(file_name)
    tag = set_file_tag(file_name)
    my_logger.info(f"\n\tIntermediate outputs will be stored in {output_directory}")
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


def run_gaia_astrometry(file_name, target, disperser_label=""):
    """Refine a World Coordinate System (WCS) using Gaia satellite astrometry catalog.

    A WCS must be already present in the exposure FITS file.

    The name of the target must be given to get its RA,DEC coordinates via a Simbad query.
    A matching is performed between the detected sources and the Gaia catalog obtained for the region of the target.
    Then the closest and brightest sources are selected and the WCS is shifted by the median of the distance between
    these stars and the detected sources.
    A new WCS is created and merged with the given exposure.

    Parameters
    ----------
    file_name: str
        Input file nam of the image to analyse.
    target: str
        The name of the targeted object.
    disperser_label: str
        The name of the disperser (default: "").

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
    ...     run_gaia_astrometry(file_name, target, disperser_label)
    ...     assert os.path.isdir('./tests/data/reduc_20170530_134_wcs')


    """
    my_logger = set_logger(__name__)
    # load image
    new_file_name = file_name.replace('.fits', '_new.fits')
    im = Image(new_file_name, target=target, disperser_label=disperser_label)
    # prepare outputs
    output_directory = set_wcs_output_directory(file_name)
    tag = set_file_tag(file_name)
    my_logger.info(f"\n\tIntermediate outputs will be stored in {output_directory}")

    # load detected sources
    hdu = fits.open(f"{os.path.join(output_directory, tag)}_sources.fits")
    sources = np.array([hdu[1].data[i][:2] for i in range(hdu[1].data.size)]).T

    # now refine the astrometry with the Gaia catalog
    wcs = load_wcs_from_file(new_file_name)
    gaia_catalog = load_gaia_catalog(im.target)
    target_coord_after_motion = update_target_coord_with_proper_motion(im.target, im.date_obs)
    gaia_coord_after_motion = update_gaia_catalog_with_proper_motion(gaia_catalog, im.date_obs)
    plot_sources_and_gaia_catalog(im, wcs, sources, target_coord_after_motion, gaia_coord_after_motion)


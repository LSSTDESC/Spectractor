import os
import sys
from copy import deepcopy
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, Distance

from scipy.spatial import ConvexHull

from spectractor import parameters
from spectractor.tools import (plot_image_simple, set_wcs_file_name, set_wcs_tag, set_wcs_output_directory,
                               set_sources_file_name, set_gaia_catalog_file_name, load_wcs_from_file, ensure_dir,
                               iraf_source_detection)
from spectractor.config import set_logger
from spectractor.extractor.images import Image
from spectractor.extractor.background import remove_image_background_sextractor


def _get_astrometry_executable_path(executable):
    """Find executable path from astrometry library.

    Parameters
    ----------
    executable: str
        The binary executable name.

    Returns
    -------
    path: str
        The absolute path of the executable.

    Examples
    --------
    >>> _get_astrometry_executable_path("solve-field")  # doctest: +ELLIPSIS
    '.../bin/solve-field'

    """
    my_logger = set_logger("get_astrometry_executable_path")
    if shutil.which(executable) is not None:
        path = shutil.which(executable)
    elif parameters.ASTROMETRYNET_DIR != "":
        if not os.path.isdir(parameters.ASTROMETRYNET_DIR):
            # reset astrometry.net path
            if 'ASTROMETRYNET_DIR' in os.environ:
                my_logger.warning(f"Reset parameters.ASTROMETRYNET_DIR={parameters.ASTROMETRYNET_DIR} (not found) "
                                  f"to {os.getenv('ASTROMETRYNET_DIR')}.")
                parameters.ASTROMETRYNET_DIR = os.getenv('ASTROMETRYNET_DIR') + '/'
            else:
                my_logger.error(f"parameters.ASTROMETRYNET_DIR={parameters.ASTROMETRYNET_DIR} but directory does "
                                f"not exist and ASTROMETRYNET_DIR is not in OS environment.")
                raise OSError(f"No {executable} binary found with parameters.ASTROMETRYNET_DIR "
                              f"or ASTROMETRYNET_DIR environment variable.")
        path = os.path.join(parameters.ASTROMETRYNET_DIR, f'bin/{executable}')
    else:
        raise OSError(f"{executable} executable not found in $PATH "
                      f"or {os.path.join(parameters.ASTROMETRYNET_DIR, f'bin/{executable}')}")
    return path


def load_gaia_catalog(coord, radius=5 * u.arcmin, gaia_mag_g_limit=23):
    """Load the Gaia catalog of stars around a given RA,DEC position within a given radius.

    Parameters
    ----------
    coord: SkyCoord
        Central coordinates for the Gaia cone search.
    radius: float
        Radius size for the cone search, with angle units (default: 5u.arcmin).
    gaia_mag_g_limit: float, optional
        Maximum g magnitude in the Gaia catalog output (default: 23).

    Returns
    -------
    gaia_catalog: Table
        The Gaia catalog.

    Examples
    --------

    >>> from astropy.coordinates import SkyCoord
    >>> c = SkyCoord(ra=0*u.deg, dec=0*u.deg)
    >>> gaia_catalog = load_gaia_catalog(c, radius=1*u.arcmin, gaia_mag_g_limit=17)  # doctest: +ELLIPSIS
    INFO: Query finished...
    >>> print(gaia_catalog)  # doctest: +SKIP
            dist        ...

    .. doctest:
        :hide:

        >>> assert len(gaia_catalog) > 0

    """
    from astroquery.gaia import Gaia
    my_logger = set_logger("load_gaia_catalog")
    Gaia.ROW_LIMIT = -1
    job = Gaia.cone_search_async(coord, radius=radius, verbose=False, columns=['ra', 'dec', 'pmra', 'pmdec', 'ref_epoch',
                                                                               'parallax', 'phot_g_mean_mag'])
    my_logger.debug(f"\n\t{job}")
    gaia_catalog = job.get_results()
    my_logger.debug(f"\n\t{gaia_catalog}")
    gaia_catalog = gaia_catalog[gaia_catalog["phot_g_mean_mag"]<gaia_mag_g_limit]
    gaia_catalog.fill_value = 0
    gaia_catalog['parallax'].fill_value = np.min(gaia_catalog['parallax'])
    return gaia_catalog


def get_gaia_coords_after_proper_motion(gaia_catalog, date_obs):
    """Use proper motion data to shift the Gaia catalog coordinates at J2000
    to current date of observation.

    Parameters
    ----------
    gaia_catalog: Table
        The Gaia catalog.
    date_obs: Time
        The time of observation.

    Returns
    -------
    gaia_coordinates_after_proper_motion: SkyCoord
        Array of RA,DEC coordinates of the Gaia stars at the observation date.

    Examples
    --------

    >>> from astropy.coordinates import SkyCoord
    >>> from astropy.time import Time
    >>> c = SkyCoord(ra=0*u.deg, dec=0*u.deg)
    >>> gaia_catalog = load_gaia_catalog(c, radius=1*u.arcmin, gaia_mag_g_limit=17)  # doctest: +ELLIPSIS
    INFO: Query finished...
    >>> t = Time("2017-01-01T00:00:00.000")
    >>> gaia_coords = get_gaia_coords_after_proper_motion(gaia_catalog, t)
    >>> print(gaia_coords)  # doctest: +ELLIPSIS
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)...

    .. doctest:
        :hide:

        >>> assert len(gaia_catalog) == len(gaia_coords)

    """
    parallax = np.array(gaia_catalog['parallax'].filled(np.nanmin(np.abs(gaia_catalog['parallax']))))
    parallax[parallax < 0] = np.min(parallax[parallax > 0])
    gaia_stars = SkyCoord(ra=gaia_catalog['ra'], dec=gaia_catalog['dec'], frame='icrs', equinox="J2000",
                          obstime=Time(np.array(gaia_catalog['ref_epoch']), format='decimalyear'),
                          pm_ra_cosdec=gaia_catalog['pmra'].filled(0) * np.cos(
                              np.array(gaia_catalog['dec']) * np.pi / 180),
                          pm_dec=gaia_catalog['pmdec'].filled(0),
                          distance=Distance(parallax=parallax * u.mas, allow_negative=True))
    gaia_coords_after_proper_motion = gaia_stars.apply_space_motion(new_obstime=Time(date_obs))
    return gaia_coords_after_proper_motion


def plot_shifts_histograms(dra, ddec):  # pragma: no cover
    dra_median = np.median(dra.to(u.arcsec).value)
    ddec_median = np.median(ddec.to(u.arcsec).value)
    dra_rms = np.std(dra.to(u.arcsec).value)
    ddec_rms = np.std(ddec.to(u.arcsec).value)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].hist(dra.to(u.arcsec).value, bins=50)
    ax[1].hist(ddec.to(u.arcsec).value, bins=50)
    ax[0].axvline(dra_median, color='k', label="median", lw=2)
    ax[1].axvline(ddec_median, color='k', label="median", lw=2)
    ax[0].set_xlabel('Shift in RA [arcsec]')
    ax[1].set_xlabel('Shift in DEC [arcsec]')
    ax[0].text(0.05, 0.96, f"Median: {dra_median:.3g}arcsec\nRMS: {dra_rms:.3g} arcsec",
               verticalalignment='top', horizontalalignment='left', transform=ax[0].transAxes)
    ax[1].text(0.05, 0.96, f"Median: {ddec_median:.3g}arcsec\nRMS: {ddec_rms:.3g} arcsec",
               verticalalignment='top', horizontalalignment='left', transform=ax[1].transAxes)
    ax[0].grid()
    ax[1].grid()
    # ax[0].legend()
    # ax[1].legend()
    if parameters.DISPLAY:
        plt.show()
    if parameters.PdfPages:
        parameters.PdfPages.savefig()


def wcs_xy_translation(wcs, shift_x, shift_y):  # pragma: no cover
    """Compute a translated WCS if image is shifted in x or y."""
    new_wcs = deepcopy(wcs)
    new_wcs.wcs.crpix[0] += shift_x
    new_wcs.wcs.crpix[1] += shift_y
    if wcs.has_distortion:
        new_wcs.sip.crpix[0] += shift_x
        new_wcs.sip.crpix[1] += shift_y
    return new_wcs


def wcs_flip_x(wcs, image):  # pragma: no cover
    """Compute a flip WCS if image is flip along x axis."""
    new_wcs = deepcopy(wcs)
    new_wcs.wcs.crpix[0] = image.data.shape[1] - new_wcs.wcs.crpix[0]
    new_wcs.wcs.cd = new_wcs.wcs.cd @ np.array([[-1, 0], [0, 1]])
    if new_wcs.has_distortion:
        new_wcs.sip.crpix[0] = image.data.shape[1] - new_wcs.wcs.crpix[0]
        for k in range(1, new_wcs.sip.a_order+1, 2):
            new_wcs.sip.a[k,:] *= -1
            new_wcs.sip.ap[k,:] *= -1
        for k in range(1, new_wcs.sip.b_order+1, 2):
            new_wcs.sip.b[k,:] *= -1
            new_wcs.sip.bp[k,:] *= -1
    return new_wcs


def wcs_flip_y(wcs, image):  # pragma: no cover
    """Compute a flip WCS if image is flip along y axis."""
    new_wcs = deepcopy(wcs)
    new_wcs.wcs.crpix[1] = image.data.shape[0] - new_wcs.wcs.crpix[1]
    new_wcs.wcs.cd = new_wcs.wcs.cd @ np.array([[1, 0], [0, -1]])
    if new_wcs.has_distortion:
        new_wcs.sip.crpix[1] = image.data.shape[0] - new_wcs.wcs.crpix[1]
        for k in range(1, new_wcs.sip.a_order+1, 2):
            new_wcs.sip.a[:,k] *= -1
            new_wcs.sip.ap[:,k] *= -1
        for k in range(1, new_wcs.sip.b_order+1, 2):
            new_wcs.sip.b[:,k] *= -1
            new_wcs.sip.bp[:,k] *= -1
    return new_wcs


def wcs_transpose(wcs, image):  # pragma: no cover
    """Compute a transposed WCS if image is transposed with np.transpose()."""
    new_wcs = wcs_flip_y(wcs, image)
    tmp_crpix = np.copy(wcs.wcs.crpix)
    new_wcs.wcs.crpix[1] = tmp_crpix[0]
    new_wcs.wcs.crpix[0] = tmp_crpix[1]
    new_wcs.wcs.cd = new_wcs.wcs.cd @ np.array([[0, 1], [-1, 0]])
    if new_wcs.has_distortion:
        # sip attributes are not writable, must fo the loop
        new_wcs.sip.crpix[0] = tmp_crpix[1]
        new_wcs.sip.crpix[1] = tmp_crpix[0]
        tmp_sip_a = np.copy(new_wcs.sip.a)
        tmp_sip_b = np.copy(new_wcs.sip.b)
        tmp_sip_ap = np.copy(new_wcs.sip.ap)
        tmp_sip_bp = np.copy(new_wcs.sip.bp)
        for i in range(new_wcs.sip.a_order+1):
            for j in range(new_wcs.sip.a_order+1):
                new_wcs.sip.a[i, j] = tmp_sip_a[j, i]
                new_wcs.sip.ap[i, j] = tmp_sip_ap[j, i]
        for i in range(new_wcs.sip.b_order+1):
            for j in range(new_wcs.sip.b_order+1):
                new_wcs.sip.b[i, j] = tmp_sip_b[j, i]
                new_wcs.sip.bp[i, j] = tmp_sip_bp[j, i]
    return new_wcs


class Astrometry():  # pragma: no cover

    def __init__(self, image, wcs_file_name="", gaia_file_name="", output_directory="",
                 gaia_mag_g_limit=23, source_extractor="iraf"):
        """Class to handle astrometric computations.

        Parameters
        ----------
        image: Image
            Input Spectractor Image.
        wcs_file_name: str, optional
            The path to a WCS fits file. WCS content will be loaded (default: "").
        gaia_file_name: str, optional
            The path to a Gaia catalog ecsv file (default: "").
        output_directory: str, optional
            The output directory path. If empty, a directory *_wcs is created next to the analyzed image (default: "").
        gaia_mag_g_limit: float, optional
            Maximum g magnitude in the Gaia catalog output (default: 23).
        source_extractor: str, optional
            Source extraction algorithm to be used for astrometry solving. Can be either:
            - iraf: uses the tools.py iraf_source_detection function which wraps the photutils IRAFStarFinder module
            - astrometrynet: uses the default astrometry.net source extraction library

        Examples
        --------
        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.wcs  # doctest: +ELLIPSIS
        WCS Keywords...
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.image = image
        self.gaia_mag_g_limit = gaia_mag_g_limit
        if source_extractor not in ["iraf", "astrometrynet"]:
            raise ValueError(f"source_extractor argument in Astrometry class must be either 'iraf' or 'astrometrynet'. "
                             f"Got {source_extractor=}")
        self.source_extractor = source_extractor
        # Use fast mode
        if parameters.CCD_REBIN > 1:
            self.image.rebin()
            if parameters.DEBUG:
                self.image.plot_image(scale='symlog', target_pixcoords=self.image.target_guess)

        self.output_directory = set_wcs_output_directory(self.image.file_name, output_directory=output_directory)
        ensure_dir(self.output_directory)
        self.tag = set_wcs_tag(self.image.file_name)
        self.new_file_name = self.image.file_name.replace('.fits', '_new.fits')
        self.sources_file_name = set_sources_file_name(self.image.file_name, output_directory=output_directory)
        self.wcs_file_name = wcs_file_name
        self.match_file_name = os.path.join(self.output_directory, self.tag) + ".match"
        self.wcs = None
        if self.wcs_file_name != "":
            if os.path.isfile(self.wcs_file_name):
                self.wcs = load_wcs_from_file(self.wcs_file_name)
            else:
                self.my_logger.warning(f"WCS file {wcs_file_name} does not exist. Skip it.")
        else:
            self.wcs_file_name = set_wcs_file_name(self.image.file_name, output_directory=output_directory)
            if os.path.isfile(self.wcs_file_name):
                self.wcs = load_wcs_from_file(self.wcs_file_name)
        if gaia_file_name != "":
            self.gaia_file_name = gaia_file_name
        else:
            self.gaia_file_name = set_gaia_catalog_file_name(self.image.file_name, output_directory=output_directory)
        self.gaia_catalog = None
        self.gaia_index = None
        self.gaia_matches = None
        self.gaia_residuals = None
        self.gaia_radec_positions_after_pm = None
        if os.path.isfile(self.gaia_file_name):
            self.my_logger.info(f"\n\tLoad Gaia catalog from {self.gaia_file_name}.")
            self.gaia_catalog = ascii.read(self.gaia_file_name, format="ecsv")
            self.gaia_radec_positions_after_pm = get_gaia_coords_after_proper_motion(self.gaia_catalog, self.image.date_obs)
        self.sources = None
        self.sources_radec_positions = None
        if os.path.isfile(self.sources_file_name):
            self.sources = self.load_sources_from_file()
        self.my_logger.info(f"\n\tIntermediate outputs will be stored in {self.output_directory}")
        self.dist_2d = None
        self.quad_stars_pixel_positions = None
        self.dist_ra = 0 * u.arcsec
        self.dist_dec = 0 * u.arcsec
        self.image.target_radec_position_after_pm = self.image.target.get_radec_position_after_pm(date_obs=self.image.date_obs)
        if os.path.isfile(self.match_file_name):
            self.quad_stars_pixel_positions = self.get_quad_stars_pixel_positions()

    def load_gaia_catalog_around_target(self):
        """Load the Gaia stars catalog around the target position.

        The radius of the search is set accordingly to the maximum range of detected sources.

        Returns
        -------
        gaia_catalog: Table
            The table of Gaia stars around the target position.

        Examples
        --------
        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.load_gaia_catalog_around_target()  #doctest: +ELLIPSIS
        INFO: Query finished...
        <Table length=...>...
        """
        radius = 0.5 * np.sqrt(2) * max(np.max(self.sources["xcentroid"]) - np.min(self.sources["xcentroid"]),
                                        np.max(self.sources["ycentroid"]) - np.min(self.sources["ycentroid"]))
        radius *= parameters.CCD_PIXEL2ARCSEC * u.arcsec
        self.my_logger.info(f"\n\tLoading Gaia catalog within radius < {radius.value} "
                            f"arcsec from {self.image.target.label} {self.image.target.radec_position}...")
        self.gaia_catalog = load_gaia_catalog(self.image.target.radec_position, radius=radius, gaia_mag_g_limit=self.gaia_mag_g_limit)
        ascii.write(self.gaia_catalog, self.gaia_file_name, format='ecsv', overwrite=True)
        return self.gaia_catalog

    def load_sources_from_file(self):
        """Load the sources from the class associated self.sources_file_name file.

        By default, the creation of an Astrometry class instance try to load the default
        source file if it exists.

        Returns
        -------
        sources: Table
            The table of detected sources.

        Examples
        --------
        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.load_sources_from_file()  # doctest: +ELLIPSIS
        <Table length=...

        """
        self.my_logger.info(f"\n\tLoad source positions and flux from {self.sources_file_name}")
        sources = Table.read(self.sources_file_name)
        sources['X'].name = "xcentroid"
        sources['Y'].name = "ycentroid"
        sources['FLUX'].name = "flux"
        sources.sort("flux", reverse=True)
        self.sources = sources
        return sources

    def get_target_pixel_position(self):
        """Gives the principal targetted object position in pixels in the image given the WCS.
        The object proper motion is taken into account.

        Returns
        -------
        target_x: float
            The target position along the x axis.
        target_y: float
            The target position along the y axis.

        Examples
        --------

        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> target_x, target_y = a.get_target_pixel_position()
        >>> print(target_x, target_y) # doctest: +ELLIPSIS
        745... 684...

        """
        target_x, target_y = self.wcs.all_world2pix(self.image.target.radec_position_after_pm.ra,
                                                    self.image.target.radec_position_after_pm.dec, 0)
        return target_x, target_y

    def get_gaia_pixel_positions(self, gaia_index=None):
        """Gives the Gaia star positions in pixels in the image given the WCS.
        The star proper motions are taken into account.

        Parameters
        ----------
        gaia_index: array_like, optional
            List of indices from the Gaia catalog for which to compute the pixel positions (default: None).

        Returns
        -------
        gaia_x: array_like
            The Gaia star positions along the x axis.
        gaia_y: array_like
            The Gaia star positions along the y axis.

        Examples
        --------

        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> gaia_x, gaia_y = a.get_gaia_pixel_positions()

        or with selected Gaia index:

        >>> gaia_x, gaia_y = a.get_gaia_pixel_positions(gaia_index=[1, 2])

        .. doctest:
            :hide:

            >>> assert np.isclose(gaia_x[0], 745, atol=1)
            >>> assert np.isclose(gaia_y[0], 684, atol=1)

        """
        if gaia_index is None:
            gaia_x, gaia_y = self.wcs.all_world2pix(self.gaia_radec_positions_after_pm.ra,
                                                    self.gaia_radec_positions_after_pm.dec, 0, quiet=True)
        else:
            gaia_x, gaia_y = self.wcs.all_world2pix(self.gaia_radec_positions_after_pm[gaia_index].ra,
                                                    self.gaia_radec_positions_after_pm[gaia_index].dec, 0, quiet=True)
        return gaia_x, gaia_y

    def get_quad_stars_pixel_positions(self):
        """Gives the star positions in pixels in the image that were used as a quad by the astrometry.net
        to compute the WCS. The positions are recovered from the *.log file generated by the solve-field command.
        At least 4 stars should be present.

        Returns
        -------
        quad_star_positions: array_like
            The array of quad star (x, y) positions.

        Examples
        --------

        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> quad_stars = a.get_quad_stars_pixel_positions()

        .. doctest:
            :hide:

            >>> assert quad_stars.shape == (4, 2)

        """
        u.set_enabled_aliases({'DEG': u.deg})
        coords = []
        with fits.open(self.match_file_name) as hdu:
            table = Table.read(hdu)
        for k in range(0, len(table["QUADPIX"][0]), 2):
            coord = [float(table["QUADPIX"][0][k]), float(table["QUADPIX"][0][k+1])]
            if np.sum(coord) > 0:
                coords.append(coord)
        if len(coords) < 4:
            self.my_logger.warning(f"\n\tOnly {len(coords)} calibration stars has been extracted from "
                                   f"{self.match_file_name}, with positions {coords}. "
                                   f"A quad of at least 4 stars is expected. "
                                   f"Please check {self.match_file_name}.")
        self.quad_stars_pixel_positions = np.array(coords)
        return self.quad_stars_pixel_positions

    def write_sources(self):
        """Write a fits file containing the source positions and fluxes.
        The name of the file is set accordingly to the name of the WCS file.

        """
        colx = fits.Column(name='X', format='D', array=self.sources['xcentroid'])
        coly = fits.Column(name='Y', format='D', array=self.sources['ycentroid'])
        colflux = fits.Column(name='FLUX', format='D', array=self.sources['flux'])
        coldefs = fits.ColDefs([colx, coly, colflux])
        hdu = fits.BinTableHDU.from_columns(coldefs)
        hdu.header['IMAGEW'] = self.image.data.shape[1]
        hdu.header['IMAGEH'] = self.image.data.shape[0]
        hdu.writeto(self.sources_file_name, overwrite=True)
        self.my_logger.info(f'\n\tSources positions saved in {self.sources_file_name}')

    def plot_sources_and_gaia_catalog(self, ax=None, sources=None, gaia_coord=None, quad=None, label="",
                                      vmax=None, margin=parameters.CCD_IMSIZE, center=None, scale="log10", swapaxes=False):
        """Plot the data image with different overlays: detected sources, Gaia stars, quad stars.

        Parameters
        ----------
        ax: Axes, optional
            Axes instance. If not given, open a new figure. (default: None).
        sources: Table, optional
            Sources. If not given use the current class instance (default: None).
        gaia_coord: SkyCoord, optional
            List of Gaia stars RA,DEC coordinates. If not given, does nothing (default: None).
        quad: array_like, optional
            List of quad stars (x,y) coordinates in pixel to plot the quad that was used by astrometry.net to
            compute the WCS solution.  If not given, does nothing (default: None).
        center: array_like, optional
            (x,y) position of the center of the image.
            If not given, center the image on the main target (default: None).
        label: str, optional
            Name of the center if particular (default: '').
        margin: int, optional
            Size of the image to plot in pixels from the center (default: parameters.CCD_IMSIZE).
        scale: str, optional
            Scaling of the image (choose between: lin, log or log10) (default: log10).
        vmax: float, optional
            Maximum z-scale value (default: None).

        Examples
        --------

        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.plot_sources_and_gaia_catalog(sources=a.sources, gaia_coord=a.gaia_radec_positions_after_pm,
        ...                                 quad=a.quad_stars_pixel_positions,
        ...                                 label=a.image.target.label)

        .. plot:
            :hide:

            from spectractor.astrometry import Astrometry

            a = Astrometry("./tests/data/reduc_20170530_134.fits", target="HD111980",
                            wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
            a.plot_sources_and_gaia_catalog(sources=a.sources, gaia_coord=a.gaia_radec_positions_after_pm,
                                             quad=np.array(a.quad_stars_pixel_positions).T,
                                             label=a.target.label)

        """
        no_plot = False
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            if self.wcs is not None:
                if swapaxes:
                    fig.add_subplot(111, projection=self.wcs.swapaxes(0, 1))
                else:
                    fig.add_subplot(111, projection=self.wcs)
            ax = plt.gca()
        else:
            no_plot = True

        plot_image_simple(ax, self.image.data, scale=scale, vmax=vmax)
        if self.wcs is not None and not no_plot:
            if swapaxes:
                ax.set_xlabel('Dec')
                ax.set_ylabel('RA')
            else:
                ax.set_xlabel('RA')
                ax.set_ylabel('Dec')
        if sources is not None:
            ax.scatter(sources['xcentroid'], sources['ycentroid'], s=300, lw=2,
                       edgecolor='black', facecolor='none', label="Detected sources")
        if gaia_coord is not None:
            gaia_x, gaia_y = self.wcs.all_world2pix(self.gaia_radec_positions_after_pm.ra,
                                                    self.gaia_radec_positions_after_pm.dec, 0, quiet=True)
            ax.scatter(gaia_x, gaia_y, s=300, marker="+", facecolor='blue', label=f"Gaia stars", lw=2)
        if center is None:
            target_x, target_y = self.get_target_pixel_position()
        else:
            target_x, target_y = center
        ax.scatter(target_x, target_y, s=300, marker="x", facecolor='cyan', label=f"{label}", lw=2)
        if quad is not None:
            if len(quad) > 3:
                points = np.concatenate([quad, [quad[-1]]])
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 'r-')
                ax.plot(points.T[0], points.T[1], 'rx', lw=2, label="Quad stars")
            else:
                self.my_logger.warning(f"\n\tNumber of quad stars is {len(quad)}: the quad can't be plotted. Skip it.")
        ax.legend()
        ax.set_xlim(max(0, target_x - margin), min(target_x + margin, self.image.data.shape[1]))
        ax.set_ylim(max(0, target_y - margin), min(target_y + margin, self.image.data.shape[0]))
        if not no_plot and parameters.DISPLAY:
            # fig.tight_layout()
            plt.show()
        else:
            plt.close("all")

    def get_sources_radec_positions(self):
        """Gives the RA,DEC position of the detected sources.

        Returns
        -------
        coords: SkyCoord
            The RA,DEC positions

        Examples
        --------
        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.get_sources_radec_positions()  # doctest: +ELLIPSIS
        <SkyCoord (ICRS): (ra, dec) in deg...

        """
        sources_coord = self.wcs.all_pix2world(self.sources['xcentroid'], self.sources['ycentroid'], 0)
        self.sources_radec_positions = SkyCoord(ra=sources_coord[0] * u.deg, dec=sources_coord[1] * u.deg,
                                                frame="icrs", obstime=self.image.date_obs, equinox="J2000")
        return self.sources_radec_positions

    def match_sources_to_gaia_catalog(self, gaia_coord=None):
        """Make the matching between the detected sources and the Gaia star catalog.

        Parameters
        ----------
        gaia_coord: SkyCoord, optional
            The Gaia stars RA,DEC positions. If None, the self.gaia_radec_positions_after_pm instance is used.

        Returns
        -------
        gaia_index: array_like
            The index in the Gaia catalog that matches best the sources.
        dist_2d: Angle
            The array of angular distances between the sources and the Gaia matches.
        dist_ra: Angle
            The array of distances in RA between the sources and the Gaia matches.
        dist_ra: Angle
            The array of distances in DEC between the sources and the Gaia matches.

        Examples
        --------
        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.get_sources_radec_positions()  # doctest: +ELLIPSIS
        <SkyCoord (ICRS): (ra, dec) in deg...
        >>> gaia_index, dist_2d, dist_ra, dist_dec = a.match_sources_to_gaia_catalog(
        ...                                                         gaia_coord=a.gaia_radec_positions_after_pm)
        >>> print(dist_2d[0: 3])  # doctest: +ELLIPSIS
        [0d00... 0d00m00... 0d00m00...]

        """
        if gaia_coord is None:
            gaia_coord = self.gaia_radec_positions_after_pm
        self.gaia_index, self.dist_2d, dist_3d = self.sources_radec_positions.match_to_catalog_sky(gaia_coord)
        matches = gaia_coord[self.gaia_index]
        self.dist_ra, self.dist_dec = self.sources_radec_positions.spherical_offsets_to(matches)
        return self.gaia_index, self.dist_2d, self.dist_ra, self.dist_dec

    def plot_astrometry_shifts(self, vmax=3, margin=parameters.CCD_IMSIZE):
        """Plot the RA,DEC distances between the detected sources and the Gaia star that match the sources.

        Parameters
        ----------
        vmax: float, optional
            Maximum z-scale value in arcsec (default: 3).
        margin: int, optional
            Size of the image to plot in pixels from the center (default: parameters.CCD_IMSIZE).

        Examples
        --------
        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.get_sources_radec_positions()  # doctest: +ELLIPSIS
        <SkyCoord (ICRS): (ra, dec) in deg...
        >>> a.gaia_index, a.dist_2d, a.dist_ra, a.dist_dec = a.match_sources_to_gaia_catalog(
        ...                                                         gaia_coord=a.gaia_radec_positions_after_pm)
        >>> a.plot_astrometry_shifts()

        .. plot:
            :hide:

            from spectractor.astrometry import Astrometry

            a = Astrometry("./tests/data/reduc_20170530_134.fits", target="HD111980",
                            wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
            a.load_gaia_catalog_around_target() # doctest: +ELLIPSIS
            a.load_sources_from_file() # doctest: +ELLIPSIS
            a.get_sources_radec_positions()  # doctest: +ELLIPSIS
            a.gaia_index, a.dist_2d, a.dist_ra, a.dist_dec = a.match_sources_to_gaia_catalog(
                                                                     gaia_coord=a.gaia_radec_positions_after_pm)
            a.plot_astrometry_shifts()
        """
        target_x, target_y = self.get_target_pixel_position()
        gaia_x, gaia_y = self.get_gaia_pixel_positions(gaia_index=self.gaia_index)

        fig = plt.figure(figsize=(6, 8))

        fig.add_subplot(211, projection=self.wcs)

        plot_image_simple(plt.gca(), self.image.data, scale="log10")
        plt.xlabel('RA')
        plt.ylabel('Dec')
        if self.sources is not None:
            plt.scatter(self.sources['xcentroid'], self.sources['ycentroid'], s=100, lw=2,
                        edgecolor='black', facecolor='none', label="Detected sources")
        vmax_2 = min(vmax, np.max(np.abs(self.dist_ra.to(u.arcsec).value)))
        sc = plt.scatter(gaia_x, gaia_y, s=100, c=self.dist_ra.to(u.arcsec).value,
                         cmap="bwr", vmin=-vmax_2, vmax=vmax_2,
                         label=f"Gaia stars", lw=1)
        plt.xlim(max(0, int(target_x - margin)), min(int(target_x + margin), self.image.data.shape[1]))
        plt.ylim(max(0, int(target_y - margin)), min(int(target_y + margin), self.image.data.shape[0]))
        plt.colorbar(sc, label="Shift in RA [arcsec]")
        plt.legend()

        fig.add_subplot(212, projection=self.wcs)
        plot_image_simple(plt.gca(), self.image.data, scale="log10")
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.grid(color='white', ls='solid')
        if self.sources is not None:
            plt.scatter(self.sources['xcentroid'], self.sources['ycentroid'], s=100, lw=2,
                        edgecolor='black', facecolor='none', label="all sources")
        vmax_2 = min(vmax, np.max(np.abs(self.dist_dec.to(u.arcsec).value)))
        sc = plt.scatter(gaia_x, gaia_y, s=100, c=self.dist_dec.to(u.arcsec).value,
                         cmap="bwr", vmin=-vmax_2, vmax=vmax_2,
                         label=f"Gaia Stars", lw=1)
        plt.xlim(max(0, int(target_x - margin)), min(int(target_x + margin), self.image.data.shape[1]))
        plt.ylim(max(0, int(target_y - margin)), min(int(target_y + margin), self.image.data.shape[0]))
        plt.colorbar(sc, label="Shift in DEC [arcsec]")
        plt.legend()
        if parameters.DISPLAY:
            # fig.tight_layout()
            plt.show()
        else:
            plt.close("all")

    def set_constraints(self, min_stars=100, flux_log10_threshold=0.1, min_range=3 * u.arcsec, max_range=5 * u.arcmin,
                        max_sep=1 * u.arcsec):
        """Gives a boolean array for sources that respect certain criterai (see below).

        Parameters
        ----------
        min_stars: int
            Minimum number of stars that have to be kept in the selection (default: 100).
        flux_log10_threshold:
            Lower cut on the log10 of the star fluxes (default: 0.1).
        min_range: astropy.Quantity
            Minimum distance for sources from image principal target in arcsec (default: 3*u.arcsec).
        max_range: astropy.Quantity
            Maximum distance for sources from image principal target in arcsec (default: 5*u.arcsec).
        max_sep: astropy.Quantity
            Maximum separation between the detected sources and the Gaia stars in arcsec (default: 1*u.arcsec).

        Returns
        -------
        indices: array_like
            Boolean array of selected sources.

        """
        self.my_logger.info(f"Initial number of sources: {len(self.dist_2d)}")
        sep = self.dist_2d < max_sep
        self.my_logger.info(f"Number of sources after {max_sep=} constraint: {np.sum(sep)=}")
        sep *= self.sources_radec_positions.separation(self.image.target.radec_position_after_pm) < max_range
        self.my_logger.info(f"Number of sources after {max_range=} constraint: {np.sum(sep)=}")
        sep *= self.sources_radec_positions.separation(self.image.target.radec_position_after_pm) > min_range
        self.my_logger.info(f"Number of sources after {min_range=} constraint: {np.sum(sep)=}")
        sep *= np.log10(self.sources['flux']) > flux_log10_threshold
        self.my_logger.info(f"Number of sources after flux constraint: {np.sum(sep)=}")
        if np.sum(sep) > min_stars:
            for r in np.arange(0, max_range.value, 0.1)[::-1]:
                range_constraint = self.sources_radec_positions.separation(self.image.target.radec_position_after_pm) \
                                   < r * u.arcmin
                if np.sum(sep * range_constraint) < min_stars:
                    break
                else:
                    sep *= range_constraint
        self.my_logger.info(f"Final number of sources: {np.sum(sep)=}")
        return sep

    def plot_shifts_profiles(self, matches, dra, ddec):
        """Plot the distances between Gaia stars and detected sources with respect to RA and DEC.
        The median of the distances is overplotted.

        Parameters
        ----------
        matches: SkyCoord
            Array of sky coordinates of the detected sources.
        dra: Angle
            Array of distances between sources and Gaia stars along the RA direction.
        ddec
            Array of distances between sources and Gaia stars along the DEC direction.

        Examples
        --------

        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.get_sources_radec_positions()  # doctest: +ELLIPSIS
        <SkyCoord (ICRS): (ra, dec) in deg...
        >>> a.gaia_index, a.dist_2d, a.dist_ra, a.dist_dec = a.match_sources_to_gaia_catalog(
        ...                                                         gaia_coord=a.gaia_radec_positions_after_pm)
        >>> sep_constraints = a.set_constraints(flux_log10_threshold=0.1)
        >>> sources_selection = a.sources_radec_positions[sep_constraints]
        >>> gaia_matches = a.gaia_radec_positions_after_pm[a.gaia_index[sep_constraints]]
        >>> dra, ddec = sources_selection.spherical_offsets_to(gaia_matches)
        >>> a.plot_shifts_profiles(gaia_matches, dra, ddec)

        .. plot:
            :hide:

            from spectractor.astrometry import Astrometry

            a = Astrometry("./tests/data/reduc_20170530_134.fits", target="HD111980",
            wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
            a.load_gaia_catalog_around_target()
            a.load_sources_from_file()
            a.get_sources_radec_positions()
            a.gaia_index, a.dist_2d, a.dist_ra, a.dist_dec = a.match_sources_to_gaia_catalog(
                                                                    gaia_coord=a.gaia_radec_positions_after_pm)
            sep_constraints = a.set_constraints(flux_log10_threshold=0.1)
            sources_selection = a.sources_radec_positions[sep_constraints]
            gaia_matches = a.gaia_radec_positions_after_pm[a.gaia_index[sep_constraints]]
            dra, ddec = sources_selection.spherical_offsets_to(gaia_matches)
            a.plot_shifts_profiles(gaia_matches, dra, ddec)


        """
        dra_median = np.median(dra.to(u.arcsec).value)
        ddec_median = np.median(ddec.to(u.arcsec).value)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        c_ra = ax[0].scatter(matches.ra, dra.to(u.arcsec).value, label="Shift in RA [arcsec]")
        c_dec = ax[0].scatter(matches.ra, ddec.to(u.arcsec).value, label="Shift in DEC [arcsec]")
        ax[1].scatter(matches.dec, dra.to(u.arcsec).value, label="Shift in RA [arcsec]")
        ax[1].scatter(matches.dec, ddec.to(u.arcsec).value, label="Shift in DEC [arcsec]")
        ax[0].axhline(dra_median, color=c_ra.get_facecolor()[0], label="median", lw=2)
        ax[0].axhline(ddec_median, color=c_dec.get_facecolor()[0], label="median", lw=2)
        ax[1].axhline(dra_median, color=c_ra.get_facecolor()[0], label="median", lw=2)
        ax[1].axhline(ddec_median, color=c_dec.get_facecolor()[0], label="median", lw=2)
        ax[0].axvline(self.image.target.radec_position_after_pm.ra.value, color='k', linestyle="--",
                      label=f"{self.image.target.label} RA")
        ax[1].axvline(self.image.target.radec_position_after_pm.dec.value, color='k', linestyle="--",
                      label=f"{self.image.target.label} DEC")
        ax[0].set_xlabel('Gaia RA [deg]')
        ax[1].set_xlabel('Gaia DEC [deg]')
        ax[0].set_ylabel('Astrometric shifts [arcsec]')
        ax[1].set_ylabel('Astrometric shifts [arcsec]')
        ax[0].grid()
        ax[1].grid()
        ax[0].set_ylim(-2, 2)
        ax[1].set_ylim(-2, 2)
        ax[0].legend()
        ax[1].legend()
        if parameters.DISPLAY:
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()

    def merge_wcs_with_new_exposure(self, log_file=None):
        """Merge the WCS solution with the current FITS file image.

        Parameters
        ----------
        log_file: file, optional
            Log file to write the output of the merge command.

        """
        exec = _get_astrometry_executable_path('new-wcs')
        command = f"{exec} -v -d -i {self.image.file_name} -w {self.wcs_file_name} -o {self.new_file_name}\n"
        self.my_logger.info(f'\n\tSave WCS in original file:\n\t{command}')
        log = subprocess.check_output(command, shell=True)
        if log_file is not None:
            log_file.write(command + "\n")
            log_file.write(log.decode("utf-8") + "\n")

    def compute_gaia_pixel_residuals(self):
        """Compute the residuals in pixels between the detected sources and the associated Gaia stars.
        A selection in proximity, range and flux is performed to keep the best matches.

        Returns
        -------
        gaia_residuals: array_like
            2D array of the pixel residuals.

        Examples
        --------

        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.load_gaia_catalog_around_target()  #doctest: +ELLIPSIS
        INFO: Query finished...
        <Table length=...>...
        >>> residuals = a.compute_gaia_pixel_residuals()

        .. doctest:
            :hide:

            >>> assert residuals.shape == (4, 2)

        """
        coords = self.quad_stars_pixel_positions.T
        ra, dec = self.wcs.all_pix2world(coords[0], coords[1], 0)
        coords_radec = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs', equinox="J2000")
        gaia_index, dist_2d, dist_3d = coords_radec.match_to_catalog_sky(self.gaia_radec_positions_after_pm)
        matches = self.gaia_radec_positions_after_pm[gaia_index]
        gaia_coords = np.array(self.wcs.all_world2pix(matches.ra, matches.dec, 0))
        self.gaia_residuals = (gaia_coords - coords).T
        return self.gaia_residuals

    def find_quad_star_index_in_sources(self, quad_star):
        """Associate a quad star with a detected source.

        Parameters
        ----------
        quad_star: array_like
            (x, y) positions of a quad star

        Returns
        -------
        index: int
            The index of the source that match the input quad star.

        Examples
        --------

        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.load_sources_from_file() # doctest: +ELLIPSIS
        <Table length=...
        >>> index = a.find_quad_star_index_in_sources(a.quad_stars_pixel_positions[0])

        """
        eps = 1e-1
        k = -1
        for k in range(len(self.sources)):
            if abs(self.sources['xcentroid'][k] - quad_star[0]) < eps \
                    and abs(self.sources['ycentroid'][k] - quad_star[1]) < eps:
                break
        return k

    def remove_worst_quad_star_from_sources(self):
        """Remove the quad star from the source table with the largest distance between the source centroid
        and the associated Gaia star position.

        Examples
        --------

        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.load_sources_from_file() # doctest: +ELLIPSIS
        <Table length=...
        >>> length = len(a.sources)
        >>> a.remove_worst_quad_star_from_sources()
        >>> assert len(a.sources) == length - 1

        """
        gaia_pixel_residuals = self.compute_gaia_pixel_residuals()
        distances = np.sqrt(np.sum(gaia_pixel_residuals ** 2, axis=1))
        max_residual_index = int(np.argmax(distances))
        worst_source_index = \
            self.find_quad_star_index_in_sources(self.quad_stars_pixel_positions[max_residual_index])
        self.my_logger.debug(f"\n\tRemove source #{worst_source_index}\n{self.sources[max_residual_index]}")
        self.sources.remove_row(worst_source_index)

    def plot_quad_stars(self, margin=10, scale="log10"):
        """Make a set of plots centered on the quad stars to check their astrometry.

        Parameters
        ----------
        margin: int, optional
            Size of the image to plot in pixels from the center (default: parameters.CCD_IMSIZE).
        scale: str, optional
            Scaling of the image (choose between: lin, log or log10) (default: log10).

        Examples
        --------

        >>> im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
        >>> a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.load_sources_from_file() # doctest: +ELLIPSIS
        <Table length=...
        >>> a.plot_quad_stars()

        """
        nquads = self.quad_stars_pixel_positions.shape[0]
        fig, ax = plt.subplots(1, nquads, figsize=(4 * nquads, 3))
        for k in range(nquads):
            row = self.sources[self.find_quad_star_index_in_sources((self.quad_stars_pixel_positions[k][0],
                                                                     self.quad_stars_pixel_positions[k][1]))]
            self.plot_sources_and_gaia_catalog(ax=ax[k], center=(self.quad_stars_pixel_positions[k][0],
                                                                 self.quad_stars_pixel_positions[k][1]),
                                               margin=margin, scale=scale, sources=self.sources,
                                               gaia_coord=self.gaia_matches, label=f"Quad star #{k}",
                                               quad=self.quad_stars_pixel_positions)
            ax[k].set_title(
                f"x={self.quad_stars_pixel_positions[k][0]:.2f}, y={self.quad_stars_pixel_positions[k][1]:.2f}, "
                f"flux={row['flux']:.2f}")
        if parameters.DISPLAY:
            fig.tight_layout()
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()

    def run_simple_astrometry(self, extent=None, sources=None):
        """Build a World Coordinate System (WCS) using astrometry.net library given an exposure as a FITS file.

        The name of the target must be given to get its RA,DEC coordinates via a Simbad query.
        If 'iraf' source_extractor is chosen, first the background of the exposure is removed using the astropy
        SExtractorBackground() method, then photutils iraf_source_detection() is used to get the positions in pixels
        and fluxes of the objects in the field. If 'astrometrynet' is chosen, astrometry.net extractor is used.
        The results are saved in the {file_name}.axy file and used by the solve_field command from the
        astrometry.net library. The solve_field path must be set using the spectractor.parameters.ASTROMETRYNET_BINDIR
        variable. A new WCS is created and saved as a new FITS file. The WCS file and the intermediate results
        are saved in a new directory named as the FITS file name with a _wcs suffix.

        Parameters
        ----------
        extent: 2-tuple
            ((xmin,xmax),(ymin,ymax)) 2 dimensional tuple to crop the exposure before any operation (default: None).
        sources: Table
            List of sources. If None, then source detection is run on the image (default: None).

        Notes
        -----
        The source file given to solve-field is understood as a FITS file with pixel origin value at 1,
        whereas pixel coordinates comes from photutils using a numpy convention with pixel origin value at 0.
        To correct for this we shift the CRPIX center of 1 pixel at the end of the function. It can be that solve-field
        using the source list or the raw FITS image then give the same WCS values.

        See Also
        --------

        iraf_source_detection()

        Examples
        --------

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> from spectractor.astrometry import Astrometry
        >>> from spectractor import parameters
        >>> parameters.VERBOSE = True
        >>> logbook = LogBook(logbook='./tests/data/ctiofulllogbook_jun2017_v5.csv')
        >>> file_name = './tests/data/reduc_20170530_134.fits'
        >>> if os.path.isfile('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs'):
        ...     os.remove('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs')
        >>> tag = file_name.split('/')[-1]
        >>> disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        >>> im = Image(file_name, target_label=target_label, disperser_label=disperser_label, config="ctio.ini")
        >>> a = Astrometry(im, source_extractor="astrometrynet")
        >>> a.run_simple_astrometry(extent=((300,1400),(300,1400)))  # doctest: +ELLIPSIS
        WCS ...

        .. doctest:
            :hide:

            >>> assert os.path.isdir('./tests/data/reduc_20170530_134_wcs')
            >>> assert os.path.isfile('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs')
            >>> assert a.sources is not None

        """
        tmp_image_file_name = self.wcs_file_name.replace(".wcs", "_tmp.fits")
        if sources is None:
            if self.source_extractor == "iraf":
                if extent is not None:
                    data = self.image.data[extent[1][0]:extent[1][1], extent[0][0]:extent[0][1]]
                else:
                    data = np.copy(self.image.data)
                # remove background
                self.my_logger.info('\n\tRemove background using astropy SExtractorBackground()...')
                data_wo_bkg = remove_image_background_sextractor(data, sigma=3.0, box_size=(50, 50),
                                                                 filter_size=(11, 11), positive=True)
                # extract source positions and fluxes
                self.my_logger.info('\n\tDetect sources using photutils iraf_source_detection()...')
                self.sources = iraf_source_detection(data_wo_bkg, sigma=3.0, fwhm=3.0, threshold_std_factor=5,
                                                     mask=None)
                if extent is not None:
                    self.sources['xcentroid'] += extent[0][0]
                    self.sources['ycentroid'] += extent[1][0]
                # write results in fits file
                self.write_sources()
                solve_field_input = self.sources_file_name
            elif self.source_extractor == "astrometrynet":
                self.my_logger.info(f"\n\tSource extraction directly with solve-field.")
                # must write a temporary image file with Spectractor flips and rotations
                fits.writeto(tmp_image_file_name, self.image.data, header=self.image.header, overwrite=True)
                solve_field_input = tmp_image_file_name
            else:
                raise ValueError(f"Got {self.source_extractor=}. Must be either 'iraf' or 'astrometrynet' "
                                 f"if sources are not given in argument.")
        else:
            self.sources = sources
            self.write_sources()
            solve_field_input = self.sources_file_name

        # run astrometry.net
        exec = _get_astrometry_executable_path('solve-field')
        command = f"{exec} --scale-unit arcsecperpix " \
                  f"--scale-low {0.95 * parameters.CCD_PIXEL2ARCSEC} " \
                  f"--scale-high {1.05 * parameters.CCD_PIXEL2ARCSEC} " \
                  f"--ra {self.image.target.radec_position.ra.value} --dec {self.image.target.radec_position.dec.value} " \
                  f"--radius {parameters.CCD_IMSIZE * parameters.CCD_PIXEL2ARCSEC / 3600.} " \
                  f"--dir {self.output_directory} --out {self.tag} " \
                  f"--overwrite --x-column X --y-column Y {solve_field_input} " \
                  f"--width {self.image.data.shape[1]} --height {self.image.data.shape[0]} --no-plots"
        self.my_logger.info(f'\n\tRun astrometry.net solve_field command:\n\t{command}')
        try:
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='ascii')
        except subprocess.CalledProcessError as e:  # pragma: nocover
            self.my_logger.warning(f"\n\tAstrometry command:\n{command}")
            self.my_logger.error(f"\n\t{e.stderr}")
            sys.exit()
        if os.path.isfile(tmp_image_file_name):
            os.remove(tmp_image_file_name)
        if os.path.isfile(os.path.join(self.output_directory, self.tag.rstrip('/')+".new")):
            os.remove(os.path.join(self.output_directory, self.tag.rstrip('/')+".new"))

        # The source file given to solve-field is understood as a FITS file with pixel origin value at 1,
        # whereas pixel coordinates comes from photutils using a numpy convention with pixel origin value at 0
        # To correct for this we shift the CRPIX center of 1 pixel
        with fits.open(self.wcs_file_name) as hdu:
            hdu[0].header['CRPIX1'] = float(hdu[0].header['CRPIX1']) + 1
            hdu[0].header['CRPIX2'] = float(hdu[0].header['CRPIX2']) + 1
            self.my_logger.info(f"\n\tWrite astrometry.net WCS solution in {self.wcs_file_name}...")
            hdu.writeto(self.wcs_file_name, overwrite=True)

        # load quad stars
        self.quad_stars_pixel_positions = self.get_quad_stars_pixel_positions()

        # load sources
        if self.sources is None:
            self.sources = self.load_sources_from_file()

        # load WCS
        self.wcs = load_wcs_from_file(self.wcs_file_name)
        return self.wcs

    # noinspection PyUnresolvedReferences
    def run_gaia_astrometry(self, min_range=3 * u.arcsec, max_range=5 * u.arcmin, max_sep=1 * u.arcsec):
        """Refine a World Coordinate System (WCS) using Gaia satellite astrometry catalog.

        A WCS must be already present in the exposure FITS file.

        The name of the target must be given to get its RA,DEC coordinates via a Simbad query.
        A matching is performed between the detected sources and the Gaia catalog obtained for the region of the target.
        Then the closest and brightest sources are selected and the WCS is shifted by the median of the distance between
        these stars and the detected sources. The original WCS FITS file is updated.

        Parameters
        ----------
        min_range: astropy.Quantity, optional
            Minimum distance for sources from image principal target in arcsec (default: 3*u.arcsec).
        max_range: astropy.Quantity, optional
            Maximum distance for sources from image principal target in arcsec (default: 5*u.arcsec).
        max_sep: astropy.Quantity, optional
            Maximum separation between the detected sources and the Gaia stars in arcsec (default: 1*u.arcsec).

        Examples
        --------

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> from spectractor import parameters
        >>> parameters.VERBOSE = True
        >>> parameters.DEBUG = True
        >>> logbook = LogBook(logbook='./tests/data/ctiofulllogbook_jun2017_v5.csv')
        >>> file_name = './tests/data/reduc_20170530_134.fits'
        >>> if os.path.isfile('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs'):
        ...     os.remove('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs')
        >>> tag = file_name.split('/')[-1]
        >>> disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        >>> im = Image(file_name, target_label=target_label, disperser_label=disperser_label, config="ctio.ini")  # doctest: +ELLIPSIS
        >>> a = Astrometry(im)
        >>> a.run_simple_astrometry(extent=((300,1400),(300,1400)))  # doctest: +ELLIPSIS
        WCS ...
        >>> dra, ddec = a.run_gaia_astrometry()

        .. doctest:
            :hide:

            >>> dra_median = np.median(dra.to(u.arcsec).value)
            >>> ddec_median = np.median(ddec.to(u.arcsec).value)
            >>> assert os.path.isdir('./tests/data/reduc_20170530_134_wcs')
            >>> assert os.path.isfile('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs')
            >>> assert np.all(np.abs([dra_median, ddec_median]) < 1e-3)

        """
        # load detected sources
        if self.sources is None:
            self.sources = self.load_sources_from_file()
        # load WCS if absent
        if self.wcs is None:
            self.wcs = load_wcs_from_file(self.wcs_file_name)
        self.sources_radec_positions = self.get_sources_radec_positions()

        # load the Gaia catalog
        if self.gaia_catalog is None:
            if os.path.isfile(self.gaia_file_name):
                self.my_logger.info(f"\n\tLoad Gaia catalog from {self.gaia_file_name}.")
                self.gaia_catalog = ascii.read(self.gaia_file_name, format="ecsv")
            else:
                self.load_gaia_catalog_around_target()
            self.my_logger.info(f"\n\tGaia catalog loaded.")

        # update coordinates with proper motion data
        self.my_logger.info(f"\n\tUpdate object coordinates with proper motion at time={self.image.date_obs}.")
        self.image.target_radec_position_after_pm = self.image.target.get_radec_position_after_pm(self.image.date_obs)
        self.gaia_radec_positions_after_pm = get_gaia_coords_after_proper_motion(self.gaia_catalog, self.image.date_obs)
        if parameters.DEBUG:
            self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_radec_positions_after_pm,
                                               quad=self.quad_stars_pixel_positions, margin=1000,
                                               label=self.image.target.label)

        # compute shifts
        self.my_logger.info(f"\n\tCompute distances between Gaia catalog and detected sources.")
        self.gaia_index, self.dist_2d, self.dist_ra, self.dist_dec = \
            self.match_sources_to_gaia_catalog(self.gaia_radec_positions_after_pm)
        if parameters.DEBUG:
            self.plot_astrometry_shifts(vmax=3)

        # select the brightest and closest stars with minimal shift
        if len(self.sources) > 50:
            flux_log10_threshold = np.log10(self.sources['flux'][int(0.5 * len(self.sources))])
        else:
            flux_log10_threshold = np.log10(self.sources['flux'][int(0.8 * len(self.sources))])
        sep_constraints = self.set_constraints(flux_log10_threshold=flux_log10_threshold, max_sep=max_sep,
                                               max_range=max_range, min_range=min_range)
        if np.sum(sep_constraints) == 0:
            raise ValueError(f"Warning! No source passes the set threshold flux>{10**flux_log10_threshold}. "
                             f"Check your filters. {self.sources=}")
        sources_selection = self.sources_radec_positions[sep_constraints]
        gaia_matches = self.gaia_radec_positions_after_pm[self.gaia_index[sep_constraints]]
        dra, ddec = sources_selection.spherical_offsets_to(gaia_matches)

        # compute statistics
        dra_median = np.median(dra.to(u.arcsec).value)
        ddec_median = np.median(ddec.to(u.arcsec).value)
        self.my_logger.info(f"\n\tMedian DeltaRA={dra_median:.3f} arcsec, median DeltaDEC={ddec_median:.3f} arcsec")
        if parameters.DEBUG:
            plot_shifts_histograms(dra, ddec)
            self.plot_shifts_profiles(gaia_matches, dra, ddec)

        # update WCS
        # tested with high latitude 20170530_120.fits exposure: dra shift must be divided by cos(dec)
        # to set new WCS system because spherical_offsets_to gives shifts angle at equator
        # (see https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html
        # #astropy.coordinates.SkyCoord.spherical_offsets_to)
        # after the shift the histograms must be centered on zero
        total_shift = np.array(
            [dra_median / np.cos(self.image.target_radec_position_after_pm.dec.radian), ddec_median]) * u.arcsec
        self.my_logger.info(f"\n\tShift original CRVAL value {self.wcs.wcs.crval * u.deg} of {total_shift}.")
        self.wcs.wcs.crval = self.wcs.wcs.crval * u.deg + total_shift
        # if parameters.DEBUG:
        #     self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_radec_positions_after_pm,
        #                                        quad=np.array(self.quad_stars_pixel_positions).T,
        #                                        margin=10, label=self.image.target.label)

        # Now, write out the WCS object as a FITS header
        dra_median = np.median(dra.to(u.arcsec).value)
        ddec_median = np.median(ddec.to(u.arcsec).value)
        dra_rms = np.std(dra.to(u.arcsec).value)
        ddec_rms = np.std(ddec.to(u.arcsec).value)
        with fits.open(self.wcs_file_name) as hdu:
            hdu[0].header['CRVAL1'] = self.wcs.wcs.crval[0]
            hdu[0].header['CRVAL2'] = self.wcs.wcs.crval[1]
            hdu[0].header['CRV1_MED'] = dra_median
            hdu[0].header['CRV2_MED'] = ddec_median
            hdu[0].header['CRV1_RMS'] = dra_rms
            hdu[0].header['CRV2_RMS'] = ddec_rms
            hdu.writeto(self.wcs_file_name, overwrite=True)

        # check histogram medians
        self.wcs = load_wcs_from_file(self.wcs_file_name)
        self.get_sources_radec_positions()
        self.gaia_index, self.dist_2d, self.dist_ra, self.dist_dec = \
            self.match_sources_to_gaia_catalog(self.gaia_radec_positions_after_pm)
        sep_constraints = self.set_constraints(flux_log10_threshold=flux_log10_threshold, max_sep=max_sep,
                                               max_range=max_range, min_range=min_range)
        sources_selection = self.sources_radec_positions[sep_constraints]
        self.gaia_matches = self.gaia_radec_positions_after_pm[self.gaia_index[sep_constraints]]
        dra, ddec = sources_selection.spherical_offsets_to(self.gaia_matches)

        # update values
        self.my_logger.info(f"\n\tUpdate WCS solution from {self.wcs_file_name} with Gaia solution.")
        dra_median = np.median(dra.to(u.arcsec).value)
        ddec_median = np.median(ddec.to(u.arcsec).value)
        dra_rms = np.std(dra.to(u.arcsec).value)
        ddec_rms = np.std(ddec.to(u.arcsec).value)
        with fits.open(self.wcs_file_name) as hdu:
            hdu[0].header['CRV1_MED'] = dra_median
            hdu[0].header['CRV2_MED'] = ddec_median
            hdu[0].header['CRV1_RMS'] = dra_rms
            hdu[0].header['CRV2_RMS'] = ddec_rms
            hdu.writeto(self.wcs_file_name, overwrite=True)

        # if parameters.DEBUG:
        #     plot_shifts_histograms(dra, ddec)
        #     self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_radec_positions_after_pm,
        #                                        quad=np.array(self.quad_stars_pixel_positions).T,
        #                                        margin=10, label=self.image.target.label)
        return dra, ddec

    def run_full_astrometry(self, extent=None, maxiter=20, min_range=3 * u.arcsec, max_range=5 * u.arcmin, max_sep=1 * u.arcsec):
        """Iterative method to get a precise World Coordinate System (WCS) using Gaia satellite astrometry catalog and
        astrometry.net fitter.

        This function runs alternatively the run_simple_astrometry and the run_gaia_astrometry.
        After an iteration, the distances between the quad stars and the associated Gaia stars are
        computed. The quad star with the largest distance is removed from the list of detected sources.
        Then the function iterates again until it reaches the maxiter maximum number of iterations
        or when astrometry.net fails because too few sources are left.
        At each iteration, the total quadratic sum of the distances between the quad stars and the
        Gaia stars is computed. The iteration with the smallest total distance (ie the smallest centroid
        distances in the pixel space) is kept as the best iteration and gives the final WCS.

        A new WCS is created and saved as a new FITS file. The WCS file and the intermediate results
        are saved in a new directory named as the FITS file name with a _wcs suffix.

        Parameters
        ----------
        extent: array_like, optional
            A ((xmin,xmax),(ymin,ymax)) to crop the image before any analysis (default: None).
        maxiter: int, optional
            The maximum number of iterations (default: 20).
        min_range: astropy.Quantity, optional
            Minimum distance for sources from image principal target in arcsec (default: 3*u.arcsec).
        max_range: astropy.Quantity, optional
            Maximum distance for sources from image principal target in arcsec (default: 5*u.arcsec).
        max_sep: astropy.Quantity, optional
            Maximum separation between the detected sources and the Gaia stars in arcsec (default: 1*u.arcsec).

        Returns
        -------
        min_gaia_residuals_quad_sum: float
            The minimum total quadratic distance between Gaia stars and the quad stars (in pixels).

        Examples
        --------

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> from spectractor import parameters
        >>> parameters.VERBOSE = True
        >>> parameters.DEBUG = True
        >>> radius = 100
        >>> maxiter = 10
        >>> logbook = LogBook(logbook='./tests/data/ctiofulllogbook_jun2017_v5.csv')
        >>> file_name = './tests/data/reduc_20170530_134.fits'
        >>> if os.path.isfile('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs'):
        ...     os.remove('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs')
        >>> tag = file_name.split('/')[-1]
        >>> disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        >>> im = Image(file_name, target_label=target_label, disperser_label=disperser_label, config="ctio.ini")  # doctest: +ELLIPSIS
        >>> a = Astrometry(im)
        >>> extent = ((max(0, xpos - radius), min(xpos + radius, parameters.CCD_IMSIZE)),
        ...           (max(0, ypos - radius), min(ypos + radius, parameters.CCD_IMSIZE)))
        >>> gaia_min_residuals = a.run_full_astrometry(extent=extent, maxiter=maxiter)  #doctest: +ELLIPSIS
        iter target_x ...

        .. doctest:
            :hide:

            >>> assert os.path.isdir(a.output_directory)
            >>> assert os.path.isfile(set_wcs_file_name(file_name))
            >>> assert a.image.data is not None
            >>> assert np.sum(a.image.data) > 1e-10
            >>> assert gaia_min_residuals < 0.8

        """

        t = Table(names=["iter", "target_x", "target_y", "gaia_residuals_abs_sum_x",
                         "gaia_residuals_abs_sum_y", "gaia_residuals_quad_sum"])
        t["iter"].format = "%d"
        for c in t.columns[1:]:
            t[c].format = "%.2f"
        sources_list = []
        for k in range(maxiter):
            self.my_logger.info(f'\n\tIteration #{k}')
            try:
                # find a simple astrometric solution using astrometry.net
                self.run_simple_astrometry(extent=extent, sources=self.sources)
                # refine with Gaia catalog
                for i in range(maxiter):
                    dra, ddec = self.run_gaia_astrometry(min_range=min_range, max_range=max_range, max_sep=max_sep)
                    dra_median = np.median(dra.to(u.mas).value)
                    ddec_median = np.median(ddec.to(u.mas).value)
                    if np.abs(dra_median) < 0.5 * parameters.CCD_PIXEL2ARCSEC and np.abs(ddec_median) < 0.5 * parameters.CCD_PIXEL2ARCSEC:
                        break
                sources_list.append(deepcopy(self.sources))
                # check the positions of quad stars with their WCS position from Gaia catalog
                gaia_residuals = self.compute_gaia_pixel_residuals()
                gaia_residuals_sum_x, gaia_residuals_sum_y = np.sum(np.abs(gaia_residuals), axis=0)
                gaia_residuals_quad_sum = np.sum(np.sqrt(np.sum(gaia_residuals ** 2, axis=1)))
                if parameters.DEBUG:
                    self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_matches, margin=20,
                                                       quad=self.quad_stars_pixel_positions,
                                                       label=self.image.target.label)
                    self.plot_astrometry_shifts(vmax=3)
                    self.plot_quad_stars()
                target_x, target_y = self.get_target_pixel_position()
                t.add_row([k, target_x, target_y, gaia_residuals_sum_x, gaia_residuals_sum_y, gaia_residuals_quad_sum])
                self.remove_worst_quad_star_from_sources()
            except FileNotFoundError or TimeoutError:
                self.my_logger.warning(f"\n\tAstrometry.net failed at iteration {k}. "
                                       f"Stop the loop here and look for best solution.")
                k -= 1
                break
        t.pprint_all()
        if len(t) == 0:
            raise IndexError(f"Astrometry has failed at every iteration, empty table {t=}.")
        best_iter = int(np.argmin(t["gaia_residuals_quad_sum"]))
        self.my_logger.info(f'\n\tBest run: iteration #{best_iter}')
        self.run_simple_astrometry(extent=extent, sources=sources_list[best_iter])
        self.run_gaia_astrometry(min_range=min_range, max_range=max_range, max_sep=max_sep)
        self.my_logger.info(f'\n\tFinal target position: {self.get_target_pixel_position()}')
        if parameters.DEBUG:
            self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_matches, margin=20,
                                               quad=self.quad_stars_pixel_positions, label="FINAL")
            self.plot_quad_stars()
        return np.min(t["gaia_residuals_quad_sum"])


if __name__ == "__main__":
    import doctest
    im = Image("./tests/data/reduc_20170530_134.fits", target_label="HD111980")
    a = Astrometry(im, wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
    a.load_gaia_catalog_around_target()
    doctest.testmod()

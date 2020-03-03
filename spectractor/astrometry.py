import os
from copy import deepcopy
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.io import fits, ascii
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, Distance

from photutils import IRAFStarFinder

from scipy.spatial import ConvexHull

from spectractor import parameters
from spectractor.tools import (plot_image_simple, set_wcs_file_name, set_wcs_tag, set_wcs_output_directory,
                               set_sources_file_name, set_gaia_catalog_file_name, load_wcs_from_file, ensure_dir)
from spectractor.config import set_logger
from spectractor.extractor.images import Image
from spectractor.extractor.background import remove_image_background_sextractor


def source_detection(data_wo_bkg, sigma=3.0, fwhm=3.0, threshold_std_factor=5, mask=None):
    """Function to detect point-like sources in a data array.

    This function use the photutils IRAFStarFinder module to search for sources in an image. This finder
    is better than DAOStarFinder for the astrometry of isolated sources but less good for photometry.

    Parameters
    ----------
    data_wo_bkg: array_like
        The image data array. It works better if the background was subtracted before.
    sigma: float
        Standard deviation value for sigma clipping function before finding sources (default: 3.0).
    fwhm: float
        Full width half maximum for the source detection algorithm (default: 3.0).
    threshold_std_factor: float
        Only sources with a flux above this value times the RMS of the images are kept (default: 5).
    mask: array_like, optional
        Boolean array to mask image pixels (default: None).

    Returns
    -------
    sources: Table
        Astropy table containing the source centroids and fluxes, ordered by decreasing magnitudes.

    Examples
    --------

    >>> N = 100
    >>> data = np.ones((N, N))
    >>> yy, xx = np.mgrid[:N, :N]
    >>> x_center, y_center = 20, 30
    >>> data += 10*np.exp(-((x_center-xx)**2+(y_center-yy)**2)/10)
    >>> sources = source_detection(data)
    >>> print(float(sources["xcentroid"]), float(sources["ycentroid"]))
    20.0 30.0

    .. doctest:
        :hide:

        >>> assert len(sources) == 1
        >>> assert sources["xcentroid"] == x_center
        >>> assert sources["ycentroid"] == y_center

    .. plot:

        from spectractor.tools import plot_image_simple
        from spectractor.astrometry import source_detection
        import numpy as np
        import matplotlib.pyplot as plt

        N = 100
        data = np.ones((N, N))
        yy, xx = np.mgrid[:N, :N]
        x_center, y_center = 20, 30
        data += 10*np.exp(-((x_center-xx)**2+(y_center-yy)**2)/10)
        sources = source_detection(data)
        fig = plt.figure(figsize=(6,5))
        plot_image_simple(plt.gca(), data, target_pixcoords=(sources["xcentroid"], sources["ycentroid"]))
        fig.tight_layout()
        plt.show()

    """
    mean, median, std = sigma_clipped_stats(data_wo_bkg, sigma=sigma)
    if mask is None:
        mask = np.zeros(data_wo_bkg.shape, dtype=bool)
    # daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_std_factor * std, exclude_border=True)
    # sources = daofind(data_wo_bkg - median, mask=mask)
    iraffind = IRAFStarFinder(fwhm=fwhm, threshold=threshold_std_factor * std, exclude_border=True)
    sources = iraffind(data_wo_bkg - median, mask=mask)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
    sources.sort('mag')
    if parameters.DEBUG:
        positions = np.array((sources['xcentroid'], sources['ycentroid']))
        plot_image_simple(plt.gca(), data_wo_bkg, scale="log10", target_pixcoords=positions)
        if parameters.DISPLAY:
            # fig.tight_layout()
            plt.show()
    return sources


def load_gaia_catalog(coord, radius=5 * u.arcmin):
    """Load the Gaia catalog of stars around a given RA,DEC position within a given radius.

    Parameters
    ----------
    coord: SkyCoord
        Central coordinates for the Gaia cone search.
    radius: float
        Radius size for the cone search, with angle units (default: 5u.arcmin).

    Returns
    -------
    gaia_catalog: Table
        The Gaia catalog.

    Examples
    --------

    >>> from astropy.coordinates import SkyCoord
    >>> c = SkyCoord(ra=0*u.deg, dec=0*u.deg)
    >>> gaia_catalog = load_gaia_catalog(c, radius=1*u.arcmin)  # doctest: +ELLIPSIS
    INFO: Query finished...
    >>> print(gaia_catalog)  # doctest: +SKIP
            dist        ...

    .. doctest:
        :hide:

        >>> assert len(gaia_catalog) > 0

    """
    from astroquery.gaia import Gaia
    my_logger = set_logger("load_gaia_catalog")
    job = Gaia.cone_search_async(coord, radius=radius, verbose=False)
    my_logger.debug(f"\n\t{job}")
    gaia_catalog = job.get_results()
    my_logger.debug(f"\n\t{gaia_catalog}")
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
    >>> gaia_catalog = load_gaia_catalog(c, radius=1*u.arcmin)  # doctest: +ELLIPSIS
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


def plot_shifts_histograms(dra, ddec):
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


class Astrometry(Image):

    def __init__(self, file_name, target_label="", disperser_label="", wcs_file_name="", output_directory=""):
        """Class to handle astrometric computations.

        Parameters
        ----------
        file_name: str
            Input file name of the image to analyse.
        target_label: str, optional
            The name of the targeted object (default: "").
        disperser_label: str, optional
            The name of the disperser (default: "").
        wcs_file_name: str, optional
            The path to a WCS fits file. WCS content will be loaded (default: "").
        output_directory: str, optional
            The output directory path. If empty, a directory *_wcs is created next to the analyzed image (default: "").
        """
        Image.__init__(self, file_name, target_label=target_label, disperser_label=disperser_label)
        self.my_logger = set_logger(self.__class__.__name__)
        self.output_directory = set_wcs_output_directory(file_name, output_directory=output_directory)
        ensure_dir(self.output_directory)
        self.tag = set_wcs_tag(file_name)
        self.new_file_name = self.file_name.replace('.fits', '_new.fits')
        self.sources_file_name = set_sources_file_name(file_name, output_directory=output_directory)
        self.wcs_file_name = wcs_file_name
        self.log_file_name = os.path.join(self.output_directory, self.tag) + ".log"
        self.wcs = None
        if self.wcs_file_name != "":
            if os.path.isfile(self.wcs_file_name):
                self.wcs = load_wcs_from_file(self.wcs_file_name)
            else:
                self.my_logger.warning(f"WCS file {wcs_file_name} does not exist. Skip it.")
        else:
            self.wcs_file_name = set_wcs_file_name(file_name, output_directory=output_directory)
            if os.path.isfile(self.wcs_file_name):
                self.wcs = load_wcs_from_file(self.wcs_file_name)
        self.gaia_file_name = set_gaia_catalog_file_name(file_name, output_directory=output_directory)
        self.gaia_catalog = None
        self.gaia_index = None
        self.gaia_matches = None
        self.gaia_residuals = None
        self.gaia_radec_positions_after_pm = None
        if os.path.isfile(self.gaia_file_name):
            self.my_logger.info(f"\n\tLoad Gaia catalog from {self.gaia_file_name}.")
            self.gaia_catalog = ascii.read(self.gaia_file_name, format="ecsv")
            self.gaia_radec_positions_after_pm = get_gaia_coords_after_proper_motion(self.gaia_catalog, self.date_obs)
        self.sources = None
        self.sources_radec_positions = None
        if os.path.isfile(self.sources_file_name):
            self.sources = self.load_sources_from_file()
        self.my_logger.info(f"\n\tIntermediate outputs will be stored in {self.output_directory}")
        self.dist_2d = None
        self.quad_stars_pixel_positions = None
        self.dist_ra = 0 * u.arcsec
        self.dist_dec = 0 * u.arcsec
        self.target_radec_position_after_pm = self.target.get_radec_position_after_pm(date_obs=self.date_obs)
        if os.path.isfile(self.log_file_name):
            self.quad_stars_pixel_positions = self.get_quad_stars_pixel_positions()

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

        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> target_x, target_y = a.get_target_pixel_position()
        >>> print(target_x, target_y) # doctest: +ELLIPSIS
        743... 683...

        """
        target_x, target_y = self.wcs.all_world2pix(self.target_radec_position_after_pm.ra,
                                                    self.target_radec_position_after_pm.dec, 0)
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

        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> gaia_x, gaia_y = a.get_gaia_pixel_positions()

        or with selected Gaia index:

        >>> gaia_x, gaia_y = a.get_gaia_pixel_positions(gaia_index=[1, 2])

        .. doctest:
            :hide:

            >>> assert np.isclose(gaia_x[0], 744, atol=0.5)
            >>> assert np.isclose(gaia_y[0], 683, atol=0.5)

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

        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> quad_stars = a.get_quad_stars_pixel_positions()

        .. doctest:
            :hide:

            >>> assert quad_stars.shape == (4, 2)

        """
        coords = []
        f = open(self.log_file_name, 'r')
        for line in f:
            if 'field_xy' in line:
                coord = line.split(' ')[5].split(',')
                coords.append([float(coord[0]), float(coord[1])])
        f.close()
        if len(coords) < 4:
            self.my_logger.warning(f"\n\tOnly {len(coords)} calibration stars has been extracted from "
                                   f"{self.log_file_name}, with positions {coords}. "
                                   f"A quad of at least 4 stars is expected. "
                                   f"Please check {self.log_file_name}.")
        self.quad_stars_pixel_positions = np.array(coords)
        return self.quad_stars_pixel_positions

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
        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.load_sources_from_file()  # doctest: +ELLIPSIS
        <Table length=...

        """
        self.my_logger.info(f"\n\tLoad source positions and flux from {self.sources_file_name}")
        sources = Table.read(self.sources_file_name)
        sources['X'].name = "xcentroid"
        sources['Y'].name = "ycentroid"
        sources['FLUX'].name = "flux"
        self.sources = sources
        return sources

    def load_gaia_catalog_around_target(self):
        """Load the Gaia stars catalog around the target position.

        The radius of the search is set accordingly to the maximum range of detected sources.

        Returns
        -------
        gaia_catalog: Table
            The table of Gaia stars around the target position.

        Examples
        --------
        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.load_gaia_catalog_around_target() # doctest: +ELLIPSIS
        INFO: Query finished...

        """
        radius = 0.5*np.sqrt(2) * max(np.max(self.sources["xcentroid"]) - np.min(self.sources["xcentroid"]),
                                      np.max(self.sources["ycentroid"]) - np.min(self.sources["ycentroid"]))
        radius *= parameters.CCD_PIXEL2ARCSEC * u.arcsec
        self.my_logger.info(f"\n\tLoading Gaia catalog within radius < {radius.value} "
                            f"arcsec from {self.target.label} {self.target.radec_position}...")
        self.gaia_catalog = load_gaia_catalog(self.target.radec_position, radius=radius)
        ascii.write(self.gaia_catalog, self.gaia_file_name, format='ecsv', overwrite=True)
        return self.gaia_catalog

    def write_sources(self):
        """Write a fits file containing the source positions and fluxes.
        The name of the file is set accordingly to the name of the WCS file.

        """
        colx = fits.Column(name='X', format='D', array=self.sources['xcentroid'])
        coly = fits.Column(name='Y', format='D', array=self.sources['ycentroid'])
        colflux = fits.Column(name='FLUX', format='D', array=self.sources['flux'])
        coldefs = fits.ColDefs([colx, coly, colflux])
        hdu = fits.BinTableHDU.from_columns(coldefs)
        hdu.header['IMAGEW'] = self.data.shape[1]
        hdu.header['IMAGEH'] = self.data.shape[0]
        hdu.writeto(self.sources_file_name, overwrite=True)
        self.my_logger.info(f'\n\tSources positions saved in {self.sources_file_name}')

    def plot_sources_and_gaia_catalog(self, ax=None, sources=None, gaia_coord=None, quad=None, label="",
                                      vmax=None, margin=parameters.CCD_IMSIZE, center=None, scale="log10"):
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

        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.plot_sources_and_gaia_catalog(sources=a.sources, gaia_coord=a.gaia_radec_positions_after_pm,
        ...                                 quad=a.quad_stars_pixel_positions,
        ...                                 label=a.target.label)

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
                fig.add_subplot(111, projection=self.wcs)
            ax = plt.gca()
        else:
            no_plot = True

        plot_image_simple(ax, self.data, scale=scale, vmax=vmax)
        if self.wcs is not None and not no_plot:
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
        if sources is not None:
            ax.scatter(sources['xcentroid'], sources['ycentroid'], s=300, lw=2,
                       edgecolor='black', facecolor='none', label="Detected sources")
        if gaia_coord is not None:
            gaia_x, gaia_y = self.wcs.all_world2pix(self.gaia_radec_positions_after_pm.ra,
                                                    self.gaia_radec_positions_after_pm.dec, 0, quiet=True)
            ax.scatter(gaia_x, gaia_y, s=300, marker="+", edgecolor='blue', facecolor='blue', label=f"Gaia stars", lw=2)
        if center is None:
            target_x, target_y = self.get_target_pixel_position()
        else:
            target_x, target_y = center
        ax.scatter(target_x, target_y, s=300, marker="x", edgecolor='cyan', facecolor='cyan', label=f"{label}", lw=2)
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
        ax.set_xlim(max(0, target_x - margin), min(target_x + margin, self.data.shape[1]))
        ax.set_ylim(max(0, target_y - margin), min(target_y + margin, self.data.shape[0]))
        if not no_plot and parameters.DISPLAY:
            # fig.tight_layout()
            plt.show()

    def get_sources_radec_positions(self):
        """Gives the RA,DEC position of the detected sources.

        Returns
        -------
        coords: SkyCoord
            The RA,DEC positions

        Examples
        --------
        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.get_sources_radec_positions()  # doctest: +ELLIPSIS
        <SkyCoord (ICRS): (ra, dec) in deg...

        """
        sources_coord = self.wcs.all_pix2world(self.sources['xcentroid'], self.sources['ycentroid'], 0)
        self.sources_radec_positions = SkyCoord(ra=sources_coord[0] * u.deg, dec=sources_coord[1] * u.deg,
                                                frame="icrs", obstime=self.date_obs, equinox="J2000")
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
        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
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
        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
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

        plot_image_simple(plt.gca(), self.data, scale="log10")
        plt.xlabel('RA')
        plt.ylabel('Dec')
        if self.sources is not None:
            plt.scatter(self.sources['xcentroid'], self.sources['ycentroid'], s=100, lw=2,
                        edgecolor='black', facecolor='none', label="Detected sources")
        sc = plt.scatter(gaia_x, gaia_y, s=100, c=self.dist_ra.to(u.arcsec).value,
                         cmap="bwr", vmin=-vmax, vmax=vmax,
                         label=f"Gaia stars", lw=1)
        plt.xlim(max(0, int(target_x - margin)), min(int(target_x + margin), self.data.shape[1]))
        plt.ylim(max(0, int(target_y - margin)), min(int(target_y + margin), self.data.shape[0]))
        plt.colorbar(sc, label="Shift in RA [arcsec]")
        plt.legend()

        fig.add_subplot(212, projection=self.wcs)
        plot_image_simple(plt.gca(), self.data, scale="log10")
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.grid(color='white', ls='solid')
        if self.sources is not None:
            plt.scatter(self.sources['xcentroid'], self.sources['ycentroid'], s=100, lw=2,
                        edgecolor='black', facecolor='none', label="all sources")
        sc = plt.scatter(gaia_x, gaia_y, s=100, c=self.dist_dec.to(u.arcsec).value,
                         cmap="bwr", vmin=-vmax, vmax=vmax,
                         label=f"Gaia Stars", lw=1)
        plt.xlim(max(0, int(target_x - margin)), min(int(target_x + margin), self.data.shape[1]))
        plt.ylim(max(0, int(target_y - margin)), min(int(target_y + margin), self.data.shape[0]))
        plt.colorbar(sc, label="Shift in DEC [arcsec]")
        plt.legend()
        if parameters.DISPLAY:
            # fig.tight_layout()
            plt.show()

    def set_constraints(self, min_stars=100, flux_log10_threshold=0.1, min_range=3 * u.arcsec, max_range=5 * u.arcmin,
                        max_sep=1 * u.arcsec):
        """Gives a boolean array for sources that respect certain criterai (see below).

        Parameters
        ----------
        min_stars: int
            Minimum number of stars that have to be kept in the selection (default: 100).
        flux_log10_threshold:
            Lower cut on the log10 of the star fluxes (default: 0.1).
        min_range:
            Minimum distance for sources from image principal target in arcsec (default: 3*u.arcsec).
        max_range
            Maximum distance for sources from image principal target in arcsec (default: 3*u.arcsec).
        max_sep
            Maximum separation between the detected sources and the Gaia stars in arcsec (default: 1*u.arcsec).

        Returns
        -------
        indices: array_like
            Boolean array of selected sources.

        """
        sep = self.dist_2d < max_sep
        sep *= self.sources_radec_positions.separation(self.target_radec_position_after_pm) < max_range
        sep *= self.sources_radec_positions.separation(self.target_radec_position_after_pm) > min_range
        sep *= np.log10(self.sources['flux']) > flux_log10_threshold
        if np.sum(sep) > min_stars:
            for r in np.arange(0, max_range.value, 0.1)[::-1]:
                range_constraint = self.sources_radec_positions.separation(self.target.radec_position_after_pm) \
                                   < r * u.arcmin
                if np.sum(sep * range_constraint) < min_stars:
                    break
                else:
                    sep *= range_constraint
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

        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
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
        ax[0].axvline(self.target_radec_position_after_pm.ra.value, color='k', linestyle="--",
                      label=f"{self.target.label} RA")
        ax[1].axvline(self.target_radec_position_after_pm.dec.value, color='k', linestyle="--",
                      label=f"{self.target.label} DEC")
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

    def merge_wcs_with_new_exposure(self, log_file=None):
        """Merge the WCS solution with the current FITS file image.

        Parameters
        ----------
        log_file: file, optional
            Log file to write the output of the merge command.

        """
        command = f"{os.path.join(parameters.ASTROMETRYNET_DIR, 'bin/new-wcs')} -v -d -i {self.file_name} " \
                  f"-w {self.wcs_file_name} -o {self.new_file_name}\n"
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

        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> residuals = a.compute_gaia_pixel_residuals()

        .. doctest:
            :hide:

            >>> assert residuals.shape == (4, 2)
            >>> assert np.all(np.abs(residuals) < 0.2)

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

        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
        >>> a.load_sources_from_file() # doctest: +ELLIPSIS
        <Table length=...
        >>> index = a.find_quad_star_index_in_sources(a.quad_stars_pixel_positions[0])
        >>> print(index)
        3

        """
        eps = 1e-1
        k = -1
        for k in range(len(self.sources)):
            if abs(self.sources['xcentroid'][k] - quad_star[0]) < eps \
                    and abs(self.sources['ycentroid'][k] - quad_star[1]):
                break
        return k

    def remove_worst_quad_star_from_sources(self):
        """Remove the quad star from the source table with the largest distance between the source centroid
        and the associated Gaia star position.

        Examples
        --------

        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
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

        >>> a = Astrometry("./tests/data/reduc_20170530_134.fits", target_label="HD111980",
        ...                wcs_file_name="./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs")
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

    def run_simple_astrometry(self, extent=None, sources=None):
        """Build a World Coordinate System (WCS) using astrometry.net library given an exposure as a FITS file.

        The name of the target must be given to get its RA,DEC coordinates via a Simbad query.
        First the background of the exposure is removed using the astropy SExtractorBackground() method.
        Then photutils source_detection() is used to get the positions in pixels en flux of the objects in the field.
        The results are saved in the {file_name}_sources.fits file and used by the solve_field command from the
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

        source_detection()

        Examples
        --------

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> from spectractor.astrometry import Astrometry
        >>> from spectractor import parameters
        >>> parameters.VERBOSE = True
        >>> logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
        >>> file_names = ['./tests/data/reduc_20170530_134.fits']
        >>> if os.path.isfile('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs'):
        ...     os.remove('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs')
        >>> for file_name in file_names:
        ...     tag = file_name.split('/')[-1]
        ...     disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        ...     if target is None or xpos is None or ypos is None:
        ...         continue
        ...     a = Astrometry(file_name, target, disperser_label)
        ...     a.run_simple_astrometry(extent=((300,1400),(300,1400)))  # doctest: +ELLIPSIS
        WCS ...

        .. doctest:
            :hide:

            >>> assert os.path.isdir('./tests/data/reduc_20170530_134_wcs')
            >>> assert os.path.isfile('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs')

        """
        # crop data
        if extent is not None:
            data = self.data[extent[1][0]:extent[1][1], extent[0][0]:extent[0][1]]
        else:
            data = np.copy(self.data)
        if sources is None:
            # remove background
            self.my_logger.info('\n\tRemove background using astropy SExtractorBackground()...')
            data_wo_bkg = remove_image_background_sextractor(data, sigma=3.0, box_size=(50, 50),
                                                             filter_size=(10, 10), positive=True)
            # extract source positions and fluxes
            self.my_logger.info('\n\tDetect sources using photutils source_detection()...')
            self.sources = source_detection(data_wo_bkg, sigma=3.0, fwhm=3.0, threshold_std_factor=5, mask=None)
            if extent is not None:
                self.sources['xcentroid'] += extent[0][0]
                self.sources['ycentroid'] += extent[1][0]
            self.my_logger.info(f'\n\t{self.sources}')
        else:
            self.sources = sources
        # write results in fits file
        self.write_sources()
        # run astrometry.net
        command = f"{os.path.join(parameters.ASTROMETRYNET_DIR, 'bin/solve-field')} --scale-unit arcsecperpix " \
                  f"--scale-low {0.999 * parameters.CCD_PIXEL2ARCSEC} " \
                  f"--scale-high {1.001 * parameters.CCD_PIXEL2ARCSEC} " \
                  f"--ra {self.target.radec_position.ra.value} --dec {self.target.radec_position.dec.value} " \
                  f"--radius {parameters.CCD_IMSIZE * parameters.CCD_PIXEL2ARCSEC / 3600.} " \
                  f"--dir {self.output_directory} --out {self.tag} " \
                  f"--overwrite --x-column X --y-column Y {self.sources_file_name}"
        self.my_logger.info(f'\n\tRun astrometry.net solve_field command:\n\t{command}')
        log = subprocess.check_output(command, shell=True)
        log_file = open(self.log_file_name, "w+")
        log_file.write(command + "\n")
        log_file.write(log.decode("utf-8") + "\n")
        # save new WCS in original fits file
        # self.merge_wcs_with_new_exposure(log_file=log_file)
        log_file.close()

        # The source file given to solve-field is understood as a FITS file with pixel origin value at 1,
        # whereas pixel coordinates comes from photutils using a numpy convention with pixel origin value at 0
        # To correct for this we shift the CRPIX center of 1 pixel
        hdu = fits.open(self.wcs_file_name)
        hdu[0].header['CRPIX1'] = float(hdu[0].header['CRPIX1']) + 1
        hdu[0].header['CRPIX2'] = float(hdu[0].header['CRPIX2']) + 1
        self.my_logger.info(f"\n\tWrite astrometry.net WCS solution in {self.wcs_file_name}...")
        hdu.writeto(self.wcs_file_name, overwrite=True)

        # load quad stars
        self.quad_stars_pixel_positions = self.get_quad_stars_pixel_positions()

        # load WCS
        self.wcs = load_wcs_from_file(self.wcs_file_name)
        return self.wcs

    # noinspection PyUnresolvedReferences
    def run_gaia_astrometry(self):
        """Refine a World Coordinate System (WCS) using Gaia satellite astrometry catalog.

        A WCS must be already present in the exposure FITS file.

        The name of the target must be given to get its RA,DEC coordinates via a Simbad query.
        A matching is performed between the detected sources and the Gaia catalog obtained for the region of the target.
        Then the closest and brightest sources are selected and the WCS is shifted by the median of the distance between
        these stars and the detected sources. The original WCS FITS file is updated.

        Examples
        --------

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> from spectractor import parameters
        >>> parameters.VERBOSE = True
        >>> parameters.DEBUG = True
        >>> logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
        >>> file_names = ['./tests/data/reduc_20170530_134.fits']
        >>> if os.path.isfile('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs'):
        ...     os.remove('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs')
        >>> for file_name in file_names:
        ...     tag = file_name.split('/')[-1]
        ...     disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        ...     if target is None or xpos is None or ypos is None:
        ...         continue
        ...     a = Astrometry(file_name, target, disperser_label)
        ...     a.run_simple_astrometry(extent=((300,1400),(300,1400)))
        ...     dra, ddec = a.run_gaia_astrometry()  # doctest: +ELLIPSIS
        WCS ...

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
                radius = 2 * max(np.max(self.sources["xcentroid"]) - np.min(self.sources["xcentroid"]),
                                 np.max(self.sources["ycentroid"]) - np.min(self.sources["ycentroid"]))
                radius *= parameters.CCD_PIXEL2ARCSEC * u.arcsec
                self.my_logger.info(f"\n\tLoading Gaia catalog within radius < {radius.value} "
                                    f"arcsec from {self.target.label} {self.target.radec_position}...")
                self.gaia_catalog = load_gaia_catalog(self.target.radec_position, radius=radius)
                ascii.write(self.gaia_catalog, self.gaia_file_name, format='ecsv', overwrite=True)
            self.my_logger.info(f"\n\tGaia catalog loaded.")

        # update coordinates with proper motion data
        self.my_logger.info(f"\n\tUpdate object coordinates with proper motion at time={self.date_obs}.")
        self.target_radec_position_after_pm = self.target.get_radec_position_after_pm(self.date_obs)
        self.gaia_radec_positions_after_pm = get_gaia_coords_after_proper_motion(self.gaia_catalog, self.date_obs)
        if parameters.DEBUG:
            self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_radec_positions_after_pm,
                                               quad=self.quad_stars_pixel_positions, margin=1000,
                                               label=self.target.label)

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
        sep_constraints = self.set_constraints(flux_log10_threshold=flux_log10_threshold)
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
            [dra_median / np.cos(self.target_radec_position_after_pm.dec.radian), ddec_median]) * u.arcsec
        self.my_logger.info(f"\n\tShift original CRVAL value {self.wcs.wcs.crval} of {total_shift}.")
        self.wcs.wcs.crval = self.wcs.wcs.crval * u.deg + total_shift
        # if parameters.DEBUG:
        #     self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_radec_positions_after_pm,
        #                                        quad=np.array(self.quad_stars_pixel_positions).T,
        #                                        margin=10, label=self.target.label)

        # Now, write out the WCS object as a FITS header
        dra_median = np.median(dra.to(u.arcsec).value)
        ddec_median = np.median(ddec.to(u.arcsec).value)
        dra_rms = np.std(dra.to(u.arcsec).value)
        ddec_rms = np.std(ddec.to(u.arcsec).value)
        hdu = fits.open(self.wcs_file_name)
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
        sep_constraints = self.set_constraints(flux_log10_threshold=flux_log10_threshold)
        sources_selection = self.sources_radec_positions[sep_constraints]
        self.gaia_matches = self.gaia_radec_positions_after_pm[self.gaia_index[sep_constraints]]
        dra, ddec = sources_selection.spherical_offsets_to(self.gaia_matches)

        # update values
        self.my_logger.info(f"\n\tUpdate WCS solution from {self.wcs_file_name} with Gaia solution.")
        dra_median = np.median(dra.to(u.arcsec).value)
        ddec_median = np.median(ddec.to(u.arcsec).value)
        dra_rms = np.std(dra.to(u.arcsec).value)
        ddec_rms = np.std(ddec.to(u.arcsec).value)
        hdu = fits.open(self.wcs_file_name)
        hdu[0].header['CRV1_MED'] = dra_median
        hdu[0].header['CRV2_MED'] = ddec_median
        hdu[0].header['CRV1_RMS'] = dra_rms
        hdu[0].header['CRV2_RMS'] = ddec_rms
        hdu.writeto(self.wcs_file_name, overwrite=True)

        # if parameters.DEBUG:
        #     plot_shifts_histograms(dra, ddec)
        #     self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_radec_positions_after_pm,
        #                                        quad=np.array(self.quad_stars_pixel_positions).T,
        #                                        margin=10, label=self.target.label)
        return dra, ddec

    def run_full_astrometry(self, extent=None, maxiter=20):
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
        >>> logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
        >>> file_names = ['./tests/data/reduc_20170530_134.fits']
        >>> if os.path.isfile('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs'):
        ...     os.remove('./tests/data/reduc_20170530_134_wcs/reduc_20170530_134.wcs')
        >>> for file_name in file_names:
        ...     tag = file_name.split('/')[-1]
        ...     disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        ...     if target is None or xpos is None or ypos is None:
        ...         continue
        ...     a = Astrometry(file_name, target, disperser_label)
        ...     extent = ((max(0, xpos - radius), min(xpos + radius, parameters.CCD_IMSIZE)),
        ...               (max(0, ypos - radius), min(ypos + radius, parameters.CCD_IMSIZE)))
        ...     gaia_min_residuals = a.run_full_astrometry(extent=extent, maxiter=maxiter)

        .. doctest:
            :hide:

            >>> assert os.path.isdir(a.output_directory)
            >>> assert os.path.isfile(set_wcs_file_name(file_name))
            >>> assert a.data is not None
            >>> assert np.sum(a.data) > 1e-10
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
                    dra, ddec = self.run_gaia_astrometry()
                    dra_median = np.median(dra.to(u.mas).value)
                    ddec_median = np.median(ddec.to(u.mas).value)
                    if np.abs(dra_median) < 1 and np.abs(ddec_median) < 1:
                        break
                sources_list.append(deepcopy(self.sources))
                # check the positions of quad stars with their WCS position from Gaia catalog
                gaia_residuals = self.compute_gaia_pixel_residuals()
                gaia_residuals_sum_x, gaia_residuals_sum_y = np.sum(np.abs(gaia_residuals), axis=0)
                gaia_residuals_quad_sum = np.sum(np.sqrt(np.sum(gaia_residuals ** 2, axis=1)))
                if parameters.DEBUG:
                    self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_matches, margin=20,
                                                       quad=self.quad_stars_pixel_positions,
                                                       label=self.target.label)
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
        best_iter = int(np.argmin(t["gaia_residuals_quad_sum"]))
        self.my_logger.info(f'\n\tBest run: iteration #{best_iter}')
        self.run_simple_astrometry(extent=extent, sources=sources_list[best_iter])
        self.run_gaia_astrometry()
        self.my_logger.info(f'\n\tFinal target position: {self.get_target_pixel_position()}')
        if parameters.DEBUG:
            self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_matches, margin=20,
                                               quad=self.quad_stars_pixel_positions, label="FINALLLL")
            self.plot_quad_stars()
        return np.min(t["gaia_residuals_quad_sum"])


if __name__ == "__main__":
    import doctest

    doctest.testmod()

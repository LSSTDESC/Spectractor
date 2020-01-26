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
    from astroquery.gaia import Gaia
    job = Gaia.cone_search_async(coord, radius=radius)
    my_logger = set_logger("load_gaia_catalog")
    my_logger.debug(f"\n\t{job}")
    gaia_catalog = job.get_results()
    my_logger.debug(f"\n\t{gaia_catalog}")
    gaia_catalog.fill_value = 0
    gaia_catalog['parallax'].fill_value = np.min(gaia_catalog['parallax'])
    return gaia_catalog


def get_gaia_coords_after_proper_motion(gaia_catalog, date_obs):
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

    def __init__(self, file_name, target="", disperser_label="", wcs_file_name="", output_directory=""):
        """
        Parameters
        ----------
        file_name: str
            Input file name of the image to analyse.
        target: str
            The name of the targeted object.
        disperser_label: str
            The name of the disperser (default: "").
        """
        Image.__init__(self, file_name, target=target, disperser_label=disperser_label)
        self.my_logger = set_logger(self.__class__.__name__)
        self.output_directory = set_wcs_output_directory(file_name, output_directory=output_directory)
        ensure_dir(self.output_directory)
        self.tag = set_wcs_tag(file_name)
        self.new_file_name = self.file_name.replace('.fits', '_new.fits')
        self.sources_file_name = set_sources_file_name(file_name, output_directory=output_directory)
        self.wcs_file_name = wcs_file_name
        self.log_file_name = os.path.join(self.output_directory, self.tag) + ".log"
        if self.wcs_file_name != "":
            self.wcs = load_wcs_from_file(self.wcs_file_name)
        else:
            self.wcs_file_name = set_wcs_file_name(file_name, output_directory=output_directory)
            if os.path.isfile(self.wcs_file_name):
                self.wcs = load_wcs_from_file(self.wcs_file_name)
        self.gaia_file_name = set_gaia_catalog_file_name(file_name, output_directory=output_directory)
        self.my_logger.info(f"\n\tIntermediate outputs will be stored in {self.output_directory}")
        self.wcs = None
        self.sources = None
        self.sources_radec_positions = None
        self.gaia_catalog = None
        self.gaia_index = None
        self.gaia_matches = None
        self.gaia_residuals = None
        self.gaia_radec_positions_after_pm = None
        self.dist_2d = None
        self.quad_stars_pixel_positions = None
        self.dist_ra = 0 * u.arcsec
        self.dist_dec = 0 * u.arcsec
        self.target_radec_position_after_pm = self.target.get_radec_position_after_pm(date_obs=self.date_obs)
        if os.path.isfile(self.log_file_name):
            self.quad_stars_pixel_positions = self.get_quad_stars_pixel_positions()

    def get_target_pixel_position(self):
        target_x, target_y = self.wcs.all_world2pix(self.target_radec_position_after_pm.ra,
                                                    self.target_radec_position_after_pm.dec, 0)
        return target_x, target_y

    def get_gaia_pixel_positions(self, gaia_index=None):
        if gaia_index is None:
            gaia_x, gaia_y = self.wcs.all_world2pix(self.gaia_radec_positions_after_pm.ra,
                                                    self.gaia_radec_positions_after_pm.dec, 0, quiet=True)
        else:
            gaia_x, gaia_y = self.wcs.all_world2pix(self.gaia_radec_positions_after_pm[gaia_index].ra,
                                                    self.gaia_radec_positions_after_pm[gaia_index].dec, 0, quiet=True)
        return gaia_x, gaia_y

    def get_quad_stars_pixel_positions(self):
        coords = []
        f = open(self.log_file_name, 'r')
        for line in f:
            if 'field_xy' in line:
                coord = line.split(' ')[5].split(',')
                coords.append([float(coord[0]), float(coord[1])])
        f.close()
        coords = np.array(coords)
        if len(coords) < 4:
            self.my_logger.warning(f"\n\tOnly {len(coords)} calibration stars has been extracted from "
                                   f"{self.log_file_name}, with positions {coords}. "
                                   f"A quad of at least 4 stars is expected. "
                                   f"Please check {self.log_file_name}.")
        self.quad_stars_pixel_positions = coords
        return coords

    def write_sources(self):
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
        if center is None:
            target_x, target_y = self.get_target_pixel_position()
        else:
            target_x, target_y = center
        ax.scatter(target_x, target_y, s=300, marker="+", edgecolor='cyan', facecolor='cyan', label=f"{label}", lw=2)
        if gaia_coord is not None:
            gaia_x, gaia_y = self.get_gaia_pixel_positions()
            ax.scatter(gaia_x, gaia_y, s=300, marker="+", edgecolor='blue', facecolor='blue', label=f"Gaia stars", lw=2)
        if quad is not None:
            points = np.concatenate([quad, [quad[-1]]])
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], 'r-')
            ax.plot(points.T[0], points.T[1], 'rx', lw=2)
        ax.legend()
        ax.set_xlim(max(0, target_x - margin), min(target_x + margin, self.data.shape[1]))
        ax.set_ylim(max(0, target_y - margin), min(target_y + margin, self.data.shape[0]))
        if not no_plot and parameters.DISPLAY:
            # fig.tight_layout()
            plt.show()

    def get_sources_radec_positions(self):
        sources_coord = self.wcs.all_pix2world(self.sources['xcentroid'], self.sources['ycentroid'], 0)
        self.sources_radec_positions = SkyCoord(ra=sources_coord[0] * u.deg, dec=sources_coord[1] * u.deg,
                                                frame="icrs", obstime=self.date_obs, equinox="J2000")
        return self.sources_radec_positions

    def shift_wcs_center_fit_gaia_catalog(self, gaia_coord):
        gaia_index, dist_2d, dist_3d = self.sources_radec_positions.match_to_catalog_sky(gaia_coord)
        matches = gaia_coord[gaia_index]
        dist_ra, dist_dec = self.sources_radec_positions.spherical_offsets_to(matches)
        return gaia_index, dist_2d, dist_ra, dist_dec

    def plot_astrometry_shifts(self, vmax=3, margin=parameters.CCD_IMSIZE, gaia_index=None):
        target_x, target_y = self.get_target_pixel_position()
        gaia_x, gaia_y = self.get_gaia_pixel_positions(gaia_index=gaia_index)

        fig = plt.figure(figsize=(6, 8))

        fig.add_subplot(211, projection=self.wcs)

        plot_image_simple(plt.gca(), self.data, scale="log10")
        plt.xlabel('RA')
        plt.ylabel('Dec')
        if self.sources is not None:
            plt.scatter(self.sources['xcentroid'], self.sources['ycentroid'], s=100, lw=2,
                        edgecolor='black', facecolor='none', label="all sources")
        sc = plt.scatter(gaia_x, gaia_y, s=100, c=self.dist_ra.to(u.arcsec).value,
                         cmap="bwr", vmin=-vmax, vmax=vmax,
                         label=f"gaia stars after motion", lw=1)
        plt.xlim(max(0, target_x - margin), min(target_x + margin, self.data.shape[1]))
        plt.ylim(max(0, target_y - margin), min(target_y + margin, self.data.shape[0]))
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
                         label=f"gaia stars after motion", lw=1)
        plt.xlim(max(0, target_x - margin), min(target_x + margin, self.data.shape[1]))
        plt.ylim(max(0, target_y - margin), min(target_y + margin, self.data.shape[0]))
        plt.colorbar(sc, label="Shift in DEC [arcsec]")
        plt.legend()
        if parameters.DISPLAY:
            # fig.tight_layout()
            plt.show()

    def set_constraints(self, min_stars=100, flux_log10_threshold=0.1, min_range=3 * u.arcsec, max_range=5 * u.arcmin,
                        max_sep=1 * u.arcsec):
        sep = self.dist_2d < max_sep
        sep *= self.sources_radec_positions.separation(self.target_radec_position_after_pm) < max_range
        sep *= self.sources_radec_positions.separation(self.target_radec_position_after_pm) > min_range
        sep *= np.log10(self.sources['flux']) > flux_log10_threshold
        if np.sum(sep) > min_stars:
            for r in np.arange(0, max_range.value, 0.1)[::-1]:
                range_constraint = self.sources_radec_positions.separation(self.target.radec_position_after_pm)\
                                   < r * u.arcmin
                if np.sum(sep * range_constraint) < min_stars:
                    break
                else:
                    sep *= range_constraint
        return sep

    def plot_shifts_profiles(self, matches, dra, ddec):
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
        command = f"{os.path.join(parameters.ASTROMETRYNET_DIR, 'bin/new-wcs')} -v -d -i {self.file_name} " \
                  f"-w {self.wcs_file_name} -o {self.new_file_name}\n"
        self.my_logger.info(f'\n\tSave WCS in original file:\n\t{command}')
        log = subprocess.check_output(command, shell=True)
        if log_file is not None:
            log_file.write(command + "\n")
            log_file.write(log.decode("utf-8") + "\n")

    def compute_gaia_pixel_residuals(self):
        coords = self.get_quad_stars_pixel_positions().T
        ra, dec = self.wcs.all_pix2world(coords[0], coords[1], 0)
        coords_radec = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs', equinox="J2000")
        gaia_index, dist_2d, dist_3d = coords_radec.match_to_catalog_sky(self.gaia_radec_positions_after_pm)
        matches = self.gaia_radec_positions_after_pm[gaia_index]
        gaia_coords = np.array(self.wcs.all_world2pix(matches.ra, matches.dec, 0))
        self.gaia_residuals = (gaia_coords - coords).T
        return self.gaia_residuals

    def find_quad_star_index_in_sources(self, quad_star):
        eps = 1e-1
        k = -1
        for k in range(len(self.sources)):
            if abs(self.sources['xcentroid'][k] - quad_star[0]) < eps \
                    and abs(self.sources['ycentroid'][k] - quad_star[1]):
                break
        return k

    def remove_worst_quad_star_from_sources(self):
        gaia_pixel_residuals = self.compute_gaia_pixel_residuals()
        distances = np.sqrt(np.sum(gaia_pixel_residuals ** 2, axis=1))
        max_residual_index = np.argmax(distances)
        worst_source_index = self.find_quad_star_index_in_sources(self.quad_stars_pixel_positions[max_residual_index])
        self.my_logger.debug(f"\n\tRemove source #{worst_source_index}\n{self.sources[max_residual_index]}")
        self.sources.remove_row(worst_source_index)

    def plot_quad_stars(self, margin=10, scale="log10"):
        nquads = len(self.quad_stars_pixel_positions)
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
        variable. A new WCS is created and saved in as a FITS file.
        The intermediate results are saved in a new directory named as the FITS file name with a _wcs suffix.

        Parameters
        ----------
        extent: 2-tuple
            ((xmin,xmax),(ymin,ymax)) 2 dimensional typle to crop the exposure before any operation (default: None).
        sources: Table
            List of sources. If None, then source detection is run on the image (default: None).

        Notes
        -----
        The source file given to solve-field is understood as a FITS file with pixel origin value at 1,
        whereas pixel coordinates comes from photutils using a numpy convention with pixel origin value at 0.
        To correct for this we shift the CRPIX center of 1 pixel at the end of the function. It can be that solve-field
        using the source list or the raw FITS image then give the same WCS values.

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
        if parameters.DEBUG:
            self.plot_image(scale="log10")
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

        Parameters
        ----------

        Examples
        --------

        >>> import os
        >>> from spectractor.logbook import LogBook
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
            self.my_logger.info(f"\n\tLoad source positions and flux from {self.sources_file_name}")
            sources = Table.read(self.sources_file_name)
            sources['X'].name = "xcentroid"
            sources['Y'].name = "ycentroid"
            sources['FLUX'].name = "flux"
            self.sources = sources

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
                radius = np.sqrt(2)*max(np.max(self.sources["xcentroid"])-np.min(self.sources["xcentroid"]),
                                        np.max(self.sources["ycentroid"])-np.min(self.sources["ycentroid"]))
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
            self.shift_wcs_center_fit_gaia_catalog(self.gaia_radec_positions_after_pm)
        if parameters.DEBUG:
            self.plot_astrometry_shifts(vmax=3, gaia_index=self.gaia_index)

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
        if parameters.DEBUG:
            self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_radec_positions_after_pm,
                                               quad=self.quad_stars_pixel_positions, margin=30, label=self.target.label)

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
            self.shift_wcs_center_fit_gaia_catalog(self.gaia_radec_positions_after_pm)
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

        if parameters.DEBUG:
            plot_shifts_histograms(dra, ddec)
        return dra, ddec

    def run_full_astrometry(self, extent=None, maxiter=20):
        t = Table(names=["target_x", "target_y", "gaia_residuals_sum_x", "gaia_residuals_sum_y", "gaia_residuals_mean"])
        for c in t.columns:
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
                gaia_residuals_mean = np.sum(np.sqrt(np.sum(gaia_residuals**2, axis=1)))
                if parameters.DEBUG:
                    self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_matches, margin=20,
                                                       quad=self.quad_stars_pixel_positions, label=self.target.label)
                    self.plot_astrometry_shifts(vmax=3, gaia_index=self.gaia_index)
                    self.plot_quad_stars()
                target_x, target_y = self.get_target_pixel_position()
                t.add_row([target_x, target_y, gaia_residuals_sum_x, gaia_residuals_sum_y, gaia_residuals_mean])
                self.remove_worst_quad_star_from_sources()
            except FileNotFoundError:
                k -= 1
                self.my_logger.warning(f"\n\tAstrometry.net failed at iteration {k}. "
                                       f"Stop the loop here and look for best solution.")
                break
        t.pprint_all()
        best_iter = int(np.argmin(t["gaia_residuals_mean"]))
        self.my_logger.info(f'\n\tBest run: iteration #{best_iter}')
        self.run_simple_astrometry(extent=extent, sources=sources_list[best_iter])
        self.run_gaia_astrometry()
        self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_matches, margin=20,
                                           quad=self.quad_stars_pixel_positions, label=self.target.label)
        self.plot_quad_stars()
        return np.min(t["gaia_residuals_mean"])

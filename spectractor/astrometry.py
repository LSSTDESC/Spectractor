import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.io import fits, ascii
import astropy.units as u
from astropy.table import Table
from astropy import wcs as WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, Distance

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
    data_wo_bkg -= np.min(data_wo_bkg)
    if parameters.DEBUG:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(bkg.background, origin='lower')
        ax[1].imshow(np.log10(1 + data_wo_bkg), origin='lower')
        plt.show()
    return data_wo_bkg


def source_detection(data_wo_bkg, sigma=3.0, fwhm=3.0, threshold_std_factor=5):
    mean, median, std = sigma_clipped_stats(data_wo_bkg, sigma=sigma)
    mask = np.zeros(data_wo_bkg.shape, dtype=bool)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_std_factor * std)
    sources = daofind(data_wo_bkg - median, mask=mask)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
    sources.sort('mag')
    if parameters.DEBUG or True:
        positions = np.array((sources['xcentroid'], sources['ycentroid']))
        fig = plt.figure(figsize=(8, 8))
        plot_image_simple(plt.gca(), data_wo_bkg, scale="log10", target_pixcoords=positions)
        fig.tight_layout()
        plt.show()
    return sources


def load_gaia_catalog(target, radius=10 * u.arcmin):
    from astroquery.gaia import Gaia
    job = Gaia.cone_search_async(target.coord,
                                 radius=0.5 * parameters.CCD_IMSIZE * parameters.CCD_PIXEL2ARCSEC * u.arcsec)
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
    return target_coord_after_proper_motion


def update_gaia_catalog_with_proper_motion(gaia_catalog, date_obs):
    parallax = np.array(gaia_catalog['parallax'].filled(np.nanmin(np.abs(gaia_catalog['parallax']))))
    parallax[parallax < 0] = np.min(parallax[parallax > 0])
    gaia_stars = SkyCoord(ra=gaia_catalog['ra'], dec=gaia_catalog['dec'], frame='icrs', equinox="J2000",
                          obstime=Time(np.array(gaia_catalog['ref_epoch']), format='decimalyear'),
                          pm_ra_cosdec=gaia_catalog['pmra'].filled(0) * np.cos(
                              np.array(gaia_catalog['dec']) * np.pi / 180),
                          pm_dec=gaia_catalog['pmdec'].filled(0),
                          distance=Distance(parallax=parallax * u.mas, allow_negative=True))
    gaia_stars_after_proper_motion = gaia_stars.apply_space_motion(new_obstime=Time(date_obs))
    return gaia_stars_after_proper_motion


class Astrometry(Image):

    def __init__(self, file_name, target="", disperser_label="", wcs_file_name=""):
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
        self.output_directory = ""
        self.tag = file_name
        self.set_wcs_output_directory()
        self.set_file_tag()
        self.new_file_name = self.file_name.replace('.fits', '_new.fits')
        self.output_sources_fitsfile = os.path.join(self.output_directory, f"{self.tag}_sources.fits")
        self.wcs_file_name = ""
        if self.wcs_file_name != "":
            self.wcs = load_wcs_from_file(self.wcs_file_name)
        else:
            self.wcs_file_name = os.path.join(self.output_directory, self.tag + '.wcs')
        self.gaia_file_name = os.path.join(self.output_directory, f"{self.tag}_gaia.ecsv")
        self.my_logger.info(f"\n\tIntermediate outputs will be stored in {self.output_directory}")
        self.wcs = None
        self.sources = None
        self.sources_coord = None
        self.gaia_catalog = None
        self.gaia_index = None
        self.gaia_coord_after_motion = None
        self.dist_2d = None
        self.dist_ra = 0 * u.arcsec
        self.dist_dec = 0 * u.arcsec
        self.target_coord_after_motion = update_target_coord_with_proper_motion(self.target, date_obs=self.date_obs)

    def set_wcs_output_directory(self):
        output_directory = os.path.join(os.path.dirname(self.file_name),
                                        os.path.splitext(os.path.basename(self.file_name))[0]) + "_wcs"
        ensure_dir(output_directory)
        self.output_directory = output_directory
        return output_directory

    def set_file_tag(self):
        tag = os.path.splitext(os.path.basename(self.file_name))[0]
        self.tag = tag
        return tag

    def write_sources(self):
        colx = fits.Column(name='X', format='D', array=self.sources['xcentroid'])
        coly = fits.Column(name='Y', format='D', array=self.sources['ycentroid'])
        colflux = fits.Column(name='FLUX', format='D', array=self.sources['flux'])
        coldefs = fits.ColDefs([colx, coly, colflux])
        hdu = fits.BinTableHDU.from_columns(coldefs)
        hdu.header['IMAGEW'] = self.data.shape[1]
        hdu.header['IMAGEH'] = self.data.shape[0]
        hdu.writeto(self.output_sources_fitsfile, overwrite=True)
        self.my_logger.info(f'\n\tSources positions saved in {self.output_sources_fitsfile}')

    def plot_sources_and_gaia_catalog(self, wcs=None, sources=None, gaia_coord=None, margin=10000):
        fig = plt.figure(figsize=(8, 8))

        if wcs is None:
            wcs = self.wcs
            if wcs is not None:
                fig.add_subplot(111, projection=wcs)

        plot_image_simple(plt.gca(), self.data, scale="log10")
        if wcs is not None:
            plt.xlabel('RA')
            plt.ylabel('Dec')
        if sources is not None:
            plt.scatter(sources['xcentroid'], sources['ycentroid'], s=300, lw=2,
                        edgecolor='black', facecolor='none', label="all detected sources")
        target_x, target_y = wcs.all_world2pix(self.target_coord_after_motion.ra, self.target_coord_after_motion.dec, 0)
        plt.scatter(target_x, target_y, s=300, marker="+",
                    edgecolor='cyan', facecolor='cyan', label=f"the target {self.target.label} after motion", lw=2)
        if gaia_coord is not None:
            gaia_x, gaia_y = wcs.all_world2pix(gaia_coord.ra, gaia_coord.dec, 0, maxiter=50, quiet=True)
            plt.scatter(gaia_x, gaia_y, s=300, marker="+",
                        edgecolor='blue', facecolor='blue', label=f"gaia stars after motion", lw=2)
        plt.legend()
        plt.xlim(max(0, target_x - margin), min(target_x + margin, self.data.shape[1]))
        plt.ylim(max(0, target_y - margin), min(target_y + margin, self.data.shape[0]))
        # fig.tight_layout()
        plt.show()

    def set_sources_coord(self):
        sources_coord = self.wcs.all_pix2world(self.sources['xcentroid'], self.sources['ycentroid'], 0)
        self.sources_coord = SkyCoord(ra=sources_coord[0] * u.deg, dec=sources_coord[1] * u.deg,
                                      frame="icrs", obstime=self.date_obs, equinox="J2000")
        return self.sources_coord

    def shift_wcs_center_fit_gaia_catalog(self, gaia_coord):
        gaia_index, self.dist_2d, dist_3d = self.sources_coord.match_to_catalog_sky(gaia_coord)
        matches = gaia_coord[gaia_index]
        dist_ra, dist_dec = self.sources_coord.spherical_offsets_to(matches)
        return gaia_index, dist_ra, dist_dec

    def plot_astrometry_shifts(self, vmax=3):
        target_x, target_y = self.wcs.all_world2pix(self.target_coord_after_motion.ra,
                                                    self.target_coord_after_motion.dec, 0)
        gaia_x, gaia_y = self.wcs.all_world2pix(self.gaia_coord_after_motion[self.gaia_index].ra,
                                                self.gaia_coord_after_motion[self.gaia_index].dec,
                                                0, maxiter=50, quiet=True)

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
        margin = 1000
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
        # fig.tight_layout()
        plt.show()

    def set_constraints(self, min_stars=50, flux_log10_threshold=0.1, min_range=3 * u.arcsec, max_range=5 * u.arcmin,
                        max_sep=1 * u.arcsec):
        sep = self.dist_2d < max_sep
        sep *= np.log10(self.sources['flux']) > flux_log10_threshold
        sep *= self.sources_coord.separation(self.target_coord_after_motion) < max_range
        sep *= self.sources_coord.separation(self.target_coord_after_motion) > min_range
        if np.sum(sep) > min_stars:
            for r in np.arange(0, max_range.value, 0.1)[::-1]:
                range_constraint = self.sources_coord.separation(self.target.coord) < r * u.arcmin
                if np.sum(sep * range_constraint) < min_stars:
                    break
                else:
                    sep *= range_constraint
        return sep

    def plot_shifts_histograms(self, dra, ddec):
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
        plt.show()

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
        ax[0].axvline(self.target_coord_after_motion.ra.value, color='k', linestyle="--",
                      label=f"{self.target.label} RA")
        ax[1].axvline(self.target_coord_after_motion.dec.value, color='k', linestyle="--",
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
        plt.show()

    def merge_wcs_with_new_exposure(self, log_file=None):
        command = f"{os.path.join(parameters.ASTROMETRYNET_BINDIR, 'new-wcs')} -v -d -i {self.file_name} " \
                  f"-w {self.wcs_file_name} -o {self.new_file_name}\n"
        # f"mv {new_file_name} {file_name}"
        self.my_logger.info(f'\n\tSave WCS in original file:\n\t{command}')
        log = subprocess.check_output(command, shell=True)
        if log_file is not None:
            log_file.write(command + "\n")
            log_file.write(log.decode("utf-8") + "\n")

    def run_simple_astrometry(self, extent=None):
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
        extent: 2-tuple
            ((xmin,xmax),(ymin,ymax)) 2 dimensional typle to crop the exposure before any operation (default: None).

        Examples
        --------

        >>> import os
        >>> from spectractor.logbook import LogBook
        >>> from spectractor.astrometry import Astrometry
        >>> from spectractor import parameters
        >>> parameters.VERBOSE = True
        >>> logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
        >>> file_names = ['./tests/data/reduc_20170530_134.fits']
        >>> file_names = ['./tests/data/reduc_20170605_028.fits']
        >>> for file_name in file_names:
        ...     tag = file_name.split('/')[-1]
        ...     disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        ...     if target is None or xpos is None or ypos is None:
        ...         continue
        ...     a = Astrometry(file_name, target, disperser_label)
        ...     a.run_simple_astrometry(extent=((300,1400),(300,1400)))
        ...     assert os.path.isdir('./tests/data/reduc_20170530_134_wcs')

        """
        # crop data
        if extent is not None:
            data = self.data[extent[1][0]:extent[1][1], extent[0][0]:extent[0][1]]
        else:
            data = np.copy(self.data)
        if parameters.DEBUG:
            self.plot_image(scale="log10")
        # remove background
        self.my_logger.info('\n\tRemove background using astropy SExtractorBackground()...')
        data_wo_bkg = remove_background(data)
        # extract source positions and fluxes
        self.my_logger.info('\n\tDetect sources using photutils source_detection()...')
        self.sources = source_detection(data_wo_bkg)
        if extent is not None:
            self.sources['xcentroid'] += extent[0][0]
            self.sources['ycentroid'] += extent[1][0]
        self.my_logger.warning(f'\n\t{self.sources}')

        # write results in fits file
        self.write_sources()
        # run astrometry.net
        command = f"{os.path.join(parameters.ASTROMETRYNET_BINDIR, 'solve-field')} --scale-unit arcsecperpix " \
                  f"--scale-low {0.95 * parameters.CCD_PIXEL2ARCSEC} " \
                  f"--scale-high {1.05 * parameters.CCD_PIXEL2ARCSEC} " \
                  f"--ra {self.target.coord.ra.value} --dec {self.target.coord.dec.value} " \
                  f"--radius {parameters.CCD_IMSIZE * parameters.CCD_PIXEL2ARCSEC / 3600.} " \
                  f"--dir {self.output_directory} --out {self.tag} " \
                  f"--overwrite --x-column X --y-column Y {self.output_sources_fitsfile}"
        self.my_logger.info(f'\n\tRun astrometry.net solve_field command:\n\t{command}')
        log = subprocess.check_output(command, shell=True)
        log_file = open(f"{os.path.join(self.output_directory, self.tag)}.log", "w+")
        log_file.write(command + "\n")
        log_file.write(log.decode("utf-8") + "\n")
        # save new WCS in original fits file
        self.merge_wcs_with_new_exposure(log_file=log_file)
        log_file.close()
        # load WCS
        self.wcs = load_wcs_from_file(self.new_file_name)
        return self.wcs

    # noinspection PyUnresolvedReferences
    def run_gaia_astrometry(self):
        """Refine a World Coordinate System (WCS) using Gaia satellite astrometry catalog.

        A WCS must be already present in the exposure FITS file.

        The name of the target must be given to get its RA,DEC coordinates via a Simbad query.
        A matching is performed between the detected sources and the Gaia catalog obtained for the region of the target.
        Then the closest and brightest sources are selected and the WCS is shifted by the median of the distance between
        these stars and the detected sources.
        A new WCS is created and merged with the given exposure.

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
        >>> file_names = ['./tests/data/reduc_20170605_028.fits']
        >>> for file_name in file_names:
        ...     tag = file_name.split('/')[-1]
        ...     disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        ...     if target is None or xpos is None or ypos is None:
        ...         continue
        ...     a = Astrometry(file_name, target, disperser_label)
        ...     a.run_gaia_astrometry()
        ...     assert os.path.isdir('./tests/data/reduc_20170530_134_wcs')

        """
        # load detected sources
        if self.sources is None or True:
            self.my_logger.info(f"\n\tLoad source positions and flux from {self.output_sources_fitsfile}")
            sources = Table.read(self.output_sources_fitsfile)
            sources['X'].name = "xcentroid"
            sources['Y'].name = "ycentroid"
            sources['FLUX'].name = "flux"
            self.sources = sources
            self.my_logger.debug(f"\n\t{self.sources}")

        # load WCS if absent
        if self.wcs is None:
            self.wcs = load_wcs_from_file(self.new_file_name)
        self.sources_coord = self.set_sources_coord()

        # load the Gaia catalog
        if os.path.isfile(self.gaia_file_name):
            self.my_logger.info(f"\n\tLoad Gaia catalog from {self.gaia_file_name}.")
            self.gaia_catalog = ascii.read(self.gaia_file_name, format="ecsv")
        else:
            self.my_logger.info(f"\n\tLoading Gaia catalog from TAP query...")
            self.gaia_catalog = load_gaia_catalog(self.target)
            ascii.write(self.gaia_catalog, self.gaia_file_name, format='ecsv', overwrite=True)
        self.my_logger.info(f"\n\tGaia catalog loaded.")

        # update coordinates with proper motion data
        self.my_logger.info(f"\n\tUpdate object coordinates with proper motion at time={self.date_obs}.")
        self.target_coord_after_motion = update_target_coord_with_proper_motion(self.target, self.date_obs)
        self.gaia_coord_after_motion = update_gaia_catalog_with_proper_motion(self.gaia_catalog, self.date_obs)
        if parameters.DEBUG:
            self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_coord_after_motion)

        # compute shifts
        self.gaia_index, self.dist_ra, self.dist_dec = \
            self.shift_wcs_center_fit_gaia_catalog(self.gaia_coord_after_motion)
        if parameters.DEBUG:
            self.plot_astrometry_shifts(vmax=3)

        # select the brightest and closest stars with maximum shift
        flux_log10_threshold = np.log10(self.sources['flux'][int(0.5 * len(self.sources))])
        sep_constraints = self.set_constraints(flux_log10_threshold=flux_log10_threshold)
        sources_selection = self.sources_coord[sep_constraints]
        gaia_matches = self.gaia_coord_after_motion[self.gaia_index[sep_constraints]]
        dra, ddec = sources_selection.spherical_offsets_to(gaia_matches)

        # compute statistics
        dra_median = np.median(dra.to(u.arcsec).value)
        ddec_median = np.median(ddec.to(u.arcsec).value)
        if parameters.DEBUG:
            self.plot_shifts_histograms(dra, ddec)
            self.plot_shifts_profiles(gaia_matches, dra, ddec)

        # update WCS
        # tested with high latitude 20170530_120.fits exposure: dra shift must be divided by cos(dec)
        # to set new WCS system because spherical_offsets_to gives shifts angle at equator
        # (see https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.spherical_offsets_to)
        # after the shift the histograms must be centered on zero
        self.wcs.wcs.crval = self.wcs.wcs.crval * u.deg + \
                             np.array([dra_median / np.cos(self.target_coord_after_motion.dec * np.pi / 180),
                                       ddec_median]) * u.arcsec
        if parameters.DEBUG:
            self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_coord_after_motion, margin=30)

        # Now, write out the WCS object as a FITS header
        hdu = fits.open(self.new_file_name)
        hdu[0].header['CRVAL1'] = self.wcs.wcs.crval[0]
        hdu[0].header['CRVAL2'] = self.wcs.wcs.crval[1]
        hdu.writeto(self.new_file_name, overwrite=True)

        # check histogram medians
        self.wcs = load_wcs_from_file(self.new_file_name)
        self.set_sources_coord()
        self.gaia_index, self.dist_ra, self.dist_dec = \
            self.shift_wcs_center_fit_gaia_catalog(self.gaia_coord_after_motion)
        sep_constraints = self.set_constraints(flux_log10_threshold=flux_log10_threshold)
        sources_selection = self.sources_coord[sep_constraints]
        self.gaia_matches = self.gaia_coord_after_motion[self.gaia_index[sep_constraints]]
        dra, ddec = sources_selection.spherical_offsets_to(self.gaia_matches)
        dra_median = np.median(dra.to(u.arcsec).value)
        ddec_median = np.median(ddec.to(u.arcsec).value)

        if parameters.DEBUG or True:
            # self.plot_sources_and_gaia_catalog(sources=self.sources, gaia_coord=self.gaia_coord_after_motion, margin=30)
            # self.plot_astrometry_shifts(vmax=3)
            self.plot_shifts_histograms(dra, ddec)
        return dra_median, ddec_median

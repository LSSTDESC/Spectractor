from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
from astropy.time import Time

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import numpy as np
import re

from astropy.io import fits

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.extractor.spectroscopy import (Lines, HGAR_LINES, HYDROGEN_LINES, ATMOSPHERIC_LINES,
                                                ISM_LINES, STELLAR_LINES)

if os.getenv("PYSYN_CDBS"):
    import pysynphot as S


def GetSimbadName(target_name):
    """
    Try to find the simbad target name from the provided target_name
    :param target_name: target name provided by the caller
    :return: existing target name in simbad
    """

    object_name = (target_name).upper()
    simbad_object_name = object_name

    # I was not able to find these entries in simbad database
    if object_name in ["1732526","1740346","1743045","1757132","1802271","1805292","1808347","1812095","1812524","2M0036+18","2M0559-14","AGK+81D266",
                   "BD02D3375","BD17D4708","BD21D0607","BD26D2606","BD29D2091","BD54D1216","BD60D1753","BD75","ETAUMA","GRW+70D5824","HS2027","KF01T5",
                  "KF06T1","KF06T2","KF08T3","KSI2CETI","P041C","P177D","P330E","SF1615001A","SF1615+001A","SNAP-1","SNAP-2","SUN_REFERENCE","WD0947_857","WD1026_453",
                   "HZ43B","DELUMI","SDSS132811","SDSSJ151421","WD-0308-565","C26202",
                  "WD0320_539"]:
        print(">>>>> SKIP TARGET {} ".format(object_name))
        return None


    # for some special target name requiring a particular format in simbad

    if object_name == "LAMLEP":
        simbad_object_name = "LAM LEP"
    elif (object_name == "MUCOL") or (object_name == "MU COL") or (object_name == "MU. COL") or (
            object_name == "MU.COL"):
        simbad_object_name = "mu. Col"
    elif object_name == 'ETA1DOR' or object_name == "ETADOR" or object_name == "ETA DOR":
        simbad_object_name = "ETA1 DOR"
    elif object_name == 'BD11D3759':
        simbad_object_name = "BD-11 3759"
    elif object_name == 'WD0320-539':
        simbad_object_name = "WD0320-539"
    elif object_name == 'WD1327_083':
        simbad_object_name = "WD1327-083"
    elif object_name == 'GJ7541A':
        simbad_object_name = "GJ754.1A"
    elif object_name.split("-")[0] == 'NGC6681':
        simbad_object_name = "NGC 6681"
    else:
        simbad_object_name = target_name

    return simbad_object_name


def GetListOfCAMSPECFiles(thedir):
    """
    Get the list of CALSPEC files (fits file) inside the directory thedir
    - thedir : directory where are the files
    """

    all_files=os.listdir(thedir)
    sorted_files=sorted(all_files)
    selected_files=[]
    for sfile in sorted_files:
        if re.search(".*fits$",sfile):
            selected_files.append(sfile)
    return selected_files

def FilterListOfCALSPECFiles(listOfFiles):
    """
    Filter list of files:

    The filename of spectrum could come in serveral sample in the input list. Only the last version is kept

    """

    all_selected_files=[]

    current_root_fn=None  # root of filename ex hd000000
    current_fn=None    # filename of calspec ex hd000000_stis.fits

    for fn in listOfFiles:

        root_fn=fn.split("_")[0]

        if root_fn == "ngc6681":
            root_fn = fn.split("_")[0]+"_"+fn.split("_")[1]


        # special case for galaxy
        #if root_fn == "ngc6681":
        #    all_selected_files.append(fn)
        #    current_root_fn=root_fn+"_"+fn.split("_")[1]
        #    current_fn=fn
        #    continue

        if current_root_fn==None:
            current_root_fn=root_fn
            current_fn=fn
            continue

        if root_fn != current_root_fn:
            all_selected_files.append(current_fn)

        current_fn=fn
        current_root_fn=root_fn

    return all_selected_files

def maptargetnametpfilename(all_selected_calspec):
    """
    Make a dictionary target-name - filename
    :param all_selected_calspec: list of calspec filename
    :return: : dictionnary simbad - target filename : calspec filename
    """


    pysynphot_root_path = os.environ['PYSYN_CDBS']
    path_sed_calspec = os.path.join(pysynphot_root_path, 'calspec')


    dict_target_tofilename = {}


    for file in all_selected_calspec:

        if file in ["WDcovar_001.fits", "WDcovar_002.fits"]:
            print(">>>>> SKIP file {} ".format(file))
            continue

        fullfilename = os.path.join(path_sed_calspec,file)

        hdu = fits.open(fullfilename)
        img = hdu[0].data
        hd = hdu[0].header

        OBJNAME = hd["TARGETID"]

        simbad_target_name = GetSimbadName(OBJNAME)

        if simbad_target_name is not None:
            dict_target_tofilename[simbad_target_name ] = file

    return dict_target_tofilename






def load_target(label, verbose=False):
    """Load the target properties according to the type set by parameters.OBS_OBJECT_TYPE.

    Currently, the type can be either "STAR", "HG-AR" or "MONOCHROMATOR". The label parameter gives the
    name of the source and allows to load its specific properties.

    Parameters
    ----------
    label: str
        The label of the target.
    verbose: bool, optional
        If True, more verbosity (default: False).

    Examples
    --------
    >>> parameters.OBS_OBJECT_TYPE = "STAR"
    >>> t = load_target("HD111980", verbose=False)
    >>> print(t.label)
    HD111980
    >>> print(t.radec_position.dec)
    -18d31m20.009s
    >>> parameters.OBS_OBJECT_TYPE = "MONOCHROMATOR"
    >>> t = load_target("XX", verbose=False)
    >>> print(t.label)
    XX
    >>> parameters.OBS_OBJECT_TYPE = "HG-AR"
    >>> t = load_target("XX", verbose=False)
    >>> print([line.wavelength for line in t.lines.lines][:5])
    [253.652, 296.728, 302.15, 313.155, 334.148]
    """
    if parameters.OBS_OBJECT_TYPE == 'STAR':
        return Star(label, verbose)
    elif parameters.OBS_OBJECT_TYPE == 'HG-AR':
        return ArcLamp(label, verbose)
    elif parameters.OBS_OBJECT_TYPE == 'MONOCHROMATOR':
        return Monochromator(label, verbose)
    else:
        raise ValueError(f'Unknown parameters.OBS_OBJECT_TYPE: {parameters.OBS_OBJECT_TYPE}')


class Target:

    def __init__(self, label, verbose=False):
        """Initialize Target class.

        Parameters
        ----------
        label: str
            String label to name the target
        verbose: bool, optional
            Set True to increase verbosity (default: False)

        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.label = label
        self.type = None
        self.wavelengths = []
        self.spectra = []
        self.verbose = verbose
        self.emission_spectrum = False
        self.hydrogen_only = False
        self.sed = None
        self.lines = None
        self.radec_position = None
        self.radec_position_after_pm = None
        self.redshift = 0
        self.image = None
        self.image_x0 = None
        self.image_y0 = None


class ArcLamp(Target):

    def __init__(self, label, verbose=False):
        """Initialize ArcLamp class.

        Parameters
        ----------
        label: str
            String label to name the lamp.
        verbose: bool, optional
            Set True to increase verbosity (default: False)

        Examples
        --------

        Mercury-Argon lamp:

        >>> t = ArcLamp("HG-AR", verbose=False)
        >>> print([line.wavelength for line in t.lines.lines][:5])
        [253.652, 296.728, 302.15, 313.155, 334.148]
        >>> print(t.emission_spectrum)
        True

        """
        Target.__init__(self, label, verbose=verbose)
        self.my_logger = set_logger(self.__class__.__name__)
        self.emission_spectrum = True
        self.lines = Lines(HGAR_LINES, emission_spectrum=True, orders=[1, 2])

    def load(self):  # pragma: no cover
        pass


class Monochromator(Target):

    def __init__(self, label, verbose=False):
        """Initialize Monochromator class.

        Parameters
        ----------
        label: str
            String label to name the monochromator.
        verbose: bool, optional
            Set True to increase verbosity (default: False)

        Examples
        --------

        >>> t = Monochromator("XX", verbose=False)
        >>> print(t.label)
        XX
        >>> print(t.emission_spectrum)
        True

        """
        Target.__init__(self, label, verbose=verbose)
        self.my_logger = set_logger(self.__class__.__name__)
        self.emission_spectrum = True
        self.lines = Lines([], emission_spectrum=True, orders=[1, 2])

    def load(self):  # pragma: no cover
        pass


class Star(Target):

    def __init__(self, label, verbose=False):
        """Initialize Star class.

        Parameters
        ----------
        label: str
            String label to name the target
        verbose: bool, optional
            Set True to increase verbosity (default: False)

        Examples
        --------

        Emission line object:

        >>> s = Star('3C273')
        >>> print(s.label)
        3C273
        >>> print(s.radec_position.dec)
        2d03m08.598s
        >>> print(s.emission_spectrum)
        True

        Standard star:

        >>> s = Star('HD111980')
        >>> print(s.label)
        HD111980
        >>> print(s.radec_position.dec)
        -18d31m20.009s
        >>> print(s.emission_spectrum)
        False

        """
        Target.__init__(self, label, verbose=verbose)
        self.my_logger = set_logger(self.__class__.__name__)
        self.simbad = None
        self.load()

    def load(self):
        """Load the coordinates of the target.

        Examples
        --------
        >>> s = Star('3C273')
        >>> print(s.radec_position.dec)
        2d03m08.598s

        """
        # currently (pending a new release) astroquery has a race
        # condition at import time, so putting here rather than at the
        # module level so that multiple test runners don't run the race
        from astroquery.simbad import Simbad
        Simbad.add_votable_fields('flux(U)', 'flux(B)', 'flux(V)', 'flux(R)', 'flux(I)', 'flux(J)', 'sptype',
                                  'parallax', 'pm', 'z_value')

        simbad_object_name = GetSimbadName(self.label)

        print(f"target_name = {self.label}, Selected object name for Simbad : {simbad_object_name}")

        simbad = Simbad.query_object(simbad_object_name)
        print(simbad)

        self.label = simbad_object_name
        self.simbad = simbad


        if simbad is not None:
            if self.verbose or True:
                self.my_logger.info(f'\n\tSimbad:\n{simbad}')
            self.radec_position = SkyCoord(simbad['RA'][0] + ' ' + simbad['DEC'][0], unit=(u.hourangle, u.deg))
        else:
            self.my_logger.warning(f'Target {self.label} not found in Simbad')
        self.get_radec_position_after_pm(date_obs="J2000")
        if not np.ma.is_masked(simbad['Z_VALUE']):
            self.redshift = float(simbad['Z_VALUE'])
        else:
            self.redshift = 0
        self.load_spectra()

    def load_spectra(self):
        """Load reference spectra from Pysynphot database or NED database.

        If the object redshift is >0.2, the LAMBDA_MIN and LAMBDA_MAX parameters
        are redshifted accordingly.

        Examples
        --------
        >>> s = Star('3C273')
        >>> print(s.spectra[0][:4])
        [0.0000000e+00 2.5048577e-14 2.4238061e-14 2.4088789e-14]
        >>> s = Star('HD111980')
        >>> print(s.spectra[0][:4])
        [2.16890002e-13 2.66480010e-13 2.03540011e-13 2.38780004e-13]
        >>> s = Star('PKS1510-089')
        >>> print(s.redshift)
        0.36
        >>> print(f'{parameters.LAMBDA_MIN:.1f}, {parameters.LAMBDA_MAX:.1f}')
        408.0, 1496.0
        >>> print(s.spectra[0][:4])
        [117.34012 139.27621  87.38032 143.0816 ]
        """
        self.wavelengths = []  # in nm
        self.spectra = []
        # first try with pysynphot
        file_names = []
        is_calspec = False
        if os.getenv("PYSYN_CDBS") is not None:
            dirname = os.path.expandvars('$PYSYN_CDBS/calspec/')
            #for fname in os.listdir(dirname):
            #    if os.path.isfile(os.path.join(dirname, fname)):
            #        if self.label.lower().replace(' ','') in fname.lower():
            #            file_names.append(os.path.join(dirname, fname))

            # get all calspec filenames
            filenames_found_incalspecdir = GetListOfCAMSPECFiles(dirname)

            # select the last version of filenames
            selected_filenames = FilterListOfCALSPECFiles(filenames_found_incalspecdir)

            # build a dictionnary to linl simbad target name with calspec filename
            dict_star_filename = maptargetnametpfilename(selected_filenames)

            # select the unique filename corresponding to the target name
            the_filename = dict_star_filename[self.label]
            file_names.append(os.path.join(dirname,the_filename))




        if len(file_names) > 0:
            is_calspec = True
            self.emission_spectrum = False
            self.hydrogen_only = False
            self.lines = Lines(HYDROGEN_LINES + ATMOSPHERIC_LINES + STELLAR_LINES,
                               redshift=self.redshift, emission_spectrum=self.emission_spectrum,
                               hydrogen_only=self.hydrogen_only)
            for k, f in enumerate(file_names):
                if '_mod_' in f:
                    continue
                if self.verbose:
                    self.my_logger.info('\n\tLoading %s' % f)
                data = S.FileSpectrum(f, keepneg=True)
                if isinstance(data.waveunits, S.units.Angstrom):
                    self.wavelengths.append(data.wave / 10.)
                    self.spectra.append(data.flux * 10.)
                else:
                    self.wavelengths.append(data.wave)
                    self.spectra.append(data.flux)
        # TODO DM-33731: the use of self.label in parameters.STAR_NAMES:
        # below works for running but breaks a test so needs fixing for DM
        elif 'HD' in self.label:  # or self.label in parameters.STAR_NAMES:  # it is a star
            self.emission_spectrum = False
            self.hydrogen_only = False
            self.lines = Lines(ATMOSPHERIC_LINES + HYDROGEN_LINES + STELLAR_LINES,
                               redshift=self.redshift, emission_spectrum=self.emission_spectrum,
                               hydrogen_only=self.hydrogen_only)
        elif 'PNG' in self.label:
            self.emission_spectrum = True
            self.lines = Lines(ATMOSPHERIC_LINES + ISM_LINES + HYDROGEN_LINES,
                               redshift=self.redshift, emission_spectrum=self.emission_spectrum,
                               hydrogen_only=self.hydrogen_only)
        else:  # maybe a quasar, try with NED query
            from astroquery.ned import Ned
            hdulists = Ned.get_spectra(self.label, show_progress=False)
            if len(hdulists) > 0:
                self.emission_spectrum = True
                self.hydrogen_only = False
                if self.redshift > 0.2:
                    self.hydrogen_only = True
                    parameters.LAMBDA_MIN *= 1 + self.redshift
                    parameters.LAMBDA_MAX *= 1 + self.redshift
                self.lines = Lines(ATMOSPHERIC_LINES+ISM_LINES+HYDROGEN_LINES,
                                   redshift=self.redshift, emission_spectrum=self.emission_spectrum,
                                   hydrogen_only=self.hydrogen_only)
                for k, h in enumerate(hdulists):
                    if h[0].header['NAXIS'] == 1:
                        self.spectra.append(h[0].data)
                    else:
                        for d in h[0].data:
                            self.spectra.append(d)
                    wave_n = len(h[0].data)
                    if h[0].header['NAXIS'] == 2:
                        wave_n = len(h[0].data.T)
                    wave_step = h[0].header['CDELT1']
                    wave_start = h[0].header['CRVAL1'] - (h[0].header['CRPIX1'] - 1) * wave_step
                    wave_end = wave_start + wave_n * wave_step
                    waves = np.linspace(wave_start, wave_end, wave_n)
                    is_angstrom = False
                    for key in list(h[0].header.keys()):
                        if 'angstrom' in str(h[0].header[key]).lower():
                            is_angstrom = True
                    if is_angstrom:
                        waves *= 0.1
                    if h[0].header['NAXIS'] > 1:
                        for i in range(h[0].header['NAXIS'] + 1):
                            self.wavelengths.append(waves)
                    else:
                        self.wavelengths.append(waves)
        self.build_sed()
        self.my_logger.debug(f"\n\tTarget label: {self.label}"
                             f"\n\tCalspec? {is_calspec}"
                             f"\n\tNumber of spectra: {len(self.spectra)}"
                             f"\n\tRedshift: {self.redshift}"
                             f"\n\tEmission spectrum ? {self.emission_spectrum}"
                             f"\n\tLines: {[l.label for l in self.lines.lines]}")

    def get_radec_position_after_pm(self, date_obs):
        target_pmra = self.simbad[0]['PMRA'] * u.mas / u.yr
        if np.isnan(target_pmra):
            target_pmra = 0 * u.mas / u.yr
        target_pmdec = self.simbad[0]['PMDEC'] * u.mas / u.yr
        if np.isnan(target_pmdec):
            target_pmdec = 0 * u.mas / u.yr
        target_parallax = self.simbad[0]['PLX_VALUE'] * u.mas
        if target_parallax == 0 * u.mas:
            target_parallax = 1e-4 * u.mas
        target_coord = SkyCoord(ra=self.radec_position.ra, dec=self.radec_position.dec,
                                distance=Distance(parallax=target_parallax),
                                pm_ra_cosdec=target_pmra, pm_dec=target_pmdec, frame='icrs', equinox="J2000",
                                obstime="J2000")
        self.radec_position_after_pm = target_coord.apply_space_motion(new_obstime=Time(date_obs))
        return self.radec_position_after_pm

    def build_sed(self, index=0):
        """Interpolate the database reference spectra and return self.sed as a function of the wavelength.

        Parameters
        ----------
        index: int
            Index of the spectrum stored in the self.spectra list

        Examples
        --------
        >>> s = Star('HD111980')
        >>> s.build_sed(index=0)
        >>> s.sed(550)
        array(1.67605113e-11)
        """
        if len(self.spectra) == 0:
            self.sed = lambda x: np.zeros_like(x)
        else:
            self.sed = interp1d(self.wavelengths[index], self.spectra[index], kind='linear', bounds_error=False,
                                fill_value=0.)

    def plot_spectra(self):
        """ Plot the spectra stored in the self.spectra list.

        Examples
        --------
        >>> s = Star('HD111980')
        >>> s.plot_spectra()
        """
        # target.load_spectra()  ## No global target object available  here (SDC)
        plt.figure()  # necessary to create a new plot (SDC)
        for isp, sp in enumerate(self.spectra):
            plt.plot(self.wavelengths[isp], sp, label='Spectrum %d' % isp)
        plt.xlim((300, 1100))
        plt.xlabel(r'$\lambda$ [nm]')
        plt.ylabel('Flux')
        plt.title(self.label)
        plt.legend()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

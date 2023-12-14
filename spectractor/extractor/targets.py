from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
from astropy.time import Time
from astroquery.simbad import SimbadClass

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.extractor.spectroscopy import (Lines, HGAR_LINES, HYDROGEN_LINES, ATMOSPHERIC_LINES,
                                                ISM_LINES, STELLAR_LINES)

from getCalspec import getCalspec


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
    >>> print(t.radec_position.dec)  # doctest: +ELLIPSIS
    -18d31m...s
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
    elif parameters.OBS_OBJECT_TYPE == "LED":
        return Led(label, verbose)
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


class Led(Target):

    def __init__(self, label, verbose=False):
        """Initialize Led class.

        Parameters
        ----------
        label: str
            String label to name the led.
        verbose: bool, optional
            Set True to increase verbosity (default: False)

        Examples
        --------

        >>> t = Led("XX", verbose=False)
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


def patchSimbadURL(simbad):
    """Monkeypatch the URL that Simbad is using to force it to use https.

    This is necessary to make the tests on github actions work, because
    for some reason it wasn't automatically upgrading to https otherwise.
    """
    simbad.SIMBAD_URL = simbad.SIMBAD_URL.replace("http:", "https:")


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

        >>> s = Star('PNG321.0+3.9')
        >>> print(s.label)
        PNG321.0+3.9
        >>> print(s.radec_position.dec)  # doctest: +ELLIPSIS
        -54d18m07.521s
        >>> print(s.emission_spectrum)
        True

        Standard star:

        >>> s = Star('HD111980')
        >>> print(s.label)
        HD111980
        >>> print(s.radec_position.dec)  # doctest: +ELLIPSIS
        -18d31m...s
        >>> print(s.emission_spectrum)
        False

        """
        Target.__init__(self, label, verbose=verbose)
        self.my_logger = set_logger(self.__class__.__name__)
        self.simbad_table = None
        self.load()

    def load(self):
        """Load the coordinates of the target.

        Examples
        --------
        >>> parameters.VERBOSE = True
        >>> s = Star('PNG321.0+3.9')
        >>> print(s.radec_position.dec)
        -54d18m07.521s
        >>> print(s.redshift)
        -0.00021
        >>> s = Star('eta dor')
        >>> print(s.radec_position.dec)
        -66d02m22.635s
        >>> s = Star('mu.col')
        >>> print(s.radec_position.dec)
        -32d18m23.162s
        """
        # explicitly make a class instance here because:
        # when using ``from astroquery.simbad import Simbad`` and then using
        # ``Simbad...`` methods secretly makes an instance, which stays around,
        # has a connection go stale, and then raises an exception seemingly
        # at some random time later
        simbadQuerier = SimbadClass()
        patchSimbadURL(simbadQuerier)

        simbadQuerier.add_votable_fields('flux(U)', 'flux(B)', 'flux(V)', 'flux(R)', 'flux(I)', 'flux(J)', 'sptype',
                                         'parallax', 'pm', 'z_value')
        if not getCalspec.is_calspec(self.label) and getCalspec.is_calspec(self.label.replace(".", " ")):
            self.label = self.label.replace(".", " ")
        astroquery_label = self.label
        if getCalspec.is_calspec(self.label):
            calspec = getCalspec.Calspec(self.label)
            astroquery_label = calspec.Astroquery_Name
        self.simbad_table = simbadQuerier.query_object(astroquery_label)

        if self.simbad_table is not None:
            if self.verbose or True:
                self.my_logger.info(f'\n\tSimbad:\n{self.simbad_table}')
            self.radec_position = SkyCoord(self.simbad_table['RA'][0] + ' ' + self.simbad_table['DEC'][0], unit=(u.hourangle, u.deg))
        else:
            raise RuntimeError(f"Target {self.label} not found in Simbad")
        self.get_radec_position_after_pm(date_obs="J2000")
        if not np.ma.is_masked(self.simbad_table['Z_VALUE']):
            self.redshift = float(self.simbad_table['Z_VALUE'])
        else:
            self.redshift = 0
        self.load_spectra()

    def load_spectra(self):
        """Load reference spectra from getCalspec database or NED database.

        If the object redshift is >0.2, the LAMBDA_MIN and LAMBDA_MAX parameters
        are redshifted accordingly.

        Examples
        --------
        >>> s = Star('HD111980')
        >>> print(s.spectra[0][:4])
        [2.2839e-13 2.0263e-13 2.0889e-13 2.3928e-13]
        >>> print(f'{parameters.LAMBDA_MIN:.1f}, {parameters.LAMBDA_MAX:.1f}')
        300.0, 1100.0
        >>> print(s.spectra[0][:4])
        [2.2839e-13 2.0263e-13 2.0889e-13 2.3928e-13]
        """
        self.wavelengths = []  # in nm
        self.spectra = []
        # first try if it is a Calspec star
        is_calspec = getCalspec.is_calspec(self.label)
        if is_calspec:
            calspec = getCalspec.Calspec(self.label)
            self.emission_spectrum = False
            self.hydrogen_only = False
            self.lines = Lines(HYDROGEN_LINES + ATMOSPHERIC_LINES + STELLAR_LINES,
                               redshift=self.redshift, emission_spectrum=self.emission_spectrum,
                               hydrogen_only=self.hydrogen_only)
            spec_dict = calspec.get_spectrum_numpy()
            # official units in spectractor are nanometers for wavelengths and erg/s/cm2/nm for fluxes
            spec_dict["WAVELENGTH"] = spec_dict["WAVELENGTH"].to(u.nm)
            spec_dict["FLUX"] = spec_dict["FLUX"].to(u.erg / u.second / u.cm**2 / u.nm)
            self.wavelengths.append(spec_dict["WAVELENGTH"].value)
            self.spectra.append(spec_dict["FLUX"].value)
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
            try:
                hdulists = Ned.get_spectra(self.label) #, show_progress=False)
            except Exception as err:
                raise err
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
                             f"\n\tEmission spectrum ? {self.emission_spectrum}")
        if self.lines is not None and len(self.lines.lines) > 0:
            self.my_logger.debug(f"\n\tLines: {[l.label for l in self.lines.lines]}")

    def get_radec_position_after_pm(self, date_obs):
        if self.simbad_table is not None:
            target_pmra = self.simbad_table[0]['PMRA'] * u.mas / u.yr
            if np.isnan(target_pmra):
                target_pmra = 0 * u.mas / u.yr
            target_pmdec = self.simbad_table[0]['PMDEC'] * u.mas / u.yr
            if np.isnan(target_pmdec):
                target_pmdec = 0 * u.mas / u.yr
            target_parallax = self.simbad_table[0]['PLX_VALUE'] * u.mas
            if target_parallax == 0 * u.mas:
                target_parallax = 1e-4 * u.mas
            target_coord = SkyCoord(ra=self.radec_position.ra, dec=self.radec_position.dec,
                                    distance=Distance(parallax=target_parallax),
                                    pm_ra_cosdec=target_pmra, pm_dec=target_pmdec, frame='icrs', equinox="J2000",
                                    obstime="J2000")
            self.radec_position_after_pm = target_coord.apply_space_motion(new_obstime=Time(date_obs))
            return self.radec_position_after_pm
        else:
            self.my_logger.warning("No Simbad table provided: can't apply proper motion correction. "
                                   "Return original (RA,DEC) coordinates of the object.")
            return self.radec_position

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
        array(1.67508011e-11)
        """
        if len(self.spectra) == 0:
            self.sed = interp1d(parameters.LAMBDAS, np.zeros_like(parameters.LAMBDAS), kind='linear', bounds_error=False,
                                fill_value=0.)
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
            plt.plot(self.wavelengths[isp], sp, label=f'Spectrum {isp}')
        plt.xlim((300, 1100))
        plt.xlabel(r'$\lambda$ [nm]')
        plt.ylabel('Flux [erg/s/cm2/nm]')
        plt.title(self.label)
        plt.legend()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as units
from astropy.coordinates import SkyCoord
from astroquery.ned import Ned
from astroquery.simbad import Simbad
from scipy.interpolate import interp1d

from spectractor.tools import *
from spectractor import parameters

if os.getenv("PYSYN_CDBS"):
    import pysynphot as S


class Target:

    def __init__(self, label, verbose=False):
        """Initialize Target class.

        Parameters
        ----------
        label: str
            String label to name the target
        verbose: bool, optional
            Set True to increase verbosity (default: False)

        Examples
        --------

        Emission line object:
        >>> t = Target('3C273')
        >>> print(t.label)
        3C273
        >>> print(t.coord.dec)
        2d03m08.598s
        >>> print(t.emission_spectrum)
        True

        Standard star:
        >>> t = Target('HD111980')
        >>> print(t.label)
        HD111980
        >>> print(t.coord.dec)
        -18d31m20.009s
        >>> print(t.emission_spectrum)
        False

        """
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.label = label
        self.type = None
        self.wavelengths = []
        self.spectra = []
        self.verbose = verbose
        self.emission_spectrum = False
        self.hydrogen_only = False
        self.sed = None
        self.lines = None

    def load(self):
        pass


class ArcLamp(Target):

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

        Mercury-Argon lamp:
        >>> s = Star('3C273')
        >>> print(s.label)
        3C273
        >>> print(s.coord.dec)
        2d03m08.598s
        >>> print(s.emission_spectrum)
        True

        """
        Target.__init__(self, label, verbose=verbose)
        self.my_logger = parameters.set_logger(self.__class__.__name__)

    def load(self):
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
        >>> print(s.coord.dec)
        2d03m08.598s
        >>> print(s.emission_spectrum)
        True

        Standard star:
        >>> s = Star('HD111980')
        >>> print(s.label)
        HD111980
        >>> print(s.coord.dec)
        -18d31m20.009s
        >>> print(s.emission_spectrum)
        False

        """
        Target.__init__(self, label, verbose=verbose)
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.coord = None
        self.redshift = 0
        self.load()

    def load(self):
        """Load the coordinates of the target.

        Examples
        --------
        >>> s = Star('3C273')
        >>> print(s.coord.dec)
        2d03m08.598s

        """
        Simbad.add_votable_fields('flux(U)', 'flux(B)', 'flux(V)', 'flux(R)', 'flux(I)', 'flux(J)', 'sptype')
        simbad = Simbad.query_object(self.label)
        if simbad is not None:
            if self.verbose:
                self.my_logger.info(f'\n\tSimbad: {simbad}')
            self.coord = SkyCoord(simbad['RA'][0] + ' ' + simbad['DEC'][0], unit=(units.hourangle, units.deg))
        else:
            self.my_logger.warning('Target {} not found in Simbad'.format(self.label))
        self.load_spectra()

    def load_spectra(self):
        """Load reference spectra from Pysynphot database or NED database.

        Examples
        --------
        >>> s = Star('3C273')
        >>> print(s.spectra[0][:4])
        [  0.00000000e+00   2.50485769e-14   2.42380612e-14   2.40887886e-14]
        >>> s = Star('HD111980')
        >>> print(s.spectra[0][:4])
        [  2.16890002e-13   2.66480010e-13   2.03540011e-13   2.38780004e-13]
        """
        self.wavelengths = []  # in nm
        self.spectra = []
        # first try with pysynphot
        file_names = []
        if os.getenv("PYSYN_CDBS") is not None:
            dirname = os.path.expandvars('$PYSYN_CDBS/calspec/')
            for fname in os.listdir(dirname):
                if os.path.isfile(dirname + fname):
                    if self.label.lower() in fname.lower():
                        file_names.append(dirname + fname)
        if len(file_names) > 0:
            self.emission_spectrum = False
            self.hydrogen_only = True
            self.lines = Lines(parameters.HYDROGEN_LINES+parameters.ATMOSPHERIC_LINES,
                                   redshift=0., emission_spectrum=self.emission_spectrum,
                                   hydrogen_only=self.hydrogen_only)
            for k, f in enumerate(file_names):
                if '_mod_' in f:
                    continue
                if self.verbose:
                    print('Loading %s' % f)
                data = S.FileSpectrum(f, keepneg=True)
                if isinstance(data.waveunits, S.units.Angstrom):
                    self.wavelengths.append(data.wave / 10.)
                    self.spectra.append(data.flux * 10.)
                else:
                    self.wavelengths.append(data.wave)
                    self.spectra.append(data.flux)
        else:
            if 'PNG' not in self.label:
                # Try with NED query
                # print 'Loading target %s from NED...' % self.label
                ned = Ned.query_object(self.label)
                hdulists = Ned.get_spectra(self.label, show_progress=False)
                self.redshift = ned['Redshift'][0]
                self.emission_spectrum = True
                self.hydrogen_only = False
                if self.redshift > 0.2:
                    self.hydrogen_only = True
                    parameters.LAMBDA_MIN *= 1 + self.redshift
                    parameters.LAMBDA_MAX *= 1 + self.redshift
                self.lines = Lines(parameters.ATMOSPHERIC_LINES+parameters.ISM_LINES+parameters.HYDROGEN_LINES,
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
            else:
                self.emission_spectrum = True
                self.lines = Lines(parameters.ATMOSPHERIC_LINES+parameters.ISM_LINES+parameters.HYDROGEN_LINES,
                                   redshift=0., emission_spectrum=self.emission_spectrum,
                                   hydrogen_only=self.hydrogen_only)
        self.build_sed()

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
        array(1.676051129017069e-11)
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
        plt.xlabel('$\lambda$ [nm]')
        plt.ylabel('Flux')
        plt.title(self.label)
        plt.legend()
        if parameters.DISPLAY:
            plt.show()


if __name__ == "__main__":
    import doctest
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    doctest.testmod()

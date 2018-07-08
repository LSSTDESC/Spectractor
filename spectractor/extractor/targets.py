import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as units
from astropy.coordinates import SkyCoord
from astroquery.ned import Ned
from astroquery.simbad import Simbad
from scipy.interpolate import interp1d

from spectractor import parameters

if os.getenv("PYSYN_CDBS"):
    import pysynphot as S


class Target:

    def __init__(self, label, verbose=False):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.label = label
        self.ra = None
        self.dec = None
        self.coord = None
        self.type = None
        self.redshift = 0
        self.wavelengths = []
        self.spectra = []
        self.verbose = verbose
        self.emission_spectrum = False
        self.hydrogen_only = False
        self.sed = None
        self.load()

    def load(self):
        Simbad.add_votable_fields('flux(U)', 'flux(B)', 'flux(V)', 'flux(R)', 'flux(I)', 'flux(J)', 'sptype')
        simbad = Simbad.query_object(self.label)
        if simbad is not None:
            if self.verbose:
                print(simbad)
            self.coord = SkyCoord(simbad['RA'][0] + ' ' + simbad['DEC'][0], unit=(units.hourangle, units.deg))
        else:
            self.my_logger.warning('Target {} not found in Simbad'.format(self.label))
        self.load_spectra()

    def load_spectra(self):
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
                hdulists = Ned.get_spectra(self.label)
                self.redshift = ned['Redshift'][0]
                self.emission_spectrum = True
                self.hydrogen_only = False
                if self.redshift > 0.2:
                    self.hydrogen_only = True
                    parameters.LAMBDA_MIN *= 1 + self.redshift
                    parameters.LAMBDA_MAX *= 1 + self.redshift
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
        self.build_sed()

    def build_sed(self, index=0):
        if len(self.spectra) == 0:
            self.sed = lambda x: np.zeros_like(x)
        else:
            self.sed = interp1d(self.wavelengths[index], self.spectra[index], kind='linear', bounds_error=False,
                                fill_value=0.)

    def plot_spectra(self):
        # target.load_spectra()  ## No global target object available  here (SDC)
        plt.figure()  # necessary to create a new plot (SDC)
        for isp, sp in enumerate(self.spectra):
            plt.plot(self.wavelengths[isp], sp, label='Spectrum %d' % isp)
        plt.xlim((300, 1100))
        plt.xlabel('$\lambda$ [nm]')
        plt.ylabel('Flux')
        plt.title(self.label)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import os
    from optparse import OptionParser
    import matplotlib as mpl

    if os.environ.get('DISPLAY', '') == '':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')

    parser = OptionParser()
    parser.add_option("-l", "--label", dest="label",
                      help="Label of the target.", default="HD111980")
    (opts, args) = parser.parse_args()

    print('Load information on target {}'.format(opts.label))
    target = Target(opts.label)
    print(target.coord)
    target.plot_spectra()

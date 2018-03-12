import numpy as np
import sys, os
from astropy import units as units
from astropy.coordinates import SkyCoord, Angle
#import pyfits
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

from astroquery.simbad import Simbad
from astroquery.ned import Ned

os.environ['PYSYN_CDBS']
import pysynphot as S


EPOCH = "J2000.0"

class Target():

    def __init__(self,label,verbose=False):
        self.label = label
        self.ra = None
        self.dec = None
        self.coord = None
        self.type = None
        self.redshift = 0
        self.spectra = []
        self.verbose = verbose
        self.emission_spectrum = False
        self.hydrogen_only = False
        self.load()

    def load(self):
        if self.label == "HD111980" :
            self.ra = "12 53 15.1"
            self.dec = "-18 31 20"
            self.type = "CALSPEC"
        elif self.label == "HD160617":
            self.ra = "17 42 49.3"
            self.dec = "-40 19 15"
            self.type = "CALSPEC"
        elif self.label == "HD185975":
            self.ra = "20 28 18.7"
            self.dec = "-87 28 19"
            self.type = "CALSPEC"
        elif self.label == "HD205905":
            self.ra = "21 39 10.1"
            self.dec = "-27 18 23"
            self.type = "CALSPEC"
        elif self.label == "3C273" :
            self.ra = "12 29 06.7"
            self.dec = "02 03 08"
            self.type = "QSO"
        #else :
        #    sys.exit('Unknown target %s' % self.label)
        Simbad.add_votable_fields('flux(U)','flux(B)','flux(V)','flux(R)','flux(I)','flux(J)','sptype')
        self.simbad = Simbad.query_object(self.label)
        if self.verbose: print self.simbad
        self.coord = SkyCoord(self.simbad['RA'][0]+' '+self.simbad['DEC'][0], unit=(units.hourangle, units.deg))
        self.load_spectra()

    def load_spectra(self):
        self.wavelengths = [] # in nm
        self.spectra = []
        # first try with pysynphot
        filenames = []
        dirname = os.path.expandvars('$PYSYN_CDBS/calspec/')
        for fname in os.listdir(dirname):
            if os.path.isfile(dirname+fname):          
                if self.label.lower() in fname.lower() :
                    filenames.append(dirname+fname)
        if len(filenames) > 0 :
            self.emission_spectrum = False
            self.hydrogen_only = True
            for k,f in enumerate(filenames) :
                if '_mod_' in f : continue
                print 'Loading %s' % f
                data = S.FileSpectrum(f,keepneg=True)
                if isinstance(data.waveunits,S.units.Angstrom) : 
                    self.wavelengths.append(data.wave/10.)
                    self.spectra.append(data.flux*10.)
                else : 
                    self.wavelengths.append(data.wave)
                    self.spectra.append(data.flux)
        else :
            if 'PNG' not in self.label:
                # Try with NED query
                #print 'Loading target %s from NED...' % self.label
                self.ned = Ned.query_object(self.label)
                hdulists = Ned.get_spectra(self.label)
                self.redshift = self.ned['Redshift'][0]
                self.emission_spectrum = True
                self.hydrogen_only = False
                for k,h in enumerate(hdulists) :
                    if h[0].header['NAXIS'] == 1 :
                        self.spectra.append(h[0].data)
                    else :
                        for d in h[0].data :
                            self.spectra.append(d)
                    wave_n = len(h[0].data)
                    if h[0].header['NAXIS'] == 2 : wave_n = len(h[0].data.T)
                    wave_step = h[0].header['CDELT1']
                    wave_start = h[0].header['CRVAL1'] - (h[0].header['CRPIX1']-1)*wave_step
                    wave_end = wave_start + wave_n*wave_step 
                    waves = np.linspace(wave_start,wave_end,wave_n)
                    is_angstrom = False
                    for key in h[0].header.keys() :
                        if 'angstrom' in str(h[0].header[key]).lower() :
                            is_angstrom=True
                    if is_angstrom : waves*=0.1
                    if h[0].header['NAXIS'] > 1 :
                        for k in range(h[0].header['NAXIS']+1) :
                            self.wavelengths.append(waves)
                    else : 
                        self.wavelengths.append(waves)
            else:
                self.emission_spectrum = True

        
    def plot_spectra(self):
        target.load_spectra()
        for isp,sp in enumerate(self.spectra):
            plt.plot(self.wavelengths[isp],sp,label='Spectrum %d' % isp)
        #plt.xlim((400,1200))
        plt.xlabel('$\lambda$ [nm]')
        plt.ylabel('Flux')
        plt.title(self.label)
        plt.legend()
        plt.show()



if __name__ == "__main__":
    import commands, string, re, time, os
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-l", "--label", dest="label",
                      help="Label of the target.",default="HD111980")
    (opts, args) = parser.parse_args()

    print 'Load informations on target %s' % opts.label
    target = Target(opts.label)
    print target.coord
    target.plot_spectra()

 

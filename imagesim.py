import sys,os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../SpectractorSim")

from scipy.signal import fftconvolve, gaussian
from astroquery.gaia import Gaia, TapPlus, GaiaClass
Gaia = GaiaClass(TapPlus(url='http://gaia.ari.uni-heidelberg.de/tap'))

from tools import *
from dispersers import *
from targets import *
from images import *
from spectroscopy import *
from spectractorsim import *
import parameters 


class StarModel():

    def __init__(self,pixcoords,A,sigma,target=None):
        """ x0, y0 coords in pixels, sigma witdth in pixels, A height in image units"""
        self.my_logger = set_logger(self.__class__.__name__)
        self.x0 = pixcoords[0]
        self.y0 = pixcoords[1]
        self.A = A
        self.sigma = sigma
        self.target = target
        self.model = models.Gaussian2D(A, self.x0, self.y0, sigma, sigma, 0)

    def plot_model(self):
        yy, xx = np.meshgrid(np.linspace(self.y0-10*self.sigma,self.y0+10*self.sigma,50),  np.linspace(self.x0-10*self.sigma,self.x0+10*self.sigma,50))
        star = self.model(xx,yy)
        fig, ax = plt.subplots(1,1)
        im = plt.imshow(star,origin='lower', cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Star model: A=%.2f, sigma=%.2f' % (self.A,self.sigma))
        cb = plt.colorbar(im,ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7,prune=None)
        cb.update_ticks()
        cb.set_label('Arbitrary units') #,fontsize=16)
        plt.show()

class StarFieldModel():

    def __init__(self,base_image,sigma,threshold=0):
        self.target = base_image.target
        self.sigma = sigma
        self.field = None
        result = Gaia.query_object_async(self.target.coord, width=IMSIZE*PIXEL2ARCSEC*units.arcsec/np.cos(self.target.coord.dec.radian), height=IMSIZE*PIXEL2ARCSEC*units.arcsec)
        if parameters.DEBUG:
            print result.pprint()
        self.stars = []
        self.pixcoords = []
        x0, y0 = base_image.target_pixcoords
        for o in result:
            if o['dist'] < 0.01 : continue
            radec = SkyCoord(ra=float(o['ra']),dec=float(o['dec']), frame='icrs', unit='deg')
            y = int(y0 - (radec.dec.arcsec - self.target.coord.dec.arcsec)/PIXEL2ARCSEC)
            x = int(x0 - (radec.ra.arcsec - self.target.coord.ra.arcsec)*np.cos(radec.dec.radian)/PIXEL2ARCSEC)
            w = 20 # search windows in pixels
            if x<w or x>IMSIZE-w: continue
            if y<w or y>IMSIZE-w: continue
            sub =  base_image.data[y-w:y+w,x-w:x+w]
            A = np.max(sub) - np.min(sub)
            if A < threshold: continue
            self.stars.append( StarModel([x,y],A,sigma) )
            self.pixcoords.append([x,y])
        self.pixcoords = np.array(self.pixcoords).T

    def model(self,x,y):
        if self.field is None:
            self.field = self.stars[0].model(x,y)
            for k in range(len(self.stars)):
                self.field += self.stars[k].model(x,y)
        return self.field

    def plot_model(self):
        yy, xx = np.mgrid[0:IMSIZE:1, 0:IMSIZE:1]
        starfield = self.model(xx,yy)
        fig, ax = plt.subplots(1,1)
        im = plt.imshow(starfield,origin='lower', cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Star field model: sigma=%.2f' % (self.sigma))
        cb = plt.colorbar(im,ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7,prune=None)
        cb.update_ticks()
        cb.set_label('Arbitrary units') #,fontsize=16)
        plt.show()

class BackgroundModel():

    def __init__(self,level,frame=None):
        self.my_logger = set_logger(self.__class__.__name__)
        self.level = level
        if self.level <= 0:
            self.my_logger.warning('\n\tBackground level must be strictly positive.')
        self.frame = frame

    def model(self,x,y):
        bkgd = self.level*np.ones_like(x)
        if self.frame is None:
            return bkgd
        else:
            xlim, ylim = self.frame
            bkgd[ylim:,:] = self.level / 100
            bkgd[:,xlim:] = self.level / 100
            kernel = np.outer(gaussian(IMSIZE, 50), gaussian(IMSIZE, 50))
            bkgd = fftconvolve(bkgd, kernel, mode='same')
            bkgd *= self.level/bkgd[IMSIZE/2,IMSIZE/2]
            return bkgd
            

    def plot_model(self):
        yy, xx = np.mgrid[0:IMSIZE:1, 0:IMSIZE:1]
        bkgd = self.model(xx,yy)
        fig, ax = plt.subplots(1,1)
        im = plt.imshow(bkgd,origin='lower', cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Background model')
        cb = plt.colorbar(im,ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7,prune=None)
        cb.update_ticks()
        cb.set_label('Arbitrary units') #,fontsize=16)
        plt.show()


    
class SpectrumModel():

    def __init__(self,base_image,spectrumsim,sigma,A1=1,A2=0,reso=None,rotation=False):
        self.my_logger = set_logger(self.__class__.__name__)
        self.base_image = base_image
        self.spectrumsim = spectrumsim
        self.disperser = base_image.disperser
        self.sigma = sigma
        self.A1 = A1
        self.A2 = A2
        self.reso = reso
        self.rotation = rotation
        self.yprofile = models.Gaussian1D(1,0,sigma)

    def model(self,x,y):
        x0, y0 = self.base_image.target_pixcoords
        if self.rotation:
            theta = self.disperser.theta(self.base_image.target_pixcoords)*np.pi/180.
            u =  np.cos(theta)*(x-x0) + np.sin(theta)*(y-y0)
            v = -np.sin(theta)*(x-x0) + np.cos(theta)*(y-y0)
            x = u + x0
            y = v + y0
        l = self.disperser.grating_pixel_to_lambda(x-x0,x0=self.base_image.target_pixcoords,order=1)
        amp = self.A1*self.spectrumsim.model(l)*self.yprofile(y-y0) + self.A1*self.A2*self.spectrumsim.model(l/2)*self.yprofile(y-y0)
        if self.reso is not None:
            kernel = gaussian(amp.shape[1],self.reso)
            kernel /= np.sum(kernel)
            for i in range(amp.shape[0]):
                amp[i] = fftconvolve(amp[i], kernel, mode='same')
        return amp

class ImageModel(Image):

    def __init__(self,filename,target=""):
        self.my_logger = set_logger(self.__class__.__name__)
        Image.__init__(self,filename,target=target)

    def compute(self,star,background,spectrum,starfield=None):
        yy, xx = np.mgrid[0:IMSIZE:1, 0:IMSIZE:1]
        self.data = star.model(xx,yy) + background.model(xx,yy) + spectrum.model(xx,yy)
        if starfield is not None:
            self.data += starfield.model(xx,yy)

    def add_poisson_noise(self):
        d = np.copy(self.data).astype(float)
        d *= self.expo*self.gain
        noisy = np.random.poisson(d).astype(float)
        self.data = noisy /(self.expo*self.gain)
        


def ImageSim(filename,outputdir,guess,target,A1=1,A2=0,with_rotation=True,with_stars=True):
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart IMAGE SIMULATOR')
    # Load reduced image
    image = ImageModel(filename,target=target)
    if parameters.DEBUG:
        image.plot_image(scale='log10',target_pixcoords=guess)
    # Find the exact target position in the raw cut image: several methods
    my_logger.info('\n\tSearch for the target in the image...')
    target_pixcoords = image.find_target_2Dprofile(guess)
    # Background model
    my_logger.info('\n\tBackground model...')
    background = BackgroundModel(level=image.target_bkgd2D(0,0),frame=(1600,1650))
    if parameters.DEBUG:
        background.plot_model()
    # Target model
    my_logger.info('\n\tStar model...')
    star = StarModel(target_pixcoords,A=image.target_gauss2D.amplitude.value,sigma=image.target_gauss2D.x_stddev.value,target=image.target)
    reso = star.sigma
    if parameters.DEBUG:
        star.plot_model()
    # Star field model
    starfield = None
    if with_stars:
        my_logger.info('\n\tStar field model...')
        starfield = StarFieldModel(image,sigma=image.target_gauss2D.x_stddev.value,threshold=0.01*star.A)
        if parameters.VERBOSE:
            image.plot_image(scale='log10',target_pixcoords=starfield.pixcoords)
            starfield.plot_model()
    # Spectrum model
    my_logger.info('\n\tSpectum model...')
    lambdas = np.arange(parameters.LAMBDA_MIN,parameters.LAMBDA_MAX)
    airmass = image.header['AIRMASS']
    pressure = image.header['OUTPRESS']
    temperature = image.header['OUTTEMP']
    telescope=TelescopeTransmission(image.filter)    
    spectrumsim = SpectractorSimCore(image, telescope, image.disperser, image.target, lambdas, airmass, pressure, temperature, pwv=5, ozone=300,aerosols=0)
    spectrum = SpectrumModel(image,spectrumsim,sigma=reso,A1=A1,A2=A2,reso=reso,rotation=with_rotation)
    # Image model
    my_logger.info('\n\tImage model...')
    image.compute(star,background,spectrum,starfield=starfield)
    image.add_poisson_noise()
    if parameters.VERBOSE:
        image.plot_image(scale="log",title="Image simulation",target_pixcoords=target_pixcoords,units=spectrumsim.units)
    # Set output path
    ensure_dir(outputdir)
    output_filename = filename.split('/')[-1]
    output_filename = (output_filename.replace('reduc','sim')).replace('trim','sim')
    output_filename = os.path.join(outputdir,output_filename)
    # Save images and parameters
    image.header['A1'] = A1
    image.header['A2'] = A2
    image.header['reso'] = reso
    image.header['ROTATION'] = int(with_rotation)
    image.header['STARS'] = int(with_stars)
    image.save_image(output_filename,overwrite=True)
    return image


if __name__ == "__main__":
    import commands, string, re, time, os
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--debug", dest="debug",action="store_true",
                      help="Enter debug mode (more verbose and plots).",default=False)
    parser.add_option("-v", "--verbose", dest="verbose",action="store_true",
                      help="Enter verbose (print more stuff).",default=False)
    parser.add_option("-o", "--output_directory", dest="output_directory", default="test/",
                      help="Write results in given output directory (default: ./tests/).")
    (opts, args) = parser.parse_args()


    parameters.VERBOSE = opts.verbose
    if opts.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True
        
    filename="../CTIOAnaJun2017/ana_29may17/OverScanRemove/trim_images/trim_20170529_150.fits"
    guess = [720, 670]
    target = "HD185975"
    #filename="../CTIOAnaJun2017/ana_31may17/OverScanRemove/trim_images/trim_20170531_150.fits"
    #guess = [840, 530]
    #target = "HD205905"

    image = ImageSim(filename,opts.output_directory,guess,target)

    
    

        

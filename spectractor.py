import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import sys,os
import copy
from tools import *
from holo_specs import *
from targets import *
import parameters 
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

class Image():

    def __init__(self,filename,target=""):
        """
        Args:
            filename (:obj:`str`): path to the image
        """
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.filename = filename
        self.units = 'ADU'
        self.load(filename)
        # Load the target if given
        self.target = None
        if target != "":
            self.target=Target(target,verbose=parameters.VERBOSE)
            self.header['TARGET'] = self.target.label
            self.header.comments['TARGET'] = 'object targeted in the image'
            self.header['REDSHIFT'] = self.target.redshift
            self.header.comments['REDSHIFT'] = 'redshift of the target'
        self.err = None

    def load(self,filename):
        """
        Args:
            filename (:obj:`str`): path to the image
        """
        self.my_logger.info('\n\tLoading image %s...' % filename)
        hdu_list = fits.open(filename)
        self.header = hdu_list[0].header
        self.data = hdu_list[0].data
        extract_info_from_CTIO_header(self,self.header)
        parameters.IMSIZE = int(self.header['XLENGTH'])
        parameters.PIXEL2ARCSEC = float(self.header['XPIXSIZE'])
        if self.header['YLENGTH'] != parameters.IMSIZE:
            self.my_logger.warning('\n\tImage rectangular: X=%d pix, Y=%d pix' % (parameters.IMSIZE, self.header['YLENGTH']))
        if self.header['YPIXSIZE'] != parameters.PIXEL2ARCSEC:
            self.my_logger.warning('\n\tPixel size rectangular: X=%d arcsec, Y=%d arcsec' % (parameters.PIXEL2ARCSEC, self.header['YPIXSIZE']))
        self.coord = SkyCoord(self.header['RA']+' '+self.header['DEC'],unit=(units.hourangle, units.deg),obstime=self.header['DATE-OBS'] )
        self.my_logger.info('\n\tImage loaded')
        # Load the disperser
        self.my_logger.info('\n\tLoading disperser %s...' % self.disperser)
        self.disperser = Hologram(self.disperser,data_dir=parameters.HOLO_DIR,verbose=parameters.VERBOSE)
        # compute CCD gain map
        self.build_gain_map()
        self.convert_to_ADU_rate_units()
        self.compute_statistical_error()
        self.compute_parallactic_angle()

    def build_gain_map(self):
        l = parameters.IMSIZE
        self.gain = np.zeros_like(self.data)
        # ampli 11
        self.gain[0:l/2,0:l/2] = self.header['GTGAIN11']
        # ampli 12
        self.gain[0:l/2,l/2:l] = self.header['GTGAIN12']
        # ampli 21
        self.gain[l/2:l,0:l] = self.header['GTGAIN21']
        # ampli 22
        self.gain[l/2:l,l/2:l] = self.header['GTGAIN22']

    def convert_to_ADU_rate_units(self):
        self.data /= self.expo
        self.units = 'ADU rate'
        
    def compute_statistical_error(self):
        # removes the zeros and negative pixels first
        # set to minimum positive value
        data = np.copy(self.data)
        zeros = np.where(data<=0)
        min_noz = np.min(data[np.where(data>0)])
        data[zeros] = min_noz
        # compute poisson noise
        self.stat_errors=np.sqrt(data)/np.sqrt(self.gain*self.expo)

    def compute_parallactic_angle(self):
        '''from A. Guyonnet.'''
        latitude = parameters.OBS_LATITUDE.split( )
        latitude = float(latitude[0])- float(latitude[1])/60. - float(latitude[2])/3600.
        latitude = Angle(latitude, units.deg).radian
        ha       = Angle(self.header['HA'], unit='hourangle').radian
        dec      = Angle(self.header['DEC'], unit=units.deg).radian
        parallactic_angle = np.arctan( np.sin(ha) / ( np.cos(dec)*np.tan(latitude) - np.sin(dec)*np.cos(ha) ) )
        self.parallactic_angle = parallactic_angle*180/np.pi
        self.header['PARANGLE'] = self.parallactic_angle
        self.header.comments['PARANGLE'] = 'parallactic angle in degree'
        return self.parallactic_angle
 
    def find_target(self,guess,rotated=False):
        """
        Find precisely the position of the targeted object.
        
        Args:
            guess (:obj:`list`): [x,y] guessed position of th target
        """
        x0 = guess[0]
        y0 = guess[1]
        Dx = parameters.XWINDOW
        Dy = parameters.YWINDOW
        if rotated:
            angle = self.rotation_angle*np.pi/180.
            rotmat = np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            vec = np.array(self.target_pixcoords) - 0.5*np.array(self.data.shape)
            guess2 =  np.dot(rotmat,vec) + 0.5*np.array(self.data_rotated.shape)
            x0 = int(guess2[0,0])
            y0 = int(guess2[0,1])
        if rotated:
            Dx = parameters.XWINDOW_ROT
            Dy = parameters.YWINDOW_ROT
            sub_image = np.copy(self.data_rotated[y0-Dy:y0+Dy,x0-Dx:x0+Dx])
        else:
            sub_image = np.copy(self.data[y0-Dy:y0+Dy,x0-Dx:x0+Dx])
        NX=sub_image.shape[1]
        NY=sub_image.shape[0]        
        X_=np.arange(NX)
        Y_=np.arange(NY)
        profile_X_raw=np.sum(sub_image,axis=0)
        profile_Y_raw=np.sum(sub_image,axis=1)
        # fit and subtract smooth polynomial background
        # with 3sigma rejection of outliers (star peaks)
        bkgd_X = fit_poly1d_outlier_removal(X_,profile_X_raw,order=2)
        bkgd_Y = fit_poly1d_outlier_removal(Y_,profile_Y_raw,order=2)
        profile_X = profile_X_raw - bkgd_X #np.min(profile_X)
        profile_Y = profile_Y_raw - bkgd_Y #np.min(profile_Y)

        avX,sigX=weighted_avg_and_std(X_,profile_X**4) 
        avY,sigY=weighted_avg_and_std(Y_,profile_Y**4)

        if profile_X[int(avX)] < 0.8*np.max(profile_X) :
            self.my_logger.warning('\n\tX position determination of the target probably wrong') 

        if profile_Y[int(avY)] < 0.8*np.max(profile_Y) :
            self.my_logger.warning('\n\tY position determination of the target probably wrong')

        theX=x0-Dx+avX
        theY=y0-Dy+avY
        
        if parameters.DEBUG:
            profile_X_max=np.max(profile_X_raw)*1.2
            profile_Y_max=np.max(profile_Y_raw)*1.2

            f, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(15,4))
            ax1.imshow(sub_image,origin='lower',vmin=0,vmax=10000,cmap='rainbow')
            ax1.plot([avX],[avY],'ko')
            ax1.grid(True)
            ax1.set_xlabel('X - pixel')
            ax1.set_ylabel('Y - pixel')

            ax2.plot(X_,profile_X_raw,'r-',lw=2)
            ax2.plot(X_,bkgd_X,'g--',lw=2,label='bkgd')
            ax2.axvline(Dx,color='y',linestyle='-',label='old',lw=2)
            ax2.axvline(avX,color='b',linestyle='-',label='new',lw=2)
            ax2.grid(True)
            ax2.set_xlabel('X - pixel')
            ax2.legend(loc=1)

            ax3.plot(Y_,profile_Y_raw,'r-',lw=2)
            ax3.plot(Y_,bkgd_Y,'g--',lw=2,label='bkgd')
            ax3.axvline(Dy,color='y',linestyle='-',label='old',lw=2)
            ax3.axvline(avY,color='b',linestyle='-',label='new',lw=2)
            ax3.grid(True)
            ax3.set_xlabel('Y - pixel')
            ax3.legend(loc=1)

            plt.show()

        self.my_logger.info('\n\tX,Y target position in pixels: %.3f,%.3f' % (theX,theY))
        if rotated:
            self.target_pixcoords_rotated = [theX,theY]
        else:
            self.target_pixcoords = [theX,theY]
        return [theX,theY]

    def compute_rotation_angle_hessian(self, deg_threshold = 10, width_cut = parameters.YWINDOW, right_edge = parameters.IMSIZE-200, margin_cut=12):
        x0, y0 = np.array(self.target_pixcoords).astype(int)
        # extract a region 
        data=np.copy(self.data[y0-width_cut:y0+width_cut,0:right_edge])
        # compute hessian matrices on the image
        Hxx, Hxy, Hyy = hessian_matrix(data, sigma=3, order='xy')
        lambda_plus = 0.5*( (Hxx+Hyy) + np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
        lambda_minus = 0.5*( (Hxx+Hyy) - np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
        theta = 0.5*np.arctan2(2*Hxy,Hyy-Hxx)*180/np.pi
        # remove the margins
        lambda_minus = lambda_minus[margin_cut:-margin_cut,margin_cut:-margin_cut]
        lambda_plus = lambda_plus[margin_cut:-margin_cut,margin_cut:-margin_cut]
        theta = theta[margin_cut:-margin_cut,margin_cut:-margin_cut]
        # thresholds
        lambda_threshold = np.min(lambda_minus)
        mask = np.where(lambda_minus>lambda_threshold)
        theta_mask = np.copy(theta)
        theta_mask[mask]=np.nan
        minimum_pixels = 0.01*2*width_cut*right_edge
        while len(theta_mask[~np.isnan(theta_mask)]) < minimum_pixels:
            lambda_threshold /= 2
            mask = np.where(lambda_minus>lambda_threshold)
            theta_mask = np.copy(theta)
            theta_mask[mask]=np.nan
            #print len(theta_mask[~np.isnan(theta_mask)]), lambda_threshold
        theta_guess = self.disperser.theta(self.target_pixcoords)
        mask2 = np.where(np.abs(theta-theta_guess)>deg_threshold)
        theta_mask[mask2] = np.nan
        theta_hist = []
        theta_hist = theta_mask[~np.isnan(theta_mask)].flatten()
        theta_median = np.median(theta_hist)
        theta_critical = 180.*np.arctan(10./parameters.IMSIZE)/np.pi
        if abs(theta_median-theta_guess)>theta_critical:
            self.my_logger.warning('\n\tInterpolated angle and fitted angle disagrees with more than 10 pixels over %d pixels:  %.2f vs %.2f' % (parameters.IMSIZE,theta_median,theta_guess))
        if parameters.DEBUG:
            f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
            xindex=np.arange(data.shape[1])
            x_new = np.linspace(xindex.min(),xindex.max(), 50)
            y_new = width_cut + (x_new-x0)*np.tan(theta_median*np.pi/180.)
            ax1.imshow(theta_mask,origin='lower',cmap=cm.brg,aspect='auto',vmin=-deg_threshold,vmax=deg_threshold)
            #ax1.imshow(np.log10(data),origin='lower',cmap="jet",aspect='auto')
            ax1.plot(x_new,y_new,'b-')
            ax1.set_ylim(0,2*width_cut)
            ax1.grid(True)
            n,bins, patches = ax2.hist(theta_hist,bins=int(np.sqrt(len(theta_hist))))
            ax2.plot([theta_median,theta_median],[0,np.max(n)])
            ax2.set_xlabel("Rotation angles [degrees]")
            plt.show()
        return theta_median    

    def turn_image(self):
        self.rotation_angle = self.compute_rotation_angle_hessian()
        self.my_logger.info('\n\tRotate the image with angle theta=%.2f degree' % self.rotation_angle)
        self.data_rotated = np.copy(self.data)
        if not np.isnan(self.rotation_angle):
            self.data_rotated=ndimage.interpolation.rotate(self.data,self.rotation_angle,prefilter=parameters.ROT_PREFILTER,order=parameters.ROT_ORDER)
            self.stat_errors_rotated=ndimage.interpolation.rotate(self.stat_errors,self.rotation_angle,prefilter=parameters.ROT_PREFILTER,order=parameters.ROT_ORDER)
        if parameters.DEBUG:
            f, (ax1,ax2) = plt.subplots(2,1,figsize=[8,8])
            y0 = int(self.target_pixcoords[1])
            ax1.imshow(np.log10(self.data[y0-parameters.YWINDOW:y0+parameters.YWINDOW,200:-200]),origin='lower',cmap='rainbow',aspect="auto")
            ax1.plot([0,self.data.shape[0]-200],[parameters.YWINDOW,parameters.YWINDOW],'w-')
            ax1.grid(color='white', ls='solid')
            ax1.grid(True)
            ax1.set_title('Raw image (log10 scale)')
            ax2.imshow(np.log10(self.data_rotated[y0-parameters.YWINDOW:y0+parameters.YWINDOW,200:-200]),origin='lower',cmap='rainbow',aspect="auto")
            ax2.plot([0,self.data_rotated.shape[0]-200],[parameters.YWINDOW,parameters.YWINDOW],'w-')
            ax2.grid(color='white', ls='solid')
            ax2.grid(True)
            ax2.set_title('Turned image (log10 scale)')
            plt.show()

    def extract_spectrum_from_image(self,w=3,ws=[8,30],right_edge=1800):
        self.my_logger.info('\n\tExtracting spectrum from image: spectrum with width 2*%d pixels and background from %d to %d pixels' % (w,ws[0],ws[1]))
        # Make a data copy
        data = np.copy(self.data_rotated)[:,0:right_edge]
        # Sum rotated image profile along y axis
        y0 = int(self.target_pixcoords_rotated[1])
        spectrum2D = np.copy(data[y0-w:y0+w,:])
        xprofile = np.mean(spectrum2D,axis=0)
        # Sum uncertainties in quadrature
        err = np.copy(self.stat_errors_rotated)[:,0:right_edge]
        err2D = np.copy(err[y0-w:y0+w,:])
        xprofile_err = np.sqrt(np.mean(err2D**2,axis=0))
        # Lateral bands to remove sky background
        Ny, Nx =  data.shape
        ymax = min(Ny,y0+ws[1])
        ymin = max(0,y0-ws[1])
        spectrum2DUp = np.copy(data[y0+ws[0]:ymax,:])
        spectrum2DUp = filter_stars_from_bgd(spectrum2DUp,margin_cut=1)
        err_spectrum2DUp = np.copy(err[y0+ws[0]:ymax,:])
        err_spectrum2DUp = filter_stars_from_bgd(err_spectrum2DUp,margin_cut=1)
        xprofileUp = np.nanmedian(spectrum2DUp,axis=0)
        xprofileUp_err = np.sqrt(np.nanmean(err_spectrum2DUp**2,axis=0))
        spectrum2DDown = np.copy(data[ymin:y0-ws[0],:])
        spectrum2DDown = filter_stars_from_bgd(spectrum2DDown,margin_cut=1)
        err_spectrum2DDown = np.copy(err[ymin:y0-ws[0],:])
        err_spectrum2DDown = filter_stars_from_bgd(err_spectrum2DDown,margin_cut=1)
        xprofileDown = np.nanmedian(spectrum2DDown,axis=0)
        xprofileDown_err = np.sqrt(np.nanmean(err_spectrum2DDown**2,axis=0))
        # Subtract mean lateral profile
        xprofile_background = 0.5*(xprofileUp+xprofileDown)
        xprofile_background_err = np.sqrt(0.5*(xprofileUp_err**2+xprofileDown_err**2))
        # Create Spectrum object
        spectrum = Spectrum(Image=self)
        spectrum.data = xprofile - xprofile_background
        spectrum.err = np.sqrt(xprofile_err**2 +  xprofile_background_err**2)
        if parameters.DEBUG:
            spectrum.plot_spectrum()    
        return spectrum

   
    def plot_image(self,scale="lin",title="",units="Image units",plot_stats=False):
        fig, ax = plt.subplots(1,1,figsize=[9.3,8])
        data = np.copy(self.data)
        if plot_stats: data = np.copy(self.stat_errors)
        if scale=="log" or scale=="log10":
            # removes the zeros and negative pixels first
            zeros = np.where(data<=0)
            min_noz = np.min(data[np.where(data>0)])
            data[zeros] = min_noz
            # apply log
            data = np.log10(data)
        im = ax.imshow(data,origin='lower',cmap='rainbow')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        cb = fig.colorbar(im,ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7,prune=None)
        cb.update_ticks()
        cb.set_label('%s (%s scale)' % (units,scale)) #,fontsize=16)
        if title!="": ax.set_title(title)
        plt.show()
        


class Spectrum():
    """ Spectrum class used to store information and methods
    relative to spectra nd their extraction.
    """
    def __init__(self,filename="",Image=None,order=1):
        """
        Args:
            filename (:obj:`str`): path to the image
            Image (:obj:`Image`): copy info from Image object
        """
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.target = None
        self.data = None
        self.err = None
        self.lambdas = None
        self.order = order
        if filename != "" :
            self.filename = filename
            self.load_spectrum(filename)
        if Image is not None:
            self.header = Image.header
            self.date_obs = Image.date_obs
            self.airmass = Image.airmass
            self.expo = Image.expo
            self.filters = Image.filters
            self.filter = Image.filter
            self.disperser = Image.disperser
            self.target = Image.target
            self.target_pixcoords = Image.target_pixcoords
            self.target_pixcoords_rotated = Image.target_pixcoords_rotated
            self.units = Image.units
            self.my_logger.info('\n\tSpectrum info copied from Image')
        self.load_filter()

    def load_filter(self):
        for f in FILTERS:
            if f['label'] == self.filter:               
                parameters.LAMBDA_MIN = f['min']
                parameters.LAMBDA_MAX = f['max']
                self.my_logger.info('\n\tLoad filter %s: lambda between %.1f and %.1f' % (f['label'],parameters.LAMBDA_MIN, parameters.LAMBDA_MAX))
                break

    def plot_spectrum(self,xlim=None,order=1,atmospheric_lines=True,nofit=False):
        xs = self.lambdas
        if xs is None : xs = np.arange(self.data.shape[0])
        fig = plt.figure(figsize=[12,6])
        if self.err is not None:
            plt.errorbar(xs,self.data,yerr=self.err,fmt='ro',lw=1,label='Order %d spectrum' % order,zorder=0)
        else:
            plt.plot(xs,self.data,'r-',lw=2,label='Order %d spectrum' % order)
        if self.lambdas is not None:
            plot_atomic_lines(plt.gca(),redshift=self.target.redshift,atmospheric_lines=atmospheric_lines,hydrogen_only=self.target.hydrogen_only,fontsize=12)
        plt.grid(True)
        plt.xlim([parameters.LAMBDA_MIN,parameters.LAMBDA_MAX])
        plt.ylim(0.,np.max(self.data)*1.2)
        plt.xlabel('$\lambda$ [nm]')
        plt.ylabel(self.units)
        if self.lambdas is None: plt.xlabel('Pixels')
        if xlim is not None :
            plt.xlim(xlim)
            plt.ylim(0.,np.max(self.data[xlim[0]:xlim[1]])*1.2)
        if not nofit and self.lambdas is not None:
            lambda_shift = detect_lines(self.lambdas,self.data,spec_err=self.err,redshift=self.target.redshift,emission_spectrum=self.target.emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=self.target.hydrogen_only,ax=plt.gca(),verbose=False)
        plt.show()

    def calibrate_spectrum(self,xlims=None):
        if xlims == None :
            left_cut, right_cut = [0,self.data.shape[0]]
        else:
            left_cut, right_cut = xlims
        self.data = self.data[left_cut:right_cut]
        pixels = np.arange(left_cut,right_cut,1)-self.target_pixcoords_rotated[0]
        self.lambdas = self.disperser.grating_pixel_to_lambda(pixels,self.target_pixcoords,order=self.order)
        # Cut spectra
        self.lambdas_indices = np.where(np.logical_and(self.lambdas > parameters.LAMBDA_MIN, self.lambdas < parameters.LAMBDA_MAX))[0]
        self.lambdas = self.lambdas[self.lambdas_indices]
        self.data = self.data[self.lambdas_indices]
        if self.err is not None: self.err = self.err[self.lambdas_indices]

    def calibrate_spectrum_with_lines(self,atmospheric_lines=True):
        self.my_logger.warning('\n\tManual settings for tests')
        atmospheric_lines = True
        self.my_logger.info('\n\tCalibrating order %d spectrum...' % self.order)
        # Detect emission/absorption lines and calibrate pixel/lambda 
        D = DISTANCE2CCD-DISTANCE2CCD_ERR
        shift = 0
        shifts = []
        counts = 0
        D_step = DISTANCE2CCD_ERR / 4
        delta_pixels = self.lambdas_indices - int(self.target_pixcoords_rotated[0])
        while D < DISTANCE2CCD+4*DISTANCE2CCD_ERR and D > DISTANCE2CCD-4*DISTANCE2CCD_ERR and counts < 30 :
            self.disperser.D = D
            lambdas_test = self.disperser.grating_pixel_to_lambda(delta_pixels,self.target_pixcoords,order=self.order)
            lambda_shift = detect_lines(lambdas_test,self.data,spec_err=self.err,redshift=self.target.redshift,emission_spectrum=self.target.emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=self.target.hydrogen_only,ax=None,verbose=parameters.DEBUG)
            shifts.append(lambda_shift)
            counts += 1
            if abs(lambda_shift)<0.1 :
                break
            elif lambda_shift > 2 :
                D_step = DISTANCE2CCD_ERR 
            elif 0.5 < lambda_shift < 2 :
                D_step = DISTANCE2CCD_ERR / 4
            elif 0 < lambda_shift < 0.5 :
                D_step = DISTANCE2CCD_ERR / 10
            elif 0 > lambda_shift > -0.5 :
                D_step = -DISTANCE2CCD_ERR / 20
            elif  lambda_shift < -0.5 :
                D_step = -DISTANCE2CCD_ERR / 6
            D += D_step
        shift = np.mean(lambdas_test - self.lambdas)
        self.lambdas = lambdas_test
        detect_lines(self.lambdas,self.data,spec_err=self.err,redshift=self.target.redshift,emission_spectrum=self.target.emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=self.target.hydrogen_only,ax=None,verbose=parameters.DEBUG)
        self.my_logger.info('\n\tWavelenght total shift: %.2fnm (after %d steps)\n\twith D = %.2f mm (DISTANCE2CCD = %.2f +/- %.2f mm, %.1f sigma shift)' % (shift,len(shifts),D,DISTANCE2CCD,DISTANCE2CCD_ERR,(D-DISTANCE2CCD)/DISTANCE2CCD_ERR))
        if parameters.VERBOSE or parameters.DEBUG:
            self.plot_spectrum(xlim=None,order=self.order,atmospheric_lines=atmospheric_lines,nofit=False)

    def save_spectrum(self,output_filename,overwrite=False):
        hdu = fits.PrimaryHDU()
        hdu.data = [self.lambdas,self.data,self.err]
        self.header['UNIT1'] = "nanometer"
        self.header['UNIT2'] = self.units
        self.header['COMMENTS'] = 'First column gives the wavelength in unit UNIT1, second column gives the spectrum in unit UNIT2, third column the corresponding errors.'
        hdu.header = self.header
        hdu.writeto(output_filename,overwrite=overwrite)
        self.my_logger.info('\n\tSpectrum saved in %s' % output_filename)


    def load_spectrum(self,input_filename):
        if os.path.isfile(input_filename):
            hdu = fits.open(input_filename)
            self.header = hdu[0].header
            self.lambdas = hdu[0].data[0]
            self.data = hdu[0].data[1]
            if len(hdu[0].data)==2:
                self.err = hdu[0].data[2]
            extract_info_from_CTIO_header(self, self.header)
            if self.header['TARGET'] != "":
                self.target=Target(self.header['TARGET'],verbose=parameters.VERBOSE)
            self.my_logger.info('\n\tSpectrum loaded from %s' % input_filename)
        else:
            self.my_logger.info('\n\tSpectrum file %s not found' % input_filename)
        

def Spectractor(filename,outputdir,guess,target):
    """ Spectractor
    Main function to extract a spectrum from an image

    Args:
        filename (:obj:`str`): path to the image
        outputdir (:obj:`str`): path to the output directory
    """
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTOR')
    # Load reduced image
    image = Image(filename,target=target)
    # Set output path
    ensure_dir(outputdir)
    output_filename = filename.split('/')[-1]
    output_filename = output_filename.replace('.fits','_spectrum.fits')
    output_filename = os.path.join(outputdir,output_filename)
    # Cut the image
  
    # Find the exact target position in the raw cut image: several methods
    my_logger.info('\n\tSearch for the target in the image...')
    target_pixcoords = image.find_target(guess)
    # Rotate the image: several methods
    image.turn_image()
    # Find the exact target position in the rotated image: several methods
    my_logger.info('\n\tSearch for the target in the rotated image...')
    target_pixcoords_rotated = image.find_target(guess,rotated=True)
    # Subtract background and bad pixels
    spectrum = image.extract_spectrum_from_image()
    # Calibrate the spectrum
    spectrum.calibrate_spectrum()
    spectrum.calibrate_spectrum_with_lines()
    # Subtract second order

    # Cut in wavelength

    # Load target and its spectrum

    # Run libratran ?

    # Save the spectra
    spectrum.save_spectrum(output_filename,overwrite=True)

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
        
        
    filename = "../../CTIODataJune2017_reducedRed/data_05jun17/reduc_20170605_00.fits"
    filename = "notebooks/fits/trim_20170605_007.fits"
    guess = [745,643]
    target = "3C273"

    #filename="../CTIOAnaJun2017/ana_05jun17/OverScanRemove/trim_images/trim_20170605_029.fits"
    #guess = [814, 585]
    #target = "PNG321.0+3.9"
    #filename="../CTIOAnaJun2017/ana_29may17/OverScanRemove/trim_images/trim_20170529_150.fits"
    #guess = [720, 670]
    #target = "HD185975"
    #filename="../CTIOAnaJun2017/ana_31may17/OverScanRemove/trim_images/trim_20170531_150.fits"
    #guess = [840, 530]
    #target = "HD205905"

    Spectractor(filename,opts.output_directory,guess,target)

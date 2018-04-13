import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import copy
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
import astropy.units as units

import parameters 
from tools import *
from dispersers import *
from targets import *
from spectroscopy import *


class Image():

    def __init__(self,filename,target=""):
        """
        Args:
            filename (:obj:`str`): path to the image
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.filename = filename
        self.units = 'ADU'
        self.load(filename)
        # Load the target if given
        self.target = None
        self.target_pixcoords = None
        self.target_pixcoords_rotated = None
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
        IMSIZE = int(self.header['XLENGTH'])
        parameters.PIXEL2ARCSEC = float(self.header['XPIXSIZE'])
        if self.header['YLENGTH'] != IMSIZE:
            self.my_logger.warning('\n\tImage rectangular: X=%d pix, Y=%d pix' % (IMSIZE, self.header['YLENGTH']))
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
        l = IMSIZE
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
        self.units = 'ADU/s'
        
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

    def find_target_init(self,guess,rotated=False):
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
        return sub_image,x0,y0,Dx,Dy

 
    def find_target_1Dprofile(self,guess,rotated=False):
        """
        Find precisely the position of the targeted object.
        
        Args:
            guess (:obj:`list`): [x,y] guessed position of th target
        """
        sub_image,x0,y0,Dx,Dy = self.find_target_init(guess=guess,rotated=rotated)
        NY, NX = sub_image.shape
        X = np.arange(NX)
        Y = np.arange(NY)
        # compute profiles
        profile_X_raw=np.sum(sub_image,axis=0)
        profile_Y_raw=np.sum(sub_image,axis=1)
        # fit and subtract smooth polynomial background
        # with 3sigma rejection of outliers (star peaks)
        bkgd_X = fit_poly1d_outlier_removal(X,profile_X_raw,order=2)
        bkgd_Y = fit_poly1d_outlier_removal(Y,profile_Y_raw,order=2)
        profile_X = profile_X_raw - bkgd_X(X) #np.min(profile_X)
        profile_Y = profile_Y_raw - bkgd_Y(Y) #np.min(profile_Y)
        avX,sigX=weighted_avg_and_std(X,profile_X**4) 
        avY,sigY=weighted_avg_and_std(Y,profile_Y**4)
        if profile_X[int(avX)] < 0.8*np.max(profile_X) :
            self.my_logger.warning('\n\tX position determination of the target probably wrong') 
        if profile_Y[int(avY)] < 0.8*np.max(profile_Y) :
            self.my_logger.warning('\n\tY position determination of the target probably wrong')
        # compute target position
        theX=x0-Dx+avX
        theY=y0-Dy+avY
        if parameters.DEBUG:
            profile_X_max=np.max(profile_X_raw)*1.2
            profile_Y_max=np.max(profile_Y_raw)*1.2

            f, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(15,4))
            im = ax1.imshow(np.log10(sub_image),origin='lower',cmap='jet')
            cb = f.colorbar(im,ax=ax1)
            cb.formatter.set_powerlimits((0, 0))
            cb.locator = MaxNLocator(7,prune=None)
            cb.update_ticks()
            cb.set_label('%s (log10 scale)' % (self.units)) #,fontsize=16)
            ax1.scatter([avX],[avY],marker='o',s=100,facecolors='none',edgecolors='k')
            ax1.grid(True)
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')

            ax2.plot(X,profile_X_raw,'r-',lw=2)
            ax2.plot(X,bkgd_X(X),'g--',lw=2,label='bkgd')
            ax2.axvline(Dx,color='y',linestyle='-',label='old',lw=2)
            ax2.axvline(avX,color='b',linestyle='-',label='new',lw=2)
            ax2.grid(True)
            ax2.set_xlabel('X (pixels)')
            ax2.legend(loc=1)

            ax3.plot(Y,profile_Y_raw,'r-',lw=2)
            ax3.plot(Y,bkgd_Y(Y),'g--',lw=2,label='bkgd')
            ax3.axvline(Dy,color='y',linestyle='-',label='old',lw=2)
            ax3.axvline(avY,color='b',linestyle='-',label='new',lw=2)
            ax3.grid(True)
            ax3.set_xlabel('Y (pixels)')
            ax3.legend(loc=1)
            f.tight_layout()
            plt.show()

        self.my_logger.info('\n\tX,Y target position in pixels: %.3f,%.3f' % (theX,theY))
        if rotated:
            self.target_pixcoords_rotated = [theX,theY]
        else:
            self.target_pixcoords = [theX,theY]
        return [theX,theY]

    def find_target_2Dprofile(self,guess,rotated=False):
        """
        Find precisely the position of the targeted object.
        
        Args:
            guess (:obj:`list`): [x,y] guessed position of th target
        """
        sub_image,x0,y0,Dx,Dy = self.find_target_init(guess=guess,rotated=rotated)
        NY, NX = sub_image.shape
        X = np.arange(NX)
        Y = np.arange(NY)
        Y, X = np.mgrid[:NY,:NX]
        # fit and subtract smooth polynomial background
        # with 3sigma rejection of outliers (star peaks)
        bkgd_2D = fit_poly2d_outlier_removal(X,Y,sub_image,order=2)
        sub_image_subtracted = sub_image-bkgd_2D(X,Y)
        # find a first guess of the target position
        avX,sigX = weighted_avg_and_std(X,(sub_image_subtracted)**4)
        avY,sigY = weighted_avg_and_std(Y,(sub_image_subtracted)**4)
        # fit a 2D gaussian close to this position
        guess = [np.max(sub_image_subtracted),avX,avY,2,2,0]
        mean_prior = 10 # in pixels
        bounds = [ [1,avX-mean_prior,avY-mean_prior,1,1,-np.pi], [2*np.max(sub_image_subtracted),avX+mean_prior,avY+mean_prior,10,10,np.pi] ]
        gauss2D = fit_gauss2d_outlier_removal(X,Y,sub_image_subtracted,guess=guess,bounds=bounds, sigma = 3, circular = True)
        # compute target positions
        avX = gauss2D.x_mean.value
        avY = gauss2D.y_mean.value
        theX=x0-Dx+avX
        theY=y0-Dy+avY
        if sub_image_subtracted[int(avY),int(avX)] < 0.8*np.max(sub_image_subtracted) :
            self.my_logger.warning('\n\tX,Y position determination of the target probably wrong') 
         # debugging plots
        if parameters.DEBUG:
            f, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(15,4))
            im = ax1.imshow(sub_image,origin='lower',cmap='jet')
            cb = f.colorbar(im,ax=ax1)
            #cb.formatter.set_powerlimits((0, 0))
            cb.locator = MaxNLocator(7,prune=None)
            cb.update_ticks()
            cb.set_label('Original image (%s)' % (self.units)) #,fontsize=16)
            ax1.scatter([Dx],[Dy],marker='o',s=100,facecolors='none',edgecolors='w',label='old')
            ax1.scatter([avX],[avY],marker='o',s=100,facecolors='none',edgecolors='k',label='new')
            ax1.grid(True)
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            ax1.legend(loc=1)

            im2 = ax2.imshow(bkgd_2D(X,Y)+gauss2D(X,Y),origin='lower',cmap='jet',vmin=np.min(sub_image),vmax=np.max(sub_image))
            cb = f.colorbar(im2,ax=ax2)
            #cb.formatter.set_powerlimits((0, 0))
            cb.locator = MaxNLocator(7,prune=None)
            cb.update_ticks()
            cb.set_label('Background + Gauss (%s)' % (self.units)) #,fontsize=16)
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
            ax2.legend(loc=1)

            im3 = ax3.imshow(sub_image_subtracted-gauss2D(X,Y),origin='lower',cmap='jet')
            cb = f.colorbar(im3,ax=ax3)
            #cb.formatter.set_powerlimits((0, 0))
            cb.locator = MaxNLocator(7,prune=None)
            cb.update_ticks()
            cb.set_label('Background+Gauss subtracted image (%s)' % (self.units)) #,fontsize=16)
            ax3.scatter([Dx],[Dy],marker='o',s=100,facecolors='none',edgecolors='w',label='old')
            ax3.scatter([avX],[avY],marker='o',s=100,facecolors='none',edgecolors='k',label='new')
            ax3.grid(True)
            ax3.set_xlabel('X (pixels)')
            ax3.set_ylabel('Y (pixels)')
            ax3.legend(loc=1)

            f.tight_layout()
            plt.show()

        self.my_logger.info('\n\tX,Y target position in pixels: %.3f,%.3f' % (theX,theY))
        if rotated:
            self.target_pixcoords_rotated = [theX,theY]
        else:
            self.target_pixcoords = [theX,theY]
        return [theX,theY]

    def compute_rotation_angle_hessian(self, deg_threshold = 10, width_cut = YWINDOW, right_edge = IMSIZE-200, margin_cut=12):
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
        theta_critical = 180.*np.arctan(10./IMSIZE)/np.pi
        if abs(theta_median-theta_guess)>theta_critical:
            self.my_logger.warning('\n\tInterpolated angle and fitted angle disagrees with more than 10 pixels over %d pixels:  %.2f vs %.2f' % (IMSIZE,theta_median,theta_guess))
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
            margin=200
            f, (ax1,ax2) = plt.subplots(2,1,figsize=[8,8])
            y0 = int(self.target_pixcoords[1])
            ax1.imshow(np.log10(self.data[y0-parameters.YWINDOW:y0+parameters.YWINDOW,margin:-margin]),origin='lower',cmap='rainbow',aspect="auto")
            ax1.plot([0,self.data.shape[0]-2*margin],[parameters.YWINDOW,parameters.YWINDOW],'k-')
            if self.target_pixcoords is not None:
                ax1.scatter(self.target_pixcoords[0]-margin,parameters.YWINDOW,marker='o',s=100,edgecolors='k',facecolors='none')
            ax1.grid(color='white', ls='solid')
            ax1.grid(True)
            ax1.set_title('Raw image (log10 scale)')
            ax2.imshow(np.log10(self.data_rotated[y0-parameters.YWINDOW:y0+parameters.YWINDOW,margin:-margin]),origin='lower',cmap='rainbow',aspect="auto")
            ax2.plot([0,self.data_rotated.shape[0]-2*margin],[parameters.YWINDOW,parameters.YWINDOW],'k-')
            if self.target_pixcoords_rotated is not None:
                ax2.scatter(self.target_pixcoords_rotated[0],self.target_pixcoords_rotated[1],marker='o',s=100,edgecolors='k',facecolors='none')
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

   
    def plot_image(self,scale="lin",title="",units="Image units",plot_stats=False,target_pixcoords=None):
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
        im = ax.imshow(data,origin='lower',cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if target_pixcoords is not None:
            plt.scatter(target_pixcoords[0],target_pixcoords[1],marker='o',s=100,edgecolors='k',facecolors='none',label='Target')
        cb = fig.colorbar(im,ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7,prune=None)
        cb.update_ticks()
        cb.set_label('%s (%s scale)' % (units,scale)) #,fontsize=16)
        if title!="": ax.set_title(title)
        plt.legend()
        plt.show()
        


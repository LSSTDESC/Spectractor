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

    def convert_to_ADU_units(self):
        self.data *= self.expo
        self.units = 'ADU'            
        
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
            self.plot_image_simple(ax1,data=sub_image,scale="log",title="",units=self.units,plot_stats=False,target_pixcoords=[avX,avY])
            ax1.legend(loc=1)

            ax2.plot(X,profile_X_raw,'r-',lw=2)
            ax2.plot(X,bkgd_X(X),'g--',lw=2,label='bkgd')
            ax2.axvline(Dx,color='y',linestyle='-',label='old',lw=2)
            ax2.axvline(avX,color='b',linestyle='-',label='new',lw=2)
            ax2.grid(True)
            ax2.set_xlabel('X [pixels]')
            ax2.legend(loc=1)

            ax3.plot(Y,profile_Y_raw,'r-',lw=2)
            ax3.plot(Y,bkgd_Y(Y),'g--',lw=2,label='bkgd')
            ax3.axvline(Dy,color='y',linestyle='-',label='old',lw=2)
            ax3.axvline(avY,color='b',linestyle='-',label='new',lw=2)
            ax3.grid(True)
            ax3.set_xlabel('Y [pixels]')
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
        self.target_gauss2D = gauss2D
        self.target_bkgd2D = bkgd_2D
        if sub_image_subtracted[int(avY),int(avX)] < 0.8*np.max(sub_image_subtracted) :
            self.my_logger.warning('\n\tX,Y position determination of the target probably wrong') 
         # debugging plots
        if parameters.DEBUG:
            f, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(15,4))
            self.plot_image_simple(ax1,data=sub_image,scale="lin",title="",units=self.units,target_pixcoords=[avX,avY])
            ax1.scatter([Dx],[Dy],marker='o',s=100,facecolors='none',edgecolors='w',label='old')
            ax1.legend(loc=1)
            
            self.plot_image_simple(ax2,data=bkgd_2D(X,Y)+gauss2D(X,Y),scale="lin",title="",units='Background + Gauss (%s)' % (self.units))
            ax2.legend(loc=1)

            self.plot_image_simple(ax3,data=sub_image_subtracted-gauss2D(X,Y),scale="lin",title="",units='Background+Gauss subtracted image (%s)' % (self.units),target_pixcoords=[avX,avY])
            ax3.scatter([Dx],[Dy],marker='o',s=100,facecolors='none',edgecolors='w',label='old')
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
            y0 = int(self.target_pixcoords[1])
            f, (ax1,ax2) = plt.subplots(2,1,figsize=[8,8])
            self.plot_image_simple(ax1,data=self.data[y0-parameters.YWINDOW:y0+parameters.YWINDOW,margin:-margin],scale="log",title='Raw image (log10 scale)',units=self.units,target_pixcoords=(self.target_pixcoords[0]-margin,parameters.YWINDOW))
            ax1.plot([0,self.data.shape[0]-2*margin],[parameters.YWINDOW,parameters.YWINDOW],'k-')
            self.plot_image_simple(ax2,data=self.data_rotated[y0-parameters.YWINDOW:y0+parameters.YWINDOW,margin:-margin],scale="log",title='Turned image (log10 scale)',units=self.units,target_pixcoords=self.target_pixcoords_rotated)
            ax2.plot([0,self.data_rotated.shape[0]-2*margin],[parameters.YWINDOW,parameters.YWINDOW],'k-')
            plt.show()

    def extract_spectrum_from_image(self,w=10,ws=[20,30],right_edge=1800):
        """
            extract_spectrum_from_image(self,w=10,ws=[20,30],right_edge=1800):
            
                Extract the 1D spectrum from the image.
                Remove background estimated from the lateral Bands
                    w: half width of central region where the spectrum is supposed to be
                    ws=[8,30]  : up/down region where the sky background is estimated 
                    right_edge : position above which no pixel should be used
        """
        self.my_logger.info('\n\tExtracting spectrum from image: spectrum with width 2*%d pixels and background from %d to %d pixels' % (w,ws[0],ws[1]))
        # Make a data copy
        data = np.copy(self.data_rotated)[:,0:right_edge]
        err = np.copy(self.stat_errors_rotated)[:,0:right_edge]
        # Lateral bands to remove sky background
        Ny, Nx =  data.shape
        y0 = int(self.target_pixcoords_rotated[1])
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
        # Sum rotated image profile along y axis
        # Subtract mean lateral profile
        xprofile_background = 0.5*(xprofileUp+xprofileDown)
        xprofile_background_err = np.sqrt(0.5*(xprofileUp_err**2+xprofileDown_err**2))
        spectrum2D = np.copy(data[y0-w:y0+w,:])
        xprofile = np.sum(spectrum2D,axis=0) - 2*w*xprofile_background
        # Sum uncertainties in quadrature
        err2D = np.copy(err[y0-w:y0+w,:])
        xprofile_err = np.sqrt( np.sum(err2D**2,axis=0) + (2*w*xprofile_background_err)**2 )
        # Create Spectrum object
        spectrum = Spectrum(Image=self)
        spectrum.data = xprofile
        spectrum.err = xprofile_err
        if parameters.DEBUG:
            spectrum.plot_spectrum()    
        return spectrum

   
    def plot_image_simple(self,ax,data=None,scale="lin",title="",units="Image units",plot_stats=False,target_pixcoords=None):
        if data is None: data = np.copy(self.data)
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
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        cb = plt.colorbar(im,ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7,prune=None)
        cb.update_ticks()
        cb.set_label('%s (%s scale)' % (units,scale)) #,fontsize=16)
        if title!="": ax.set_title(title)
        if target_pixcoords is not None:
            ax.scatter(target_pixcoords[0],target_pixcoords[1],marker='o',s=100,edgecolors='k',facecolors='none',label='Target',linewidth=2)
        
    def plot_image(self,data=None,scale="lin",title="",units="Image units",plot_stats=False,target_pixcoords=None):
        fig, ax = plt.subplots(1,1,figsize=[9.3,8])
        self.plot_image_simple(ax,data=data,scale=scale,title=title,units=units,plot_stats=plot_stats,target_pixcoords=target_pixcoords)
        plt.legend()
        plt.show()

    def save_image(self,output_filename,overwrite=False):
        hdu = fits.PrimaryHDU()
        hdu.data = self.data
        hdu.header = self.header
        hdu.writeto(output_filename,overwrite=overwrite)
        self.my_logger.info('\n\tImage saved in %s' % output_filename)

        

    def extract_spectrum_from_image_sylvie(self,w=3,ws=[8,30],right_edge=1800,meanflag=False,filterstarflag=False):
        """
            extract_spectrum_from_image(self,w=3,ws=[8,30],right_edge=1800,meanflag=False,filterstarflag=False):
            
                Extract the 1D spectrum from the image.
                Remove background estimated from the lateral Bands

                    w: half width of central region where the spectrum is supposed to be
                    ws=[8,30]  : up/down region where the sky background is estimated 

                    right_edge : position above which no pixel should be used
                    meanflag   : flag to compute signal from the mean
                    filterflag : flag to clean region from stars 
                
                Original values : 
                    w=3,ws=[8,30],right_edge=1800,meanflag=True,filterstarflag=True
                    
                New Values ( Sylvie April 12th)
                    w=10,ws=[20,30],right_edge=1800,meanflag=False,filterstarflag=False
                
                
        """
        self.my_logger.info('\n\tExtracting spectrum from image: spectrum with width 2*%d pixels and background from %d to %d pixels' % (w,ws[0],ws[1]))
        # Make a data copy of the image portion
        data = np.copy(self.data_rotated)[:,0:right_edge]
        # Sum rotated image profile along y axis
        y0 = int(self.target_pixcoords_rotated[1])
        x0 = int(self.target_pixcoords_rotated[0])
        # 1 first consider a big area around the spectrum
        WBIG=50
        spectrum2Dbig = np.copy(data[y0-WBIG:y0+WBIG,:])
        if parameters.DEBUG:
           plt.figure(figsize=(20,5))
           img=plt.imshow(spectrum2Dbig,origin='lower',cmap='jet',vmin=0,vmax=500)
           cbar=plt.colorbar(img,orientation='horizontal')
           plt.plot([0,right_edge],[WBIG,WBIG],'y-',lw=2)
           plt.title("extract_spectrum_from_image : spectrum2DBig : with central star")
           plt.grid()
           plt.show()
           
        # 2 erase the central star   
        spectrum2Dbig2 = np.copy(spectrum2Dbig)
        spectrum2Dbig2[:,x0-2*WBIG:x0+2*WBIG]=0
        SPECMAX=spectrum2Dbig2[:,:right_edge].max()
        if parameters.DEBUG:
           plt.figure(figsize=(20,5))
           img=plt.imshow(spectrum2Dbig2,origin='lower',cmap='jet',vmin=0,vmax=SPECMAX)
           cbar=plt.colorbar(img,orientation='horizontal')
           plt.plot([0,right_edge],[WBIG,WBIG],'y-',lw=2)
           plt.title("extract_spectrum_from_image : spectrum2DBig2 : with central star ERASED")
           plt.grid()
           plt.show()   
        # 3 find the new central y0
        yprofileBig=np.sum(spectrum2Dbig2[:,x0+2*WBIG:right_edge],axis=1)
        delta_y0=np.where(yprofileBig==yprofileBig.max())[0][0]-WBIG
        print ' Delta y0 =',delta_y0
        
        if parameters.DEBUG or parameters.VERBOSE:
             plt.figure(figsize=(8,4))
             plt.plot(yprofileBig,'b-')
             plt.plot([delta_y0+WBIG,delta_y0+WBIG],[0,yprofileBig.max()],'r-',lw=2)
             plt.plot([delta_y0+WBIG-w,delta_y0+WBIG-w],[0,yprofileBig.max()/2],'r:',lw=2)
             plt.plot([delta_y0+WBIG+w,delta_y0+WBIG+w],[0,yprofileBig.max()/2],'r:',lw=2)
             
             plt.plot([delta_y0+WBIG-ws[0],delta_y0+WBIG-ws[0]],[0,yprofileBig.max()/5],'g:',lw=2)
             plt.plot([delta_y0+WBIG+ws[0],delta_y0+WBIG+ws[0]],[0,yprofileBig.max()/5],'g:',lw=2)
             plt.plot([delta_y0+WBIG-ws[1],delta_y0+WBIG-ws[1]],[0,yprofileBig.max()/5],'g:',lw=2)
             plt.plot([delta_y0+WBIG+ws[1],delta_y0+WBIG+ws[1]],[0,yprofileBig.max()/5],'g:',lw=2)
             
             ws=[8,30]
             
             plt.title('yprofile: check the center')
             plt.grid()
             plt.xlabel('y (pix)')
             plt.show()
             
             self.my_logger.info('\n\t extract_spectrum_from_image::Correct vertical center delta_y0=%s' % (delta_y0))
        
        # readjust the center of vertical profile
        y0=y0+delta_y0
        
        # 3 Extract the image corresponding to the spectrum
        spectrum2D = np.copy(data[y0-w:y0+w,:])
        spectrum2Dsmall = np.copy(spectrum2D)
        spectrum2Dsmall[:,x0-2*WBIG:x0+2*WBIG]=0
        yprofilesmall=np.sum(spectrum2Dsmall[:,x0+2*WBIG:right_edge],axis=1)

        
        if parameters.DEBUG:
            plt.figure(figsize=(8,4))
            plt.plot(yprofilesmall,'b-')
            plt.plot([w,w],[0,yprofilesmall.max()],'r-',lw=2)
 
            plt.title('yprofile: selected spectra')
            plt.grid()
            plt.xlabel('y (pix)')
            plt.show() 
            
           
            plt.figure(figsize=(20,5))
            img=plt.imshow(spectrum2D,origin='lower',cmap='jet',vmin=0,vmax=SPECMAX)
            cbar=plt.colorbar(img,orientation='horizontal')
            plt.plot([0,right_edge],[w,w],'y-',lw=2)
            plt.title("extract_spectrum_from_image : spectrum2D")
            plt.grid()
            plt.show()


            
        
        # Simulatio can only provide the sum (SDC)
        if meanflag:  # Jeremy's method
            xprofile = np.mean(spectrum2D,axis=0)
        else:             # Sylvie's method unsing the sum in a band y of width 2*w
            xprofile = np.sum(spectrum2D,axis=0)
            
            
        # Sum uncertainties in quadrature
        err = np.copy(self.stat_errors_rotated)[:,0:right_edge]
        err2D = np.copy(err[y0-w:y0+w,:])
        if meanflag:
            xprofile_err = np.sqrt(np.mean(err2D**2,axis=0))
        else:
            xprofile_err = np.sqrt(np.sum(err2D**2,axis=0))
        
        # Lateral bands to remove sky background
        Ny, Nx =  data.shape
        ymax = min(Ny,y0+ws[1])
        ymin = max(0,y0-ws[1])
        
        # Upper band with width ws[1]-ws[0]
        spectrum2DUp = np.copy(data[y0+ws[0]:ymax,:])
        if filterstarflag:
            spectrum2DUp = filter_stars_from_bgd(spectrum2DUp,margin_cut=1)
        
        err_spectrum2DUp = np.copy(err[y0+ws[0]:ymax,:])
        if filterstarflag:
            err_spectrum2DUp = filter_stars_from_bgd(err_spectrum2DUp,margin_cut=1)
        
        if meanflag:
            xprofileUp = np.nanmedian(spectrum2DUp,axis=0)
            xprofileUp_err = np.sqrt(np.nanmean(err_spectrum2DUp**2,axis=0))
        else:
            xprofileUp = np.sum(spectrum2DUp,axis=0)
            xprofileUp_err = np.sqrt(np.sum(err_spectrum2DUp**2,axis=0))
            
         # lower band with width ws[1]-ws[0]
        spectrum2DDown = np.copy(data[ymin:y0-ws[0],:])
        if filterstarflag:
            spectrum2DDown = filter_stars_from_bgd(spectrum2DDown,margin_cut=1)
        
        err_spectrum2DDown = np.copy(err[ymin:y0-ws[0],:])
        if filterstarflag:
            err_spectrum2DDown = filter_stars_from_bgd(err_spectrum2DDown,margin_cut=1)
        
        
        if meanflag:
            xprofileDown = np.nanmedian(spectrum2DDown,axis=0)
            xprofileDown_err = np.sqrt(np.nanmean(err_spectrum2DDown**2,axis=0))
        else:
            xprofileDown = np.sum(spectrum2DDown,axis=0)
            xprofileDown_err = np.sqrt(np.sum(err_spectrum2DDown**2,axis=0))
        
        if parameters.DEBUG:
            plt.figure(figsize=(20,5))
            plt.subplot(311)
            plt.imshow( spectrum2DUp ,origin='lower',cmap='jet',vmin=0,vmax=SPECMAX/5.)
            plt.grid()
            plt.subplot(312) 
            img=plt.imshow( spectrum2D ,origin='lower',cmap='jet',vmin=0,vmax=SPECMAX/5.)
            plt.subplot(313) 
            plt.grid()
            plt.imshow( spectrum2DDown ,origin='lower',cmap='jet',vmin=0,vmax=SPECMAX/5.)
            plt.grid()
            #plt.subplot(111)
            #cbar=plt.colorbar(img)
            plt.suptitle("extract_spectrum_from_image() : spectrum2D Backgrounds")
            #plt.grid()
            plt.show()
        
        
        
      
        
        # Subtract mean lateral profile by renormalisationof the surface
        xprofile_background = 0.5*(xprofileUp+xprofileDown)*2*w/(ws[1]-ws[0])
        
        xprofile_background_err = np.sqrt(0.5*(xprofileUp_err**2+xprofileDown_err**2))
        
        
        if parameters.DEBUG:
            
            plt.figure(figsize=(8,4))
            plt.plot(xprofile,'b-')
            plt.plot(xprofile_background,'r-')
            plt.ylim(0.,xprofile.max())
            plt.grid()
            plt.show()
            
            plt.figure(figsize=(8,4))
            plt.plot(xprofile,'b-')
            plt.plot(xprofile_background,'r-')
            plt.ylim(0.,xprofile.max()/20.)
            plt.grid()
            plt.show()
        
        
        # Suppressed code to select spectrum in 90% CL band
        if 0:
            # first check about background subtraction
            spectrum2D_nobkg=np.copy(spectrum2D)-xprofile_background
            they=np.arange(spectrum2D_nobkg.shape[0])
            thex=np.arange(spectrum2D_nobkg.shape[1])
        
            all_aver=np.zeros(spectrum2D_nobkg.shape[1])
            all_sig=np.zeros(spectrum2D_nobkg.shape[1])
      
            # compute average and sigma
            for x in thex:
                if np.sum(spectrum2D_nobkg[:,x]>5) and x<right_edge:
                    all_aver[x],all_sig[x]=weighted_avg_and_std(they, spectrum2D_nobkg[:,x])
        
            indexes_wthsig=np.where(all_sig>3)[0]
            indexes_nosig=np.where(all_sig<=3)[0]
        
        
            av_sig=np.median(all_sig[indexes_wthsig])
            
            if parameters.VERBOSE:
                self.my_logger.info('\n\t extract_spectrum_from_image:: average sigma=%4.5f' % (av_sig))
        
        
            all_aver[indexes_nosig]=10
            all_sig[indexes_nosig]=av_sig
        
            if parameters.DEBUG:
            
                plt.figure(figsize=(20,5))
                img=plt.imshow( spectrum2D_nobkg ,origin='lower',cmap='jet',vmin=0,vmax=SPECMAX)
                cbar=plt.colorbar(img,orientation='horizontal')            
                plt.grid()
                plt.show()
            
                plt.figure(figsize=(16,8))
                for x in thex:
                    if (x> x0+2*WBIG) and (all_aver[x]>0) and (x<right_edge) :
                        if x%10==0: # sample 10%
                            plt.plot(spectrum2D_nobkg[:,x])
                plt.xlabel('ypix')
                plt.title('transverse profiles')
                plt.grid()
                plt.show()
            
                up=all_aver+1.645*all_sig
                do=all_aver-1.645*all_sig
            
           
     
                plt.figure(figsize=(16,4))
                #plt.errorbar(thex,all_aver,yerr=1.645*all_sig,color='red',fmt='o',lw=2)
                plt.plot(thex,all_aver,'k-',lw=2)           
                plt.fill_between(thex,up, do, alpha=.25)          
                plt.ylim(0,spectrum2D_nobkg.shape[0])
                plt.title("Confidence belt 90% CL")
                plt.xlabel('xpix')
                plt.ylabel('ypix')
                plt.grid()
                plt.show()
        
        
            # accumulate the signal in 90% CL
            # init
            xprofile_sum=np.zeros(spectrum2D.shape[1])
            xprofile_width_inCL=np.zeros(spectrum2D.shape[1])
            xprofile_sum_bg=np.zeros(spectrum2D.shape[1])
            for x in thex:                
                if  x<right_edge:
                    y_max= all_aver[x]+1.645*all_sig[x]
                    y_min= all_aver[x]-1.645*all_sig[x]
                    xprofile_width_inCL[x]=ymax-ymin
                    index_sel=np.where(np.logical_and(they>=y_min,they<=y_max))[0]
                    xprofile_sum[x]=np.sum(spectrum2D[index_sel,x])
                    xprofile_sum_bg[x] = 0.5*(xprofileUp[x]+xprofileDown[x])*xprofile_width_inCL[x]/(ws[1]-ws[0])
 
        
            SPECMAX=xprofile_sum[x0+2*WBIG:right_edge].max()
    
            if parameters.DEBUG:
                plt.figure(figsize=(8,4))
                plt.plot(xprofile_sum,'b-')
                plt.plot(xprofile_sum_bg,'r-')
                plt.ylim(0.,SPECMAX)
                plt.grid()
                plt.title("Spectrum 90%CL")
                plt.show()
            
                plt.figure(figsize=(8,4))
                plt.plot(xprofile_sum,'b-')
                plt.plot(xprofile_sum_bg,'r-')
                plt.ylim(0.,SPECMAX/20.)
                plt.grid()
                plt.title("Spectrum 90%CL")
                plt.show()

       
        # Create Spectrum object
        spectrum = Spectrum(Image=self)
        spectrum.data = xprofile - xprofile_background
        spectrum.err = np.sqrt(xprofile_err**2 +  xprofile_background_err**2)
        if parameters.DEBUG:
            spectrum.plot_spectrum()    
        return spectrum


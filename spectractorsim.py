import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import sys,os
import copy
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

from scipy.interpolate import interp1d

sys.path.append("../Spectractor")

from tools import *
#from holo_specs import *
from targets import *
from optics import *
import parameters 


import libsimulateTranspCTIOScattAbsAer as atmsim
import libCTIOTransm as ctio

#
WLMIN=300. # Minimum wavelength : PySynPhot works with Angstrom
WLMAX=1100. # Minimum wavelength : PySynPhot works with Angstrom

NBWLBINS=800 # Number of bins between WLMIN and WLMAX
BinWidth=(WLMAX-WLMIN)/float(NBWLBINS) # Bin width in Angstrom
WL=np.linspace(WLMIN,WLMAX,NBWLBINS) # Array of wavelength in Angstrom


# specify parameters for the atmospheric grid

#aerosols
#NB_AER_POINTS=20
NB_AER_POINTS=3
AER_MIN=0.
AER_MAX=0.1

#ozone
#NB_OZ_POINTS=5
NB_OZ_POINTS=3
OZ_MIN=200
OZ_MAX=300

# pwv
#NB_PWV_POINTS=11
NB_PWV_POINTS=3
PWV_MIN=0.
PWV_MAX=10.

# definition of the grid
AER_Points=np.linspace(AER_MIN,AER_MAX,NB_AER_POINTS)
OZ_Points=np.linspace(OZ_MIN,OZ_MAX,NB_OZ_POINTS)
PWV_Points=np.linspace(PWV_MIN,PWV_MAX,NB_PWV_POINTS)

# total number of points
NB_ATM_POINTS=NB_AER_POINTS*NB_OZ_POINTS*NB_PWV_POINTS

#  column 0 : count number
#  column 1 : aerosol value
#  column 2 : pwv value
#  column 3 : ozone value
#  column 4 : data start 
#
index_atm_count=0
index_atm_aer=1
index_atm_pwv=2
index_atm_oz=3
index_atm_data=4

NB_atm_HEADER=5
NB_atm_DATA=len(WL)-1


class SED():
    """
    SED()
    
    """
    def __init__(self,target=""):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.targetname=target
        self.target=None
        self.wl=None
        self.flux=None
        self.sed=None
        
    def get_sed(self):
        self.target=Target(self.targetname,verbose=parameters.VERBOSE)
        self.wl=self.target.wavelengths[-1]
        self.flux=self.target.spectra[-1]
        func=interp1d(self.wl,self.flux,kind='linear') 
        self.sed=func(WL)
        return self.sed

    def plot_sed(self):
        self.target.plot_spectra()
        plt.figure()
        plt.plot(WL,self.sed,'r-',lw=2,label='sed')
        plt.legend()
        plt.xlabel('$\lambda$ [nm]')
        plt.ylabel('Flux')
        plt.title('SED')
        plt.legend()
        plt.grid()
        plt.show()


#----------------------------------------------------------------------------------
class Disperser():
    """
    Disperser():
        
    """
    def __init__(self,dispersername=""):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.disperser=dispersername
        self.td=0.0
        
    def load_transm(self):
        self.td=np.ones(len(WL))
        return self.td
    
    def plot_transm(self,xlim=None):
        plt.figure()
        if(len(self.td)!=0):
            plt.plot(WL,self.td,'b-',label=dispersername)
            plt.legend()
            plt.grid()
            plt.xlabel("$\lambda$ (nm)")
            plt.ylabel("transmission")
            plt.title("Disperser Transmissions")
#----------------------------------------------------------------------------------
            
#----------------------------------------------------------------------------------
class Atmosphere():
    """
    Atmospheres(): 
        classes to simulate series of atmospheres by calling libradtran
    
    """
    def __init__(self,airmass,pressure,temperature,filenamedata):
        """
        Args:
            filename (:obj:`str`): path to the image
            Image (:obj:`Image`): copy info from Image object
        """
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.airmass = airmass
        self.pressure = pressure
        self.temperature= temperature
        self.filename=""
        self.filenamedata=filenamedata   
        
        # create the numpy array that will contains the atmospheric grid    
        self.atmgrid=np.zeros((NB_ATM_POINTS+1,NB_atm_HEADER+NB_atm_DATA))
        self.atmgrid[0,index_atm_data:]=WL

        
    def simulate(self):
        # first determine the length
        if parameters.VERBOSE or parameters.DEBUG:
            self.my_logger.info('\n\tAtmosphere.simulate am=%4.2f, P=%4.2f, for data-file=%s ' % (self.airmass,self.pressure,self.filenamedata))
        count=0
        for  aer in AER_Points:
            for pwv in PWV_Points:
                for oz in OZ_Points:
                    count+=1
                    # fills headers info in the numpy array
                    self.atmgrid[count,index_atm_count]=count
                    self.atmgrid[count,index_atm_aer]=aer
                    self.atmgrid[count,index_atm_pwv]=pwv
                    self.atmgrid[count,index_atm_oz]=oz
                    
                    path,thefile=atmsim.ProcessSimulationaer(self.airmass,pwv,oz,aer,self.pressure)
                    fullfilename=os.path.join(path,thefile)
                    data=np.loadtxt(fullfilename)
                    wl=data[:,0]
                    atm=data[:,1]
                    func=interp1d(wl,atm,kind='linear')   # interpolation to conform to wavelength grid required
                    transm=func(WL)
                    
                    
                    self.atmgrid[count,index_atm_data:]=transm    # each of atmospheric transmission
                    
        return self.atmgrid
    
    def savefile(self,filename=""):
        
        if filename != "" :
            self.filename = filename
        
        if self.filename=="":
            return
        else:
        
            hdr = fits.Header()
        
       
            hdr['ATMSIM'] = "libradtran"
            hdr['SIMVERS'] = "2.0.1"
            hdr['DATAFILE']=self.filenamedata
            hdr['SIMUFILE']=os.path.basename(self.filename)
            
            hdr['AIRMASS'] = self.airmass
            hdr['PRESSURE'] = self.pressure
            hdr['TEMPERAT'] = self.temperature
            hdr['NBATMPTS'] = NB_ATM_POINTS
        
            hdr['NBAERPTS'] = NB_AER_POINTS
            hdr['AERMIN'] = AER_MIN
            hdr['AERMAX'] = AER_MAX

            hdr['NBPWVPTS'] = NB_PWV_POINTS
            hdr['PWVMIN'] = PWV_MIN
            hdr['PWVMAX'] = PWV_MAX
        
            hdr['NBOZPTS'] = NB_OZ_POINTS
            hdr['OZMIN'] = OZ_MIN
            hdr['OZMAX'] = OZ_MAX

            hdr['AER_PTS'] =np.array_str(AER_Points)
            hdr['PWV_PTS'] =np.array_str(PWV_Points)
            hdr['OZ_PTS'] =np.array_str(OZ_Points)
            hdr['NBWLBIN']=NBWLBINS
            hdr['WLMIN']=WLMIN
            hdr['WLMAX']=WLMAX
    
            hdr['IDX_CNT']=index_atm_count
            hdr['IDX_AER']=index_atm_aer
            hdr['IDX_PWV']=index_atm_pwv
            hdr['IDX_OZ']=index_atm_oz
            hdr['IDX_DATA']=index_atm_data
    
            if parameters.VERBOSE:
                print hdr
    
            hdu = fits.PrimaryHDU(self.atmgrid,header=hdr)
            hdu.writeto(self.filename,overwrite=True)
            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info('\n\tAtmosphere.save atm-file=%s' % (self.filename))
 #----------------------------------------------------------------------------------
  
#----------------------------------------------------------------------------------
class TelesTransm():
    """
    TelesTransm : Transmission of the telescope
    - mirrors
    - throughput
    - QE
    - Filter
    
    """
    
    def __init__(self,filtername=""):
        """
        Args:
        filename (:obj:`str`): path to the image
        Image (:obj:`Image`): copy info from Image object
        """
    
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.filtername = filtername
        self.qe= []
        self.to=[]
        self.tm=[]
        self.tf=[]
        self.tt=[]
        self.tfr=[]
        self.tfb=[]
        
    def load_transm(self):
        # QE
        wl,qe=ctio.Get_QE()
        # extend
        wl=np.concatenate([[WLMIN],wl,[WLMAX]])
        qe=np.concatenate([[0.],qe,[qe[-1]]])
        func=interp1d(wl,qe,kind='linear')   # interpolation to conform to wavelength grid required
        QE=func(WL)
        self.qe=QE
        
        #  Throughput
        wl,trt=ctio.Get_Throughput()
        wl=np.concatenate([[WLMIN],wl,[WLMAX]])
        trt=np.concatenate([[0.],trt,[trt[-1]]])
        func=interp1d(wl,trt,kind='linear')   # interpolation to conform to wavelength grid required
        TO=func(WL)
        self.to=TO
        
        # Mirrors 
        wl,trm=ctio.Get_Mirror()
        wl=np.concatenate([[WLMIN],wl,[WLMAX]])
        trm=np.concatenate([[0.],trm,[trm[-1]]])
        func=interp1d(wl,trm,kind='linear') 
        TM=func(WL)
        self.tm=TM
          
        
       
        wl,trg=ctio.Get_RG715()
        wl=np.concatenate([[WLMIN],wl,[WLMAX]])
        trg=np.concatenate([[0.],trg,[trg[-1]]])
        func=interp1d(wl,trg,kind='linear')
        TFR=func(WL)
        self.tfr=TFR
        
        wl,trb=ctio.Get_FGB37()
        wl=np.concatenate([[WLMIN],wl,[WLMAX]])
        trb=np.concatenate([[0.],trb,[0.]])
        func=interp1d(wl,trb,kind='linear')
        TFB=func(WL)
        self.tfb=TFB
            
            
        if self.filtername == "RG715" :
            TF=TFR
        elif self.filtername =="FGB37":
            
            TF=TFB
        else:
            TF=np.ones(len(WL))
            
        self.tf=TF
        
        self.tt=QE*TO*TM*TM*TF
                
        return self.tt
        
    def plot_transm(self,xlim=None):
        plt.figure()
        if(len(self.tt)!=0):
            plt.plot(WL,self.qe,'b-',label='qe')
            plt.plot(WL,self.to,'g-',label='othr')
            plt.plot(WL,self.tm,'y-',label='mirr')
            plt.plot(WL,self.tf,'k-',label='filt')
            plt.plot(WL,self.tfr,'k:',label='RG715')
            plt.plot(WL,self.tfb,'k--',label='FGB37')
            plt.plot(WL,self.tt,'r-',lw=2,label='tot')
            plt.legend()
            plt.grid()
            plt.xlabel("$\lambda$ (nm)")
            plt.ylabel("transmission")
            plt.title("Telescope Transmissions")
#----------------------------------------------------------------------------------        
               
        
        
#----------------------------------------------------------------------------------
class SpectrumSim():
    """ SpectrumSim class used to store information and methods
    relative to spectrum simulation.
    """
    def __init__(self,filename="",Image=None,atmospheric_lines=True,order=1):
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
        self.atmospheric_lines = atmospheric_lines
        self.lines = Lines(self.target.redshift,atmospheric_lines=self.atmospheric_lines,hydrogen_only=self.target.hydrogen_only,emission_spectrum=self.target.emission_spectrum)
        self.load_filter()

    def load_filter(self):
        for f in FILTERS:
            if f['label'] == self.filter:               
                parameters.LAMBDA_MIN = f['min']
                parameters.LAMBDA_MAX = f['max']
                self.my_logger.info('\n\tLoad filter %s: lambda between %.1f and %.1f' % (f['label'],parameters.LAMBDA_MIN, parameters.LAMBDA_MAX))
                break

    def plot_spectrum(self,xlim=None,nofit=False):
        xs = self.lambdas
        if xs is None : xs = np.arange(self.data.shape[0])
        fig = plt.figure(figsize=[12,6])
        if self.err is not None:
            plt.errorbar(xs,self.data,yerr=self.err,fmt='ro',lw=1,label='Order %d spectrum' % self.order,zorder=0)
        else:
            plt.plot(xs,self.data,'r-',lw=2,label='Order %d spectrum' % self.order)
        #if len(self.target.spectra)>0:
        #    for k in range(len(self.target.spectra)):
        #        s = self.target.spectra[k]/np.max(self.target.spectra[k])*np.max(self.data)
        #        plt.plot(self.target.wavelengths[k],s,lw=2,label='Tabulated spectra #%d' % k)
        plt.grid(True)
        plt.xlim([parameters.LAMBDA_MIN,parameters.LAMBDA_MAX])
        plt.ylim(0.,np.max(self.data)*1.2)
        plt.xlabel('$\lambda$ [nm]')
        plt.ylabel(self.units)
        if self.lambdas is None: plt.xlabel('Pixels')
        if xlim is not None :
            plt.xlim(xlim)
            plt.ylim(0.,np.max(self.data[xlim[0]:xlim[1]])*1.2)
        if self.lambdas is not None:
            self.lines.plot_atomic_lines(plt.gca(),fontsize=12)
        if not nofit and self.lambdas is not None:
            lambda_shift = self.lines.detect_lines(self.lambdas,self.data,spec_err=self.err,ax=plt.gca(),verbose=parameters.VERBOSE)
        plt.legend(loc='best',title=self.filters)
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

    def calibrate_spectrum_with_lines(self):
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
            lambda_shift = self.lines.detect_lines(lambdas_test,self.data,spec_err=self.err,ax=None,verbose=parameters.DEBUG)
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
        lambda_shift = self.lines.detect_lines(self.lambdas,self.data,spec_err=self.err,ax=None,verbose=parameters.DEBUG)
        self.my_logger.info('\n\tWavelenght total shift: %.2fnm (after %d steps)\n\twith D = %.2f mm (DISTANCE2CCD = %.2f +/- %.2f mm, %.1f sigma shift)' % (shift,len(shifts),D,DISTANCE2CCD,DISTANCE2CCD_ERR,(D-DISTANCE2CCD)/DISTANCE2CCD_ERR))
        if parameters.VERBOSE or parameters.DEBUG:
            self.plot_spectrum(xlim=None,nofit=False)

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
            if len(hdu[0].data)>2:
                self.err = hdu[0].data[2]
            extract_info_from_CTIO_header(self, self.header)
            if self.header['TARGET'] != "":
                self.target=Target(self.header['TARGET'],verbose=parameters.VERBOSE)
            if self.header['UNIT2'] != "":
                self.units = self.header['UNIT2']
            self.my_logger.info('\n\tSpectrum loaded from %s' % input_filename)
        else:
            self.my_logger.info('\n\tSpectrum file %s not found' % input_filename)
#----------------------------------------------------------------------------------        


#----------------------------------------------------------------------------------
def SpectractorSim(filename,outputdir,target,index,airmass,pressure,temperature,rhumidity,exposure,filtername,dispersername,atmospheric_lines=True):
    
    """ SpectractorSim
    Main function to simulate several spectra 
    A grid of spectra will be produced for a given target, airmass and pressure

    Args:
        filename (:obj:`str`): filename of the image (data)
        outputdir (:obj:`str`): path to the output directory
        
        then the following parameters are required to do the spectum simulation.
        
        target           : 
        index            :
        airmass          :
        pressure         : 
        temperature      :
        rhumidity        : 
        exposure         :
        filtername       :
        dispersername    :
    """
    my_logger = parameters.set_logger(__name__)
    my_logger.info('\n\tStart SPECTRACTORSIM')
    # Load reduced image
    #image = Image(filename,target=target)
 
    # Set output path
    ensure_dir(outputdir)
    # extract the basename : simimar as os.path.basename(file)
    base_filename = filename.split('/')[-1]  # get "reduc_20170530_213.fits"
    tag_filename=base_filename.split('_')[0] # get "reduc_"
    search_str ='^%s_(.*)' % (tag_filename)  # get "^reduc_(.*)"
    root_filename=re.findall(search_str,base_filename)[0]   # get "20170530_213.fits'
    
    #output_filename = root_filename.replace('.fits','_spectrumsim.fits')
    #output_atmfilename = root_filename.replace('_spectrumsim.fits','_atmsim.fits')
    output_filename='spectrasim_'+root_filename # get "spectrasim_20170530_213.fits"
    output_atmfilename='atmsim_'+root_filename  # get "atmsim__20170530_213.fits"
    
    output_filename = os.path.join(outputdir,output_filename)
    output_atmfilename = os.path.join(outputdir,output_atmfilename)
    # Find the exact target position in the raw cut image: several methods
    my_logger.info('\n\tWill simulate the spectrum...')
    if parameters.DEBUG:
            infostring='\n\tWill debug simulated the spectrum into file %s ...'%(output_filename)
            my_logger.info(infostring)
 
    
    # SIMULATE ATMOSPHERE GRID
    # ------------------------
    if parameters.VERBOSE:
        infostring='\n\t ========= Atmospheric simulation :  ==============='
    atm=Atmosphere(airmass,pressure,temperature,filename)
    arr=atm.simulate()
    atm.savefile(filename=output_atmfilename)
    atmsim.CleanSimDir()
    
    # TELESCOPE TRANSMISSION
    # ------------------------
    tel=TelesTransm(filtername)
    tr=tel.load_transm()
    
    if parameters.VERBOSE:
        infostring='\n\t ========= Telescope transmission :  ==============='
        my_logger.info(infostring)
        tel.plot_transm()
        
    # DISPERSER TRANSMISSION
    # ------------------------
    disp=Disperser(dispersername)
    td=disp.load_transm()
    if parameters.VERBOSE:
        infostring='\n\t ========= Disperser transmission :  ==============='
        my_logger.info(infostring)
        disp.plot_transm()
    
    # STAR SPECTRUM
    # ------------------------
    sed=SED(target)
    flux=sed.get_sed()
    
    if parameters.VERBOSE:
        infostring='\n\t ========= SED : %s  ===============' % target
        my_logger.info(infostring)
        sed.plot_sed()
    
    # SPECTRA-GRID  -> TBD
    #-----------------------
    spec=flux*td*tr*arr[-1,index_atm_data:]*WL
    
    plt.figure()
    plt.plot(WL,spec,'b-',label='spectrum')
    plt.xlabel('$\lambda$ [nm]')
    plt.ylabel('Flux')
    plt.title('SPECTRUM')
    plt.legend()
    plt.grid()
    plt.show()
    
    #spectrum = image.extract_spectrum_from_image()
    #spectrum.atmospheric_lines = atmospheric_lines
    # Calibrate the spectrum
    #spectrum.calibrate_spectrum()
    #spectrum.calibrate_spectrum_with_lines()
    # Subtract second order

    # Save the spectra
    #spectrum.save_spectrum(output_filename,overwrite=True)
    
    
    
    
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


if __name__ == "__main__":
    import commands, string,  time
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
        
        
    #filename = "../../CTIODataJune2017_reducedRed/data_05jun17/reduc_20170605_00.fits"
    #filename = "notebooks/fits/trim_20170605_007.fits"
    #guess = [745,643]
    #target = "3C273"

    opts.debug=True
    parameters.DEBUG = True
    parameters.VERBOSE = True

    filename="reduc_20170530_213.fits"
    airmass=1.094
    pressure=782
    temperature=9.5
    rhumidity=24.0
    target = "HD205905"
    index=213
    exposure=60.
    filtername='dia'
    dispersername='HoloPhP'
    
    
    SpectractorSim(filename,opts.output_directory,target,index,airmass,pressure,temperature,rhumidity,exposure,filtername,dispersername,atmospheric_lines=True)
#----------------------------------------------------------------------------------    

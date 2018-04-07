"""
spectractorsim
=============----

author : Sylvie Dagoret-Campagne
affiliation : LAL/CNRS/IN2P3/FRANCE
Collaboration : DESC-LSST

Purpose : Simulate a series of spectra for each experimental spectra measured by auxiliary telescope.
Structure in parallel to Spectractor.
For each experimental spectra a fits file image is generated which holds all possible auxiliary telescope spectra
corresponding to different conditions in aerosols, pwv, and ozone. 

creation date : April 18th 
Last updaten : April 6th

"""

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
from astropy import constants as const

from scipy.interpolate import interp1d

sys.path.append("../Spectractor")

from tools import *
#from holo_specs import *
from targets import *
from optics import *
import parameters 
#----------------------------------------------------------------------------
# where is spectractorsim
#----------------------------------------------------------------------------
spectractorsim_path = os.path.dirname(__file__)


#---------------------------------------------------------------------------
# Libraries to interface LibRadTran and CTIO 0.9m telescope transparencies
#-------------------------------------------------------------------------

import libsimulateTranspCTIOScattAbsAer as atmsim
import libCTIOTransm as ctio
#--------------------------------------------------------------------------
# Telescope parameter
#
#   The goal is to calculate the numerical factor to get spectra into ADU
#
#  the SED is supposed to be in flam units ie erg/s/cm^-2 per angtrom
#   however the binning is 10 A or 1 nm, the the SED has been multiplied by 10
#--------------------------------------------------------------------------
Tel_Diam=0.9*units.m                     # Diameter of the telescope
Tel_Surf=np.pi*Tel_Diam**2/4.            # collection surface of telescope
Time_unit=1*units.s                      # flux for 1 second
SED_unit=1*units.erg/units.s/(units.cm)**2/(units.nanometer)          # Units of SEDs in flam (erg/s/cm2/nm)
hc=const.h*const.c                        # h.c product of fontamental constants c and h 
wl_dwl_unit=(units.nanometer)**2          # lambda.dlambda  in wavelength in nm
g_elec=3.0                                # electronic gain : elec/ADU
g_disperser_ronchi=0.2                   # theoretical gain for order+1 : 20%
#Factor=2.1350444e11
Factor=(Tel_Surf*SED_unit*Time_unit*wl_dwl_unit/hc/g_elec*g_disperser_ronchi).decompose()



#------------------------------------------------------------------------
# Definition of data format for the atmospheric grid
#-----------------------------------------------------------------------------
WLMIN=300. # Minimum wavelength : PySynPhot works with Angstrom
WLMAX=1100. # Minimum wavelength : PySynPhot works with Angstrom

NBWLBINS=800 # Number of bins between WLMIN and WLMAX
BinWidth=(WLMAX-WLMIN)/float(NBWLBINS) # Bin width in Angstrom
WL=np.linspace(WLMIN,WLMAX,NBWLBINS) # Array of wavelength in Angstrom


# specify parameters for the atmospheric grid

#aerosols
#NB_AER_POINTS=20
NB_AER_POINTS=5
AER_MIN=0.
AER_MAX=0.1

#ozone
#NB_OZ_POINTS=5
NB_OZ_POINTS=3
OZ_MIN=200
OZ_MAX=400

# pwv
#NB_PWV_POINTS=11
NB_PWV_POINTS=5
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
#----------------------------------------------------------------------------------------------
# Specific class tools
#---------------------------------------------------------------------------------------
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
            plt.plot(WL,self.td,'b-',label=self.disperser)
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
    
    #---------------------------------------------------------------------------
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
        self.header=fits.Header()
    #---------------------------------------------------------------------------        
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
    #---------------------------------------------------------------------------  
    def plot_transm(self):
        plt.figure()
        counts=self.atmgrid[1:,index_atm_count]
        
        for count in counts:
            
            plt.plot(WL,self.atmgrid[int(count),index_atm_data:])
        plt.grid()
        plt.xlabel("$\lambda$ (nm)")
        plt.ylabel("atmospheric transparency")
        plt.title("Atmospheric variations")
        plt.show()
    #---------------------------------------------------------------------------   
    def plot_transm_img(self):
        plt.figure()
        img=plt.imshow(self.atmgrid[1:,index_atm_data:],origin='lower',cmap='jet')
        plt.grid(True)
        plt.xlabel("Wavelength bins [nm]")
        plt.ylabel("simulation number")
        plt.title(" Atmospheric variations",size=15)
        cbar=plt.colorbar(img)
        cbar.set_label('atmospheric transparency',size=15)
        plt.show()
    #---------------------------------------------------------------------------    
    def savefile(self,filename=""):
             
        hdr = fits.Header()
               
        if filename != "" :
            self.filename = filename
        
        if self.filename=="":
            infostring='\n\t Atmosphere:savefile no input file given ...'
            self.my_logger.info(infostring)
            return
        else:
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
                
            return hdr
    #---------------------------------------------------------------------------   
    def loadfile(self,filename):
             
        if filename != "" :
            self.filename = filename
               
        
        if self.filename=="":
            infostring='\n\t Atmosphere:loadfile no input file given ...'
            self.my_logger.info(infostring)
 
            return
        else:
        
            hdu= fits.open(self.filename)
            hdr=hdu[0].header
       
            #hdr['ATMSIM'] = "libradtran"
            #hdr['SIMVERS'] = "2.0.1"
            self.filenamedata=hdr['DATAFILE']
            #hdr['SIMUFILE']=os.path.basename(self.filename)
            
            self.airmass=hdr['AIRMASS'] 
            self.pressure=hdr['PRESSURE']
            self.temperature=hdr['TEMPERAT']
            
            # hope those are the same parameters : TBD !!!!
            NB_ATM_POINTS=hdr['NBATMPTS']
        
            NB_AER_POINTS=hdr['NBAERPTS'] 
            AER_MIN=hdr['AERMIN']  
            AER_MAX=hdr['AERMAX'] 

            NB_PWV_POINTS=hdr['NBPWVPTS'] 
            PWV_MIN=hdr['PWVMIN'] 
            PWV_MAX=hdr['PWVMAX'] 
        
            NB_OZ_POINTS=hdr['NBOZPTS'] 
            OZ_MIN=hdr['OZMIN'] 
            OZ_MAX=hdr['OZMAX'] 
            
            AER_Points=np.linspace(AER_MIN,AER_MAX,NB_AER_POINTS)
            OZ_Points=np.linspace(OZ_MIN,OZ_MAX,NB_OZ_POINTS)
            PWV_Points=np.linspace(PWV_MIN,PWV_MAX,NB_PWV_POINTS)

            #hdr['AER_PTS'] =np.array_str(AER_Points)
            #hdr['PWV_PTS'] =np.array_str(PWV_Points)
            #hdr['OZ_PTS'] =np.array_str(OZ_Points)
            
            NBWLBINS= hdr['NBWLBIN']
            WLMIN= hdr['WLMIN']
            WLMAX= hdr['WLMAX']
    
            index_atm_count=hdr['IDX_CNT']
            index_atm_aer=hdr['IDX_AER']
            index_atm_pwv=hdr['IDX_PWV']
            index_atm_oz=hdr['IDX_OZ']
            index_atm_data=hdr['IDX_DATA']
    
            self.atmgrid=np.zeros((NB_ATM_POINTS+1,NB_atm_HEADER+NB_atm_DATA))
    
            self.atmgrid[:,:]=hdu[0].data[:,:]
           
            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info('\n\tAtmosphere.load atm-file=%s' % (self.filename))
                
            return self.atmgrid,self.header
        #---------------------------------------------------------------------------
        
 
  
#----------------------------------------------------------------------------------
class TelesTransm():
    """
    TelesTransm : Transmission of the telescope
    - mirrors
    - throughput
    - QE
    - Filter
    
    """
    #---------------------------------------------------------------------------
    def __init__(self,filtername=""):
        """
        Args:
        filename (:obj:`str`): path to the data filename (for info only)
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
    #---------------------------------------------------------------------------    
    def load_transm(self):
        """
        load_transm(self) :
            load the telescope transmission
            return the total telescope transmission, disperser excluded
        """
        
        # defines the datapath relative to the Spectractor sim path
        datapath=os.path.join(spectractorsim_path,"CTIOThroughput")
        
        # QE
        wl,qe=ctio.Get_QE(datapath)
        # extend
        wl=np.concatenate([[WLMIN],wl,[WLMAX]])
        qe=np.concatenate([[0.],qe,[qe[-1]]])
        func=interp1d(wl,qe,kind='linear')   # interpolation to conform to wavelength grid required
        QE=func(WL)
        self.qe=QE
        
        #  Throughput
        wl,trt=ctio.Get_Throughput(datapath)
        wl=np.concatenate([[WLMIN],wl,[WLMAX]])
        trt=np.concatenate([[0.],trt,[trt[-1]]])
        func=interp1d(wl,trt,kind='linear')   # interpolation to conform to wavelength grid required
        TO=func(WL)
        self.to=TO
        
        # Mirrors 
        wl,trm=ctio.Get_Mirror(datapath)
        wl=np.concatenate([[WLMIN],wl,[WLMAX]])
        trm=np.concatenate([[0.],trm,[trm[-1]]])
        func=interp1d(wl,trm,kind='linear') 
        TM=func(WL)
        self.tm=TM
          
        # Filter RG715
        wl,trg=ctio.Get_RG715(datapath)
        wl=np.concatenate([[WLMIN],wl,[WLMAX]])
        trg=np.concatenate([[0.],trg,[trg[-1]]])
        func=interp1d(wl,trg,kind='linear')
        TFR=func(WL)
        self.tfr=TFR
        
        # Filter FGB37
        wl,trb=ctio.Get_FGB37(datapath)
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
    #---------------------------------------------------------------------------    
    def plot_transm(self,xlim=None):
        """
        plot_transm()
            plot the various transmission
        """
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
    #---------------------------------------------------------------------------
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

        
        self.header = None
        self.date_obs = None
        self.airmass = None
        self.expo = None
        self.filters = None
        self.filter = None
        self.disperser = None
        self.target = None
        
        self.atmgrid = None
        self.spectragrid= None

        self.filename=""

        self.atmospheric_lines = atmospheric_lines
        #self.lines = Lines(self.target.redshift,atmospheric_lines=self.atmospheric_lines,hydrogen_only=self.target.hydrogen_only,emission_spectrum=self.target.emission_spectrum)
    
        if filename != "" :
            self.filename = filename
            self.load_spectrum(filename)
    #----------------------------------------------------------------------------    
    def compute(self,atmgrid,filtertransm , dispersertransm,sed, header):
        self.header=header
        
        if parameters.VERBOSE :
            print self.header
            
            
        self.atmgrid=atmgrid
         
        self.spectragrid=np.zeros(self.atmgrid.shape)
         
        # product of all sed and transmission except atmosphere
        all_transm=filtertransm*dispersertransm*sed*WL*BinWidth
         
        # copy atmospheric grid parameters into spectra grid 
        self.spectragrid[0,index_atm_data:]=WL
        self.spectragrid[:,index_atm_count:index_atm_data]=self.atmgrid[:,index_atm_count:index_atm_data] 
        # Is boradcasting working OK ?
        self.spectragrid[1:,index_atm_data:]=self.atmgrid[1:,index_atm_data:]* all_transm *Factor
         
        return self.spectragrid
    #---------------------------------------------------------------------------
    def plot_spectra(self):
        plt.figure()
        counts=self.spectragrid[1:,index_atm_count]
        for count in counts:
            plt.plot(WL,self.spectragrid[int(count),index_atm_data:])
        plt.grid()
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Flux  [ADU/s]")
        plt.title("Spectra for Atmospheric variations")
        plt.show()
    #---------------------------------------------------------------------------   
    def plot_spectra_img(self):
        plt.figure()
        img=plt.imshow(self.spectragrid[1:,index_atm_data:],origin='lower',cmap='jet')
        plt.xlabel("Wavelength bins [nm]")
        plt.ylabel("simulation number")
        plt.title("Spectra for Atmospheric variations")
        cbar=plt.colorbar(img)
        cbar.set_label('ADU',size=15)
        plt.grid(True)
        plt.show()
    #---------------------------------------------------------------------------  
    def save_spectra(self,filename):
                   
        if filename != "" :
            self.filename = filename
        
        if self.filename=="":
            return
        else:
         
            hdu = fits.PrimaryHDU(self.spectragrid,header=self.header)
            hdu.writeto(self.filename,overwrite=True)
            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info('\n\tSPECTRA.save atm-file=%s' % (self.filename))
    #---------------------------------------------------------------------------            
                
   
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

    atm=Atmosphere(airmass,pressure,temperature,filename)
    
    # test if file already exists
    if os.path.exists(output_atmfilename):
        infostring=" atmospheric simulation file %s already exist, thus load it ..." % (output_atmfilename)
        my_logger.info(infostring)
        atmgrid,header=atm.loadfile(output_atmfilename)
    else:
        atmgrid=atm.simulate()
        header=atm.savefile(filename=output_atmfilename)
        atmsim.CleanSimDir()
    
    if parameters.VERBOSE:
        infostring='\n\t ========= Atmospheric simulation :  ==============='
        my_logger.info(infostring)
        atm.plot_transm()
        atm.plot_transm_img()
    
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
    
    # SPECTRA-GRID  
    #-------------   
   
    spectra=SpectrumSim()
    
    spectragrid=spectra.compute(atmgrid,tr,td,flux,header)
    spectra.save_spectra(output_filename)
    
    
    if parameters.VERBOSE:
        infostring='\n\t ========= Spectra simulation :  ==============='
        spectra.plot_spectra()
        spectra.plot_spectra_img()
    #--------------------------------------------------------------------------- 
    
       
    
    
#----------------------------------------------------------------------------------
#  START SPECTRACTORSIM HERE !
#----------------------------------------------------------------------------------


if __name__ == "__main__":
    #import commands, string,  time
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

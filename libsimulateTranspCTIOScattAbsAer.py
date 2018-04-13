################################################################
#
# Script to simulate air transparency with LibRadTran
# With a pure absorbing atmosphere
# Here we vary PWV
# author: sylvielsstfr
# creation date : November 2nd 2016
# update : April 2018
#
#################################################################
import os
import re
import math
import numpy as np
import pandas as pd
from astropy.io import fits
import sys,getopt

import UVspec

FLAG_DEBUG=False

# Definitions and configuration
#-------------------------------------

# LibRadTran installation directory
home = os.environ['HOME']+'/'       
libradtranpath = os.getenv('LIBRADTRANDIR')+'/'

# Filename : RT_LS_pp_us_sa_rt_z15_wv030_oz30.txt
#          : Prog_Obs_Rte_Atm_proc_Mod_zXX_wv_XX_oz_XX
  
Prog='RT'  #definition the simulation programm is libRadTran
Obs='CT'   # definition of observatory site (LS,CT,OH,MK,...)
Rte='pp'   # pp for parallel plane of ps for pseudo-spherical
Atm=['us']   # short name of atmospheric sky here US standard and  Subarctic winter
Proc='sa'  # light interaction processes : sc for pure scattering,ab for pure absorption
           # sa for scattering and absorption, ae with aerosols default, as with aerosol special
Mod='rt'   # Models for absorption bands : rt for REPTRAN, lt for LOWTRAN, k2 for Kato2
ZXX='z'        # XX index for airmass z :   XX=int(10*z)
WVXX='wv'      # XX index for PWV       :   XX=int(pwv*10)
OZXX='oz'      # XX index for OZ        :   XX=int(oz/10)
AEXX='aer'
AEXX2='aer2'


#LSST_Altitude = 2.750  # in k meters from astropy package (Cerro Pachon)
#OBS_Altitude = str(LSST_Altitude)


CTIO_Altitude = 2.200  # in k meters from astropy package (Cerro Pachon)
OBS_Altitude = str(CTIO_Altitude)


TOPDIR='simulations/RT/2.0.1/CT'

def CleanSimDir():   
    os.system("rm -rf simulations")


############################################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
#########################################################################


def usage0():
    print "*******************************************************************"
    print sys.argv[0],' -z <airmass> -w <pwv> -o <oz>'
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    
    
    print "*******************************************************************"
    
def usageaer():
    print "*******************************************************************"
    print sys.argv[0],' -z <airmass> -w <pwv> -o <oz> -l <wl> -t <tau>'
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    
    
    print "*******************************************************************"    


def usage():
    print "*******************************************************************"
    print sys.argv[0],' -z <airmass> -w <pwv> -o <oz> -a<aer> -p <P>'
    print ' \t - airmass from 1.0 to 3.0, typical z=1 '
    print ' \t - pwv is precipitable watr vapor in kg per m2 or mm, typical pwv = 5.18 mm'
    print ' \t - oz ozone in Dobson units from 200 DU to 400 DU'
    print ' \t - aer means Aerosols, typical a=0.04  '
    print ' \t - p Pressure in hPa, typical P=775.3 hPa  '
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)    
    print "*******************************************************************"
    


#----------------------------------------------------------------------------
def ApplyAerosols(wl,tr,thelambda0,tau0,alpha0):
    """
     ApplyAerosols(wl,tr,thelambda0,tau0,alpha0)
     
     - input :
     -- wl np array of wavelength in nm
     -- tr transparency array without aerosols
     -- thelambda0 : the reference point where to have tau0 in nm
     -- tau0 is the extinction at thelambda0
     -- alpha0 the Angstrom exponent
     
    """
    #extinc_aer=tau0*(thelambda0/wl)**alpha0
    extinc_aer=tau0*np.power(wl/thelambda0,-alpha0)
    tr_aer=np.exp(-extinc_aer)
    tr_tot=tr*tr_aer
    return tr_tot
    
    
#-----------------------------------------------------------------------------


def ProcessSimulation(airmass_num,pwv_num,oz_num,press_num):
    """
    ProcessSimulation(airmass_num,pwv_num,oz_num) 
    No aerosol simulation is performed
    """
    
 
    print '--------------------------------------------'
    print ' 1) airmass = ', airmass_num
    print ' 2) pwv = ', pwv_num
    print ' 3) oz = ', oz_num
    print ' 4) pressure  = ',press_num
    print '--------------------------------------------'    
   
    
    ensure_dir(TOPDIR)

    
    Proc='sa'  # Pure absorption and Rayleigh scattering : Clear sky without aerosols
    
    # build the part 1 of filename
    BaseFilename_part1=Prog+'_'+Obs+'_'+Rte+'_'
    

    # Set up type of run
    runtype='clearsky' #'no_scattering' #aerosol_special #aerosol_default# #'clearsky'#     
    if Proc == 'sc':
        runtype='no_absorption'
        outtext='no_absorption'
    elif Proc == 'ab':
        runtype='no_scattering'
        outtext='no_scattering'
    elif Proc == 'sa':
        runtype=='clearsky'
        outtext='clearsky'
    elif Proc == 'ae':   
        runtype='aerosol_default'
        outtext='aerosol_default'
    elif Proc == 'as':   
        runtype='aerosol_special'
        outtext='aerosol_special'
    else:
        runtype=='clearsky'
        outtext='clearsky'

#   Selection of RTE equation solver        
    if Rte == 'pp': # parallel plan
        rte_eq='disort'
    elif Rte=='ps':   # pseudo spherical
        rte_eq='sdisort'
        
 
#   Selection of absorption model 
    molmodel='reptran'
    if Mod == 'rt':
        molmodel='reptran'
    if Mod == 'lt':
        molmodel='lowtran'
    if Mod == 'kt':
        molmodel='kato'
    if Mod == 'k2':
        molmodel='kato2'
    if Mod == 'fu':
        molmodel='fu'    
    if Mod == 'cr':
        molmodel='crs'     
               


    	  
    # for simulation select only two atmosphere   
    #theatmospheres = np.array(['afglus','afglms','afglmw','afglt','afglss','afglsw'])
    atmosphere_map=dict()  # map atmospheric names to short names 
    atmosphere_map['afglus']='us'
    atmosphere_map['afglms']='ms'
    atmosphere_map['afglmw']='mw'  
    atmosphere_map['afglt']='tp'  
    atmosphere_map['afglss']='ss'  
    atmosphere_map['afglsw']='sw'  
      
    theatmospheres= []
    for skyindex in Atm:
        if re.search('us',skyindex):
            theatmospheres.append('afglus')
        if re.search('sw',skyindex):
            theatmospheres.append('afglsw')
            
   
   

    # 1) LOOP ON ATMOSPHERE
    for atmosphere in theatmospheres:
        #if atmosphere != 'afglus':  # just take us standard sky
        #    break
        atmkey=atmosphere_map[atmosphere]
       
        # manage input and output directories and vary the ozone
        TOPDIR2=TOPDIR+'/'+Rte+'/'+atmkey+'/'+Proc+'/'+Mod
        ensure_dir(TOPDIR2)
        INPUTDIR=TOPDIR2+'/'+'in'
        ensure_dir(INPUTDIR)
        OUTPUTDIR=TOPDIR2+'/'+'out'
        ensure_dir(OUTPUTDIR)
    
    
        # loop on molecular model resolution
        #molecularresolution = np.array(['COARSE','MEDIUM','FINE']) 
        # select only COARSE Model
        molecularresolution = np.array(['COARSE'])    
        for molres in molecularresolution:
            if molres=='COARSE':
                molresol ='coarse'
            elif molres=='MEDIUM':
                molresol ='medium'
            else:
                molresol ='fine'
           
        
        #water vapor   
        pwv_val=pwv_num
        pwv_str='H2O '+str(pwv_val)+ ' MM'
        wvfileindex=int(10*pwv_val)
           
           
        # airmass
        airmass=airmass_num
        amfileindex=int(airmass_num*10)
        
        # Ozone    
        oz_str='O3 '+str(oz_num)+ ' DU'
        ozfileindex=int(oz_num/10.)
        
            
        BaseFilename=BaseFilename_part1+atmkey+'_'+Proc+'_'+Mod+'_z'+str(amfileindex)+'_'+WVXX+str(wvfileindex) +'_'+OZXX+str(ozfileindex)                   
                    
        verbose=True
        uvspec = UVspec.UVspec()
        uvspec.inp["data_files_path"]  =  libradtranpath+'data'
                
        uvspec.inp["atmosphere_file"] = libradtranpath+'data/atmmod/'+atmosphere+'.dat'
        uvspec.inp["albedo"]           = '0.2'
    
        uvspec.inp["rte_solver"] = rte_eq
            
            
                
        if Mod == 'rt':
            uvspec.inp["mol_abs_param"] = molmodel + ' ' + molresol
        else:
            uvspec.inp["mol_abs_param"] = molmodel

        # Convert airmass into zenith angle 
        am=airmass
        sza=math.acos(1./am)*180./math.pi

        # Should be no_absorption
        if runtype=='aerosol_default':
            uvspec.inp["aerosol_default"] = ''
        elif runtype=='aerosol_special':
            uvspec.inp["aerosol_default"] = ''
            uvspec.inp["aerosol_set_tau_at_wvl"] = '500 0.02'
                        
        if runtype=='no_scattering':
            uvspec.inp["no_scattering"] = ''
        if runtype=='no_absorption':
            uvspec.inp["no_absorption"] = ''
     
        # set up the ozone value               
        uvspec.inp["mol_modify"] = pwv_str
        uvspec.inp["mol_modify2"] = oz_str
        
        # rescale pressure   if reasonable pressure values are provided
        if press_num>600. and press_num<1015.:
            uvspec.inp["pressure"] = press_num
                    
                
        uvspec.inp["output_user"] = 'lambda edir'
        uvspec.inp["altitude"] = OBS_Altitude   # Altitude LSST observatory
        uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_1.0nm.dat'
        #uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_0.1nm.dat'
        uvspec.inp["sza"]        = str(sza)
        uvspec.inp["phi0"]       = '0'
        uvspec.inp["wavelength"]       = '250.0 1200.0'
        uvspec.inp["output_quantity"] = 'reflectivity' #'transmittance' #
#       uvspec.inp["verbose"] = ''
        uvspec.inp["quiet"] = ''

  

        if "output_quantity" in uvspec.inp.keys():
            outtextfinal=outtext+'_'+uvspec.inp["output_quantity"]

           
            
        inputFilename=BaseFilename+'.INP'
        outputFilename=BaseFilename+'.OUT'
        inp=os.path.join(INPUTDIR,inputFilename)
        out=os.path.join(OUTPUTDIR,outputFilename)
                    
            
        uvspec.write_input(inp)
        uvspec.run(inp,out,verbose,path=libradtranpath)
        
        
    return OUTPUTDIR,outputFilename

#---------------------------------------------------------------------------


#------------------------------------------------------------------------------
def ProcessSimulationaer(airmass_num,pwv_num,oz_num,aer_num,press_num):  
    """
    ProcessSimulationaer(airmass_num,pwv_num,oz_num,aer_num,press_num) 
    with aerosol simulation is performed
    default profile
    """
 
    if FLAG_DEBUG:
        print '--------------------------------------------'
        print 'ProcessSimulationaer'
        print ' 1) airmass = ', airmass_num
        print ' 2) pwv = ', pwv_num
        print ' 3) oz = ', oz_num
        print ' 4) aer = ',aer_num
        print ' 5) pressure =',press_num
        print '--------------------------------------------'    
   
    
    ensure_dir(TOPDIR)

    
    # build the part 1 of filename
    BaseFilename_part1=Prog+'_'+Obs+'_'+Rte+'_'
    
    aerosol_string = '500 '+str(aer_num)
    #aerosol_str=str(wl0_num)+ ' '+str(tau0_num)
    aer_index=int(aer_num*100.)

    # Set up type of run
    runtype='aerosol_special' #'no_scattering' #aerosol_special #aerosol_default# #'clearsky'#     
    
    Proc='as'  # Absoprtion + Rayleigh + aerosols special
    
    
    if Proc == 'sc':
        runtype='no_absorption'
        outtext='no_absorption'
    elif Proc == 'ab':
        runtype='no_scattering'
        outtext='no_scattering'
    elif Proc == 'sa':
        runtype=='clearsky'
        outtext='clearsky'
    elif Proc == 'ae':   
        runtype='aerosol_default'
        outtext='aerosol_default'
    elif Proc == 'as':   
        runtype='aerosol_special'
        outtext='aerosol_special'
    else:
        runtype=='clearsky'
        outtext='clearsky'

#   Selection of RTE equation solver        
    if Rte == 'pp': # parallel plan
        rte_eq='disort'
    elif Rte=='ps':   # pseudo spherical
        rte_eq='sdisort'
        
 
#   Selection of absorption model 
    molmodel='reptran'
    if Mod == 'rt':
        molmodel='reptran'
    if Mod == 'lt':
        molmodel='lowtran'
    if Mod == 'kt':
        molmodel='kato'
    if Mod == 'k2':
        molmodel='kato2'
    if Mod == 'fu':
        molmodel='fu'    
    if Mod == 'cr':
        molmodel='crs'     
               


    	  
    # for simulation select only two atmosphere   
    #theatmospheres = np.array(['afglus','afglms','afglmw','afglt','afglss','afglsw'])
    atmosphere_map=dict()  # map atmospheric names to short names 
    atmosphere_map['afglus']='us'
    atmosphere_map['afglms']='ms'
    atmosphere_map['afglmw']='mw'  
    atmosphere_map['afglt']='tp'  
    atmosphere_map['afglss']='ss'  
    atmosphere_map['afglsw']='sw'  
      
    theatmospheres= []
    for skyindex in Atm:
        if re.search('us',skyindex):
            theatmospheres.append('afglus')
        if re.search('sw',skyindex):
            theatmospheres.append('afglsw')
            
   
   

    # 1) LOOP ON ATMOSPHERE
    for atmosphere in theatmospheres:
        #if atmosphere != 'afglus':  # just take us standard sky
        #    break
        atmkey=atmosphere_map[atmosphere]
       
        # manage input and output directories and vary the ozone
        TOPDIR2=TOPDIR+'/'+Rte+'/'+atmkey+'/'+Proc+'/'+Mod
        ensure_dir(TOPDIR2)
        INPUTDIR=TOPDIR2+'/'+'in'
        ensure_dir(INPUTDIR)
        OUTPUTDIR=TOPDIR2+'/'+'out'
        ensure_dir(OUTPUTDIR)
    
    
        # loop on molecular model resolution
        #molecularresolution = np.array(['COARSE','MEDIUM','FINE']) 
        # select only COARSE Model
        molecularresolution = np.array(['COARSE'])    
        for molres in molecularresolution:
            if molres=='COARSE':
                molresol ='coarse'
            elif molres=='MEDIUM':
                molresol ='medium'
            else:
                molresol ='fine'
           
        
        #water vapor   
        pwv_val=pwv_num
        pwv_str='H2O '+str(pwv_val)+ ' MM'
        wvfileindex=int(10*pwv_val)
        
        #aerosols
       
           
        # airmass
        airmass=airmass_num
        amfileindex=int(airmass_num*10)
        
        # Ozone    
        oz_str='O3 '+str(oz_num)+ ' DU'
        ozfileindex=int(oz_num/10.)
        
            
        BaseFilename=BaseFilename_part1+atmkey+'_'+Proc+'_'+Mod+'_z'+str(amfileindex)+'_'+WVXX+str(wvfileindex) +'_'+OZXX+str(ozfileindex)+'_'+AEXX+str(aer_index)                   
                    
        verbose=FLAG_DEBUG
        
        uvspec = UVspec.UVspec()
        uvspec.inp["data_files_path"]  =  libradtranpath+'data'
                
        uvspec.inp["atmosphere_file"] = libradtranpath+'data/atmmod/'+atmosphere+'.dat'
        uvspec.inp["albedo"]           = '0.2'
    
        uvspec.inp["rte_solver"] = rte_eq
            
            
                
        if Mod == 'rt':
            uvspec.inp["mol_abs_param"] = molmodel + ' ' + molresol
        else:
            uvspec.inp["mol_abs_param"] = molmodel

        # Convert airmass into zenith angle 
        am=airmass
        sza=math.acos(1./am)*180./math.pi

        # Should be no_absorption
        if runtype=='aerosol_default':
            uvspec.inp["aerosol_default"] = ''
        elif runtype=='aerosol_special':
            uvspec.inp["aerosol_default"] = ''
            uvspec.inp["aerosol_set_tau_at_wvl"] = aerosol_string
                   
        if runtype=='no_scattering':
            uvspec.inp["no_scattering"] = ''
        if runtype=='no_absorption':
            uvspec.inp["no_absorption"] = ''
     
        # set up the ozone value               
        uvspec.inp["mol_modify"] = pwv_str
        uvspec.inp["mol_modify2"] = oz_str
        
        # rescale pressure   if reasonable pressure values are provided
        if press_num>600. and press_num<1015.:
            uvspec.inp["pressure"] = press_num
        else:
            print "creazy pressure p=",press_num, ' hPa'
                    
                
        uvspec.inp["output_user"] = 'lambda edir'
        uvspec.inp["altitude"] = OBS_Altitude   # Altitude LSST observatory
        uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_1.0nm.dat'
        #uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_0.1nm.dat'
        uvspec.inp["sza"]        = str(sza)
        uvspec.inp["phi0"]       = '0'
        uvspec.inp["wavelength"]       = '250.0 1200.0'
        uvspec.inp["output_quantity"] = 'reflectivity' #'transmittance' #
#       uvspec.inp["verbose"] = ''
        uvspec.inp["quiet"] = ''

  

        if "output_quantity" in uvspec.inp.keys():
            outtextfinal=outtext+'_'+uvspec.inp["output_quantity"]

           
            
        inputFilename=BaseFilename+'.INP'
        outputFilename=BaseFilename+'.OUT'
        inp=os.path.join(INPUTDIR,inputFilename)
        out=os.path.join(OUTPUTDIR,outputFilename)
                    
            
        uvspec.write_input(inp)
        uvspec.run(inp,out,verbose,path=libradtranpath)
        
        
    return OUTPUTDIR,outputFilename

#---------------------------------------------------------------------------


#------------------------------------------------------------------------------
def ProcessSimulationaer1(airmass_num,pwv_num,oz_num,wl0_num,tau0_num,press_num):  
    """
    ProcessSimulationaer(airmass_num,pwv_num,oz_num) 
    with aerosol simulation is performed
    default profile
    """
 
    print '--------------------------------------------'
    print ' 1) airmass = ', airmass_num
    print ' 2) pwv = ', pwv_num
    print ' 3) oz = ', oz_num
    print ' 4) wl0 = ',wl0_num
    print ' 5) tau0 = ',tau0_num
    print ' 6) pressure =',press_num
    print '--------------------------------------------'    
   
    
    ensure_dir(TOPDIR)

    
    # build the part 1 of filename
    BaseFilename_part1=Prog+'_'+Obs+'_'+Rte+'_'
    

    # Set up type of run
    runtype='aerosol_special' #'no_scattering' #aerosol_special #aerosol_default# #'clearsky'#     
    
    Proc='as'  # Absoprtion + Rayleigh + aerosols
    
    
    if Proc == 'sc':
        runtype='no_absorption'
        outtext='no_absorption'
    elif Proc == 'ab':
        runtype='no_scattering'
        outtext='no_scattering'
    elif Proc == 'sa':
        runtype=='clearsky'
        outtext='clearsky'
    elif Proc == 'ae':   
        runtype='aerosol_default'
        outtext='aerosol_default'
    elif Proc == 'as':   
        runtype='aerosol_special'
        outtext='aerosol_special'
    else:
        runtype=='clearsky'
        outtext='clearsky'

#   Selection of RTE equation solver        
    if Rte == 'pp': # parallel plan
        rte_eq='disort'
    elif Rte=='ps':   # pseudo spherical
        rte_eq='sdisort'
        
 
#   Selection of absorption model 
    molmodel='reptran'
    if Mod == 'rt':
        molmodel='reptran'
    if Mod == 'lt':
        molmodel='lowtran'
    if Mod == 'kt':
        molmodel='kato'
    if Mod == 'k2':
        molmodel='kato2'
    if Mod == 'fu':
        molmodel='fu'    
    if Mod == 'cr':
        molmodel='crs'     
               


    	  
    # for simulation select only two atmosphere   
    #theatmospheres = np.array(['afglus','afglms','afglmw','afglt','afglss','afglsw'])
    atmosphere_map=dict()  # map atmospheric names to short names 
    atmosphere_map['afglus']='us'
    atmosphere_map['afglms']='ms'
    atmosphere_map['afglmw']='mw'  
    atmosphere_map['afglt']='tp'  
    atmosphere_map['afglss']='ss'  
    atmosphere_map['afglsw']='sw'  
      
    theatmospheres= []
    for skyindex in Atm:
        if re.search('us',skyindex):
            theatmospheres.append('afglus')
        if re.search('sw',skyindex):
            theatmospheres.append('afglsw')
            
   
   

    # 1) LOOP ON ATMOSPHERE
    for atmosphere in theatmospheres:
        #if atmosphere != 'afglus':  # just take us standard sky
        #    break
        atmkey=atmosphere_map[atmosphere]
       
        # manage input and output directories and vary the ozone
        TOPDIR2=TOPDIR+'/'+Rte+'/'+atmkey+'/'+Proc+'/'+Mod
        ensure_dir(TOPDIR2)
        INPUTDIR=TOPDIR2+'/'+'in'
        ensure_dir(INPUTDIR)
        OUTPUTDIR=TOPDIR2+'/'+'out'
        ensure_dir(OUTPUTDIR)
    
    
        # loop on molecular model resolution
        #molecularresolution = np.array(['COARSE','MEDIUM','FINE']) 
        # select only COARSE Model
        molecularresolution = np.array(['COARSE'])    
        for molres in molecularresolution:
            if molres=='COARSE':
                molresol ='coarse'
            elif molres=='MEDIUM':
                molresol ='medium'
            else:
                molresol ='fine'
           
        
        #water vapor   
        pwv_val=pwv_num
        pwv_str='H2O '+str(pwv_val)+ ' MM'
        wvfileindex=int(10*pwv_val)
        
        #aerosols
        aerosol_str=str(wl0_num)+ ' '+str(tau0_num)
        aer_index=int(tau0_num*100.)
           
        # airmass
        airmass=airmass_num
        amfileindex=int(airmass_num*10)
        
        # Ozone    
        oz_str='O3 '+str(oz_num)+ ' DU'
        ozfileindex=int(oz_num/10.)
        
            
        BaseFilename=BaseFilename_part1+atmkey+'_'+Proc+'_'+Mod+'_z'+str(amfileindex)+'_'+WVXX+str(wvfileindex) +'_'+OZXX+str(ozfileindex)+'_'+AEXX+str(aer_index)                   
                    
        verbose=True
        uvspec = UVspec.UVspec()
        uvspec.inp["data_files_path"]  =  libradtranpath+'data'
                
        uvspec.inp["atmosphere_file"] = libradtranpath+'data/atmmod/'+atmosphere+'.dat'
        uvspec.inp["albedo"]           = '0.2'
    
        uvspec.inp["rte_solver"] = rte_eq
            
            
                
        if Mod == 'rt':
            uvspec.inp["mol_abs_param"] = molmodel + ' ' + molresol
        else:
            uvspec.inp["mol_abs_param"] = molmodel

        # Convert airmass into zenith angle 
        am=airmass
        sza=math.acos(1./am)*180./math.pi

        # Should be no_absorption
        if runtype=='aerosol_default':
            uvspec.inp["aerosol_default"] = ''
        elif runtype=='aerosol_special':
            uvspec.inp["aerosol_default"] = ''
            uvspec.inp["aerosol_set_tau_at_wvl"] = aerosol_str
                        
        if runtype=='no_scattering':
            uvspec.inp["no_scattering"] = ''
        if runtype=='no_absorption':
            uvspec.inp["no_absorption"] = ''
     
        # set up the ozone value               
        uvspec.inp["mol_modify"] = pwv_str
        uvspec.inp["mol_modify2"] = oz_str
        
        # rescale pressure   if reasonable pressure values are provided
        if press_num>600. and press_num<1015.:
            uvspec.inp["pressure"] = press_num
        else:
            print "creazy pressure p=",press_num, ' hPa'
                    
                
        uvspec.inp["output_user"] = 'lambda edir'
        uvspec.inp["altitude"] = OBS_Altitude   # Altitude LSST observatory
        uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_1.0nm.dat'
        #uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_0.1nm.dat'
        uvspec.inp["sza"]        = str(sza)
        uvspec.inp["phi0"]       = '0'
        uvspec.inp["wavelength"]       = '250.0 1200.0'
        uvspec.inp["output_quantity"] = 'reflectivity' #'transmittance' #
#       uvspec.inp["verbose"] = ''
        uvspec.inp["quiet"] = ''

  

        if "output_quantity" in uvspec.inp.keys():
            outtextfinal=outtext+'_'+uvspec.inp["output_quantity"]

           
            
        inputFilename=BaseFilename+'.INP'
        outputFilename=BaseFilename+'.OUT'
        inp=os.path.join(INPUTDIR,inputFilename)
        out=os.path.join(OUTPUTDIR,outputFilename)
                    
            
        uvspec.write_input(inp)
        uvspec.run(inp,out,verbose,path=libradtranpath)
        
        
    return OUTPUTDIR,outputFilename

#---------------------------------------------------------------------------




def ProcessSimulationaer2(airmass_num,pwv_num,oz_num,alpha_num,beta_num,press_num):  
    """
    ProcessSimulationaer2(airmass_num,pwv_num,oz_num,alpha_num,beta_num)
    with aerosol simulation is performed
    default profile
    """
 
    print '--------------------------------------------'
    print ' 1) airmass = ', airmass_num
    print ' 2) pwv = ', pwv_num
    print ' 3) oz = ', oz_num
    print ' 4) alpha = ',alpha_num
    print ' 5) beta = ',beta_num
    print ' 6) pressure =',press_num
    print '--------------------------------------------'    
   
    
    ensure_dir(TOPDIR)

    
    # build the part 1 of filename
    BaseFilename_part1=Prog+'_'+Obs+'_'+Rte+'_'
    
    Proc='as'  # Absoprtion + Rayleigh + aerosols

    # Set up type of run
    runtype='aerosol_special' #'no_scattering' #aerosol_special #aerosol_default# #'clearsky'#     
    if Proc == 'sc':
        runtype='no_absorption'
        outtext='no_absorption'
    elif Proc == 'ab':
        runtype='no_scattering'
        outtext='no_scattering'
    elif Proc == 'sa':
        runtype=='clearsky'
        outtext='clearsky'
    elif Proc == 'ae':   
        runtype='aerosol_default'
        outtext='aerosol_default'
    elif Proc == 'as':   
        runtype='aerosol_special'
        outtext='aerosol_special'
    else:
        runtype=='clearsky'
        outtext='clearsky'

#   Selection of RTE equation solver        
    if Rte == 'pp': # parallel plan
        rte_eq='disort'
    elif Rte=='ps':   # pseudo spherical
        rte_eq='sdisort'
        
 
#   Selection of absorption model 
    molmodel='reptran'
    if Mod == 'rt':
        molmodel='reptran'
    if Mod == 'lt':
        molmodel='lowtran'
    if Mod == 'kt':
        molmodel='kato'
    if Mod == 'k2':
        molmodel='kato2'
    if Mod == 'fu':
        molmodel='fu'    
    if Mod == 'cr':
        molmodel='crs'     
               


    	  
    # for simulation select only two atmosphere   
    #theatmospheres = np.array(['afglus','afglms','afglmw','afglt','afglss','afglsw'])
    atmosphere_map=dict()  # map atmospheric names to short names 
    atmosphere_map['afglus']='us'
    atmosphere_map['afglms']='ms'
    atmosphere_map['afglmw']='mw'  
    atmosphere_map['afglt']='tp'  
    atmosphere_map['afglss']='ss'  
    atmosphere_map['afglsw']='sw'  
      
    theatmospheres= []
    for skyindex in Atm:
        if re.search('us',skyindex):
            theatmospheres.append('afglus')
        if re.search('sw',skyindex):
            theatmospheres.append('afglsw')
            
   
   

    # 1) LOOP ON ATMOSPHERE
    for atmosphere in theatmospheres:
        #if atmosphere != 'afglus':  # just take us standard sky
        #    break
        atmkey=atmosphere_map[atmosphere]
       
        # manage input and output directories and vary the ozone
        TOPDIR2=TOPDIR+'/'+Rte+'/'+atmkey+'/'+Proc+'/'+Mod
        ensure_dir(TOPDIR2)
        INPUTDIR=TOPDIR2+'/'+'in'
        ensure_dir(INPUTDIR)
        OUTPUTDIR=TOPDIR2+'/'+'out'
        ensure_dir(OUTPUTDIR)
    
    
        # loop on molecular model resolution
        #molecularresolution = np.array(['COARSE','MEDIUM','FINE']) 
        # select only COARSE Model
        molecularresolution = np.array(['COARSE'])    
        for molres in molecularresolution:
            if molres=='COARSE':
                molresol ='coarse'
            elif molres=='MEDIUM':
                molresol ='medium'
            else:
                molresol ='fine'
           
        
        #water vapor   
        pwv_val=pwv_num
        pwv_str='H2O '+str(pwv_val)+ ' MM'
        wvfileindex=int(10*pwv_val)
        
        #aerosols
        #aerosol_angstrom alpha beta
        # tau = beta * lambda^-alpha, lambda in microns
        aerosol_str=str(alpha_num)+ ' '+str(beta_num)
        aer_index=int(alpha_num*100.)
           
        # airmass
        airmass=airmass_num
        amfileindex=int(airmass_num*10)
        
        # Ozone    
        oz_str='O3 '+str(oz_num)+ ' DU'
        ozfileindex=int(oz_num/10.)
        
            
        BaseFilename=BaseFilename_part1+atmkey+'_'+Proc+'_'+Mod+'_z'+str(amfileindex)+'_'+WVXX+str(wvfileindex) +'_'+OZXX+str(ozfileindex)+'_'+AEXX2+str(aer_index)                   
                    
        verbose=True
        uvspec = UVspec.UVspec()
        uvspec.inp["data_files_path"]  =  libradtranpath+'data'
                
        uvspec.inp["atmosphere_file"] = libradtranpath+'data/atmmod/'+atmosphere+'.dat'
        uvspec.inp["albedo"]           = '0.2'
    
        uvspec.inp["rte_solver"] = rte_eq
            
            
                
        if Mod == 'rt':
            uvspec.inp["mol_abs_param"] = molmodel + ' ' + molresol
        else:
            uvspec.inp["mol_abs_param"] = molmodel

        # Convert airmass into zenith angle 
        am=airmass
        sza=math.acos(1./am)*180./math.pi

        # Should be no_absorption
        if runtype=='aerosol_default':
            uvspec.inp["aerosol_default"] = ''
        elif runtype=='aerosol_special':
            uvspec.inp["aerosol_default"] = '' # wrong effect
            #uvspec.inp["aerosol_vulcan"]= 1          # Aerosol type above 2km
            #uvspec.inp["aerosol_haze"]=6            # Aerosol type below 2km
            #uvspec.inp["aerosol_season"]=1          # Summer season
            #uvspec.inp["aerosol_visibility"]= 50.0   # Visibility
    
            uvspec.inp["aerosol_angstrom"] = aerosol_str
                        
        if runtype=='no_scattering':
            uvspec.inp["no_scattering"] = ''
        if runtype=='no_absorption':
            uvspec.inp["no_absorption"] = ''
     
        # set up the ozone value               
        uvspec.inp["mol_modify"] = pwv_str
        uvspec.inp["mol_modify2"] = oz_str
        
        # rescale pressure   if reasonable pressure values are provided
        if press_num>600. and press_num<1015.:
            uvspec.inp["pressure"] = press_num
        else:
             print "creazy pressure p=",press_num, ' hPa'
                    
                
        uvspec.inp["output_user"] = 'lambda edir'
        uvspec.inp["altitude"] = OBS_Altitude   # Altitude LSST observatory
        uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_1.0nm.dat'
        #uvspec.inp["source"] = 'solar '+libradtranpath+'data/solar_flux/kurudz_0.1nm.dat'
        uvspec.inp["sza"]        = str(sza)
        uvspec.inp["phi0"]       = '0'
        uvspec.inp["wavelength"]       = '250.0 1200.0'
        uvspec.inp["output_quantity"] = 'reflectivity' #'transmittance' #
#       uvspec.inp["verbose"] = ''
        uvspec.inp["quiet"] = ''

  

        if "output_quantity" in uvspec.inp.keys():
            outtextfinal=outtext+'_'+uvspec.inp["output_quantity"]

           
            
        inputFilename=BaseFilename+'.INP'
        outputFilename=BaseFilename+'.OUT'
        inp=os.path.join(INPUTDIR,inputFilename)
        out=os.path.join(OUTPUTDIR,outputFilename)
                    
            
        uvspec.write_input(inp)
        uvspec.run(inp,out,verbose,path=libradtranpath)
        
        
    return OUTPUTDIR,outputFilename

#---------------------------------------------------------------------------













#####################################################################
# The program simulation start here
#    NEED TO BE RE-WRITTEN 
#
####################################################################

if __name__ == "__main__":
    
    
    AerosolTest_Flag=False
    
    # init string variables
    airmass_str=""
    pwv_str=""
    oz_str=""
    press_str=""
    aer_str=""
    wl0_str=""
    tau0_str=""
    
    # Case No Aerosols
    
    if AerosolTest_Flag==False:
        try:
            opts, args = getopt.getopt(sys.argv[1:],"hz:w:o:p:",["z=","w=","o=","p="])
        except getopt.GetoptError:
            print ' Exception bad getopt with :: '+sys.argv[0]+ ' -z <airmass> -w <pwv> -o <oz> -p <press>'
            sys.exit(2)
        
    
        
        print 'opts = ',opts
        print 'args = ',args    
        
        
        for opt, arg in opts:
            if opt == '-h':
                usage()
                sys.exit()
            elif opt in ("-z", "--airmass"):
                airmass_str = arg
            elif opt in ("-w", "--pwv"):
                pwv_str = arg
            elif opt in ("-o", "--oz"):
                oz_str = arg  
            elif opt in ("-p", "--pr"):
                press_str = arg 
            else:
                print 'Do not understand arguments : ',argv
            
         
        print '--------------------------------------------'     
        print '1) airmass-str = ', airmass_str
        print '2) pwv-str = ', pwv_str
        print "3) oz-str = ", oz_str  
        print "4) pr = ", press_str
        print '--------------------------------------------' 

        if airmass_str=="":
            usage()
            sys.exit()

        if pwv_str=="":
            usage()
            sys.exit()

        if oz_str=="":
            usage()
            sys.exit()
            
        if press_str=="":
            usage()
            sys.exit()
        
    
        
	
	
        airmass_nb=float(airmass_str)
        pwv_nb=float(pwv_str)
        oz_nb=float(oz_str)	
        press_nb=float(press_str)
        
        print '--------------------------------------------'     
        print '1) airmass  = ', airmass_nb
        print '2) pwv = ', pwv_nb
        print "3) oz = ", oz_nb
        print "4) press = ", press_nb
        print '--------------------------------------------' 
        
    
        if airmass_nb<1 or airmass_nb >3 :
            print "bad airmass value z=",airmass_nb
            sys.exit()
            
        if pwv_nb<0 or pwv_nb >50 :
            print "bad PWV value pwv=",pwv_nb
            sys.exit()
        
        if oz_nb<0 or oz_nb >600 :
            print "bad Ozone value oz=",oz_nb
            sys.exit()
            
        
        
        if press_nb<0 or press_nb >1500 :
            print "bad Pressure value press=",press_nb
            sys.exit()
        
        
        # do the simulation now 
        print "values are OK"
    
        path, outputfile=ProcessSimulation(airmass_nb,pwv_nb,oz_nb,press_nb)
    
        print '*****************************************************'
        print ' path       = ', path
        print ' outputfile =  ', outputfile 
        print '*****************************************************'   
    
    
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:],"hz:w:o:a:p:",["z=","w=","o=","a=","p="])
        except getopt.GetoptError:
            print ' Exception bad getopt with :: '+sys.argv[0]+ ' -z <airmass> -w <pwv> -o <oz> -a <aer> -p <press>'
            sys.exit(2)
        
    
        
        print 'opts = ',opts
        print 'args = ',args    
        
        
        for opt, arg in opts:
            if opt == '-h':
                usage()
                sys.exit()
            elif opt in ("-z", "--airmass"):
                airmass_str = arg
            elif opt in ("-w", "--pwv"):
                pwv_str = arg
            elif opt in ("-o", "--oz"):
                oz_str = arg  
            elif opt in ("-a", "--aer"):
                aer_str = arg 
            elif opt in ("-p", "--pr"):
                press_str = arg 
            else:
                print 'Do not understand arguments : ',argv
            
         
        print '--------------------------------------------'     
        print '1) airmass-str = ', airmass_str
        print '2) pwv-str = ', pwv_str
        print "3) oz-str = ", oz_str  
        print "4) aer = ", aer_str
        print "5) pr = ", press_str
        print '--------------------------------------------' 

        if airmass_str=="":
            usage()
            sys.exit()

        if pwv_str=="":
            usage()
            sys.exit()

        if oz_str=="":
            usage()
            sys.exit()
            
        if press_str=="":
            usage()
            sys.exit()
        
        if aer_str=="":
            usage()
            sys.exit()
        
	
	
        airmass_nb=float(airmass_str)
        pwv_nb=float(pwv_str)
        oz_nb=float(oz_str)	
        aer_nb=float(aer_str)
        press_nb=float(press_str)
        
        print '--------------------------------------------'     
        print '1) airmass  = ', airmass_nb
        print '2) pwv = ', pwv_nb
        print "3) oz = ", oz_nb
        print "4) aer = ", aer_nb
        print "5) press = ", press_nb
        print '--------------------------------------------' 
        
    
        if airmass_nb<1 or airmass_nb >3 :
            print "bad airmass value z=",airmass_nb
            sys.exit()
            
        if pwv_nb<0 or pwv_nb >50 :
            print "bad PWV value pwv=",pwv_nb
            sys.exit()
        
        if oz_nb<0 or oz_nb >600 :
            print "bad Ozone value oz=",oz_nb
            sys.exit()
            
        if aer_nb<0 or aer_nb >0.5 :
            print "bad Aerosol value aer=",aer_nb
            sys.exit()
        
        
        if press_nb<0 or press_nb >1500 :
            print "bad Pressure value press=",press_nb
            sys.exit()
        
        
        # do the simulation now 
        print "values are OK"
    
        path, outputfile=ProcessSimulationaer(airmass_nb,pwv_nb,oz_nb,aer_nb,press_nb)
    
        print '*****************************************************'
        print ' path       = ', path
        print ' outputfile =  ', outputfile 
        print '*****************************************************'
        
        

        
        
       
   

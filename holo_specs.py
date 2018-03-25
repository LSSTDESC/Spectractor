import numpy as np
import os, sys
import copy
from scipy import ndimage
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tools import *
from scipy.signal import argrelextrema
from astropy.table import Table

# CCD characteristics
IMSIZE = 2048 # size of the image in pixel
PIXEL2MM = 24e-3 # pixel size in mm
PIXEL2ARCSEC = 0.401 # pixel size in arcsec
ARCSEC2RADIANS = np.pi/(180.*3600.) # conversion factor from arcsec to radians
DISTANCE2CCD = 55.45 # distance between hologram and CCD in mm
DISTANCE2CCD_ERR = 0.19 # uncertainty on distance between hologram and CCD in mm
MAXADU = 60000 # approximate maximum ADU output of the CCD

# Making of the holograms
LAMBDA_CONSTRUCTOR = 639e-6 # constructor wavelength to make holograms in mm
GROOVES_PER_MM = 350 # approximate effective number of lines per millimeter of the hologram
PLATE_CENTER_SHIFT_X = -6. # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y = -8. # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_X_ERR = 2. # estimate uncertainty on plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y_ERR = 2. # estimate uncertainty on plate center shift on x in mm in filter frame

# CTIO latitude
CTIO_LATITUDE = '-30 10 07.90'

# Main emission/absorption lines in nm
HALPHA = {'lambda':656.3,'atmospheric':False,'label':'$H\\alpha$','pos':[-0.016,0.02]}
HBETA = {'lambda': 486.3,'atmospheric':False,'label':'$H\\beta$','pos':[0.007,0.02]} 
HGAMMA = {'lambda':434.0,'atmospheric':False,'label':'$H\\gamma$','pos':[0.007,0.02]} 
HDELTA = {'lambda': 410.2,'atmospheric':False,'label':'$H\\delta$','pos':[0.007,0.02]}
OIII = {'lambda': 500.7,'atmospheric':False,'label':'$O_{III}$','pos':[0.007,0.02]}
CII1 =  {'lambda': 723.5,'atmospheric':False,'label':'$C_{II}$','pos':[0.005,0.92]}
CII2 =  {'lambda': 711.0,'atmospheric':False,'label':'$C_{II}$','pos':[0.005,0.02]}
CIV =  {'lambda': 706.0,'atmospheric':False,'label':'$C_{IV}$','pos':[-0.016,0.92]}
CII3 =  {'lambda': 679.0,'atmospheric':False,'label':'$C_{II}$','pos':[0.005,0.02]}
CIII1 =  {'lambda': 673.0,'atmospheric':False,'label':'$C_{III}$','pos':[-0.016,0.92]}
CIII2 =  {'lambda': 570.0,'atmospheric':False,'label':'$C_{III}$','pos':[0.007,0.02]}
HEI =  {'lambda': 587.5,'atmospheric':False,'label':'$He_{I}$','pos':[0.007,0.02]}
HEII =  {'lambda': 468.6,'atmospheric':False,'label':'$He_{II}$','pos':[0.007,0.02]}
CAII1 =  {'lambda': 393.366,'atmospheric':True,'label':'$Ca_{II}$','pos':[0.007,0.02]} # https://en.wikipedia.org/wiki/Fraunhofer_lines
CAII2 =  {'lambda': 396.847,'atmospheric':True,'label':'$Ca_{II}$','pos':[0.007,0.02]} # https://en.wikipedia.org/wiki/Fraunhofer_lines
O2 = {'lambda': 762.1,'atmospheric':True,'label':'$O_2$','pos':[0.007,0.02]} # http://onlinelibrary.wiley.com/doi/10.1029/98JD02799/pdf
#O2_1 = {'lambda': 760.6,'atmospheric':True,'label':'','pos':[0.007,0.02]} # libradtran paper fig.3
#O2_2 = {'lambda': 763.2,'atmospheric':True,'label':'$O_2$','pos':[0.007,0.02]}  # libradtran paper fig.3
O2B = {'lambda': 686.719,'atmospheric':True,'label':'$O_2(B)$','pos':[0.007,0.02]} # https://en.wikipedia.org/wiki/Fraunhofer_lines
O2Y = {'lambda': 898.765,'atmospheric':True,'label':'$O_2(Y)$','pos':[0.007,0.02]} # https://en.wikipedia.org/wiki/Fraunhofer_lines
O2Z = {'lambda': 822.696,'atmospheric':True,'label':'$O_2(Z)$','pos':[0.007,0.02]} # https://en.wikipedia.org/wiki/Fraunhofer_lines
#H2O = {'lambda': 960,'atmospheric':True,'label':'$H_2 O$','pos':[0.007,0.02]}  # 
H2O_1 = {'lambda': 950,'atmospheric':True,'label':'$H_2 O$','pos':[0.007,0.02]}  # libradtran paper fig.3
H2O_2 = {'lambda': 970,'atmospheric':True,'label':'$H_2 O$','pos':[0.007,0.02]}  # libradtran paper fig.3
LINES = [HALPHA,HBETA,HGAMMA,HDELTA,O2,O2B,O2Y,O2Z,H2O_1,H2O_2,OIII,CII1,CII2,CIV,CII3,CIII1,CIII2,HEI,HEII,CAII1,CAII2]
LINES = sorted(LINES, key=lambda x: x['lambda'])


# H-alpha filter
HALPHA_CENTER = 655.9e-6 # center of the filter in mm
HALPHA_WIDTH = 6.4e-6 # width of the filter in mm

# Other filters
FGB37 = {'label':'FGB37','min':300,'max':800}
RG715 = {'label':'RG715','min':690,'max':1100}
HALPHA_FILTER = {'label':'Halfa','min':HALPHA_CENTER-2*HALPHA_WIDTH,'max':HALPHA_CENTER+2*HALPHA_WIDTH}
ZGUNN = {'label':'Z-Gunn','min':800,'max':1100}
FILTERS = [RG715,FGB37,HALPHA_FILTER,ZGUNN]

def sort_redshifted_lines(redshift=0):
    if redshift > 0 :
        sorted_LINES = copy.deepcopy(LINES)
        for LINE in sorted_LINES:
            if not LINE['atmospheric'] : LINE['lambda'] *= (1+redshift)
        sorted_LINES = sorted(sorted_LINES, key=lambda x: x['lambda'])
        return sorted_LINES
    else:
        return LINES


DATA_DIR = "../../common_tools/data/"


def plot_atomic_lines(ax,redshift=0,atmospheric_lines=True,hydrogen_only=False,color_atomic='g',color_atmospheric='b',fontsize=16):
    xlim = ax.get_xlim()
    for LINE in LINES:
        if hydrogen_only :
            if not atmospheric_lines :
                if LINE['atmospheric'] : continue
                if '$H\\' not in LINE['label'] : continue
            else :
                if not LINE['atmospheric'] and '$H\\' not in LINE['label'] : continue
        else :
            if not atmospheric_lines and LINE['atmospheric'] : continue
        #if not atmospheric_lines and line['atmospheric']: continue
        #if hydrogen_only and '$H\\' not in line['label'] : continue
        color = color_atomic
        l = LINE['lambda']
        if not LINE['atmospheric'] : l = l*(1+redshift)
        if LINE['atmospheric']: color = color_atmospheric
        ax.axvline(l,lw=2,color=color)
        xpos = (l-xlim[0])/(xlim[1]-xlim[0])+LINE['pos'][0]
        if xpos > 0 and xpos < 1 :
            ax.annotate(LINE['label'],xy=(xpos,LINE['pos'][1]),rotation=90,ha='left',va='bottom',xycoords='axes fraction',color=color,fontsize=fontsize)

def detect_lines(lambdas,spec,redshift=0,emission_spectrum=False,snr_minlevel=3,atmospheric_lines=True,hydrogen_only=False,ax=None,verbose=False):
    bgd_npar = parameters.BGD_NPARAMS
    peak_look = 7 # half range to look for local maximum in pixels
    bgd_width = 7 # size of the peak sides to use to fit spectrum base line
    if hydrogen_only :
        peak_look = 15
        bgd_width = 15
    baseline_prior = 3 # gaussian prior on base line fit
    lambda_shifts = []
    snrs = []
    index_list = []
    guess_list = []
    bounds_list = []
    lines_list = []
    sorted_LINES = sort_redshifted_lines(redshift)
    for LINE in sorted_LINES:
        if hydrogen_only :
            if not atmospheric_lines :
                if LINE['atmospheric'] : continue
                if '$H\\' not in LINE['label'] : continue
            else :
                if not LINE['atmospheric'] and '$H\\' not in LINE['label'] : continue
        else :
            if not atmospheric_lines and LINE['atmospheric'] : continue
        #if not((atmospheric_lines and LINE['atmospheric']) or (hydrogen_only and '$H\\' in LINE['label'])) : continue
        # wavelength of the line
        l = LINE['lambda']
        #if not LINE['atmospheric'] : l = l*(1+redshift)
        l_index, l_lambdas = find_nearest(lambdas,l)
        if l_index < peak_look or l_index > len(lambdas)-peak_look : continue
        # look for local extrema to detect emission or absorption line
        line_strategy = np.greater  # look for emission line
        bgd_strategy = np.less
        if not emission_spectrum or LINE['atmospheric']:
            line_strategy = np.less # look for absorption line
            bgd_strategy = np.greater
        index = range(l_index-peak_look,l_index+peak_look)
        extrema = argrelextrema(spec[index], line_strategy)
        if len(extrema[0])==0 : continue
        peak_index = index[0] + extrema[0][0]
        # if several extrema, look for the greatest
        if len(extrema[0])>1 :
            if line_strategy == np.greater :
                test = -1e20
                for m in extrema[0]:
                    idx = index[0] + m
                    if spec[idx] > test :
                        peak_index = idx
                        test = spec[idx]
            elif line_strategy == np.less :
                test = 1e20
                for m in extrema[0]:
                    idx = index[0] + m
                    if spec[idx] < test :
                        peak_index = idx
                        test = spec[idx]
        # look for first local minima around the local maximum
        index_inf = peak_index - 1
        while index_inf > max(0,peak_index - 3*peak_look) :
            test_index = range(index_inf,peak_index)
            minm = argrelextrema(spec[test_index], bgd_strategy)
            if len(minm[0]) > 0 :
                index_inf = index_inf + minm[0][0] 
                break
            else :
                index_inf -= 1
        index_sup = peak_index + 1
        while index_sup < min(len(spec)-1,peak_index + 3*peak_look) :
            test_index = range(peak_index,index_sup)
            minm = argrelextrema(spec[test_index], bgd_strategy)
            if len(minm[0]) > 0 :
                index_sup = peak_index + minm[0][0] 
                break
            else :
                index_sup += 1
        index = range(max(0,index_inf-bgd_width),min(len(lambdas),index_sup+bgd_width))
        # guess and bounds to fit this line
        # min sigma at 1 pixel and max sigma at 10 pixels (half width)
        guess = [0]*bgd_npar+[0*abs(spec[peak_index]),lambdas[peak_index],3]
        bounds = [[-np.inf]*bgd_npar+[-np.inf,lambdas[index_inf],1], [np.inf]*bgd_npar+[2*np.max(spec[index]),lambdas[index_sup],10]  ]
        # gaussian amplitude bounds depend if line is emission/absorption
        if line_strategy == np.less :
            bounds[1][bgd_npar] = 0
        else :
            bounds[0][bgd_npar] = 0
        index_list.append(index)
        lines_list.append(LINE)
        guess_list.append(guess)
        bounds_list.append(bounds)
    # Now gather lines together if index ranges overlap
    idx = 0
    merges = [[0]]
    while idx < len(index_list) - 1 :
        idx = merges[-1][-1]
        if idx == len(index_list)-1 : break
        if index_list[idx][-1] > index_list[idx+1][0] :
            merges[-1].append(idx+1)
        else :
            merges.append([idx+1])
            idx += 1
    # reorder merge list with respect to lambdas in guess list
    new_merges = []
    for merge in merges:
        tmp_guess = [guess_list[i][bgd_npar+1] for i in merge]
        new_merges.append( [x for _,x in sorted(zip(tmp_guess,merge))] )
    # reorder lists with merges
    new_index_list = []
    new_guess_list = []
    new_bounds_list = []
    new_lines_list = []
    for merge in new_merges :
        new_index_list.append([])
        new_guess_list.append([])
        new_bounds_list.append([[],[]])
        new_lines_list.append([])
        for i in merge :
            # add the bgd parameters 
            if i == merge[0] :
                new_guess_list[-1] += guess_list[i][:bgd_npar]
                new_bounds_list[-1][0] += bounds_list[i][0][:bgd_npar]
                new_bounds_list[-1][1] += bounds_list[i][1][:bgd_npar]
            # add the gauss parameters
            new_index_list[-1] += index_list[i]
            new_guess_list[-1] += guess_list[i][bgd_npar:]
            new_bounds_list[-1][0] += bounds_list[i][0][bgd_npar:]
            new_bounds_list[-1][1] += bounds_list[i][1][bgd_npar:]
            new_lines_list[-1].append(lines_list[i])
        # set central peak bounds at middle of the lines
        for k in range(len(merge)-1) :
            new_bounds_list[-1][0][bgd_npar+3*(k+1)+1]  = 0.5*(new_guess_list[-1][bgd_npar+3*k+1]+new_guess_list[-1][bgd_npar+3*(k+1)+1])
            new_bounds_list[-1][1][bgd_npar+3*k+1] = 0.5*(new_guess_list[-1][bgd_npar+3*k+1]+new_guess_list[-1][bgd_npar+3*(k+1)+1])
        new_index_list[-1] = sorted(list(set(new_index_list[-1])))
    rows = []
    for k in range(len(new_index_list)):
        # first guess for the base line
        index = new_index_list[k]
        guess = new_guess_list[k]
        bounds = new_bounds_list[k]
        bgd_index = index[:bgd_width]+index[-bgd_width:]
        line_popt, line_pcov = fit_bgd(lambdas[bgd_index],spec[bgd_index])
        for n in range(bgd_npar):
            guess[n] = line_popt[n]
            bounds[0][n] = line_popt[n]-baseline_prior*np.sqrt(line_pcov[n][n])
            bounds[1][n] = line_popt[n]+baseline_prior*np.sqrt(line_pcov[n][n])
        # fit local maximum with a gaussian + line
        popt, pcov = fit_multigauss_and_bgd(lambdas[index],spec[index],guess=guess, bounds=bounds)
        # compute the base line subtracting the gaussians
        base_line = spec[index]
        for j in range(len(new_lines_list[k])) :
            base_line -= gauss(lambdas[index],*popt[bgd_npar+3*j:bgd_npar+3*j+3])
        # noise level defined as the std of the residuals
        noise_level = np.std(spec[index]-multigauss_and_bgd(lambdas[index],*popt))
        for j in range(len(new_lines_list[k])) :
            LINE = new_lines_list[k][j]
            l = LINE['lambda']
            peak_pos = popt[bgd_npar+3*j+1]
            # SNR computation
            signal_level = popt[bgd_npar+3*j]
            snr = np.abs(signal_level / noise_level)
            if snr < snr_minlevel : continue
            # FWHM
            FWHM = np.abs(popt[bgd_npar+3*j+bgd_npar])*2.355
            rows.append((LINE["label"],l,peak_pos,peak_pos-l,FWHM,signal_level,snr))
            # wavelength shift between tabulate and observed lines
            lambda_shifts.append(peak_pos-l)
            snrs.append(snr)
        if ax is not None :
            ax.plot(lambdas[index],multigauss_and_bgd(lambdas[index],*popt),lw=2,color='b')
            ax.plot(lambdas[index],np.polyval(popt[:bgd_npar],lambdas[index]),lw=2,color='b',linestyle='--')
    if len(rows) > 0 :
        t = Table(rows=rows,names=('Line','Tabulated','Detected','Shift','FWHM','Amplitude','SNR'),dtype=('a10','f4','f4','f4','f4','f4','f4'))
        for col in t.colnames[1:-2] : t[col].unit = 'nm'
        if verbose : print t
        shift =  np.average(lambda_shifts,weights=np.array(snrs)**2)
    else :
        shift = 0
    return shift

def build_hologram(order0_position,order1_position,theta_tilt,lambda_plot=256000):
    # wavelength in nm, hologram porduced at 639nm
    # spherical wave centered in 0,0,0
    U = lambda x,y,z : np.exp(2j*np.pi*np.sqrt(x*x + y*y + z*z)*1e6/lambda_plot)/np.sqrt(x*x + y*y + z*z) 
    # superposition of two spherical sources centered in order 0 and order 1 positions
    plot_center = 0.5*IMSIZE*PIXEL2MM
    xA = [order0_position[0]*PIXEL2MM,order0_position[1]*PIXEL2MM]
    xB = [order1_position[0]*PIXEL2MM,order1_position[1]*PIXEL2MM]
    A = lambda x,y : U(x-xA[0],y-xA[1],-DISTANCE2CCD)+U(x-xB[0],y-xB[1],-DISTANCE2CCD)
    intensity = lambda x,y : np.abs(A(x,y))**2
    xholo = np.linspace(0,IMSIZE*PIXEL2MM,IMSIZE)
    yholo = np.linspace(0,IMSIZE*PIXEL2MM,IMSIZE)
    xxholo, yyholo = np.meshgrid(xholo,yholo)
    holo = intensity(xxholo,yyholo)
    return(holo)


def build_ronchi(x_center,y_center,theta_tilt,grooves=400):
    intensity = lambda x,y : 2*np.sin(2*np.pi*(x-x_center*PIXEL2MM)*0.5*grooves)**2

    xronchi = np.linspace(0,IMSIZE*PIXEL2MM,IMSIZE)
    yronchi = np.linspace(0,IMSIZE*PIXEL2MM,IMSIZE)
    xxronchi, yyronchi = np.meshgrid(xronchi,yronchi)
    ronchi = (intensity(xxronchi,yyronchi)).astype(int)
    rotated_ronchi=ndimage.interpolation.rotate(ronchi,theta_tilt)
    return(ronchi)



def get_theta0(x0):
    """ Return incident angle on grating in radians.
    x0: the order 0 position in the full raw image."""
    if isinstance(x0, (list, tuple, np.ndarray)) :
        return (x0[0] - IMSIZE/2)*PIXEL2ARCSEC*ARCSEC2RADIANS
    else :
        return (x0 - IMSIZE/2)*PIXEL2ARCSEC*ARCSEC2RADIANS

    
def get_delta_pix_ortho(deltaX,x0,D=DISTANCE2CCD):
    """ Return the distance in pixels between pixel x and 
    projected incident point on grating. D is in mm.
    x0 is the order 0 position in the full raw image.
    deltaX is the distance in pixels between order 0 and signal point 
    in the rotated image."""
    theta0 = get_theta0(x0)
    deltaX0 = np.tan(theta0)*D/PIXEL2MM 
    return deltaX + deltaX0

def get_refraction_angle(deltaX,x0,D=DISTANCE2CCD):
    """ Return the refraction angle from order 0 and x positions.
    x0 is the order 0 position in the full raw image.
    deltaX is the distance in pixels between order 0 and signal point 
    in the rotated image."""
    delta = get_delta_pix_ortho(deltaX,x0,D=D)
    theta = np.arctan2(delta*PIXEL2MM,D)
    return theta

def get_N(deltaX,x0,D=DISTANCE2CCD,l=HALPHA_CENTER,order=1):
    """ Return grooves per mm given the signal x position with 
    its wavelength in mm, the distance to CCD in mm and the order number.
    x0 is the order 0 position in the full raw image.
    deltaX is the distance in pixels between order 0 and signal point 
    in the rotated image."""
    theta = get_refraction_angle(deltaX,x0,D=D)
    theta0 = get_theta0(x0)
    N = (np.sin(theta)-np.sin(theta0))/(order*HALPHA_CENTER)
    return N
    
def neutral_lines(x_center,y_center,theta_tilt):
    xs = np.linspace(0,IMSIZE,20)
    line1 = np.tan(theta_tilt*np.pi/180)*(xs-x_center)+y_center
    line2 = np.tan((theta_tilt+90)*np.pi/180)*(xs-x_center)+y_center
    return(xs,line1,line2)

def order01_positions(holo_center,N,theta_tilt,theta0=0,verbose=True):
    # refraction angle between order 0 and order 1 at construction
    alpha = np.arcsin(N*LAMBDA_CONSTRUCTOR + np.sin(theta0)) 
    # distance between order 0 and order 1 in pixels
    AB = (np.tan(alpha)-np.tan(theta0))*DISTANCE2CCD/PIXEL2MM
    # position of order 1 in pixels
    x_center = holo_center[0]
    y_center = holo_center[1]
    order1_position = [ 0.5*AB*np.cos(theta_tilt*np.pi/180)+x_center, 0.5*AB*np.sin(theta_tilt*np.pi/180)+y_center] 
    # position of order 0 in pixels
    order0_position = [ -0.5*AB*np.cos(theta_tilt*np.pi/180)+x_center, -0.5*AB*np.sin(theta_tilt*np.pi/180)+y_center]
    if verbose :
        print 'Order  0 position at x0 = %.1f and y0 = %.1f' % (order0_position[0],order0_position[1])
        print 'Order +1 position at x0 = %.1f and y0 = %.1f' % (order1_position[0],order1_position[1])
        print 'Distance between the orders: %.2f pixels (%.2f mm)' % (AB,AB*PIXEL2MM)
    return(order0_position,order1_position,AB)
    
def find_order01_positions(holo_center,N_interp,theta_interp,verbose=True):
    N= N_interp(holo_center)
    theta_tilt = theta_interp(holo_center)
    theta0 = 0
    convergence = 0
    while abs(N - convergence) > 1e-6:
        order0_position, order1_position, AB = order01_positions(holo_center,N,theta_tilt,theta0,verbose=False)
        convergence = np.copy(N)
        N = N_interp(order0_position)
        theta_tilt = theta_interp(order0_position)
        theta0 = get_theta0(order0_position)
    order0_position, order1_position, AB = order01_positions(holo_center,N,theta_tilt,theta0,verbose=verbose)
    return(order0_position,order1_position,AB)






class Grating():
    def __init__(self,N,label="",D=DISTANCE2CCD,data_dir=DATA_DIR,verbose=False):
        self.N_input = N
        self.N_err = 1
        self.D = D
        self.label = label
        self.data_dir = data_dir
        self.load_files(verbose=verbose)

    def N(self,x) :
        return self.N_input

    def load_files(self,verbose=False):
        filename = self.data_dir+self.label+"/N.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.N_input = a[0]
            self.N_err = a[1]
        filename = self.data_dir+self.label+"/hologram_center.txt"
        if os.path.isfile(filename):
            lines = [line.rstrip('\n') for line in open(filename)]
            self.theta_tilt = float(lines[1].split(' ')[2])
        else :
            self.theta_tilt = 0
            return
        self.plate_center = [0.5*IMSIZE,0.5*IMSIZE]
        if verbose : print 'Grating plate center at x0 = %.1f and y0 = %.1f with average tilt of %.1f degrees' % (self.plate_center[0],self.plate_center[1],self.theta_tilt)
        
    def refraction_angle(self,deltaX,x0):
        """ Refraction angle in radians. 
        x0: the order 0 position on the full raw image.
        deltaX: the distance in pixels between order 0 and signal point 
        in the rotated image."""
        theta = get_refraction_angle(deltaX,x0,D=self.D)
        return( theta )
        
    def refraction_angle_lambda(self,l,x0,order=1):
        """ Return refraction angle in radians with lambda in mm. 
        x0: the order 0 position on the full raw image."""
        theta0 = get_theta0(x0)
        return( np.arcsin(order*l*self.N(x0) + np.sin(theta0) ) )
        
    def grating_pixel_to_lambda(self,deltaX,x0,order=1):
        """ Convert pixels into wavelength in nm.
        x0: the order 0 position on the full raw image.
        deltaX: the distance in pixels between order 0 and signal point 
        in the rotated image."""
        theta = self.refraction_angle(deltaX,x0)
        theta0 = get_theta0(x0)
        l = (np.sin(theta)-np.sin(theta0))/(order*self.N(x0))
        return(l*1e6)

    def grating_resolution(self,deltaX,x0,order=1):
        """ Return wavelength resolution in nm per pixel.
        See mathematica notebook: derivative of the grating formula.
        x0: the order 0 position on the full raw image.
        deltaX: the distance in pixels between order 0 and signal point 
        in the rotated image."""
        delta = get_delta_pix_ortho(deltaX,x0,D=self.D)*PIXEL2MM
        #theta = self.refraction_angle(x,x0,order=order)
        #res = (np.cos(theta)**3*PIXEL2MM*1e6)/(order*self.N(x0)*self.D)
        res = (self.D**2/pow(self.D**2+delta**2,1.5))*PIXEL2MM*1e6/(order*self.N(x0))
        return(res)


        
class Hologram(Grating):

    def __init__(self,label,D=DISTANCE2CCD,lambda_plot=256000,data_dir=DATA_DIR,verbose=True):
        Grating.__init__(self,GROOVES_PER_MM,D=D,label=label,data_dir=data_dir,verbose=False)
        self.holo_center = None # center of symmetry of the hologram interferences in pixels
        self.plate_center = None # center of the hologram plate
        self.theta = None # interpolated rotation angle map of the hologram from data in degrees
        self.theta_data = None # rotation angle map data of the hologram from data in degrees
        self.theta_x = None # x coordinates for the interpolated rotation angle map 
        self.theta_y = None # y coordinates for the interpolated rotation angle map
        self.N_x = None
        self.N_y = None
        self.N_data = None
        self.lambda_plot = lambda_plot
        self.load_specs(verbose=verbose)
        self.is_hologram = True

    def N(self,x):
        N = GROOVES_PER_MM
        if x[0] < np.min(self.N_x) or x[0] > np.max(self.N_x) or x[1] < np.min(self.N_y) or x[1] > np.max(self.N_y) :
            N = self.N_fit(x[0],x[1])
        else :
            N = self.N_interp(x)[0]
        return N
    
    def load_specs(self,verbose=True):
        if verbose :
            print 'Load disperser %s:' % self.label
            print '\tfrom %s' % self.data_dir+self.label
        filename = self.data_dir+self.label+"/hologram_grooves_per_mm.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.N_x, self.N_y, self.N_data = a.T
            N_interp = interpolate.interp2d(self.N_x, self.N_y, self.N_data, kind='cubic')
            self.N_fit = fit_poly2d(self.N_x,self.N_y, self.N_data, degree=2) 
            self.N_interp = lambda x : N_interp(x[0],x[1])
        else :
            self.is_hologram = False
            filename = self.data_dir+self.label+"/N.txt"
            if os.path.isfile(filename):
                a = np.loadtxt(filename)            
                self.N_interp = lambda x : a[0]
                self.N_fit = lambda x,y : a[0]
            else :
                self.N_interp = lambda x : GROOVES_PER_MM
                self.N_fit = lambda x,y : GROOVES_PER_MM
        filename = self.data_dir+self.label+"/hologram_center.txt"
        if os.path.isfile(filename):
            lines = [line.rstrip('\n') for line in open(filename)]
            self.holo_center = map(float,lines[1].split(' ')[:2])
            self.theta_tilt = float(lines[1].split(' ')[2])
        else :
            self.holo_center = [0.5*IMSIZE,0.5*IMSIZE]
            self.theta_tilt = 0
        filename = self.data_dir+self.label+"/hologram_rotation_angles.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.theta_x, self.theta_y, self.theta_data = a.T
            theta_interp = interpolate.interp2d(self.theta_x, self.theta_y, self.theta_data, kind='cubic')
            self.theta = lambda x : theta_interp(x[0],x[1])[0]
        else :
            self.theta = lambda x: self.theta_tilt
        self.plate_center = [0.5*IMSIZE+PLATE_CENTER_SHIFT_X/PIXEL2MM,0.5*IMSIZE+PLATE_CENTER_SHIFT_Y/PIXEL2MM] 
        self.x_lines, self.line1, self.line2 = neutral_lines(self.holo_center[0],self.holo_center[1],self.theta_tilt)
        if verbose :
            if self.is_hologram:
                print 'Hologram characteristics:'
                print '\tN = %.2f +/- %.2f grooves/mm at plate center' % (self.N(self.plate_center), self.N_err)
                print '\tPlate center at x0 = %.1f and y0 = %.1f with average tilt of %.1f degrees' % (self.plate_center[0],self.plate_center[1],self.theta_tilt)
                print '\tHologram center at x0 = %.1f and y0 = %.1f with average tilt of %.1f degrees' % (self.holo_center[0],self.holo_center[1],self.theta_tilt)
            else:
                print 'Grating characteristics:'
                print '\tN = %.2f +/- %.2f grooves/mm' % (self.N([0,0]), self.N_err)
                print '\tAverage tilt of %.1f degrees' % (self.theta_tilt)
        if self.is_hologram:
            self.order0_position, self.order1_position, self.AB = find_order01_positions(self.holo_center,self.N_interp,self.theta,verbose=verbose)
        #if verbose :
        #    print 'At order 0 position: N=%.2f grooves/mm and theta=%.2f degrees' % (self.N(self.order0_position),self.theta(self.order0_position))
        #self.hologram_shape = build_hologram(self.order0_position,self.order1_position,self.theta_tilt,lambda_plot=self.lambda_plot)
            




                
def EmissionLineFit(spectra,left_edge=1200,right_edge=1600,guess=[10,1400,200],bounds=(-np.inf,np.inf)):
    xs = np.arange(left_edge,right_edge,1)
    right_spectrum = spectra[left_edge:right_edge]
    popt, pcov = fit_gauss(xs,right_spectrum,guess=guess)
    return(popt, pcov)

def CalibrateDistance2CCD_OneOrder(thecorrspectra,thex0,order0_positions,all_filt,xlim=(1200,1600),guess=[10,1400,200],bounds=(-np.inf,np.inf),order=1):
    NBSPEC=0
    for index in range(len(thecorrspectra)):
        if "Ron400" not in all_filt[index] and "Thor300" not in all_filt[index] :
            continue
        NBSPEC += 1
    f, axarr = plt.subplots(NBSPEC,2,figsize=(25,5*NBSPEC))
    count = 0
    D_range = 1 # in mm
    print 'Present distance to CCD : %.2f mm (to update if necessary)' % DISTANCE2CCD
    print '-------------------------------'    
    distances = []
    distances_err = []
    for index in range(len(thecorrspectra)):
        if "Ron400" not in all_filt[index] and "Thor300" not in all_filt[index] :
            continue
        if "Thor300" in all_filt[index] : N_theo = 300
        if "Ron400" in all_filt[index] : N_theo = 400
        # set x limits
        if type(xlim[0]) is list :
            left_edge = int(xlim[index][0])
            right_edge = int(xlim[index][1])
            guess[1] = 0.5*(left_edge+right_edge)
        else :
            left_edge = int(xlim[0])
            right_edge = int(xlim[1])
        if right_edge-left_edge < 10 :
            distances.append(np.nan)
            distances_err.append(np.nan)
            count += 1
            continue
        # dispersion axis analysis
        spectra = thecorrspectra[index]
        popt, pcov = EmissionLineFit(spectra,left_edge,right_edge,guess,bounds)
        x0 = popt[1]
        x0_err = np.sqrt(pcov[1][1]) 
        theta0 = get_theta0(order0_positions[index])
        deltaX = x0 - thex0[index]
        print all_filt[index]
        print 'Position of the H-alpha emission line : %.2f +/- %.2f pixels (%.2f percent) std: %.2f pixels' % (deltaX,x0_err,x0_err/deltaX*100,popt[2])
        Ds = np.linspace(DISTANCE2CCD-D_range,DISTANCE2CCD+D_range,100)
        Ns = []
        diffs = []
        optimal_D = DISTANCE2CCD
        optimal_D_inf = DISTANCE2CCD
        optimal_D_sup = DISTANCE2CCD
        test = 1e20
        test_sup = 1e20
        test_inf = 1e20
        for D in Ds :
            N = get_N(deltaX,order0_positions[index],D=D,l=HALPHA_CENTER,order=order)
            Ns.append( N )
            diff = np.abs(N-N_theo)
            diff_sup = np.abs(N-N_theo+1)
            diff_inf = np.abs(N-N_theo-1)
            diffs.append(diff)
            if diff < test :
                test = diff
                optimal_D = D
            if diff_sup < test_sup :
                test_sup = diff_sup
                optimal_D_sup = D
            if diff_inf < test_inf :
                test_inf = diff_inf
                optimal_D_inf = D
        optimal_D_err  = 0.5*(optimal_D_sup-optimal_D_inf)
        distances.append(optimal_D)
        distances_err.append(optimal_D_err)
        print 'Deduced distance to CCD with %s : %.2f +/- %.2f mm (%.2f percent)' % (all_filt[index],optimal_D,optimal_D_err,100*optimal_D_err/optimal_D)
        # plot Ns vs Ds
        axarr[count,0].plot(Ds,Ns,'b-',lw=2)
        axarr[count,0].plot([np.min(Ds),np.max(Ds)],[N_theo,N_theo],'r-',lw=2)
        axarr[count,0].plot([optimal_D,optimal_D],[np.min(Ns),np.max(Ns)],'r-',lw=2)
        axarr[count,0].plot([np.min(Ds),np.max(Ds)],[N_theo+1,N_theo+1],'k--',lw=2)
        axarr[count,0].plot([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],'k--',lw=2)
        axarr[count,0].plot([optimal_D_inf,optimal_D_inf],[np.min(Ns),np.max(Ns)],'k--',lw=2)
        axarr[count,0].plot([optimal_D_sup,optimal_D_sup],[np.min(Ns),np.max(Ns)],'k--',lw=2)
        axarr[count,0].fill_between([optimal_D_inf,optimal_D_sup],[np.min(Ns),np.min(Ns)],[np.max(Ns),np.max(Ns)],color='red',alpha=0.2)
        axarr[count,0].plot([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],'k--',lw=2)
        axarr[count,0].fill_between([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],[N_theo+1,N_theo+1],color='red',alpha=0.2)
        axarr[count,0].scatter([optimal_D],[N_theo],s=200,color='r')
        axarr[count,0].set_xlim([np.min(Ds),np.max(Ds)])
        axarr[count,0].set_ylim([np.min(Ns),np.max(Ns)])
        axarr[count,0].grid(True)
        axarr[count,0].annotate(all_filt[index],xy=(0.05,0.05),xytext=(0.05,0.05),verticalalignment='bottom', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
        axarr[count,0].set_xlabel('Distance to CCD [mm]',fontsize=16)
        axarr[count,0].set_ylabel('Grooves per mm',fontsize=16)
        # plot diffs vs Ds
        axarr[count,1].plot(Ds,diffs,'b-',lw=2)
        axarr[count,1].plot([optimal_D,optimal_D],[np.min(diffs),np.max(diffs)],'r-',lw=2)
        axarr[count,1].plot([np.min(Ds),np.max(Ds)],[1,1],'k--',lw=2)
        #axarr[count,1].scatter([N_theo],[optimal_D],s=200,color='r')
        axarr[count,1].set_xlim([np.min(Ds),np.max(Ds)])
        axarr[count,1].set_ylim([np.min(diffs),np.max(diffs)])
        axarr[count,1].grid(True)
        axarr[count,1].set_xlabel('Distance to CCD [mm]',fontsize=16)
        axarr[count,1].set_ylabel('Difference to $N_{\mathrm{theo}}$ [grooves per mm]',fontsize=16)
        axarr[count,1].plot([optimal_D_inf,optimal_D_inf],[np.min(diffs),np.max(diffs)],'k--',lw=2)
        axarr[count,1].plot([optimal_D_sup,optimal_D_sup],[np.min(diffs),np.max(diffs)],'k--',lw=2)
        axarr[count,1].fill_between([optimal_D_inf,optimal_D_sup],[np.min(diffs),np.min(diffs)],[np.max(diffs),np.max(diffs)],color='red',alpha=0.2)
        axarr[count,1].annotate(all_filt[index],xy=(0.05,0.05),xytext=(0.05,0.05),verticalalignment='bottom', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
        count += 1
        print '-------------------------------'
    d = []
    d_err = []
    for k in range(len(distances)):
        if not np.isnan(distances[k]):
            d.append(distances[k])
            d_err.append(distances_err[k])
    distances_mean = np.mean(d)
    distances_mean_err = np.sqrt(np.mean(np.array(d_err)**2))
    print 'Average distance to CCD : %.2f +/- %.2f mm (%.2f percent)' % (distances_mean,distances_mean_err,100*distances_mean_err/distances_mean)

    plt.show()
    return(distances_mean,distances_mean_err,distances)


def CalibrateDistance2CCD_TwoOrder(thecorrspectra,all_filt,leftorder_edges=[100,400],rightorder_edges=[1200,1600],guess=[[10,200,100],[10,1400,200]],bounds=(-np.inf,np.inf)):
    NBSPEC=0
    for index in range(len(thecorrspectra)):
        if "Ron400" not in all_filt[index] and "Thor300" not in all_filt[index] :
            continue
        NBSPEC += 1
    f, axarr = plt.subplots(NBSPEC,2,figsize=(25,5*NBSPEC))
    count = 0
    D_range = 1 # in mm
    print 'Present distance to CCD : %.2f mm (to update if necessary)' % DISTANCE2CCD
    print '-------------------------------'    
    distances = []
    distances_err = []
    for index in range(len(thecorrspectra)):
        if "Ron400" not in all_filt[index] and "Thor300" not in all_filt[index] :
            continue
        if "Thor300" in all_filt[index] : N_theo = 300
        if "Ron400" in all_filt[index] : N_theo = 400  
        if type(leftorder_edges[0]) is list :
            xlim_left = leftorder_edges[index]
            guess[0][1] = 0.5*(xlim_left[0]+xlim_left[1])
        else :
            xlim_left = leftorder_edges
        if abs(xlim_left[0]-xlim_left[1]) < 10 :
            distances.append(np.nan)
            distances_err.append(np.nan)
            count += 1
            continue
        if type(rightorder_edges[0]) is list :
            xlim_right = rightorder_edges[index]
            guess[1][1] = 0.5*(xlim_right[0]+xlim_right[1])
        else :
            xlim_right = rightorder_edges
        if abs(xlim_right[0]-xlim_right[1]) < 10 :
            distances.append(np.nan)
            distances_err.append(np.nan)
            count += 1
            continue
        # dispersion axis analysis
        spectra = thecorrspectra[index]
        popt, pcov = EmissionLineFit(spectra,int(xlim_left[0]),int(xlim_left[1]),guess[0],bounds)
        x0_left = popt[1]
        x0_left_err = np.sqrt(pcov[1][1]) 
        popt, pcov = EmissionLineFit(spectra,int(xlim_right[0]),int(xlim_right[1]),guess[1],bounds)
        x0_right = popt[1]
        x0_right_err = np.sqrt(pcov[1][1]) 
        deltaX = 0.5*np.abs(x0_right - x0_left)
        x0_err = 0.5*np.sqrt(x0_right_err**2+x0_left_err**2)
        print all_filt[index]
        print 'Position of the H-alpha emission line : %.2f +/- %.2f pixels (%.2f percent)' % (deltaX,x0_err,x0_err/deltaX*100)
        Ds = np.linspace(DISTANCE2CCD-D_range,DISTANCE2CCD+D_range,100)
        Ns = []
        diffs = []
        optimal_D = DISTANCE2CCD
        optimal_D_inf = DISTANCE2CCD
        optimal_D_sup = DISTANCE2CCD
        test = 1e20
        test_sup = 1e20
        test_inf = 1e20
        for D in Ds :
            theta = np.arctan2(deltaX*PIXEL2MM,D)
            N = np.sin(theta)/HALPHA_CENTER
            Ns.append( N )
            diff = np.abs(N-N_theo)
            diff_sup = np.abs(N-N_theo+1)
            diff_inf = np.abs(N-N_theo-1)
            diffs.append(diff)
            if diff < test :
                test = diff
                optimal_D = D
            if diff_sup < test_sup :
                test_sup = diff_sup
                optimal_D_sup = D
            if diff_inf < test_inf :
                test_inf = diff_inf
                optimal_D_inf = D
        optimal_D_err  = 0.5*(optimal_D_sup-optimal_D_inf)
        distances.append(optimal_D)
        distances_err.append(optimal_D_err)
        print 'Deduced distance to CCD with %s : %.2f +/- %.2f mm (%.2f percent)' % (all_filt[index],optimal_D,optimal_D_err,100*optimal_D_err/optimal_D)
        # plot Ns vs Ds
        axarr[count,0].plot(Ds,Ns,'b-',lw=2)
        axarr[count,0].plot([np.min(Ds),np.max(Ds)],[N_theo,N_theo],'r-',lw=2)
        axarr[count,0].plot([optimal_D,optimal_D],[np.min(Ns),np.max(Ns)],'r-',lw=2)
        axarr[count,0].plot([np.min(Ds),np.max(Ds)],[N_theo+1,N_theo+1],'k--',lw=2)
        axarr[count,0].plot([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],'k--',lw=2)
        axarr[count,0].plot([optimal_D_inf,optimal_D_inf],[np.min(Ns),np.max(Ns)],'k--',lw=2)
        axarr[count,0].plot([optimal_D_sup,optimal_D_sup],[np.min(Ns),np.max(Ns)],'k--',lw=2)
        axarr[count,0].fill_between([optimal_D_inf,optimal_D_sup],[np.min(Ns),np.min(Ns)],[np.max(Ns),np.max(Ns)],color='red',alpha=0.2)
        axarr[count,0].plot([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],'k--',lw=2)
        axarr[count,0].fill_between([np.min(Ds),np.max(Ds)],[N_theo-1,N_theo-1],[N_theo+1,N_theo+1],color='red',alpha=0.2)
        axarr[count,0].scatter([optimal_D],[N_theo],s=200,color='r')
        axarr[count,0].set_xlim([np.min(Ds),np.max(Ds)])
        axarr[count,0].set_ylim([np.min(Ns),np.max(Ns)])
        axarr[count,0].grid(True)
        axarr[count,0].annotate(all_filt[index],xy=(0.05,0.05),xytext=(0.05,0.05),verticalalignment='bottom', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
        axarr[count,0].set_xlabel('Distance to CCD [mm]',fontsize=16)
        axarr[count,0].set_ylabel('Grooves per mm',fontsize=16)
        # plot diffs vs Ds
        axarr[count,1].plot(Ds,diffs,'b-',lw=2)
        axarr[count,1].plot([optimal_D,optimal_D],[np.min(diffs),np.max(diffs)],'r-',lw=2)
        axarr[count,1].plot([np.min(Ds),np.max(Ds)],[1,1],'k--',lw=2)
        #axarr[count,1].scatter([N_theo],[optimal_D],s=200,color='r')
        axarr[count,1].set_xlim([np.min(Ds),np.max(Ds)])
        axarr[count,1].set_ylim([np.min(diffs),np.max(diffs)])
        axarr[count,1].grid(True)
        axarr[count,1].set_xlabel('Distance to CCD [mm]',fontsize=16)
        axarr[count,1].set_ylabel('Difference to $N_{\mathrm{theo}}$ [grooves per mm]',fontsize=16)
        axarr[count,1].plot([optimal_D_inf,optimal_D_inf],[np.min(diffs),np.max(diffs)],'k--',lw=2)
        axarr[count,1].plot([optimal_D_sup,optimal_D_sup],[np.min(diffs),np.max(diffs)],'k--',lw=2)
        axarr[count,1].fill_between([optimal_D_inf,optimal_D_sup],[np.min(diffs),np.min(diffs)],[np.max(diffs),np.max(diffs)],color='red',alpha=0.2)
        axarr[count,1].annotate(all_filt[index],xy=(0.05,0.05),xytext=(0.05,0.05),verticalalignment='bottom', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
        count += 1
        print '-------------------------------'  
    d = []
    d_err = []
    for k in range(len(distances)):
        if not np.isnan(distances[k]):
            d.append(distances[k])
            d_err.append(distances_err[k])
    distances_mean = np.mean(d)
    distances_mean_err = np.sqrt(np.mean(np.array(d_err)**2))
    print 'Average distance to CCD : %.2f +/- %.2f mm (%.2f percent)' % (distances_mean,distances_mean_err,100*distances_mean_err/distances_mean)

    plt.show()
    return(distances_mean,distances_mean_err, distances)


def GratingResolution_OneOrder(thecorrspectra,thex0,order0_positions,all_images,all_filt,xlim=[1200,1600],guess=[10,1400,200],bounds=(-np.inf,np.inf),order=1):
    print 'H-alpha filter center: %.1fnm ' % (HALPHA_CENTER*1e6)
    print 'H-alpha filter width: %.1fnm\n' % (HALPHA_WIDTH*1e6)
    Ns = []
    N_errs = []
    for index in range(len(thecorrspectra)):
        if type(xlim[0]) is list :
            left_edge = int(xlim[index][0])
            right_edge = int(xlim[index][1])
            guess[1] = 0.5*(left_edge+right_edge)
        else :
            left_edge = int(xlim[0])
            right_edge = int(xlim[1])
        if right_edge-left_edge < 10 :
            Ns.append(np.nan)
            N_errs.append(np.nan)
            continue
        print all_filt[index]
        # dispersion axis analysis
        spectra = thecorrspectra[index]
        popt, pcov = EmissionLineFit(spectra,left_edge,right_edge,guess=guess,bounds=bounds)
        x0 = popt[1]
        # compute N
        deltaX = x0 - thex0[index]
        N = get_N(deltaX,order0_positions[index],D=DISTANCE2CCD,l=HALPHA_CENTER,order=order)
        Ns.append(N)
        # compute N uncertainty 
        N_up = get_N(deltaX,order0_positions[index],D=DISTANCE2CCD+DISTANCE2CCD_ERR,l=HALPHA_CENTER,order=order)
        N_low = get_N(deltaX,order0_positions[index],D=DISTANCE2CCD-DISTANCE2CCD_ERR,l=HALPHA_CENTER,order=order)
        N_err = 0.5*np.abs(N_up-N_low)
        N_errs.append(N_err)
        # look at finesse
        fwhm_line = np.abs(popt[2])*2.355
        g = Grating(N,label=all_filt[index],verbose=False)
        res = g.grating_resolution(x0,thex0[index])
        finesse = HALPHA_CENTER/(res*fwhm_line*1e-6-HALPHA_WIDTH)
        # transverse profile analysis
        right_edge = all_images[index].shape[1]
        yprofile=np.copy(all_images[index])[:,min(int(x0),right_edge-1)]
        popt2, pcov2 = EmissionLineFit(yprofile,0,len(yprofile),[np.max(yprofile),0.5*len(yprofile),10])
        fwhm_profile = np.abs(popt2[2])*2.355
        finesse_profile = HALPHA_CENTER*1e6/(res*fwhm_profile)
    
        print 'N=%.1f +/- %.1f lines/mm\t H-alpha FWHM=%.1fpix with res=%.3fnm/pix : FWHM=%.1fnm\t ie finesse=%.1f' % (N,N_err,fwhm_line,res,res*fwhm_line,finesse)
        print 'Transverse profile FWHM=%.1fpix ' % (fwhm_profile)
        print '-------------------------------'
    return(Ns,N_errs)



def GratingResolution_TwoOrder(thecorrspectra,all_images,all_filt,leftorder_edges=[100,400],rightorder_edges=[1200,1600],guess=[[10,200,100],[10,1400,200]],bounds=(-np.inf,np.inf)):
    print 'H-alpha filter center: %.1fnm ' % (HALPHA_CENTER*1e6)
    print 'H-alpha filter width: %.1fnm\n' % (HALPHA_WIDTH*1e6)

    Ns = []
    N_errs = []
    for index in range(len(thecorrspectra)):
        if type(leftorder_edges[0]) is list :
            xlim_left = leftorder_edges[index]
            guess[0][1] = 0.5*(xlim_left[0]+xlim_left[1])
        else :
            xlim_left = leftorder_edges
        if abs(xlim_left[0]-xlim_left[1]) < 10 :
            Ns.append(np.nan)
            N_errs.append(np.nan)
            continue
        if type(rightorder_edges[0]) is list :
            xlim_right = rightorder_edges[index]
            guess[1][1] = 0.5*(xlim_right[0]+xlim_right[1])
        else :
            xlim_right = rightorder_edges
        if abs(xlim_right[0]-xlim_right[1]) < 10 :
            Ns.append(np.nan)
            N_errs.append(np.nan)
            continue
        # dispersion axis analysis
        spectra = thecorrspectra[index]
        popt, pcov = EmissionLineFit(spectra,xlim_left[0],xlim_left[1],guess[0],bounds)
        x0_left = popt[1]
        popt2, pcov = EmissionLineFit(spectra,xlim_right[0],xlim_right[1],guess[1],bounds)
        x0_right = popt2[1]
        deltaX = 0.5*np.abs(x0_left-x0_right)
        # compute N
        theta = np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD)
        N = np.sin(theta)/HALPHA_CENTER
        Ns.append(N)
        # compute N uncertainty 
        theta = np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD+DISTANCE2CCD_ERR)
        N_up = np.sin(theta)/HALPHA_CENTER
        theta = np.arctan2(deltaX*PIXEL2MM,DISTANCE2CCD-DISTANCE2CCD_ERR)
        N_low = np.sin(theta)/HALPHA_CENTER
        N_err = 0.5*np.abs(N_up-N_low)
        N_errs.append(N_err)
        # look at finesse
        g = Grating(N,label=all_filt[index])
        res = g.grating_resolution(deltaX,[IMSIZE/2,IMSIZE/2])
        # right
        fwhm_line_right = np.abs(popt2[2])*2.355
        finesse_right = HALPHA_CENTER/(res*fwhm_line_right*1e-6-HALPHA_WIDTH)
        # left
        fwhm_line_left = np.abs(popt[2])*2.355
        finesse_left = HALPHA_CENTER/(res*fwhm_line_left*1e-6-HALPHA_WIDTH)
        # transverse profile analysis
        # right
        right_edge = all_images[index].shape[1]
        yprofile=np.copy(all_images[index])[:,min(int(x0_right),right_edge-1)]
        popt2, pcov2 = EmissionLineFit(yprofile,0,len(yprofile),[np.max(yprofile),0.5*len(yprofile),10])
        fwhm_profile_right = np.abs(popt2[2])*2.355
        finesse_profile_right = HALPHA_CENTER*1e6/(res*fwhm_profile_right)
        # left
        yprofile=np.copy(all_images[index])[:,max(0,int(x0_left))]
        popt2, pcov2 = EmissionLineFit(yprofile,0,len(yprofile),[np.max(yprofile),0.5*len(yprofile),10])
        fwhm_profile_left = np.abs(popt2[2])*2.355
        finesse_profile_left = HALPHA_CENTER*1e6/(res*fwhm_profile_left)
    
        print all_filt[index]
        print 'N=%.1f +/- %.1f lines/mm' % (N,N_err)
        print 'Right order: H-alpha FWHM=%.1fpix with res=%.3fnm/pix : FWHM=%.1fnm\t ie finesse=%.1f' % (fwhm_line_right,res,res*fwhm_line_right,finesse_right)
        print 'Left  order: H-alpha FWHM=%.1fpix with res=%.3fnm/pix : FWHM=%.1fnm\t ie finesse=%.1f' % (fwhm_line_left,res,res*fwhm_line_left,finesse_left)
        print 'Transverse profile FWHM :  %.1fpix (right)  %.1fpix (left)' % (fwhm_profile_right, fwhm_profile_left)
        print '-------------------------------'
    return(Ns,N_errs)


def extract_spectrum(thecorrspectrum,holo,xlims,thex0,order0_position,order=1):
    left_cut, right_cut = xlims
    spec = thecorrspectrum[left_cut:right_cut]
    pixels = np.arange(left_cut,right_cut,1)-thex0
    lambdas = holo.grating_pixel_to_lambda(pixels,order0_position,order=order)
    return [lambdas,spec]


def CalibrateSpectra(spectra,redshift,thex0,order0_positions,all_titles,object_name,all_filt,xlim=(1000,1800),target=None,order=1,emission_spectrum=False,atmospheric_lines=True,hydrogen_only=False,nofit=False,verbose=False,dir_top_images=None):
    """
    CalibrateSpectra show the right part of spectrum with identified lines
    =====================
    """
    NBSPEC=len(spectra)
    Ds = []
    specs = []
    for index in np.arange(0,NBSPEC):
        if isinstance(xlim[0], (list, tuple, np.ndarray)) :
            left_cut = xlim[index][0]
            right_cut = min(len(spectra[index]),xlim[index][1])
        else :
            left_cut = xlim[0]
            right_cut = min(len(spectra[index]),xlim[1])
        ######## convert pixels to wavelengths #########
        holo = Hologram(all_filt[index],verbose=verbose)
        if verbose : print '---------------------------------------------------'
        lambdas, spec = extract_spectrum(spectra[index],holo,[left_cut,right_cut],thex0[index],order0_positions[index],order=order)
        ###### detect emission/absorption lines and calibrate pixel/lambda #####
        D = DISTANCE2CCD-DISTANCE2CCD_ERR
        shift = 0
        shifts = []
        counts = 0
        #lambda_shift = 1e20
        #while abs(lambda_shift) > 0.1 :
        #    lambda_shift = detect_lines(lambdas,spec,redshift=redshift,emission_spectrum=emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only,ax=None,verbose=False)
        #    lambdas -= lambda_shift
        #    shifts.append(lambda_shift)
        #    counts += 1
        #    if len(shifts)>2 :
        #        if abs(shifts[-1]+shifts[-2]) < 0.1 : break
        #    if counts > 30 : break
        #if verbose : print 'Wavelenght total shift: %.2fnm (after %d steps)' % (np.sum(shifts),len(shifts))
        D_step = DISTANCE2CCD_ERR / 4
        while D < DISTANCE2CCD+4*DISTANCE2CCD_ERR and D > DISTANCE2CCD-4*DISTANCE2CCD_ERR and counts < 30 :
            holo.D = D
            lambdas_test, spec = extract_spectrum(spectra[index],holo,[left_cut,right_cut],thex0[index],order0_positions[index],order=order)
            #lambdas_test = holo.grating_pixel_to_lambda(pixels,order0_positions[index],order=order)
            lambda_shift = detect_lines(lambdas_test,spec,redshift=redshift,emission_spectrum=emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only,ax=None,verbose=False)
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
        Ds.append(D)
        shift = np.mean(lambdas_test - lambdas)
        lambdas = lambdas_test
        if verbose :
            print 'Wavelenght total shift: %.2fnm (after %d steps)' % (shift,len(shifts))
            print '\twith D = %.2f mm (DISTANCE2CCD = %.2f +/- %.2f mm, %.1f sigma shift)' % (D,DISTANCE2CCD,DISTANCE2CCD_ERR,(D-DISTANCE2CCD)/DISTANCE2CCD_ERR)
        specs.append([lambdas,spec])
    PlotCalibratedSpectra(specs,redshift,all_titles,object_name,all_filt,target=target,order=order,emission_spectrum=emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only,nofit=nofit,verbose=verbose,dir_top_images=dir_top_images)
    return specs, Ds

def PlotCalibratedSpectra(specs,redshift,all_titles,object_name,all_filt,target=None,order=1,emission_spectrum=False,atmospheric_lines=True,hydrogen_only=False,nofit=False,verbose=False,dir_top_images=None):
    NBSPEC=len(specs)
    if target is not None :
        target.load_spectra()
    f, axarr = plt.subplots(NBSPEC,1,figsize=(20,7*NBSPEC))
    for index in np.arange(0,NBSPEC):
        lambdas, spec = specs[index]
        axarr[index].set_xlim(lambdas[0],lambdas[-1])
        axarr[index].plot(lambdas,spec,'r-',lw=2,label='Order %d spectrum' % order)
        plot_atomic_lines(axarr[index],redshift=redshift,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only)
        if not nofit :
            lambda_shift = detect_lines(lambdas,spec,redshift=redshift,emission_spectrum=emission_spectrum,atmospheric_lines=atmospheric_lines,hydrogen_only=hydrogen_only,ax=axarr[index],verbose=verbose)
        if verbose : print '-----------------------------------------------------'
        ######## add target spectrum #######
        if target is not None :
            for isp,sp in enumerate(target.spectra):
                if isp==0 or isp==2 : continue
                axarr[index].plot(target.wavelengths[isp],0.3*sp*spec.max()/np.max(sp),label='NED spectrum %d' % isp,lw=2)
        ######## set plot #######
        axarr[index].set_title(all_titles[index])
        axarr[index].annotate(all_filt[index],xy=(0.05,0.8),xytext=(0.05,0.8),verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
        axarr[index].legend(fontsize=16,loc='best')
        axarr[index].set_xlabel('Wavelength [nm]', fontsize=16)
        axarr[index].grid(True)
        axarr[index].set_ylim(spec.min(),spec.max()*1.2)
    if dir_top_images is not None :
        figfilename=os.path.join(dir_top_images,'calibrated_spectrum_profile.pdf')
        plt.savefig(figfilename) 
    plt.show()


def EstimateSecondOrderRatios(specs1,specs2,lambdas,all_titles,object_name,all_filt,bin_width=1,verbose=False,dir_top_images=None):
    """ bins in nm : with of binned ratio """
    NBSPEC=len(specs1)
    f, axarr = plt.subplots(NBSPEC,1,figsize=(20,7*NBSPEC))
    ratios = []
    for index in np.arange(0,NBSPEC):
        ####### compute order1/order2 ratio ########
        spec1_interp = np.interp(lambdas,specs1[index][0],specs1[index][1])
        spec2_interp = np.interp(lambdas,specs2[index][0],specs2[index][1])
        ratio = np.abs(spec2_interp/spec1_interp)
        ###### bin ratio ########
        if bin_width > 1 :
            binned_ratio = []
            binned_ratio_err = []
            binned_lambdas = []
            binned_lambdas_err = []
            i = 0
            while lambdas[0] + (i+1)*bin_width <= lambdas[-1] :
                lambdas_indices = np.where(np.logical_and(lambdas > lambdas[0]+i*bin_width, lambdas < lambdas[0] + (i+1)*bin_width))
                binned_ratio.append(np.median(ratio[lambdas_indices]))
                binned_ratio_err.append(np.std(ratio[lambdas_indices]))
                binned_lambdas.append(0.5*(lambdas[lambdas_indices[0][0]]+lambdas[lambdas_indices[0][-1]]))
                binned_lambdas_err.append(-0.5*(lambdas[lambdas_indices[0][0]]-lambdas[lambdas_indices[0][-1]]))
                i += 1
            ratios.append([2*lambdas,ratio,2*np.array(binned_lambdas),binned_ratio,2*np.array(binned_lambdas_err),binned_ratio_err])
        else : 
            ratios.append([2*lambdas,ratio])
        if verbose : print '---------------------------------------------------'
        ######## set plot #######
        axarr[index].plot(2*lambdas,ratio,'r-',lw=2) # plot with respect to order 1 wavelength
        if bin_width > 1 :
            axarr[index].errorbar(2*np.array(binned_lambdas),binned_ratio,xerr=2*np.array(binned_lambdas_err),yerr=binned_ratio_err,lw=2,elinewidth=3,fmt='ko',markersize=10)
        axarr[index].set_title(all_titles[index])
        axarr[index].annotate(all_filt[index],xy=(0.05,0.8),xytext=(0.05,0.8),verticalalignment='top', horizontalalignment='left',color='blue',fontweight='bold', fontsize=20, xycoords='axes fraction')
        #axarr[index].legend(fontsize=16,loc='best')
        axarr[index].set_xlabel('Order 1 wavelength [nm]', fontsize=16)
        axarr[index].grid(True)
        axarr[index].set_xlim(2*lambdas[0],2*lambdas[-1])
        axarr[index].set_ylim(0,1.2*np.max(ratio))
    if dir_top_images is not None :
        figfilename=os.path.join(dir_top_images,'second_order_contamination.pdf')
        plt.savefig(figfilename) 
    plt.show()
    return ratios



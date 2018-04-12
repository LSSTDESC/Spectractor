import numpy as np
import os, sys
from scipy import ndimage
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from tools import *
from parameters import *

# Making of the holograms
DISTANCE2CCD = 55.45 # distance between hologram and CCD in mm
DISTANCE2CCD_ERR = 0.19 # uncertainty on distance between hologram and CCD in mm
LAMBDA_CONSTRUCTOR = 639e-6 # constructor wavelength to make holograms in mm
GROOVES_PER_MM = 350 # approximate effective number of lines per millimeter of the hologram
PLATE_CENTER_SHIFT_X = -6. # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y = -8. # plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_X_ERR = 2. # estimate uncertainty on plate center shift on x in mm in filter frame
PLATE_CENTER_SHIFT_Y_ERR = 2. # estimate uncertainty on plate center shift on x in mm in filter frame


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

def get_N(deltaX,x0,D=DISTANCE2CCD,l=656,order=1):
    """ Return grooves per mm given the signal x position with 
    its wavelength in mm, the distance to CCD in mm and the order number.
    x0 is the order 0 position in the full raw image.
    deltaX is the distance in pixels between order 0 and signal point 
    in the rotated image."""
    theta = get_refraction_angle(deltaX,x0,D=D)
    theta0 = get_theta0(x0)
    N = (np.sin(theta)-np.sin(theta0))/(order*l)
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
    def __init__(self,N,label="",D=DISTANCE2CCD,data_dir=HOLO_DIR,verbose=False):
        self.N_input = N
        self.N_err = 1
        self.D = D
        self.label = label
        self.data_dir = data_dir
        self.load_files(verbose=verbose)
        self.transmission = lambda wavelength: 1.0

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

    def plot_transmission(self,xlim=None):
        wavelengths = np.linspace(parameters.LAMBDA_MIN,parameters.LAMBDA_MAX,100)
        plt.figure()
        plt.plot(wavelengths,self.transmission(wavelengths),'b-',label=self.label)
        plt.xlabel("$\lambda$ [nm]")
        plt.ylabel("Transmission")
        plt.legend()
        plt.grid()
        plt.show()

        
class Hologram(Grating):

    def __init__(self,label,D=DISTANCE2CCD,lambda_plot=256000,data_dir=HOLO_DIR,verbose=True):
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
        self.is_hologram = True
        self.load_specs(verbose=verbose)

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



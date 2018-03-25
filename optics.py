import numpy as np
import copy


from parameters import *

class Line():

    def __init__(self,wavelength,label,atmospheric=False,emission=False,label_pos=[0.007,0.02],width_bounds=[1,10]):
        self.my_logger = set_logger(self.__class__.__name__)
        self.wavelength = wavelength # in nm
        self.label = label
        self.label_pos = label_pos
        self.atmospheric = atmospheric
        self.emission = emission
        if self.atmospheric: self.emission = False
        self.width_bounds = width_bounds


class Lines():

    def __init__(self,redshift=0,atmospheric_lines=True,hydrogen_only=False):
        # Main emission/absorption lines in nm
        HALPHA = Line(656.3,atmospheric=False,label='$H\\alpha$',label_pos=[-0.016,0.02])
        HBETA = Line( 486.3,atmospheric=False,label='$H\\beta$',label_pos=[0.007,0.02]) 
        HGAMMA = Line(434.0,atmospheric=False,label='$H\\gamma$',label_pos=[0.007,0.02]) 
        HDELTA = Line( 410.2,atmospheric=False,label='$H\\delta$',label_pos=[0.007,0.02])
        OIII = Line( 500.7,atmospheric=False,label='$O_{III}$',label_pos=[0.007,0.02])
        CII1 =  Line( 723.5,atmospheric=False,label='$C_{II}$',label_pos=[0.005,0.92])
        CII2 =  Line( 711.0,atmospheric=False,label='$C_{II}$',label_pos=[0.005,0.02])
        CIV =  Line( 706.0,atmospheric=False,label='$C_{IV}$',label_pos=[-0.016,0.92])
        CII3 =  Line( 679.0,atmospheric=False,label='$C_{II}$',label_pos=[0.005,0.02])
        CIII1 =  Line( 673.0,atmospheric=False,label='$C_{III}$',label_pos=[-0.016,0.92])
        CIII2 =  Line( 570.0,atmospheric=False,label='$C_{III}$',label_pos=[0.007,0.02])
        HEI =  Line( 587.5,atmospheric=False,label='$He_{I}$',label_pos=[0.007,0.02])
        HEII =  Line( 468.6,atmospheric=False,label='$He_{II}$',label_pos=[0.007,0.02])
        CAII1 =  Line( 393.366,atmospheric=True,label='$Ca_{II}$',label_pos=[0.007,0.02]) # https://en.wikipedia.org/wiki/Fraunhofer_lines
        CAII2 =  Line( 396.847,atmospheric=True,label='$Ca_{II}$',label_pos=[0.007,0.02]) # https://en.wikipedia.org/wiki/Fraunhofer_lines
        O2 = Line( 762.1,atmospheric=True,label='$O_2$',label_pos=[0.007,0.02]) # http://onlinelibrary.wiley.com/doi/10.1029/98JD02799/pdf
        #O2_1 = Line( 760.6,atmospheric=True,label='',label_pos=[0.007,0.02]) # libradtran paper fig.3
        #O2_2 = Line( 763.2,atmospheric=True,label='$O_2$',label_pos=[0.007,0.02])  # libradtran paper fig.3
        O2B = Line( 686.719,atmospheric=True,label='$O_2(B)$',label_pos=[0.007,0.02]) # https://en.wikipedia.org/wiki/Fraunhofer_lines
        O2Y = Line( 898.765,atmospheric=True,label='$O_2(Y)$',label_pos=[0.007,0.02]) # https://en.wikipedia.org/wiki/Fraunhofer_lines
        O2Z = Line( 822.696,atmospheric=True,label='$O_2(Z)$',label_pos=[0.007,0.02]) # https://en.wikipedia.org/wiki/Fraunhofer_lines
        #H2O = Line( 960,atmospheric=True,label='$H_2 O$',label_pos=[0.007,0.02])  # 
        H2O_1 = Line( 950,atmospheric=True,label='$H_2 O$',label_pos=[0.007,0.02])  # libradtran paper fig.3
        H2O_2 = Line( 970,atmospheric=True,label='$H_2 O$',label_pos=[0.007,0.02])  # libradtran paper fig.3
        
        self.lines = [HALPHA,HBETA,HGAMMA,HDELTA,O2,O2B,O2Y,O2Z,H2O_1,H2O_2,OIII,CII1,CII2,CIV,CII3,CIII1,CIII2,HEI,HEII,CAII1,CAII2]
        self.redshift = redshift
        self.atmospheric_lines = atmospheric_lines
        self.hydrogen_only = hydrogen_only
        self.lines = self.sort_lines()

    def sort_lines(self):
        sorted_lines = []
        for l in self.lines:
            if self.hydrogen_only :
                if not self.atmospheric_lines :
                    if l.atmospheric : continue
                    if '$H\\' not in l.label : continue
                else :
                    if not l.atmospheric and '$H\\' not in l.label : continue
            else :
                if not self.atmospheric_lines and l.atmospheric : continue
            sorted_lines.append(l)        
        if self.redshift > 0 :
            for line in sorted_lines:
                if not line.atmospheric : line.wavelength *= (1+self.redshift)
        sorted_lines = sorted(sorted_lines, key=lambda x: x.wavelength)
        return sorted_lines

    
    def plot_atomic_lines(self,ax,color_atomic='g',color_atmospheric='b',fontsize=12):
        xlim = ax.get_xlim()
        for l in self.lines:
            color = color_atomic
            if l.atmospheric: color = color_atmospheric
            ax.axvline(l.wavelength,lw=2,color=color)
            xpos = (l.wavelength-xlim[0])/(xlim[1]-xlim[0])+l.label_pos[0]
            if xpos > 0 and xpos < 1 :
                ax.annotate(l.label,xy=(xpos,l.label_pos[1]),rotation=90,ha='left',va='bottom',xycoords='axes fraction',color=color,fontsize=fontsize)


class Filter():

    def __init__(self,wavelength_min,wavelength_max,label):
        self.min = wavelength_min
        self.max = wavelength_max
        self.label = label


import numpy as np
import copy

from parameters import *
from tools import *
from scipy.signal import argrelextrema
from astropy.table import Table

# H-alpha filter
HALPHA_CENTER = 655.9e-6 # center of the filter in mm
HALPHA_WIDTH = 6.4e-6 # width of the filter in mm

# Other filters
FGB37 = {'label':'FGB37','min':300,'max':800}
RG715 = {'label':'RG715','min':690,'max':1100}
HALPHA_FILTER = {'label':'Halfa','min':HALPHA_CENTER-2*HALPHA_WIDTH,'max':HALPHA_CENTER+2*HALPHA_WIDTH}
ZGUNN = {'label':'Z-Gunn','min':800,'max':1100}
FILTERS = [RG715,FGB37,HALPHA_FILTER,ZGUNN]

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
        self.high_snr = False


class Lines():

    def __init__(self,redshift=0,atmospheric_lines=True,hydrogen_only=False,emission_spectrum=False):
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
        #H2O = Line( 960,atmospheric=True,label='$H_2 O$',label_pos=[0.007,0.02],width_bounds=(1,100))  # 
        H2O_1 = Line( 950,atmospheric=True,label='$H_2 O$',label_pos=[0.007,0.02],width_bounds=(5,30))  # libradtran paper fig.3, broad line
        H2O_2 = Line( 970,atmospheric=True,label='$H_2 O$',label_pos=[0.007,0.02],width_bounds=(5,30))  # libradtran paper fig.3, broad line
        
        self.lines = [HALPHA,HBETA,HGAMMA,HDELTA,O2,O2B,O2Y,O2Z,H2O_1,H2O_2,OIII,CII1,CII2,CIV,CII3,CIII1,CIII2,HEI,HEII,CAII1,CAII2]
        self.redshift = redshift
        self.atmospheric_lines = atmospheric_lines
        self.hydrogen_only = hydrogen_only
        self.emission_spectrum = emission_spectrum
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
            if not l.high_snr : continue
            color = color_atomic
            if l.atmospheric: color = color_atmospheric
            ax.axvline(l.wavelength,lw=2,color=color)
            xpos = (l.wavelength-xlim[0])/(xlim[1]-xlim[0])+l.label_pos[0]
            if xpos > 0 and xpos < 1 :
                ax.annotate(l.label,xy=(xpos,l.label_pos[1]),rotation=90,ha='left',va='bottom',xycoords='axes fraction',color=color,fontsize=fontsize)


    def detect_lines(self,lambdas,spec,spec_err=None,snr_minlevel=3,ax=None,verbose=False):
        # main settings
        bgd_npar = BGD_NPARAMS
        peak_look = 7 # half range to look for local maximum in pixels
        bgd_width = 7 # size of the peak sides to use to fit spectrum base line
        if self.hydrogen_only :
            peak_look = 15
            bgd_width = 15
        baseline_prior = 3 # *sigma gaussian prior on base line fit
        # initialisation
        lambda_shifts = []
        snrs = []
        index_list = []
        guess_list = []
        bounds_list = []
        lines_list = []
        for line in self.lines:
            # wavelength of the line: find the nearest pixel index
            l = line.wavelength
            l_index, l_lambdas = find_nearest(lambdas,l)
            # reject if pixel index is too close to image bounds
            if l_index < peak_look or l_index > len(lambdas)-peak_look : continue
            # search for local extrema to detect emission or absorption line
            # around pixel index +/- peak_look
            line_strategy = np.greater  # look for emission line
            bgd_strategy = np.less
            if not self.emission_spectrum or line.atmospheric:
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
            # search for first local minima around the local maximum
            # or for first local maxima around the local minimum
            # around +/- 3*peak_look
            index_inf = peak_index - 1 # extrema on the left
            while index_inf > max(0,peak_index - 3*peak_look) :
                test_index = range(index_inf,peak_index)
                minm = argrelextrema(spec[test_index], bgd_strategy)
                if len(minm[0]) > 0 :
                    index_inf = index_inf + minm[0][0] 
                    break
                else :
                    index_inf -= 1
            index_sup = peak_index + 1  # extrema on the right
            while index_sup < min(len(spec)-1,peak_index + 3*peak_look) :
                test_index = range(peak_index,index_sup)
                minm = argrelextrema(spec[test_index], bgd_strategy)
                if len(minm[0]) > 0 :
                    index_sup = peak_index + minm[0][0] 
                    break
                else :
                    index_sup += 1
            # pixel range to consider around the peak, adding bgd_width pixels
            # to fit for background around the peak
            index = range(max(0,index_inf-bgd_width),min(len(lambdas),index_sup+bgd_width))
            # first guess and bounds to fit the line properties and
            # the background with BGD_ORDER order polynom
            guess = [0]*bgd_npar+[0*abs(spec[peak_index]),lambdas[peak_index],0.5*(line.width_bounds[0]+line.width_bounds[1])]
            bounds = [[-np.inf]*bgd_npar+[-np.inf,lambdas[index_inf],line.width_bounds[0]], [np.inf]*bgd_npar+[2*np.max(spec[index]),lambdas[index_sup],line.width_bounds[1]]  ]
            # gaussian amplitude bounds depend if line is emission/absorption
            if line_strategy == np.less :
                bounds[1][bgd_npar] = 0 # look for abosrption under bgd
            else :
                bounds[0][bgd_npar] = 0 # look for emission above bgd
            index_list.append(index)
            lines_list.append(line)
            guess_list.append(guess)
            bounds_list.append(bounds)
        # now gather lines together if pixel index ranges overlap
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
            # set central peak bounds exactly between two close lines
            for k in range(len(merge)-1) :
                new_bounds_list[-1][0][bgd_npar+3*(k+1)+1]  = 0.5*(new_guess_list[-1][bgd_npar+3*k+1]+new_guess_list[-1][bgd_npar+3*(k+1)+1])
                new_bounds_list[-1][1][bgd_npar+3*k+1] = 0.5*(new_guess_list[-1][bgd_npar+3*k+1]+new_guess_list[-1][bgd_npar+3*(k+1)+1])
            # sort pixel indices and remove doublons
            new_index_list[-1] = sorted(list(set(new_index_list[-1])))
        # fit the line subsets and background
        rows = []
        for k in range(len(new_index_list)):
            # first guess for the base line with the lateral bands
            index = new_index_list[k]
            guess = new_guess_list[k]
            bounds = new_bounds_list[k]
            bgd_index = index[:bgd_width]+index[-bgd_width:]
            sigma = None
            if spec_err is not None: sigma = spec_err[bgd_index]
            line_popt, line_pcov = fit_bgd(lambdas[bgd_index],spec[bgd_index],sigma=sigma)
            for n in range(bgd_npar):
                guess[n] = line_popt[n]
                bounds[0][n] = line_popt[n]-baseline_prior*np.sqrt(line_pcov[n][n])
                bounds[1][n] = line_popt[n]+baseline_prior*np.sqrt(line_pcov[n][n])
            # fit local extrema with a multigaussian + BGD_ORDER polynom
            # account for the spectrum uncertainties if provided
            sigma = None
            if spec_err is not None: sigma = spec_err[index]
            popt, pcov = fit_multigauss_and_bgd(lambdas[index],spec[index],guess=guess, bounds=bounds,sigma=sigma)
            # compute the base line subtracting the gaussians
            base_line = spec[index]
            for j in range(len(new_lines_list[k])) :
                base_line -= gauss(lambdas[index],*popt[bgd_npar+3*j:bgd_npar+3*j+3])
            # noise level defined as the std of the residuals if no error
            noise_level = np.std(spec[index]-multigauss_and_bgd(lambdas[index],*popt))
            # otherwise mean of error bars of bgd lateral bands
            if spec_err is not None:
                noise_level = np.mean(spec_err[bgd_index])
            plot_line_subset = False
            for j in range(len(new_lines_list[k])) :
                line = new_lines_list[k][j]
                l = line.wavelength
                peak_pos = popt[bgd_npar+3*j+1]
                # SNR computation
                signal_level = popt[bgd_npar+3*j]
                snr = np.abs(signal_level / noise_level)
                if snr < snr_minlevel : continue
                # FWHM
                FWHM = np.abs(popt[bgd_npar+3*j+bgd_npar])*2.355
                rows.append((line.label,l,peak_pos,peak_pos-l,FWHM,signal_level,snr))
                # save fit results
                plot_line_subset = True
                line.high_snr = True
                line.fit_lambdas = lambdas[index]
                line.fit_gauss = gauss(lambdas[index],*popt[bgd_npar+3*j:bgd_npar+3*j+3])
                line.fit_bgd = base_line
                line.fit_snr = snr
                line.fit_fwhm = FWHM
                # wavelength shift between tabulate and observed lines
                lambda_shifts.append(peak_pos-l)
                snrs.append(snr)
            if ax is not None and plot_line_subset:
                ax.plot(lambdas[index],multigauss_and_bgd(lambdas[index],*popt),lw=2,color='b')
                ax.plot(lambdas[index],np.polyval(popt[:bgd_npar],lambdas[index]),lw=2,color='b',linestyle='--')
        if len(rows) > 0 :
            t = Table(rows=rows,names=('Line','Tabulated','Detected','Shift','FWHM','Amplitude','SNR'),dtype=('a10','f4','f4','f4','f4','f4','f4'))
            for col in t.colnames[1:-2] : t[col].unit = 'nm'
            if verbose : print t
            shift =  np.average(lambda_shifts,weights=np.array(snrs)**2)
            # remove lines with low SNR from plot
            for l in self.lines:
                if l.label not in t['Line'][:]: l.high_snr = False
        else :
            shift = 0
        return shift


                
class Filter():

    def __init__(self,wavelength_min,wavelength_max,label):
        self.min = wavelength_min
        self.max = wavelength_max
        self.label = label


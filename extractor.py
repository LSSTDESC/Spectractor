import sys, os
sys.path.append('../SpectractorSim')

from spectractorsim import *
from spectractor import *
import parameters

import emcee as mc
from scipy.optimize import minimize, least_squares
from scipy.signal import fftconvolve, gaussian


class Extractor():

    def __init__(self,filename):
        self.pwv = 5
        self.ozone = 300
        self.aerosols = 0.05
        self.A1 = 1.0
        self.A2 = 0.2
        self.reso = 3
        self.p = [self.A1, self.A2, self.pwv, self.ozone, self.aerosols, self.reso ]
        self.bounds = ((0,0,0,0,0,1), (np.inf,1.0,10,np.inf,1.0,10))

        self.spectrum, self.telescope, self.disperser, self.target = SpectractorInit(filename)
        self.airmass = self.spectrum.header['AIRMASS']
        self.pressure = self.spectrum.header['OUTPRESS']
        self.temperature = self.spectrum.header['OUTTEMP']
        self.atmosphere = Atmosphere(self.airmass,self.pressure,self.temperature)


    def simulation(self,lambdas,A1,A2,pwv,ozone,aerosols,reso):
        #print 'Parameters:',A1,A2,pwv,ozone,aerosols,reso
        title = 'Parameters: A1=%.3f, A2=%.3f, PWV=%.3f, OZ=%.3g, VAOD=%.3f, reso=%.2f' % (A1,A2,pwv,ozone,aerosols,reso)
        print title
        self.atmosphere.simulate(pwv, ozone, aerosols)    
        simulation = SpectrumSimulation(self.spectrum,self.atmosphere,self.telescope,self.disperser)
        simulation.simulate(lambdas)    
        kernel = gaussian(lambdas.size,reso)
        kernel /= np.sum(kernel)
        tmp = A1*np.copy(simulation.data)
        sim_conv = fftconvolve(simulation.data, kernel, mode='same')
        sim_conv = interp1d(lambdas,sim_conv,kind="linear",bounds_error=False,fill_value=(0,0))
        simulation = lambda x: A1*sim_conv(x) + A1*A2*sim_conv(x/2)
        if True:
            fig, ax = plt.subplots(1,1)
            #plt.errorbar(lambdas,self.spectrum.data,yerr=self.spectrum.err,label='data')
            self.spectrum.plot_spectrum_simple(ax)
            ax.plot(lambdas,tmp,label='before conv')
            ax.plot(lambdas,simulation(lambdas),label='model')
            ax.legend()
            ax.set_title(title,fontsize=10)
            plt.draw()
            plt.pause(5)
            plt.close()
        return simulation(lambdas)

    def chisq(self,p):
        simulation = self.simulation(self.spectrum.lambdas,*p)
        chisq = np.sum(((simulation - self.spectrum.data)/self.spectrum.err)**2)
        print '\tChisq=',chisq
        return chisq

    def minimizer(self):
        self.p[0] *= np.max(self.spectrum.data)/np.max(self.simulation(self.spectrum.lambdas,*self.p))
        #self.popt, self.pcov = minimize(self.simulation,self.spectrum.lambdas,self.spectrum.data,sigma=self.spectrum.err,p0=self.p,bounds=self.bounds)
        #print self.popt
        #print self.pcov
        res = least_squares(self.chisq,x0=self.p,bounds=self.bounds,diff_step=(0.1,0.1,0.2,0.5,0.2,0.5),verbose=0)
        self.popt = res.x
        print res
        print res.x
        # reduc_061.fits
        # popt = [  3.40487137e-01,   3.51815612e-21,   6.85293068e+01,   2.64999888e+02, 6.87424403e-18,   2.57321071e+01]
        # pcov = [[  7.27035555e-04,  -4.06484360e-05,  -2.98707685e-19,   2.28750956e-19 ,   1.60926623e-03 , -1.03127993e-03] ,[ -4.06484360e-05 ,  1.03716244e-04 ,  5.47966342e-20,  -2.10871168e-18,   -6.33161029e-05,   9.49676729e-03], [ -2.98707685e-19,   5.47966342e-20,   3.73350474e-34,  -1.29187966e-32,   -6.24999401e-19 ,  5.81815069e-17] ,[  2.28750956e-19,  -2.10871168e-18,  -1.29187966e-32 ,  6.62477509e-31,   -2.15274280e-18,  -2.98353193e-15], [  1.60926623e-03 , -6.33161029e-05,  -6.24999401e-19,  -2.15274280e-18  ,  3.67355523e-03 ,  9.69261415e-03], [ -1.03127993e-03 ,  9.49676729e-03 ,  5.81815069e-17 , -2.98353193e-15  ,  9.69261415e-03,   1.34366264e+01]]
        # reduc_060.fits
        # popt = [  1.67477902e+00,   2.52140635e-19 ,  7.86923985e+01  , 3.08111559e+02,   1.71430380e-18 ,  3.10480892e+01]
        # pcov = [[  5.93723419e-03 , -4.54034315e-04 , -2.83356734e-18,  -2.43329541e-18,    2.04236587e-03,  -2.18969937e-02], [ -4.54034315e-04 ,  1.41440222e-04 ,  5.06837982e-19 ,  6.64690985e-19,   -1.42903896e-04,   5.98540014e-03], [ -2.83356734e-18,   5.06837982e-19,   2.52819423e-32,   4.61449136e-32,   -7.62579706e-19  , 4.15628906e-16], [ -2.43329541e-18,   6.64690985e-19  , 4.61449136e-32 ,  8.56021586e-32,   -4.43632344e-19 ,  7.71031445e-16], [  2.04236587e-03  -1.42903896e-04,  -7.62579706e-19 , -4.43632344e-19,    7.38426569e-04  ,-3.98891340e-03], [ -2.18969937e-02 ,  5.98540014e-03 ,  4.15628906e-16 ,  7.71031445e-16,   -3.98891340e-03 ,  6.94479562e+00]]

    def plot_bestfit(self):
        fig, ax = plt.subplots(1,1)
        self.spectrum.plot_spectrum_simple(ax)
        ax.plot(self.spectrum.lambdas,self.simulation(*self.popt),label='Best fit')
        plt.legend()
        plt.show()
        
        


parameters.VERBOSE = False
filename = 'output/data_28may17/reduc_20170528_062_spectrum.fits'
m = Extractor(filename)
m.minimizer()
m.plot_bestfit()

#p0 = [0.1,0.1,5,300,0.01,150]
#constraints = ({'type':'eq', 'fun':lambda p: p[1]=300},
#               {'type':'eq', 'fun':lambda p: p[1]=300})
#bounds = ((0,None), (0,1), (0,20), (200,400), (0,0.2),  (0,500))
#res = minimize(m.chisq,p0,method='Powell',bounds=bounds,
#               options={'gtol': 1e-2, 'disp': True, 'xtol': 0.1})
#popt, pcov = curve_fit(func, xdata, ydata)

import sys, os
sys.path.append('../SpectractorSim')

from spectractorsim import *
from spectractor import *
import parameters

import emcee 
import corner
from scipy.optimize import minimize, least_squares
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from multiprocessing import Pool

class Extractor():

    def __init__(self,filename,atmgrid_filename="",live_fit=False):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.live_fit = live_fit
        self.A1 = 1.0
        self.A2 = 0.1
        self.ozone = 300.
        self.pwv = 3
        self.aerosols = 0.03
        self.reso = 3.
        self.shift = 1.
        self.dilatation = 1.
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols, self.reso, self.shift ])
        self.labels = ["$A_1$", "$A_2$", "ozone", "PWV", "VAOD", "reso", "$\lambda_{\\mathrm{shift}}$" ]
        self.bounds = ((0,0,0,0,0,1,-20), (np.inf,1.0,np.inf,10,1.0,10,20))
        self.title = ""
        self.spectrum, self.telescope, self.disperser, self.target = SpectractorSimInit(filename)
        self.airmass = self.spectrum.header['AIRMASS']
        self.pressure = self.spectrum.header['OUTPRESS']
        self.temperature = self.spectrum.header['OUTTEMP']
        self.use_grid = False
        if atmgrid_filename == "":
            self.atmosphere = Atmosphere(self.airmass,self.pressure,self.temperature)
        else:
            self.use_grid = True
            self.atmosphere = AtmosphereGrid(filename,atmgrid_filename)
            if parameters.VERBOSE:
                self.my_logger.info('\n\tUse atmospheric grid models from file %s. ' % atmgrid_filename)
        self.p[0] *= np.max(self.spectrum.data)/np.max(self.simulation(self.spectrum.lambdas,*self.p))
        self.get_truth()
        if 0. in self.spectrum.err :
            self.spectrum.err = np.ones_like(self.spectrum.err)
        if parameters.DEBUG:
            fig = plt.figure()
            for i in range(10):
                a = self.atmosphere.interpolate(300,i,0.05)
                plt.plot(self.atmosphere.lambdas,a,label='pwv=%dmm' % i)
            plt.grid()
            plt.xlabel('$\lambda$ [nm]')
            plt.ylabel('Atmospheric transmission')
            plt.legend(loc='best')
            plt.show()

    def get_truth(self):
        if 'A1' in self.spectrum.header.keys():
            A1_truth = self.spectrum.header['A1']
            if 'A2' in self.spectrum.header.keys():
                A2_truth = self.spectrum.header['A2']
            if 'OZONE' in self.spectrum.header.keys():
                ozone_truth = self.spectrum.header['OZONE']
            if 'PWV' in self.spectrum.header.keys():
                pwv_truth = self.spectrum.header['PWV']
            if 'VAOD' in self.spectrum.header.keys():
                aerosols_truth = self.spectrum.header['VAOD']
            if 'RESO' in self.spectrum.header.keys():
                reso_truth = self.spectrum.header['RESO']
            self.truth = (A1_truth, A2_truth, ozone_truth, pwv_truth, aerosols_truth, reso_truth, None, None)
        else:
            self.truth = None


    def simulation(self,lambdas,A1,A2,ozone,pwv,aerosols,reso,shift=0.):
        self.title = 'Parameters: A1=%.3f, A2=%.3f, PWV=%.3f, OZ=%.3g, VAOD=%.3f, reso=%.2f, shift=%.2f' % (A1,A2,pwv,ozone,aerosols,reso,shift)
        #print self.title
        self.atmosphere.simulate(ozone, pwv, aerosols)
        simulation = SpectrumSimulation(self.spectrum,self.atmosphere,self.telescope,self.disperser)
        simulation.simulate(lambdas-shift)    
        self.model_noconv = A1*np.copy(simulation.data)
        sim_conv = fftconvolve_gaussian(simulation.data,reso)
        sim_conv = interp1d(lambdas,sim_conv,kind="linear",bounds_error=False,fill_value=(0,0))
        self.lambdas = lambdas
        self.model = lambda x: A1*sim_conv(x) + A1*A2*sim_conv(x/2)
        if self.live_fit: self.plot_fit()
        return self.model(lambdas)

    def chisq(self,p):
        model = self.simulation(self.spectrum.lambdas,*p)
        chisq = np.sum(((model - self.spectrum.data)/self.spectrum.err)**2)
        #chisq /= self.spectrum.data.size
        #print '\tReduced chisq =',chisq/self.spectrum.data.size
        return chisq

    def minimizer(self):
        res = least_squares(self.chisq,x0=self.p,bounds=self.bounds,xtol=1e-6,ftol=1e-5,method='trf',verbose=0,x_scale=(1,1000,10000,10000,10,1000,1000,1000),loss='soft_l1')
        # diff_step=(0.1,0.1,0.5,1,0.5,0.2),
        self.popt = res.x
        print res
        print res.x

    def plot_fit(self):
        fig = plt.figure(figsize=(12,6))
        ax1 = plt.subplot(222)
        ax2 = plt.subplot(224)
        ax3 = plt.subplot(121)
        # main plot
        self.spectrum.plot_spectrum_simple(ax3)
        ax3.plot(self.lambdas,self.model_noconv,label='before conv')
        ax3.plot(self.lambdas,self.model(self.lambdas),label='model')
        ax3.set_title(self.title,fontsize=10)
        ax3.legend()
        # zoom O2
        sub = np.where((self.lambdas>730) & (self.lambdas<800))
        self.spectrum.plot_spectrum_simple(ax2)
        ax2.plot(self.lambdas[sub],self.model_noconv[sub],label='before conv')
        ax2.plot(self.lambdas[sub],self.model(self.lambdas[sub]),label='model')
        ax2.set_xlim((self.lambdas[sub][0],self.lambdas[sub][-1]))
        ax2.set_ylim((0.9*np.min(self.spectrum.data[sub]),1.1*np.max(self.spectrum.data[sub])))
        ax2.set_title('Zoom $O_2$',fontsize=10)
        # zoom H2O
        sub = np.where((self.lambdas>870) & (self.lambdas<1000))
        self.spectrum.plot_spectrum_simple(ax1)
        ax1.plot(self.lambdas[sub],self.model_noconv[sub],label='before conv')
        ax1.plot(self.lambdas[sub],self.model(self.lambdas[sub]),label='model')
        ax1.set_xlim((self.lambdas[sub][0],self.lambdas[sub][-1]))
        ax1.set_ylim((0.9*np.min(self.spectrum.data[sub]),1.1*np.max(self.spectrum.data[sub])))
        ax1.set_title('Zoom $H_2 O$',fontsize=10)
        fig.tight_layout()
        if self.live_fit:
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            plt.show()

class Extractor_MCMC(Extractor):

    def __init__(self,filename,atmgrid_filename="",live_fit=False):
        Extractor.__init__(self,filename,atmgrid_filename=atmgrid_filename,live_fit=live_fit)
        self.ndim = len(self.p)
        self.nwalkers = 4*self.ndim

    def lnprior(self,p):
        in_bounds = True
        for npar,par in enumerate(p):
            if par < self.bounds[0][npar] or par > self.bounds[1][npar]:
                in_bounds = False
                break
        if in_bounds:
            return 0.0
        else:
            return -1e20

    def lnlike(self,p):
        return -0.5*self.chisq(p) 

    def lnprob(self,p):
        lp = self.lnprior(p)
        if not np.isfinite(lp):
            return -1e20
        return lp + self.lnlike(p)
            
    def mcmc(self):
        pos = np.array([self.p  + 0.1*self.p*np.random.randn(self.ndim) for i in range(self.nwalkers)])
        # Set up the backend
        # Don't forget to clear it in case the file already exists
        filename = "tutorial.h5"
        #backend = emcee.backends.HDFBackend(filename)
        #backend.reset(self.nwalkers, self.ndim)

        #with Pool() as pool:
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, args=())
        nsamples = 4000
        self.sampler.run_mcmc(pos, nsamples)
        #tau = sampler.get_autocorr_time()
        burnin = nsamples / 2
        thin = nsamples / 4
        #self.samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        self.samples = self.sampler.chain[:, burnin:, :].reshape((-1, self.ndim))
        #log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
        #log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

        print("burn-in: {0}".format(burnin))
        print("thin: {0}".format(thin))
        #print("flat chain shape: {0}".format(samples.shape))
        #print("flat log prob shape: {0}".format(log_prob_samples.shape))
        #print("flat log prior shape: {0}".format(log_prior_samples.shape))

        fig = corner.corner(self.samples, labels=self.labels, truths=self.truth, quantiles=[0.16, 0.5, 0.84], show_titles=True)
        plt.show()
        fig.savefig("triangle.png")

    def load_walkers(self):
        reader = emcee.backends.HDFBackend(filename)
        tau = reader.get_autocorr_time()
        burnin = int(2*np.max(tau))
        thin = int(0.5*np.min(tau))
        samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
        log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
        log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

        print("burn-in: {0}".format(burnin))
        print("thin: {0}".format(thin))
        print("flat chain shape: {0}".format(samples.shape))
        print("flat log prob shape: {0}".format(log_prob_samples.shape))
        print("flat log prior shape: {0}".format(log_prior_samples.shape))



    def test_convergence(self):
        fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(self.samples[:, i, :], "k", alpha=0.3)
            ax.set_xlim(0, len(self.samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.show()
        
if __name__ == "__main__":
    import commands, string, re, time, os
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--debug", dest="debug",action="store_true",
                      help="Enter debug mode (more verbose and plots).",default=False)
    parser.add_option("-v", "--verbose", dest="verbose",action="store_true",
                      help="Enter verbose (print more stuff).",default=False)
    parser.add_option("-o", "--output_directory", dest="output_directory", default="test/",
                      help="Write results in given output directory (default: ./tests/).")
    (opts, args) = parser.parse_args()


    parameters.VERBOSE = False
    filename = 'output/data_28may17/sim_20170528_060_spectrum.fits'
    atmgrid_filename = filename.replace('sim','reduc').replace('spectrum','atmsim')
    #filename = 'output/data_28may17/reduc_20170528_060_sim.fits'
 

    
    #m = Extractor(filename,atmgrid_filename)
    #m.minimizer(live_fit=True)
    m = Extractor_MCMC(filename,atmgrid_filename=atmgrid_filename,live_fit=False)
    m.mcmc()
    m.test_convergence()
    m.plot_fit()

    #p0 = [0.1,0.1,5,300,0.01,150]
    #constraints = ({'type':'eq', 'fun':lambda p: p[1]=300},
    #               {'type':'eq', 'fun':lambda p: p[1]=300})
    #bounds = ((0,None), (0,1), (0,20), (200,400), (0,0.2),  (0,500))
    #res = minimize(m.chisq,p0,method='Powell',bounds=bounds,
    #               options={'gtol': 1e-2, 'disp': True, 'xtol': 0.1})
    #popt, pcov = curve_fit(func, xdata, ydata)

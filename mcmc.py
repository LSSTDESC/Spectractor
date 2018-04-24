import numpy as np
import sys, os
from texttable import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import parameters
from statistics import *


class Chain(txttableclass):

    def __init__(self,filename,covfile,nchain=0,nsteps=1000):
        txttableclass.__init__(self)
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.nchain = nchain
        self.nsteps = nsteps
        self.start_index = -1
        self.labels = []
        self.axisnames = []
        self.proposal_cov = []
        self.mean_vec = []
        self.start_vec = []
        self.dim = 0
        self.gelman = 2.4
        self.read_covfile(covfile)
        self.load(filename,createit=True)
        self.filename = filename
        self.best_chisq = 1e20
        self.best_key = -1
        self.best_sample = None
        self.set_chain_start()
        
    def load(self,filename,createit=True):
        if(os.path.isfile(filename)):
            if parameters.VERBOSE:
                self.my_logger.info('\n\tLoading '+filename+'...')
            self.loadfile(filename)
            self.config_columns_chain()
        else:
            if(not createit):
                self.my_logger.error('\n\tFile '+filename+' does not exist.')
                sys.exit()
            else:
                self.config_columns_chain()

    def load_and_init(filename,createtit=True):
        self.load(filename,createit)
        self.nsteps = len(self.chain.allrowkeys)

    def config_columns_chain(self):
        # Respect this order
        self.configcols(['Chain', 'Index'],'d','%d',visible=1)
        self.configcols(['Chi2'],'f','%.4g',visible=1)
        for i in range(self.dim):
            self.configcols([self.labels[i]],'f','%.3g',visible=1)

    def read_covfile(self,covfile):
        if(os.path.isfile(covfile)):
            f = open(covfile,'r')
            irow = 0
            for line in f:
                words = line.split()
                if words[0] == 'Parameter' :
                    self.labels.append(words[1])
                    self.axisnames.append(words[2])
                    self.start_vec.append(float(words[3]))
                    self.dim += 1
                else :
                    self.proposal_cov.append(np.zeros(self.dim))
                    for iw,word in enumerate(words) :
                        self.proposal_cov[irow][iw] = float(word)
                    irow += 1
            self.proposal_cov = np.array(self.proposal_cov)
        else:
            self.my_logger.error('\n\tCovariance file %s does not exist.' % covfile)
            sys.exit()

    def set_chain_start(self):
        keys = self.selectkeys(mask=[self.nchain,'exact'],col4mask='Chain')
        if len(keys)>0 :
            self.start_index = np.max(self.getentries(keys,'Index'))
            self.start_key = self.search4entry('Index',self.start_index,mask=[self.nchain,'exact'],col4mask='Chain')
            self.start_vec = []
            for i in range(self.dim):
                self.start_vec.append(self.getentry(self.start_key,self.labels[i]))
        else :
            self.start_index=0
            self.start_key=-1
                
    def make_dictline(self,index,chisq,vec,nbetafit=None):
        dictline = {'Chain':self.nchain,'Index':index}
        dictline.update({'Chi2':chisq})
        for i in range(self.dim) :
            dictline.update({self.labels[i]:vec[i]})
        return(dictline)
 
    def append2filelastkey(self,filename):
        key = self.allrowkeys[-1]
        if(not os.path.isfile(filename)): 
            self.append2file(filename,keys=key,printheader=True,autoformat=True)
        else:
            self.append2file(filename,keys=key,printheader=False,autoformat=True)

    def draw_vector(self,vec):
        # Gelman et al. 1996 Efficient Metropolis Jumping Rules :
        #       cov_proposed / cov_posterior = 2.4/sqrt(dim)
        vec2 = np.random.multivariate_normal(vec,np.array(self.proposal_cov)*(self.gelman*self.gelman)/self.dim) 
        return(vec2)

    def update_proposal_cov(self,vec):
        keys = self.selectkeys(mask=[self.nchain,'exact'],col4mask='Chain')
        # build first mean vector
        if len(self.mean_vec)==0 :
            for i in range(self.dim):
                self.mean_vec.append(np.mean(self.getentries(keys,self.labels[i])) )
            self.mean_vec = np.array(self.mean_vec)
        # update the mean vector
        index = self.getentry(keys[-1],'Index')
        mean_corr = (np.array(vec) - self.mean_vec)/index
        self.mean_vec += mean_corr
        # update the proposal cov matrix
        delta = np.array(vec) - self.mean_vec
        cov_corr = (np.einsum('i,j',delta,delta) - self.proposal_cov)/index
        self.proposal_cov += cov_corr
        # compute acceptance rate over last 100 jumps
        window = 100
        l = len(keys)
        start_index = 1
        if len(keys) > window : start_index = l-window
        ar = self.compute_local_acceptance_rate(start_index,l,keys)
        # update the gelman factor
        c = self.gelman*self.gelman / self.dim
        c_corr = (ar-0.25)/index
        c = c*(1.-1./index) + c_corr
        self.gelman = np.sqrt(c*self.dim)
        if parameters.DEBUG:
            print 'Gelman coefficient: %.3f (index=%d)' % (self.gelman,index)
             
    def compute_local_acceptance_rate(self,start_index,last_index,keys=None):
        if keys==None :
            keys = self.selectkeys(mask=[self.nchain,'exact'],col4mask='Chain')
        frequences = []
        test = self.getentry(keys[start_index],'Chi2')
        counts = 1
        for index in range(start_index+1,last_index):
            chi2 = self.getentry(keys[index],'Chi2')
            if np.isclose(chi2,test) :
                counts += 1
            else :
                frequences.append(float(counts))
                counts = 1
                test = chi2
        frequences.append(counts)
        return(1.0/np.mean(frequences))


class Chains(Chain):

    def __init__(self,filename,covfile,nchains=1,nsteps=1000,burnin=100,nbins=10,truth=None):
        self.filename = filename
        self.chains_filename = self.build_chains_filename()
        Chain.__init__(self,self.chains_filename,covfile,nchain=0,nsteps=nsteps)
        self.nchains = nchains
        self.burnin = burnin
        self.nbins = nbins
        self.truth = truth
        self.chains = []
        for i in range(nchains):
            self.chains.append( Chain(self.chains_filename, covfile, nchain=i, nsteps=nsteps) )

    def build_chains_filename(self):
        chain_name = self.filename.replace('spectrum.fits','chains.txt')
        return chain_name

    def chains_to_likelihood(self,nchain=-1,pdfonly=False):
        self.load(self.chains_filename)
        columns = []
        keys = []
        for key in self.allrowkeys:
            if self.getentry(key,'Index')>self.burnin: keys.append(key)
        rangedim = range(self.dim)
        if nchain!= -1 : keys = self.selectkeys(keys=keys,mask=[nchain,'exact'],col4mask='Chain')
        for i in rangedim:
            columns.append(self.getentries(keys,self.labels[i]))
        weights = None #self.getentries(keys,'Prior')
        centers = []
        for n in rangedim:
            centers.append(np.linspace(np.min(columns[n]),np.max(columns[n]),self.nbins-1))
        self.likelihood = Likelihood(centers,labels=self.labels,axisnames=self.axisnames,truth=self.truth)
        for i in rangedim:
            self.likelihood.pdfs[i].fill_histogram(columns[i],weights)
            if not pdfonly :
                for j in rangedim:
                    if(i != j) :
                        self.likelihood.contours[i][j].fill_histogram(columns[i],columns[j],weights)
        if not pdfonly :
            self.best_chisq = 1e6
            self.best_key = -1
            for key in keys:
                chisq = self.getentry(key,'Chi2')
                if(chisq < self.best_chisq):
                    self.best_chisq = chisq
                    self.best_key = key
            self.best_sample_stats()
        return self.likelihood
            

    def best_sample_stats(self):
        if self.best_key > -1 :
            best_row = self.getrow(self.best_key)
            #self.best_sample.loaddict(best_row)
            print 'Maximum likelihood sample: chi2=%.3g' % self.best_chisq
            for i in range(self.dim): 
                print "\t"+self.labels[i]+": "+str(best_row[self.labels[i]])



    def convergence_tests(self):
        self.load(self.chains_filename)
        nchains = np.sort(np.unique(self.getentries(self.allrowkeys,'Chain')))
        fig = plt.figure(figsize=(16,9))
        ax = [None]*(self.dim+2)
        nrow = self.dim / 2 + self.dim % 2
        # Select keys
        chain_keys = []
        keys = []
        for key in self.allrowkeys:
            if self.getentry(key,'Index')>self.burnin: keys.append(key)
        for nchain in nchains:
            chain_keys.append(self.selectkeys(keys=keys,mask=[nchain,'exact'],col4mask='Chain'))
        print "Computing Parameter vs Index plots..."
        # Parameter vs Index
        for i in range(self.dim):
            ax[i] = fig.add_subplot(nrow,3,i+1)
            for n in range(len(nchains)):
                ax[i].plot(self.getentries(chain_keys[n],'Index'),self.getentries(chain_keys[n],self.labels[i]),label='Chain '+str(nchains[n]))
                ax[i].set_xlabel('Index')
                ax[i].set_ylabel(self.axisnames[i])
            ax[i].legend(loc='upper left', ncol=2, fontsize=10)
        # Chi2 vs Index
        ax[self.dim] = fig.add_subplot(nrow,3,self.dim+1)
        print "Chisq statistics:"
        for n in range(len(nchains)):
            chisqs = self.getentries(chain_keys[n],'Chi2')
            ax[self.dim].plot(self.getentries(chain_keys[n],'Index'),chisqs,label='Chain '+str(nchains[n]))
            print "\tChain %d: %.3f +/- %.3f" % (nchains[n],np.mean(chisqs),np.std(chisqs))
            ax[self.dim].set_xlabel('Index')
            ax[self.dim].set_ylabel('$\chi^2$')
        ax[self.dim].legend(loc='upper left', ncol=2, fontsize=10)
        print "Computing acceptance rate..."
        # Acceptance rate vs Index
        min_len = np.min(map(len, chain_keys))
        window = 100
        if min_len > window :
            ax[self.dim+1] = fig.add_subplot(nrow,3,self.dim+2)
            for n in range(len(nchains)):
                ARs = []
                indices = []
                for l in range(window,min_len,window):
                    #tmp = []
                    #test = self.getentry(chain_keys[n][l-window],'Chi2')
                    #ar = 1
                    #for t in range(l-window+1,l):
                    #    chi2 = self.getentry(chain_keys[n][t],'Chi2')
                    #    if np.isclose(chi2,test) :
                    #        ar += 1
                    #    else :
                    #        tmp.append(float(ar))
                    #        ar = 1
                    #        test = chi2
                    #tmp.append(ar)
                    #ARs.append(1.0/np.mean(tmp))
                    ARs.append(self.compute_local_acceptance_rate(l-window,l,chain_keys[n]))
                    indices.append(l+self.burnin)
                ax[self.dim+1].plot(indices,ARs,label='Chain '+str(nchains[n]))
                ax[self.dim+1].set_xlabel('Index')
                ax[self.dim+1].set_ylabel('Aceptance rate')
                ax[self.dim+1].legend(loc='upper left', ncol=2, fontsize=10)
        # Parameter PDFs by chain
        print "Computing chain by chain PDFs..."
        fig2 = plt.figure(figsize=(16,9))
        ax2 = [None]*(self.dim+1)
        nrow = self.dim / 2 + self.dim % 2
        for n in range(len(nchains)):
            self.chains_to_likelihood(nchain=nchains[n],pdfonly=True)
            self.likelihood.stats(pdfonly=True,verbose=False)
            for i in range(self.dim):
                ax2[i] = fig2.add_subplot(nrow,3,i+1)
                ax2[i].plot(self.likelihood.pdfs[i].axe.axis,self.likelihood.pdfs[i].grid,lw=parameters.LINEWIDTH,label='Chain '+str(nchains[n]))
                ax2[i].set_xlabel(self.axisnames[i])
                ax2[i].set_ylabel('PDF')
                ax2[i].legend(loc='upper right', ncol=2, fontsize=10)
        # Gelman-Rubin test
        if len(nchains) > 1 :
            step = max(1,min_len / 20)
            #if min_len < 200 and min_len > 100 :
            #    step = 20
            #elif min_len < 100 :
            #    step = 2
            print 'Gelman-Rubin tests (burnin=%d, step=%d):' % (self.burnin, step)
            ax2[self.dim] = fig2.add_subplot(nrow,3,self.dim+1)
            for i in range(self.dim):
                Rs = []
                lens = []
                for l in range(step,min_len,step):
                    chain_averages = []
                    chain_variances = []
                    all_keys = []
                    for n in range(len(nchains)):
                        all_keys.append(chain_keys[n][:l])
                    all_keys = np.array(all_keys).flatten()
                    global_average = np.mean(self.getcol(self.labels[i],keys=all_keys))
                    for n in range(len(nchains)):
                        #chain_averages.append(np.mean(self.getentries(chain_keys[n][int(l/2):l],self.input_labels[i])))
                        #chain_variances.append(np.var(self.getentries(chain_keys[n][int(l/2):l],self.input_labels[i]),ddof=1))
                        chain_averages.append(np.mean(self.getentries(chain_keys[n][:l],self.labels[i])))
                        chain_variances.append(np.var(self.getentries(chain_keys[n][:l],self.labels[i]),ddof=1))
                    W = np.mean(chain_variances)
                    B = 0
                    for n in range(len(nchains)):
                        B += (chain_averages[n]-global_average)**2
                    B *= ((l+1)/(len(nchains)-1))
                    R = ( W*l/(l+1) + B/(l+1)*(len(nchains)+1)/len(nchains) ) / W
                    Rs.append(R-1)
                    lens.append(self.burnin+l+1)
                plt.plot(lens,Rs,lw=parameters.LINEWIDTH,label=self.axisnames[i])
                print '\t'+self.labels[i]+' : R-1 = %.3f (l = %d)' % (Rs[-1],lens[-1]-1)
            plt.plot(lens,[0.03]*len(lens),'k--')
            plt.xlabel('Chain length')
            plt.ylabel('$R-1$')
            plt.ylim(0,0.6)
            ax2[self.dim].legend(loc='best', ncol=2, fontsize=10)
        fig.tight_layout()
        fig2.tight_layout()
        plt.show()


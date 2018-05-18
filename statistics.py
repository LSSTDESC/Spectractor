import numpy as np
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tools import *
import parameters


class Parameter():
    __slots__=('label','title','value','set')
    "Class to store parameter value, name and title"
    def __init__(self,value=None,label="",title=""):
        self.label=label
        self.title=title
        self.value=value

    def set(self,value):
        self.value=value
        
class FitParameter():
    __slots__=('label','title','value','cov','chi2','set','write','read')
    "Class to store parameter result from curve fitting"
    def __init__(self,value,cov=-1,chi2=-1,label="",title=""):
        self.label=label
        self.title=title
        self.value=value
        self.cov=cov
        self.chi2=chi2

    def set(self,value,cov,chi2):
        self.value=value
        self.cov=cov
        self.chi2=chi2

    def write(self):
        txt = "%.3g %.3g %.3g " % (self.value,self.cov,self.chi2)
        return(txt)

    def read(self,words,start):
        self.value = float(words[start])
        self.cov = float(words[start+1])
        self.chi2 = float(words[start+2])
        return(3)

class ParameterList():
    __slots__=('data','label','title','mu','std','size','new_sample','set_stats','append','stats','write','read','write_header')
    "Class to store 1D lists and their statistical properties"
    def __init__(self,label="",title=""):
        self.data=[]
        self.label=label
        self.title=title
        self.mu=[]
        self.std=[]
        self.size=[]
        self.new_sample=True

    def reset(self):
        self.mu = []
        self.std = []
        self.size = []
        self.data = []
        self.new_sample = True
        
    def set_stats(self,param):
        if(len(param.mu)!=0):
            self.mu.append(param.mu[0])
            self.std.append(param.std[0])
            self.new_sample=True

    def append(self,item):
        if(self.new_sample):
            self.data.append([])
            self.new_sample=False
        self.data[-1].append(item)

    def stats(self,weights=None,squared=False):
        "Use to close data serie"
        if(len(self.data)!=0):
            if(len(self.data[-1])!=0):
                self.size.append(len(self.data[-1]))
                if(weights is None):
                    if(squared):
                        self.mu.append(np.sqrt(np.mean(np.array(self.data[-1])**2)))
                    else:
                        self.mu.append(np.mean(self.data[-1]))
                    self.std.append(np.std(self.data[-1]))
                else:
                    if(squared):
                        self.mu.append(np.average(np.array(self.data[-1])**2,weights=weights))
                    else:
                        self.mu.append(np.average(self.data[-1],weights=weights))
                    self.std.append(np.sqrt(np.average((self.data[-1]-self.mu[-1])**2,weights=weights)))
                self.new_sample=True

    def write(self):
        txt = "x x "
        if(len(self.mu)!=0):
            txt = "%.3g %.3g " % (self.mu[0],self.std[0])
        return(txt)

    def write_header(self,tag=''):
        txt = self.label+tag+"\t"+self.label+tag+"_std"+"\t"
        return(txt)

    def read(self,words,start):
        if type(words[start]) is str :
            self.mu.append(float(words[start]))
            self.std.append(float(words[start+1]))
        else:
            self.mu.append(words[start])
            self.std.append(words[start+1])
        return(2)


class Axis():
    def __init__(self,l,axisname):
        self.axisname = axisname
        self.set_axis(l)
        self.size = len(l)

    def getAxisVal(self,index):
        return(self.axis[index])

    def getAxisIndex(self,x):
        # optimized for sorted array
        index = np.searchsorted(self.axis, x)
        return(index)

    def set_axis(self,l):
        if isinstance(l,float):
            l = [l]
        else:
            if len(l)==1: l=[l[0]]
        self.axis = np.sort(l)
        self.min = np.min(self.axis)
        self.max = np.max(self.axis)
        self.n = len(self.axis)
        if len(self.axis)>1:
            self.step = np.gradient(self.axis)
        else:
            self.step = [0]
        self.bins = np.zeros(self.n+1)
        self.bins[0] = self.axis[0] - self.step[0]/2
        self.bins[1:] = self.axis[:] + self.step/2

class Grid():
    def __init__(self,dim,ranges,axisnames):
        self.dim=dim
        self.axisnames=axisnames
        self.grid = np.zeros_like(1)   
        self.max = 0
        self.maxindex =  - np.ones(self.dim)
        self.total = 0
        self.rangedim = range(dim)
        if(dim==1 and len(ranges)!=dim): ranges = [ranges]
        if(dim==1 and len(axisnames)!=dim): axisnames = [axisnames]
        self.axes = [Axis(ranges[i],axisnames[i]) for i in self.rangedim]

    def getAxisVal(self,iaxis,index):
        return(self.axes[iaxis].getAxisVal(index))

    def getAxisIndex(self,iaxis,value):
        return(self.axes[iaxis].getAxisIndex(value))
        
    def getAxisIndices(self,values):
        return([self.getAxisIndex(i,values[i]) for i in self.rangedim])

    def getMaximum(self):
        self.max = max(self.grid.flatten())
        self.maxindex = np.where(self.grid==self.max)[0][0]
        return(self.max)

    def getTotal(self):
        if len(self.axes[-1].axis) > 1:
            self.total = np.trapz(y=self.grid,x=self.axes[-1].axis,axis=self.rangedim[-1])
        else:
            self.total = np.sum(self.grid)
        if self.dim > 1 :
            for i in reversed(self.rangedim[:-1]):
                if len(self.axes[i].axis) > 1:
                    self.total = np.trapz(y=self.total,x=self.axes[i].axis,axis=i)
                else:
                    self.total = np.sum(self.total,axis=i)
        return(self.total)

    def normalize(self):
        self.total = self.getTotal()
        if(self.total==0):
            sys.exit('Warning! Sum of likelihood is zero: can not normalize grid %s.' % self.axisnames)
        else:
            self.grid = self.grid/self.total
        return(self.total)

    def marginalizeAlongAxis(self,iaxis):
        #return(np.sum(self.grid,axis=iaxis))
        return(np.trapz(self.grid,self.axes[iaxis].axis,axis=iaxis))

class PDF(Grid):
    def __init__(self,ticks,label='',axisname=''):
        Grid.__init__(self,1,ticks,axisname)
        self.axe = Axis(ticks,axisname)
        self.label = label
        self.axisname = axisname
        self.grid = np.zeros_like(ticks)
        self.mean = None
        self.maxpdf = None
        self.errorlow = None
        self.errorhigh = None
        self.problevels = [ 0.682690, 0.954500 ] #[ 0.682690, 0.954500, 0.9973 ]
        self.variance = None
        self.title = ""
        
    def fill(self,pdf):
        self.grid = pdf
        self.normalize()
        return(pdf)

    def fill_histogram(self,data,weights=None):
        self.grid = np.histogram(data,bins=self.axe.bins,normed=False,weights=weights)[0]
        self.normalize()
    
    def plot(self,truth=None):
        x = self.axe.axis
        y = self.grid
        tck = interpolate.splrep(self.axe.axis,self.grid,s=0)
        if parameters.PAPER:
            x = np.linspace(self.axe.min,self.axe.max,num=101,endpoint=True)
            y = interpolate.splev(x,tck,der=0)
        plt.plot(x,y,lw=parameters.LINEWIDTH)
        # shaded 1 sigma band
        if self.errorlow is not None and self.errorhigh is not None :
            x1s = np.linspace(self.mean-self.errorlow,self.mean+self.errorhigh,num=101,endpoint=True)
            y1s = interpolate.splev(x1s,tck,der=0)
            plt.fill_between(x1s,y1s,0,alpha=0.5,color='cornflowerblue')
            plt.plot([self.mean,self.mean],[0,np.interp(self.mean,x,y)],'b--',lw=parameters.LINEWIDTH)
        # truth
        if truth is not None :
            plt.plot([truth,truth],[0,1.2*max(self.grid)],'r-',lw=parameters.LINEWIDTH)
        plt.xlabel(self.axisname)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.ylim(0,1.2*max(self.grid))
        plt.xlim(self.axe.min,self.axe.max)
        # delete last and first tick labels
        xticks = plt.gca().axes.get_xticks()
        #if xticks[-1] > 1: xticks = xticks[:-1]
        xticks = xticks[:-1]
        plt.gca().axes.set_xticks(xticks[1:])
        if self.title != "": plt.title(self.title)

    def stats(self,verbose=True):
        self.getMaximum()
        self.maxpdf = self.axe.axis[self.maxindex]
        #self.mean = np.average(self.axe.axis,weights=self.grid)
        xprod = np.zeros_like(self.grid)
        for xindex,x in enumerate(self.axe.axis):
            xprod[xindex] = x*self.grid[xindex]
        self.mean = np.trapz(xprod,x=self.axe.axis)
        #cumprob = np.cumsum(self.grid)
        cumprob = np.zeros_like(self.grid)
        cumprob[0]=self.grid[0]
        for xindex in range(len(self.axe.axis))[1:]:
            cumprob[xindex] = cumprob[xindex-1] + 0.5*(self.grid[xindex]+self.grid[xindex-1])*(self.axe.axis[xindex]-self.axe.axis[xindex-1])
        cumbest = np.interp(self.mean,self.axe.axis,cumprob)
        if (cumbest > 1 - self.problevels[0]/2.0):
            if(verbose > 2): print '\tWarning!',self.label,"estimate is too close to cumulative prob upper limit of 1. Errors may not be accurate." 
        if (cumbest < self.problevels[0]/2.0):
            if(verbose > 2): print '\tWarning!',self.label,"estimate is too close to cumulative prob lower limit of 0. Errors may not be accurate."
        upcum = cumbest + self.problevels[0]/2.0
        if(upcum > 1.0): 
            uplimit = np.interp(1,cumprob,self.axe.axis)
        else:
            uplimit = np.interp(upcum,cumprob,self.axe.axis)
        self.errorhigh = uplimit - self.mean
        botcum = cumbest - self.problevels[0]/2.0
        if(botcum < 0.0): 
            lowlimit = np.interp(0,cumprob,self.axe.axis)
        else:
            lowlimit = np.interp(botcum,cumprob,self.axe.axis)
        self.errorlow = self.mean - lowlimit
        # MCI interval
        #print 'Warning! MCI disabled'
        for y in np.linspace(0,self.max,num=1000)[::-1][1:-1]:
            limits = np.where(self.grid>y)[0]
            if len(limits)>1 :
                if limits[0] > 0 :
                    lowx = np.interp(y,self.grid[limits[0]-1:limits[0]+1],self.axe.axis[limits[0]-1:limits[0]+1])
                else :
                    lowx = self.axe.axis[limits[0]]
                if limits[-1] < len(self.axe.axis)-1 :
                    upx = np.interp(y,self.grid[limits[-1]:limits[-1]+2][::-1],self.axe.axis[limits[-1]:limits[-1]+2][::-1])
                else:
                    upx = self.axe.axis[limits[-1]]
                prob = np.interp(upx,self.axe.axis,cumprob) - np.interp(lowx,self.axe.axis,cumprob)
                if prob > self.problevels[0]: break
                self.errorhigh = upx-self.mean
                self.errorlow = -(lowx-self.mean)
        # Estimate variance
        xxprod = np.zeros_like(self.grid)
        for xindex,x in enumerate(self.axe.axis):
            xxprod[xindex] = self.grid[xindex]*(x-self.mean)**2
        self.variance = np.trapz(xxprod,self.axe.axis)
        txt = "\t%s: %s +%s -%s (std: %s)" % formatting_numbers(self.mean,self.errorhigh,self.errorlow,std=np.sqrt(self.variance),label=self.label)
        self.title =  '$%s^{+%s}_{-%s}$' % formatting_numbers(self.mean,self.errorhigh,self.errorlow)
        if(verbose): print txt
        
     

class Contours(Grid):
    def __init__(self,list0,list1,labels,axisnames):
        Grid.__init__(self,2,[list0,list1],axisnames)
        self.grid = np.zeros([len(list0),len(list1)])
        self.axes = [Axis(list0,axisnames[0]),Axis(list1,axisnames[1])]
        self.pdfs = [PDF(list0,labels[0],axisnames[0]),PDF(list1,labels[1],axisnames[1])]
        self.problevels = [ 0.682690, 0.954500 ] #[ 0.682690, 0.954500, 0.9973 ]
        self.labels = labels
        self.cov=0
        self.rho=0
        
    def fill_histogram(self,data1,data2,weights=None):
        self.grid = (np.histogram2d(data1,data2,bins=[list(self.axes[1].bins),list(self.axes[0].bins)],normed=False,weights=weights)[0]).T
        self.normalize()

    def covariance(self):
        self.normalize()
        pdf = np.trapz(y=self.grid,x=self.axes[1].axis,axis=1)
        self.pdfs[0].fill(pdf)
        pdf = np.trapz(y=self.grid,x=self.axes[0].axis,axis=0)
        self.pdfs[1].fill(pdf)
        if self.pdfs[0].maxpdf is None : self.pdfs[0].stats(verbose=False)
        if self.pdfs[1].maxpdf is None : self.pdfs[1].stats(verbose=False)
        self.cov = 0
        #for xindex,x in enumerate(self.axes[0].axis):
        #    for yindex,y in enumerate(self.axes[1].axis):
        #        self.cov += (x-self.pdfs[0].mean)*(y-self.pdfs[1].mean)*self.grid[xindex][yindex]
        xyprod = np.zeros_like(self.grid)
        for xindex,x in enumerate(self.axes[0].axis):
            for yindex,y in enumerate(self.axes[1].axis):
                xyprod[xindex][yindex] = (x-self.pdfs[0].mean)*(y-self.pdfs[1].mean)*self.grid[xindex][yindex]
        self.cov = np.trapz(np.trapz(y=xyprod,x=self.axes[1].axis,axis=1),self.axes[0].axis,axis=0)
        self.rho = self.cov / (np.sqrt(self.pdfs[0].variance*self.pdfs[1].variance))
        return(self.cov)
        

    def plot(self,plot=False,truth=None):
        self.normalize()
        dxdyprod = np.zeros_like(self.grid)
        sortgrid = []
        # Trapezoidal 2D integration with irregular axes
        for xindex,x in enumerate(self.axes[0].axis[:-1]):
            for yindex,y in enumerate(self.axes[1].axis[:-1]):
                val = 0.25*(self.grid[xindex][yindex]+self.grid[xindex+1][yindex]+self.grid[xindex][yindex+1]+self.grid[xindex+1][yindex+1])
                dxdyprod[xindex][yindex] = val*(self.axes[0].axis[xindex+1]-x)*(self.axes[1].axis[yindex+1]-y)
                #dxdyprod[xindex][yindex] = (self.grid[xindex][yindex])*(self.axes[0].axis[xindex+1]-x)*(self.axes[1].axis[yindex+1]-y)
                sortgrid.append((dxdyprod[xindex][yindex],val))
        # Sort dxdyprod keeping a trace of grid sorting
        sortgrid = np.array(sortgrid,dtype=[('dxdyprod',float),('grid',float)])
        sortprob = np.sort(sortgrid,order='dxdyprod')
        dxdyprod, sortgrid = zip(*sortprob)
        # Cumulative integration
        totprob = np.zeros_like(dxdyprod)
        totprob[0] = dxdyprod[0]
        for i in range(1,len(dxdyprod)):
            totprob[i] = totprob[i-1] + dxdyprod[i]
        totprob = 1.0 - totprob
        # Contour levels
        ilevels = []
        for i in range(len(self.problevels)):
            levels = np.where(totprob>self.problevels[i])
            if len(levels[0]) != 0 :
                ilevels.append(levels[0][-1])
        contlevels = np.sort(np.array(sortgrid)[ilevels])
        #if var.PAPER:
        #    f = interpolate.interp2d(self.axes[0].axis,self.axes[1].axis,self.grid.T, kind='linear')
        #    x = np.linspace(self.axes[0].min,self.axes[0].max,2*self.axes[0].size)
        #    y = np.linspace(self.axes[1].min,self.axes[1].max,2*self.axes[1].size)
        #    z = f(x,y)
        #    c = plt.contourf(x,y,z,levels=np.sort(list(contlevels) + [0,np.max(self.grid)]),colors=('w','cornflowerblue','b'),origin='lower')
        #    c2 =  plt.contour(c,levels=contlevels,linewidths=[var.LINEWIDTH,var.LINEWIDTH],colors='b',origin='lower')
        #else:
        c = plt.contourf(self.axes[0].axis,self.axes[1].axis,self.grid.T,levels=np.sort(list(contlevels) + [0,np.max(self.grid)]),colors=('w','cornflowerblue','b'),origin='lower')
        c2 = plt.contour(self.axes[0].axis,self.axes[1].axis,self.grid.T,levels=contlevels,linewidths=[parameters.LINEWIDTH,parameters.LINEWIDTH],colors='b',origin='lower')
        # plot mean values and truth
        plt.plot([self.pdfs[0].mean],[self.pdfs[1].mean],'k*',markersize=10)
        if truth is not None :
            plt.plot([truth[0]],[truth[1]],'ro',markersize=10)
        # set axes
        plt.xlim(self.axes[0].min,self.axes[0].max)
        plt.ylim(self.axes[1].min,self.axes[1].max)
        # delete last and first tick labels
        yticks = plt.gca().axes.get_yticks()
        #if yticks[-1] > 1: yticks = yticks[:-1]
        yticks = yticks[:-1]
        plt.gca().axes.set_yticks(yticks[1:])
        xticks = plt.gca().axes.get_xticks()
        #if xticks[-1] > 1: xticks = xticks[:-1]
        xticks = xticks[:-1]
        plt.gca().axes.set_xticks(xticks[1:])
        if(plot):
            plt.xlabel(self.axisnames[0])
            plt.ylabel(self.axisnames[1])
            plt.show()

class Likelihood(Grid):

    def __init__(self,ranges,labels=['Beta','Bbeta','Depth','Yasked','YBasked'],axisnames=["$\\beta_n$","$\\beta_B$","depth","$Y_n$","$Y_B$"],truth=None):
        Grid.__init__(self,len(ranges),ranges,axisnames)
        self.ranges = ranges
        self.axes = [Axis(self.ranges[i],axisnames[i]) for i in self.rangedim]
        self.pdfs = [PDF(self.ranges[i],labels[i],axisnames[i]) for i in self.rangedim]
        self.contours = [ [ Contours(self.ranges[i],self.ranges[j],[labels[i], labels[j]],[axisnames[i],axisnames[j]]) for i in self.rangedim ] for j in self.rangedim ]
        self.grid = None
        if self.dim <= 6 : self.grid = np.zeros(map(len, ranges))
        self.labels = labels
        self.best_chisq = 1e20
        self.best_key = -1
        self.truth = truth
                      
    def setValue(self,indices,value):
        if(self.dim==4):
            self.grid[indices[0]][indices[1]][indices[2]][indices[3]] = value
        elif(self.dim==5):
            self.grid[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]] = value
        elif(self.dim==6):
            self.grid[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]][indices[5]] = value
        elif(self.dim==7):
            self.grid[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]][indices[5]][indices[6]] = value
        else:
            sys.exit('Warning! In setValue, unsupported grid dimension.')

    def marginalize(self,pdfonly=False):
        for i in self.rangedim:
            for j in self.rangedim:
                axes =  [n for n in self.rangedim if not n in [i,j]]
                if(i==j): 
                    if(j>i): continue
                    if self.dim > 1 :
                        if len(self.axes[axes[-1]].axis)==1:
                            pdf = np.sum(self.grid,axis=axes[-1])
                        else :
                            pdf = np.trapz(y=self.grid,x=self.axes[axes[-1]].axis,axis=axes[-1])
                        for axe in reversed(axes[:-1]):
                            if len(self.axes[axe].axis)==1:
                                pdf = np.sum(pdf,axis=axe)
                            else:
                                pdf = np.trapz(y=pdf,x=self.axes[axe].axis,axis=axe)
                        self.pdfs[i].fill(pdf)
                    else :
                        self.pdfs[i].fill(self.grid)
                else:
                    if(not pdfonly):
                        if self.dim > 2 :
                            if len(self.axes[axes[-1]].axis)==1:
                                self.contours[i][j].grid = np.sum(self.grid,axis=axes[-1])
                            else:
                                self.contours[i][j].grid = np.trapz(y=self.grid,x=self.axes[axes[-1]].axis,axis=axes[-1])
                            for axe in reversed(axes[:-1]):
                                if len(self.axes[axe].axis)==1:
                                    self.contours[i][j].grid = np.sum(self.contours[i][j].grid,axis=axe)
                                else:
                                    self.contours[i][j].grid = np.trapz(y=self.contours[i][j].grid,x=self.axes[axe].axis,axis=axe)
                            if(i<j): self.contours[i][j].grid = self.contours[i][j].grid.T
                        else : 
                            if(i<j): self.contours[i][j].grid = self.grid.T
                            if(j<i): self.contours[i][j].grid = self.grid


    def triangle_plots(self): 
        n = self.dim
        fig=plt.figure(1, figsize=(16,9))
        if parameters.PAPER: fig.set_size_inches(18,13)
        fig.clf()
        for i in range(n):
            n1=self.axisnames[i]
            fig.add_subplot(n,n,i+i*n+1 )
            truth = None
            if self.truth is not None : truth = self.truth[i]
            self.pdfs[i].plot(truth=truth)
            plt.gca().axes.get_xaxis().set_major_locator(MaxNLocator(4,prune='upper'))
            if (i==n-1):
                plt.xlabel(self.axisnames[i])
                plt.gca().axes.get_xaxis().set_label_coords(0.5, -0.3)
            else:
                plt.gca().axes.get_xaxis().set_ticklabels([])
            #print 'hist ',n1
            for j in range(i):
                n1=self.axisnames[i]
                n2=self.axisnames[j]
                #print 'plot ',n1,' vs. ',n2
                fig.add_subplot(n,n,i*n+j+1)
                truth = None
                if self.truth is not None : truth = [self.truth[j],self.truth[i]]
                self.contours[i][j].plot(plot=False,truth=truth)
                plt.gca().axes.get_xaxis().set_major_locator(MaxNLocator(4,prune='upper'))
                plt.gca().axes.get_yaxis().set_major_locator(MaxNLocator(4,prune='upper'))
                if (i==n-1):
                    plt.xlabel(n2)
                    plt.gca().axes.get_xaxis().set_label_coords(0.5, -0.3)
                    #formatter = mpl.ticker.ScalarFormatter(useOffset=False)
                    #plt.gca().axes.get_xaxis().set_major_formatter(formatter)
                    #tick_params(axis='both', which='major')
                if (j==0):
                    plt.ylabel(n1)
                    plt.gca().axes.get_yaxis().set_label_coords(-0.4, 0.5)
                else:
                    plt.gca().axes.get_yaxis().set_ticklabels([])
        fig.tight_layout()
        # embed correlation matrix plot
        inset=plt.axes([0.65,0.65,0.3,0.3]) #This defines the inset
        cax = plt.imshow(self.rho_matrix, interpolation="nearest", cmap='bwr',vmin=-1,vmax=1)
        plt.title('Correlation matrix')
        plt.xticks(range(self.dim),self.axisnames,rotation='vertical',fontsize=11)
        plt.yticks(range(self.dim),self.axisnames, fontsize=11)
        cbar = fig.colorbar(cax)
        cbar.ax.tick_params(labelsize=9) 
        # plot the triangle
        fig.subplots_adjust(hspace=0,wspace=0)
        plt.show()
        if(parameters.SAVE):
            print 'Save '+parameters.PLOT_DIR+'triangle_plot.eps'
            fig.savefig(parameters.PLOT_DIR+'triangle_plot.eps', bbox_inches='tight')

    def max_likelihood_stats(self):
        self.getMaximum()
        if self.best_chisq > 1e6 :
            self.best_chisq = -2.0*np.log( self.total*self.getMaximum() )
        print 'Maximum likelihood position: chi2=%.3g' % self.best_chisq
        for i in self.rangedim:
            print "\t"+self.labels[i]+": "+str(self.getAxisVal(i,self.maxindex[i][0]))

    def stats(self,output='',pdfonly=False,verbose=True):
        #self.max_likelihood_stats()
        if(verbose): print 'Marginalised best fit values (Mean and MCI):'
        self.mean_vec = np.zeros(self.dim)
        for i in self.rangedim:
            self.pdfs[i].stats(verbose=verbose)
            self.mean_vec[i] = self.pdfs[i].mean
        if not pdfonly:
            # Estimate covariance
            self.cov_matrix = np.zeros([self.dim,self.dim])
            self.rho_matrix = np.zeros([self.dim,self.dim])
            for i in self.rangedim:
                for j in self.rangedim:
                    if i==j :
                        self.cov_matrix[i][j] = self.pdfs[i].variance
                        self.rho_matrix[i][j] = 1.0
                    else:
                        self.cov_matrix[i][j] = self.contours[i][j].covariance()
                        self.rho_matrix[i][j] = self.contours[i][j].rho
            if(verbose): print 'Correlation matrix:'
            if(verbose): print('\n'.join([''.join(['\t{0:4.3f}'.format(item) for item in row]) for row in self.rho_matrix]))
            # Output results
            if output is not '' :
                txt = ''
                for i in self.rangedim:
                    txt += 'Parameter '+self.labels[i]+' '+self.axisnames[i]+' '+str(self.pdfs[i].maxpdf)+'\n'
                    if self.labels[i]=='Depth' and self.cov_matrix[i][i] > 20*20:
                        print 'Warning! Depth covariance above 15 pixels. Reduce this to have a correct sampling inside depth prior.'
                cov = '\n'.join([''.join(['\t{0:8.6f}'.format(item) for item in row]) for row in self.cov_matrix])
                f = open(output,'w')
                f.write(txt+cov)
                f.close()



import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
import numpy as np
import sys
import os

from schwimmbad import MPIPool
import emcee
import multiprocessing

from spectractor import parameters
from spectractor.tools import formatting_numbers
from spectractor.config import set_logger
from spectractor.fit.fitter import FitWorkspace, FitParameters


class Axis(object):
    def __init__(self, axis, axisname):
        self.axisname = axisname
        self.axis = []
        self.bins = []
        self.min = 0
        self.max = 0
        self.step = 0
        self.set_axis(axis)
        self.size = len(axis)
        self.txt = ''

    def getAxisVal(self, index):
        return self.axis[index]

    def getAxisIndex(self, x):
        # optimized for sorted array
        index = np.searchsorted(self.axis, x)
        return index

    def set_axis(self, axis):
        if isinstance(axis, float):
            axis = [axis]
        else:
            if len(axis) == 1:
                axis = [axis[0]]
        self.axis = np.sort(axis)
        self.min = np.min(self.axis)
        self.max = np.max(self.axis)
        self.size = len(self.axis)
        if len(self.axis) > 1:
            self.step = np.gradient(self.axis)
        else:
            self.step = [0]
        self.bins = np.zeros(self.size + 1)
        self.bins[0] = self.axis[0] - self.step[0] / 2
        self.bins[1:] = self.axis[:] + self.step / 2


class Grid:
    def __init__(self, dim, ranges, axis_names):
        self.dim = dim
        self.axis_names = axis_names
        self.grid = np.zeros_like(1)
        self.max = 0
        self.max_index = -1
        self.total = 0
        self.rangedim = list(range(dim))
        if dim == 1 and len(ranges) != dim:
            ranges = [ranges]
        if dim == 1 and len(axis_names) != dim:
            self.axis_names = [axis_names]
        self.axes = [Axis(ranges[i], self.axis_names[i]) for i in self.rangedim]

    def getAxisVal(self, axis_index, index):
        return self.axes[axis_index].getAxisVal(index)

    def getAxisIndex(self, axis_index, value):
        return self.axes[axis_index].getAxisIndex(value)

    def getAxisIndices(self, values):
        return [self.getAxisIndex(i, values[i]) for i in self.rangedim]

    def getMaximum(self):
        self.max = np.max(self.grid.flatten())
        self.max_index = np.argmax(self.grid)
        return self.max

    def getTotal(self):
        if len(self.axes[-1].axis) > 1:
            self.total = np.trapz(y=self.grid, x=self.axes[-1].axis, axis=self.rangedim[-1])
        else:
            self.total = np.sum(self.grid)
        if self.dim > 1:
            for i in reversed(self.rangedim[:-1]):
                if len(self.axes[i].axis) > 1:
                    self.total = np.trapz(y=self.total, x=self.axes[i].axis, axis=i)
                else:
                    self.total = np.sum(self.total, axis=i)
        return self.total

    def normalize(self):
        self.total = self.getTotal()
        if self.total == 0:
            sys.exit('Warning! Sum of likelihood is zero: can not normalize grid {}.'.format(self.axis_names))
        else:
            self.grid = self.grid / self.total
        return self.total

    def marginalizeAlongAxis(self, axis_index):
        # return(np.sum(self.grid,axis=axis_index))
        return np.trapz(self.grid, self.axes[axis_index].axis, axis=axis_index)


class PDF(Grid):
    def __init__(self, ticks, label='', axis_name=''):
        Grid.__init__(self, 1, ticks, axis_name)
        self.axe = Axis(ticks, axis_name)
        self.label = label
        self.axis_name = axis_name
        self.grid = np.zeros_like(ticks)
        self.mean = None
        self.max_pdf = None
        self.error_low = None
        self.error_high = None
        self.probability_levels = [0.682690, 0.954500]  # [ 0.682690, 0.954500, 0.9973 ]
        self.variance = None
        self.title = ""

    def fill(self, pdf):
        self.grid = pdf
        self.normalize()
        return pdf

    def fill_histogram(self, data, weights=None):
        self.grid = np.histogram(data, bins=self.axe.bins, density=False, weights=weights)[0]
        self.normalize()

    def plot(self, truth=None):
        """

        Returns:
            :
        """
        x = self.axe.axis
        y = self.grid
        tck = interpolate.splrep(self.axe.axis, self.grid, s=0)
        if parameters.PAPER:
            x = np.linspace(self.axe.min, self.axe.max, num=101, endpoint=True)
            y = interpolate.splev(x, tck, der=0)
        plt.plot(x, y, lw=parameters.LINEWIDTH)
        # shaded 1 sigma band
        if self.error_low is not None and self.error_high is not None:
            x1s = np.linspace(self.mean - self.error_low, self.mean + self.error_high, num=101, endpoint=True)
            y1s = interpolate.splev(x1s, tck, der=0)
            plt.fill_between(x1s, y1s, 0, alpha=0.5, color='cornflowerblue')
            plt.plot([self.mean, self.mean], [0, np.interp(self.mean, x, y)], 'b--', lw=parameters.LINEWIDTH)
        # truth
        if truth is not None:
            plt.plot([truth, truth], [0, 1.2 * max(self.grid)], 'r-', lw=parameters.LINEWIDTH)
        plt.xlabel(self.axis_name)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.ylim(0, 1.2 * max(self.grid))
        plt.xlim(self.axe.min, self.axe.max)
        # delete last and first tick labels
        xticks = plt.gca().axes.get_xticks()
        # if xticks[-1] > 1: xticks = xticks[:-1]
        xticks = xticks[:-1]
        plt.gca().axes.set_xticks(xticks[1:])
        if self.title != "":
            plt.title(self.title)

    def stats(self, verbose=True):
        self.getMaximum()
        self.max_pdf = self.axe.axis[self.max_index]
        # self.mean = np.average(self.axe.axis,weights=self.grid)
        xprod = np.zeros_like(self.grid)
        for xindex, x in enumerate(self.axe.axis):
            xprod[xindex] = x * self.grid[xindex]
        self.mean = np.trapz(xprod, x=self.axe.axis)
        # cumprob = np.cumsum(self.grid)
        cumprob = np.zeros_like(self.grid)
        cumprob[0] = self.grid[0]
        for xindex in range(len(self.axe.axis))[1:]:
            cumprob[xindex] = cumprob[xindex - 1] + 0.5 * (self.grid[xindex] + self.grid[xindex - 1]) * (
                    self.axe.axis[xindex] - self.axe.axis[xindex - 1])
        cumbest = np.interp(self.mean, self.axe.axis, cumprob)
        if cumbest > 1 - self.probability_levels[0] / 2.0:
            if verbose > 2:
                print('\tWarning! {{}} estimate is too close to cumulative prob upper limit of 1. '
                      'Errors may not be accurate.'.format(self.label))
        if cumbest < self.probability_levels[0] / 2.0:
            if verbose > 2:
                print('\tWarning! {{}} estimate is too close to cumulative prob lower limit of 0. '
                      'Errors may not be accurate.'.format(self.label))
        upcum = cumbest + self.probability_levels[0] / 2.0
        if upcum > 1.0:
            uplimit = np.interp(1, cumprob, self.axe.axis)
        else:
            uplimit = np.interp(upcum, cumprob, self.axe.axis)
        self.error_high = uplimit - self.mean
        botcum = cumbest - self.probability_levels[0] / 2.0
        if botcum < 0.0:
            lowlimit = np.interp(0, cumprob, self.axe.axis)
        else:
            lowlimit = np.interp(botcum, cumprob, self.axe.axis)
        self.error_low = self.mean - lowlimit
        # MCI interval
        # print 'Warning! MCI disabled'
        for y in np.linspace(0, self.max, num=1000)[::-1][1:-1]:
            limits = np.where(self.grid > y)[0]
            if len(limits) > 1:
                if limits[0] > 0:
                    lowx = np.interp(y, self.grid[limits[0] - 1:limits[0] + 1],
                                     self.axe.axis[limits[0] - 1:limits[0] + 1])
                else:
                    lowx = self.axe.axis[limits[0]]
                if limits[-1] < len(self.axe.axis) - 1:
                    upx = np.interp(y, self.grid[limits[-1]:limits[-1] + 2][::-1],
                                    self.axe.axis[limits[-1]:limits[-1] + 2][::-1])
                else:
                    upx = self.axe.axis[limits[-1]]
                prob = np.interp(upx, self.axe.axis, cumprob) - np.interp(lowx, self.axe.axis, cumprob)
                if prob > self.probability_levels[0]:
                    break
                self.error_high = upx - self.mean
                self.error_low = -(lowx - self.mean)
        # Estimate variance
        xxprod = np.zeros_like(self.grid)
        for xindex, x in enumerate(self.axe.axis):
            xxprod[xindex] = self.grid[xindex] * (x - self.mean) ** 2
        self.variance = np.trapz(xxprod, self.axe.axis)
        txt = "%s: %s +%s -%s (std: %s)" % formatting_numbers(self.mean, self.error_high, self.error_low,
                                                                std=np.sqrt(self.variance), label=self.label)
        self.title = '$%s^{+%s}_{-%s}$' % formatting_numbers(self.mean, self.error_high, self.error_low)
        self.txt = txt
        self.txt_long = f"{self.label}: {self.mean} +{self.error_high} -{self.error_low} (std: {np.sqrt(self.variance)})"
        if verbose:
            print('\t'+txt)


class Contours(Grid):
    def __init__(self, list0, list1, labels, axisnames):
        Grid.__init__(self, 2, [list0, list1], axisnames)
        self.grid = np.zeros([len(list0), len(list1)])
        self.axes = [Axis(list0, axisnames[0]), Axis(list1, axisnames[1])]
        self.pdfs = [PDF(list0, labels[0], axisnames[0]), PDF(list1, labels[1], axisnames[1])]
        self.problevels = [0.682690, 0.954500]  # [ 0.682690, 0.954500, 0.9973 ]
        self.labels = labels
        self.cov = 0
        self.rho = 0

    def fill_histogram(self, data1, data2, weights=None):
        self.grid = (np.histogram2d(data1, data2, bins=[list(self.axes[1].bins), list(self.axes[0].bins)], density=False,
                                    weights=weights)[0]).T
        self.normalize()

    def covariance(self):
        self.normalize()
        pdf = np.trapz(y=self.grid, x=self.axes[1].axis, axis=1)
        self.pdfs[0].fill(pdf)
        pdf = np.trapz(y=self.grid, x=self.axes[0].axis, axis=0)
        self.pdfs[1].fill(pdf)
        if self.pdfs[0].max_pdf is None:
            self.pdfs[0].stats(verbose=False)
        if self.pdfs[1].max_pdf is None:
            self.pdfs[1].stats(verbose=False)
        self.cov = 0
        # for xindex,x in enumerate(self.axes[0].axis):
        #    for yindex,y in enumerate(self.axes[1].axis):
        #        self.cov += (x-self.pdfs[0].mean)*(y-self.pdfs[1].mean)*self.grid[xindex][yindex]
        xyprod = np.zeros_like(self.grid)
        for xindex, x in enumerate(self.axes[0].axis):
            for yindex, y in enumerate(self.axes[1].axis):
                xyprod[xindex][yindex] = (x - self.pdfs[0].mean) * (y - self.pdfs[1].mean) * self.grid[xindex][yindex]
        self.cov = np.trapz(np.trapz(y=xyprod, x=self.axes[1].axis, axis=1), self.axes[0].axis, axis=0)
        self.rho = self.cov / (np.sqrt(self.pdfs[0].variance * self.pdfs[1].variance))
        return self.cov

    def plot(self, plot=False, truth=None):
        """

        Returns:
            :
        """
        self.normalize()
        dxdyprod = np.zeros_like(self.grid)
        sortgrid = []
        # Trapezoidal 2D integration with irregular axes
        for xindex, x in enumerate(self.axes[0].axis[:-1]):
            for yindex, y in enumerate(self.axes[1].axis[:-1]):
                val = 0.25 * (
                        self.grid[xindex][yindex] + self.grid[xindex + 1][yindex] + self.grid[xindex][yindex + 1] +
                        self.grid[xindex + 1][yindex + 1])
                dxdyprod[xindex][yindex] = val * (self.axes[0].axis[xindex + 1] - x) * (
                        self.axes[1].axis[yindex + 1] - y)
                # dxdyprod[xindex][yindex] =
                # (self.grid[xindex][yindex])*(self.axes[0].axis[xindex+1]-x)*(self.axes[1].axis[yindex+1]-y)
                sortgrid.append((dxdyprod[xindex][yindex], val))
        # Sort dxdyprod keeping a trace of grid sorting
        sortgrid = np.array(sortgrid, dtype=[('dxdyprod', float), ('grid', float)])
        sortprob = np.sort(sortgrid, order='dxdyprod')
        dxdyprod, sortgrid = list(zip(*sortprob))
        # Cumulative integration
        totprob = np.zeros_like(dxdyprod)
        totprob[0] = dxdyprod[0]
        for i in range(1, len(dxdyprod)):
            totprob[i] = totprob[i - 1] + dxdyprod[i]
        totprob = 1.0 - totprob
        # Contour levels
        ilevels = []
        for i in range(len(self.problevels)):
            levels = np.where(totprob > self.problevels[i])
            if len(levels[0]) != 0:
                ilevels.append(levels[0][-1])
        contlevels = np.sort(np.array(sortgrid)[ilevels])
        # if var.PAPER:
        #    f = interpolate.interp2d(self.axes[0].axis,self.axes[1].axis,self.grid.T, kind='linear')
        #    x = np.linspace(self.axes[0].min,self.axes[0].max,2*self.axes[0].size)
        #    y = np.linspace(self.axes[1].min,self.axes[1].max,2*self.axes[1].size)
        #    z = f(x,y)
        #    c = plt.contourf(x,y,z,levels=np.sort(list(contlevels) + [0,np.max(self.grid)]),
        # colors=('w','cornflowerblue','b'),origin='lower')
        #    c2 =  plt.contour(c,levels=contlevels,linewidths=[var.LINEWIDTH,var.LINEWIDTH],colors='b',origin='lower')
        # else:
        plt.contourf(self.axes[0].axis, self.axes[1].axis, self.grid.T,
                     levels=np.sort(list(contlevels) + [0, np.max(self.grid)]), colors=('w', 'cornflowerblue', 'b'),
                     origin='lower')
        plt.contour(self.axes[0].axis, self.axes[1].axis, self.grid.T, levels=contlevels,
                    linewidths=[parameters.LINEWIDTH, parameters.LINEWIDTH], colors='b', origin='lower')
        # plot mean values and truth
        plt.plot([self.pdfs[0].mean], [self.pdfs[1].mean], 'k*', markersize=10)
        if truth is not None:
            plt.plot([truth[0]], [truth[1]], 'ro', markersize=10)
            plt.axhline(truth[1], color='r', linewidth=parameters.LINEWIDTH)
            plt.axvline(truth[0], color='r', linewidth=parameters.LINEWIDTH)
        # set axes
        plt.xlim(self.axes[0].min, self.axes[0].max)
        plt.ylim(self.axes[1].min, self.axes[1].max)
        # delete last and first tick labels
        yticks = plt.gca().axes.get_yticks()
        # if yticks[-1] > 1: yticks = yticks[:-1]
        yticks = yticks[:-1]
        plt.gca().axes.set_yticks(yticks[1:])
        xticks = plt.gca().axes.get_xticks()
        # if xticks[-1] > 1: xticks = xticks[:-1]
        xticks = xticks[:-1]
        plt.gca().axes.set_xticks(xticks[1:])
        if plot:
            plt.xlabel(self.axis_names[0])
            plt.ylabel(self.axis_names[1])
            if parameters.DISPLAY:
                plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()


class Likelihood(Grid):

    def __init__(self, ranges, labels=[], axis_names=[], truth=None):
        Grid.__init__(self, len(ranges), ranges, axis_names)
        self.ranges = ranges
        self.axes = [Axis(self.ranges[i], axis_names[i]) for i in self.rangedim]
        self.pdfs = [PDF(self.ranges[i], labels[i], axis_names[i]) for i in self.rangedim]
        self.contours = [
            [Contours(self.ranges[i], self.ranges[j], [labels[i], labels[j]], [axis_names[i], axis_names[j]]) for i in
             self.rangedim] for j in self.rangedim]
        self.grid = None
        if self.dim <= 6:
            self.grid = np.zeros(list(map(len, ranges)))
        self.labels = labels
        self.best_chisq = 1e20
        self.truth = truth
        self.cov_matrix = None
        self.rho_matrix = None
        self.mean_vec = None

    def setValue(self, indices, value):  # pragma: no cover
        if self.dim == 4:
            self.grid[indices[0]][indices[1]][indices[2]][indices[3]] = value
        elif self.dim == 5:
            self.grid[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]] = value
        elif self.dim == 6:
            self.grid[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]][indices[5]] = value
        elif self.dim == 7:
            self.grid[indices[0]][indices[1]][indices[2]][indices[3]][indices[4]][indices[5]][indices[6]] = value
        else:
            sys.exit('Warning! In setValue, unsupported grid dimension.')

    def marginalize(self, pdfonly=False):  # pragma: no cover
        for i in self.rangedim:
            for j in self.rangedim:
                axes = [n for n in self.rangedim if n not in [i, j]]
                if i == j:
                    if j > i:
                        continue
                    if self.dim > 1:
                        if len(self.axes[axes[-1]].axis) == 1:
                            pdf = np.sum(self.grid, axis=axes[-1])
                        else:
                            pdf = np.trapz(y=self.grid, x=self.axes[axes[-1]].axis, axis=axes[-1])
                        for axe in reversed(axes[:-1]):
                            if len(self.axes[axe].axis) == 1:
                                pdf = np.sum(pdf, axis=axe)
                            else:
                                pdf = np.trapz(y=pdf, x=self.axes[axe].axis, axis=axe)
                        self.pdfs[i].fill(pdf)
                    else:
                        self.pdfs[i].fill(self.grid)
                else:
                    if not pdfonly:
                        if self.dim > 2:
                            if len(self.axes[axes[-1]].axis) == 1:
                                self.contours[i][j].grid = np.sum(self.grid, axis=axes[-1])
                            else:
                                self.contours[i][j].grid = np.trapz(y=self.grid, x=self.axes[axes[-1]].axis,
                                                                    axis=axes[-1])
                            for axe in reversed(axes[:-1]):
                                if len(self.axes[axe].axis) == 1:
                                    self.contours[i][j].grid = np.sum(self.contours[i][j].grid, axis=axe)
                                else:
                                    self.contours[i][j].grid = np.trapz(y=self.contours[i][j].grid,
                                                                        x=self.axes[axe].axis, axis=axe)
                            if i < j:
                                self.contours[i][j].grid = self.contours[i][j].grid.T
                        else:
                            if i < j:
                                self.contours[i][j].grid = self.grid.T
                            if j < i:
                                self.contours[i][j].grid = self.grid

    def triangle_plots(self, output_filename=''):
        n = self.dim
        fig = plt.figure(1, figsize=(16, 9))
        if parameters.PAPER:
            fig.set_size_inches(18, 13)
        fig.clf()
        for i in range(n):
            fig.add_subplot(n, n, i + i * n + 1)
            truth = None
            if self.truth is not None:
                truth = self.truth[i]
            self.pdfs[i].plot(truth=truth)
            plt.gca().axes.get_xaxis().set_major_locator(MaxNLocator(4, prune='upper'))
            if i == n - 1:
                plt.xlabel(self.axis_names[i])
                plt.gca().axes.get_xaxis().set_label_coords(0.5, -0.3)
            else:
                plt.gca().axes.get_xaxis().set_ticklabels([])
            # print 'hist ',n1
            for j in range(i):
                n1 = self.axis_names[i]
                n2 = self.axis_names[j]
                # print 'plot ',n1,' vs. ',n2
                fig.add_subplot(n, n, i * n + j + 1)
                truth = None
                if self.truth is not None:
                    truth = [self.truth[j], self.truth[i]]
                self.contours[i][j].plot(plot=False, truth=truth)
                plt.gca().axes.get_xaxis().set_major_locator(MaxNLocator(4, prune='upper'))
                plt.gca().axes.get_yaxis().set_major_locator(MaxNLocator(4, prune='upper'))
                if i == n - 1:
                    plt.xlabel(n2)
                    plt.gca().axes.get_xaxis().set_label_coords(0.5, -0.3)
                    # formatter = mpl.ticker.ScalarFormatter(useOffset=False)
                    # plt.gca().axes.get_xaxis().set_major_formatter(formatter)
                    # tick_params(axis='both', which='major')
                if j == 0:
                    plt.ylabel(n1)
                    plt.gca().axes.get_yaxis().set_label_coords(-0.32, 0.5)
                else:
                    plt.gca().axes.get_yaxis().set_ticklabels([])
        fig.tight_layout()
        # embed correlation matrix plot
        plt.axes([0.65, 0.65, 0.3, 0.3])  # This defines the inset
        cax = plt.imshow(self.rho_matrix, interpolation="nearest", cmap='bwr', vmin=-1, vmax=1)
        plt.title('Correlation matrix')
        plt.xticks(list(range(self.dim)), self.axis_names, rotation='vertical', fontsize=11)
        plt.yticks(list(range(self.dim)), self.axis_names, fontsize=11)
        cbar = fig.colorbar(cax)
        cbar.ax.tick_params(labelsize=9)
        # plot the triangle
        fig.subplots_adjust(hspace=0, wspace=0)
        if parameters.DISPLAY:
            plt.show()
        if output_filename != '':
            print(f'Save figure: {output_filename}')
            fig.savefig(output_filename, dpi=100, bbox_inches='tight')
        if parameters.PdfPages:
            parameters.PdfPages.savefig()
        return fig

    def max_likelihood_stats(self):
        self.getMaximum()
        if self.best_chisq > 1e6:
            self.best_chisq = -2.0 * np.log(self.total * self.getMaximum())
        print('Maximum likelihood position: chi2={:.3g}'.format(self.best_chisq))
        for i in self.rangedim:
            print("\t" + self.labels[i] + ": " + str(self.getAxisVal(i, self.max_index)))

    def stats(self, output='', pdfonly=False, verbose=True):
        #self.max_likelihood_stats()
        if verbose:
            print('Marginalised best fit values (Mean and MCI):')
        self.mean_vec = np.zeros(self.dim)
        for i in self.rangedim:
            self.pdfs[i].stats(verbose=verbose)
            self.mean_vec[i] = self.pdfs[i].mean
        if not pdfonly:
            # Estimate covariance
            self.cov_matrix = np.zeros([self.dim, self.dim])
            self.rho_matrix = np.zeros([self.dim, self.dim])
            for i in self.rangedim:
                for j in self.rangedim:
                    if i == j:
                        self.cov_matrix[i][j] = self.pdfs[i].variance
                        self.rho_matrix[i][j] = 1.0
                    else:
                        self.cov_matrix[i][j] = self.contours[i][j].covariance()
                        self.rho_matrix[i][j] = self.contours[i][j].rho
            if verbose:
                print('Correlation matrix:')
            if verbose:
                print(('\n'.join([''.join(['\t{0:4.3f}'.format(item) for item in row]) for row in self.rho_matrix])))
            # Output results
            if output != '':
                txt = ''
                for i in self.rangedim:
                    txt += f'{self.pdfs[i].txt_long}\n'
                cov = '\n'.join([''.join(['\t{0:8.6f}'.format(item) for item in row]) for row in self.cov_matrix])
                print(f'Save best fit parameters in {output}')
                f = open(output, 'w')
                f.write(txt + cov)
                f.close()



class MCMCFitWorkspace(FitWorkspace):

    def __init__(self, params, file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=False, plot=False, live_fit=False, truth=None):
        """Generic class to create a fit workspace with parameters, bounds and general fitting methods.

        Parameters
        ----------
        params: FitParameters
            The parameters to fit to data.
        file_name: str, optional
            The generic file name to save results. If file_name=="", nothing is saved ond disk (default: "").
        nwalkers: int, optional
            Number of walkers for MCMC exploration (default: 18).
        nsteps: int, optional
            Number of steps for MCMC exploration (default: 1000).
        burnin: int, optional
            Number of burn-in steps for MCMC exploration (default: 100).
        nbins: int, optional
            Number of bins to make histograms after MCMC exploration (default: 10).
        verbose: bool, optional
            Level of verbosity (default: False).
        plot: bool, optional
            Level of plotting (default: False).
        live_fit: bool, optional
            If True, model, data and residuals plots are made along the fitting procedure (default: False).
        truth: array_like, optional
            Array of true parameters (default: None).

        Examples
        --------
        >>> params = FitParameters(values=[1, 1, 1, 1, 1])
        >>> w = MCMCFitWorkspace(params)
        >>> w.nwalkers
        18
        """
        FitWorkspace.__init__(self, params, file_name=file_name, verbose=verbose, plot=plot, live_fit=live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        self.nwalkers = max(2 * self.params.ndim, nwalkers)
        self.nsteps = nsteps
        self.nbins = nbins
        self.burnin = burnin
        self.start = []
        self.likelihood = np.array([[]])
        self.gelmans = np.array([])
        self.chains = np.array([[]])
        self.lnprobs = np.array([[]])
        self.flat_chains = np.array([[]])
        self.valid_chains = [False] * self.nwalkers
        self.global_average = None
        self.global_std = None
        self.use_grid = False
        if self.filename != "":
            if "." in self.filename:
                self.emcee_filename = os.path.splitext(self.filename)[0] + "_emcee.h5"
            else:
                self.my_logger.warning("\n\tFile name must have an extension.")
        else:
            self.emcee_filename = "emcee.h5"

    def set_start(self, percent=0.02, a_random=1e-5):
        """Set the random starting points for MCMC exploration.

        A set of parameters are drawn with a uniform distribution between +/- percent times the starting guess.
        For null guess parameters, starting points are drawn from a uniform distribution between +/- a_random.

        Parameters
        ----------
        percent: float, optional
            Percent of the guess parameters to set the uniform interval to draw random points (default: 0.02).
        a_random: float, optional
            Absolute value to set the +/- uniform interval to draw random points
            for null guess parameters (default: 1e-5).

        Returns
        -------
        start: np.array
            Array of starting points of shape (ndim, nwalkers).

        """
        self.start = np.array([np.random.uniform(self.params.values[i] - percent * self.params.values[i],
                                                 self.params.values[i] + percent * self.params.values[i],
                                                 self.nwalkers) for i in range(self.params.ndim)]).T
        self.start[self.start == 0] = a_random * np.random.uniform(-1, 1)
        return self.start

    def load_chains(self):
        """Load the MCMC chains from a hdf5 file. The burn-in points are not rejected at this stage.

        Returns
        -------
        chains: np.array
            Array of the chains.
        lnprobs: np.array
            Array of the logarithmic posterior probability.

        """
        self.chains = [[]]
        self.lnprobs = [[]]
        self.nsteps = 0
        # tau = -1
        reader = emcee.backends.HDFBackend(self.emcee_filename)
        try:
            tau = reader.get_autocorr_time()
        except emcee.autocorr.AutocorrError:
            tau = -1
        self.chains = reader.get_chain(discard=0, flat=False, thin=1)
        self.lnprobs = reader.get_log_prob(discard=0, flat=False, thin=1)
        self.nsteps = self.chains.shape[0]
        self.nwalkers = self.chains.shape[1]
        print(f"Auto-correlation time: {tau}")
        print(f"Burn-in: {self.burnin}")
        print(f"Chains shape: {self.chains.shape}")
        print(f"Log prob shape: {self.lnprobs.shape}")
        return self.chains, self.lnprobs

    def build_flat_chains(self):
        """Flatten the chains array and apply burn-in.

        Returns
        -------
        flat_chains: np.array
            Flat chains.

        """
        self.flat_chains = self.chains[self.burnin:, self.valid_chains, :].reshape((-1, self.params.ndim))
        return self.flat_chains

    def analyze_chains(self):
        """Load the chains, build the probability densities for the parameters, compute the best fitting values
        and the uncertainties and covariance matrices, and plot.

        """
        self.load_chains()
        self.set_chain_validity()
        self.convergence_tests()
        self.build_flat_chains()
        self.likelihood = self.chain2likelihood()
        self.params.cov = self.likelihood.cov_matrix
        self.params.values = self.likelihood.mean_vec
        self.simulate(*self.params.values)
        self.plot_fit()
        figure_name = os.path.splitext(self.emcee_filename)[0] + '_triangle.pdf'
        self.likelihood.triangle_plots(output_filename=figure_name)

    def chain2likelihood(self, pdfonly=False, walker_index=-1):
        """Convert the chains to a psoterior probability density function via histograms.

        Parameters
        ----------
        pdfonly: bool, optional
            If True, do not compute the covariances and the 2D correlation plots (default: False).
        walker_index: int, optional
            The walker index to plot. If -1, all walkers are selected (default: -1).

        Returns
        -------
        likelihood: np.array
            Posterior density function.

        """
        if walker_index >= 0:
            chains = self.chains[self.burnin:, walker_index, :]
        else:
            chains = self.flat_chains
        rangedim = range(chains.shape[1])
        centers = []
        for i in rangedim:
            centers.append(np.linspace(np.min(chains[:, i]), np.max(chains[:, i]), self.nbins - 1))
        likelihood = Likelihood(centers, labels=self.params.labels, axis_names=self.params.axis_names, truth=self.params.truth)
        if walker_index < 0:
            for i in rangedim:
                likelihood.pdfs[i].fill_histogram(chains[:, i], weights=None)
                if not pdfonly:
                    for j in rangedim:
                        if i != j:
                            likelihood.contours[i][j].fill_histogram(chains[:, i], chains[:, j], weights=None)
            output_file = ""
            if self.filename != "":
                output_file = os.path.splitext(self.filename)[0] + "_bestfit.txt"
            likelihood.stats(output=output_file)
        else:
            for i in rangedim:
                likelihood.pdfs[i].fill_histogram(chains[:, i], weights=None)
        return likelihood

    def compute_local_acceptance_rate(self, start_index, last_index, walker_index):
        """Compute the local acceptance rate in a chain.

        Parameters
        ----------
        start_index: int
            Beginning index.
        last_index: int
            End index.
        walker_index: int
            Index of the walker.

        Returns
        -------
        freq: float
            The acceptance rate.

        """
        frequences = []
        test = -2 * self.lnprobs[start_index, walker_index]
        counts = 1
        for index in range(start_index + 1, last_index):
            chi2 = -2 * self.lnprobs[index, walker_index]
            if np.isclose(chi2, test):
                counts += 1
            else:
                frequences.append(float(counts))
                counts = 1
                test = chi2
        frequences.append(counts)
        return 1.0 / np.mean(frequences)

    def set_chain_validity(self):
        """Test the validity of a chain: reject chains whose chi2 is far from the mean of the others.

        Returns
        -------
        valid_chains: list
            List of boolean values, True if the chain is valid, or False if invalid.

        """
        nchains = [k for k in range(self.nwalkers)]
        chisq_averages = []
        chisq_std = []
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            # if np.mean(chisqs) < 1e5:
            chisq_averages.append(np.mean(chisqs))
            chisq_std.append(np.std(chisqs))
        self.global_average = np.mean(chisq_averages)
        self.global_std = np.mean(chisq_std)
        self.valid_chains = [False] * self.nwalkers
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            chisq_average = np.mean(chisqs)
            chisq_std = np.std(chisqs)
            if 3 * self.global_std + self.global_average < chisq_average < 1e5:
                self.valid_chains[k] = False
            elif chisq_std < 0.1 * self.global_std:
                self.valid_chains[k] = False
            else:
                self.valid_chains[k] = True
        return self.valid_chains

    def convergence_tests(self):
        """Compute the convergence tests (Gelman-Rubin, acceptance rate).

        """
        chains = self.chains[self.burnin:, :, :]  # .reshape((-1, self.ndim))
        nchains = [k for k in range(self.nwalkers)]
        fig, ax = plt.subplots(self.params.ndim + 1, 2, figsize=(16, 7), sharex='all')
        fontsize = 8
        steps = np.arange(self.burnin, self.nsteps)
        # Chi2 vs Index
        print("Chisq statistics:")
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            text = f"\tWalker {k:d}: {float(np.mean(chisqs)):.3f} +/- {float(np.std(chisqs)):.3f}"
            if not self.valid_chains[k]:
                text += " -> excluded"
                ax[self.params.ndim, 0].plot(steps, chisqs, c='0.5', linestyle='--')
            else:
                ax[self.params.ndim, 0].plot(steps, chisqs)
            print(text)
        # global_average = np.mean(-2*self.lnprobs[self.valid_chains, self.burnin:])
        # global_std = np.std(-2*self.lnprobs[self.valid_chains, self.burnin:])
        ax[self.params.ndim, 0].set_ylim(
            [self.global_average - 5 * self.global_std, self.global_average + 5 * self.global_std])
        # Parameter vs Index
        print("Computing Parameter vs Index plots...")
        for i in range(self.params.ndim):
            ax[i, 0].set_ylabel(self.params.axis_names[i], fontsize=fontsize)
            for k in nchains:
                if self.valid_chains[k]:
                    ax[i, 0].plot(steps, chains[:, k, i])
                else:
                    ax[i, 0].plot(steps, chains[:, k, i], c='0.5', linestyle='--')
                ax[i, 0].get_yaxis().set_label_coords(-0.05, 0.5)
        ax[self.params.ndim, 0].set_ylabel(r'$\chi^2$', fontsize=fontsize)
        ax[self.params.ndim, 0].set_xlabel('Steps', fontsize=fontsize)
        ax[self.params.ndim, 0].get_yaxis().set_label_coords(-0.05, 0.5)
        # Acceptance rate vs Index
        print("Computing acceptance rate...")
        min_len = self.nsteps
        window = 100
        if min_len > window:
            for k in nchains:
                ARs = []
                indices = []
                for pos in range(self.burnin + window, self.nsteps, window):
                    ARs.append(self.compute_local_acceptance_rate(pos - window, pos, k))
                    indices.append(pos)
                if self.valid_chains[k]:
                    ax[self.params.ndim, 1].plot(indices, ARs, label=f'Walker {k:d}')
                else:
                    ax[self.params.ndim, 1].plot(indices, ARs, label=f'Walker {k:d}', c='gray', linestyle='--')
                ax[self.params.ndim, 1].set_xlabel('Steps', fontsize=fontsize)
                ax[self.params.ndim, 1].set_ylabel('Aceptance rate', fontsize=fontsize)
                # ax[self.dim + 1, 2].legend(loc='upper left', ncol=2, fontsize=10)
        # Parameter PDFs by chain
        print("Computing chain by chain PDFs...")
        for k in nchains:
            likelihood = self.chain2likelihood(pdfonly=True, walker_index=k)
            likelihood.stats(pdfonly=True, verbose=False)
            # for i in range(self.dim):
            # ax[i, 1].plot(likelihood.pdfs[i].axe.axis, likelihood.pdfs[i].grid, lw=var.LINEWIDTH,
            #               label=f'Walker {k:d}')
            # ax[i, 1].set_xlabel(self.axis_names[i])
            # ax[i, 1].set_ylabel('PDF')
            # ax[i, 1].legend(loc='upper right', ncol=2, fontsize=10)
        # Gelman-Rubin test.py
        if len(nchains) > 1:
            step = max(1, (self.nsteps - self.burnin) // 20)
            self.gelmans = []
            print(f'Gelman-Rubin tests (burnin={self.burnin:d}, step={step:d}, nsteps={self.nsteps:d}):')
            for i in range(self.params.ndim):
                Rs = []
                lens = []
                for pos in range(self.burnin + step, self.nsteps, step):
                    chain_averages = []
                    chain_variances = []
                    global_average = np.mean(self.chains[self.burnin:pos, self.valid_chains, i])
                    for k in nchains:
                        if not self.valid_chains[k]:
                            continue
                        chain_averages.append(np.mean(self.chains[self.burnin:pos, k, i]))
                        chain_variances.append(np.var(self.chains[self.burnin:pos, k, i], ddof=1))
                    W = np.mean(chain_variances)
                    B = 0
                    for n in range(len(chain_averages)):
                        B += (chain_averages[n] - global_average) ** 2
                    B *= ((pos + 1) / (len(chain_averages) - 1))
                    R = (W * pos / (pos + 1) + B / (pos + 1) * (len(chain_averages) + 1) / len(chain_averages)) / W
                    Rs.append(R - 1)
                    lens.append(pos)
                print(f'\t{self.params.labels[i]}: R-1 = {Rs[-1]:.3f} (l = {lens[-1] - 1:d})')
                self.gelmans.append(Rs[-1])
                ax[i, 1].plot(lens, Rs, lw=1, label=self.params.axis_names[i])
                ax[i, 1].axhline(0.03, c='k', linestyle='--')
                ax[i, 1].set_xlabel('Walker length', fontsize=fontsize)
                ax[i, 1].set_ylabel('$R-1$', fontsize=fontsize)
                ax[i, 1].set_ylim(0, 0.6)
                # ax[self.dim, 3].legend(loc='best', ncol=2, fontsize=10)
        self.gelmans = np.array(self.gelmans)
        fig.tight_layout()
        plt.subplots_adjust(hspace=0)
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()
        figure_name = self.emcee_filename.replace('.h5', '_convergence.pdf')
        print(f'Save figure: {figure_name}')
        fig.savefig(figure_name, dpi=100)

    def print_settings(self):
        """Print the main settings of the FitWorkspace.

        """
        print('************************************')
        print(f"Input file: {self.filename}\nWalkers: {self.nwalkers}\t Steps: {self.nsteps}")
        print(f"Output file: {self.emcee_filename}")
        print('************************************')

    def lnprior(self, p):
        """Compute the logarithmic prior for a set of model parameters p.

        The function returns 0 for good parameters, and -np.inf for parameters out of their boundaries.

        Parameters
        ----------
        p: array_like
            The array of model parameters.

        Returns
        -------
        lnprior: float
            The logarithmic value fo the prior.

        """
        in_bounds = True
        for npar, par in enumerate(p):
            if par < self.params.bounds[npar][0] or par > self.params.bounds[npar][1]:
                in_bounds = False
                break
        if in_bounds:
            return 0.0
        else:
            return -np.inf

def lnprob(p):  # pragma: no cover
    global fit_workspace
    lp = fit_workspace.lnprior(p)
    if not np.isfinite(lp):
        return -1e20
    return lp + fit_workspace.lnlike(p)


def run_emcee(mcmc_fit_workspace, ln=lnprob):
    my_logger = set_logger(__name__)
    mcmc_fit_workspace.print_settings()
    nsamples = mcmc_fit_workspace.nsteps
    p0 = mcmc_fit_workspace.set_start()
    filename = mcmc_fit_workspace.emcee_filename
    backend = emcee.backends.HDFBackend(filename)
    try:  # pragma: no cover
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(mcmc_fit_workspace.nwalkers, mcmc_fit_workspace.ndim, ln, args=(),
                                        pool=pool, backend=backend)
        my_logger.info(f"\n\tInitial size: {backend.iteration}")
        if backend.iteration > 0:
            p0 = backend.get_last_sample()
        if nsamples - backend.iteration > 0:
            sampler.run_mcmc(p0, nsteps=max(0, nsamples - backend.iteration), progress=True)
        pool.close()
    except ValueError:
        sampler = emcee.EnsembleSampler(mcmc_fit_workspace.nwalkers, mcmc_fit_workspace.params.ndim, ln, args=(),
                                        threads=multiprocessing.cpu_count(), backend=backend)
        my_logger.info(f"\n\tInitial size: {backend.iteration}")
        if backend.iteration > 0:
            p0 = sampler.get_last_sample()
        for _ in sampler.sample(p0, iterations=max(0, nsamples - backend.iteration), progress=True, store=True):
            continue
    mcmc_fit_workspace.chains = sampler.chain
    mcmc_fit_workspace.lnprobs = sampler.lnprobability

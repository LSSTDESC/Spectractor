"""
throughput.py
=============----

author : Sylvie Dagoret-Campagne, Jérémy Neveu
affiliation : LAL/CNRS/IN2P3/FRANCE
Collaboration : DESC-LSST

Purpose : Provide the various useful transmission functions
update : July 2018

"""

import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii

import spectractor.parameters as parameters


class Throughput():

    def __init__(self,input_directory=parameters.THROUGHPUT_DIR):
        self.path_transmission = input_directory
        self.filename_quantum_efficiency = os.path.join(self.path_transmission, "qecurve.txt")
        self.filename_FGB37 = os.path.join(self.path_transmission, "FGB37.txt")
        self.filename_RG715 = os.path.join(self.path_transmission, "RG175.txt")
        self.filename_telescope_throughput = os.path.join(self.path_transmission, 'ctio_throughput.txt')
        self.filename_mirrors = os.path.join(self.path_transmission, 'lsst_mirrorthroughput.txt')
        self.filename_total_throughput = os.path.join(self.path_transmission, 'ctio_throughput_300517_v1.txt')
        # self.filename_total_throughput = os.path.join(self.path_transmission, '20171006_RONCHI400_clear_45_median_tpt.txt')

    def get_quantum_efficiency(self):
        data_qe = ascii.read(self.filename_quantum_efficiency)
        x = data_qe["col1"]
        y = data_qe["col2"] / 100.
        indexes = np.where(np.logical_and(x > parameters.LAMBDA_MIN, x < parameters.LAMBDA_MAX))
        return x[indexes], y[indexes]

    def get_RG715(self):
        data_rg = ascii.read(self.filename_RG715)
        x = data_rg["col2"]
        y = data_rg["col3"] / 100.
        indexes = np.where(np.logical_and(x > parameters.LAMBDA_MIN, x < parameters.LAMBDA_MAX))
        return x[indexes], y[indexes]

    def get_FGB37(self):
        data_fgb = ascii.read(self.filename_FGB37)
        x = data_fgb["col2"]
        y = data_fgb["col3"] / 100.
        indexes = np.where(np.logical_and(x > parameters.LAMBDA_MIN, x < parameters.LAMBDA_MAX))
        return x[indexes], y[indexes]

    def get_telescope_throughput(self):
        data_rg = ascii.read(self.filename_telescope_throughput)
        x = data_rg["col1"]
        y = data_rg["col2"]
        indexes = np.where(np.logical_and(x > parameters.LAMBDA_MIN, x < parameters.LAMBDA_MAX))
        return x[indexes], y[indexes]

    def get_total_throughput(self):
        # data_rg = ascii.read(self.filename_total_throughput)
        # x = data_rg["col1"]
        # y = data_rg["col2"]
        # z = data_rg["col3"]
        data = np.loadtxt(self.filename_total_throughput)
        x = data.T[0]
        y = data.T[1]
        z = data.T[2]
        indexes = np.where(np.logical_and(x > parameters.LAMBDA_MIN, x < parameters.LAMBDA_MAX))
        return x[indexes], y[indexes], z[indexes]

    def get_mirror_reflectivity(self):
        data_m = ascii.read(self.filename_mirrors)
        x = data_m["col1"]
        y = data_m["col2"]
        indexes = np.where(np.logical_and(x > parameters.LAMBDA_MIN, x < parameters.LAMBDA_MAX))
        return x[indexes], y[indexes]

    def PlotQE(self):
        wl, qe = self.get_quantum_efficiency()
        plt.figure()
        plt.plot(wl, qe, "-")
        plt.title('Quantum efficiency')
        plt.xlabel("$\lambda$")
        plt.ylabel("QE")
        plt.xlim(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX)

    def PlotRG(self):
        wl, trg = self.get_RG715()
        plt.figure()
        plt.plot(wl, trg, "-")
        plt.title('RG175')
        plt.xlabel("$\lambda$")
        plt.ylabel("transmission")
        plt.xlim(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX)

    def PlotFGB(self):
        wl, trg = self.get_FGB37()
        plt.figure()
        plt.plot(wl, trg, "-")
        plt.title('FGB37')
        plt.xlabel("$\lambda$")
        plt.ylabel("transmission")
        plt.xlim(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX)

    def PlotThroughput(self):
        wl, trt = self.get_telescope_throughput()
        plt.figure()
        plt.plot(wl, trt, "-")
        plt.title('throughput')
        plt.xlabel("$\lambda$")
        plt.ylabel("transmission")
        plt.xlim(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX)

    def PlotMirror(self):
        wl, tr = self.get_mirror_reflectivity()
        plt.figure()
        plt.plot(wl, tr, "b-", label='1 mirror')
        plt.plot(wl, tr * tr, "r-", label='2 mirrors')
        plt.title('Mirror')
        plt.xlabel("$\lambda$")
        plt.ylabel("transmission")
        plt.legend()
        plt.xlim(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX)


if __name__ == "__main__":

    throughput = Throughput()
    throughput.PlotQE()
    throughput.PlotRG()
    throughput.PlotFGB()
    throughput.PlotThroughput()
    throughput.PlotMirror()
    if parameters.DISPLAY: plt.show()
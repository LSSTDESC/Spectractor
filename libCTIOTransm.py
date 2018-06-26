"""
libCTIOSTransm.py
=============----

author : Sylvie Dagoret-Campagne
affiliation : LAL/CNRS/IN2P3/FRANCE
Collaboration : DESC-LSST

Purpose : Provide the various transmission usefull
update : October 17 : set the path with environnment variables

"""

import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io import ascii

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

home = os.environ['HOME'] + '/'
path_CTIOtransm = 'CTIOThroughput'

#
filename_qe = "qecurve.txt"
filename_FGB37 = "FGB37.txt"
filename_RG715 = "RG175.txt"
filename_Throughput = 'ctio_throughput.txt'
filename_Total_Throughput = 'ctio_throughput_300517_v1.txt'
filename_mirrors = 'lsst_mirrorthroughput.txt'

WLMIN = 300.
WLMAX = 1100.


# -----------------------------------------------------------------------------
def Get_QE(path):
    filename = os.path.join(path, filename_qe)

    data_qe = ascii.read(filename)
    x = data_qe["col1"]
    y = data_qe["col2"] / 100.
    indexes = np.where(np.logical_and(x > WLMIN, x < WLMAX))
    return x[indexes], y[indexes]


# ------------------------------------------------------------------------------

def Get_RG715(path):
    filename = os.path.join(path, filename_RG715)

    data_rg = ascii.read(filename)
    x = data_rg["col2"]
    y = data_rg["col3"] / 100.
    indexes = np.where(np.logical_and(x > WLMIN, x < WLMAX))
    return x[indexes], y[indexes]


# ------------------------------------------------------------------------------
def Get_FGB37(path):
    filename = os.path.join(path, filename_FGB37)

    data_fgb = ascii.read(filename)
    x = data_fgb["col2"]
    y = data_fgb["col3"] / 100.
    indexes = np.where(np.logical_and(x > WLMIN, x < WLMAX))
    return x[indexes], y[indexes]


# -----------------------------------------------------------------------------
def Get_Throughput(path):
    filename = os.path.join(path, filename_Throughput)

    data_rg = ascii.read(filename)
    x = data_rg["col1"]
    y = data_rg["col2"]
    indexes = np.where(np.logical_and(x > WLMIN, x < WLMAX))
    return x[indexes], y[indexes]


# -----------------------------------------------------------------------------
def Get_Total_Throughput(path):
    filename = os.path.join(path, filename_Total_Throughput)

    data_rg = ascii.read(filename)
    x = data_rg["col1"]
    y = data_rg["col2"]
    z = data_rg["col3"]
    indexes = np.where(np.logical_and(x > WLMIN, x < WLMAX))
    return x[indexes], y[indexes], z[indexes]


# ---------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def Get_Mirror(path):
    filename = os.path.join(path, filename_mirrors)

    data_m = ascii.read(filename)
    x = data_m["col1"]
    y = data_m["col2"]
    indexes = np.where(np.logical_and(x > WLMIN, x < WLMAX))
    return x[indexes], y[indexes]


# ---------------------------------------------------------------
def PlotQE(path):
    wl, qe = Get_QE(path)

    plt.figure()
    plt.plot(wl, qe, "-")
    plt.title('Quantum efficiency')
    plt.xlabel("$\lambda$")
    plt.ylabel("QE")
    plt.xlim(WLMIN, WLMAX)


# ---------------------------------------------------------------
def PlotRG(path):
    wl, trg = Get_RG715(path)

    plt.figure()
    plt.plot(wl, trg, "-")
    plt.title('RG175')
    plt.xlabel("$\lambda$")
    plt.ylabel("transmission")
    plt.xlim(WLMIN, WLMAX)


# ---------------------------------------------------------------
def PlotFGB(path):
    wl, trg = Get_FGB37(path)

    plt.figure()
    plt.plot(wl, trg, "-")
    plt.title('FGB37')
    plt.xlabel("$\lambda$")
    plt.ylabel("transmission")
    plt.xlim(WLMIN, WLMAX)

    # ---------------------------------------------------------------


def PlotThroughput(path):
    wl, trt = Get_Throughput(path)

    plt.figure()
    plt.plot(wl, trt, "-")
    plt.title('throughput')
    plt.xlabel("$\lambda$")
    plt.ylabel("transmission")
    plt.xlim(WLMIN, WLMAX)

    # ---------------------------------------------------------------


def PlotMirror(path):
    wl, tr = Get_Mirror(path)

    plt.figure()
    plt.plot(wl, tr, "b-", label='1 mirror')
    plt.plot(wl, tr * tr, "r-", label='2 mirrors')
    plt.title('Mirror')
    plt.xlabel("$\lambda$")
    plt.ylabel("transmission")
    plt.legend()
    plt.xlim(WLMIN, WLMAX)


# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    mypath = os.path.dirname(__file__)
    datapath = os.path.join(mypath, "CTIOThroughput")

    PlotQE(datapath)

    PlotRG(datapath)

    PlotFGB(datapath)

    PlotThroughput(datapath)

    PlotMirror(datapath)

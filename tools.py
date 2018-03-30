import os, re
import scipy
from scipy.optimize import curve_fit
from scipy.misc import imresize
import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
import warnings

from skimage.feature import hessian_matrix

import parameters

def gauss(x,A,x0,sigma):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))

def gauss_and_line(x,a,b,A,x0,sigma):
    return gauss(x,A,x0,sigma) + line(x,a,b)

def line(x, a, b):
    return a*x + b

def parabola(x, a, b, c):
    return a*x*x + b*x + c

def fit_gauss(x,y,guess=[10,1000,1],bounds=(-np.inf,np.inf)):
    popt,pcov = curve_fit(gauss,x,y,p0=guess,bounds=bounds)
    return popt, pcov

def multigauss_and_line(x,*params):
    out = line(x,params[0],params[1])
    for k in range((len(params)-2)/3) :
        out += gauss(x,*params[2+3*k:2+3*k+3])
    return out

def fit_multigauss_and_line(x,y,guess=[10,1000,1,0,0,1],bounds=(-np.inf,np.inf)):
    maxfev=100000
    popt,pcov = curve_fit(multigauss_and_line,x,y,p0=guess,bounds=bounds,maxfev=maxfev)
    return popt, pcov

def multigauss_and_bgd(x,*params):
    bgd_nparams = parameters.BGD_NPARAMS
    out = np.polyval(params[0:bgd_nparams],x)
    for k in range((len(params)-bgd_nparams)/3) :
        out += gauss(x,*params[bgd_nparams+3*k:bgd_nparams+3*k+3])
    return out

def fit_multigauss_and_bgd(x,y,guess=[10,1000,1,0,0,1],bounds=(-np.inf,np.inf),sigma=None):
    maxfev=10000
    popt,pcov = curve_fit(multigauss_and_bgd,x,y,p0=guess,bounds=bounds,maxfev=maxfev,sigma=sigma)
    return popt, pcov


def fit_line(x,y,guess=[1,1],bounds=(-np.inf,np.inf),sigma=None):
    popt,pcov = curve_fit(line,x,y,p0=guess,bounds=bounds,sigma=sigma)
    return popt, pcov

def fit_bgd(x,y,guess=[1]*parameters.BGD_NPARAMS,bounds=(-np.inf,np.inf),sigma=None):
    bgd = lambda x, *p: np.polyval(p,x) 
    popt,pcov = curve_fit(bgd,x,y,p0=guess,bounds=bounds,sigma=sigma)
    return popt, pcov

def fit_poly(x,y,degree,w=None):
    cov = -1
    if(len(x)> order):
        if(w is None):
            fit, cov = np.polyfit(x,y,degree,cov=True)
        else:
            fit, cov = np.polyfit(x,y,degree,cov=True,w=w)
        model = lambda x : np.polyval(fit,x)
    else:
        fit = [0] * (degree+1)
        model = y
    return(fit,cov,model)

def fit_poly2d(x,y,z,degree):
    # Fit the data using astropy.modeling
    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)
        return p

def fit_poly1d_outlier_removal(x,y,order=2,sigma=3.0,niter=3):
    gg_init = models.Polynomial1D(order)
    gg_init.c0.min = np.min(y)
    gg_init.c0.max = 2*np.max(y)
    gg_init.c1 = 0
    gg_init.c2 = 0
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        filtered_data, or_fitted_model = or_fit(gg_init, x, y)
        return or_fitted_model

def fit_poly2d_outlier_removal(x,y,z,order=2,sigma=3.0,niter=30):
    gg_init = models.Polynomial2D(order)
    gg_init.c0_0.min = np.min(z)
    gg_init.c0_0.max = 2*np.max(z)
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        filtered_data, or_fitted_model = or_fit(gg_init, x, y, z)
        return or_fitted_model

def tied_circular_gauss2d(g1):
    std = g1.x_stddev
    return std

def fit_gauss2d_outlier_removal(x,y,z,sigma=3.0,niter=30,guess=None,bounds=None,circular=False):
    '''Gauss2D parameters: amplitude, x_mean,y_mean,x_stddev, y_stddev,theta'''
    gg_init = models.Gaussian2D()
    if guess is not None:
        for ip,p in enumerate(gg_init.param_names):
            getattr(gg_init,p).value = guess[ip]
    if bounds is not None:
        for ip,p in enumerate(gg_init.param_names):
            getattr(gg_init,p).min = bounds[0][ip]
            getattr(gg_init,p).max = bounds[1][ip]
    if circular:
        gg_init.y_stddev.tied = tied_circular_gauss2d
        gg_init.theta.fixed = True
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        filtered_data, or_fitted_model = or_fit(gg_init, x, y, z)
        return or_fitted_model

def fit_moffat2d_outlier_removal(x,y,z,sigma=3.0,niter=30,guess=None,bounds=None):
    '''Moffat2D parameters: amplitude, x_mean,y_mean,gamma,alpha'''
    gg_init = models.Moffat2D()
    if guess is not None:
        for ip,p in enumerate(gg_init.param_names):
            getattr(gg_init,p).value = guess[ip]
    if bounds is not None:
        for ip,p in enumerate(gg_init.param_names):
            getattr(gg_init,p).min = bounds[0][ip]
            getattr(gg_init,p).max = bounds[1][ip]
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        fit = fitting.LevMarLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
        # get fitted model and filtered data
        filtered_data, or_fitted_model = or_fit(gg_init, x, y, z)
        print or_fitted_model
        return or_fitted_model

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx,array[idx]

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
        

def resize_image(data,Nx_final,Ny_final):
    # F mode stands for 'float' : output is float and not int between 0 and 255
    data_resampled=scipy.misc.imresize(data,(Nx_final,Ny_final),mode='F')
    data_rescaled = data_resampled
    # following line is not useful with mode='F', uncomment it with mode='I'
    #data_rescaled=(np.max(data)-np.min(data))*(data_resampled-np.min(data_resampled))/(np.max(data_resampled)-np.min(data_resampled))+np.min(data)
    return(data_rescaled)


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    
    For example for the PSF
    
    x=pixel number
    y=Intensity in pixel
    
    values-x
    weights=y=f(x)
    
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))

def hessian_and_theta(data,margin_cut=1):
    # compute hessian matrices on the image
    Hxx, Hxy, Hyy = hessian_matrix(data, sigma=3, order='xy')
    lambda_plus = 0.5*( (Hxx+Hyy) + np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
    lambda_minus = 0.5*( (Hxx+Hyy) - np.sqrt( (Hxx-Hyy)**2 +4*Hxy*Hxy) )
    theta = 0.5*np.arctan2(2*Hxy,Hyy-Hxx)*180/np.pi
    # remove the margins
    lambda_minus = lambda_minus[margin_cut:-margin_cut,margin_cut:-margin_cut]
    lambda_plus = lambda_plus[margin_cut:-margin_cut,margin_cut:-margin_cut]
    theta = theta[margin_cut:-margin_cut,margin_cut:-margin_cut]
    return lambda_plus, lambda_minus, theta

def filter_stars_from_bgd(data,margin_cut=1):
    lambda_plus, lambda_minus, theta = hessian_and_theta(np.copy(data), margin_cut=1)
    # thresholds
    lambda_threshold = np.median(lambda_minus)-2*np.std(lambda_minus)
    mask = np.where(lambda_minus<lambda_threshold)
    data[mask]=np.nan
    return data


def extract_info_from_CTIO_header(obj,header):
    obj.date_obs = header['DATE-OBS']
    obj.airmass = header['AIRMASS']
    obj.expo = header['EXPTIME']
    obj.filters = header['FILTERS']
    obj.filter = header['FILTER1']
    obj.disperser = header['FILTER2']

    

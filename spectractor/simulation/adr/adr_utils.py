import matplotlib.pyplot as plt
import astropy.coordinates as AC
from astropy import units as u
import spectractor.simulation.adr.adr as ADR
import numpy as np

#from spectractor import parameters

"""
Functions to:
  - get adr object from fits file
  - get list of shift due to adr corresponding to a list of lambdas
  - get maximum shift due to adr between (in th range of lambdas given)
  - plot the adr along both axis and the biggest one against the wavelength
Example to get shift in pixels:
  instanciation_adr(fitsfile, latitude, lbda_ref)
  xs, ys = get_adr_shift_for_lbdas(adr_object, lbdas)
  xs_pix, ys_pix = in_pixel(xs, fitsfile), adru.in_pixel(ys, fitsfile)
"""

__author__ = "Maxime Rey"

#adapted by Vincent Brémaud


# ================================= #
#          Main functions           #
# ================================= #


def adr_calib(lambdas,params,lat,lambda_ref=550):

    if isinstance(lat,str):
        lat=AC.Latitude(lat, unit=u.deg)
    elif isinstance(lat,AC):
        lat=lat
    else:
        raise TypeError('latitude type is neither a str nor an astropy.coordinates')

    meadr = instanciation_adr(params, lat, lambdas[0]*10)

    xs, ys= get_adr_shift_for_lbdas(meadr, lambdas*10, params)
    xs_pix = in_pixel(xs, params)

    indice_ref=min(len(lambdas), np.argmin(np.abs(lambdas - lambda_ref)))
    x_0=xs_pix[indice_ref] #550 nm

    x_shift=x_0-xs_pix

    return x_shift


def instanciation_adr(params,latitude, lbda_ref):

    dec,hour_angle=params[:2]

    if isinstance(dec, str) and isinstance(hour_angle, str):
        dec = AC.Angle(params[0], unit=u.deg)
        hour_angle = AC.Angle(params[1], unit=u.hourangle)

    elif isinstance(dec, AC) and isinstance(hour_angle, AC):
        dec = dec
        hour_angle = hour_angle
    else:
        raise TypeError('dec/hour_angle type is neither a str nor an astropy.coordinates')

    temperature,pressure,humidity,airmass,rotangle=params[2:-2]

    _, parangle = ADR.hadec2zdpar(hour_angle.degree, dec.degree, latitude.degree, deg=True)
    adr = ADR.ADR(airmass=airmass, parangle=parangle, temperature=temperature,
    pressure=pressure, lbdaref=lbda_ref, relathumidity=humidity)
  
    return adr


def get_adr_shift_for_lbdas(adr_object, lbdas, params):
  """
  Returns shift in x and y due to adr as arrays in arcsec.
  """

  arcsecshift = adr_object.refract(0, 0, lbdas, params[-3])

  x_shift = (arcsecshift[0])
  y_shift = (arcsecshift[1])

  return x_shift, y_shift


# ================================= #
#        Utilitary functions        #
# ================================= #


def in_pixel(thing_in_arcsec, params):
  """
  Transform something in arcsec in pixels
  """

  xpixsize=params[-2]
  ypixsize=params[-1]

  if xpixsize!=ypixsize:
    raise ValueError('Pixels in X and Y do not have the same length, too complicated, did not work')


  thing_in_pix = thing_in_arcsec/xpixsize

  return thing_in_pix





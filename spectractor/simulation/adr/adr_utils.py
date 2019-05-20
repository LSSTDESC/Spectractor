import matplotlib.pyplot as plt
import astropy.coordinates as AC
from astropy import units as u
import astropy.io.fits as F

import adr as ADR

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


# ================================= #
#          Main functions           #
# ================================= #


def instanciation_adr(fitsfile, latitude, lbda_ref):
  """
  Returns an ADR object (from the ADR library, don't get confused between both).
  hdr should be a dict composed of the keynames of the header and their associated value and the latitude should be geodetic.
  By default, the latitude is from CTIO, otherwise replace with astropy.coordinates.Latitude object.
  """ 

  hdu = F.open(fitsfile) 
  hdr = hdu[0].header

  test_key_in_hdr(hdr, 'DEC')
  test_key_in_hdr(hdr, 'HA')
  test_key_in_hdr(hdr, 'OUTTEMP')
  test_key_in_hdr(hdr, 'OUTPRESS')
  test_key_in_hdr(hdr, 'OUTHUM')
  test_key_in_hdr(hdr, 'AIRMASS')


  dec = AC.Angle(hdr['DEC'], unit=u.deg)
  hour_angle = AC.Angle(hdr['HA'], unit=u.hourangle)

  temperature = hdr['OUTTEMP']                          # outside temp (C) 
  pressure = hdr['OUTPRESS']                            # outside pressure (mbar)
  humidity = hdr['OUTHUM']                              # outside humidity (%)
  airmass = hdr['AIRMASS']                              # airmass    

  _, parangle = ADR.hadec2zdpar(hour_angle.degree, dec.degree, latitude.degree, deg=True)
  adr = ADR.ADR(airmass=airmass, parangle=parangle, temperature=temperature,
    pressure=pressure, lbdaref=lbda_ref, relathumidity=humidity)

  return adr


def get_adr_shift_for_lbdas(adr_object, lbdas):
  """
  Returns shift in x and y due to adr as arrays in arcsec.
  """

  arcsecshift = adr_object.refract(0, 0, lbdas)
  
  x_shift = (arcsecshift[0])
  y_shift = (arcsecshift[1])

  return x_shift, y_shift


def get_max_shift_for_lbdas(adr_object, lbdas):
  """
  Returns absolute maximum shift in x and y due to adr in arcsec.
  """

  arcsecshift = adr_object.refract(0, 0, lbdas)
  
  x_shift = (arcsecshift[0,-1]-arcsecshift[0,0])
  y_shift = (arcsecshift[1,-1]-arcsecshift[1,0])

  return x_shift, y_shift


def plot_adr(lbdas, x_shift, y_shift, units='arcsec'):
  """
  Plot adr in x against y (arcsec) and in wavelength against the strongest shift (x or y).
  Of course lbdas has to be a list with the same size as the x and y shift.
  """

  test_size_2list(x_shift, y_shift)
  try:
    test_size_2list(x_shift, lbdas)
  except ValueError:   ############################################# Autre moyen de faire ça (remplacer texte de l'erreur) ? M'a pas l'air intelligent mais ai pas trouvé mieux
    raise ValueError('You used an array of array, didn\'t you ? \n Please select an array of values')             # as e and e.args += machin mais si pas +, ça split par lettre

  xshift = abs(x_shift[-1]-x_shift[0])
  yshift = abs(y_shift[-1]-y_shift[0])
  shift_xminusy = xshift - yshift

  if shift_xminusy>0:                        # returns axis with biggest shift
      bigshiftaxis = x_shift
      title2 = 'Shift in lambdas against x'
  else:
      bigshiftaxis = y_shift
      title2 = 'Shift in lambdas against y'

  _, [ax1, ax2] = plt.subplots(nrows=1, ncols=2) #################################### Why often see fig, axes et pas _, axes ?

  ax1.plot(x_shift, y_shift)
  ax1.set_xlabel('centered x shift in ' + units)
  ax1.set_ylabel('centered y shift in ' + units)
  ax1.set_title('Shift in x against y')

  ax2.plot(lbdas, bigshiftaxis)
  ax2.set_xlabel('wavelength (A)')
  ax2.set_ylabel('biggest shift in ' + units)
  ax2.set_title(title2)

  plt.tight_layout()
  plt.show()


# ================================= #
#        Utilitary functions        #
# ================================= #


def in_pixel(thing_in_arcsec, fitsfile):
  """
  Transform something in arcsec in pixels
  """

  hdu = F.open(fitsfile) 
  hdr = hdu[0].header

  test_key_in_hdr(hdr, 'XPIXSIZE')
  test_key_in_hdr(hdr, 'YPIXSIZE')

  if hdr['XPIXSIZE']!=hdr['YPIXSIZE']:
    raise ValueError('Pixels in X and Y do not have the same length, too complicated, did not work')


  thing_in_pix = thing_in_arcsec/hdr['XPIXSIZE']

  return thing_in_pix


def get_lat_tel(telescope):
  """
  Returns the latitude of the telescope given as a string.
  """

  if telescope=='CTIO':
    geod_lat = AC.Latitude('-30:10:07.90', unit=u.deg)     #From CTIO doc: http://www.ctio.noao.edu/noao/node/2085
    geod_lon = AC.Angle('-70:48:23.86', unit=u.deg)
    return geod_lat, geod_lon
  elif telescope=='Auxtel':
    raise ValueError('not documented yet (you can add the value in the code')




# ================================= #
#          Test functions           #
# ================================= #


def test_size_2list(list1, list2): ############# How do I return the name of the list ?
  """
  Check that 2 lists have the same size.
  """

  if len(list1)!=len(list2):
    raise ValueError("{} and {} don't have the same length".format(str(list1), str(list2)))


def test_key_in_hdr(dict,key):
  """
  Check that keys are indeed in a dictionnary.
  """

  try:
    dict[key]
  except KeyError:
    raise KeyError('{} is not contained in the fits files and is necessary'.format(key))
#! /usr/bin/env python
# -*- coding: utf-8 -*-


import astropy.coordinates as AC
from astropy import units as u
import numpy as np
from spectractor import parameters
from spectractor.tools import flip_and_rotate_radec_to_image_xy_coordinates

"""
Atmospheric Differential Refraction: Evolution of the spatial position as a function of wavelength.
Credit: Y.Copin (y.copin@ipnl.in2p3.fr)
Adapted by: M. Rigault
Inspiration: SNfactory ToolBox (see also Don Neill's Fork of KPY)
Original Source: http://emtoolbox.nist.gov/Wavelength/Documentation.asp
"""


class ADR:

    def __init__(self, airmass, lbdaref, pressure, parangle, relathumidity, temperature):

        self.airmass = airmass
        self.lbdaref = lbdaref
        self.pressure = pressure
        self.parangle = parangle
        self.relathumidity = relathumidity
        self.temperature = temperature

    # =================== #
    #   Methods           #
    # =================== #
    def refract(self, x, y, lbda, backward=False, unit=1.):
        """If forward (default), return refracted position(s) at
        wavelength(s) *lbda* [Å] from reference position(s) *x*,*y*
        (in units of *unit* in arcsec).  Return shape is
        (2,[nlbda],[npos]), where nlbda and npos are the number of
        input wavelengths and reference positions.
        If backward, one should have `len(x) == len(y) ==
        len(lbda)`. Return shape is then (2,npos).
        Coordinate *x* is counted westward, and coordinate *y* counted
        northward (standard North-up, Est-left orientation).
        Anonymous `kwargs` will be propagated to :meth:`set`.
        """

        x0 = np.atleast_1d(x)  # (npos,)
        y0 = np.atleast_1d(y)
        if np.shape(x0) != np.shape(y0):
            raise TypeError("x and y do not have the same shape")

        npos = len(x0)

        dz = np.tan(np.arccos(1. / self.airmass)) * self.get_scale(lbda) / unit  # [unit]
        if backward:
            nlbda = len(np.atleast_1d(lbda))
            assert npos == nlbda, "Incompatible x,y and lbda vectors."
            x = x0 - dz * np.sin(self.parangle / 180. * np.pi)
            y = y0 + dz * np.cos(self.parangle / 180. * np.pi)  # (nlbda=npos,)
            out = np.vstack((x, y))  # (2,npos)
        else:
            dz = dz[:, np.newaxis]  # (nlbda,1)
            x = x0 + dz * np.sin(self.parangle / 180. * np.pi)  # (nlbda,npos)
            y = y0 - dz * np.cos(self.parangle / 180. * np.pi)  # (nlbda,npos)
            out = np.dstack((x.T, y.T)).T  # (2,nlbda,npos)
        return out.squeeze()  # (2,[nlbda],[npos])

    # ------- #
    # GETTER  #
    # ------- #
    def get_scale(self, lbda):
        """
        Return ADR scale [arcsec] for wavelength(s) `lbda` [A].
        Anonymous `kwargs` will be propagated to :meth:`set`.
        """

        lbda = np.atleast_1d(lbda)  # (nlbda,)

        # Approximate ADR to 1st order in (n - 1). The difference
        # between exact expression and 1st-order approximation reaches
        # 4e-9 at 3200 A.
        # dz = self.nref - \
        #      refractiveIndex(lbda, P=self.P, T=self.T, RH=self.RH)

        # Exact ADR expression
        dz = (self.get_refractive_index(lbda) ** -2 - self.get_refractive_index(self.lbdaref) ** -2) * 0.5

        return dz * 180. * 3600 / np.pi  # (nlbda,) [arcsec]

    def get_refractive_index(self, lbda):
        """ relative index for the given wavelength.
        Anonymous `kwargs` will be propagated to :meth:`set`.
        """

        return refractive_index(lbda, self.pressure, self.temperature, self.relathumidity)


##############################
#                            #
#   General Functions        #
#                            #
##############################

def refractive_index(lbda, pressure=617., temperature=2., relathumidity=0):
    """Compute refractive index at vacuum wavelength.
    source: NIST/IAPWS via SNfactory ToolBox
            http://emtoolbox.nist.gov/Wavelength/Documentation.asp
    Parameters:
    -----------
    lbda: [float / array of]
        wavelength in [Å]

    pressure: [float] -optional-
        air pressure in mbar
    temperature: [float] -optional-
        air temperature in Celcius
    relathumidity: [float] -optional-
        air relative humidity in percent.
        [the water vapor pressure will be derived from it,
        according to (modified) Edlén Calculation of the Index of
        Refraction from NIST 'Refractive Index of Air Calculator'.  CO2
        concentration is fixed to 450 µmol/mol.]
    Returns
    -------
    float/array (depending on lbda input)
    """

    A = 8342.54
    B = 2406147.
    C = 15998.
    D = 96095.43
    E = 0.601
    F = 0.00972
    G = 0.003661

    iml2 = (lbda * 1e-4) ** -2  # 1/(lambda in microns)**2
    nsm1e2 = 1e-6 * (A + B / (130. - iml2) + C / (38.9 - iml2))  # (ns - 1)*1e2
    # P in mbar = 1e2 Pa
    X = (1. + 1e-6 * (E - F * temperature) * pressure) / (1. + G * temperature)
    n = 1. + pressure * nsm1e2 * X / D  # ref. index corrected for P,T

    if relathumidity:  # Humidity correction
        pv = relathumidity / 100. * saturation_vapor_pressure(temperature)  # [Pa]
        n -= 1e-10 * (292.75 / (temperature + 273.15)) * (3.7345 - 0.0401 * iml2) * pv

    return n


def saturation_vapor_pressure(temperature):
    """Compute saturation vapor pressure [Pa] for temperature *T* [°C]
    according to Edlén Calculation of the Index of Refraction from
    NIST 'Refractive Index of Air Calculator'.
    Source: http://emtoolbox.nist.gov/Wavelength/Documentation.asp
    """

    t = np.atleast_1d(temperature)
    psv = np.where(t >= 0,
                   _saturationVaporPressureOverWater(temperature),
                   _saturationVaporPressureOverIce(temperature))

    return psv  # [Pa]


def _saturationVaporPressureOverIce(temperature):
    """See :func:`saturation_vapor_pressure`"""

    A1 = -13.928169
    A2 = 34.7078238
    th = (temperature + 273.15) / 273.16
    Y = A1 * (1 - th ** -1.5) + A2 * (1 - th ** -1.25)
    psv = 611.657 * np.exp(Y)

    return psv


def _saturationVaporPressureOverWater(temperature):
    """See :func:`saturation_vapor_pressure`"""

    K1 = 1.16705214528e+03
    K2 = -7.24213167032e+05
    K3 = -1.70738469401e+01
    K4 = 1.20208247025e+04
    K5 = -3.23255503223e+06
    K6 = 1.49151086135e+01
    K7 = -4.82326573616e+03
    K8 = 4.05113405421e+05
    K9 = -2.38555575678e-01
    K10 = 6.50175348448e+02

    t = temperature + 273.15  # °C → K
    x = t + K9 / (t - K10)
    A = x ** 2 + K1 * x + K2
    B = K3 * x ** 2 + K4 * x + K5
    C = K6 * x ** 2 + K7 * x + K8
    X = -B + np.sqrt(B ** 2 - 4 * A * C)
    psv = 1e6 * (2 * C / X) ** 4

    return psv


########################
#
#  Conversions         #
#
########################
RAD2DEG = 180. / np.pi


def hadec2zdpar(ha, dec, lat, deg=True):
    """
    Conversion of equatorial coordinates *(ha, dec)* (in degrees if *deg*) to
    zenithal distance and parallactic angle *(zd, par)* (in degrees if *deg*),
    for a given geodetic latitude *lat* (in degrees if *deg*).
    .. Author: Y. Copin (y.copin@ipnl.in2p3.fr)
    """
    if deg:  # Convert to radians
        ha = ha / RAD2DEG
        dec = dec / RAD2DEG
        lat = lat / RAD2DEG

    cha, sha = np.cos(ha), np.sin(ha)
    cdec, sdec = np.cos(dec), np.sin(dec)
    clat, slat = np.cos(lat), np.sin(lat)

    sz_sp = clat * sha
    sz_cp = slat * cdec - clat * cha * sdec
    cz = slat * sdec + clat * cha * cdec

    sz, p = rec2pol(sz_cp, sz_sp)
    r, z = rec2pol(cz, sz)

    assert np.allclose(r, 1), "Precision error"

    if deg:
        z *= RAD2DEG
        p *= RAD2DEG

    return z, p


def rec2pol(x, y, deg=False):
    """
    Conversion of rectangular *(x, y)* to polar *(r, theta)* coordinates.
    .. Author: Y. Copin (y.copin@ipnl.in2p3.fr)
    """
    r = np.hypot(x, y)
    t = np.arctan2(y, x)
    if deg:  # Convert to radians
        t *= RAD2DEG

    return r, t


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


# ================================= #
#          Main functions           #
# ================================= #


def adr_calib(lambdas, params, lat, lambda_ref=550):
    if isinstance(lat, str) or isinstance(lat, float):
        lat = AC.Latitude(lat, unit=u.deg)
    elif isinstance(lat, AC.Latitude):
        lat = lat
    else:
        raise TypeError('Latitude type is neither a str, float nor an astropy.coordinates')

    meadr = instanciation_adr(params, lat, lambda_ref * 10)

    disp_axis, trans_axis = get_adr_shift_for_lbdas(meadr, lambdas * 10)
    disp_axis_pix = in_pixel(disp_axis)
    trans_axis_pix = in_pixel(trans_axis)

    return disp_axis_pix, trans_axis_pix


def instanciation_adr(params, latitude, lbda_ref):
    dec, hour_angle = params[:2]

    if (isinstance(dec, str) and isinstance(hour_angle, str)) \
            or (isinstance(dec, float) and isinstance(hour_angle, float)):
        dec = AC.Angle(params[0], unit=u.deg)
        hour_angle = AC.Angle(params[1], unit=u.hourangle)
    elif isinstance(dec, AC.Angle) and isinstance(hour_angle, AC.Angle):
        dec = dec
        hour_angle = hour_angle
    else:
        raise TypeError('dec/hour_angle type is neither a str nor an astropy.coordinates')

    temperature, pressure, humidity, airmass = params[2:]

    _, parangle = hadec2zdpar(hour_angle.degree, dec.degree, latitude.degree, deg=True)
    adr = ADR(airmass=airmass, parangle=parangle, temperature=temperature,
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


# ================================= #
#        Utilitary functions        #
# ================================= #


def in_pixel(thing_in_arcsec):
    """
    Transform something in arcsec in pixels
    """

    xpixsize = parameters.CCD_PIXEL2ARCSEC
    ypixsize = parameters.CCD_PIXEL2ARCSEC

    if xpixsize != ypixsize:
        raise ValueError('Pixels in X and Y do not have the same length, too complicated, did not work')

    thing_in_pix = thing_in_arcsec / xpixsize

    return thing_in_pix


def flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=0):
    """Flip and rotate the ADR shifts in pixels along (RA,DEC) directions to (x, y) image coordinates.

    Parameters
    ----------
    adr_ra: array_like
        ADR shift in pixel along the RA direction.
    adr_dec: array_like
        ADR shift in pixel along the DEC direction.
    dispersion_axis_angle: float, optional
        Optional additional angle of the dispersion axis in the (x,y) frame (default: 0).

    Returns
    -------
    adr_x: array_like
        ADR shift in pixel along the x direction.
    adr_y: array_like
        ADR shift in pixel along the y direction.

    Examples
    --------

    >>> from spectractor.extractor.spectrum import Spectrum
    >>> spec = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits", config="./config/ctio.ini")

    Compute ADR in (RA, DEC) frame

    >>> adr_ra, adr_dec = adr_calib(spec.lambdas, spec.adr_params, lat=parameters.OBS_LATITUDE, lambda_ref=550)

    Compute ADR in (x, y) frame

    >>> adr_x, adr_y = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=0)
    >>> assert np.all(np.isclose(adr_x, adr_ra))
    >>> assert np.all(np.isclose(adr_y, -adr_dec))

    Compute ADR in (u, v) spectrogram frame

    >>> adr_u, adr_v = flip_and_rotate_adr_to_image_xy_coordinates(adr_ra, adr_dec, dispersion_axis_angle=-1.54)
    >>> assert adr_x[0] < adr_u[0]
    >>> assert adr_y[0] < adr_v[0]

    """
    adr_x, adr_y = flip_and_rotate_radec_to_image_xy_coordinates(adr_ra, adr_dec,
                                                                 camera_angle=parameters.OBS_CAMERA_ROTATION,
                                                                 flip_ra_sign=parameters.OBS_CAMERA_RA_FLIP_SIGN,
                                                                 flip_dec_sign=parameters.OBS_CAMERA_DEC_FLIP_SIGN)
    if not np.isclose(dispersion_axis_angle, 0, atol=0.001):
        # minus sign as rotation matrix is apply on the right on the adr vector
        a = - dispersion_axis_angle * np.pi / 180
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=float)
        adr_u, adr_v = (np.asarray([adr_x, adr_y]).T @ rotation).T
        return adr_u, adr_v
    else:
        return adr_x, adr_y

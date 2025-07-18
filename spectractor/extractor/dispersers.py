from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import os

from spectractor import parameters
from spectractor.tools import fit_poly2d
from spectractor.logbook import set_logger
from spectractor.config import from_config_to_dict


def build_hologram(order0_position, order1_position, theta_tilt=0, D=parameters.DISTANCE2CCD, lambda_plot=256000):
    """Produce the interference pattern printed on a hologram, with two sources
    located at order0_position and order1_position, with an angle theta_tilt with respect
    to the X axis. For plotting reasons, the wavelength can be set very large with
    the lambda_plot parameter.

    Parameters
    ----------
    order0_position: array
        List [x0,y0] of the pixel coordinates of the order 0 source position (source A).
    order1_position: array
        List [x1,y1] of the pixel coordinates of the order 1 source position (source B).
    theta_tilt: float
        Angle (in degree) to tilt the interference pattern with respect to X axis (default: 0)
    D: float
    lambda_plot: float
        Wavelength to produce the interference pattern (default: 256000)

    Returns
    -------
    hologram: 2D-array,
        The hologram figure, of shape (CCD_IMSIZE,CCD_IMSIZE)

    Examples
    --------
    >>> hologram = build_hologram([500,500],[800,500],theta_tilt=-1,lambda_plot=200000)
    >>> assert np.all(np.isclose(hologram[:5,:5],np.zeros((5,5))))
    """
    # wavelength in nm, hologram produced at 639nm
    # spherical wave centered in 0,0,0
    U = lambda x, y, z: np.exp(2j * np.pi * np.sqrt(x * x + y * y + z * z) * 1e6 /
                               lambda_plot) / np.sqrt(x * x + y * y + z * z)
    # superposition of two spherical sources centered in order 0 and order 1 positions
    xA = [order0_position[0] * parameters.CCD_PIXEL2MM, order0_position[1] * parameters.CCD_PIXEL2MM]
    xB = [order1_position[0] * parameters.CCD_PIXEL2MM, order1_position[1] * parameters.CCD_PIXEL2MM]
    A = lambda x, y: U(x - xA[0], y - xA[1], -D) + U(x - xB[0], y - xB[1], -D)
    intensity = lambda x, y: np.abs(A(x, y)) ** 2
    xholo = np.linspace(0, parameters.CCD_IMSIZE * parameters.CCD_PIXEL2MM, parameters.CCD_IMSIZE)
    yholo = np.linspace(0, parameters.CCD_IMSIZE * parameters.CCD_PIXEL2MM, parameters.CCD_IMSIZE)
    xxholo, yyholo = np.meshgrid(xholo, yholo)
    holo = intensity(xxholo, yyholo)
    rotated_holo = ndimage.rotate(holo, theta_tilt)
    return rotated_holo


def build_ronchi(x_center, theta_tilt=0, grooves=400):
    """Produce the Ronchi pattern (alternance of recatngular stripes of transparancy 0 and 1),
    centered at x_center, with an angle theta_tilt with respect
    to the X axis. Grooves parameter set the number of grooves per mm.

    Parameters
    ----------
    x_center: float
        Center pixel to start the figure with a black stripe.
    theta_tilt: float
        Angle (in degree) to tilt the interference pattern with respect to X axis (default: 0)
    grooves: float
        Number of grooves per mm (default: 400)

    Returns
    -------
    hologram: 2D-array,
        The hologram figure, of shape (CCD_IMSIZE,CCD_IMSIZE)

    Examples
    --------
    >>> ronchi = build_ronchi(0,theta_tilt=0,grooves=400)
    >>> print(ronchi[:5,:5])
    [[0 1 0 0 1]
     [0 1 0 0 1]
     [0 1 0 0 1]
     [0 1 0 0 1]
     [0 1 0 0 1]]

    """
    intensity = lambda x, y: 2 * np.sin(2 * np.pi *
                                        (x - x_center * parameters.CCD_PIXEL2MM) * 0.5 * grooves) ** 2
    xronchi = np.linspace(0, parameters.CCD_IMSIZE * parameters.CCD_PIXEL2MM, parameters.CCD_IMSIZE)
    yronchi = np.linspace(0, parameters.CCD_IMSIZE * parameters.CCD_PIXEL2MM, parameters.CCD_IMSIZE)
    xxronchi, yyronchi = np.meshgrid(xronchi, yronchi)
    ronchi = (intensity(xxronchi, yyronchi)).astype(int)
    rotated_ronchi = ndimage.rotate(ronchi, theta_tilt)
    return rotated_ronchi


def get_theta0(x0):
    """ Return the incident angle on the disperser in radians, with respect to the disperser normal and the X axis.

    Parameters
    ----------
    x0: float, tuple, list
        The order 0 position in the full non-rotated image.

    Returns
    -------
    theta0: float
        The incident angle in radians

    Examples
    --------
    >>> get_theta0((parameters.CCD_IMSIZE/2,parameters.CCD_IMSIZE/2))
    0.0
    >>> get_theta0(parameters.CCD_IMSIZE/2)
    0.0
    """
    pix_to_rad = parameters.CCD_PIXEL2ARCSEC * np.pi / (180. * 3600.)
    if isinstance(x0, (list, tuple, np.ndarray)):
        return (x0[0] - parameters.CCD_IMSIZE / 2) * pix_to_rad
    else:
        return (x0 - parameters.CCD_IMSIZE / 2) * pix_to_rad


def get_delta_pix_ortho(deltaX, x0, D):
    """ Subtract from the distance deltaX in pixels between a pixel x the order 0 the distance between
    the projected incident point on the disperser and the order 0. In other words, the projection of the incident
    angle theta0 from the disperser to the CCD is removed. The distance to the CCD D is in mm.

    Parameters
    ----------
    deltaX: float
        The distance in pixels between the order 0 and a spectrum pixel in the rotated image.
    x0: list, [x0,y0]
        The order 0 position in the full non-rotated image.
    D: float
        The distance between the CCD and the disperser in mm.

    Returns
    -------
    distance: float
        The projected distance in pixels

    Examples
    --------
    >>> from spectractor.config import load_config
    >>> load_config("default.ini")
    >>> delta, D = 500, 55
    >>> get_delta_pix_ortho(delta, [parameters.CCD_IMSIZE/2,  parameters.CCD_IMSIZE/2], D=D)
    500.0
    >>> get_delta_pix_ortho(delta, [500,500], D=D)
    497.6654556732099
    """
    theta0 = get_theta0(x0)
    deltaX0 = np.tan(theta0) * D / parameters.CCD_PIXEL2MM
    return deltaX + deltaX0


def get_refraction_angle(deltaX, x0, D):
    """ Return the refraction angle with respect to the disperser normal, using geometrical consideration.

    Parameters
    ----------
    deltaX: float
        The distance in pixels between the order 0 and a spectrum pixel in the rotated image.
    x0: list, [x0,y0]
        The order 0 position in the full non-rotated image.
    D: float
        The distance between the CCD and the disperser in mm.

    Returns
    -------
    theta: float
        The refraction angle in radians.

    Examples
    --------
    >>> delta, D = 500, 55
    >>> theta = get_refraction_angle(delta, [parameters.CCD_IMSIZE/2,  parameters.CCD_IMSIZE/2], D=D)
    >>> assert np.isclose(theta, np.arctan2(delta*parameters.CCD_PIXEL2MM, D))
    >>> theta = get_refraction_angle(delta, [500,500], D=D)
    >>> print('{:.2f}'.format(theta))
    0.21
    """
    delta = get_delta_pix_ortho(deltaX, x0, D=D)
    theta = np.arctan2(delta * parameters.CCD_PIXEL2MM, D)
    return theta


def get_N(deltaX, x0, D, wavelength=656, order=1):
    """ Return the grooves per mm number given the spectrum pixel x position with
    its wavelength in mm, the distance to the CCD in mm and the order number. It
    uses the disperser formula.

    Parameters
    ----------
    deltaX: float
        The distance in pixels between the order 0 and a spectrum pixel in the rotated image.
    x0: list, [x0,y0]
        The order 0 position in the full non-rotated image.
    D: float
        The distance between the CCD and the disperser in mm.
    wavelength: float
        The wavelength at pixel x in nm (default: 656).
    order: int
        The order of the spectrum (default: 1).

    Returns
    -------
    theta: float
        The number of grooves per mm.

    Examples
    --------
    >>> delta, D, w = 500, 55, 600
    >>> N = get_N(delta, [500,500], D=D, wavelength=w, order=1)
    >>> print('{:.0f}'.format(N))
    355
    """
    theta = get_refraction_angle(deltaX, x0, D=D)
    theta0 = get_theta0(x0)
    N = (np.sin(theta) - np.sin(theta0)) / (order * wavelength * 1e-6)
    return N


def neutral_lines(x_center, y_center, theta_tilt):
    """Return the neutral lines of a hologram."""
    xs = np.linspace(0, parameters.CCD_IMSIZE, 20)
    line1 = np.tan(theta_tilt * np.pi / 180) * (xs - x_center) + y_center
    line2 = np.tan((theta_tilt + 90) * np.pi / 180) * (xs - x_center) + y_center
    return xs, line1, line2


def order01_positions(holo_center, N, theta_tilt, theta0=0, D=parameters.DISTANCE2CCD, lambda_constructor=639e-6, verbose=True):  # pragma: no cover
    """Return the order 0 and order 1 positions of a hologram."""
    # refraction angle between order 0 and order 1 at construction
    alpha = np.arcsin(N * lambda_constructor + np.sin(theta0))
    # distance between order 0 and order 1 in pixels
    AB = (np.tan(alpha) - np.tan(theta0)) * D / parameters.CCD_PIXEL2MM
    # position of order 1 in pixels
    x_center = holo_center[0]
    y_center = holo_center[1]
    order1_position = [0.5 * AB * np.cos(theta_tilt * np.pi / 180) + x_center,
                       0.5 * AB * np.sin(theta_tilt * np.pi / 180) + y_center]
    # position of order 0 in pixels
    order0_position = [-0.5 * AB * np.cos(theta_tilt * np.pi / 180) + x_center,
                       -0.5 * AB * np.sin(theta_tilt * np.pi / 180) + y_center]
    if verbose:
        my_logger = set_logger(__name__)
        my_logger.info(f'\n\tOrder  0 position at x0 = {order0_position[0]:.1f} and y0 = {order0_position[1]:.1f}'
                       f'\n\tOrder +1 position at x0 = {order1_position[0]:.1f} and y0 = {order1_position[1]:.1f}'
                       f'\n\tDistance between the orders: {AB:.2f} pixels ({AB * parameters.CCD_PIXEL2MM:.2f} mm)')
    return order0_position, order1_position, AB


def find_order01_positions(holo_center, N_interp, theta_interp, lambda_constructor=639e-6, verbose=True):  # pragma: no cover
    """Find the order 0 and order 1 positions of a hologram."""
    N = N_interp(holo_center)
    theta_tilt = theta_interp(holo_center)
    theta0 = 0
    convergence = 0
    while abs(N - convergence) > 1e-6:
        order0_position, order1_position, AB = order01_positions(holo_center, N, theta_tilt, theta0,
                                                                 lambda_constructor=lambda_constructor, verbose=False)
        convergence = np.copy(N)
        N = N_interp(order0_position)
        theta_tilt = theta_interp(order0_position)
        theta0 = get_theta0(order0_position)
    order0_position, order1_position, AB = order01_positions(holo_center, N, theta_tilt, theta0,
                                                             lambda_constructor=lambda_constructor, verbose=verbose)
    return order0_position, order1_position, AB


class Disperser:
    """Generic class for dispersers."""

    def __init__(self, N=-1, label="", data_dir=parameters.DISPERSER_DIR):
        """Initialize a standard grating object.

        Parameters
        ----------
        N: float
            The number of grooves per mm of the grating (default: -1)
        label: str
            String label for the grating (default: '')
        data_dir: str
            The directory where information about this disperser is stored. If relative, then the starting point is the
            installation package directory spectractor/. If absolute, it is taken as it is.
            (default: parameters.DISPERSER_DIR)

        Examples
        --------
        >>> g = Disperser(N=400)
        >>> print(g.N_input)
        400
        >>> g = Disperser(N=400, label="Ron400", data_dir=parameters.DISPERSER_DIR)
        >>> print(f"{g.N_input:6f}")
        400.869182

        Hologram case

        >>> h = Hologram(label='HoloPhP')
        >>> np.round(h.N((500,500)), 3)
        345.479
        >>> np.round(h.theta((700,700)), 3)
        -0.834
        >>> h.center
        [856.004, 562.34]

        """
        self.my_logger = set_logger(self.__class__.__name__)
        if N <= 0 and label == '':
            raise ValueError("Set either N grooves per mm or the grating label.")
        self.is_hologram = False
        self.center = [0.5 * parameters.CCD_IMSIZE, 0.5 * parameters.CCD_IMSIZE]
        self.N_input = N
        self.N_err = 1
        self.label = label
        self.full_name = label
        if os.path.isabs(data_dir):
            self.data_dir = data_dir
        else:
            mypath = os.path.dirname(os.path.dirname(__file__))
            self.data_dir = os.path.join(mypath, parameters.DISPERSER_DIR)
        self.theta_tilt = 0

        # transmissions
        ones = np.ones_like(parameters.LAMBDAS).astype(float)
        self.transmission = interpolate.interp1d(parameters.LAMBDAS, ones, bounds_error=False, fill_value=0.)
        self.transmission_err = interpolate.interp1d(parameters.LAMBDAS, 0 * ones, bounds_error=False, fill_value=0.)
        ratio = parameters.GRATING_ORDER_2OVER1 * np.ones_like(parameters.LAMBDAS).astype(float)
        self.ratio_order_2over1 = interpolate.interp1d(parameters.LAMBDAS, ratio, bounds_error=False, kind="linear",
                                                       fill_value="extrapolate")  # "(0, t[-1]))
        self.ratio_order_3over2 = None
        self.ratio_order_3over1 = None
        self.flat_ratio_order_2over1 = True

        # N and theta interp grids
        self.N_x = np.arange(0, parameters.CCD_IMSIZE)
        self.N_y = np.arange(0, parameters.CCD_IMSIZE)
        self.N_interp = self.N_flat
        self.N_fit = self.N_flat
        self.theta_interp = self.theta_flat
        self.D = None

        if self.label != "":
            if os.path.isfile(os.path.join(self.data_dir, self.label, self.label+'.ini')):
                # should be the default in near future
                self.load_config(path=os.path.join(self.data_dir, self.label, self.label+'.ini'))
            else:
                # going obsolete
                self.load_files()

        if self.is_hologram:
            self.x_lines, self.line1, self.line2 = neutral_lines(self.center[0], self.center[1], self.theta_tilt)
        self.my_logger.info(f"\n\t{self}")

    def __str__(self):
        if self.is_hologram:
            return(f'Hologram characteristics:'
                   f'\nN = {self.N(self.center):.2f} +/- {self.N_err:.2f} '
                   f'grooves/mm at hologram center'
                   f'\nHologram center at x0 = {self.center[0]:.1f} '
                   f'and y0 = {self.center[1]:.1f} with average tilt of {self.theta_tilt:.1f} '
                   f'degrees')
        else:
            return(f'Disperser characteristics:'
                   f'\nN = {self.N([0, 0]):.2f} +/- {self.N_err:.2f} grooves/mm'
                   f'\nAverage tilt of {self.theta_tilt:.1f} degrees')

    def N_flat(self, x, y):
        return self.N_input

    def theta_flat(self, x, y):
        return self.theta_tilt

    def N(self, x):
        """Return the number of grooves per mm of the grating at position x.

        Parameters
        ----------
        x: array
            The [x,y] pixel position.

        Returns
        -------
        N: float
            The number of grooves per mm at position x

        Examples
        --------
        >>> g = Disperser(400)
        >>> g.N((500,500))
        400

        >>> h = Hologram(label='HoloPhP')
        >>> np.round(h.N((500,500)), 3)
        345.479
        >>> np.round(h.N((0,0)), 3)
        283.569

        """
        if x[0] < np.min(self.N_x) or x[0] > np.max(self.N_x) \
                or x[1] < np.min(self.N_y) or x[1] > np.max(self.N_y):
            N = float(self.N_fit(*x))
        else:
            N = int(self.N_interp(*x))
        return N

    def theta(self, x):
        """Return the mean dispersion angle of the grating at position x.

        Parameters
        ----------
        x: float, array
            The [x,y] pixel position on the CCD.

        Returns
        -------
        theta: float
            The mean dispersion angle at position x in degrees.

        Examples
        --------
        >>> g = Disperser(400)
        >>> g.theta((500,500))
        0.0

        >>> h = Hologram('HoloPhP')
        >>> np.round(h.theta((700,700)), 3)
        -0.834
        """
        return float(self.theta_interp(*x))

    def load_files(self):
        """OBSOLETE. If they exist, load the files in data_dir/label/ to set the main
        characteristics of the grating. Overrides the N input at initialisation.

        Examples
        --------

        The files exist:

        >>> g = Disperser(400, label='Ron400')
        >>> g.N_input
        400.86918248709316
        >>> print(g.theta_tilt)
        -0.277

        The files do not exist:

        >>> g = Disperser(400, label='XXX')
        >>> g.N_input
        400
        >>> print(g.theta_tilt)
        0

        Hologram case

        >>> h = Hologram(label='HoloPhP')
        >>> h.N((500,500))
        345.47941688229855
        >>> h.theta((700,700))
        -0.8335087452358715
        >>> h.center
        [856.004, 562.34]

        """
        self.my_logger.warning(f'Obsolete: no config file found for {self.label}. '
                               f'Consider converting the disperser text files into a config .ini file.')
        filename = os.path.join(self.data_dir, self.label, "N.txt")
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.N_input = a[0]
            self.N_err = a[1]

        filename = os.path.join(self.data_dir, self.label, "full_name.txt")
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                for line in f:  # MFL: you really just want the last line of the file?
                    self.full_name = line.rstrip('\n')

        filename = os.path.join(self.data_dir, self.label, "transmission.txt")
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            l, t, e = a.T
            self.transmission = interpolate.interp1d(l, t, bounds_error=False, fill_value=0.)
            self.transmission_err = interpolate.interp1d(l, e, bounds_error=False, fill_value=0.)

        filename = os.path.join(self.data_dir, self.label, "ratio_order_2over1.txt")
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            if a.T.shape[0] == 2:
                l, t = a.T
            else:
                l, t, e = a.T
            self.ratio_order_2over1 = interpolate.interp1d(l, t, bounds_error=False, kind="linear",
                                                           fill_value="extrapolate")  # "(0, t[-1]))
            self.flat_ratio_order_2over1 = False
        filename = os.path.join(self.data_dir, self.label, "ratio_order_3over2.txt")
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            if a.T.shape[0] == 2:
                l, t = a.T
            else:
                l, t, e = a.T
            self.ratio_order_3over2 = interpolate.interp1d(l, t, bounds_error=False, kind="linear", fill_value="extrapolate")
            self.ratio_order_3over1 = interpolate.interp1d(l, self.ratio_order_3over2(l)*self.ratio_order_2over1(l),
                                                           bounds_error=False, kind="linear", fill_value="extrapolate")

        filename = os.path.join(self.data_dir, self.label, "hologram_center.txt")
        if os.path.isfile(filename):
            with open(filename) as f:
                lines = [ll.rstrip('\n') for ll in f]
            self.theta_tilt = float(lines[1].split(' ')[2])

        filename = os.path.join(self.data_dir, self.label, "hologram_grooves_per_mm.txt")
        if os.path.isfile(filename):
            self.is_hologram = True
            a = np.loadtxt(filename)
            self.N_x, self.N_y, self.N_data = a.T
            if parameters.CCD_REBIN > 1:
                self.N_x /= parameters.CCD_REBIN
                self.N_y /= parameters.CCD_REBIN
            self.N_interp = interpolate.CloughTocher2DInterpolator((self.N_x, self.N_y), self.N_data)
            self.N_fit = fit_poly2d(self.N_x, self.N_y, self.N_data, order=2)

        filename = os.path.join(self.data_dir, self.label, "hologram_center.txt")
        if os.path.isfile(filename):
            with open(filename) as f:
                lines = [ll.rstrip('\n') for ll in f]
            self.center = list(map(float, lines[1].split(' ')[:2]))
            self.theta_tilt = float(lines[1].split(' ')[2])

        filename = os.path.join(self.data_dir, self.label, "hologram_rotation_angles.txt")
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.theta_x, self.theta_y, self.theta_data = a.T
            if parameters.CCD_REBIN > 1:
                self.theta_x /= parameters.CCD_REBIN
                self.theta_y /= parameters.CCD_REBIN
            self.theta_interp = interpolate.CloughTocher2DInterpolator((self.theta_x, self.theta_y), self.theta_data,
                                                                       fill_value=self.theta_tilt)


    def load_config(self, path):
        """If they exist, load the config file in data_dir/label/ to set the main
        characteristics of the grating. Overrides the N input at initialisation.

        Parameters
        ----------
        path: str
            The path to the config file.

        Examples
        --------

        The files exist:

        >>> g = Disperser(400, label='Ron400')
        >>> g.N_input
        400.86918248709316
        >>> print(g.theta_tilt)
        -0.277

        Hologram case

        >>> h = Hologram(label='HoloPhP')
        >>> np.round(h.N((500,500)), 3)
        345.479
        >>> np.round(h.theta((700,700)), 3)
        -0.834
        >>> h.center
        [856.004, 562.34]

        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} is not a file.')
        self.my_logger.info(f'\n\tLoad disperser {self.label}:\n\tfrom {os.path.join(self.data_dir, self.label)}')
        d = from_config_to_dict(path)
        self.full_name = d["main"]["full_name"].rstrip('\n')

        # Disperser grooves per mm
        if type(d["main"]["n"]) == float:
            self.is_hologram = False
            self.N_input = float(d["main"]["n"])
            self.N_x = np.arange(0, parameters.CCD_IMSIZE)
            self.N_y = np.arange(0, parameters.CCD_IMSIZE)
            self.N_interp = self.N_flat
            self.N_fit = self.N_flat
        elif type(d["main"]["n"]) == str:
            self.is_hologram = True
            N_filename = os.path.join(self.data_dir, self.label, d["main"]["n"])
            a = np.loadtxt(N_filename)
            self.N_x, self.N_y, self.N_data = a.T
            if parameters.CCD_REBIN > 1:
                self.N_x /= parameters.CCD_REBIN
                self.N_y /= parameters.CCD_REBIN
            self.N_interp = interpolate.CloughTocher2DInterpolator((self.N_x, self.N_y), self.N_data)
            self.N_fit = fit_poly2d(self.N_x, self.N_y, self.N_data, order=2)
        else:
            raise ValueError("Unknown N type. Must be path or float.")
        self.N_err = float(d["main"]["n_err"])
        if self.is_hologram:
            self.center = [d["main"]["x_center"], d["main"]["y_center"]]

        # Disperser axis orientation
        if type(d["main"]["theta_tilt"]) == float:
            self.theta_tilt = float(d["main"]["theta_tilt"])
            self.theta_interp = self.theta_flat
        elif type(d["main"]["theta_tilt"]) == str:
            theta_filename = os.path.join(self.data_dir, self.label, d["main"]["theta_tilt"])
            a = np.loadtxt(theta_filename)
            self.theta_x, self.theta_y, self.theta_data = a.T
            if parameters.CCD_REBIN > 1:
                self.theta_x /= parameters.CCD_REBIN
                self.theta_y /= parameters.CCD_REBIN
            self.theta_interp = interpolate.CloughTocher2DInterpolator((self.theta_x, self.theta_y), self.theta_data,
                                                                       fill_value=self.theta_tilt)
        else:
            raise ValueError("Unknown theta_tilt type. Must be path or float.")

        if "transmission" in d["transmissions"].keys():
            if type(d["transmissions"]["transmission"]) == str:
                tr_filename = os.path.join(self.data_dir, self.label, d["transmissions"]["transmission"].rstrip('\n'))
                a = np.loadtxt(tr_filename)
                l, tr, e = a.T
                self.transmission = interpolate.interp1d(l, tr, bounds_error=False, fill_value=0.)
                self.transmission_err = interpolate.interp1d(l, e, bounds_error=False, fill_value=0.)
            elif type(d["transmissions"]["transmission"]) == float:
                tr = d["transmissions"]["transmission"] * np.ones_like(parameters.LAMBDAS).astype(float)
                self.transmission = interpolate.interp1d(parameters.LAMBDAS, tr, bounds_error=False,
                                                         fill_value=0.)
                self.transmission_err = interpolate.interp1d(parameters.LAMBDAS, 0 * tr, bounds_error=False,
                                                             fill_value=0.)
            else:
                raise ValueError("Unknown transmission type. Must be path or float.")
        if "ratio_order_2over1" in d["transmissions"].keys():
            if type(d["transmissions"]["ratio_order_2over1"]) == str:
                tr_filename = os.path.join(self.data_dir, self.label, d["transmissions"]["ratio_order_2over1"].rstrip('\n'))
                a = np.loadtxt(tr_filename)
                if a.T.shape[0] == 2:
                    l, t = a.T
                else:
                    l, t, e = a.T
                self.ratio_order_2over1 = interpolate.interp1d(l, t, bounds_error=False, kind="linear",
                                                               fill_value="extrapolate")  # "(0, t[-1]))
                self.flat_ratio_order_2over1 = False
            elif type(d["transmissions"]["ratio_order_2over1"]) == float:
                ratio = d["transmissions"]["ratio_order_2over1"] * np.ones_like(parameters.LAMBDAS).astype(float)
                self.ratio_order_2over1 = interpolate.interp1d(parameters.LAMBDAS, ratio, bounds_error=False, kind="linear",
                                                       fill_value="extrapolate")  # "(0, t[-1]))
                self.flat_ratio_order_2over1 = True
            else:
                raise ValueError("Unknown ratio_order_2over1 type. Must be path or float.")
        if "ratio_order_3over2" in d["transmissions"].keys():
            if type(d["transmissions"]["ratio_order_3over2"]) == str:
                tr_filename = os.path.join(self.data_dir, self.label, d["transmissions"]["ratio_order_3over2"].rstrip('\n'))
                a = np.loadtxt(tr_filename)
                if a.T.shape[0] == 2:
                    l, t = a.T
                else:
                    l, t, e = a.T
                self.ratio_order_3over2 = interpolate.interp1d(l, t, bounds_error=False, kind="linear",
                                                               fill_value="extrapolate")  # "(0, t[-1]))
            elif type(d["transmissions"]["ratio_order_3over2"]) == float:
                ratio = d["transmissions"]["ratio_order_3over2"] * np.ones_like(parameters.LAMBDAS).astype(float)
                self.ratio_order_3over2 = interpolate.interp1d(parameters.LAMBDAS, ratio, bounds_error=False, kind="linear",
                                                       fill_value="extrapolate")  # "(0, t[-1]))
            else:
                raise ValueError("Unknown ratio_order_2over1 type. Must be path or float.")

            self.ratio_order_3over1 = interpolate.interp1d(l, self.ratio_order_3over2(l)*self.ratio_order_2over1(l),
                                                           bounds_error=False, kind="linear", fill_value="extrapolate")

    def refraction_angle(self, deltaX, x0, D):
        """ Return the refraction angle with respect to the disperser normal, using geometrical consideration,
        given the distance to order 0 in pixels.

        Parameters
        ----------
        deltaX: float
            The distance in pixels between the order 0 and a spectrum pixel in the rotated image.
        x0: array
            The order 0 position [x0,y0] in the full non-rotated image.
        D: float
            The distance between the CCD and the disperser in mm.

        Returns
        -------
        theta: float
            The refraction angle in radians.

        Examples
        --------
        >>> from spectractor.config import load_config
        >>> load_config("ctio.ini")
        >>> g = Disperser(400)
        >>> theta = g.refraction_angle(500, [parameters.CCD_IMSIZE/2,  parameters.CCD_IMSIZE/2], D=55)
        >>> assert np.isclose(theta, np.arctan2(500*parameters.CCD_PIXEL2MM, 55))
        """
        theta = get_refraction_angle(deltaX, x0, D=D)
        return theta

    def refraction_angle_lambda(self, lambdas, x0, order=1):
        """ Return the refraction angle with respect to the disperser normal, using geometrical consideration,
        given the wavelength in nm and the order of the spectrum.

        Parameters
        ----------
        lambdas: float, array
            The distance in pixels between the order 0 and a spectrum pixel in the rotated image.
        x0: float, array
            The order 0 pixel position [x0,y0] in the full non-rotated image.
        order: int
            The order of the spectrum.

        Returns
        -------
        theta: float
            The refraction angle in radians.

        Examples
        --------
        >>> from spectractor.config import load_config
        >>> load_config("ctio.ini")
        >>> g = Disperser(400)
        >>> theta = g.refraction_angle(500, [parameters.CCD_IMSIZE/2,  parameters.CCD_IMSIZE/2], D=55)
        >>> assert np.isclose(theta, np.arctan2(500*parameters.CCD_PIXEL2MM, 55))
        """
        theta0 = get_theta0(x0)
        return np.arcsin(np.clip(order * lambdas * 1e-6 * self.N(x0) + np.sin(theta0),-1, 1))

    def grating_refraction_angle_to_lambda(self, thetas, x0, order=1):
        """ Convert refraction angles into wavelengths (in nm) with.

        Parameters
        ----------
        thetas: array, float
            Refraction angles in radian.
        x0: float or [float, float]
            Order 0 position detected in the non-rotated image.
        order: int
            Order of the spectrum (default: 1)

        Examples
        --------
        >>> from spectractor.config import load_config
        >>> load_config("default.ini")
        >>> disperser = Disperser(N=300)
        >>> x0 = [800,800]
        >>> lambdas = np.arange(300, 900, 100)
        >>> thetas = disperser.refraction_angle_lambda(lambdas, x0, order=1)
        >>> print(thetas)
        [0.0896847  0.11985125 0.15012783 0.18054376 0.21112957 0.24191729]
        >>> lambdas = disperser.grating_refraction_angle_to_lambda(thetas, x0, order=1)
        >>> print(lambdas)
        [300. 400. 500. 600. 700. 800.]
        """
        theta0 = get_theta0(x0)
        lambdas = (np.sin(thetas) - np.sin(theta0)) / (order * self.N(x0))
        return lambdas * 1e6

    def grating_pixel_to_lambda(self, deltaX, x0, D, order=1):
        """ Convert pixels into wavelengths (in nm) with.

        Parameters
        ----------
        deltaX: array, float
            *Algebraic* pixel distances to order 0 along the dispersion axis.
        x0: float or [float, float]
            Order 0 position detected in the non-rotated image.
        D: float
            The distance between the CCD and the disperser in mm.
        order: int
            Order of the spectrum (default: 1).

        Examples
        --------
        >>> from spectractor.config import load_config
        >>> load_config("default.ini")
        >>> disperser = Disperser(N=300)
        >>> x0 = [800,800]
        >>> deltaX = np.arange(0,1000,1).astype(float)
        >>> lambdas = disperser.grating_pixel_to_lambda(deltaX, x0, D=55, order=1)
        >>> print(lambdas[:5])
        [0.         1.45454532 2.90909063 4.36363511 5.81817793]
        >>> pixels = disperser.grating_lambda_to_pixel(lambdas, x0, D=55, order=1)
        >>> print(pixels[:5])
        [0. 1. 2. 3. 4.]
        """
        theta = self.refraction_angle(deltaX, x0, D=D)
        theta0 = get_theta0(x0)
        lambdas = (np.sin(theta) - np.sin(theta0)) / (order * self.N(x0))
        return lambdas * 1e6

    def grating_lambda_to_pixel(self, lambdas, x0, D, order=1):
        """ Convert wavelength in nm into pixel distance with order 0.

        Parameters
        ----------
        lambdas: array, float
            Wavelengths in nm.
        x0: float or [float, float]
            Order 0 position detected in the raw image.
        D: float
            The distance between the CCD and the disperser in mm.
        order: int
            Order of the spectrum (default: 1)

        Examples
        --------
        >>> from spectractor.config import load_config
        >>> load_config("default.ini")
        >>> disperser = Disperser(N=300)
        >>> x0 = [800,800]
        >>> deltaX = np.arange(0,1000,1).astype(float)
        >>> lambdas = disperser.grating_pixel_to_lambda(deltaX, x0, D=55, order=1)
        >>> print(lambdas[:5])
        [0.         1.45454532 2.90909063 4.36363511 5.81817793]
        >>> pixels = disperser.grating_lambda_to_pixel(lambdas, x0, D=55, order=1)
        >>> print(pixels[:5])
        [0. 1. 2. 3. 4.]
        """
        lambdas = np.copy(lambdas)
        theta0 = get_theta0(x0)
        theta = self.refraction_angle_lambda(lambdas, x0, order=order)
        deltaX = D * (np.tan(theta) - np.tan(theta0)) / parameters.CCD_PIXEL2MM
        return deltaX

    def grating_resolution(self, deltaX, x0, D, order=1):
        """ Return wavelength resolution in nm per pixel.
        See mathematica notebook: derivative of the grating formula.
        x0: the order 0 position on the full raw image.
        deltaX: the distance in pixels between order 0 and signal point
        in the rotated image."""
        delta = get_delta_pix_ortho(deltaX, x0, D=D) * parameters.CCD_PIXEL2MM
        # theta = self.refraction_angle(x,x0,order=order)
        # res = (np.cos(theta)**3*CCD_PIXEL2MM*1e6)/(order*self.N(x0)*self.D)
        res = (D ** 2 / pow(D ** 2 + delta ** 2, 1.5)) * parameters.CCD_PIXEL2MM * 1e6 / (order * self.N(x0))
        return res

    def plot_transmission(self, xlim=None):
        """Plot the transmission of the grating with respect to the wavelength (in nm).

        Parameters
        ----------
        xlim: [xmin,xmax], optional
            List of the X axis extrema (default: None).

        Examples
        --------
        >>> g = Disperser(400, label='Ron400')
        >>> g.plot_transmission(xlim=(400,800))
        >>> g = Hologram(label='holo4_003')
        >>> g.plot_transmission(xlim=(400,800))
        """
        wavelengths = np.linspace(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX, 100)
        if xlim is not None:
            wavelengths = np.linspace(xlim[0], xlim[1], 100)
        plt.plot(wavelengths, self.transmission(wavelengths), 'b-', label=self.label)
        plt.plot(wavelengths, self.ratio_order_2over1(wavelengths), 'r-', label="Ratio 2/1")
        if self.ratio_order_3over2:
            plt.plot(wavelengths, self.ratio_order_3over2(wavelengths), 'g-', label="Ratio 3/2")
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel(r"Transmission")
        plt.grid()
        plt.legend(loc='best')
        if parameters.DISPLAY:
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()


class Hologram(Disperser):

    def __init__(self, label, **kwargs):
        Disperser.__init__(self, N=-1, label=label, **kwargs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import ndimage

from spectractor.tools import *


def build_hologram(order0_position, order1_position, theta_tilt=0, lambda_plot=256000):
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
    A = lambda x, y: U(x - xA[0], y - xA[1], -parameters.DISTANCE2CCD) + U(x-xB[0], y-xB[1], -parameters.DISTANCE2CCD)
    intensity = lambda x, y: np.abs(A(x, y)) ** 2
    xholo = np.linspace(0, parameters.CCD_IMSIZE * parameters.CCD_PIXEL2MM, parameters.CCD_IMSIZE)
    yholo = np.linspace(0, parameters.CCD_IMSIZE * parameters.CCD_PIXEL2MM, parameters.CCD_IMSIZE)
    xxholo, yyholo = np.meshgrid(xholo, yholo)
    holo = intensity(xxholo, yyholo)
    rotated_holo = ndimage.interpolation.rotate(holo, theta_tilt)
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
    rotated_ronchi = ndimage.interpolation.rotate(ronchi, theta_tilt)
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
    if isinstance(x0, (list, tuple, np.ndarray)):
        return (x0[0] - parameters.CCD_IMSIZE / 2) * parameters.CCD_PIXEL2ARCSEC * parameters.CCD_ARCSEC2RADIANS
    else:
        return (x0 - parameters.CCD_IMSIZE / 2) * parameters.CCD_PIXEL2ARCSEC * parameters.CCD_ARCSEC2RADIANS


def get_delta_pix_ortho(deltaX, x0, D=parameters.DISTANCE2CCD):
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
    >>> delta, D = 500, 55
    >>> get_delta_pix_ortho(delta, [parameters.CCD_IMSIZE/2,  parameters.CCD_IMSIZE/2], D=D)
    500.0
    >>> get_delta_pix_ortho(delta, [500,500], D=D)
    497.66545567320992
    """
    theta0 = get_theta0(x0)
    deltaX0 = np.tan(theta0) * D / parameters.CCD_PIXEL2MM
    return deltaX + deltaX0


def get_refraction_angle(deltaX, x0, D=parameters.DISTANCE2CCD):
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


def get_N(deltaX, x0, D=parameters.DISTANCE2CCD, wavelength=656, order=1):
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
    """Return the nuetrla lines of an hologram."""
    xs = np.linspace(0, parameters.CCD_IMSIZE, 20)
    line1 = np.tan(theta_tilt * np.pi / 180) * (xs - x_center) + y_center
    line2 = np.tan((theta_tilt + 90) * np.pi / 180) * (xs - x_center) + y_center
    return xs, line1, line2


def order01_positions(holo_center, N, theta_tilt, theta0=0, verbose=True):
    """Return the order 0 and order 1 positions of an hologram."""
    # refraction angle between order 0 and order 1 at construction
    alpha = np.arcsin(N * parameters.LAMBDA_CONSTRUCTOR + np.sin(theta0))
    # distance between order 0 and order 1 in pixels
    AB = (np.tan(alpha) - np.tan(theta0)) * parameters.DISTANCE2CCD / parameters.CCD_PIXEL2MM
    # position of order 1 in pixels
    x_center = holo_center[0]
    y_center = holo_center[1]
    order1_position = [0.5 * AB * np.cos(theta_tilt * np.pi / 180) + x_center,
                       0.5 * AB * np.sin(theta_tilt * np.pi / 180) + y_center]
    # position of order 0 in pixels
    order0_position = [-0.5 * AB * np.cos(theta_tilt * np.pi / 180) + x_center,
                       -0.5 * AB * np.sin(theta_tilt * np.pi / 180) + y_center]
    if verbose:
        print('Order  0 position at x0 = %.1f and y0 = %.1f' % (order0_position[0], order0_position[1]))
        print('Order +1 position at x0 = %.1f and y0 = %.1f' % (order1_position[0], order1_position[1]))
        print('Distance between the orders: %.2f pixels (%.2f mm)' % (AB, AB * parameters.CCD_PIXEL2MM))
    return order0_position, order1_position, AB


def find_order01_positions(holo_center, N_interp, theta_interp, verbose=True):
    """Find the order 0 and order 1 positions of an hologram."""
    N = N_interp(holo_center)
    theta_tilt = theta_interp(holo_center)
    theta0 = 0
    convergence = 0
    while abs(N - convergence) > 1e-6:
        order0_position, order1_position, AB = order01_positions(holo_center, N, theta_tilt, theta0, verbose=False)
        convergence = np.copy(N)
        N = N_interp(order0_position)
        theta_tilt = theta_interp(order0_position)
        theta0 = get_theta0(order0_position)
    order0_position, order1_position, AB = order01_positions(holo_center, N, theta_tilt, theta0, verbose=verbose)
    return order0_position, order1_position, AB


class Grating:
    """Generic class for dispersers."""

    def __init__(self, N, label="", D=parameters.DISTANCE2CCD, data_dir=parameters.HOLO_DIR, verbose=False):
        """Initialize a standard grating object.

        Parameters
        ----------
        N: float
            The number of grooves per mm of the grating
        label: str
            String label for the grating (default: '')
        D: float
            The distance between the CCD and the disperser in mm.
        data_dir: str
            The directory where information about this disperser is stored. Must be in the form data_dir/label/...
            (default: parameters.HOLO_DIR)
        verbose: bool
            Set to True to increase the verbosity of the initialisation (default: False)

        Examples
        --------
        >>> g = Grating(400)
        >>> print(g.N_input)
        400
        >>> g = Grating(400, label="Ron400", data_dir=parameters.HOLO_DIR)
        >>> print(g.N_input)
        400.869182487
        >>> assert g.D is parameters.DISTANCE2CCD
        """
        self.N_input = N
        self.N_err = 1
        self.D = D
        self.label = label
        self.full_name = label
        self.data_dir = data_dir
        self.plate_center = [0.5 * parameters.CCD_IMSIZE, 0.5 * parameters.CCD_IMSIZE]
        self.theta_tilt = 0
        self.transmission = None
        self.transmission_err = None
        self.load_files(verbose=verbose)

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
        >>> g = Grating(400)
        >>> g.N((500,500))
        400
        """
        return self.N_input

    def load_files(self, verbose=False):
        """If they exists, load the files in data_dir/label/ to set the main
        characteristics of the grating. Overrides the N input at initialisation.

        Parameters
        ----------
        verbose: bool
            Set to True to get more verbosity.

        Examples
        --------

        The files exist:
        >>> g = Grating(400, label='Ron400')
        >>> g.N_input
        400.86918248709316
        >>> print(g.theta_tilt)
        -0.277

        The files do not exist:
        >>> g = Grating(400, label='XXX')
        >>> g.N_input
        400
        >>> print(g.theta_tilt)
        0

        """
        filename = self.data_dir + self.label + "/N.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.N_input = a[0]
            self.N_err = a[1]
        filename = self.data_dir + self.label + "/full_name.txt"
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                for line in f:
                    self.full_name = line.rstrip('\n')
        filename = self.data_dir + self.label + "/transmission.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            l, t, e = a.T
            self.transmission = interpolate.interp1d(l, t, bounds_error=False, fill_value=0.)
            self.transmission_err = interpolate.interp1d(l, e, bounds_error=False, fill_value=0.)
        else:
            self.transmission = lambda x: np.ones_like(x).astype(float)
            self.transmission_err = lambda x: np.zeros_like(x).astype(float)
        filename = self.data_dir + self.label + "/hologram_center.txt"
        if os.path.isfile(filename):
            lines = [ll.rstrip('\n') for ll in open(filename)]
            self.theta_tilt = float(lines[1].split(' ')[2])
        else:
            self.theta_tilt = 0
        if verbose:
            print('Grating plate center at x0 = {:.1f} and y0 = {:.1f} with average tilt of {:.1f} degrees'.format(
                self.plate_center[0], self.plate_center[1], self.theta_tilt))

    def refraction_angle(self, deltaX, x0):
        """ Return the refraction angle with respect to the disperser normal, using geometrical consideration,
        given the distance to order 0 in pixels.

        Parameters
        ----------
        deltaX: float
            The distance in pixels between the order 0 and a spectrum pixel in the rotated image.
        x0: array
            The order 0 position [x0,y0] in the full non-rotated image.

        Returns
        -------
        theta: float
            The refraction angle in radians.

        Examples
        --------
        >>> g = Grating(400)
        >>> theta = g.refraction_angle(500, [parameters.CCD_IMSIZE/2,  parameters.CCD_IMSIZE/2])
        >>> assert np.isclose(theta, np.arctan2(500*parameters.CCD_PIXEL2MM, parameters.DISTANCE2CCD))
        """
        theta = get_refraction_angle(deltaX, x0, D=self.D)
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
        >>> g = Grating(400)
        >>> theta = g.refraction_angle(500, [parameters.CCD_IMSIZE/2,  parameters.CCD_IMSIZE/2])
        >>> assert np.isclose(theta, np.arctan2(500*parameters.CCD_PIXEL2MM, parameters.DISTANCE2CCD))
        """
        theta0 = get_theta0(x0)
        return np.arcsin(order * lambdas*1e-6 * self.N(x0) + np.sin(theta0))

    def grating_pixel_to_lambda(self, deltaX, x0, order=1):
        """ Convert pixels into wavelengths (in nm) with.

        Parameters
        ----------
        deltaX: array, float
            Pixel distance to order 0.
        x0: float or [float, float]
            Order 0 position detected in the non-rotated image.
        order: int
            Order of the spectrum (default: 1)

        Examples
        --------
        >>> disperser = Grating(N=300, D=55)
        >>> x0 = [800,800]
        >>> deltaX = np.arange(0,1000,1).astype(float)
        >>> lambdas = disperser.grating_pixel_to_lambda(deltaX, x0, order=1)
        >>> print(lambdas[:5])
        [ 0.          1.45454532  2.90909063  4.36363511  5.81817793]
        >>> pixels = disperser.grating_lambda_to_pixel(lambdas, x0, order=1)
        >>> print(pixels[:5])
        [ 0.  1.  2.  3.  4.]
        """
        theta = self.refraction_angle(deltaX, x0)
        theta0 = get_theta0(x0)
        lambdas = (np.sin(theta) - np.sin(theta0)) / (order * self.N(x0))
        return lambdas * 1e6

    def grating_lambda_to_pixel(self, lambdas, x0, order=1):
        """ Convert wavelength in nm into pixel distance with order 0.

        Parameters
        ----------
        lambdas: array, float
            Wavelengths in nm.
        x0: float or [float, float]
            Order 0 position detected in the raw image.
        order: int
            Order of the spectrum (default: 1)

        Examples
        --------
        >>> disperser = Grating(N=300, D=55)
        >>> x0 = [800,800]
        >>> deltaX = np.arange(0,1000,1).astype(float)
        >>> lambdas = disperser.grating_pixel_to_lambda(deltaX, x0, order=1)
        >>> print(lambdas[:5])
        [ 0.          1.45454532  2.90909063  4.36363511  5.81817793]
        >>> pixels = disperser.grating_lambda_to_pixel(lambdas, x0, order=1)
        >>> print(pixels[:5])
        [ 0.  1.  2.  3.  4.]
        """
        lambdas = np.copy(lambdas)
        theta0 = get_theta0(x0)
        theta = self.refraction_angle_lambda(lambdas, x0, order=order)
        deltaX = self.D * (np.tan(theta) - np.tan(theta0)) / parameters.CCD_PIXEL2MM
        return deltaX

    def grating_resolution(self, deltaX, x0, order=1):
        """ Return wavelength resolution in nm per pixel.
        See mathematica notebook: derivative of the grating formula.
        x0: the order 0 position on the full raw image.
        deltaX: the distance in pixels between order 0 and signal point 
        in the rotated image."""
        delta = get_delta_pix_ortho(deltaX, x0, D=self.D) * parameters.CCD_PIXEL2MM
        # theta = self.refraction_angle(x,x0,order=order)
        # res = (np.cos(theta)**3*CCD_PIXEL2MM*1e6)/(order*self.N(x0)*self.D)
        res = (self.D ** 2 / pow(self.D ** 2 + delta ** 2, 1.5)) * parameters.CCD_PIXEL2MM * 1e6 / (order * self.N(x0))
        return res

    def plot_transmission(self, xlim=None):
        """Plot the transmission of the grating with respect to the wavelength (in nm).

        Parameters
        ----------
        xlim: [xmin,xmax], optional
            List of the X axis extrema (default: None).

        Examples
        --------
        >>> g = Grating(400, label='Ron400')
        >>> g.plot_transmission(xlim=(400,800))
        """
        wavelengths = np.linspace(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX, 100)
        if xlim is not None:
            wavelengths = np.linspace(xlim[0], xlim[1], 100)
        plt.plot(wavelengths, self.transmission(wavelengths), 'b-', label=self.label)
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel(r"Transmission")
        plt.grid()
        plt.legend(loc='best')
        if parameters.DISPLAY:
            plt.show()


class Hologram(Grating):

    def __init__(self, label, D=parameters.DISTANCE2CCD, data_dir=parameters.HOLO_DIR,
                 lambda_plot=256000, verbose=False):
        """Initialize an Hologram object, given its label. Specification are loaded from text files
        in data_dir/label/... Inherit from the Grating class.

        Parameters
        ----------
        label: str
            String label for the grating (default: '')
        D: float
            The distance between the CCD and the disperser in mm.
        data_dir: str
            The directory where information about this disperser is stored. Must be in the form data_dir/label/...
            (default: parameters.HOLO_DIR)
        lambda_plot: float, optional
            Wavelength to plot the hologram pattern (default: 256000).
        verbose: bool
            Set to True to increase the verbosity of the initialisation (default: False)

        Examples
        --------
        >>> h = Hologram(label="HoloPhP", data_dir=parameters.HOLO_DIR)
        >>> h.label
        'HoloPhP'
        >>> h.N((500,500))
        345.4794168822986

        """
        Grating.__init__(self, parameters.GROOVES_PER_MM, D=D, label=label, data_dir=data_dir, verbose=False)
        self.holo_center = None  # center of symmetry of the hologram interferences in pixels
        self.plate_center = None  # center of the hologram plate
        self.theta = None  # interpolated rotation angle map of the hologram from data in degrees
        self.theta_data = None  # rotation angle map data of the hologram from data in degrees
        self.theta_x = None  # x coordinates for the interpolated rotation angle map
        self.theta_y = None  # y coordinates for the interpolated rotation angle map
        self.x_lines = None
        self.line1 = None
        self.line2 = None
        self.order0_position = None
        self.order1_position = None
        self.AB = None
        self.N_x = None
        self.N_y = None
        self.N_data = None
        self.N_interp = None
        self.N_fit = None
        self.lambda_plot = lambda_plot
        self.is_hologram = True
        self.load_specs(verbose=verbose)

    def N(self, x):
        """Return the number of grooves per mm of the grating at position x. If the position is inside
        the data provided by the text files, this number is computed from an interpolation. If it lies outside,
        it is computed from a 2D polynomial fit.

        Parameters
        ----------
        x: float, array
            The [x,y] pixel position on the CCD.

        Returns
        -------
        N: float
            The number of grooves per mm at position x

        Examples
        --------
        >>> h = Hologram('HoloPhP')
        >>> h.N((500,500))
        345.4794168822986
        >>> h.N((0,0))
        283.56876727310373
        """

        if x[0] < np.min(self.N_x) or x[0] > np.max(self.N_x) \
                or x[1] < np.min(self.N_y) or x[1] > np.max(self.N_y):
            N = self.N_fit(x[0], x[1])
        else:
            N = int(self.N_interp(x))
        return N

    def load_specs(self, verbose=True):
        """Load the files in data_dir/label/ to set the main
        characteristics of the hologram. If they do not exist, default values are used.

        Parameters
        ----------
        verbose: bool
            Set to True to get more verbosity.

        Examples
        --------

        The files exist:
        >>> h = Hologram(label='HoloPhP')
        >>> h.N((500,500))
        345.4794168822986
        >>> h.theta((500,500))
        -1.3393287109201792
        >>> h.holo_center
        [856.004, 562.34]

        The files do not exist:
        >>> h = Hologram(label='XXX')
        >>> h.N((500,500))
        350
        >>> h.theta((500,500))
        0
        >>> h.holo_center
        [1024.0, 1024.0]

        """
        if verbose:
            print('Load disperser {}:'.format(self.label))
            print('\tfrom {}'.format(self.data_dir + self.label))
        filename = self.data_dir + self.label + "/hologram_grooves_per_mm.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.N_x, self.N_y, self.N_data = a.T
            N_interp = interpolate.interp2d(self.N_x, self.N_y, self.N_data, kind='cubic')
            self.N_fit = fit_poly2d(self.N_x, self.N_y, self.N_data, order=2)
            self.N_interp = lambda x: float(N_interp(x[0], x[1]))
        else:
            self.is_hologram = False
            self.N_x = np.arange(0, parameters.CCD_IMSIZE)
            self.N_y = np.arange(0, parameters.CCD_IMSIZE)
            filename = self.data_dir + self.label + "/N.txt"
            if os.path.isfile(filename):
                a = np.loadtxt(filename)
                self.N_interp = lambda x: a[0]
                self.N_fit = lambda x, y: a[0]
            else:
                self.N_interp = lambda x: parameters.GROOVES_PER_MM
                self.N_fit = lambda x, y: parameters.GROOVES_PER_MM
        filename = self.data_dir + self.label + "/hologram_center.txt"
        if os.path.isfile(filename):
            lines = [ll.rstrip('\n') for ll in open(filename)]
            self.holo_center = list(map(float, lines[1].split(' ')[:2]))
            self.theta_tilt = float(lines[1].split(' ')[2])
        else:
            self.holo_center = [0.5 * parameters.CCD_IMSIZE, 0.5 * parameters.CCD_IMSIZE]
            self.theta_tilt = 0
        filename = self.data_dir + self.label + "/hologram_rotation_angles.txt"
        if os.path.isfile(filename):
            a = np.loadtxt(filename)
            self.theta_x, self.theta_y, self.theta_data = a.T
            theta_interp = interpolate.interp2d(self.theta_x, self.theta_y, self.theta_data, kind='cubic')
            self.theta = lambda x: float(theta_interp(x[0], x[1]))
        else:
            self.theta = lambda x: self.theta_tilt
        self.plate_center = [0.5 * parameters.CCD_IMSIZE + parameters.PLATE_CENTER_SHIFT_X / parameters.CCD_PIXEL2MM,
                             0.5 * parameters.CCD_IMSIZE + parameters.PLATE_CENTER_SHIFT_Y / parameters.CCD_PIXEL2MM]
        self.x_lines, self.line1, self.line2 = neutral_lines(self.holo_center[0], self.holo_center[1], self.theta_tilt)
        if verbose:
            if self.is_hologram:
                print('Hologram characteristics:')
                print('\tN = {:.2f} +/- {:.2f} grooves/mm at plate center'.format(self.N(self.plate_center), self.N_err))
                print('\tPlate center at x0 = {:.1f} and y0 = {:.1f} with average tilt of {:.1f} degrees'.format(
                    self.plate_center[0], self.plate_center[1], self.theta_tilt))
                print('\tHologram center at x0 = {:.1f} and y0 = {:.1f} with average tilt of {:.1f} degrees'.format(
                    self.holo_center[0], self.holo_center[1], self.theta_tilt))
            else:
                print('Grating characteristics:')
                print('\tN = {:.2f} +/- {:.2f} grooves/mm'.format(self.N([0, 0]), self.N_err))
                print('\tAverage tilt of {:.1f} degrees'.format(self.theta_tilt))
        if self.is_hologram:
            self.order0_position, self.order1_position, self.AB = find_order01_positions(self.holo_center,
                                                                                         self.N_interp, self.theta,
                                                                                         verbose=verbose)


if __name__ == "__main__":
    import doctest
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    doctest.testmod()

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.interpolate import interp1d, RegularGridInterpolator

from spectractor.config import set_logger
import spectractor.parameters as parameters

import spectractor.simulation.libradtran as libradtran
from spectractor.simulation.throughput import plot_transmission_simple


angstrom_exponent_default = 0.0192


class Atmosphere:

    def __init__(self, airmass, pressure, temperature, lambda_min=250, lambda_max=1200, altitude=parameters.OBS_ALTITUDE):
        """Class to evaluate an atmospheric transmission using Libradtran.

        Parameters
        ----------
        airmass: float
            Airmass of the source object.
        pressure: float
            Pressure of the atmosphere at observatory altitude in hPa.
        temperature: float
            Temperature of the atmosphere at observatory altitude in Celsius degrees.
        lambda_min: float
            Minimum wavelength for simulation in nm (default: 250).
        lambda_max: float
            Maximum wavelength for simulation in nm (default: 1200).
        altitude: float
            Observatory altitude in km (default: parameters.OBS_ALTITUDE).

        Examples
        --------
        >>> a = Atmosphere(airmass=1.2, pressure=800, temperature=5)
        >>> print(a.airmass)
        1.2
        >>> print(a.pressure)
        800
        >>> print(a.temperature)
        5
        >>> print(a.transmission(500))
        1.0
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.airmass = airmass
        self.pressure = pressure
        self.temperature = temperature
        self.altitude = altitude
        self.pwv = None
        self.ozone = None
        self.aerosols = None
        self.transmission = lambda x: np.ones_like(x).astype(float)
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.title = ""
        self.label = ""

    def set_title(self):
        """Make a title string for the simulation.

        """
        self.title = f'Atmospheric transmission with z={self.airmass:4.2f}, P={self.pressure:4.2f} hPa, ' \
                     rf'T={self.temperature:4.2f}$\degree$C'

    def set_label(self):
        """Make a label string for the simulation.

        """
        self.label = f'PWV={self.pwv:4.2f}mm, OZ={self.ozone:4.2f}DB, VAOD={self.aerosols:4.2f} '

    def set_lambda_range(self, lambdas):
        """Reset the Atmosphere wavelength range for optimized computations.

        Parameters
        ----------
        lambdas: array_like
            Wavelength array in nm.

        Examples
        --------
        >>> a = Atmosphere(airmass=1.2, pressure=800, temperature=5, lambda_min=350, lambda_max=1000)
        >>> a.lambda_min
        350
        >>> a.lambda_max
        1000
        >>> a.set_lambda_range(np.arange(400, 810, 10))
        >>> a.lambda_min
        400
        >>> a.lambda_max
        800

        """
        self.lambda_min = int(np.min(lambdas))
        self.lambda_max = int(np.ceil(np.max(lambdas)))

    def simulate(self, aerosols, ozone, pwv, angstrom_exponent=None):
        """Simulate the atmosphere transparency with Libradtran given atmospheric composition.

        Values outside the Libradtran simulation range are set to zero.

        Parameters
        ----------
        aerosols: float
            VAOD Vertical Aerosols Optical Depth.
        ozone: float
            Ozone quantity in Dobson.
        pwv: float
            Precipitable Water Vapor quantity in mm.
        angstrom_exponent: float, optional
            Angstrom exponent for aerosols. If negative or None, default aerosol model from Libradtran is used.
            If value is 0.0192, the atmospheric transmission is very close to the case with angstrom_exponent=None (default: None).

        Returns
        -------
        transmission: callable
            The transmission function of wavelengths in nm.

        Examples
        --------
        >>> a = Atmosphere(airmass=1.2, pressure=800, temperature=5, lambda_min=350, lambda_max=1000)
        >>> transmission = a.simulate(aerosols=0.05, ozone=400, pwv=5, angstrom_exponent=-1)
        >>> a.ozone
        400
        >>> a.pwv
        5
        >>> a.aerosols
        0.05
        >>> transmission([350, 550, 600, 800, 950])
        array([0.5035478, 0.8303832, 0.8381782, 0.9382188, 0.7130625])
        >>> a.plot_transmission()
        >>> transmission_ang_exp = a.simulate(aerosols=0.05, ozone=400, pwv=5, angstrom_exponent=0.02)
        >>> transmission_ang_exp([350, 550, 600, 800, 950])
        array([0.5018349, 0.8318764, 0.839858 , 0.938896 , 0.7123196])

        .. doctest::
            :hide:

            >>> assert transmission is not None
            >>> assert transmission_ang_exp is not None
            >>> assert a.transmission(500) > 0
            >>> assert a.transmission(1100) == 0

        .. plot::

            from spectractor.simulation.atmosphere import Atmosphere
            a = Atmosphere(airmass=1.2, pressure=800, temperature=5, lambda_min=300, lambda_max=1000)
            transmission = a.simulate(ozone=400, pwv=5, aerosols=0.05)
            a.plot_transmission()
        """
        self.pwv = pwv
        self.ozone = ozone
        self.aerosols = aerosols
        self.set_title()
        self.set_label()
        self.my_logger.debug(f'\n\t{self.title}\n\t\t{self.label}')

        lib = libradtran.Libradtran()
        wl, atm = lib.simulate(self.airmass, aerosols, ozone, pwv, self.pressure, angstrom_exponent=angstrom_exponent,
                               lambda_min=self.lambda_min, lambda_max=self.lambda_max, altitude=self.altitude)
        self.transmission = interp1d(wl, atm, kind='linear', bounds_error=False, fill_value=(0, 0))
        return self.transmission

    def plot_transmission(self, lambdas=parameters.LAMBDAS):
        """Plot the atmospheric transmission computed with Libradtran.

        Parameters
        ----------
        lambdas: array_like, optional
            Array of wavelengths in nm (default: parameters.LAMBDAS).

        Examples
        --------
        >>> a = Atmosphere(airmass=1.2, pressure=800, temperature=5)
        >>> transmission = a.simulate(ozone=400, pwv=5, aerosols=0.05)
        >>> a.plot_transmission()

        """
        plot_transmission_simple(plt.gca(), lambdas, self.transmission(lambdas),
                                 title=self.title, label=self.label)
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        else:
            plt.close('all')


class AtmosphereGrid(Atmosphere):

    def __init__(self, image_filename="", spectrum_filename="", atmgrid_filename="",
                 airmass=1., pressure=800., temperature=10.,
                 pwv_grid=[0, 10, 10], ozone_grid=[100, 700, 7], aerosol_grid=[0, 0.1, 10],
                 lambdas=parameters.LAMBDAS, altitude=parameters.OBS_ALTITUDE):
        """Class to load and interpolate grids of atmospheric transmission computed with Libradtran.

        Parameters
        ----------
        image_filename: str, optional
            The original image fits file name from which the grid was computed or has to be computed (default: "").
        spectrum_filename: str, optional
            The file name of the spectrum fits file name from which the grid was computed or has to be computed (default: "").
        atmgrid_filename: str, optional
            The file name of the atmospheric grid if it exists (default: "").
        airmass: float, optional
            Airmass of the source object (default: 1). Overwritten if spectrum_filename is given.
        pressure: float, optional
            Pressure of the atmosphere at observatory altitude in hPa (default: 800). Overwritten if spectrum_filename is given.
        temperature: float, optional
            Temperature of the atmosphere at observatory altitude in Celsius degrees (default: 10). Overwritten if spectrum_filename is given.
        pwv_grid: list, optional
            List of 3 numbers for the PWV quantity: min, max, number of simulations (default: [0, 10, 10]).
        ozone_grid: list, optional
            List of 3 numbers for the ozone quantity: min, max, number of simulations (default: [100, 700, 7]).
        aerosol_grid: list, optional
            List of 3 numbers for the aerosol quantity: min, max, number of simulations (default: [0, 0.1, 10]).
        lambdas: array_like, optional
            Array of wavelengths (default: parameters.LAMBDAS).
        altitude: float
            Observatory altitude in km (default: parameters.OBS_ALTITUDE).

        Examples
        --------
        >>> a = AtmosphereGrid(atmgrid_filename='./tests/data/reduc_20170530_134_atmsim.fits')
        >>> a.image_filename.split('/')[-1]
        'reduc_20170530_134_spectrum.fits'
        """
        Atmosphere.__init__(self, airmass, pressure, temperature,
                            lambda_min=np.min(lambdas), lambda_max=np.max(lambdas), altitude=altitude)
        self.my_logger = set_logger(self.__class__.__name__)
        self.image_filename = image_filename
        if spectrum_filename != "":
            self.image_filename = spectrum_filename
        self.filename = atmgrid_filename

        # Definition of data format for the atmospheric grid
        self.index_atm_count = 0  # row 0 : count number
        self.index_atm_aer = 1  # row 1 : aerosol value
        self.index_atm_pwv = 2  # row 2 : pwv value
        self.index_atm_oz = 3  # row 3 : ozone value
        self.index_atm_data = 4  # row 4 : data start

        # specify parameters for the atmospheric grid
        self.lambdas = lambdas
        self.model = None
        self.atmgrid = None
        self.NB_ATM_HEADER = self.index_atm_data + 1
        self.NB_ATM_DATA = len(self.lambdas) - 1
        self.NB_ATM_POINTS = 0
        self.AER_Points = np.array([])
        self.OZ_Points = np.array([])
        self.PWV_Points = np.array([])

        # set the initial grid
        self.set_grid(pwv_grid=pwv_grid, ozone_grid=ozone_grid, aerosol_grid=aerosol_grid, lambdas=self.lambdas)

        self.header = fits.Header()
        if atmgrid_filename != "":
            self.load_file(atmgrid_filename)
        if spectrum_filename != "":
            hdr = fits.getheader(spectrum_filename)
            self.pressure = hdr["OUTPRESS"]
            self.temperature = hdr["OUTTEMP"]
            self.airmass = hdr["AIRMASS"]

    def set_grid(self, pwv_grid=[0, 10, 10], ozone_grid=[100, 700, 7], aerosol_grid=[0, 0.1, 10],
                 lambdas=parameters.LAMBDAS):
        """Set the size of the simulation grid self.atmgrid before compute it.

        The first column of self.atmgrid will contain the wavelengths set by lambdas argument,
        the other columns the future simulations.

        Parameters
        ----------
        pwv_grid: list
            List of 3 numbers for the PWV quantity: min, max, number of simulations (default: [0, 10, 10]).
        ozone_grid: list
            List of 3 numbers for the ozone quantity: min, max, number of simulations (default: [100, 700, 7]).
        aerosol_grid: list
            List of 3 numbers for the aerosol quantity: min, max, number of simulations (default: [0, 0.1, 10]).
        lambdas: array_like, optional
            Array of wavelengths (default: parameters.LAMBDAS).
        """
        self.lambdas = lambdas

        # aerosols
        NB_AER_POINTS = int(aerosol_grid[2])
        AER_MIN = float(aerosol_grid[0])
        AER_MAX = float(aerosol_grid[1])

        # ozone
        NB_OZ_POINTS = int(ozone_grid[2])
        OZ_MIN = float(ozone_grid[0])
        OZ_MAX = float(ozone_grid[1])

        # pwv
        NB_PWV_POINTS = int(pwv_grid[2])
        PWV_MIN = float(pwv_grid[0])
        PWV_MAX = float(pwv_grid[1])

        # definition of the grid
        self.AER_Points = np.linspace(AER_MIN, AER_MAX, NB_AER_POINTS)
        self.OZ_Points = np.linspace(OZ_MIN, OZ_MAX, NB_OZ_POINTS)
        self.PWV_Points = np.linspace(PWV_MIN, PWV_MAX, NB_PWV_POINTS)

        # total number of points
        self.NB_ATM_POINTS = NB_AER_POINTS * NB_OZ_POINTS * NB_PWV_POINTS

        # create the numpy array that will contain the atmospheric grid
        self.atmgrid = np.zeros((self.NB_ATM_POINTS + 1, self.NB_ATM_HEADER + self.NB_ATM_DATA))
        self.atmgrid[0, self.index_atm_data:] = self.lambdas

    def compute(self):
        """Compute atmospheric transmissions and fill self.atmgrid.

        The wavelengths used for the computation are the ones set by self.lambdas.

        Returns
        -------
        atmospheric_grid: array_like
            The atmospheric grid self.atmgrid.

        Examples
        --------
        >>> a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits',
        ... pwv_grid=[5, 5, 1], ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
        >>> atmospheric_grid = a.compute()
        >>> atmospheric_grid  # doctest: +ELLIPSIS
        array([[0.000000e+00, ...
               ...])
        >>> a.save_file(a.image_filename.replace('.fits', '_atmsim.fits'))
        >>> a.plot_transmission()

        .. plot::

            from spectractor.simulation.atmosphere import AtmosphereGrid
            a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits', pwv_grid=[5, 5, 1],
                               ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
            atmospheric_grid = a.compute()
            a.plot_transmission()

        .. doctest::
            :hide:

            >>> assert os.path.isfile(a.image_filename.replace('.fits', '_atmsim.fits'))
            >>> assert np.all(np.isclose(a.atmgrid[0, a.index_atm_data:], parameters.LAMBDAS))
            >>> assert not np.any(np.isclose(a.atmgrid[1, a.index_atm_data:],
            ... np.zeros_like(parameters.LAMBDAS), rtol=1e-6))
            >>> assert a.atmgrid.shape == (3, a.index_atm_data+len(parameters.LAMBDAS))
        """
        # first determine the length
        self.my_logger.debug(f'\n\tAtmosphere simulations for z={self.airmass:4.2f}, P={self.pressure:4.2f}hPa, '
                             rf'T={self.temperature:4.2f}$\degree$C, for data-file={self.image_filename} ')
        count = 0
        for aer in self.AER_Points:
            for pwv in self.PWV_Points:
                for oz in self.OZ_Points:
                    count += 1
                    # fills headers info in the numpy array
                    self.atmgrid[count, self.index_atm_count] = count
                    self.atmgrid[count, self.index_atm_aer] = aer
                    self.atmgrid[count, self.index_atm_pwv] = pwv
                    self.atmgrid[count, self.index_atm_oz] = oz
                    transmission = super(AtmosphereGrid, self).simulate(aerosols=aer, ozone=oz, pwv=pwv)
                    transm = transmission(self.lambdas)
                    self.atmgrid[count, self.index_atm_data:] = transm  # each of atmospheric spectrum
        return self.atmgrid

    def plot_transmission(self, lambdas=parameters.LAMBDAS):
        """Plot the atmospheric transmission contained in the grid.

        Parameters
        ----------
        lambdas: array_like, optional
            Array of wavelengths in nm (default: parameters.LAMBDAS).

        Examples
        --------
        >>> a = AtmosphereGrid(atmgrid_filename='tests/data/reduc_20170530_134_atmsim.fits')
        >>> a.plot_transmission()

        .. plot::

            from spectractor.simulation.atmosphere import AtmosphereGrid
            a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits', pwv_grid=[5, 5, 1],
                               ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
            atmospheric_grid = a.compute()
            a.plot_transmission()

        """
        plt.figure()
        counts = self.atmgrid[1:, self.index_atm_count]
        for count in counts:
            label = f'PWV={self.atmgrid[int(count), self.index_atm_pwv]} ' \
                    f'OZ={self.atmgrid[int(count), self.index_atm_oz]} ' \
                    f'VAOD={self.atmgrid[int(count), self.index_atm_aer]}'
            plot_transmission_simple(plt.gca(), lambdas, np.interp(lambdas, self.lambdas, self.atmgrid[int(count), self.index_atm_data:]),
                                     title="Atmospheric grid", label=label)
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        else:
            plt.close('all')

    def plot_transmission_image(self):
        """Plot the atmospheric transmission contained in the grid using imshow.

        Examples
        --------
        >>> a = AtmosphereGrid(atmgrid_filename='tests/data/reduc_20170530_134_atmsim.fits')
        >>> a.plot_transmission_image()

        .. plot::

            from spectractor.simulation.atmosphere import AtmosphereGrid
            a = AtmosphereGrid(image_filename='tests/data/reduc_20170530_134_atmsim.fits', pwv_grid=[5, 5, 1],
                               ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
            atmospheric_grid = a.compute()
            a.plot_transmission_image()

        """
        plt.figure()
        img = plt.imshow(self.atmgrid[1:, self.index_atm_data:], origin='lower', cmap='jet', aspect="auto")
        plt.grid(True)
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel("Simulation number")
        plt.title("Atmospheric variations")
        cbar = plt.colorbar(img)
        cbar.set_label('Atmospheric transmission')
        if parameters.DISPLAY:
            plt.show()
        else:
            plt.close('all')

    def save_file(self, filename=""):
        """Save the atmospheric grid in a fits file.

        Parameters
        ----------
        filename: str
            The output file name.

        Examples
        --------
        >>> a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits',
        ... pwv_grid=[5, 5, 1], ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
        >>> atmospheric_grid = a.compute()
        >>> a.save_file(a.image_filename.replace('.fits', '_atmsim.fits'))

        .. doctest::
            :hide:

            >>> assert os.path.isfile('tests/data/reduc_20170605_028_atmsim.fits')
        """
        hdr = fits.Header()

        if filename != "":
            self.filename = filename

        if self.filename == "":
            self.my_logger.error('\n\tNo file name is given...')
        else:
            hdr['ATMSIM'] = "libradtran"
            hdr['SIMVERS'] = "2.0.1"
            hdr['DATAFILE'] = self.image_filename
            hdr['SIMUFILE'] = os.path.basename(self.filename)

            hdr['AIRMASS'] = self.airmass
            hdr['PRESSURE'] = self.pressure
            hdr['TEMPERAT'] = self.temperature
            hdr['NBATMPTS'] = self.NB_ATM_POINTS

            hdr['NBAERPTS'] = self.AER_Points.size
            hdr['AERMIN'] = self.AER_Points.min()
            hdr['AERMAX'] = self.AER_Points.max()

            hdr['NBPWVPTS'] = self.PWV_Points.size
            hdr['PWVMIN'] = self.PWV_Points.min()
            hdr['PWVMAX'] = self.PWV_Points.max()

            hdr['NBOZPTS'] = self.OZ_Points.size
            hdr['OZMIN'] = self.OZ_Points.min()
            hdr['OZMAX'] = self.OZ_Points.max()

            hdr['NBWLBIN'] = self.lambdas.size
            hdr['WLMIN'] = np.min(self.lambdas)
            hdr['WLMAX'] = np.max(self.lambdas)

            hdr['IDX_CNT'] = self.index_atm_count
            hdr['IDX_AER'] = self.index_atm_aer
            hdr['IDX_PWV'] = self.index_atm_pwv
            hdr['IDX_OZ'] = self.index_atm_oz
            hdr['IDX_DATA'] = self.index_atm_data

            hdu = fits.PrimaryHDU(self.atmgrid, header=hdr)
            hdu.writeto(self.filename, overwrite=True)
            self.my_logger.info(f'\n\tAtmosphere.save atm-file={self.filename}')

    def load_file(self, filename):
        """Load the atmospheric grid from a fits file and interpolate across the points
        using RegularGridInterpolator. Automatically called from __init__.

        Parameters
        ----------
        filename: str
            The input file name.

        Examples
        --------
        >>> a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits',
        ... pwv_grid=[5, 5, 1], ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
        >>> atmospheric_grid = a.compute()
        >>> a.save_file(a.image_filename.replace('.fits', '_atmsim.fits'))
        >>> assert os.path.isfile('tests/data/reduc_20170605_028_atmsim.fits')
        >>> a.load_file(a.image_filename.replace('.fits', '_atmsim.fits'))
        >>> a.AER_Points
        array([0. , 0.1])
        >>> a.PWV_Points
        array([5.])
        >>> a.OZ_Points
        array([400.])
        """

        if filename != "":
            self.filename = filename

        if self.filename == "":
            self.my_logger.error('\n\tNo file name is given...')
        else:

            with fits.open(self.filename) as hdu:
                hdr = hdu[0].header
                self.header = hdr
                self.image_filename = hdr['DATAFILE']

                self.airmass = hdr['AIRMASS']
                self.pressure = hdr['PRESSURE']
                self.temperature = hdr['TEMPERAT']

                self.NB_ATM_POINTS = hdr['NBATMPTS']

                NB_AER_POINTS = hdr['NBAERPTS']
                AER_MIN = hdr['AERMIN']
                AER_MAX = hdr['AERMAX']

                NB_PWV_POINTS = hdr['NBPWVPTS']
                PWV_MIN = hdr['PWVMIN']
                PWV_MAX = hdr['PWVMAX']

                NB_OZ_POINTS = hdr['NBOZPTS']
                OZ_MIN = hdr['OZMIN']
                OZ_MAX = hdr['OZMAX']

                self.AER_Points = np.linspace(AER_MIN, AER_MAX, NB_AER_POINTS)
                self.OZ_Points = np.linspace(OZ_MIN, OZ_MAX, NB_OZ_POINTS)
                self.PWV_Points = np.linspace(PWV_MIN, PWV_MAX, NB_PWV_POINTS)

                NBWLBINS = hdr['NBWLBIN']

                self.index_atm_count = hdr['IDX_CNT']
                self.index_atm_aer = hdr['IDX_AER']
                self.index_atm_pwv = hdr['IDX_PWV']
                self.index_atm_oz = hdr['IDX_OZ']
                self.index_atm_data = hdr['IDX_DATA']

                self.atmgrid = np.zeros((self.NB_ATM_POINTS + 1, self.NB_ATM_HEADER + NBWLBINS - 1))

                self.atmgrid[:, :] = hdu[0].data[:, :]

            self.my_logger.debug(f'\n\tAtmosphere.load_image atm-file={self.filename}')

            # interpolate the grid
            self.lambdas = self.atmgrid[0, self.index_atm_data:]
            self.model = RegularGridInterpolator((self.lambdas, self.OZ_Points, self.PWV_Points, self.AER_Points), (
                self.atmgrid[1:, self.index_atm_data:].reshape(NB_AER_POINTS, NB_PWV_POINTS,
                                                               NB_OZ_POINTS,
                                                               len(self.lambdas))).T, bounds_error=False, fill_value=0)

    def simulate(self, ozone, pwv, aerosols, angstrom_exponent=None):
        """Interpolate from the atmospheric grid to get the atmospheric transmission.

        First ozone, second pwv, last aerosols, to respect order of loops when generating the grid

        Parameters
        ----------
        ozone: float
            Ozone quantity in Dobson.
        pwv: float
            Precipitable Water Vapor quantity in mm.
        aerosols: float
            VAOD Vertical Aerosols Optical Depth.
        angstrom_exponent: float, optional
            Angstrom exponent for aerosols. If negative or None, default aerosol model from Libradtran is used. If value is 0.0192,
            atmospheric transmission is very close to the case angstrom_exponent negative (default: None).

        Examples
        --------

        .. plot::
            :include-source:

            >>> from spectractor.simulation.atmosphere import AtmosphereGrid, plot_transmission_simple
            >>> from spectractor import parameters
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> a = AtmosphereGrid(atmgrid_filename='tests/data/reduc_20170530_134_atmsim.fits')
            >>> lambdas = np.arange(200, 1200)
            >>> fig = plt.figure()
            >>> for pwv in np.arange(5):
            ...     transmission = a.simulate(ozone=400, pwv=pwv, aerosols=0.05)
            ...     plot_transmission_simple(plt.gca(), lambdas, transmission(lambdas),
            ...     title=a.title, label=a.label)
            >>> if parameters.DISPLAY: plt.show()
        """
        if angstrom_exponent is not None and angstrom_exponent > 0:
            raise ValueError(f"Angstrom exponent not implemented in AtmosphericGrid() yet. "
                             f"Please provide angstrom_exponent=None. Got {angstrom_exponent=} instead.")
        self.pwv = pwv
        self.ozone = ozone
        self.aerosols = aerosols
        self.set_title()
        self.set_label()
        ones = np.ones_like(self.lambdas)
        points = np.array([self.lambdas, ozone * ones, pwv * ones, aerosols * ones]).T
        atm = self.model(points)
        self.transmission = interp1d(self.lambdas, atm, kind='linear', bounds_error=False, fill_value=(0, 0))
        return self.transmission


if __name__ == "__main__":
    import doctest

    doctest.testmod()

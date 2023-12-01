import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from spectractor.config import set_logger
import spectractor.parameters as parameters


def load_transmission(file_name):
    """Load the transmission files and crop in wavelength using LAMBDA_MIN and LAMBDA_MAX.

    The input file must have two or three columns:
        1. wavelengths in nm
        2. transmissions between 0 and 1.
        3. uncertainties on the transmissions (optional)

    Returns
    -------
    lambdas: array_like
        The wavelengths array in nm.
    transmissions: array_like
        The transmission array, values are between 0 and 1.
    uncertainties: array_like
        The uncertainty on the transmission array (0 if file does not contain the info).

    Examples
    --------
    >>> parameters.LAMBDA_MIN = 500
    >>> lambdas, transmissions, errors = load_transmission(os.path.join(parameters.THROUGHPUT_DIR, "qecurve.txt"))
    >>> print(lambdas[:3])
    [500.81855389 508.18553888 519.23601637]
    >>> print(transmissions[:3])
    [0.74355972 0.75526932 0.76932084]
    >>> print(errors[:3])
    [0. 0. 0.]

    """
    if os.path.isabs(file_name):
        path = file_name
    else:
        mypath = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(mypath, file_name)
    data = np.loadtxt(path).T
    lambdas = data[0]
    sorted_indices = lambdas.argsort()
    lambdas = lambdas[sorted_indices]
    y = data[1][sorted_indices]
    err = np.zeros_like(y)
    if data.shape[0] == 3:
        err = data[2][sorted_indices]
    indexes = np.logical_and(lambdas > parameters.LAMBDA_MIN, lambdas < parameters.LAMBDA_MAX)
    return lambdas[indexes], y[indexes], err[indexes]


def plot_transmission_simple(ax, lambdas, transmissions,  uncertainties=None, label="", title="", lw=2):
    """Plot the transmission with respect to the wavelength.

    Parameters
    ----------
    ax: Axes
        An Axes instance.
    lambdas: array_like
        The wavelengths array in nm.
    transmissions: array_like
        The transmission array, values are between 0 and 1.
    uncertainties: array_like, optional
        The uncertainty on the transmission array (default: None).
    label: str, optional
        The label of the curve for the legend (default: "")
    title: str, optional
        The title of the plot (default: "")
    lw: int
        Line width (default: 2).

    Examples
    --------

    .. plot::
        :include-source:

        >>> from spectractor.simulation.atmosphere import plot_transmission_simple
        >>> from spectractor import parameters
        >>> fig = plt.figure()
        >>> ax = plt.gca()
        >>> parameters.LAMBDA_MIN = 500
        >>> lambdas, transmissions, errors = load_transmission(os.path.join(parameters.THROUGHPUT_DIR, "qecurve.txt"))
        >>> plot_transmission_simple(ax, lambdas, transmissions, errors, title="CTIO", label="Quantum efficiency")
        >>> lambdas, transmissions, errors = load_transmission(os.path.join(parameters.THROUGHPUT_DIR, "lsst_mirrorthroughput.txt"))
        >>> plot_transmission_simple(ax, lambdas, transmissions, errors, title="CTIO", label="Mirror 1")
        >>> lambdas, transmissions, errors = load_transmission(os.path.join(parameters.THROUGHPUT_DIR, "FGB37.txt"))
        >>> plot_transmission_simple(ax, lambdas, transmissions, errors, title="CTIO", label="FGB37")
        >>> lambdas, transmissions, errors = load_transmission(os.path.join(parameters.THROUGHPUT_DIR, "RG715.txt"))
        >>> plot_transmission_simple(ax, lambdas, transmissions, errors, title="CTIO", label="RG715")
        >>> lambdas, transmissions, errors = load_transmission(os.path.join(parameters.THROUGHPUT_DIR, parameters.OBS_FULL_INSTRUMENT_TRANSMISSON))
        >>> plot_transmission_simple(ax, lambdas, transmissions, errors, title="CTIO", label="Full instrument")
        >>> if parameters.DISPLAY: plt.show()

    """
    if uncertainties is None or np.all(np.isclose(uncertainties, np.zeros_like(transmissions))):
        ax.plot(lambdas, transmissions, "-", label=label, lw=lw)
    else:
        ax.errorbar(lambdas, transmissions, yerr=uncertainties, label=label, lw=lw)
    if title != "":
        ax.set_title(title)
    ax.set_xlabel(r"$\lambda$ [nm]")
    ax.set_ylabel("Transmission")
    ax.set_xlim(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX)
    ax.grid()
    if label != "":
        ax.legend(loc="best")


class TelescopeTransmission:

    def __init__(self, filter_label=""):
        """Transmission of the telescope as product of the following transmissions:

        - mirrors
        - throughput
        - quantum efficiency
        - Filter

        Parameters
        ----------
        filter_label: str, optional
            The filter string name.

        Examples
        --------
        >>> t = TelescopeTransmission()
        >>> t.plot_transmission()
        """

        self.my_logger = set_logger(self.__class__.__name__)
        self.filter_label = filter_label
        self.transmission = None
        self.transmission_err = None
        self.load_transmission()

    def load_transmission(self):
        """Load the transmission files and make a function.

        Returns
        -------
        transmission: callable
            The transmission function using wavelengths in nm.

        Examples
        --------
        >>> t = TelescopeTransmission()
        >>> t.plot_transmission()

        >>> t2 = TelescopeTransmission(filter_label="RG715")
        >>> t2.plot_transmission()

        .. doctest:
            :hide:

            >>> assert t.transmission is not None
            >>> assert t.transmission_err is not None
            >>> assert t2.transmission is not None
            >>> assert t2.transmission_err is not None
            >>> assert np.sum(t.transmission(parameters.LAMBDAS)) > np.sum(t2.transmission(parameters.LAMBDAS))

        """
        wl, trm, err = load_transmission(os.path.join(parameters.THROUGHPUT_DIR,
                                                      parameters.OBS_FULL_INSTRUMENT_TRANSMISSON))
        to = interp1d(wl, trm, kind='linear', bounds_error=False, fill_value=0.)
        err = np.sqrt(err ** 2 + parameters.OBS_TRANSMISSION_SYSTEMATICS ** 2)
        to_err = interp1d(wl, err, kind='linear', bounds_error=False, fill_value=0.)

        TF = lambda x: 1
        TF_err = lambda x: 0
        if self.filter_label != "" and "empty" not in self.filter_label.lower():
            if ".txt" in self.filter_label:
                filter_filename = self.filter_label
            else:
                filter_filename = self.filter_label + ".txt"
            wl, trb, err = load_transmission(os.path.join(parameters.THROUGHPUT_DIR, filter_filename))
            TF = interp1d(wl, trb, kind='linear', bounds_error=False, fill_value=0.)
            TF_err = interp1d(wl, err, kind='linear', bounds_error=False, fill_value=0.)

        # self.transmission=lambda x: self.qe(x)*self.to(x)*(self.tm(x)**2)*self.tf(x)
        self.transmission = lambda x: to(x) * TF(x)
        self.transmission_err = lambda x: np.sqrt(to_err(x)**2 + TF_err(x)**2)
        return self.transmission

    def plot_transmission(self):
        """Plot the transmission function and the associated uncertainties.

        Examples
        --------
        >>> t = TelescopeTransmission()
        >>> t.plot_transmission()

        """
        plt.figure()
        plot_transmission_simple(plt.gca(), parameters.LAMBDAS, self.transmission(parameters.LAMBDAS),
                                 uncertainties=self.transmission_err(parameters.LAMBDAS))
        if parameters.DISPLAY:
            plt.show()
        else:
            plt.close('all')

    def reset_lambda_range(self, transmission_threshold=1e-4):
        integral = np.cumsum(self.transmission(parameters.LAMBDAS))
        lambda_min = parameters.LAMBDAS[0]
        for k, tr in enumerate(integral):
            if tr > transmission_threshold:
                lambda_min = parameters.LAMBDAS[k]
                break
        lambda_max = parameters.LAMBDAS[0]
        for k, tr in enumerate(integral):
            if tr > integral[-1] - transmission_threshold:
                lambda_max = parameters.LAMBDAS[k]
                break
        parameters.LAMBDA_MIN = max(lambda_min, parameters.LAMBDA_MIN)
        parameters.LAMBDA_MAX = min(lambda_max, parameters.LAMBDA_MAX)
        parameters.LAMBDAS = np.arange(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX, parameters.LAMBDA_STEP)
        self.my_logger.info(f"\n\tWith filter {self.filter_label}, set parameters.LAMBDA_MIN={parameters.LAMBDA_MIN} "
                            f"and parameters.LAMBDA_MAX={parameters.LAMBDA_MAX}.")


if __name__ == "__main__":
    import doctest

    doctest.testmod()

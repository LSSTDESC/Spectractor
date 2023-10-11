from scipy import optimize
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy
import os
import json
from dataclasses import dataclass
from typing import Optional, Union

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.tools import (formatting_numbers, compute_correlation_matrix, plot_correlation_matrix_simple,
                               NumpyArrayEncoder)

from lsst.utils.threads import disable_implicit_threading
disable_implicit_threading()


@dataclass()
class FitParameter:
    """Container for a parameter to fit on data with FitWorkspace.

    Attributes
    ----------
    value: float
        Array containing the parameter values.
    label: str
        Parameter labels for screen print.
    axis_name: str
        Parameter labels for plot print.
    bounds: list
        2-element list giving the (lower, upper) bounds for the parameter.
    fixed: bool
        Boolean indicating whether the parameter is fixed.
    err: float
        Uncertainty on parameter value.
    truth: float, optional
        Truth value.

    Examples
    --------
    >>> from spectractor.fit.fitter import FitParameter
    >>> p = FitParameter(value=1, label="x", axis_name="$x$", bounds=[0, 1], fixed=False, err=0.1, truth=1)
    >>> p
    x: 1.0 +/- 0.1 (truth=1)
    >>> p = FitParameter(value=0.01234, label="t", axis_name="$t$", bounds=[-1, 1], fixed=False, err=0.02)
    >>> p
    t: 0.01 +/- 0.02
    >>> p = FitParameter(value=1, label="x", axis_name="$x$", bounds=[0, 1], fixed=True, err=0)
    >>> p
    x: 1 (fixed)

    """
    value: float
    label: str
    axis_name: str
    bounds: list
    fixed: bool
    err: float
    truth: Optional[float] = None

    def __repr__(self):
        """Print the parameter on screen.

        """
        txt = ""
        if self.fixed:
            txt += f"{self.label}: {self.value} (fixed)"
        else:
            if self.err != 0:
                val, err, _ = formatting_numbers(self.value, self.err, self.err)
                txt += f"{self.label}: {val} +/- {err}"
            else:
                txt += f"{self.label}: {self.value} +/- {self.err}"
        if self.truth:
            txt += f" (truth={self.truth})"
        return txt


@dataclass()
class FitParameters:
    """Container for the parameters to fit on data with FitWorkspace.

    Attributes
    ----------
    values: np.ndarray
        Array containing the parameter values.
    labels: list, optional
        List of the parameter labels for screen print.
        If None, make a default list with parameters labelled par (default: None).
    axis_names: list, optional
        List of the parameter labels for plot print.
        If None, make a default list with parameters labelled par (default: None).
    bounds: list, optional
        List of 2-element list giving the (lower, upper) bounds for every parameter.
        If None, make a default list with np.infinity boundaries (default: None).
    fixed: list, optional
        List of boolean: True to fix a parameter, False to let it free.
        If None, make a default list with False values: all parameters are free (default: None).
    truth: np.ndarray, optional
        Array of truth parameters (default: None).
    filename: str, optional
        File name associated to the fitted parameters (usually _spectrum.fits file name) (default: '').

    Examples
    --------
    >>> from spectractor.fit.fitter import FitParameters
    >>> params = FitParameters(values=[1, 1, 1, 1, 1])
    >>> params.ndim
    5
    >>> params.values
    array([1., 1., 1., 1., 1.])
    >>> params.labels
    ['par0', 'par1', 'par2', 'par3', 'par4']
    >>> params.bounds
    [[-inf, inf], [-inf, inf], [-inf, inf], [-inf, inf], [-inf, inf]]
    """
    values: Union[np.ndarray, list]
    labels: Optional[list] = None
    axis_names: Optional[list] = None
    bounds: Optional[list] = None
    fixed: Optional[list] = None
    truth: Optional[list] = None
    filename: Optional[str] = ""
    extra: Optional[dict] = None

    def __post_init__(self):
        if type(self.values) is list:
            self.values = np.array(self.values, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if not self.labels:
            self.labels = [f"par{k}" for k in range(self.ndim)]
        else:
            if len(self.labels) != self.ndim:
                raise ValueError("labels argument must have same size as values argument.")
        if not self.axis_names:
            self.axis_names = [f"$p_{k}$" for k in range(self.ndim)]
        else:
            if len(self.axis_names) != self.ndim:
                raise ValueError("labels argument must have same size as values argument.")
        if self.bounds is None:
            self.bounds = [[-np.inf, np.inf]] * self.ndim
        else:
            if np.array(self.bounds).shape != (self.ndim, 2):
                raise ValueError(f"bounds argument size {np.array(self.bounds).shape} must be same as values argument {(self.ndim, 2)}.")
        if not self.fixed:
            self.fixed = [False] * self.ndim
        else:
            if len(list(self.fixed)) != self.ndim:
                raise ValueError("fixed argument must have same size as values argument.")
        self.cov = np.zeros((self.nfree, self.nfree))

    def __getitem__(self, label):
        """Get parameter value given its label.

        Parameters
        ----------
        label: str
            The parameter label.

        Returns
        -------
        value: float
            The parameter value.

        Examples
        --------
        >>> params = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> params["z"]
        3.0
        """
        index = self.get_index(label=label)
        return self.values[index]

    def __len__(self):
        """Length of the parameter array, equivalent to self.ndim.

        Examples
        --------
        >>> params = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> len(params)
        4
        >>> len(params) == params.ndim
        True
        """
        return len(self.values)

    def __eq__(self, other):
        """Test parameter instances equality.

        Examples
        --------
        >>> p1 = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> p2 = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> p1 == p2
        True
        >>> p3 = FitParameters(values=[1, 2, 3, 4.1], labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> p1 == p3
        False
        >>> p4 = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"], fixed=[False, False, True, False])
        >>> p1 == p4
        False
        """

        if not isinstance(other, FitParameters):
            return NotImplemented
        out = True
        for key in self.__dict__.keys():
            if isinstance(getattr(self, key), np.ndarray):
                if len(getattr(self, key).flatten()) == len(getattr(other, key).flatten()):
                    out *= np.all(np.equal(getattr(self, key).flatten(), getattr(other, key).flatten()))
                else:
                    # if fixed parameters are not equal, covariance matrices have not the same shape
                    # and multiplication above is forbidden
                    out = False
            else:
                out *= getattr(self, key) == getattr(other, key)
        return out

    def get_index(self, label):
        """Get parameter index given its label.

        Parameters
        ----------
        label: str
            The parameter label.

        Returns
        -------
        index: int
            The parameter index.

        Examples
        --------
        >>> params = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> params.get_index("z")
        2
        """
        if label in self.labels:
            index = self.labels.index(label)
            return index
        else:
            raise KeyError(f"{label=} not in FitParameters.labels ({self.labels=}).")

    def set(self, label, value):
        """Set value parameter given its label.

        Parameters
        ----------
        label: str
            The parameter label.
        value: float
            The new parameter value.

        Examples
        --------
        >>> params = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> params["z"]
        3.0
        >>> params.set("z", 0)
        >>> params["z"]
        0.0
        """
        key = self.get_index(label)
        self.values[key] = value

    @property
    def rho(self):
        """Correlation matrix computed from the covariance matrix

        Returns
        -------
        rho: np.ndarray
            The correlation matrix array.

        Examples
        --------
        >>> from spectractor.fit.fitter import FitParameters
        >>> params = FitParameters(values=[1, 1, 1], axis_names=["x", "y", "z"])
        >>> params.cov = np.array([[2,-0.5,0],[-0.5,2,-1],[0,-1,2]])
        >>> params.rho
        array([[ 1.  , -0.25,  0.  ],
               [-0.25,  1.  , -0.5 ],
               [ 0.  , -0.5 ,  1.  ]])

        """
        return compute_correlation_matrix(self.cov)

    @property
    def err(self):
        """Uncertainties on fitted parameters, as the square root of the covariance matrix diagonal.

        Returns
        -------
        err: np.ndarray
            The uncertainty array.

        Examples
        -------
        >>> from spectractor.fit.fitter import FitParameters
        >>> import numpy as np
        >>> params = FitParameters(values=[1, 1, 1, 1, 1], fixed=[False, True, False, True, False])
        >>> params.cov = np.array([[1,-0.5,0],[-0.5,4,-1],[0,-1,9]])
        >>> params.err
        array([1., 0., 2., 0., 3.])
        """
        err = np.zeros_like(self.values, dtype=float)
        if np.sum(self.fixed) != len(self.fixed):
            err[~np.asarray(self.fixed)] = np.sqrt(np.diag(self.cov))
        return err

    @property
    def ndim(self):
        """Number of parameters.

        Returns
        -------
        ndim: int

        Examples
        --------
        >>> from spectractor.fit.fitter import FitParameters
        >>> params = FitParameters(values=[1, 1, 1, 1, 1])
        >>> params.ndim
        5
        """
        return len(self.values)

    @property
    def nfree(self):
        """Number of free parameters.

        Returns
        -------
        nfree: int

        Examples
        --------
        >>> from spectractor.fit.fitter import FitParameters
        >>> params = FitParameters(values=[1, 1, 1, 1, 1], fixed=[True, False, True, False, True])
        >>> params.nfree
        2
        """
        return len(self.get_free_parameters())

    @property
    def nfixed(self):
        """Number of fixed parameters.

        Returns
        -------
        nfixed: int

        Examples
        --------
        >>> from spectractor.fit.fitter import FitParameters
        >>> params = FitParameters(values=[1, 1, 1, 1, 1], fixed=[True, False, True, False, True])
        >>> params.nfixed
        3
        """
        return len(self.get_fixed_parameters())

    def get_free_parameters(self):
        """Return indices array of free parameters.

        Examples
        --------
        >>> params = FitParameters(values=[1, 1, 1, 1, 1], fixed=None)
        >>> params.fixed
        [False, False, False, False, False]
        >>> params.get_free_parameters()
        array([0, 1, 2, 3, 4])
        >>> params = FitParameters(values=[1, 1, 1, 1, 1], fixed=[True, False, True, False, True])
        >>> params.fixed
        [True, False, True, False, True]
        >>> params.get_free_parameters()
        array([1, 3])

        """
        return np.array(np.where(np.array(self.fixed).astype(int) == 0)[0])

    def get_fixed_parameters(self):
        """Return indices array of fixed parameters.

        Examples
        --------
        >>> params = FitParameters(values=[1, 1, 1, 1, 1], fixed=None)
        >>> params.fixed
        [False, False, False, False, False]
        >>> params.get_fixed_parameters()
        array([], dtype=int64)
        >>> params = FitParameters(values=[1, 1, 1, 1, 1], fixed=[True, False, True, False, True])
        >>> params.fixed
        [True, False, True, False, True]
        >>> params.get_fixed_parameters()
        array([0, 2, 4])

        """
        return np.array(np.where(np.array(self.fixed).astype(int) == 1)[0])

    def print_parameters_summary(self):
        """Print the best fitting parameters on screen.
        Labels are from self.labels.

        Returns
        -------
        txt: str
            The printed text.

        Examples
        --------
        >>> parameters.VERBOSE = True
        >>> params = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> params.cov = np.array([[1, -0.5], [-0.5, 4]])
        >>> _ = params.print_parameters_summary()
        """
        txt = ""
        ifree = self.get_free_parameters()
        icov = 0
        for ip in range(self.ndim):
            if ip in ifree:
                txt += "%s: %s +%s -%s" % formatting_numbers(self.values[ip], np.sqrt(np.abs(self.cov[icov, icov])),
                                                             np.sqrt(np.abs(self.cov[icov, icov])),
                                                             label=self.labels[ip])
                txt += f" bounds={self.bounds[ip]}\n\t"
                icov += 1
            else:
                txt += f"{self.labels[ip]}: {self.values[ip]} (fixed)\n\t"
        return txt

    def get_parameter(self, key):
        """Return a FitParameter instance. key argument can be the parameter label or its index value.

        Parameters
        ----------
        key: str, int
            The parameter key.

        Returns
        -------
        param: FitParameter
            A FitParameter instance containing all information about a parameter.

        Examples
        --------
        >>> params = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> params.cov = np.array([[1, -0.5], [-0.5, 4]])
        >>> p3 = params.get_parameter("t")
        >>> p3
        t: 4 +/- 2
        >>> p0 = params.get_parameter(0)
        >>> p0
        x: 1.0 (fixed)
        """
        if type(key) is str:
            index = self.get_index(key)
        else:
            index = key
        if self.truth is None:
            truth = None
        else:
            truth = self.truth[index]
        p = FitParameter(value=self.values[index], label=self.labels[index],
                         axis_name=self.axis_names[index], bounds=self.bounds[index],
                         fixed=self.fixed[index], err=self.err[index], truth=truth)
        return p

    def plot_correlation_matrix(self, live_fit=False):
        """Compute and plot a correlation matrix.

        Save the plot if parameters.SAVE is True. The output file name is build from self.file_name,
        adding the suffix _correlation.pdf.

        Parameters
        ----------
        live_fit: bool, optional, optional
            If True, model, data and residuals plots are made along the fitting procedure (default: False).

        Examples
        --------
        >>> from spectractor.fit.fitter import FitParameters
        >>> params = FitParameters(values=[1, 1, 1], axis_names=["x", "y", "z"])
        >>> params.cov = np.array([[1,-0.5,0],[-0.5,1,-1],[0,-1,1]])
        >>> params.plot_correlation_matrix()
        """
        ipar = self.get_free_parameters()
        fig = plt.figure()
        plot_correlation_matrix_simple(plt.gca(), self.rho, axis_names=[self.axis_names[i] for i in ipar])
        fig.tight_layout()
        if (parameters.SAVE or parameters.LSST_SAVEFIGPATH) and self.filename != "":  # pragma: no cover
            figname = os.path.splitext(self.filename)[0] + "_correlation.pdf"
            fig.savefig(figname, dpi=100, bbox_inches='tight')
        if parameters.LSST_SAVEFIGPATH:  # pragma: no cover
            figname = os.path.join(parameters.LSST_SAVEFIGPATH, "parameters_correlation.pdf")
            fig.savefig(figname, dpi=100, bbox_inches='tight')
        if parameters.DISPLAY:  # pragma: no cover
            if live_fit:
                plt.draw()
                plt.pause(1e-8)
            else:
                plt.show()

    @property
    def txt_filename(self):
        return os.path.splitext(self.filename)[0] + "_bestfit.txt"

    @property
    def json_filename(self):
        return os.path.splitext(self.filename)[0] + "_bestfit.json"

    def write_text(self, header=""):
        """Save the best fitting parameter summary in a text file.

        The file name is build from self.file_name, adding the suffix _bestfit.txt.

        Parameters
        ----------
        header: str, optional
            A header to add to the file (default: "").

        Examples
        --------
        >>> params = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"],  fixed=[True, False, True, False], filename="test_spectrum.fits")
        >>> params.cov = np.array([[1,-0.5,0],[-0.5,1,-1],[0,-1,1]])
        >>> params.write_text(header="chi2: 1")

        .. doctest::
            :hide:

            >>> assert os.path.isfile(params.txt_filename)
            >>> os.remove(params.txt_filename)
        """
        txt = self.filename + "\n"
        if header != "":
            txt += header + "\n"
        txt += self.print_parameters_summary()
        for row in self.cov:
            txt += np.array_str(row, max_line_width=20 * self.cov.shape[0]) + '\n'
        output_filename = os.path.splitext(self.filename)[0] + "_bestfit.txt"
        f = open(output_filename, 'w')
        f.write(txt)
        f.close()

    def write_json(self):
        pass


def write_fitparameter_json(json_filename, params, extra=None):
    """Save FitParameters attributes as a JSON file.

    Parameters
    ----------
    json_filename: str
        JSON file name.
    params: FitParameters
        A FitParameters instance to save in JSON json_filename.
    extra: dict, optional
        Extra information to write in the JSON file.

    Returns
    -------
    jsontxt: str
        The JSON dictionnary as string.

    Examples
    --------
    >>> params = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"],  fixed=[True, False, True, False], filename="test_spectrum.fits")
    >>> params.cov = np.array([[1,-0.5,0],[-0.5,1,-1],[0,-1,1]])
    >>> jsonstr = write_fitparameter_json(params.json_filename, params, extra={"chi2": 1})
    >>> jsonstr  # doctest: +ELLIPSIS
    '{"values": [1.0, 2.0, 3.0, 4.0], "labels": ["x", "y", "z", "t"],..."extra": {"chi2": 1}...

    .. doctest::
        :hide:

        >>> assert os.path.isfile(params.json_filename)
        >>> os.remove(params.json_filename)
    """
    if json_filename == "":
        raise ValueError("Must provide attribute a JSON filename.")
    if extra:
        params.extra = extra
    jsontxt = json.dumps(params.__dict__, cls=NumpyArrayEncoder)
    with open(json_filename, 'w') as output_json:
        output_json.write(jsontxt)
    return jsontxt


def read_fitparameter_json(json_filename):
    """Read JSON file and store data in FitParameters instance.

    Parameters
    ----------
    json_filename: str
        The JSON file name.

    Returns
    -------
    params: FitParameters
        A FitParameters instance to loaded from JSON json_filename.

    Examples
    --------
    >>> params = FitParameters(values=[1, 2, 3, 4], labels=["x", "y", "z", "t"],  fixed=[True, False, True, False], filename="test_spectrum.fits")
    >>> params.cov = np.array([[1,-0.5,0],[-0.5,1,-1],[0,-1,1]])
    >>> _ = write_fitparameter_json(params.json_filename, params, extra={"chi2": 1})
    >>> new_params = read_fitparameter_json(params.json_filename)
    >>> new_params.values
    array([1., 2., 3., 4.])

    .. doctest::
        :hide:

        >>> assert os.path.isfile(params.json_filename)
        >>> assert params == new_params
        >>> os.remove(params.json_filename)

    """
    params = FitParameters(values=[0])
    with open(json_filename, 'r') as f:
        data = json.load(f)
    for key in ["values", "cov"]:
        data[key] = np.asarray(data[key])
    for key in data:
        setattr(params, key, data[key])
    return params


class FitWorkspace:

    def __init__(self, params=None, file_name="", verbose=False, plot=False, live_fit=False, truth=None):
        """Generic class to create a fit workspace with parameters, bounds and general fitting methods.

        Parameters
        ----------
        params: FitParameters, optional
            The parameters to fit to data (default: None).
        file_name: str, optional
            The generic file name to save results. If file_name=="", nothing is saved ond disk (default: "").
        verbose: bool, optional
            Level of verbosity (default: False).
        plot: bool, optional
            Level of plotting (default: False).
        live_fit: bool, optional
            If True, model, data and residuals plots are made along the fitting procedure (default: False).
        truth: array_like, optional
            Array of true parameters (default: None).

        Examples
        --------
        >>> params = FitParameters(values=[1, 1, 1, 1, 1])
        >>> w = FitWorkspace(params)
        >>> w.params.ndim
        5
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.params = params
        self.filename = file_name
        self.truth = truth
        self.verbose = verbose
        self.plot = plot
        self.live_fit = live_fit
        self.data = None
        self.err = None
        self.data_cov = None
        self.W = None
        self.x = None
        self.outliers = []
        self.mask = []
        self.sigma_clip = 5
        self.model = None
        self.model_err = None
        self.model_noconv = None
        self.params_table = None
        self.costs = np.array([[]])

    def get_bad_indices(self):
        """List of indices that are outliers rejected by a sigma-clipping method or other masking method.

        Returns
        -------
        outliers: list

        Examples
        --------
        >>> w = FitWorkspace()
        >>> w.data = np.array([np.array([1,2,3]), np.array([1,2,3,4])], dtype=object)
        >>> w.outliers = [2, 6]
        >>> w.get_bad_indices()
        [array([2]), array([3])]
        """
        bad_indices = np.asarray(self.outliers, dtype=int)
        if self.data.dtype == object:
            if len(self.outliers) > 0:
                bad_indices = []
                start_index = 0
                for k in range(self.data.shape[0]):
                    mask = np.zeros(self.data[k].size, dtype=bool)
                    outliers = np.asarray(self.outliers)[np.logical_and(np.asarray(self.outliers) > start_index,
                                                                        np.asarray(self.outliers) < start_index +
                                                                        self.data[k].size)]
                    mask[outliers - start_index] = True
                    bad_indices.append(np.arange(self.data[k].size)[mask])
                    start_index += self.data[k].size
            else:
                bad_indices = [[] for _ in range(self.data.shape[0])]
        return bad_indices

    def simulate(self, *p):
        """Compute the model prediction given a set of parameters.

        Parameters
        ----------
        p: array_like
            Array of parameters for the computation of the model.

        Returns
        -------
        x: array_like
            The abscisse of the model prediction.
        model: array_like
            The model prediction.
        model_err: array_like
            The uncertainty on the model prediction.

        Examples
        --------
        >>> w = FitWorkspace()
        >>> p = np.zeros(3)
        >>> x, model, model_err = w.simulate(*p)

        .. doctest::
            :hide:
            >>> assert x is not None

        """
        self.x = np.array([])
        self.model = np.array([])
        self.model_err = np.array([])
        return self.x, self.model, self.model_err

    def plot_fit(self):  # pragma: no cover
        """Generic function to plot the result of the fit for 1D curves.

        Returns
        -------
        fig: plt.FigureClass
            The figure.

        """
        fig = plt.figure()
        plt.errorbar(self.x, self.data, yerr=self.err, fmt='ko', label='Data')
        if self.truth is not None:
            x, truth, truth_err = self.simulate(*self.truth)
            plt.plot(self.x, truth, label="Truth")
        plt.plot(self.x, self.model, label='Best fitting model')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        title = ""
        for i, label in enumerate(self.params.labels):
            if self.params.cov.size > 0:
                err = np.sqrt(self.params.cov[i, i])
                formatting_numbers(self.params.values[i], err, err)
                _, par, err, _ = formatting_numbers(self.params.values[i], err, err, label=label)
                title += rf"{label} = {par} $\pm$ {err}"
            else:
                title += f"{label} = {self.params.values[i]:.3g}"
            if i < self.params.ndim - 1:
                title += ", "
        plt.title(title)
        plt.legend()
        plt.grid()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        return fig

    def weighted_residuals(self, p):  # pragma: nocover
        """Compute the weighted residuals array for a set of model parameters p.

        Parameters
        ----------
        p: array_like
            The array of model parameters.

        Returns
        -------
        residuals: np.array
            The array of weighted residuals.

        """
        x, model, model_err = self.simulate(*p)
        if self.data_cov is None:
            if len(self.outliers) > 0:
                model_err = model_err.flatten()
                err = self.err.flatten()
                res = (model.flatten() - self.data.flatten()) / np.sqrt(model_err * model_err + err * err)
            else:
                res = ((model - self.data) / np.sqrt(model_err * model_err + self.err * self.err)).flatten()
        else:
            if self.data_cov.ndim > 2:
                K = self.data_cov.shape[0]
                if np.any(model_err > 0):
                    cov = [self.data_cov[k] + np.diag(model_err[k] ** 2) for k in range(K)]
                    L = [np.linalg.inv(np.linalg.cholesky(cov[k])) for k in range(K)]
                else:
                    L = [np.linalg.cholesky(self.W[k]) for k in range(K)]
                res = [L[k] @ (model[k] - self.data[k]) for k in range(K)]
                res = np.concatenate(res).ravel()
            else:
                if np.any(model_err > 0):
                    cov = self.data_cov + np.diag(model_err * model_err)
                    L = np.linalg.inv(np.linalg.cholesky(cov))
                else:
                    if not scipy.sparse.issparse(self.W):
                        if self.W.ndim == 1 and self.W.dtype != object:
                            L = np.diag(np.sqrt(self.W))
                        elif self.W.ndim == 2 and self.W.dtype != object:
                            L = np.linalg.cholesky(self.W)
                        else:
                            raise ValueError(f"Case not implemented with self.W.ndim={self.W.ndim} "
                                             f"and self.W.dtype={self.W.dtype}")
                    else:
                        if scipy.sparse.isspmatrix_dia(self.W):
                            L = self.W.sqrt()
                        else:
                            L = np.linalg.cholesky(self.W.toarray())
                res = L @ (model - self.data)
        return res

    def prepare_weight_matrices(self):
        r"""Compute weight matrix :math:`\mathbf{W}` `self.W` as the inverse of data covariance matrix `self.data_cov`.
        Cancel weights of data outliers given by `self.outliers`.

        Examples
        --------
        1D case:

        >>> w = FitWorkspace()
        >>> w.data_cov = 2 * np.ones(4)
        >>> w.prepare_weight_matrices()
        >>> w.W
        array([0.5, 0.5, 0.5, 0.5])

        2D case:

        >>> w = FitWorkspace()
        >>> w.data = np.array([1,2,3])
        >>> w.data_cov = np.diag([1,2,4])
        >>> w.prepare_weight_matrices()
        >>> w.W
        array([[1.  , 0.  , 0.  ],
               [0.  , 0.5 , 0.  ],
               [0.  , 0.  , 0.25]])

        Add outliers:

        >>> w.outliers = np.array([2])
        >>> w.prepare_weight_matrices()
        >>> w.W
        array([[1. , 0. , 0. ],
               [0. , 0.5, 0. ],
               [0. , 0. , 0. ]])

        Use sparse matrix:

        >>> w.W = scipy.sparse.diags(np.ones(3))
        >>> w.W.toarray()
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> w.prepare_weight_matrices()
        >>> w.W.getformat()
        'dia'
        >>> w.W.toarray()
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 0.]])


        """
        # Prepare covariance matrix for data
        if self.data_cov is None:
            self.data_cov = np.asarray(self.err.flatten() ** 2)
        # Prepare inverse covariance matrix for data
        if self.W is None:
            if self.data_cov.ndim == 1 and self.data_cov.dtype != object:
                self.W = 1 / self.data_cov
            elif self.data_cov.ndim == 2 and self.data_cov.dtype != object:
                self.W = np.linalg.inv(self.data_cov)
            elif self.data_cov.dtype is object:
                if self.data_cov[0].ndim == 1:
                    self.W = np.array([1 / self.data_cov[k] for k in range(self.data_cov.shape[0])])
                else:
                    self.W = []
                    for k in range(len(self.data_cov)):
                        L = np.linalg.inv(np.linalg.cholesky(self.data_cov[k]))
                        self.W[k] = L.T @ L
                    self.W = np.asarray(self.W)
        if len(self.outliers) > 0:
            bad_indices = self.get_bad_indices()
            if not scipy.sparse.issparse(self.W):
                if self.W.ndim == 1 and self.W.dtype != object:
                    self.W[bad_indices] = 0
                elif self.W.ndim == 2 and self.W.dtype != object:
                    self.W[:, bad_indices] = 0
                    self.W[bad_indices, :] = 0
                elif self.W.dtype == object:
                    if self.data_cov[0].ndim == 1:
                        for k in range(len(self.W)):
                            self.W[k][bad_indices[k]] = 0
                    else:
                        for k in range(len(self.W)):
                            self.W[k][:, bad_indices[k]] = 0
                            self.W[k][bad_indices[k], :] = 0
                else:
                    raise ValueError(
                        f"Data inverse covariance matrix must be a np.ndarray of dimension 1 or 2,"
                        f"either made of 1D or 2D arrays of equal lengths or not for block diagonal matrices."
                        f"\nHere W type is {type(self.W)}, shape is {self.W.shape} and W is {self.W}.")
            else:
                format = self.W.getformat()
                W = self.W.tocsr()
                W[:, bad_indices] = 0
                W[bad_indices, :] = 0
                W.eliminate_zeros()
                self.W = W.asformat(format=format)

    def compute_W_with_model_error(self, model_err):
        """Propagate model uncertainties to weight matrix W.
        The method add the mode uncertainties in quadrature to the inverse of the weight matrix W
        `self.data_cov` and re-invert it.

        Parameters
        ----------
        model_err: np.ndarray
            Flat array of model uncertainties.

        Returns
        -------
        W: array_like
            Weight matrix with model uncertainties propagated.

        Examples
        --------
        1D case:

        >>> w = FitWorkspace()
        >>> w.data_cov = 2 * np.ones(4)
        >>> w.prepare_weight_matrices()
        >>> w.compute_W_with_model_error(np.sqrt(2) * np.ones(4))
        array([0.25, 0.25, 0.25, 0.25])

        2D case:

        >>> w = FitWorkspace()
        >>> w.data = np.array([1,2,3])
        >>> w.data_cov = np.diag([1,1,1])
        >>> w.prepare_weight_matrices()
        >>> w.compute_W_with_model_error(np.sqrt(3) * np.ones(3))
        array([[0.25, 0.  , 0.  ],
               [0.  , 0.25, 0.  ],
               [0.  , 0.  , 0.25]])

        Add outliers:

        >>> w.outliers = np.array([2])
        >>> w.prepare_weight_matrices()
        >>> w.compute_W_with_model_error(np.sqrt(3) * np.ones(3))
        array([[0.25, 0.  , 0.  ],
               [0.  , 0.25, 0.  ],
               [0.  , 0.  , 0.  ]])

        Use sparse matrix:

        >>> w.W = scipy.sparse.diags(np.ones(3)).tocsr()
        >>> w.W[-1, -1] = 0  # mask last data point
        >>> w.W.toarray()
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 0.]])
        >>> w.compute_W_with_model_error(np.sqrt(3) * np.ones(3)).toarray()
        array([[0.25, 0.  , 0.  ],
               [0.  , 0.25, 0.  ],
               [0.  , 0.  , 0.  ]])

        """
        if np.any(model_err > 0):
            if not scipy.sparse.issparse(self.W):
                zeros = self.W == 0
                if self.W.ndim == 1 and self.W.dtype != object:
                    self.W = 1 / (self.data_cov + model_err * model_err)
                elif self.W.dtype == object:
                    K = len(self.W)
                    if self.W[0].ndim == 1:
                        self.W = [1 / (self.data_cov[k] + model_err[k] * model_err[k]) for k in range(K)]
                    elif self.W[0].ndim == 2:
                        K = len(self.W)
                        cov = [self.data_cov[k] + np.diag(model_err[k] ** 2) for k in range(K)]
                        L = [np.linalg.inv(np.linalg.cholesky(cov[k])) for k in range(K)]
                        self.W = [L[k].T @ L[k] for k in range(K)]
                    else:
                        raise ValueError(f"First element of fitworkspace.W has no ndim attribute or has a dimension above 2. "
                                         f"I get W[0]={self.W[0]}")
                elif self.W.ndim == 2 and self.W.dtype != object:
                    cov = self.data_cov + np.diag(model_err * model_err)
                    self.W = np.linalg.inv(cov)
                # needs to reapply the mask of outliers
                self.W[zeros] = 0
            else:
                cov = self.data_cov + np.diag(model_err * model_err)
                L = np.linalg.inv(np.linalg.cholesky(cov))
                W = L.T @ L
                # needs to reapply the mask of outliers
                rows, cols = self.W.nonzero()
                self.W = scipy.sparse.csr_matrix((W[rows, cols], (rows, cols)), dtype=self.W.dtype, shape=self.W.shape)
        return self.W

    def chisq(self, p, model_output=False):
        """Compute the chi square for a set of model parameters p.

        Four cases are implemented: diagonal W, 2D W, array of diagonal Ws, array of 2D Ws. The two latter cases
        are for multiple independent data vectors with W being block diagonal.

        Parameters
        ----------
        p: array_like
            The array of model parameters.
        model_output: bool, optional
            If true, the simulated model is output.

        Returns
        -------
        chisq: float
            The chi square value.

        """
        # check data format
        if (self.data.dtype != object and self.data.ndim > 1) or (self.err.dtype != object and self.err.ndim > 1):
            raise ValueError("Fitworkspace.data and Fitworkspace.err must be a flat 1D array,"
                             " or an array of flat arrays of unequal lengths.")
        # prepare weight matrices in case they have not been built before
        self.prepare_weight_matrices()
        x, model, model_err = self.simulate(*p)
        W = self.compute_W_with_model_error(model_err)
        if not scipy.sparse.issparse(self.W):
            if W.ndim == 1 and W.dtype != object:
                res = (model - self.data)
                chisq = res @ (W * res)
            elif W.dtype == object:
                K = len(W)
                res = [model[k] - self.data[k] for k in range(K)]
                if W[0].ndim == 1:
                    chisq = np.sum([res[k] @ (W[k] * res[k]) for k in range(K)])
                elif W[0].ndim == 2:
                    chisq = np.sum([res[k] @ W[k] @ res[k] for k in range(K)])
                else:
                    raise ValueError(f"First element of fitworkspace.W has no ndim attribute or has a dimension above 2. "
                                     f"I get W[0]={W[0]}")
            elif W.ndim == 2 and W.dtype != object:
                res = (model - self.data)
                chisq = res @ W @ res
            else:
                raise ValueError(
                    f"Weight covariance matrix must be a np.ndarray of dimension 1 or 2 if not sparse,"
                    f"either made of 1D or 2D arrays of equal lengths or not for block diagonal matrices."
                    f"\nHere W type is {type(W)}, shape is {W.shape} and W is {W}.")
        else:
            res = (model - self.data)
            chisq = res @ W @ res
        if model_output:
            return chisq, x, model, model_err
        else:
            return chisq

    def lnlike(self, p):
        """Compute the logarithmic likelihood for a set of model parameters p as -0.5*chisq.

        Parameters
        ----------
        p: array_like
            The array of model parameters.

        Returns
        -------
        lnlike: float
            The logarithmic likelihood value.

        """
        return -0.5 * self.chisq(p)

    def jacobian(self, params, epsilon, model_input=None):
        """Generic function to compute the Jacobian matrix of a model, with numerical derivatives.

        Parameters
        ----------
        params: array_like
            The array of model parameters.
        epsilon: array_like
            The array of small steps to compute the partial derivatives of the model.
        model_input: array_like, optional
            A model input as a list with (x, model, model_err) to avoid an additional call to simulate().

        Returns
        -------
        J: np.array
            The Jacobian matrix.

        """
        if model_input:
            x, model, model_err = model_input
        else:
            x, model, model_err = self.simulate(*params)
        if self.W.dtype == object and self.W[0].ndim == 2:
            J = [[] for _ in range(params.size)]
        else:
            model = model.flatten()
            J = np.zeros((params.size, model.size))
        for ip, p in enumerate(params):
            if self.params.fixed[ip]:
                continue
            tmp_p = np.copy(params)
            if tmp_p[ip] + epsilon[ip] < self.params.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.params.bounds[ip][1]:
                epsilon[ip] = - epsilon[ip]
            tmp_p[ip] += epsilon[ip]
            tmp_x, tmp_model, tmp_model_err = self.simulate(*tmp_p)
            if self.W.dtype == object and self.W[0].ndim == 2:
                for k in range(model.shape[0]):
                    J[ip].append((tmp_model[k] - model[k]) / epsilon[ip])
            else:
                J[ip] = (tmp_model.flatten() - model) / epsilon[ip]
        return np.asarray(J)

    def hessian(self, params, epsilon):  # pragma: nocover
        """Experimental function to compute the hessian of a model.

        Parameters
        ----------
        params: array_like
            The array of model parameters.
        epsilon: array_like
            The array of small steps to compute the partial derivatives of the model.

        Returns
        -------

        """
        x, model, model_err = self.simulate(*params)
        model = model.flatten()
        J = self.jacobian(params, epsilon)
        H = np.zeros((params.size, params.size, model.size))
        tmp_p = np.copy(params)
        for ip, p1 in enumerate(params):
            print(ip, p1, params[ip], tmp_p[ip], self.params.bounds[ip], epsilon[ip], tmp_p[ip] + epsilon[ip])
            if self.params.fixed[ip]:
                continue
            if tmp_p[ip] + epsilon[ip] < self.params.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.params.bounds[ip][1]:
                epsilon[ip] = - epsilon[ip]
            tmp_p[ip] += epsilon[ip]
            print(tmp_p)
            # tmp_x, tmp_model, tmp_model_err = self.simulate(*tmp_p)
            # J[ip] = (tmp_model.flatten() - model) / epsilon[ip]
        tmp_J = self.jacobian(tmp_p, epsilon)
        for ip, p1 in enumerate(params):
            if self.params.fixed[ip]:
                continue
            for jp, p2 in enumerate(params):
                if self.params.fixed[jp]:
                    continue
                x, modelplus, model_err = self.simulate(params + epsilon)
                x, modelmoins, model_err = self.simulate(params - epsilon)
                model = model.flatten()

                print("hh", ip, jp, tmp_J[ip], J[jp], tmp_p[ip], params, (tmp_J[ip] - J[jp]) / epsilon)
                print((modelplus + modelmoins - 2 * model) / (np.asarray(epsilon) ** 2))
                H[ip, jp] = (tmp_J[ip] - J[jp]) / epsilon
                H[ip, jp] = (modelplus + modelmoins - 2 * model) / (np.asarray(epsilon) ** 2)
        return H

    def plot_gradient_descent(self):
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex="all")
        iterations = np.arange(self.params_table.shape[0])
        ax[0].plot(iterations, self.costs, lw=2)
        for ip in range(self.params_table.shape[1]):
            ax[1].plot(iterations, self.params_table[:, ip], label=f"{self.params.axis_names[ip]}")
        ax[1].set_yscale("symlog")
        ax[1].legend(ncol=6, loc=9)
        ax[1].grid()
        ax[0].set_yscale("log")
        ax[0].set_ylabel(r"$\chi^2$")
        ax[1].set_ylabel("Parameters")
        ax[0].grid()
        ax[1].set_xlabel("Iterations")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        if parameters.SAVE and self.filename != "":  # pragma: no cover
            figname = os.path.splitext(self.filename)[0] + "_fitting.pdf"
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.PdfPages:  # args from the above? MFL
            parameters.PdfPages.savefig()

        self.simulate(*self.params.values)
        self.live_fit = False
        self.plot_fit()

    def save_gradient_descent(self):
        iterations = np.arange(self.params_table.shape[0]).astype(int)
        t = np.zeros((self.params_table.shape[1] + 2, self.params_table.shape[0]))
        t[0] = iterations
        t[1] = self.costs
        t[2:] = self.params_table.T
        h = 'iter,costs,' + ','.join(self.params.labels)
        output_filename = os.path.splitext(self.filename)[0] + "_fitting.txt"
        np.savetxt(output_filename, t.T, header=h, delimiter=",")
        self.my_logger.info(f"\n\tSave gradient descent log {output_filename}.")


def gradient_descent(fit_workspace, epsilon, niter=10, xtol=1e-3, ftol=1e-3, with_line_search=True):
    """

    Four cases are implemented: diagonal W, 2D W, array of diagonal Ws, array of 2D Ws. The two latter cases
    are for multiple independent data vectors with W being block diagonal.

    Parameters
    ----------
    fit_workspace: FitWorkspace
    epsilon
    niter
    xtol
    ftol
    with_line_search

    Returns
    -------

    """
    my_logger = set_logger(__name__)
    tmp_params = np.copy(fit_workspace.params.values).astype(float)
    fit_workspace.prepare_weight_matrices()
    n_data_masked = len(fit_workspace.mask) + len(fit_workspace.outliers)
    ipar = fit_workspace.params.get_free_parameters()
    costs = []
    params_table = []
    inv_JT_W_J = np.zeros((len(ipar), len(ipar)), dtype=float)

    for i in range(niter):
        start = time.time()
        cost, tmp_lambdas, tmp_model, tmp_model_err = fit_workspace.chisq(tmp_params, model_output=True)
        if i == 0 and fit_workspace.verbose:
            my_logger.info(f"\n\tIteration={i}:\tinitial cost={cost:.5g}\tinitial chisq_red={cost / (tmp_model.size - n_data_masked):.5g}")
        W = fit_workspace.compute_W_with_model_error(tmp_model_err)
        # residuals
        if (isinstance(W, np.ndarray) or scipy.sparse.issparse(W)) and W.dtype != object:
            residuals = (tmp_model - fit_workspace.data).flatten()
        elif isinstance(W, np.ndarray) and W.dtype == object:
            residuals = [(tmp_model[k] - fit_workspace.data[k]) for k in range(len(W))]
        else:
            raise TypeError(f"Type of fit_workspace.W is {type(W)}. It must be a np.ndarray.")
        # Jacobian
        J = fit_workspace.jacobian(tmp_params, epsilon, model_input=[tmp_lambdas, tmp_model, tmp_model_err])
        # remove parameters with unexpected null Jacobian vectors or that are degenerated
        J_vectors = [np.array(J[ip]).ravel() for ip in range(J.shape[0])]
        J_norms = [np.linalg.norm(J_vectors[ip]) for ip in range(J.shape[0])]
        for ip in range(J.shape[0]):
            if ip not in ipar:
                continue
            # check for null vectors
            if J_norms[ip] < 1e-20:
                ipar = np.delete(ipar, list(ipar).index(ip))
                fit_workspace.params.fixed[ip] = True
                my_logger.warning(
                    f"\n\tStep {i}: {fit_workspace.params.labels[ip]} has a null Jacobian; parameter is fixed "
                    f"at its last known current value ({tmp_params[ip]}).")
                continue
            # check for degeneracies using Cauchy-Schwartz inequality; fix the second parameter
            for jp in range(ip, J.shape[0]):
                if ip == jp or jp not in ipar:
                    continue
                inner = np.abs(J_vectors[ip] @  J_vectors[jp])
                if np.abs(inner - J_norms[ip] * J_norms[jp]) < 1e-8 * inner:
                    ipar = np.delete(ipar, list(ipar).index(jp))
                    fit_workspace.params.fixed[jp] = True
                    my_logger.warning(
                        f"\n\tStep {i}: {fit_workspace.params.labels[ip]} is degenerated with {fit_workspace.params.labels[jp]}; "
                        f"parameter {fit_workspace.params.labels[jp]} is fixed at its last known current value ({tmp_params[jp]}).")
                    continue
        # remove fixed and degenerated parameters; then transpose
        J = J[ipar].T

        # compute J.T @ W @ J matrix and invert it
        if W.ndim == 1 and W.dtype != object:
            JT_W = J.T * W
            JT_W_J = JT_W @ J
        elif W.ndim == 2 and W.dtype != object:
            JT_W = J.T @ W
            JT_W_J = JT_W @ J
        else:
            if W[0].ndim == 1:
                JT_W = np.array([j for j in J]).T * np.concatenate(W).ravel()
                JT_W_J = JT_W @ np.array([j for j in J])
            else:
                # warning ! here the data arrays indexed by k can have different lengths because outliers
                # because W inverse covariance is block diagonal and blocks can have different sizes
                # the philosophy is to temporarily flatten the data arrays
                JT_W = [np.concatenate([J[ip][k].T @ W[k]
                                        for k in range(W.shape[0])]).ravel()
                        for ip in range(len(J))]
                JT_W_J = np.array([[JT_W[ip2] @ np.concatenate(J[ip1][:]).ravel() for ip1 in range(len(J))]
                                   for ip2 in range(len(J))])
        try:
            L = np.linalg.inv(np.linalg.cholesky(JT_W_J))  # cholesky is too sensible to the numerical precision
            inv_JT_W_J = L.T @ L
        except np.linalg.LinAlgError:
            inv_JT_W_J = np.linalg.inv(JT_W_J)
        if fit_workspace.W.dtype != object:
            JT_W_R0 = JT_W @ residuals
        else:
            JT_W_R0 = JT_W @ np.concatenate(residuals).ravel()

        # Gauss-Newton step:
        dparams = - inv_JT_W_J @ JT_W_R0
        new_params = np.copy(tmp_params)
        new_params[ipar] = tmp_params[ipar] + dparams

        # check bounds
        for ip, p in enumerate(new_params):
            if p < fit_workspace.params.bounds[ip][0]:
                new_params[ip] = fit_workspace.params.bounds[ip][0]
            if p > fit_workspace.params.bounds[ip][1]:
                new_params[ip] = fit_workspace.params.bounds[ip][1]

        fval = fit_workspace.chisq(new_params)

        if with_line_search or fval > (1 + 10 * ftol) * cost:
            def line_search(alpha):
                tmp_params_2 = np.copy(tmp_params)
                tmp_params_2[ipar] = tmp_params[ipar] + alpha * dparams
                for ipp, pp in enumerate(tmp_params_2):
                    if pp < fit_workspace.params.bounds[ipp][0]:
                        tmp_params_2[ipp] = fit_workspace.params.bounds[ipp][0]
                    if pp > fit_workspace.params.bounds[ipp][1]:
                        tmp_params_2[ipp] = fit_workspace.params.bounds[ipp][1]
                return fit_workspace.chisq(tmp_params_2)

            # tol parameter acts on alpha (not func)
            alpha_min, fval, iter, funcalls = optimize.brent(line_search, full_output=True, tol=5e-1, brack=(0, 1))
            new_params[ipar] = tmp_params[ipar] + alpha_min * dparams
            # check bounds
            for ip, p in enumerate(new_params):
                if p < fit_workspace.params.bounds[ip][0]:
                    new_params[ip] = fit_workspace.params.bounds[ip][0]
                if p > fit_workspace.params.bounds[ip][1]:
                    new_params[ip] = fit_workspace.params.bounds[ip][1]
        else:
            alpha_min = 1
            funcalls = 0
            iter = 0

        tmp_params[ipar] = new_params[ipar]

        # prepare outputs
        costs.append(fval)
        params_table.append(np.copy(tmp_params))
        fit_workspace.params.values = tmp_params
        if fit_workspace.verbose:
            my_logger.info(f"\n\tIteration={i}:\tfinal cost={fval:.5g}\tfinal chisq_red={fval / (tmp_model.size - n_data_masked):.5g} "
                           f"\tcomputed in {time.time() - start:.2f}s"
                           f"\n\tNew parameters: {tmp_params[ipar]}")
            my_logger.debug(f"\n\t Parameter shifts: {alpha_min * dparams}\n"
                            f"\n\t Line search: alpha_min={alpha_min:.3g} iter={iter} funcalls={funcalls}")
        if fit_workspace.live_fit:  # pragma: no cover
            fit_workspace.simulate(*tmp_params)
            fit_workspace.plot_fit()
            fit_workspace.cov = inv_JT_W_J
            # fit_workspace.params.plot_correlation_matrix(ipar)
        if len(ipar) == 0:
            my_logger.warning(f"\n\tGradient descent terminated in {i} iterations because all parameters "
                              f"have null Jacobian.")
            break
        if np.sum(np.abs(alpha_min * dparams)) / np.sum(np.abs(tmp_params[ipar])) < xtol:
            my_logger.info(f"\n\tGradient descent terminated in {i} iterations because the sum of parameter shift "
                           f"relative to the sum of the parameters is below xtol={xtol}.")
            break
        if len(costs) > 1 and np.abs(costs[-2] - fval) / np.max([np.abs(fval), np.abs(costs[-2])]) < ftol:
            my_logger.info(f"\n\tGradient descent terminated in {i} iterations because the "
                           f"relative change of cost is below ftol={ftol}.")
            break
    return tmp_params, inv_JT_W_J, np.array(costs), np.array(params_table)


def simple_newton_minimisation(fit_workspace, epsilon, niter=10, xtol=1e-3, ftol=1e-3):  # pragma: no cover
    """Experimental function to minimize a function.

    Parameters
    ----------
    fit_workspace: FitWorkspace
    epsilon
    niter
    xtol
    ftol

    """
    my_logger = set_logger(__name__)
    tmp_params = np.copy(fit_workspace.params.values)
    ipar = fit_workspace.params.get_free_parameters()
    funcs = []
    params_table = []
    inv_H = np.zeros((len(ipar), len(ipar)))
    for i in range(niter):
        start = time.time()
        tmp_lambdas, tmp_model, tmp_model_err = fit_workspace.simulate(*tmp_params)
        # if fit_workspace.live_fit:
        #    fit_workspace.plot_fit()
        J = fit_workspace.jacobian(tmp_params, epsilon)
        # remove parameters with unexpected null Jacobian vectors
        for ip in range(J.shape[0]):
            if ip not in ipar:
                continue
            if np.all(J[ip] == np.zeros(J.shape[1])):
                ipar = np.delete(ipar, list(ipar).index(ip))
                # tmp_params[ip] = 0
                my_logger.warning(
                    f"\n\tStep {i}: {fit_workspace.params.labels[ip]} has a null Jacobian; parameter is fixed "
                    f"at its last known current value ({tmp_params[ip]}).")
        # remove fixed parameters
        J = J[ipar].T
        # hessian
        H = fit_workspace.hessian(tmp_params, epsilon)
        try:
            L = np.linalg.inv(np.linalg.cholesky(H))  # cholesky is too sensible to the numerical precision
            inv_H = L.T @ L
        except np.linalg.LinAlgError:
            inv_H = np.linalg.inv(H)
        dparams = - inv_H[:, :, 0] @ J[:, 0]
        print("dparams", dparams, inv_H, J, H)
        tmp_params[ipar] += dparams

        # check bounds
        print("tmp_params", tmp_params, dparams, inv_H, J)
        for ip, p in enumerate(tmp_params):
            if p < fit_workspace.params.bounds[ip][0]:
                tmp_params[ip] = fit_workspace.params.bounds[ip][0]
            if p > fit_workspace.params.bounds[ip][1]:
                tmp_params[ip] = fit_workspace.params.bounds[ip][1]

        tmp_lambdas, new_model, tmp_model_err = fit_workspace.simulate(*tmp_params)
        new_func = new_model[0]
        funcs.append(new_func)

        r = np.log10(fit_workspace.regs)
        js = [fit_workspace.jacobian(np.asarray([rr]), epsilon)[0] for rr in np.array(r)]
        plt.plot(r, js, label="J")
        plt.grid()
        plt.legend()
        plt.show()

        if parameters.DISPLAY:
            fig = plt.figure()
            plt.plot(r, js, label="prior")
            mod = tmp_model + J[0] * (r - (tmp_params - dparams)[0])
            plt.plot(r, mod)
            plt.axvline(tmp_params)
            plt.axhline(tmp_model)
            plt.grid()
            plt.legend()
            plt.draw()
            plt.pause(1e-8)
            plt.close(fig)

        # prepare outputs
        params_table.append(np.copy(tmp_params))
        if fit_workspace.verbose:
            my_logger.info(f"\n\tIteration={i}: initial func={tmp_model[0]:.5g}"
                           f"\n\tParameter shifts: {dparams}"
                           f"\n\tNew parameters: {tmp_params[ipar]}"
                           f"\n\tFinal func={new_func:.5g}"
                           f" computed in {time.time() - start:.2f}s")
        if fit_workspace.live_fit:
            fit_workspace.simulate(*tmp_params)
            fit_workspace.plot_fit()
            fit_workspace.cov = inv_H[:, :, 0]
            print("shape", fit_workspace.cov.shape)
            # fit_workspace.params.plot_correlation_matrix(ipar)
        if len(ipar) == 0:
            my_logger.warning(f"\n\tGradient descent terminated in {i} iterations because all parameters "
                              f"have null Jacobian.")
            break
        if np.sum(np.abs(dparams)) / np.sum(np.abs(tmp_params[ipar])) < xtol:
            my_logger.info(f"\n\tGradient descent terminated in {i} iterations because the sum of parameter shift "
                           f"relative to the sum of the parameters is below xtol={xtol}.")
            break
        if len(funcs) > 1 and np.abs(funcs[-2] - new_func) / np.max([np.abs(new_func), np.abs(funcs[-2])]) < ftol:
            my_logger.info(f"\n\tGradient descent terminated in {i} iterations because the "
                           f"relative change of cost is below ftol={ftol}.")
            break
    plt.close()
    return tmp_params, inv_H[:, :, 0], np.array(funcs), np.array(params_table)


def run_gradient_descent(fit_workspace, epsilon, xtol, ftol, niter, verbose=False, with_line_search=True):
    if fit_workspace.costs.size == 0:
        fit_workspace.costs = np.array([fit_workspace.chisq(fit_workspace.params.values)])
        fit_workspace.params_table = np.array([fit_workspace.params.values])
    p, cov, tmp_costs, tmp_params_table = gradient_descent(fit_workspace, epsilon, niter=niter, xtol=xtol, ftol=ftol,
                                                           with_line_search=with_line_search)
    fit_workspace.params.values, fit_workspace.params.cov = p, cov
    fit_workspace.params_table = np.concatenate([fit_workspace.params_table, tmp_params_table])
    fit_workspace.costs = np.concatenate([fit_workspace.costs, tmp_costs])
    if verbose or fit_workspace.verbose:
        fit_workspace.my_logger.info(f"\n\t{fit_workspace.params.print_parameters_summary()}")
    if parameters.DEBUG and (verbose or fit_workspace.verbose):
        fit_workspace.plot_gradient_descent()
        if len(fit_workspace.params.get_free_parameters()) > 1:
            fit_workspace.params.plot_correlation_matrix()


def run_simple_newton_minimisation(fit_workspace, epsilon, xtol=1e-8, ftol=1e-8, niter=50, verbose=False):  # pragma: no cover
    fit_workspace.values, fit_workspace.cov, funcs, params_table = simple_newton_minimisation(fit_workspace,
                                                                                              epsilon, niter=niter,
                                                                                              xtol=xtol, ftol=ftol)
    if verbose or fit_workspace.verbose:
        fit_workspace.my_logger.info(f"\n\t{fit_workspace.params.print_parameters_summary()}")
    if parameters.DEBUG and (verbose or fit_workspace.verbose):
        fit_workspace.plot_gradient_descent()
        if len(fit_workspace.params.get_free_parameters()) > 1:
            fit_workspace.params.plot_correlation_matrix()
    return params_table, funcs


def run_minimisation(fit_workspace, method="newton", epsilon=None, xtol=1e-4, ftol=1e-4, niter=50,
                     verbose=False, with_line_search=True, minimizer_method="L-BFGS-B"):
    my_logger = set_logger(__name__)

    bounds = fit_workspace.params.bounds

    nll = lambda params: -fit_workspace.lnlike(params)

    guess = fit_workspace.params.values.astype('float64')
    if verbose:
        my_logger.debug(f"\n\tStart guess: {guess}")

    if method == "minimize":
        start = time.time()
        result = optimize.minimize(nll, fit_workspace.params.values, method=minimizer_method,
                                   options={'maxiter': 100000}, bounds=bounds)
        fit_workspace.params.values = result['x']
        if verbose:
            my_logger.debug(f"\n\t{result}")
            my_logger.debug(f"\n\tMinimize: total computation time: {time.time() - start}s")
            if parameters.DEBUG:
                fit_workspace.plot_fit()
    elif method == 'basinhopping':
        start = time.time()
        minimizer_kwargs = dict(method=minimizer_method, bounds=bounds)
        result = optimize.basinhopping(nll, guess, minimizer_kwargs=minimizer_kwargs)
        fit_workspace.params.values = result['x']
        if verbose:
            my_logger.debug(f"\n\t{result}")
            my_logger.debug(f"\n\tBasin-hopping: total computation time: {time.time() - start}s")
            if parameters.DEBUG:
                fit_workspace.plot_fit()
    elif method == "least_squares":  # pragma: no cover
        fit_workspace.my_logger.warning("least_squares might not work, use with caution... "
                                        "or repair carefully the function weighted_residuals()")
        start = time.time()
        x_scale = np.abs(guess)
        x_scale[x_scale == 0] = 0.1
        p = optimize.least_squares(fit_workspace.weighted_residuals, guess, verbose=2, ftol=1e-6, x_scale=x_scale,
                                   diff_step=0.001, bounds=bounds.T)
        fit_workspace.params.values = p.x  # m.np_values()
        if verbose:
            my_logger.debug(f"\n\t{p}")
            my_logger.debug(f"\n\tLeast_squares: total computation time: {time.time() - start}s")
            if parameters.DEBUG:
                fit_workspace.plot_fit()
    elif method == "lm":  # pragma: no cover
        if epsilon is None:
            epsilon = 1e-4 * guess
            epsilon[epsilon == 0] = 1e-4

        def Dfun(params):
            return fit_workspace.jacobian(params, epsilon=epsilon, model_input=None).T

        start = time.time()
        x, cov, infodict, mesg, ier = optimize.leastsq(fit_workspace.weighted_residuals, guess, Dfun=Dfun,
                                                       ftol=ftol, xtol=xtol, full_output=True)
        fit_workspace.params.values = x
        fit_workspace.params.cov = cov
        if verbose:
            my_logger.debug(f"\n\t{x}")
            my_logger.debug(f"\n\tLeast_squares: total computation time: {time.time() - start}s")
            if parameters.DEBUG:
                fit_workspace.plot_fit()
    elif method == "newton":
        if epsilon is None:
            epsilon = 1e-4 * guess
            epsilon[epsilon == 0] = 1e-4

        start = time.time()
        run_gradient_descent(fit_workspace, epsilon, xtol=xtol, ftol=ftol, niter=niter, verbose=verbose,
                             with_line_search=with_line_search)
        if verbose:
            my_logger.debug(f"\n\tNewton: total computation time: {time.time() - start}s")
        if fit_workspace.filename != "":
            write_fitparameter_json(fit_workspace.params.json_filename, fit_workspace.params)
            fit_workspace.save_gradient_descent()


def run_minimisation_sigma_clipping(fit_workspace, method="newton", epsilon=None, xtol=1e-4, ftol=1e-4,
                                    niter=50, sigma_clip=5.0, niter_clip=3, verbose=False, with_line_search=True):
    my_logger = set_logger(__name__)
    fit_workspace.sigma_clip = sigma_clip
    for step in range(niter_clip):
        if verbose:
            my_logger.info(f"\n\tSigma-clipping step {step}/{niter_clip} (sigma={sigma_clip})")
        run_minimisation(fit_workspace, method=method, epsilon=epsilon, xtol=xtol, ftol=ftol, niter=niter,
                         with_line_search=with_line_search, verbose=verbose)
        # remove outliers
        if fit_workspace.data.dtype == object:
            # indices_no_nan = ~np.isnan(np.concatenate(fit_workspace.data).ravel())
            data = np.concatenate(fit_workspace.data).ravel()  # [indices_no_nan]
            model = np.concatenate(fit_workspace.model).ravel()  # [indices_no_nan]
            err = np.concatenate(fit_workspace.err).ravel()  # [indices_no_nan]
        else:
            # indices_no_nan = ~np.isnan(fit_workspace.data.flatten())
            data = fit_workspace.data.flatten()  # [indices_no_nan]
            model = fit_workspace.model.flatten()  # [indices_no_nan]
            err = fit_workspace.err.flatten()  # [indices_no_nan]
        residuals = np.abs(data - model) / err
        outliers = residuals > sigma_clip
        outliers = [i for i in range(data.size) if outliers[i]]
        outliers.sort()
        if len(outliers) > 0:
            my_logger.debug(f'\n\tOutliers flat index list: {outliers}')
            my_logger.info(f'\n\tOutliers: {len(outliers)} / {data.size - len(fit_workspace.mask)} data points '
                           f'({100 * len(outliers) / (data.size - len(fit_workspace.mask)):.2f}%) '
                           f'at more than {sigma_clip}-sigma from best-fit model.')
            if np.all(fit_workspace.outliers == outliers):
                my_logger.info(f'\n\tOutliers flat index list unchanged since last iteration: '
                               f'break the sigma clipping iterations.')
                break
            else:
                fit_workspace.outliers = outliers
        else:
            my_logger.info(f'\n\tNo outliers detected at first iteration: break the sigma clipping iterations.')
            break


class RegFitWorkspace(FitWorkspace):

    def __init__(self, w, opt_reg=parameters.PSF_FIT_REG_PARAM, verbose=False, live_fit=False):
        """

        Parameters
        ----------
        w: ChromaticPSFFitWorkspace, FullFowardModelFitWorkspace
            FitWorkspace instance where to apply regularisation.
        opt_reg: float
            Input value for optimal regularisation parameter (default: parameters.PSF_FIT_REG_PARAM).
        verbose: bool, optional
            Level of verbosity (default: False).
        live_fit: bool, optional
            If True, model, data and residuals plots are made along the fitting procedure (default: False).

        """
        params = FitParameters(np.asarray([np.log10(opt_reg)]), labels=["log10_reg"],
                               axis_names=[r"$\log_{10} r$"], fixed=None,
                               bounds=[(-20, np.log10(w.amplitude_priors.size) + 2)])
        FitWorkspace.__init__(self, params, verbose=verbose, live_fit=live_fit)
        self.x = np.array([0])
        self.data = np.array([0])
        self.err = np.array([1])
        self.w = w
        self.opt_reg = opt_reg
        self.resolution = np.zeros_like((self.w.amplitude_params.size, self.w.amplitude_params.size))
        self.G = 0
        self.chisquare = -1

    def print_regularisation_summary(self):
        self.my_logger.info(f"\n\tOptimal regularisation parameter: {self.opt_reg}"
                            f"\n\tTr(R) = N_dof = {np.trace(self.resolution)}"
                            f"\n\tN_params = {len(self.w.amplitude_params)}"
                            f"\n\tN_data = {self.w.data.size - len(self.w.mask) - len(self.w.outliers)}"
                            f" (excluding masked pixels and outliers)")

    def simulate(self, log10_r):
        reg = 10 ** log10_r
        M_dot_W_dot_M_plus_Q = self.w.M_dot_W_dot_M + reg * self.w.Q
        try:
            L = np.linalg.inv(np.linalg.cholesky(M_dot_W_dot_M_plus_Q))
            cov = L.T @ L
        except np.linalg.LinAlgError:
            cov = np.linalg.inv(M_dot_W_dot_M_plus_Q)
        if self.w.W.ndim == 1:
            A = cov @ (self.w.M.T @ (self.w.W * self.w.data) + reg * self.w.Q_dot_A0)
        else:
            A = cov @ (self.w.M.T @ (self.w.W @ self.w.data) + reg * self.w.Q_dot_A0)
        if A.ndim == 2:  # ndim == 2 when A comes from a sparse matrix computation
            A = np.asarray(A).reshape(-1)
        self.resolution = np.eye(A.size) - reg * cov @ self.w.Q
        diff = self.w.data - self.w.M @ A
        if self.w.W.ndim == 1:
            self.chisquare = diff @ (self.w.W * diff)
        else:
            self.chisquare = diff @ self.w.W @ diff
        self.w.amplitude_params = A
        self.w.amplitude_cov_matrix = cov
        self.w.amplitude_params_err = np.array([np.sqrt(np.abs(cov[x, x])) for x in range(cov.shape[0])])
        self.G = self.chisquare / ((self.w.data.size - len(self.w.mask) - len(self.w.outliers)) - np.trace(self.resolution)) ** 2
        return np.asarray([log10_r]), np.asarray([self.G]), np.zeros_like(self.data)

    def plot_fit(self):
        log10_opt_reg = self.params.values[0]
        regs = 10 ** np.linspace(min(-7, 0.9 * log10_opt_reg), max(3, 1.2 * log10_opt_reg), 50)
        Gs = []
        chisqs = []
        resolutions = []
        for r in regs:
            self.simulate(np.log10(r))
            Gs.append(self.G)
            chisqs.append(self.chisquare)
            resolutions.append(np.trace(self.resolution))
        opt_reg = 10 ** log10_opt_reg
        fig, ax = plt.subplots(3, 1, figsize=(7, 5), sharex="all")
        ax[0].plot(regs, Gs)
        ax[0].axvline(opt_reg, color="k")
        ax[1].axvline(opt_reg, color="k")
        ax[2].axvline(opt_reg, color="k")
        ax[0].set_ylabel(r"$G(r)$")
        ax[0].set_xlabel("Regularisation hyper-parameter $r$")
        ax[0].grid()
        ax[0].set_title(f"Optimal regularisation parameter: {opt_reg:.3g}")
        ax[1].plot(regs, chisqs)
        ax[1].set_ylabel(r"$\chi^2(\mathbf{A}(r) \vert \mathbf{\theta})$")
        ax[1].set_xlabel("Regularisation hyper-parameter $r$")
        ax[1].grid()
        ax[1].set_xscale("log")
        ax[2].set_xscale("log")
        ax[2].plot(regs, resolutions)
        ax[2].set_ylabel(r"$\mathrm{Tr}\,\mathbf{R}$")
        ax[2].set_xlabel("Regularisation hyper-parameter $r$")
        ax[2].grid()
        # fig.tight_layout()
        plt.subplots_adjust(hspace=0)
        if parameters.DISPLAY:
            plt.show()
        if parameters.LSST_SAVEFIGPATH:
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'regularisation.pdf'))

        fig = plt.figure(figsize=(7, 5))
        rho = compute_correlation_matrix(self.w.amplitude_cov_matrix)
        plot_correlation_matrix_simple(plt.gca(), rho, axis_names=[''] * len(self.w.amplitude_params))
        # ipar=np.arange(10, 20))
        plt.gca().set_title(r"Correlation matrix $\mathbf{\rho}$")
        fig.tight_layout()
        if parameters.LSST_SAVEFIGPATH:
            fig.savefig(os.path.join(parameters.LSST_SAVEFIGPATH, 'amplitude_correlation_matrix.pdf'))
        if parameters.DISPLAY:
            plt.show()

    def run_regularisation(self, Ndof=None):
        run_minimisation(self, method="minimize", ftol=1e-4, xtol=1e-2, verbose=self.verbose, epsilon=[1e-1],
                         minimizer_method="Nelder-Mead")
        self.opt_reg = 10 ** self.params.values[0]
        self.simulate(np.log10(self.opt_reg))
        self.print_regularisation_summary()
        if Ndof is not None:
            r_Ndof = self.set_regularisation_with_Ndof(Ndof)
            if self.opt_reg < 1e-3 * r_Ndof or self.opt_reg > 1e3 * r_Ndof:
                self.my_logger.warning(f"\n\tRegularisation parameter r minimizing G(r) is 3 orders of magnitude away "
                                       f"from optimal regularisation parameter {r_Ndof} using {Ndof=}. "
                                       f"Probably that the model does not fit well data at this stage. "
                                       f"Switch to optimal parameter.")
                self.opt_reg = r_Ndof
                self.simulate(np.log10(self.opt_reg))
                self.print_regularisation_summary()

    def set_regularisation_with_Ndof(self, Ndof):
        """Find regularisation parameter $r$ that checks $Tr(R) = Ndof$.

        Parameters
        ----------
        Ndof: float
            Number of degrees of freedom, ie $Tr(R)$.

        Returns
        -------
        r: float
            Regularisation parameter.

        """
        log10_opt_reg = self.params.values[0]
        regs = 10 ** np.linspace(min(-7, 0.9 * log10_opt_reg), max(3, 1.2 * log10_opt_reg), 50)
        Gs = np.zeros_like(regs)
        chisqs = np.zeros_like(regs)
        resolutions = np.zeros_like(regs)
        for ir, r in enumerate(regs):
            self.simulate(np.log10(r))
            Gs[ir] = self.G
            chisqs[ir] = self.chisquare
            resolutions[ir] = np.trace(self.resolution)
        Ndof_index = np.argmin(np.abs(resolutions - Ndof))
        return regs[Ndof_index]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

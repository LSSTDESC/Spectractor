from iminuit import Minuit
from scipy import optimize
from schwimmbad import MPIPool
import emcee
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import sys
import os
import json
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Union

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.tools import (formatting_numbers, compute_correlation_matrix, plot_correlation_matrix_simple,
                               NumpyArrayEncoder)
from spectractor.fit.statistics import Likelihood


@dataclass()
class FitParameters:
    """Container for the parameters to fit on data with FitWorkspace.

    Attributes
    ----------
    p: np.ndarray
        Array containing the parameter values.
    input_labels: list, optional
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
    >>> params = FitParameters(p=[1, 1, 1, 1, 1])
    >>> params.ndim
    5
    >>> params.p
    array([1., 1., 1., 1., 1.])
    >>> params.input_labels
    ['par0', 'par1', 'par2', 'par3', 'par4']
    >>> params.bounds
    [[-inf, inf], [-inf, inf], [-inf, inf], [-inf, inf], [-inf, inf]]
    """
    p: Union[np.ndarray, list]
    input_labels: Optional[list] = None
    axis_names: Optional[list] = None
    bounds: Optional[list] = None
    fixed: Optional[list] = None
    truth: Optional[list] = None
    filename: Optional[str] = ""
    extra: Optional[dict] = None

    def __post_init__(self):
        if type(self.p) is list:
            self.p = np.array(self.p, dtype=float)
        self.p = np.asarray(self.p, dtype=float)
        if not self.input_labels:
            self.input_labels = [f"par{k}" for k in range(self.ndim)]
        else:
            if len(self.input_labels) != self.ndim:
                raise ValueError("input_labels argument must have same size as values argument.")
        if not self.axis_names:
            self.axis_names = [f"$p_{k}$" for k in range(self.ndim)]
        else:
            if len(self.axis_names) != self.ndim:
                raise ValueError("input_labels argument must have same size as values argument.")
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
        >>> params = FitParameters(p=[1, 1, 1], axis_names=["x", "y", "z"])
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
        >>> params = FitParameters(p=[1, 1, 1, 1, 1], fixed=[True, False, True, False, True])
        >>> params.cov = np.array([[1,-0.5,0],[-0.5,4,-1],[0,-1,9]])
        >>> params.err
        array([1., 0., 2., 0., 3.])
        """
        err = np.zeros_like(self.p, dtype=float)
        err[self.fixed] = np.sqrt(np.diag(self.cov))
        return err

    def __eq__(self, other):
        if not isinstance(other, FitParameters):
            return NotImplemented
        out = True
        for key in self.__dict__.keys():
            if isinstance(getattr(self, key), np.ndarray):
                out *= np.all(np.equal(getattr(self, key).flatten(), getattr(other, key).flatten()))
            else:
                out *= getattr(self, key) == getattr(other, key)
        return out

    @property
    def ndim(self):
        """Number of parameters.

        Returns
        -------
        ndim: int

        Examples
        --------
        >>> from spectractor.fit.fitter import FitParameters
        >>> params = FitParameters(p=[1, 1, 1, 1, 1])
        >>> params.ndim
        5
        """
        return len(self.p)

    @property
    def nfree(self):
        """Number of free parameters.

        Returns
        -------
        nfree: int

        Examples
        --------
        >>> from spectractor.fit.fitter import FitParameters
        >>> params = FitParameters(p=[1, 1, 1, 1, 1], fixed=[True, False, True, False, True])
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
        >>> params = FitParameters(p=[1, 1, 1, 1, 1], fixed=[True, False, True, False, True])
        >>> params.nfixed
        3
        """
        return len(self.get_fixed_parameters())

    def get_free_parameters(self):
        """Return indices array of free parameters.

        Examples
        --------
        >>> params = FitParameters(p=[1, 1, 1, 1, 1], fixed=None)
        >>> params.fixed
        [False, False, False, False, False]
        >>> params.get_free_parameters()
        array([0, 1, 2, 3, 4])
        >>> params = FitParameters(p=[1, 1, 1, 1, 1], fixed=[True, False, True, False, True])
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
        >>> params = FitParameters(p=[1, 1, 1, 1, 1], fixed=None)
        >>> params.fixed
        [False, False, False, False, False]
        >>> params.get_fixed_parameters()
        array([], dtype=int64)
        >>> params = FitParameters(p=[1, 1, 1, 1, 1], fixed=[True, False, True, False, True])
        >>> params.fixed
        [True, False, True, False, True]
        >>> params.get_fixed_parameters()
        array([0, 2, 4])

        """
        return np.array(np.where(np.array(self.fixed).astype(int) == 1)[0])

    def print_parameters_summary(self):
        """Print the best fitting parameters on screen.
        Labels are from self.input_labels.

        Returns
        -------
        txt: str
            The printed text.

        Examples
        --------
        >>> parameters.VERBOSE = True
        >>> params = FitParameters(p=[1, 2, 3, 4], input_labels=["x", "y", "z", "t"], fixed=[True, False, True, False])
        >>> params.cov = np.array([[1, -0.5], [-0.5, 4]])
        >>> _ = params.print_parameters_summary()
        """
        txt = ""
        ifree = self.get_free_parameters()
        icov = 0
        for ip in range(self.ndim):
            if ip in ifree:
                txt += "%s: %s +%s -%s\n\t" % formatting_numbers(self.p[ip], np.sqrt(self.cov[icov, icov]),
                                                                 np.sqrt(self.cov[icov, icov]),
                                                                 label=self.input_labels[ip])
                icov += 1
            else:
                txt += f"{self.input_labels[ip]}: {self.p[ip]} (fixed)\n\t"
        return txt

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
        >>> params = FitParameters(p=[1, 1, 1], axis_names=["x", "y", "z"])
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
        >>> params = FitParameters(p=[1, 2, 3, 4], input_labels=["x", "y", "z", "t"],  fixed=[True, False, True, False], filename="test_spectrum.fits")
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
    >>> params = FitParameters(p=[1, 2, 3, 4], input_labels=["x", "y", "z", "t"],  fixed=[True, False, True, False], filename="test_spectrum.fits")
    >>> params.cov = np.array([[1,-0.5,0],[-0.5,1,-1],[0,-1,1]])
    >>> jsonstr = write_fitparameter_json(params.json_filename, params, extra={"chi2": 1})
    >>> jsonstr  # doctest: +ELLIPSIS
    '{"p": [1, 2, 3, 4], "input_labels": ["x", "y", "z", "t"],..."extra": {"chi2": 1}...

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
    >>> params = FitParameters(p=[1, 2, 3, 4], input_labels=["x", "y", "z", "t"],  fixed=[True, False, True, False], filename="test_spectrum.fits")
    >>> params.cov = np.array([[1,-0.5,0],[-0.5,1,-1],[0,-1,1]])
    >>> _ = write_fitparameter_json(params.json_filename, params, extra={"chi2": 1})
    >>> new_params = read_fitparameter_json(params.json_filename)
    >>> new_params.p
    array([1, 2, 3, 4])

    .. doctest::
        :hide:

        >>> assert os.path.isfile(params.json_filename)
        >>> assert params == new_params
        >>> os.remove(params.json_filename)

    """
    params = FitParameters(p=[0])
    with open(json_filename, 'r') as f:
        data = json.load(f)
    for key in ["p", "cov"]:
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
        >>> params = FitParameters(p=[1, 1, 1, 1, 1])
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

    def plot_fit(self):
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
        for i, label in enumerate(self.params.input_labels):
            if self.params.cov.size > 0:
                err = np.sqrt(self.params.cov[i, i])
                formatting_numbers(self.params.p[i], err, err)
                _, par, err, _ = formatting_numbers(self.params.p[i], err, err, label=label)
                title += rf"{label} = {par} $\pm$ {err}"
            else:
                title += f"{label} = {self.params.p[i]:.3g}"
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
                    if self.W.ndim == 1 and self.W.dtype != object:
                        L = np.sqrt(self.W)
                    elif self.W.ndim == 2 and self.W.dtype != object:
                        L = np.linalg.cholesky(self.W)
                    else:
                        raise ValueError(f"Case not implemented with self.W.ndim={self.W.ndim} "
                                         f"and self.W.dtype={self.W.dtype}")
                res = L @ (model - self.data)
        return res

    def compute_W_with_model_error(self, model_err):
        W = self.W
        zeros = W == 0
        if self.W.ndim == 1 and self.W.dtype != object:
            if np.any(model_err > 0):
                W = 1 / (self.data_cov + model_err * model_err)
        elif self.W.dtype == object:
            K = len(self.W)
            if self.W[0].ndim == 1:
                if np.any(model_err > 0):
                    W = [1 / (self.data_cov[k] + model_err[k] * model_err[k]) for k in range(K)]
            elif self.W[0].ndim == 2:
                K = len(self.W)
                if np.any(model_err > 0):
                    cov = [self.data_cov[k] + np.diag(model_err[k] ** 2) for k in range(K)]
                    L = [np.linalg.inv(np.linalg.cholesky(cov[k])) for k in range(K)]
                    W = [L[k].T @ L[k] for k in range(K)]
            else:
                raise ValueError(f"First element of fitworkspace.W has no ndim attribute or has a dimension above 2. "
                                 f"I get W[0]={self.W[0]}")
        elif self.W.ndim == 2 and self.W.dtype != object:
            if np.any(model_err > 0):
                cov = self.data_cov + np.diag(model_err * model_err)
                L = np.linalg.inv(np.linalg.cholesky(cov))
                W = L.T @ L
        W[zeros] = 0
        return W

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
                f"Data inverse covariance matrix must be a np.ndarray of dimension 1 or 2,"
                f"either made of 1D or 2D arrays of equal lengths or not for block diagonal matrices."
                f"\nHere W type is {type(W)}, shape is {W.shape} and W is {W}.")
        if model_output:
            return chisq, x, model, model_err
        else:
            return chisq

    def prepare_weight_matrices(self):
        # Prepare covariance matrix for data
        if self.data_cov is None:
            self.data_cov = np.asarray(self.err.flatten() ** 2)
        # Prepare inverse covariance matrix for data
        if self.W is None:
            if self.data_cov.ndim == 1 and self.data_cov.dtype != object:
                self.W = 1 / self.data_cov
            elif self.data_cov.ndim == 2 and self.data_cov.dtype != object:
                L = np.linalg.inv(np.linalg.cholesky(self.data_cov))
                self.W = L.T @ L
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

    def lnprior(self, p):
        """Compute the logarithmic prior for a set of model parameters p.

        The function returns 0 for good parameters, and -np.inf for parameters out of their boundaries.

        Parameters
        ----------
        p: array_like
            The array of model parameters.

        Returns
        -------
        lnprior: float
            The logarithmic value fo the prior.

        """
        in_bounds = True
        for npar, par in enumerate(p):
            if par < self.params.bounds[npar][0] or par > self.params.bounds[npar][1]:
                in_bounds = False
                break
        if in_bounds:
            return 0.0
        else:
            return -np.inf

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

        self.simulate(*self.params.p)
        self.live_fit = False
        self.plot_fit()

    def save_gradient_descent(self):
        iterations = np.arange(self.params_table.shape[0]).astype(int)
        t = np.zeros((self.params_table.shape[1] + 2, self.params_table.shape[0]))
        t[0] = iterations
        t[1] = self.costs
        t[2:] = self.params_table.T
        h = 'iter,costs,' + ','.join(self.params.input_labels)
        output_filename = os.path.splitext(self.filename)[0] + "_fitting.txt"
        np.savetxt(output_filename, t.T, header=h, delimiter=",")
        self.my_logger.info(f"\n\tSave gradient descent log {output_filename}.")


class MCMCFitWorkspace(FitWorkspace):

    def __init__(self, params, file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=False, plot=False, live_fit=False, truth=None):
        """Generic class to create a fit workspace with parameters, bounds and general fitting methods.

        Parameters
        ----------
        params: FitParameters
            The parameters to fit to data.
        file_name: str, optional
            The generic file name to save results. If file_name=="", nothing is saved ond disk (default: "").
        nwalkers: int, optional
            Number of walkers for MCMC exploration (default: 18).
        nsteps: int, optional
            Number of steps for MCMC exploration (default: 1000).
        burnin: int, optional
            Number of burn-in steps for MCMC exploration (default: 100).
        nbins: int, optional
            Number of bins to make histograms after MCMC exploration (default: 10).
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
        >>> params = FitParameters(p=[1, 1, 1, 1, 1])
        >>> w = MCMCFitWorkspace(params)
        >>> w.nwalkers
        18
        """
        FitWorkspace.__init__(self, params, file_name=file_name, verbose=verbose, plot=plot, live_fit=live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        self.nwalkers = max(2 * self.params.ndim, nwalkers)
        self.nsteps = nsteps
        self.nbins = nbins
        self.burnin = burnin
        self.start = []
        self.likelihood = np.array([[]])
        self.gelmans = np.array([])
        self.chains = np.array([[]])
        self.lnprobs = np.array([[]])
        self.flat_chains = np.array([[]])
        self.valid_chains = [False] * self.nwalkers
        self.global_average = None
        self.global_std = None
        self.use_grid = False
        if self.filename != "":
            if "." in self.filename:
                self.emcee_filename = os.path.splitext(self.filename)[0] + "_emcee.h5"
            else:
                self.my_logger.warning("\n\tFile name must have an extension.")
        else:
            self.emcee_filename = "emcee.h5"

    def set_start(self, percent=0.02, a_random=1e-5):
        """Set the random starting points for MCMC exploration.

        A set of parameters are drawn with a uniform distribution between +/- percent times the starting guess.
        For null guess parameters, starting points are drawn from a uniform distribution between +/- a_random.

        Parameters
        ----------
        percent: float, optional
            Percent of the guess parameters to set the uniform interval to draw random points (default: 0.02).
        a_random: float, optional
            Absolute value to set the +/- uniform interval to draw random points
            for null guess parameters (default: 1e-5).

        Returns
        -------
        start: np.array
            Array of starting points of shape (ndim, nwalkers).

        """
        self.start = np.array([np.random.uniform(self.params.p[i] - percent * self.params.p[i],
                                                 self.params.p[i] + percent * self.params.p[i],
                                                 self.nwalkers) for i in range(self.params.ndim)]).T
        self.start[self.start == 0] = a_random * np.random.uniform(-1, 1)
        return self.start

    def load_chains(self):
        """Load the MCMC chains from a hdf5 file. The burn-in points are not rejected at this stage.

        Returns
        -------
        chains: np.array
            Array of the chains.
        lnprobs: np.array
            Array of the logarithmic posterior probability.

        """
        self.chains = [[]]
        self.lnprobs = [[]]
        self.nsteps = 0
        # tau = -1
        reader = emcee.backends.HDFBackend(self.emcee_filename)
        try:
            tau = reader.get_autocorr_time()
        except emcee.autocorr.AutocorrError:
            tau = -1
        self.chains = reader.get_chain(discard=0, flat=False, thin=1)
        self.lnprobs = reader.get_log_prob(discard=0, flat=False, thin=1)
        self.nsteps = self.chains.shape[0]
        self.nwalkers = self.chains.shape[1]
        print(f"Auto-correlation time: {tau}")
        print(f"Burn-in: {self.burnin}")
        print(f"Chains shape: {self.chains.shape}")
        print(f"Log prob shape: {self.lnprobs.shape}")
        return self.chains, self.lnprobs

    def build_flat_chains(self):
        """Flatten the chains array and apply burn-in.

        Returns
        -------
        flat_chains: np.array
            Flat chains.

        """
        self.flat_chains = self.chains[self.burnin:, self.valid_chains, :].reshape((-1, self.params.ndim))
        return self.flat_chains

    def analyze_chains(self):
        """Load the chains, build the probability densities for the parameters, compute the best fitting values
        and the uncertainties and covariance matrices, and plot.

        """
        self.load_chains()
        self.set_chain_validity()
        self.convergence_tests()
        self.build_flat_chains()
        self.likelihood = self.chain2likelihood()
        self.params.cov = self.likelihood.cov_matrix
        self.params.p = self.likelihood.mean_vec
        self.simulate(*self.params.p)
        self.plot_fit()
        figure_name = os.path.splitext(self.emcee_filename)[0] + '_triangle.pdf'
        self.likelihood.triangle_plots(output_filename=figure_name)

    def chain2likelihood(self, pdfonly=False, walker_index=-1):
        """Convert the chains to a psoterior probability density function via histograms.

        Parameters
        ----------
        pdfonly: bool, optional
            If True, do not compute the covariances and the 2D correlation plots (default: False).
        walker_index: int, optional
            The walker index to plot. If -1, all walkers are selected (default: -1).

        Returns
        -------
        likelihood: np.array
            Posterior density function.

        """
        if walker_index >= 0:
            chains = self.chains[self.burnin:, walker_index, :]
        else:
            chains = self.flat_chains
        rangedim = range(chains.shape[1])
        centers = []
        for i in rangedim:
            centers.append(np.linspace(np.min(chains[:, i]), np.max(chains[:, i]), self.nbins - 1))
        likelihood = Likelihood(centers, labels=self.params.input_labels, axis_names=self.params.axis_names, truth=self.params.truth)
        if walker_index < 0:
            for i in rangedim:
                likelihood.pdfs[i].fill_histogram(chains[:, i], weights=None)
                if not pdfonly:
                    for j in rangedim:
                        if i != j:
                            likelihood.contours[i][j].fill_histogram(chains[:, i], chains[:, j], weights=None)
            output_file = ""
            if self.filename != "":
                output_file = os.path.splitext(self.filename)[0] + "_bestfit.txt"
            likelihood.stats(output=output_file)
        else:
            for i in rangedim:
                likelihood.pdfs[i].fill_histogram(chains[:, i], weights=None)
        return likelihood

    def compute_local_acceptance_rate(self, start_index, last_index, walker_index):
        """Compute the local acceptance rate in a chain.

        Parameters
        ----------
        start_index: int
            Beginning index.
        last_index: int
            End index.
        walker_index: int
            Index of the walker.

        Returns
        -------
        freq: float
            The acceptance rate.

        """
        frequences = []
        test = -2 * self.lnprobs[start_index, walker_index]
        counts = 1
        for index in range(start_index + 1, last_index):
            chi2 = -2 * self.lnprobs[index, walker_index]
            if np.isclose(chi2, test):
                counts += 1
            else:
                frequences.append(float(counts))
                counts = 1
                test = chi2
        frequences.append(counts)
        return 1.0 / np.mean(frequences)

    def set_chain_validity(self):
        """Test the validity of a chain: reject chains whose chi2 is far from the mean of the others.

        Returns
        -------
        valid_chains: list
            List of boolean values, True if the chain is valid, or False if invalid.

        """
        nchains = [k for k in range(self.nwalkers)]
        chisq_averages = []
        chisq_std = []
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            # if np.mean(chisqs) < 1e5:
            chisq_averages.append(np.mean(chisqs))
            chisq_std.append(np.std(chisqs))
        self.global_average = np.mean(chisq_averages)
        self.global_std = np.mean(chisq_std)
        self.valid_chains = [False] * self.nwalkers
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            chisq_average = np.mean(chisqs)
            chisq_std = np.std(chisqs)
            if 3 * self.global_std + self.global_average < chisq_average < 1e5:
                self.valid_chains[k] = False
            elif chisq_std < 0.1 * self.global_std:
                self.valid_chains[k] = False
            else:
                self.valid_chains[k] = True
        return self.valid_chains

    def convergence_tests(self):
        """Compute the convergence tests (Gelman-Rubin, acceptance rate).

        """
        chains = self.chains[self.burnin:, :, :]  # .reshape((-1, self.ndim))
        nchains = [k for k in range(self.nwalkers)]
        fig, ax = plt.subplots(self.params.ndim + 1, 2, figsize=(16, 7), sharex='all')
        fontsize = 8
        steps = np.arange(self.burnin, self.nsteps)
        # Chi2 vs Index
        print("Chisq statistics:")
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            text = f"\tWalker {k:d}: {float(np.mean(chisqs)):.3f} +/- {float(np.std(chisqs)):.3f}"
            if not self.valid_chains[k]:
                text += " -> excluded"
                ax[self.params.ndim, 0].plot(steps, chisqs, c='0.5', linestyle='--')
            else:
                ax[self.params.ndim, 0].plot(steps, chisqs)
            print(text)
        # global_average = np.mean(-2*self.lnprobs[self.valid_chains, self.burnin:])
        # global_std = np.std(-2*self.lnprobs[self.valid_chains, self.burnin:])
        ax[self.params.ndim, 0].set_ylim(
            [self.global_average - 5 * self.global_std, self.global_average + 5 * self.global_std])
        # Parameter vs Index
        print("Computing Parameter vs Index plots...")
        for i in range(self.params.ndim):
            ax[i, 0].set_ylabel(self.params.axis_names[i], fontsize=fontsize)
            for k in nchains:
                if self.valid_chains[k]:
                    ax[i, 0].plot(steps, chains[:, k, i])
                else:
                    ax[i, 0].plot(steps, chains[:, k, i], c='0.5', linestyle='--')
                ax[i, 0].get_yaxis().set_label_coords(-0.05, 0.5)
        ax[self.params.ndim, 0].set_ylabel(r'$\chi^2$', fontsize=fontsize)
        ax[self.params.ndim, 0].set_xlabel('Steps', fontsize=fontsize)
        ax[self.params.ndim, 0].get_yaxis().set_label_coords(-0.05, 0.5)
        # Acceptance rate vs Index
        print("Computing acceptance rate...")
        min_len = self.nsteps
        window = 100
        if min_len > window:
            for k in nchains:
                ARs = []
                indices = []
                for pos in range(self.burnin + window, self.nsteps, window):
                    ARs.append(self.compute_local_acceptance_rate(pos - window, pos, k))
                    indices.append(pos)
                if self.valid_chains[k]:
                    ax[self.params.ndim, 1].plot(indices, ARs, label=f'Walker {k:d}')
                else:
                    ax[self.params.ndim, 1].plot(indices, ARs, label=f'Walker {k:d}', c='gray', linestyle='--')
                ax[self.params.ndim, 1].set_xlabel('Steps', fontsize=fontsize)
                ax[self.params.ndim, 1].set_ylabel('Aceptance rate', fontsize=fontsize)
                # ax[self.dim + 1, 2].legend(loc='upper left', ncol=2, fontsize=10)
        # Parameter PDFs by chain
        print("Computing chain by chain PDFs...")
        for k in nchains:
            likelihood = self.chain2likelihood(pdfonly=True, walker_index=k)
            likelihood.stats(pdfonly=True, verbose=False)
            # for i in range(self.dim):
            # ax[i, 1].plot(likelihood.pdfs[i].axe.axis, likelihood.pdfs[i].grid, lw=var.LINEWIDTH,
            #               label=f'Walker {k:d}')
            # ax[i, 1].set_xlabel(self.axis_names[i])
            # ax[i, 1].set_ylabel('PDF')
            # ax[i, 1].legend(loc='upper right', ncol=2, fontsize=10)
        # Gelman-Rubin test.py
        if len(nchains) > 1:
            step = max(1, (self.nsteps - self.burnin) // 20)
            self.gelmans = []
            print(f'Gelman-Rubin tests (burnin={self.burnin:d}, step={step:d}, nsteps={self.nsteps:d}):')
            for i in range(self.params.ndim):
                Rs = []
                lens = []
                for pos in range(self.burnin + step, self.nsteps, step):
                    chain_averages = []
                    chain_variances = []
                    global_average = np.mean(self.chains[self.burnin:pos, self.valid_chains, i])
                    for k in nchains:
                        if not self.valid_chains[k]:
                            continue
                        chain_averages.append(np.mean(self.chains[self.burnin:pos, k, i]))
                        chain_variances.append(np.var(self.chains[self.burnin:pos, k, i], ddof=1))
                    W = np.mean(chain_variances)
                    B = 0
                    for n in range(len(chain_averages)):
                        B += (chain_averages[n] - global_average) ** 2
                    B *= ((pos + 1) / (len(chain_averages) - 1))
                    R = (W * pos / (pos + 1) + B / (pos + 1) * (len(chain_averages) + 1) / len(chain_averages)) / W
                    Rs.append(R - 1)
                    lens.append(pos)
                print(f'\t{self.params.input_labels[i]}: R-1 = {Rs[-1]:.3f} (l = {lens[-1] - 1:d})')
                self.gelmans.append(Rs[-1])
                ax[i, 1].plot(lens, Rs, lw=1, label=self.params.axis_names[i])
                ax[i, 1].axhline(0.03, c='k', linestyle='--')
                ax[i, 1].set_xlabel('Walker length', fontsize=fontsize)
                ax[i, 1].set_ylabel('$R-1$', fontsize=fontsize)
                ax[i, 1].set_ylim(0, 0.6)
                # ax[self.dim, 3].legend(loc='best', ncol=2, fontsize=10)
        self.gelmans = np.array(self.gelmans)
        fig.tight_layout()
        plt.subplots_adjust(hspace=0)
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()
        figure_name = self.emcee_filename.replace('.h5', '_convergence.pdf')
        print(f'Save figure: {figure_name}')
        fig.savefig(figure_name, dpi=100)

    def print_settings(self):
        """Print the main settings of the FitWorkspace.

        """
        print('************************************')
        print(f"Input file: {self.filename}\nWalkers: {self.nwalkers}\t Steps: {self.nsteps}")
        print(f"Output file: {self.emcee_filename}")
        print('************************************')


def lnprob(p):  # pragma: no cover
    global fit_workspace
    lp = fit_workspace.lnprior(p)
    if not np.isfinite(lp):
        return -1e20
    return lp + fit_workspace.lnlike(p)


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
    tmp_params = np.copy(fit_workspace.params.p)
    fit_workspace.prepare_weight_matrices()
    n_data_masked = len(fit_workspace.mask) + len(fit_workspace.outliers)
    ipar = fit_workspace.params.get_free_parameters()
    costs = []
    params_table = []
    inv_JT_W_J = np.zeros((len(ipar), len(ipar)))
    for i in range(niter):
        start = time.time()
        cost, tmp_lambdas, tmp_model, tmp_model_err = fit_workspace.chisq(tmp_params, model_output=True)
        # W matrix
        W = fit_workspace.compute_W_with_model_error(tmp_model_err)
        # residuals
        if isinstance(W, np.ndarray) and W.dtype != object:
            residuals = (tmp_model - fit_workspace.data).flatten()
        elif isinstance(W, np.ndarray) and W.dtype == object:
            residuals = [(tmp_model[k] - fit_workspace.data[k]) for k in range(len(W))]
        else:
            raise TypeError(f"Type of fit_workspace.W is {type(W)}. It must be a np.ndarray.")
        # Jacobian
        J = fit_workspace.jacobian(tmp_params, epsilon, model_input=[tmp_lambdas, tmp_model, tmp_model_err])
        # remove parameters with unexpected null Jacobian vectors
        for ip in range(J.shape[0]):
            if ip not in ipar:
                continue
            if np.all(np.array(J[ip]).flatten() == np.zeros(np.array(J[ip]).size)):
                ipar = np.delete(ipar, list(ipar).index(ip))
                fit_workspace.params.fixed[ip] = True
                my_logger.warning(
                    f"\n\tStep {i}: {fit_workspace.params.input_labels[ip]} has a null Jacobian; parameter is fixed "
                    f"at its last known current value ({tmp_params[ip]}).")
        # remove fixed parameters
        J = J[ipar].T
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
        dparams = - inv_JT_W_J @ JT_W_R0

        if with_line_search:
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
        else:
            alpha_min = 1
            fval = np.copy(cost)
            funcalls = 0
            iter = 0

        tmp_params[ipar] += alpha_min * dparams
        # check bounds
        for ip, p in enumerate(tmp_params):
            if p < fit_workspace.params.bounds[ip][0]:
                tmp_params[ip] = fit_workspace.params.bounds[ip][0]
            if p > fit_workspace.params.bounds[ip][1]:
                tmp_params[ip] = fit_workspace.params.bounds[ip][1]

        # prepare outputs
        costs.append(fval)
        params_table.append(np.copy(tmp_params))
        fit_workspace.p = tmp_params
        if fit_workspace.verbose:
            my_logger.info(f"\n\tIteration={i}: initial cost={cost:.5g} initial chisq_red={cost / (tmp_model.size - n_data_masked):.5g}"
                           f"\n\t\t Line search: alpha_min={alpha_min:.3g} iter={iter} funcalls={funcalls}"
                           f"\n\tParameter shifts: {alpha_min * dparams}"
                           f"\n\tNew parameters: {tmp_params[ipar]}"
                           f"\n\tFinal cost={fval:.5g} final chisq_red={fval / (tmp_model.size - n_data_masked):.5g} "
                           f"computed in {time.time() - start:.2f}s")
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
    plt.close()
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
    tmp_params = np.copy(fit_workspace.params.p)
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
                    f"\n\tStep {i}: {fit_workspace.params.input_labels[ip]} has a null Jacobian; parameter is fixed "
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
        fit_workspace.costs = np.array([fit_workspace.chisq(fit_workspace.params.p)])
        fit_workspace.params_table = np.array([fit_workspace.params.p])
    p, cov, tmp_costs, tmp_params_table = gradient_descent(fit_workspace, epsilon, niter=niter, xtol=xtol, ftol=ftol,
                                                           with_line_search=with_line_search)
    fit_workspace.params.p, fit_workspace.params.cov = p, cov
    fit_workspace.params_table = np.concatenate([fit_workspace.params_table, tmp_params_table])
    fit_workspace.costs = np.concatenate([fit_workspace.costs, tmp_costs])
    if verbose or fit_workspace.verbose:
        fit_workspace.my_logger.info(f"\n\t{fit_workspace.params.print_parameters_summary()}")
    if parameters.DEBUG and (verbose or fit_workspace.verbose):
        fit_workspace.plot_gradient_descent()
        if len(fit_workspace.params.get_free_parameters()) > 1:
            fit_workspace.params.plot_correlation_matrix()


def run_simple_newton_minimisation(fit_workspace, epsilon, xtol=1e-8, ftol=1e-8, niter=50, verbose=False):  # pragma: no cover
    fit_workspace.p, fit_workspace.cov, funcs, params_table = simple_newton_minimisation(fit_workspace,
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

    guess = fit_workspace.params.p.astype('float64')
    if verbose:
        my_logger.debug(f"\n\tStart guess: {guess}")

    if method == "minimize":
        start = time.time()
        result = optimize.minimize(nll, fit_workspace.params.p, method=minimizer_method,
                                   options={'ftol': ftol, 'maxiter': 100000}, bounds=bounds)
        fit_workspace.params.p = result['x']
        if verbose:
            my_logger.debug(f"\n\t{result}")
            my_logger.debug(f"\n\tMinimize: total computation time: {time.time() - start}s")
            if parameters.DEBUG:
                fit_workspace.plot_fit()
    elif method == 'basinhopping':
        start = time.time()
        minimizer_kwargs = dict(method=minimizer_method, bounds=bounds)
        result = optimize.basinhopping(nll, guess, minimizer_kwargs=minimizer_kwargs)
        fit_workspace.params.p = result['x']
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
        fit_workspace.params.p = p.x  # m.np_values()
        if verbose:
            my_logger.debug(f"\n\t{p}")
            my_logger.debug(f"\n\tLeast_squares: total computation time: {time.time() - start}s")
            if parameters.DEBUG:
                fit_workspace.plot_fit()
    elif method == "minuit":
        start = time.time()
        error = 0.1 * np.abs(guess) * np.ones_like(guess)
        error[2:5] = 0.3 * np.abs(guess[2:5]) * np.ones_like(guess[2:5])
        z = np.where(np.isclose(error, 0.0, 1e-6))
        error[z] = 1.
        # noinspection PyArgumentList
        # m = Minuit(fcn=nll, values=guess, error=error, errordef=1, fix=fix, print_level=verbose, limit=bounds)
        m = Minuit(nll, np.copy(guess))
        m.errors = error
        m.errordef = 1
        m.fixed = fit_workspace.params.fixed
        m.print_level = verbose
        m.limits = bounds
        m.tol = 10
        m.migrad()
        fit_workspace.p = np.array(m.values[:])
        if verbose:
            my_logger.debug(f"\n\t{m}")
            my_logger.debug(f"\n\tMinuit: total computation time: {time.time() - start}s")
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
                                    niter=50, sigma_clip=5.0, niter_clip=3, verbose=False):
    my_logger = set_logger(__name__)
    fit_workspace.sigma_clip = sigma_clip
    for step in range(niter_clip):
        if verbose:
            my_logger.info(f"\n\tSigma-clipping step {step}/{niter_clip} (sigma={sigma_clip})")
        run_minimisation(fit_workspace, method=method, epsilon=epsilon, xtol=xtol, ftol=ftol, niter=niter)
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


def run_emcee(mcmc_fit_workspace, ln=lnprob):
    my_logger = set_logger(__name__)
    mcmc_fit_workspace.print_settings()
    nsamples = mcmc_fit_workspace.nsteps
    p0 = mcmc_fit_workspace.set_start()
    filename = mcmc_fit_workspace.emcee_filename
    backend = emcee.backends.HDFBackend(filename)
    try:  # pragma: no cover
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(mcmc_fit_workspace.nwalkers, mcmc_fit_workspace.ndim, ln, args=(),
                                        pool=pool, backend=backend)
        my_logger.info(f"\n\tInitial size: {backend.iteration}")
        if backend.iteration > 0:
            p0 = backend.get_last_sample()
        if nsamples - backend.iteration > 0:
            sampler.run_mcmc(p0, nsteps=max(0, nsamples - backend.iteration), progress=True)
        pool.close()
    except ValueError:
        sampler = emcee.EnsembleSampler(mcmc_fit_workspace.nwalkers, mcmc_fit_workspace.params.ndim, ln, args=(),
                                        threads=multiprocessing.cpu_count(), backend=backend)
        my_logger.info(f"\n\tInitial size: {backend.iteration}")
        if backend.iteration > 0:
            p0 = sampler.get_last_sample()
        for _ in sampler.sample(p0, iterations=max(0, nsamples - backend.iteration), progress=True, store=True):
            continue
    mcmc_fit_workspace.chains = sampler.chain
    mcmc_fit_workspace.lnprobs = sampler.lnprobability


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
        params = FitParameters(np.asarray([np.log10(opt_reg)]), input_labels=["log10_reg"],
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
                            f"\n\tTr(R) = {np.trace(self.resolution)}"
                            f"\n\tN_params = {len(self.w.amplitude_params)}"
                            f"\n\tN_data = {self.w.data.size - len(self.w.mask) - len(self.w.outliers)}"
                            f" (without mask and outliers)")

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
        self.w.amplitude_params_err = np.array([np.sqrt(cov[x, x]) for x in range(cov.shape[0])])
        self.G = self.chisquare / ((self.w.data.size - len(self.w.mask) - len(self.w.outliers)) - np.trace(self.resolution)) ** 2
        return np.asarray([log10_r]), np.asarray([self.G]), np.zeros_like(self.data)

    def plot_fit(self):
        log10_opt_reg = self.params.p[0]
        opt_reg = 10 ** log10_opt_reg
        regs = 10 ** np.linspace(min(-10, 0.9 * log10_opt_reg), max(3, 1.2 * log10_opt_reg), 50)
        Gs = []
        chisqs = []
        resolutions = []
        x = np.arange(len(self.w.amplitude_priors))
        for r in regs:
            self.simulate(np.log10(r))
            if parameters.DISPLAY and False:  # pragma: no cover
                fig = plt.figure()
                plt.errorbar(x, self.w.amplitude_params, yerr=[np.sqrt(self.w.amplitude_cov_matrix[i, i]) for i in x],
                             label=f"fit r={r:.2g}")
                plt.plot(x, self.w.amplitude_priors, label="prior")
                plt.grid()
                plt.legend()
                plt.draw()
                plt.pause(1e-8)
                plt.close(fig)
            Gs.append(self.G)
            chisqs.append(self.chisquare)
            resolutions.append(np.trace(self.resolution))
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
        fig.tight_layout()
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

    def run_regularisation(self):
        run_minimisation(self, method="minimize", ftol=1e-4, xtol=1e-2, verbose=self.verbose, epsilon=[1e-1],
                         minimizer_method="Nelder-Mead")
        self.opt_reg = 10 ** self.params.p[0]
        self.simulate(np.log10(self.opt_reg))
        self.print_regularisation_summary()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

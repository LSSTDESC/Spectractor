import sys
import numpy as np
import matplotlib.pyplot as plt
from deprecated import deprecated

from scipy.optimize import basinhopping, minimize
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
from iminuit import Minuit

from astropy.modeling import Fittable1DModel, Fittable2DModel, Parameter

from spectractor.tools import (dichotomie, plot_image_simple)
from spectractor import parameters
from spectractor.config import set_logger
from spectractor.fit.fitter import FitWorkspace, run_minimisation

from numba import njit


@njit
def evaluate_moffat1d(y, amplitude, y_mean, gamma, alpha):
    rr = (y - y_mean) * (y - y_mean)
    rr_gg = rr / (gamma * gamma)
    a = np.power(1 + rr_gg, -alpha)
    dx = y[1] - y[0]
    integral = np.sum(a) * dx
    norm = amplitude
    if integral != 0:
        norm /= integral
    a *= norm
    return a.T


@njit
def evaluate_moffatgauss1d(y, amplitude, y_mean, gamma, alpha, eta_gauss, stddev):
    rr = (y - y_mean) * (y - y_mean)
    rr_gg = rr / (gamma * gamma)
    a = np.power(1 + rr_gg, -alpha) + eta_gauss * np.exp(-(rr / (2. * stddev * stddev)))
    dx = y[1] - y[0]
    integral = np.sum(a) * dx
    norm = amplitude
    if integral != 0:
        norm /= integral
    a *= norm
    return a.T


@njit
def evaluate_moffat2d(x, y, amplitude, x_mean, y_mean, gamma, alpha):
    rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
    rr_gg = rr / (gamma * gamma)
    a = np.power(1 + rr_gg, -alpha)
    norm = (np.pi * gamma * gamma) / (alpha - 1)
    a *= amplitude / norm
    return a.T


@njit
def evaluate_moffatgauss2d(x, y, amplitude, x_mean, y_mean, gamma, alpha, eta_gauss, stddev):
    rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
    rr_gg = rr / (gamma * gamma)
    a = np.power(1 + rr_gg, -alpha) + eta_gauss * np.exp(-(rr / (2. * stddev * stddev)))
    norm = (np.pi * gamma * gamma) / (alpha - 1) + eta_gauss * 2 * np.pi * stddev * stddev
    a *= amplitude / norm
    return a.T


class PSF:
    """Generic PSF model class.

    The PSF models must contain at least the "amplitude", "x_mean" and "y_mean" parameters as the first three parameters
    (in this order) and "saturation" parameter as the last parameter. "amplitude", "x_mean" and "y_mean"
    stands respectively for the general amplitude of the model, the position along the dispersion axis and the
    transverse position with respect to the dispersion axis (assumed to be the X axis).
    Last "saturation" parameter must be express in the same units as the signal to model and as the "amplitude"
    parameter. The PSF models must be normalized to one in total flux divided by the first parameter (amplitude).
    Then the PSF model integral is equal to the "amplitude" parameter.

    """

    def __init__(self):
        self.my_logger = set_logger(self.__class__.__name__)
        self.p = np.array([])
        self.param_names = ["amplitude", "x_mean", "y_mean", "saturation"]
        self.axis_names = ["$A$", r"$x_0$", r"$y_0$", "saturation"]
        self.bounds = [[]]
        self.p_default = np.array([1, 0, 0, 1])
        self.max_half_width = np.inf

    def evaluate(self, pixels, p=None):  # pragma: no cover
        if p is not None:
            self.p = np.asarray(p).astype(float)
        # amplitude, x_mean, y_mean, saturation = self.p
        if pixels.ndim == 3 and pixels.shape[0] == 2:
            return np.zeros_like(pixels)
        elif pixels.ndim == 1:
            return np.zeros_like(pixels)
        else:
            self.my_logger.error(f"\n\tPixels array must have dimension 1 or shape=(2,Nx,Ny). "
                                 f"Here pixels.ndim={pixels.shape}.")
            return None

    def apply_max_width_to_bounds(self, max_half_width=None):  # pragma: no cover
        pass

    def fit_psf(self, data, data_errors=None, bgd_model_func=None):
        """
        Fit a PSF model on 1D or 2D data.

        Parameters
        ----------
        data: array_like
            1D or 2D array containing the data.
        data_errors: np.array, optional
            The 1D or 2D array of uncertainties.
        bgd_model_func: callable, optional
            A 1D or 2D function to model the extracted background (default: None -> null background).

        Returns
        -------
        fit_workspace: PSFFitWorkspace
            The PSFFitWorkspace instance to get info about the fitting.

        Examples
        --------

        Build a mock PSF2D without background and with random Poisson noise:

        >>> p0 = np.array([200000, 20, 30, 5, 2, -0.1, 2, 400000])
        >>> psf0 = MoffatGauss(p0)
        >>> xx, yy = np.mgrid[:50, :60]
        >>> data = psf0.evaluate(np.array([xx, yy]), p0)
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        Fit the data in 2D:

        >>> p = np.array([100000, 19, 31, 3, 3, -0.1, 3, 400000])
        >>> psf = MoffatGauss(p)
        >>> w = psf.fit_psf(data, data_errors=data_errors, bgd_model_func=None)
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert w.model is not None
            >>> residuals = (w.data-w.model)/w.err
            >>> assert w.costs[-1] / w.pixels.size < 1.3
            >>> assert np.abs(np.mean(residuals)) < 0.4
            >>> assert np.std(residuals) < 1.2
            >>> assert np.all(np.isclose(psf.p[1:3], p0[1:3], atol=1e-1))

        Fit the data in 1D:

        >>> data1d = data[:,int(p[1])]
        >>> data1d_err = data_errors[:,int(p[1])]
        >>> p = np.array([10000, 20, 32, 3, 3, -0.2, 1, 400000])
        >>> psf1d = MoffatGauss(p)
        >>> w = psf1d.fit_psf(data1d, data_errors=data1d_err, bgd_model_func=None)
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert w.model is not None
            >>> residuals = (w.data-w.model)/w.err
            >>> assert w.costs[-1] / w.pixels.size < 1.2
            >>> assert np.abs(np.mean(residuals)) < 0.2
            >>> assert np.std(residuals) < 1.2
            >>> assert np.all(np.isclose(w.p[2], p0[2], atol=1e-1))

        .. plot::
            :include-source:

            import numpy as np
            import matplotlib.pyplot as plt
            from spectractor.extractor.psf import *
            p = np.array([200000, 20, 30, 5, 2, -0.1, 2, 400000])
            psf = MoffatGauss(p)
            xx, yy = np.mgrid[:50, :60]
            data = psf.evaluate(np.array([xx, yy]), p)
            data = np.random.poisson(data)
            data_errors = np.sqrt(data+1)
            data = np.random.poisson(data)
            data_errors = np.sqrt(data+1)
            psf = MoffatGauss(p)
            w = psf.fit_psf(data, data_errors=data_errors, bgd_model_func=None)
            w.plot_fit()

        """
        w = PSFFitWorkspace(self, data, data_errors, bgd_model_func=bgd_model_func,
                            verbose=False, live_fit=False)
        run_minimisation(w, method="newton", ftol=1 / w.pixels.size, xtol=1e-6, niter=50, fix=w.fixed)
        self.p = np.copy(w.p)
        return w


class Moffat(PSF):

    def __init__(self, p=None):
        PSF.__init__(self)
        self.p_default = np.array([1, 0, 0, 3, 2, 1]).astype(float)
        if p is not None:
            self.p = np.asarray(p).astype(float)
        else:
            self.p = np.copy(self.p_default)
        self.param_names = ["amplitude", "x_mean", "y_mean", "gamma", "alpha", "saturation"]
        self.axis_names = ["$A$", r"$x_0$", r"$y_0$", r"$\gamma$", r"$\alpha$", "saturation"]
        self.bounds = np.array([(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0.1, np.inf), (1.1, 10),
                                (0, np.inf)])

    def apply_max_width_to_bounds(self, max_half_width=None):
        if max_half_width is not None:
            self.max_half_width = max_half_width
        self.bounds = np.array([(0, np.inf), (-np.inf, np.inf), (0, 2 * self.max_half_width),
                                (0.1, self.max_half_width), (1.1, 10), (0, np.inf)])

    def evaluate(self, pixels, p=None):
        """Evaluate the Moffat function.

        The function is normalized to have an integral equal to amplitude parameter.

        Parameters
        ----------
        pixels: list
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        p: array_like
            The parameter array. If None, the array used to instanciate the class is taken.
            If given, the class instance parameter array is updated.

        Returns
        -------
        output: array_like
            The PSF function evaluated.

        Examples
        --------
        >>> p = [2,20,30,4,2,10]
        >>> psf = Moffat(p)
        >>> xx, yy = np.mgrid[:50, :60]
        >>> out = psf.evaluate(pixels=np.array([xx, yy]))

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np
            from spectractor.extractor.psf import Moffat
            p = [2,20,30,4,2,10]
            psf = Moffat(p)
            xx, yy = np.mgrid[:50, :60]
            out = psf.evaluate(pixels=np.array([xx, yy]))
            fig = plt.figure(figsize=(5,5))
            plt.imshow(out, origin="lower")
            plt.xlabel("X [pixels]")
            plt.ylabel("Y [pixels]")
            plt.show()

        """
        if p is not None:
            self.p = np.asarray(p).astype(float)
        amplitude, x_mean, y_mean, gamma, alpha, saturation = self.p
        if pixels.ndim == 3 and pixels.shape[0] == 2:
            x, y = pixels  # .astype(np.float32)  # float32 to increase rapidity
            return np.clip(evaluate_moffat2d(x, y, amplitude, x_mean, y_mean, gamma, alpha), 0, saturation)
        elif pixels.ndim == 1:
            y = np.array(pixels)
            return np.clip(evaluate_moffat1d(y, amplitude, y_mean, gamma, alpha), 0, saturation)
        else:  # pragma: no cover
            self.my_logger.error(f"\n\tPixels array must have dimension 1 or shape=(2,Nx,Ny). "
                                 f"Here pixels.ndim={pixels.shape}.")
            return None


class MoffatGauss(PSF):

    def __init__(self, p=None):
        PSF.__init__(self)
        self.p_default = np.array([1, 0, 0, 3, 2, 0, 1, 1]).astype(float)
        if p is not None:
            self.p = np.asarray(p).astype(float)
        else:
            self.p = np.copy(self.p_default)
        self.param_names = ["amplitude", "x_mean", "y_mean", "gamma", "alpha", "eta_gauss", "stddev",
                            "saturation"]
        self.axis_names = ["$A$", r"$x_0$", r"$y_0$", r"$\gamma$", r"$\alpha$", r"$\eta$", r"$\sigma$", "saturation"]
        self.bounds = np.array([(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0.1, np.inf), (1.1, 10),
                                (-1, 0), (0.1, np.inf), (0, np.inf)])

    def apply_max_width_to_bounds(self, max_half_width=None):
        if max_half_width is not None:
            self.max_half_width = max_half_width
        self.bounds = np.array([(0, np.inf), (-np.inf, np.inf), (0, 2 * self.max_half_width),
                                (0.1, self.max_half_width), (1.1, 10), (-1, 0), (0.1, self.max_half_width / 2),
                                (0, np.inf)])

    def evaluate(self, pixels, p=None):
        """Evaluate the MoffatGauss function.

        The function is normalized to have an integral equal to amplitude parameter.

        Parameters
        ----------
        pixels: list
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        p: array_like
            The parameter array. If None, the array used to instanciate the class is taken.
            If given, the class instance parameter array is updated.

        Returns
        -------
        output: array_like
            The PSF function evaluated.

        Examples
        --------
        >>> p = [2,20,30,4,2,-0.5,1,10]
        >>> psf = MoffatGauss(p)
        >>> xx, yy = np.mgrid[:50, :60]
        >>> out = psf.evaluate(pixels=np.array([xx, yy]))

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np
            from spectractor.extractor.psf import MoffatGauss
            p = [2,20,30,4,2,-0.5,1,10]
            psf = MoffatGauss(p)
            xx, yy = np.mgrid[:50, :60]
            out = psf.evaluate(pixels=np.array([xx, yy]))
            fig = plt.figure(figsize=(5,5))
            plt.imshow(out, origin="lower")
            plt.xlabel("X [pixels]")
            plt.ylabel("Y [pixels]")
            plt.show()

        """
        if p is not None:
            self.p = np.asarray(p).astype(float)
        amplitude, x_mean, y_mean, gamma, alpha, eta_gauss, stddev, saturation = self.p
        if pixels.ndim == 3 and pixels.shape[0] == 2:
            x, y = pixels  # .astype(np.float32)  # float32 to increase rapidity
            return np.clip(evaluate_moffatgauss2d(x, y, amplitude, x_mean, y_mean,
                                                  gamma, alpha, eta_gauss, stddev), 0, saturation)
        elif pixels.ndim == 1:
            y = np.array(pixels)
            return np.clip(evaluate_moffatgauss1d(y, amplitude, y_mean, gamma, alpha, eta_gauss, stddev), 0, saturation)
        else:  # pragma: no cover
            self.my_logger.error(f"\n\tPixels array must have dimension 1 or shape=(2,Nx,Ny). "
                                 f"Here pixels.ndim={pixels.shape}.")
            return None


class PSFFitWorkspace(FitWorkspace):
    """Generic PSF fitting workspace.

    """

    def __init__(self, psf, data, data_errors, bgd_model_func=None, file_name="",
                 nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        """

        Parameters
        ----------
        psf
        data: array_like
            The data array (background subtracted) of dimension 1 or 2.
        data_errors
        bgd_model_func
        file_name
        nwalkers
        nsteps
        burnin
        nbins
        verbose
        plot
        live_fit
        truth

        Examples
        --------

        Build a mock spectrogram with random Poisson noise:

        >>> p = np.array([100, 50, 50, 3, 2, -0.1, 2, 200])
        >>> psf = MoffatGauss(p)
        >>> data = psf.evaluate(p)
        >>> data_errors = np.sqrt(data+1)

        Fit the data:

        >>> w = PSFFitWorkspace(psf, data, data_errors, bgd_model_func=None, verbose=True)

        """
        FitWorkspace.__init__(self, file_name, nwalkers, nsteps, burnin, nbins, verbose=verbose, plot=plot,
                              live_fit=live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        if data.shape != data_errors.shape:
            self.my_logger.error(f"\n\tData and uncertainty arrays must have the same shapes. "
                                 f"Here data.shape={data.shape} and data_errors.shape={data_errors.shape}.")
        self.psf = psf
        self.data = data
        self.err = data_errors
        self.bgd_model_func = bgd_model_func
        self.p = np.copy(self.psf.p)  # [1:])
        self.guess = np.copy(self.psf.p)
        self.saturation = self.psf.p[-1]
        self.fixed = [False] * len(self.p)
        self.fixed[-1] = True  # fix saturation parameter
        self.input_labels = list(np.copy(self.psf.param_names))  # [1:]))
        self.axis_names = list(np.copy(self.psf.axis_names))  # [1:]))
        self.bounds = self.psf.bounds  # [1:]
        self.nwalkers = max(2 * self.ndim, nwalkers)

        # prepare the fit
        if data.ndim == 2:
            self.Ny, self.Nx = self.data.shape
            self.psf.apply_max_width_to_bounds(self.Ny // 2)
            self.pixels = np.mgrid[:self.Nx, :self.Ny]
        elif data.ndim == 1:
            self.Ny = self.data.size
            self.Nx = 1
            self.psf.apply_max_width_to_bounds(self.Ny // 2)
            self.pixels = np.arange(self.Ny)
            self.fixed[1] = True
        else:
            self.my_logger.error(f"\n\tData array must have dimension 1 or 2. Here pixels.ndim={data.ndim}.")

        # update bounds
        self.bounds = self.psf.bounds  # [1:]
        total_flux = np.sum(data)
        self.bounds[0] = (0.1 * total_flux, 2 * total_flux)

        # error matrix
        # here image uncertainties are assumed to be uncorrelated
        # (which is not exactly true in rotated images)
        self.W = 1. / (self.err * self.err)
        self.W = self.W.flatten()  # np.diag(self.W.flatten())
        self.W_dot_data = self.W * self.data.flatten()

    def simulate(self, *psf_params):
        """
        Compute a PSF model given PSF parameters and minimizing
        amplitude parameter given a data array.

        Parameters
        ----------
        psf_params: array_like
            PSF shape parameter array (all parameters except amplitude).

        Examples
        --------

        Build a mock PSF2D without background and with random Poisson noise:

        >>> p = np.array([200000, 20, 30, 5, 2, -0.1, 2, 400000])
        >>> psf = MoffatGauss(p)
        >>> xx, yy = np.mgrid[:50, :60]
        >>> data = psf.evaluate(np.array([xx, yy]), p)
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        Fit the data in 2D:

        >>> w = PSFFitWorkspace(psf, data, data_errors, bgd_model_func=None, verbose=True)
        >>> x, mod, mod_err = w.simulate(*p)
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert mod is not None
            >>> assert np.mean(np.abs(mod-data)/data_errors) < 1

        Fit the data in 1D:

        >>> data1d = data[:,int(p[1])]
        >>> data1d_err = data_errors[:,int(p[1])]
        >>> psf.p[0] = p[0] / 10.5
        >>> w = PSFFitWorkspace(psf, data1d, data1d_err, bgd_model_func=None, verbose=True)
        >>> x, mod, mod_err = w.simulate(*psf.p)
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert mod is not None
            >>> assert np.mean(np.abs(mod-data1d)/data1d_err) < 1

        .. plot::

            import numpy as np
            import matplotlib.pyplot as plt
            from spectractor.extractor.psf import *
            p = np.array([2000, 20, 30, 5, 2, -0.1, 2, 400])
            psf = MoffatGauss(p)
            xx, yy = np.mgrid[:50, :60]
            data = psf.evaluate(np.array([xx, yy]), p)
            data = np.random.poisson(data)
            data_errors = np.sqrt(data+1)
            data = np.random.poisson(data)
            data_errors = np.sqrt(data+1)
            w = PSFFitWorkspace(psf, data, data_errors, bgd_model_func=bgd_model_func, verbose=True)
            x, mod, mod_err = w.simulate(*p[:-1])
            w.plot_fit()

        """
        # Initialization of the regression
        self.p = np.copy(psf_params)
        # if not self.fixed_amplitude:
        #     # Matrix filling
        #     M = self.psf.evaluate(self.pixels, p=np.array([1] + list(self.p))).flatten()
        #     M_dot_W_dot_M = M.T @ self.W @ M
        #     # Regression
        #     amplitude = M.T @ self.W_dot_data / M_dot_W_dot_M
        #     self.p[0] = amplitude
        # Save results
        self.model = self.psf.evaluate(self.pixels, p=self.p)
        self.model_err = np.zeros_like(self.model)
        return self.pixels, self.model, self.model_err

    def plot_fit(self):
        fig = plt.figure()
        if self.data.ndim == 1:
            fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex='all', gridspec_kw={'height_ratios': [5, 1]})
            data = np.copy(self.data)
            if self.bgd_model_func is not None:
                data = data + self.bgd_model_func(self.pixels)
            ax[0].errorbar(self.pixels, data, yerr=self.err, fmt='ro', label="Data")
            if len(self.outliers) > 0:
                ax[0].errorbar(self.outliers, data[self.outliers], yerr=self.err[self.outliers], fmt='go',
                               label=rf"Outliers ({self.sigma_clip}$\sigma$)")
            if self.bgd_model_func is not None:
                ax[0].plot(self.pixels, self.bgd_model_func(self.pixels), 'b--', label="fitted bgd")
            if self.guess is not None:
                if self.bgd_model_func is not None:
                    ax[0].plot(self.pixels, self.psf.evaluate(self.pixels, p=self.guess)
                               + self.bgd_model_func(self.pixels), 'k--', label="Guess")
                else:
                    ax[0].plot(self.pixels, self.psf.evaluate(self.pixels, p=self.guess),
                               'k--', label="Guess")
                self.psf.p = np.copy(self.p)
            model = np.copy(self.model)
            # if self.bgd_model_func is not None:
            #    model = self.model + self.bgd_model_func(self.pixels)
            ax[0].plot(self.pixels, model, 'b-', label="Model")
            ylim = list(ax[0].get_ylim())
            ylim[1] = 1.2 * np.max(self.model)
            ax[0].set_ylim(ylim)
            ax[0].set_ylabel('Transverse profile')
            ax[0].legend(loc=2, numpoints=1)
            ax[0].grid(True)
            txt = ""
            for ip, p in enumerate(self.input_labels):
                txt += f'{p}: {self.p[ip]:.4g}\n'
            ax[0].text(0.95, 0.95, txt, horizontalalignment='right', verticalalignment='top', transform=ax[0].transAxes)
            # residuals
            residuals = (data - model) / self.err
            residuals_err = np.ones_like(self.err)
            ax[1].errorbar(self.pixels, residuals, yerr=residuals_err, fmt='ro')
            if len(self.outliers) > 0:
                residuals_outliers = (data[self.outliers] - model[self.outliers]) / self.err[self.outliers]
                residuals_outliers_err = np.ones_like(residuals_outliers)
                ax[1].errorbar(self.outliers, residuals_outliers, yerr=residuals_outliers_err, fmt='go')
            ax[1].axhline(0, color='b')
            ax[1].grid(True)
            std = np.std(residuals)
            ax[1].set_ylim([-3. * std, 3. * std])
            ax[1].set_xlabel(ax[0].get_xlabel())
            ax[1].set_ylabel('(data-fit)/err')
            ax[0].set_xticks(ax[1].get_xticks()[1:-1])
            ax[0].get_yaxis().set_label_coords(-0.1, 0.5)
            ax[1].get_yaxis().set_label_coords(-0.1, 0.5)
            # fig.tight_layout()
            # fig.subplots_adjust(wspace=0, hspace=0)
        elif self.data.ndim == 2:
            gs_kw = dict(width_ratios=[3, 0.15], height_ratios=[1, 1, 1, 1])
            fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(5, 7), gridspec_kw=gs_kw)
            norm = np.nanmax(self.data)
            plot_image_simple(ax[0, 0], data=self.model / norm, aspect='auto', cax=ax[0, 1], vmin=0, vmax=1,
                              units='1/max(data)')
            ax[0, 0].set_title("Model", fontsize=10, loc='center', color='white', y=0.8)
            plot_image_simple(ax[1, 0], data=self.data / norm, title='Data', aspect='auto',
                              cax=ax[1, 1], vmin=0, vmax=1, units='1/max(data)')
            ax[1, 0].set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
            residuals = (self.data - self.model)
            # residuals_err = self.spectrum.spectrogram_err / self.model
            norm = self.err
            residuals /= norm
            std = float(np.std(residuals))
            plot_image_simple(ax[2, 0], data=residuals, vmin=-5 * std, vmax=5 * std, title='(Data-Model)/Err',
                              aspect='auto', cax=ax[2, 1], units='', cmap="bwr")
            ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, loc='center', color='black', y=0.8)
            ax[2, 0].text(0.05, 0.05, f'mean={np.mean(residuals):.3f}\nstd={np.std(residuals):.3f}',
                          horizontalalignment='left', verticalalignment='bottom',
                          color='black', transform=ax[2, 0].transAxes)
            ax[0, 0].set_xticks(ax[2, 0].get_xticks()[1:-1])
            ax[0, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[1, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[2, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[3, 1].remove()
            ax[3, 0].plot(np.arange(self.Nx), self.data.sum(axis=0), label='Data')
            ax[3, 0].plot(np.arange(self.Nx), self.model.sum(axis=0), label='Model')
            ax[3, 0].set_ylabel('Transverse sum')
            ax[3, 0].set_xlabel(r'X [pixels]')
            ax[3, 0].legend(fontsize=7)
            ax[3, 0].grid(True)
        else:
            self.my_logger.error(f"\n\tData array must have dimension 1 or 2. Here data.ndim={self.data.ndim}.")
        if self.live_fit:  # pragma: no cover
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            if parameters.DISPLAY:
                plt.show()
            else:
                plt.close(fig)
        if parameters.SAVE:  # pragma: no cover
            figname = self.filename.replace(self.filename.split('.')[-1], "_bestfit.pdf")
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')


def PSF2D_chisq(params, model, xx, yy, zz, zz_err=None):
    mod = model.evaluate(xx, yy, *params)
    if zz_err is None:
        return np.nansum((mod - zz) ** 2)
    else:
        return np.nansum(((mod - zz) / zz_err) ** 2)


def PSF2D_chisq_jac(params, model, xx, yy, zz, zz_err=None):
    diff = model.evaluate(xx, yy, *params) - zz
    jac = model.fit_deriv(xx, yy, *params)
    if zz_err is None:
        return np.array([np.nansum(2 * jac[p] * diff) for p in range(len(params))])
    else:
        zz_err2 = zz_err * zz_err
        return np.array([np.nansum(2 * jac[p] * diff / zz_err2) for p in range(len(params))])


# DO NOT WORK
# def fit_PSF2D_outlier_removal(x, y, data, sigma=3.0, niter=3, guess=None, bounds=None):
#     """Fit a PSF 2D model with parameters:
#         amplitude_gauss, x_mean, stddev, amplitude, alpha, gamma, saturation
#     using scipy. Find outliers data point above sigma*data_errors from the fit over niter iterations.
#
#     Parameters
#     ----------
#     x: np.array
#         2D array of the x coordinates.
#     y: np.array
#         2D array of the y coordinates.
#     data: np.array
#         the 1D array profile.
#     guess: array_like, optional
#         list containing a first guess for the PSF parameters (default: None).
#     bounds: list, optional
#         2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
#     sigma: int
#         the sigma limit to exclude data points (default: 3).
#     niter: int
#         the number of loop iterations to exclude  outliers and refit the model (default: 2).
#
#     Returns
#     -------
#     fitted_model: MoffatGauss2D
#         the MoffatGauss2D fitted model.
#
#     Examples
#     --------
#
#     Create the model:
#     >>> X, Y = np.mgrid[:50,:50]
#     >>> PSF = MoffatGauss2D()
#     >>> p = (1000, 25, 25, 5, 1, -0.2, 1, 6000)
#     >>> Z = PSF.evaluate(X, Y, *p)
#     >>> Z += 100*np.exp(-((X-10)**2+(Y-10)**2)/4)
#     >>> Z_err = np.sqrt(1+Z)
#
#     Prepare the fit:
#     >>> guess = (1000, 27, 23, 3.2, 1.2, -0.1, 2,  6000)
#     >>> bounds = np.array(((0, 6000), (10, 40), (10, 40), (0.5, 10), (0.5, 5), (-1, 0), (0.01, 10), (0, 8000)))
#     >>> bounds = bounds.T
#
#     Fit without bars:
#     >>> model = fit_PSF2D_outlier_removal(X, Y, Z, guess=guess, bounds=bounds, sigma=7, niter=5)
#     >>> res = [getattr(model, p).value for p in model.param_names]
#     >>> print(res, p)
#     >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))
#     """
#     gg_init = MoffatGauss2D()
#     if guess is not None:
#         for ip, p in enumerate(gg_init.param_names):
#             getattr(gg_init, p).value = guess[ip]
#     if bounds is not None:
#         for ip, p in enumerate(gg_init.param_names):
#             getattr(gg_init, p).min = bounds[0][ip]
#             getattr(gg_init, p).max = bounds[1][ip]
#     gg_init.saturation.fixed = True
#     with warnings.catch_warnings():
#         # Ignore model linearity warning from the fitter
#         warnings.simplefilter('ignore')
#         fit = LevMarLSQFitterWithNan()
#         or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)
#         # get fitted model and filtered data
#         or_fitted_model, filtered_data = or_fit(gg_init, x, y, data)
#         if parameters.VERBOSE:
#             print(or_fitted_model)
#         if parameters.DEBUG:
#             print(fit.fit_info)
#         print(fit.fit_info)
#         return or_fitted_model


def fit_PSF2D(x, y, data, guess=None, bounds=None, data_errors=None, method='minimize'):
    """
    Fit a PSF 2D model with parameters: amplitude, x_mean, y_mean, stddev, eta, alpha, gamma, saturation
    using basin hopping global minimization method.

    Parameters
    ----------
    x: np.array
        2D array of the x coordinates from meshgrid.
    y: np.array
        2D array of the y coordinates from meshgrid.
    data: np.array
        the 2D array image.
    guess: array_like, optional
        List containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    data_errors: np.array
        the 2D array uncertainties.
    method: str, optional
        the minimisation method: 'minimize' or 'basinhopping' (default: 'minimize').

    Returns
    -------
    fitted_model: PSF2DAstropy
        the PSF fitted model.

    Examples
    --------

    Create the model

    >>> import numpy as np
    >>> X, Y = np.mgrid[:50,:50]
    >>> psf = PSF2DAstropy()
    >>> p = (50, 25, 25, 5, 1, -0.4, 1, 60)
    >>> Z = psf.evaluate(X, Y, *p)
    >>> Z_err = np.sqrt(Z)/10.

    Prepare the fit

    >>> guess = (52, 22, 22, 3.2, 1.2, -0.1, 2, 60)
    >>> bounds = ((1, 200), (10, 40), (10, 40), (0.5, 10), (0.5, 5), (-100, 200), (0.01, 10), (0, 400))

    Fit with error bars

    >>> model = fit_PSF2D(X, Y, Z, guess=guess, bounds=bounds, data_errors=Z_err)
    >>> res = [getattr(model, p).value for p in model.param_names]

    ..  doctest::
        :hide:

        >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit without error bars

    >>> model = fit_PSF2D(X, Y, Z, guess=guess, bounds=bounds, data_errors=None)
    >>> res = [getattr(model, p).value for p in model.param_names]

    ..  doctest::
        :hide:

        >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit with error bars and basin hopping method

    >>> model = fit_PSF2D(X, Y, Z, guess=guess, bounds=bounds, data_errors=Z_err, method='basinhopping')
    >>> res = [getattr(model, p).value for p in model.param_names]

    ..  doctest::
        :hide:

        >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    """

    model = PSF2DAstropy()
    my_logger = set_logger(__name__)
    if method == 'minimize':
        res = minimize(PSF2D_chisq, guess, method="L-BFGS-B", bounds=bounds,
                       args=(model, x, y, data, data_errors), jac=PSF2D_chisq_jac)
    elif method == 'basinhopping':
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, jac=PSF2D_chisq_jac,
                                args=(model, x, y, data, data_errors))
        res = basinhopping(PSF2D_chisq, guess, niter=20, minimizer_kwargs=minimizer_kwargs)
    else:
        my_logger.error(f'\n\tUnknown method {method}.')
        sys.exit()
    my_logger.debug(f'\n{res}')
    psf = PSF2DAstropy(*res.x)
    my_logger.debug(f'\n\tPSF best fitting parameters:\n{psf}')
    return psf


def fit_PSF2D_minuit(x, y, data, guess=None, bounds=None, data_errors=None):
    """
    Fit a PSF 2D model with parameters: amplitude, x_mean, y_mean, stddev, eta, alpha, gamma, saturation
    using basin hopping global minimization method.

    Parameters
    ----------
    x: np.array
        2D array of the x coordinates from meshgrid.
    y: np.array
        2D array of the y coordinates from meshgrid.
    data: np.array
        the 2D array image.
    guess: array_like, optional
        List containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    data_errors: np.array
        the 2D array uncertainties.

    Returns
    -------
    fitted_model: PSF2DAstropy
        the PSF2D fitted model.

    Examples
    --------

    Create the model

    >>> import numpy as np
    >>> X, Y = np.mgrid[:50,:50]
    >>> psf = PSF2DAstropy()
    >>> p = (50, 25, 25, 5, 1, -0.4, 1, 60)
    >>> Z = psf.evaluate(X, Y, *p)
    >>> Z_err = np.sqrt(Z)/10.

    Prepare the fit

    >>> guess = (52, 22, 22, 3.2, 1.2, -0.1, 2, 60)
    >>> bounds = ((1, 200), (10, 40), (10, 40), (0.5, 10), (0.5, 5), (-100, 200), (0.01, 10), (0, 400))

    Fit with error bars

    >>> model = fit_PSF2D_minuit(X, Y, Z, guess=guess, bounds=bounds, data_errors=Z_err)
    >>> res = [getattr(model, p).value for p in model.param_names]

    ..  doctest::
        :hide:

        >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    Fit without error bars

    >>> model = fit_PSF2D_minuit(X, Y, Z, guess=guess, bounds=bounds, data_errors=None)
    >>> res = [getattr(model, p).value for p in model.param_names]

    ..  doctest::
        :hide:

        >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))
    """

    model = PSF2DAstropy()
    my_logger = set_logger(__name__)

    if bounds is not None:
        bounds = np.array(bounds)
        if bounds.shape[0] == 2 and bounds.shape[1] > 2:
            bounds = bounds.T

    guess = np.array(guess)
    error = 0.001 * np.abs(guess) * np.ones_like(guess)
    z = np.where(np.isclose(error, 0.0, 1e-6))
    error[z] = 0.001

    def chisq_PSF2D(params):
        return PSF2D_chisq(params, model, x, y, data, data_errors)

    def chisq_PSF2D_jac(params):
        return PSF2D_chisq_jac(params, model, x, y, data, data_errors)

    fix = [False] * error.size
    fix[-1] = True
    # noinspection PyArgumentList
    m = Minuit.from_array_func(fcn=chisq_PSF2D, start=guess, error=error, errordef=1,
                               fix=fix, print_level=0, limit=bounds, grad=chisq_PSF2D_jac)

    m.tol = 0.001
    m.migrad()
    popt = m.np_values()

    my_logger.debug(f'\n{popt}')
    psf = PSF2DAstropy(*popt)
    my_logger.debug(f'\n\tPSF best fitting parameters:\n{psf}')
    return psf


@deprecated(reason='Use MoffatGauss1D class instead.')
class PSF1DAstropy(Fittable1DModel):
    n_inputs = 1
    n_outputs = 1
    # inputs = ('x',)
    # outputs = ('y',)

    amplitude_moffat = Parameter('amplitude_moffat', default=0.5)
    x_mean = Parameter('x_mean', default=0)
    gamma = Parameter('gamma', default=3)
    alpha = Parameter('alpha', default=3)
    eta_gauss = Parameter('eta_gauss', default=1)
    stddev = Parameter('stddev', default=1)
    saturation = Parameter('saturation', default=1)

    axis_names = ["A", "y", r"\gamma", r"\alpha", r"\eta", r"\sigma", "saturation"]

    @staticmethod
    def evaluate(x, amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = (x - x_mean) * (x - x_mean)
        rr_gg = rr / (gamma * gamma)
        # use **(-alpha) instead of **(alpha) to avoid overflow power errors due to high alpha exponents
        # import warnings
        # warnings.filterwarnings('error')
        try:
            a = amplitude_moffat * ((1 + rr_gg) ** (-alpha) + eta_gauss * np.exp(-(rr / (2. * stddev * stddev))))
        except RuntimeWarning:  # pragma: no cover
            my_logger = set_logger(__name__)
            my_logger.warning(f"{[amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation]}")
            a = amplitude_moffat * eta_gauss * np.exp(-(rr / (2. * stddev * stddev)))
        return np.clip(a, 0, saturation)

    @staticmethod
    def fit_deriv(x, amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = (x - x_mean) * (x - x_mean)
        rr_gg = rr / (gamma * gamma)
        gauss_norm = np.exp(-(rr / (2. * stddev * stddev)))
        d_eta_gauss = amplitude_moffat * gauss_norm
        moffat_norm = (1 + rr_gg) ** (-alpha)
        d_amplitude_moffat = moffat_norm + eta_gauss * gauss_norm
        d_x_mean = amplitude_moffat * (eta_gauss * (x - x_mean) / (stddev * stddev) * gauss_norm
                                       - alpha * moffat_norm * (-2 * x + 2 * x_mean) / (
                                               gamma * gamma * (1 + rr_gg)))
        d_stddev = amplitude_moffat * eta_gauss * rr / (stddev ** 3) * gauss_norm
        d_alpha = - amplitude_moffat * moffat_norm * np.log(1 + rr_gg)
        d_gamma = 2 * amplitude_moffat * alpha * moffat_norm * (rr_gg / (gamma * (1 + rr_gg)))
        d_saturation = saturation * np.zeros_like(x)
        return np.array([d_amplitude_moffat, d_x_mean, d_gamma, d_alpha, d_eta_gauss, d_stddev, d_saturation])

    @staticmethod
    def deriv(x, amplitude_moffat, x_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = (x - x_mean) * (x - x_mean)
        rr_gg = rr / (gamma * gamma)
        d_eta_gauss = np.exp(-(rr / (2. * stddev * stddev)))
        d_gauss = - eta_gauss * (x - x_mean) / (stddev * stddev) * d_eta_gauss
        d_moffat = -  alpha * 2 * (x - x_mean) / (gamma * gamma * (1 + rr_gg) ** (alpha + 1))
        return amplitude_moffat * (d_gauss + d_moffat)

    def interpolation(self, x_array):
        """

        Parameters
        ----------
        x_array: array_like
            The abscisse array to interpolate the model.

        Returns
        -------
        interp: callable
            Function corresponding to the interpolated model on the x_array array.

        Examples
        --------
        >>> x = np.arange(0, 60, 1)
        >>> p = [2,0,2,2,1,2,10]
        >>> psf = PSF1DAstropy(*p)
        >>> interp = psf.interpolation(x)

        ..  doctest::
            :hide:

            >>> assert np.isclose(interp(p[1]), psf.evaluate(p[1], *p))

        """
        params = [getattr(self, p).value for p in self.param_names]
        return interp1d(x_array, self.evaluate(x_array, *params), fill_value=0, bounds_error=False)

    def integrate(self, bounds=(-np.inf, np.inf), x_array=None):
        """
        Compute the integral of the PSF model. Bounds are -np.inf, np.inf by default, or provided
        if no x_array is provided. Otherwise the bounds comes from x_array edges.

        Parameters
        ----------
        x_array: array_like, optional
            If not None, the interpoalted PSF modelis used for integration (default: None).
        bounds: array_like, optional
            The bounds of the integral (default bounds=(-np.inf, np.inf)).

        Returns
        -------
        result: float
            The integral of the PSF model.

        Examples
        --------

        .. doctest::

            >>> x = np.arange(0, 60, 1)
            >>> p = [2,30,4,2,-0.5,1,10]
            >>> psf = PSF1DAstropy(*p)
            >>> xx = np.arange(0, 60, 0.01)
            >>> plt.plot(xx, psf.evaluate(xx, *p)) # doctest: +ELLIPSIS
            [<matplotlib.lines.Line2D object at ...>]
            >>> plt.plot(x, psf.evaluate(x, *p)) # doctest: +ELLIPSIS
            [<matplotlib.lines.Line2D object at ...>]
            >>> if parameters.DISPLAY: plt.show()

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np
            from spectractor.extractor.psf import PSF1DAstropy
            p = [2,30,4,2,-0.5,1,10]
            x = np.arange(0, 60, 1)
            xx = np.arange(0, 60, 0.01)
            psf = PSF1DAstropy(*p)
            fig = plt.figure(figsize=(5,3))
            plt.plot(xx, psf.evaluate(xx, *p), label="high sampling")
            plt.plot(x, psf.evaluate(x, *p), label="low sampling")
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('PSF(x)')
            plt.legend()
            plt.show()

        .. doctest::

            >>> psf.integrate()  # doctest: +ELLIPSIS
            10.0597...
            >>> psf.integrate(bounds=(0,60), x_array=x)  # doctest: +ELLIPSIS
            10.0466...

        """
        params = [getattr(self, p).value for p in self.param_names]
        if x_array is None:
            i = quad(self.evaluate, bounds[0], bounds[1], args=tuple(params), limit=200)
            return i[0]
        else:
            return np.trapz(self.evaluate(x_array, *params), x_array)

    def fwhm(self, x_array=None):
        """
        Compute the full width half maximum of the PSF model with a dichotomie method.

        Parameters
        ----------
        x_array: array_like, optional
            An abscisse array is one wants to find FWHM on the interpolated PSF model
            (to smooth the spikes from spurious parameter sets).

        Returns
        -------
        FWHM: float
            The full width half maximum of the PSF model.

        Examples
        --------
        >>> x = np.arange(0, 60, 1)
        >>> p = [2,30,4,2,-0.4,1,10]
        >>> psf = PSF1DAstropy(*p)
        >>> a, b =  p[1], p[1]+3*max(p[-2], p[2])
        >>> fwhm = psf.fwhm(x_array=None)
        >>> assert np.isclose(fwhm, 7.25390625)
        >>> fwhm = psf.fwhm(x_array=x)
        >>> assert np.isclose(fwhm, 7.083984375)
        >>> print(fwhm)
        7.083984375
        >>> import matplotlib.pyplot as plt
        >>> x = np.arange(0, 60, 0.01)
        >>> plt.plot(x, psf.evaluate(x, *p)) # doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object at 0x...>]
        >>> if parameters.DISPLAY: plt.show()
        """
        params = [getattr(self, p).value for p in self.param_names]
        interp = None
        if x_array is not None:
            interp = self.interpolation(x_array)
            values = self.evaluate(x_array, *params)
            maximum = np.max(values)
            imax = np.argmax(values)
            a = imax + np.argmin(np.abs(values[imax:] - 0.95 * maximum))
            b = imax + np.argmin(np.abs(values[imax:] - 0.05 * maximum))

            def eq(x):
                return interp(x) - 0.5 * maximum
        else:
            maximum = self.amplitude_moffat.value * (1 + self.eta_gauss.value)
            a = self.x_mean.value
            b = self.x_mean.value + 3 * max(self.gamma.value, self.stddev.value)

            def eq(x):
                return self.evaluate(x, *params) - 0.5 * maximum
        res = dichotomie(eq, a, b, 1e-2)
        # res = newton()
        return abs(2 * (res - self.x_mean.value))


@deprecated(reason='Use MoffatGauss1D class instead.')
def PSF1D_chisq(params, model, xx, yy, yy_err=None):
    m = model.evaluate(xx, *params)
    if len(m) == 0 or len(yy) == 0:
        return 1e20
    if np.any(m < 0) or np.any(m > 1.5 * np.max(yy)) or np.max(m) < 0.5 * np.max(yy):
        return 1e20
    diff = m - yy
    if yy_err is None:
        return np.nansum(diff * diff)
    else:
        return np.nansum((diff / yy_err) ** 2)


@deprecated(reason='Use MoffatGauss1D class instead.')
def PSF1D_chisq_jac(params, model, xx, yy, yy_err=None):
    diff = model.evaluate(xx, *params) - yy
    jac = model.fit_deriv(xx, *params)
    if yy_err is None:
        return np.array([np.nansum(2 * jac[p] * diff) for p in range(len(params))])
    else:
        yy_err2 = yy_err * yy_err
        return np.array([np.nansum(2 * jac[p] * diff / yy_err2) for p in range(len(params))])


@deprecated(reason='Use MoffatGauss1D class instead.')
def fit_PSF1D(x, data, guess=None, bounds=None, data_errors=None, method='minimize'):
    """Fit a PSF 1D Astropy model with parameters :
        amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation

    using basin hopping global minimization method.

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates.
    data: np.array
        the 1D array profile.
    guess: array_like, optional
        list containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    data_errors: np.array
        the 1D array uncertainties.
    method: str, optional
        method to use for the minimisation: choose between minimize and basinhopping.

    Returns
    -------
    fitted_model: PSF1DAstropy
        the PSF fitted model.

    Examples
    --------

    Create the model:

    >>> import numpy as np
    >>> X = np.arange(0, 50)
    >>> psf = PSF1DAstropy()
    >>> p = (50, 25, 5, 1, -0.2, 1, 60)
    >>> Y = psf.evaluate(X, *p)
    >>> Y_err = np.sqrt(Y)/10.

    Prepare the fit:

    >>> guess = (60, 20, 3.2, 1.2, -0.1, 2,  60)
    >>> bounds = ((0, 200), (10, 40), (0.5, 10), (0.5, 5), (-10, 200), (0.01, 10), (0, 400))

    Fit with error bars:
    # >>> model = fit_PSF1D(X, Y, guess=guess, bounds=bounds, data_errors=Y_err)
    # >>> res = [getattr(model, p).value for p in model.param_names]
    # >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))
    #
    # Fit without error bars:
    # >>> model = fit_PSF1D(X, Y, guess=guess, bounds=bounds, data_errors=None)
    # >>> res = [getattr(model, p).value for p in model.param_names]
    # >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))
    #
    # Fit with error bars and basin hopping method:
    # >>> model = fit_PSF1D(X, Y, guess=guess, bounds=bounds, data_errors=Y_err, method='basinhopping')
    # >>> res = [getattr(model, p).value for p in model.param_names]
    # >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-3))

    """
    my_logger = set_logger(__name__)
    model = PSF1DAstropy()
    if method == 'minimize':
        res = minimize(PSF1D_chisq, guess, method="L-BFGS-B", bounds=bounds,
                       args=(model, x, data, data_errors), jac=PSF1D_chisq_jac)
    elif method == 'basinhopping':
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds,
                                args=(model, x, data, data_errors), jac=PSF1D_chisq_jac)
        res = basinhopping(PSF1D_chisq, guess, niter=20, minimizer_kwargs=minimizer_kwargs)
    else:
        my_logger.error(f'\n\tUnknown method {method}.')
        sys.exit()
    my_logger.debug(f'\n{res}')
    psf = PSF1DAstropy(*res.x)
    my_logger.debug(f'\n\tPSF best fitting parameters:\n{psf}')
    return psf


@deprecated(reason='Use MoffatGauss1D class instead. Mainly because PSF integral must be normalized to one.')
def fit_PSF1D_outlier_removal(x, data, data_errors=None, sigma=3.0, niter=3, guess=None, bounds=None, method='minimize',
                              niter_basinhopping=5, T_basinhopping=0.2):
    """Fit a PSF 1D Astropy model with parameters:
        amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation

    using scipy. Find outliers data point above sigma*data_errors from the fit over niter iterations.

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates.
    data: np.array
        the 1D array profile.
    data_errors: np.array
        the 1D array uncertainties.
    guess: array_like, optional
        list containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    sigma: int
        the sigma limit to exclude data points (default: 3).
    niter: int
        the number of loop iterations to exclude  outliers and refit the model (default: 2).
    method: str
        Can be 'minimize' or 'basinhopping' (default: 'minimize').
    niter_basinhopping: int, optional
        The number of basin hops (default: 5)
    T_basinhopping: float, optional
        The temperature for the basin hops (default: 0.2)

    Returns
    -------
    fitted_model: PSF1DAstropy
        the PSF fitted model.
    outliers: list
        the list of the outlier indices.

    Examples
    --------

    Create the model:

    >>> import numpy as np
    >>> X = np.arange(0, 50)
    >>> psf = PSF1DAstropy()
    >>> p = (1000, 25, 5, 1, -0.2, 1, 6000)
    >>> Y = psf.evaluate(X, *p)
    >>> Y += 100*np.exp(-((X-10)/2)**2)
    >>> Y_err = np.sqrt(1+Y)

    Prepare the fit:

    >>> guess = (600, 27, 3.2, 1.2, -0.1, 2,  6000)
    >>> bounds = ((0, 6000), (10, 40), (0.5, 10), (0.5, 5), (-1, 0), (0.01, 10), (0, 8000))

    Fit without bars:
    # >>> model, outliers = fit_PSF1D_outlier_removal(X, Y, guess=guess, bounds=bounds,
    # ... sigma=3, niter=5, method="minimize")
    # >>> res = [getattr(model, p).value for p in model.param_names]
    # >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))
    #
    # Fit with error bars:
    # >>> model, outliers = fit_PSF1D_outlier_removal(X, Y, guess=guess, bounds=bounds, data_errors=Y_err,
    # ... sigma=3, niter=2, method="minimize")
    # >>> res = [getattr(model, p).value for p in model.param_names]
    # >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))
    #
    # Fit with error bars and basinhopping:
    # >>> model, outliers = fit_PSF1D_outlier_removal(X, Y, guess=guess, bounds=bounds, data_errors=Y_err,
    # ... sigma=3, niter=5, method="basinhopping", niter_basinhopping=20)
    # >>> res = [getattr(model, p).value for p in model.param_names]
    # >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))
    """

    my_logger = set_logger(__name__)
    indices = np.mgrid[:x.shape[0]]
    outliers = np.array([])
    model = PSF1DAstropy()

    for step in range(niter):
        # first fit
        if data_errors is None:
            err = None
        else:
            err = data_errors[indices]
        if method == 'minimize':
            res = minimize(PSF1D_chisq, guess, method="L-BFGS-B", bounds=bounds, jac=PSF1D_chisq_jac,
                           args=(model, x[indices], data[indices], err))
        elif method == 'basinhopping':
            minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, jac=PSF1D_chisq_jac,
                                    args=(model, x[indices], data[indices], err))
            res = basinhopping(PSF1D_chisq, guess, T=T_basinhopping, niter=niter_basinhopping,
                               minimizer_kwargs=minimizer_kwargs)
        else:
            my_logger.error(f'\n\tUnknown method {method}.')
            sys.exit()
        if parameters.DEBUG:
            my_logger.debug(f'\n\tniter={step}\n{res}')
        # update the model and the guess
        for ip, p in enumerate(model.param_names):
            setattr(model, p, res.x[ip])
        guess = res.x
        # remove outliers
        indices_no_nan = ~np.isnan(data)
        diff = model(x[indices_no_nan]) - data[indices_no_nan]
        if data_errors is not None:
            outliers = np.where(np.abs(diff) / data_errors[indices_no_nan] > sigma)[0]
        else:
            std = np.std(diff)
            outliers = np.where(np.abs(diff) / std > sigma)[0]
        if len(outliers) > 0:
            indices = [i for i in range(x.shape[0]) if i not in outliers]
        else:
            break
    my_logger.debug(f'\n\tPSF best fitting parameters:\n{model}')
    return model, outliers


@deprecated(reason='Use MoffatGauss1D class instead.')
def fit_PSF1D_minuit(x, data, guess=None, bounds=None, data_errors=None):
    """Fit a PSF 1D Astropy model with parameters:
        amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation

    using Minuit.

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates.
    data: np.array
        the 1D array profile.
    guess: array_like, optional
        list containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    data_errors: np.array
        the 1D array uncertainties.

    Returns
    -------
    fitted_model: PSF1DAstropy
        the PSF fitted model.

    Examples
    --------

    Create the model:

    >>> import numpy as np
    >>> X = np.arange(0, 50)
    >>> psf = PSF1DAstropy()
    >>> p = (50, 25, 5, 1, -0.2, 1, 60)
    >>> Y = psf.evaluate(X, *p)
    >>> Y_err = np.sqrt(1+Y)

    Prepare the fit:

    >>> guess = (60, 20, 3.2, 1.2, -0.1, 2,  60)
    >>> bounds = ((0, 200), (10, 40), (0.5, 10), (0.5, 5), (-1, 0), (0.01, 10), (0, 400))

    Fit with error bars:
    # >>> model = fit_PSF1D_minuit(X, Y, guess=guess, bounds=bounds, data_errors=Y_err)
    # >>> res = [getattr(model, p).value for p in model.param_names]
    # >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-2))
    #
    # Fit without error bars:
    # >>> model = fit_PSF1D_minuit(X, Y, guess=guess, bounds=bounds, data_errors=None)
    # >>> res = [getattr(model, p).value for p in model.param_names]
    # >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-2))

    """

    my_logger = set_logger(__name__)
    model = PSF1DAstropy()

    def PSF1D_chisq_v2(params):
        mod = model.evaluate(x, *params)
        diff = mod - data
        if data_errors is None:
            return np.nansum(diff * diff)
        else:
            return np.nansum((diff / data_errors) ** 2)

    error = 0.1 * np.abs(guess) * np.ones_like(guess)
    fix = [False] * len(guess)
    fix[-1] = True
    # noinspection PyArgumentList
    # 3 times faster with gradient
    m = Minuit.from_array_func(fcn=PSF1D_chisq_v2, start=guess, error=error, errordef=1, limit=bounds, fix=fix,
                               print_level=parameters.DEBUG)
    m.migrad()
    psf = PSF1DAstropy(*m.np_values())

    my_logger.debug(f'\n\tPSF best fitting parameters:\n{psf}')
    return psf


@deprecated(reason='Use MoffatGauss1D class instead.')
def fit_PSF1D_minuit_outlier_removal(x, data, data_errors, guess=None, bounds=None, sigma=3, niter=2, consecutive=3):
    """Fit a PSF Astropy 1D model with parameters:
        amplitude_gauss, x_mean, stddev, amplitude_moffat, alpha, gamma, saturation

    using Minuit. Find outliers data point above sigma*data_errors from the fit over niter iterations.
    Only at least 3 consecutive outliers are considered.

    Parameters
    ----------
    x: np.array
        1D array of the x coordinates.
    data: np.array
        the 1D array profile.
    data_errors: np.array
        the 1D array uncertainties.
    guess: array_like, optional
        list containing a first guess for the PSF parameters (default: None).
    bounds: list, optional
        2D list containing bounds for the PSF parameters with format ((min,...), (max...)) (default: None)
    sigma: int
        the sigma limit to exclude data points (default: 3).
    niter: int
        the number of loop iterations to exclude  outliers and refit the model (default: 2).
    consecutive: int
        the number of outliers that have to be consecutive to be considered (default: 3).

    Returns
    -------
    fitted_model: PSF1DAstropy
        the PSF fitted model.
    outliers: list
        the list of the outlier indices.

    Examples
    --------

    Create the model:

    >>> import numpy as np
    >>> X = np.arange(0, 50)
    >>> psf = PSF1DAstropy()
    >>> p = (1000, 25, 5, 1, -0.2, 1, 6000)
    >>> Y = psf.evaluate(X, *p)
    >>> Y += 100*np.exp(-((X-10)/2)**2)
    >>> Y_err = np.sqrt(1+Y)

    Prepare the fit:

    >>> guess = (600, 20, 3.2, 1.2, -0.1, 2,  6000)
    >>> bounds = ((0, 6000), (10, 40), (0.5, 10), (0.5, 5), (-1, 0), (0.01, 10), (0, 8000))

    Fit with error bars:
    # >>> model, outliers = fit_PSF1D_minuit_outlier_removal(X, Y, guess=guess, bounds=bounds, data_errors=Y_err,
    # ... sigma=3, niter=2, consecutive=3)
    # >>> res = [getattr(model, p).value for p in model.param_names]
    # >>> assert np.all(np.isclose(p[:-1], res[:-1], rtol=1e-1))
    """

    psf = PSF1DAstropy(*guess)
    model = PSF1DAstropy()
    outliers = np.array([])
    indices = [i for i in range(x.shape[0]) if i not in outliers]

    def PSF1D_chisq_v2(params):
        mod = model.evaluate(x, *params)
        diff = mod[indices] - data[indices]
        if data_errors is None:
            return np.nansum(diff * diff)
        else:
            return np.nansum((diff / data_errors[indices]) ** 2)

    error = 0.1 * np.abs(guess) * np.ones_like(guess)
    fix = [False] * len(guess)
    fix[-1] = True

    consecutive_outliers = []
    for step in range(niter):
        # noinspection PyArgumentList
        # it seems that minuit with a jacobian function works less good...
        m = Minuit.from_array_func(fcn=PSF1D_chisq_v2, start=guess, error=error, errordef=1, limit=bounds, fix=fix,
                                   print_level=0, grad=None)
        m.migrad()
        guess = m.np_values()
        psf = PSF1DAstropy(*m.np_values())
        for ip, p in enumerate(model.param_names):
            setattr(model, p, guess[ip])
        # remove outliers
        indices_no_nan = ~np.isnan(data)
        diff = model(x[indices_no_nan]) - data[indices_no_nan]
        if data_errors is not None:
            outliers = np.where(np.abs(diff) / data_errors[indices_no_nan] > sigma)[0]
        else:
            std = np.std(diff)
            outliers = np.where(np.abs(diff) / std > sigma)[0]
        if len(outliers) > 0:
            # test if 3 consecutive pixels are in the outlier list
            test = 0
            consecutive_outliers = []
            for o in range(1, len(outliers)):
                t = outliers[o] - outliers[o - 1]
                if t == 1:
                    test += t
                else:
                    test = 0
                if test >= consecutive - 1:
                    for i in range(consecutive):
                        consecutive_outliers.append(outliers[o - i])
            consecutive_outliers = list(set(consecutive_outliers))
            # my_logger.debug(f"\n\tConsecutive oultlier indices: {consecutive_outliers}")
            indices = [i for i in range(x.shape[0]) if i not in outliers]
        else:
            break

    # my_logger.debug(f'\n\tPSF best fitting parameters:\n{PSF}')
    return psf, consecutive_outliers


@deprecated(reason="Use new MoffatGauss2D class.")
class PSF2DAstropy(Fittable2DModel):
    n_inputs = 2
    n_outputs = 1
    # inputs = ('x', 'y',)
    # outputs = ('z',)

    amplitude_moffat = Parameter('amplitude_moffat', default=1)
    x_mean = Parameter('x_mean', default=0)
    y_mean = Parameter('y_mean', default=0)
    gamma = Parameter('gamma', default=3)
    alpha = Parameter('alpha', default=3)
    eta_gauss = Parameter('eta_gauss', default=0.)
    stddev = Parameter('stddev', default=1)
    saturation = Parameter('saturation', default=1)

    param_titles = ["A", "x", "y", r"\gamma", r"\alpha", r"\eta", r"\sigma", "saturation"]

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
        rr_gg = rr / (gamma * gamma)
        a = amplitude * ((1 + rr_gg) ** (-alpha) + eta_gauss * np.exp(-(rr / (2. * stddev * stddev))))
        return np.clip(a, 0, saturation)

    @staticmethod
    def normalisation(amplitude, gamma, alpha, eta_gauss, stddev):
        return amplitude * ((np.pi * gamma * gamma) / (alpha - 1) + eta_gauss * 2 * np.pi * stddev * stddev)

    @staticmethod
    def fit_deriv(x, y, amplitude, x_mean, y_mean, gamma, alpha, eta_gauss, stddev, saturation):
        rr = ((x - x_mean) ** 2 + (y - y_mean) ** 2)
        rr_gg = rr / (gamma * gamma)
        gauss_norm = np.exp(-(rr / (2. * stddev * stddev)))
        d_eta_gauss = amplitude * gauss_norm
        moffat_norm = (1 + rr_gg) ** (-alpha)
        d_amplitude = moffat_norm + eta_gauss * gauss_norm
        d_x_mean = amplitude * eta_gauss * (x - x_mean) / (stddev * stddev) * gauss_norm \
                   - amplitude * alpha * moffat_norm * (-2 * x + 2 * x_mean) / (gamma ** 2 * (1 + rr_gg))
        d_y_mean = amplitude * eta_gauss * (y - y_mean) / (stddev * stddev) * gauss_norm \
                   - amplitude * alpha * moffat_norm * (-2 * y + 2 * y_mean) / (gamma ** 2 * (1 + rr_gg))
        d_stddev = amplitude * eta_gauss * rr / (stddev ** 3) * gauss_norm
        d_alpha = - amplitude * moffat_norm * np.log(1 + rr_gg)
        d_gamma = 2 * amplitude * alpha * moffat_norm * (rr_gg / (gamma * (1 + rr_gg)))
        d_saturation = saturation * np.zeros_like(x)
        return [d_amplitude, d_x_mean, d_y_mean, d_gamma, d_alpha, d_eta_gauss, d_stddev, d_saturation]

    def interpolation(self, x_array, y_array):
        """

        Parameters
        ----------
        x_array: array_like
            The x array to interpolate the model.
        y_array: array_like
            The y array to interpolate the model.

        Returns
        -------
        interp: callable
            Function corresponding to the interpolated model on the (x_array,y_array) array.

        Examples
        --------
        >>> x = np.arange(0, 60, 1)
        >>> y = np.arange(0, 30, 1)
        >>> p = [2,30,15,2,2,1,2,10]
        >>> psf = PSF2DAstropy(*p)
        >>> interp = psf.interpolation(x, y)

        ..  doctest::
            :hide:

            >>> assert np.isclose(interp(p[1], p[2]), psf.evaluate(p[1], p[2], *p))

        """
        params = [getattr(self, p).value for p in self.param_names]
        xx, yy = np.meshgrid(x_array, y_array)
        return interp2d(x_array, y_array, self.evaluate(xx, yy, *params), fill_value=0, bounds_error=False)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

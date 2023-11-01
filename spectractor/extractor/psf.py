import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import RegularGridInterpolator

from spectractor.tools import plot_image_simple
from spectractor import parameters
from spectractor.config import set_logger
from spectractor.fit.fitter import FitWorkspace, FitParameters, run_minimisation

from numba import njit


def evaluate_moffat1d_normalisation(gamma, alpha):
    r"""Compute 1D Moffat normalisation.

    .. math ::

        A = \frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}
        \quad\text{with}, \alpha > 1/2

    Note that this function is defined only for :math:`\alpha > 1/2`.

    Parameters
    ----------
    gamma: float
        Width  :math:`\gamma` of the function.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.

    Returns
    -------
    norm: float
        1D Moffat normalisation.

    Examples
    --------
    >>> print(f"{evaluate_moffat1d_normalisation(5, 2):.6f}")
    0.127324
    """
    return special.gamma(alpha) / (gamma * np.sqrt(np.pi) * special.gamma(alpha - 0.5))


def evaluate_moffat1d_normalisation_dalpha(norm, alpha):
    r"""Compute 1D Moffat normalisation.

    .. math ::

        A = \frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}
        \quad\text{with}, \alpha > 1/2

    Note that this function is defined only for :math:`\alpha > 1/2`.

    Parameters
    ----------
    norm: float
        1D Moffat normalisation.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.

    Returns
    -------
    dalpha: float
        1D Moffat normalisation derivatives with respect to alpha.

    Examples
    --------
    >>> print(f"{evaluate_moffat1d_normalisation_dalpha(5, 2):.6f}")
    1.931472
    """
    return norm * (special.digamma(alpha) - special.digamma(alpha - 0.5))


@njit(["float32[:](int64[:], float32, float32, float32, float32, float32)",
       "float32[:](float32[:], float32, float32, float32, float32, float32)"], fastmath=True, cache=True)
def evaluate_moffat1d(y, amplitude, y_c, gamma, alpha, norm):  # pragma: no cover
    r"""Compute a 1D Moffat function, whose integral is not normalised to unity.

    .. math ::

        f(y) \propto \frac{A}{\left[ 1 +\left(\frac{y-y_c}{\gamma}\right)^2 \right]^\alpha}
        \quad\text{with}, \alpha > 1/2

    Note that this function is defined only for :math:`\alpha > 1/2`. The normalisation factor
    :math:`\frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}` is not included as special functions
    are not supported by numba library.

    Parameters
    ----------
    y: array_like
        1D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    y_c: float
        Center  :math:`y_c` of the function.
    gamma: float
        Width  :math:`\gamma` of the function.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.
    norm: float
        Normalisation :math:`\frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}`.

    Returns
    -------
    output: array_like
        1D array of the function evaluated on the y pixel array.

    Examples
    --------

    >>> Ny = 50
    >>> y = np.arange(Ny)
    >>> amplitude = 10
    >>> alpha = 2
    >>> gamma = 5
    >>> norm = evaluate_moffat1d_normalisation(gamma, alpha)
    >>> a = evaluate_moffat1d(y, amplitude=amplitude, y_c=Ny/2, gamma=gamma, alpha=alpha, norm=norm)
    >>> print(f"{np.sum(a):.6f}")
    9.967563

    .. doctest::
        :hide:

        >>> assert np.isclose(np.argmax(a), Ny/2, atol=0.5)
        >>> assert np.isclose(np.argmax(a), Ny/2, atol=0.5)

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from spectractor.extractor.psf import *
        Ny = 50
        y = np.arange(Ny)
        amplitude = 10
        norm = evaluate_moffat1d_normalisation(gamma, alpha)
        a = evaluate_moffat1d(y, amplitude=amplitude, y_c=Ny/2, gamma=5, alpha=2, norm=norm)
        plt.plot(a)
        plt.grid()
        plt.xlabel("y")
        plt.ylabel("Moffat")
        plt.show()

    """
    rr = (y - y_c) * (y - y_c)
    rr_gg = rr / (gamma * gamma)
    a = (1 + rr_gg) ** -alpha
    a *= (amplitude * norm)
    return a


@njit(["float32[:,:](int64[:], float32, float32, float32, float32, float32, float32, boolean[:])"], fastmath=True, cache=True)
def evaluate_moffat1d_jacobian(y, amplitude, y_c, gamma, alpha, norm, dnormda, fixed):  # pragma: no cover
    r"""Compute a 1D Moffat Jacobian, whose integral is normalised to unity.

    .. math ::

        f(y) \propto \frac{A}{\left[ 1 +\left(\frac{y-y_c}{\gamma}\right)^2 \right]^\alpha}\times \frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}
        \quad\text{with}, \alpha > 1/2

    Note that this function is defined only for :math:`\alpha > 1/2`.

    Parameters
    ----------
    y: array_like
        1D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    y_c: float
        Center  :math:`y_c` of the function.
    gamma: float
        Width  :math:`\gamma` of the function.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.
    norm: float
        Normalisation :math:`\frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}`.
    dnormda: float
        Derivatives of the normalisation with respect to alpha.
    fixed: array_like
        Array of booleans, with True values for fixed parameters.

    Returns
    -------
    J: array_like
        2D array of the model Jacobian.

    Examples
    --------

    >>> Ny = 50
    >>> y = np.arange(Ny)
    >>> amplitude = 10
    >>> alpha = 2
    >>> gamma = 5
    >>> norm = evaluate_moffat1d_normalisation(gamma, alpha)
    >>> dnormda = evaluate_moffat1d_normalisation_dalpha(norm, alpha)
    >>> a = evaluate_moffat1d(y, amplitude=amplitude, y_c=Ny/2, gamma=gamma, alpha=alpha, norm=norm)
    >>> J = evaluate_moffat1d_jacobian(y, amplitude=amplitude, y_c=Ny/2, gamma=gamma, alpha=alpha, norm=norm, dnormda=dnormda, fixed=np.array([False, False, True, False]))
    >>> J.shape
    (5, 50)
    >>> J.dtype
    dtype('float32')
    >>> np.allclose(J[2], 0)
    True

    .. doctest::
        :hide:

        >>> assert np.allclose(J[0], a.ravel()/amplitude)

    """
    yc = y - y_c
    rr = yc * yc
    rr_gg = rr / (gamma * gamma)
    inv = 1 / (1 + rr_gg)
    psf = inv ** alpha
    dpsf = alpha * inv * psf
    A = norm * amplitude
    J = np.zeros((5, y.size), dtype=np.float32)
    if not fixed[0]:
        J[0] = norm * psf  # amplitude
    # fixed x_c so J[1] = 0
    if not fixed[2]:
        J[2] = (2 * A / (gamma * gamma)) * yc * dpsf  # y_c
    if not fixed[3]:
        J[3] = (2 * A / gamma) * rr_gg * dpsf - (A / gamma) * psf  # gamma
    if not fixed[4]:
        J[4] = - A * psf * np.log(1 + rr_gg) + amplitude * psf * dnormda  # alpha
    return J


@njit(["float32[:](int64[:], float32, float32, float32, float32, float32, float32, float32)",
       "float32[:](float32[:], float32, float32, float32, float32, float32, float32, float32)"], fastmath=True, cache=True)
def evaluate_moffatgauss1d(y, amplitude, y_c, gamma, alpha, eta_gauss, sigma, norm_moffat):  # pragma: no cover
    r"""Compute a 1D Moffat-Gaussian function, whose integral is normalised to unity.

    .. math ::

        f(y) \propto A \left\lbrace
        \frac{1}{\left[ 1 +\left(\frac{y-y_c}{\gamma}\right)^2 \right]^\alpha}+ \eta e^{-(y-y_c)^2/(2\sigma^2)}\right\rbrace
        \quad\text{ and } \quad \eta < 0, \alpha > 1/2

    Note that this function is defined only for :math:`\alpha > 1/2`. The normalisation factor for the Moffat+Gauss
    :math:`\frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)} + \eta \sqrt{2\pi} \sigma` is not included as special functions
    are not supproted by the numba library.

    Parameters
    ----------
    y: array_like
        1D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    y_c: float
        Center  :math:`y_c` of the function.
    gamma: float
        Width  :math:`\gamma` of the Moffat function.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.
    eta_gauss: float
        Relative negative amplitude of the Gaussian function.
    sigma: float
        Width :math:`\sigma` of the Gaussian function.
    norm_moffat: float
        Normalisation :math:`\frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}`.

    Returns
    -------
    output: array_like
        1D array of the function evaluated on the y pixel array.

    Examples
    --------

    >>> Ny = 50
    >>> y = np.arange(Ny)
    >>> amplitude = 10
    >>> gamma = 5
    >>> alpha = 2
    >>> eta_gauss = -0.1
    >>> sigma = 1
    >>> norm = evaluate_moffat1d_normalisation(gamma, alpha)
    >>> a = evaluate_moffatgauss1d(y, amplitude=amplitude, y_c=Ny/2, gamma=gamma, alpha=alpha, eta_gauss=eta_gauss, sigma=sigma, norm_moffat=norm)
    >>> print(f"{np.sum(a):.6f}")
    9.966492
    >>> a.dtype
    dtype('float32')

    .. doctest::
        :hide:

        >>> assert np.isclose(np.sum(a), amplitude, atol=0.5)
        >>> assert np.isclose(np.argmax(a), Ny/2, atol=0.5)

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from spectractor.extractor.psf import *
        Ny = 50
        y = np.arange(Ny)
        amplitude = 10
        norm = evaluate_moffat1d_normalisation(gamma, alpha)
        a = evaluate_moffatgauss1d(y, amplitude=amplitude, y_c=Ny/2, gamma=5, alpha=2, eta_gauss=-0.1, sigma=1, norm_moffat=norm)
        plt.plot(a)
        plt.grid()
        plt.xlabel("y")
        plt.ylabel("Moffat+Gauss")
        plt.show()

    """
    yc = y - y_c
    rr = yc * yc
    rr_gg = rr / (gamma * gamma)
    rr_ss = rr / (sigma * sigma)
    norm = (1. / norm_moffat) + eta_gauss * np.sqrt(2 * np.pi) * sigma
    a = (1 + rr_gg) ** -alpha + eta_gauss * np.exp(-(rr_ss / 2))
    a *= (amplitude / norm)
    return a


@njit(["float32[:,:](int64[:], float32, float32, float32, float32, float32, float32, float32, float32, boolean[:])"], fastmath=True, cache=True)
def evaluate_moffatgauss1d_jacobian(y, amplitude, y_c, gamma, alpha, eta_gauss, sigma, norm_moffat, dnormda, fixed):  # pragma: no cover
    r"""Compute a 1D Moffat-Gaussian Jacobian, whose integral is normalised to unity.

    .. math ::

        f(y) \propto A \left\lbrace
        \frac{1}{\left[ 1 +\left(\frac{y-y_c}{\gamma}\right)^2 \right]^\alpha} \times \frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}
         - \eta e^{-(y-y_c)^2/(2\sigma^2)}\right\rbrace
        \quad\text{ and } \quad \eta < 0, \alpha > 1/2

    Note that this function is defined only for :math:`\alpha > 1/2`. The normalisation factor for the Moffat
    :math:`\frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)} + \eta \sqrt{2\pi} \sigma` is not included as special functions
    are not supproted by the numba library.

    Parameters
    ----------
    y: array_like
        1D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    y_c: float
        Center  :math:`y_c` of the function.
    gamma: float
        Width  :math:`\gamma` of the Moffat function.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.
    eta_gauss: float
        Relative negative amplitude of the Gaussian function.
    sigma: float
        Width :math:`\sigma` of the Gaussian function.
    norm_moffat: float
        Normalisation :math:`\frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}`.
    dnormda: float
        Derivatives of the normalisation with respect to alpha.
    fixed: array_like
        Array of booleans, with True values for fixed parameters.

    Returns
    -------
    J: array_like
        2D array of the model Jacobian.

    Examples
    --------

    >>> Ny = 50
    >>> y = np.arange(Ny)
    >>> amplitude = 10
    >>> gamma = 5
    >>> alpha = 2
    >>> eta_gauss = -0.1
    >>> sigma = 1
    >>> norm = evaluate_moffat1d_normalisation(gamma, alpha)
    >>> dnormda = evaluate_moffat1d_normalisation_dalpha(norm, alpha)
    >>> a = evaluate_moffatgauss1d(y, amplitude=amplitude, y_c=Ny/2, gamma=gamma, alpha=alpha,
    ... eta_gauss=eta_gauss, sigma=sigma, norm_moffat=norm)
    >>> J = evaluate_moffatgauss1d_jacobian(y, amplitude=amplitude, y_c=Ny/2, gamma=gamma, alpha=alpha,
    ... eta_gauss=eta_gauss, sigma=sigma, norm_moffat=norm, dnormda=dnormda, fixed=np.array([False, False, True, False, False]))
    >>> J.shape
    (7, 50)
    >>> J.dtype
    dtype('float32')
    >>> np.allclose(J[2], 0)
    True

    .. doctest::
        :hide:

        >>> assert np.allclose(J[0], a.ravel()/amplitude)

    """
    yc = y - y_c
    rr = yc * yc
    rr_gg = rr / (gamma * gamma)
    rr_ss = rr / (sigma * sigma)
    inv_moffat = 1 / (1 + rr_gg)
    psf_moffat = inv_moffat ** alpha
    dpsf_moffat = alpha * inv_moffat * psf_moffat
    psf_gauss = np.exp(-(rr_ss / 2))
    psf = psf_moffat + eta_gauss * psf_gauss
    norm = amplitude / ((1. / norm_moffat) + eta_gauss * np.sqrt(2 * np.pi) * sigma)
    J = np.zeros((7, y.size), dtype=np.float32)
    if not fixed[0]:
        J[0] = (norm / amplitude) * psf  # amplitude
    # fixed x_c so J[1] = 0
    if not fixed[2]:
        J[2] = (norm / (sigma * sigma)) * yc * eta_gauss * psf_gauss + (2 * norm / (gamma * gamma)) * yc * dpsf_moffat  # y_c
    if not fixed[3]:
        J[3] = (2 * norm / gamma) * rr_gg * dpsf_moffat - (norm * norm / (amplitude * norm_moffat * gamma)) * psf   # gamma
    if not fixed[4]:
        J[4] = -norm * psf_moffat * np.log(1 + rr_gg) + psf * (norm * norm / amplitude) * (dnormda / (norm_moffat * norm_moffat))   # alpha
    if not fixed[5]:
        J[5] = norm * psf_gauss - (np.sqrt(2 * np.pi) * sigma * norm * norm / amplitude) * psf  # eta
    if not fixed[6]:
        J[6] = (-eta_gauss * np.sqrt(2*np.pi) * norm * norm / amplitude) * psf + (eta_gauss * norm / sigma) * rr_ss * psf_gauss  # sigma
    return J


@njit(["float32[:,:](int64[:,:], int64[:,:], float32, float32, float32, float32, float32)",
       "float64[:,:](float64[:,:], float64[:,:], float32, float32, float32, float32, float32)"], fastmath=True, cache=True)
def evaluate_moffat2d(x, y, amplitude, x_c, y_c, gamma, alpha):  # pragma: no cover
    r"""Compute a 2D Moffat function, whose integral is normalised to unity.

    .. math ::

        f(x, y) = \frac{A (\alpha - 1)}{\pi \gamma^2} \frac{1}{
        \left[ 1 +\frac{\left(x-x_c\right)^2+\left(y-y_c\right)^2}{\gamma^2} \right]^\alpha}
        \quad\text{with}\quad
        \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y) \mathrm{d}x \mathrm{d}y = A

    Note that this function is defined only for :math:`\alpha > 1`.

    Note that the normalisation of a 2D Moffat function is analytical so it is not expected that
    the sum of the output array is equal to :math:`A`, but lower.

    Parameters
    ----------
    x: array_like
        2D array of pixels :math:`x`, regularly spaced.
    y: array_like
        2D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    x_c: float
        X axis center  :math:`x_c` of the function.
    y_c: float
        Y axis center  :math:`y_c` of the function.
    gamma: float
        Width  :math:`\gamma` of the function.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.

    Returns
    -------
    output: array_like
        2D array of the function evaluated on the y pixel array.

    Examples
    --------

    >>> Nx = 50
    >>> Ny = 50
    >>> yy, xx = np.mgrid[:Ny, :Nx]
    >>> amplitude = 10
    >>> a = evaluate_moffat2d(xx, yy, amplitude=amplitude, x_c=Nx/2, y_c=Ny/2, gamma=5, alpha=2)
    >>> print(f"{np.sum(a):.4f}")
    9.6831
    >>> a.dtype
    dtype('float32')

    .. doctest::
        :hide:

        >>> assert not np.isclose(np.sum(a), amplitude)

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from spectractor.extractor.psf import *
        Nx = 50
        Ny = 50
        yy, xx = np.mgrid[:Nx, :Ny]
        amplitude = 10
        a = evaluate_moffat2d(xx, yy, amplitude=amplitude, y_c=Ny/2, x_c=Nx/2, gamma=5, alpha=2)
        im = plt.pcolor(xx, yy, a)
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(im, label="Moffat 2D")
        plt.show()

    """
    xc = x - x_c
    yc = y - y_c
    rr_gg = (xc * xc + yc * yc) / (gamma * gamma)
    a = (1 + rr_gg) ** -alpha
    norm = (np.pi * gamma * gamma) / (alpha - 1)
    a *= amplitude / norm
    return a


@njit(["float32[:,:](int64[:,:], int64[:,:], float32, float32, float32, float32, float32, boolean[:])"], fastmath=True, cache=True)
def evaluate_moffat2d_jacobian(x, y, amplitude, x_c, y_c, gamma, alpha, fixed):  # pragma: no cover
    r"""Compute a 2D Moffat Jacobian, whose integral is normalised to unity.

    Parameters
    ----------
    x: array_like
        2D array of pixels :math:`x`, regularly spaced.
    y: array_like
        2D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    x_c: float
        X axis center  :math:`x_c` of the function.
    y_c: float
        Y axis center  :math:`y_c` of the function.
    gamma: float
        Width  :math:`\gamma` of the function.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.
    fixed: array_like
        Array of booleans, with True values for fixed parameters.

    Returns
    -------
    J: array_like
        2D array of the model Jacobian.

    Examples
    --------

    >>> Nx = 50
    >>> Ny = 50
    >>> yy, xx = np.mgrid[:Ny, :Nx]
    >>> amplitude = 10
    >>> a = evaluate_moffat2d(xx, yy, amplitude=amplitude, x_c=Nx/2, y_c=Ny/2, gamma=5, alpha=2)
    >>> J = evaluate_moffat2d_jacobian(xx, yy, amplitude=amplitude, x_c=Nx/2, y_c=Ny/2, gamma=5, alpha=2, fixed=np.array([False, False, True, False, False]))
    >>> J.shape
    (5, 2500)
    >>> J.dtype
    dtype('float32')
    >>> np.allclose(J[2], 0)
    True

    .. doctest::
        :hide:

        >>> assert np.allclose(J[0], a.ravel()/amplitude)

    """
    xc = (x - x_c).ravel()
    yc = (y - y_c).ravel()
    rr_gg = (xc * xc + yc * yc) / (gamma * gamma)
    inv = 1 / (1 + rr_gg)
    psf = (1 + rr_gg) ** -alpha
    dpsf = alpha * inv * psf
    norm = np.float32(amplitude * (alpha-1) / (np.pi * gamma * gamma))
    J = np.zeros((5, x.size), dtype=np.float32)
    if not fixed[0]:
        J[0] = (norm / amplitude) * psf  # amplitude
    if not fixed[1]:
        J[1] = (2 * norm / (gamma * gamma)) * xc * dpsf  # x_c
    if not fixed[2]:
        J[2] = (2 * norm / (gamma * gamma)) * yc * dpsf  # y_c
    if not fixed[3]:
        J[3] = (2 * norm / (gamma)) * rr_gg * dpsf - (2 * norm / gamma) * psf  # gamma
    if not fixed[4]:
        J[4] = (norm / (alpha - 1)) * psf - norm * psf * np.log(1 + rr_gg)  # alpha
    return J


@njit(["float32[:,:](int64[:,:], int64[:,:], float32, float32, float32, float32, float32, float32, float32)"], fastmath=True, cache=True)
def evaluate_moffatgauss2d(x, y, amplitude, x_c, y_c, gamma, alpha, eta_gauss, sigma):  # pragma: no cover
    r"""Compute a 2D Moffat-Gaussian function, whose integral is normalised to unity.

    .. math ::

        f(x, y) = \frac{A}{\frac{\pi \gamma^2}{\alpha-1} + 2 \pi \eta \sigma^2}\left\lbrace \frac{1}{
        \left[ 1 +\frac{\left(x-x_c\right)^2+\left(y-y_c\right)^2}{\gamma^2} \right]^\alpha}
         + \eta e^{-\left[ \left(x-x_c\right)^2+\left(y-y_c\right)^2\right]/(2 \sigma^2)}
        \right\rbrace

    .. math ::
        \quad\text{with}\quad
        \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y) \mathrm{d}x \mathrm{d}y = A
        \quad\text{and} \quad \eta < 0

    Note that this function is defined only for :math:`\alpha > 1`.

    Parameters
    ----------
    x: array_like
        2D array of pixels :math:`x`, regularly spaced.
    y: array_like
        2D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    x_c: float
        X axis center  :math:`x_c` of the function.
    y_c: float
        Y axis center  :math:`y_c` of the function.
    gamma: float
        Width  :math:`\gamma` of the function.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.
    eta_gauss: float
        Relative negative amplitude of the Gaussian function.
    sigma: float
        Width :math:`\sigma` of the Gaussian function.

    Returns
    -------
    output: array_like
        2D array of the function evaluated on the y pixel array.

    Examples
    --------

    >>> Nx = 50
    >>> Ny = 50
    >>> yy, xx = np.mgrid[:Ny, :Nx]
    >>> amplitude = 10
    >>> a = evaluate_moffatgauss2d(xx, yy, amplitude=amplitude, x_c=Nx/2, y_c=Ny/2, gamma=5, alpha=2,
    ... eta_gauss=-0.1, sigma=1)
    >>> print(f"{np.sum(a):.6f}")
    9.680574
    >>> a.dtype
    dtype('float32')

    .. doctest::
        :hide:

        >>> assert not np.isclose(np.sum(a), amplitude)

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from spectractor.extractor.psf import *
        Nx = 50
        Ny = 50
        yy, xx = np.mgrid[:Nx, :Ny]
        amplitude = 10
        a = evaluate_moffatgauss2d(xx, yy, amplitude, Nx/2, Ny/2, gamma=5, alpha=2, eta_gauss=-0.1, sigma=1)
        im = plt.pcolor(xx, yy, a)
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(im, label="Moffat 2D")
        plt.show()

    """
    xc = x - x_c
    yc = y - y_c
    rr = xc * xc + yc * yc
    rr_gg = rr / (gamma * gamma)
    rr_ss = rr / (sigma * sigma)
    a = (1 + rr_gg) ** -alpha + eta_gauss * np.exp(-(rr_ss / 2))
    norm = (np.pi * gamma * gamma) / (alpha - 1) + eta_gauss * 2 * np.pi * sigma * sigma
    a *= amplitude / norm
    return a


@njit(["float32[:](int64[:], float32, float32, float32)",
       "float32[:](float32[:], float32, float32, float32)"], fastmath=True, cache=False)
def evaluate_gauss1d(y, amplitude, y_c, sigma):  # pragma: no cover
    r"""Compute a 1D Gaussian function, whose integral is normalised to unity.

    .. math ::

        f(x, y) = \frac{A}{\sigma \sqrt{2 \pi}\left\lbrace e^{-\left[ \left(x-x_c\right)^2\right]/(2 \sigma^2)}
        \right\rbrace

    .. math ::
        \quad\text{with}\quad
        \int_{-\infty}^{\infty}f(y) \mathrm{d}y = A

    Parameters
    ----------
    y: array_like
        1D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    y_c: float
        X axis center  :math:`y_c` of the function.
    sigma: float
        Width :math:`\sigma` of the Gaussian function.

    Returns
    -------
    output: array_like
        1D array of the function evaluated on the y pixel array.

    Examples
    --------

    >>> Ny = 50
    >>> y = np.arange(Ny)
    >>> amplitude = 10
    >>> sigma = 2
    >>> a = evaluate_gauss1d(y, amplitude=amplitude, y_c=Ny/2, sigma=sigma)
    >>> print(f"{np.sum(a):.6f}")
    10.000000

    .. doctest::
        :hide:

        >>> assert np.isclose(np.argmax(a), Ny/2, atol=0.5)
        >>> assert np.isclose(np.argmax(a), Ny/2, atol=0.5)

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from spectractor.extractor.psf import *
        Ny = 50
        y = np.arange(Ny)
        amplitude = 10
        a = evaluate_gauss1d(y, amplitude=amplitude, y_c=Ny/2, sigma=2)
        plt.plot(a)
        plt.grid()
        plt.xlabel("y")
        plt.ylabel("Gauss")
        plt.show()

    """
    yc = (y - y_c) / sigma
    rr = yc * yc
    a = np.exp(-(rr / 2))
    norm = np.sqrt(2 * np.pi) * sigma
    a *= amplitude / norm
    return a


@njit(["float32[:,:](int64[:], float32, float32, float32, boolean[:])",
       "float32[:,:](float32[:], float32, float32, float32, boolean[:])"], fastmath=True, cache=False)
def evaluate_gauss1d_jacobian(y, amplitude, y_c, sigma, fixed):  # pragma: no cover
    r"""Compute a 1D Gaussian function, whose integral is normalised to unity.

    .. math ::

        f(x, y) = \frac{A}{\sigma \sqrt{2 \pi}\left\lbrace e^{-\left[ \left(x-x_c\right)^2\right]/(2 \sigma^2)}
        \right\rbrace

    .. math ::
        \quad\text{with}\quad
        \int_{-\infty}^{\infty}f(y) \mathrm{d}y = A

    Parameters
    ----------
    y: array_like
        2D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    y_c: float
        X axis center  :math:`y_c` of the function.
    sigma: float
        Width :math:`\sigma` of the Gaussian function.
    fixed: array_like
        Array of booleans, with True values for fixed parameters.

    Returns
    -------
    J: array_like
        2D array of the model Jacobian.

    Examples
    --------

    >>> Ny = 50
    >>> y = np.arange(Ny)
    >>> amplitude = 10
    >>> sigma = 2
    >>> a = evaluate_gauss1d(y, amplitude=amplitude, y_c=Ny/2, sigma=2)
    >>> J = evaluate_gauss1d_jacobian(y, amplitude=amplitude, y_c=Ny/2, sigma=2, fixed=np.array([False, False, True]))
    >>> J.shape
    (4, 50)
    >>> J.dtype
    dtype('float32')
    >>> np.allclose(J[2], 0)
    True

    .. doctest::
        :hide:

        >>> assert np.allclose(J[0], a/amplitude)

    """
    yc = (y - y_c) / sigma
    rr_ss = yc * yc
    psf = np.exp(-(rr_ss / 2))
    norm = amplitude / (np.sqrt(2 * np.pi) * sigma)
    J = np.zeros((4, y.size), dtype=np.float32)
    if not fixed[0]:
        J[0] = (norm / amplitude) * psf  # amplitude
    # x_c is fixed so J[1] = 0
    if not fixed[2]:
        J[2] = (norm / sigma) * yc * psf  # y_c
    if not fixed[3]:
        J[3] = (norm / sigma) * rr_ss * psf - (norm / sigma) * psf  # sigma
    return J


@njit(["float32[:,:](int64[:,:], int64[:,:], float32, float32, float32, float32)"], fastmath=True, cache=True)
def evaluate_gauss2d(x, y, amplitude, x_c, y_c, sigma):  # pragma: no cover
    r"""Compute a 2D Gaussian function, whose integral is normalised to unity.

    .. math ::

        f(x, y) = \frac{A}{2 \pi \sigma^2}\left\lbrace e^{-\left[ \left(x-x_c\right)^2+\left(y-y_c\right)^2\right]/(2 \sigma^2)}
        \right\rbrace

    .. math ::
        \quad\text{with}\quad
        \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y) \mathrm{d}x \mathrm{d}y = A

    Parameters
    ----------
    x: array_like
        2D array of pixels :math:`x`, regularly spaced.
    y: array_like
        2D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    x_c: float
        X axis center  :math:`x_c` of the function.
    y_c: float
        Y axis center  :math:`y_c` of the function.
    sigma: float
        Width :math:`\sigma` of the Gaussian function.

    Returns
    -------
    output: array_like
        2D array of the function evaluated on the y pixel array.

    Examples
    --------

    >>> Nx = 50
    >>> Ny = 50
    >>> yy, xx = np.mgrid[:Ny, :Nx]
    >>> amplitude = 10
    >>> a = evaluate_gauss2d(xx, yy, amplitude=amplitude, x_c=Nx/2, y_c=Ny/2, sigma=1)
    >>> print(f"{np.sum(a):.6f}")
    10.000000
    >>> a.dtype
    dtype('float32')

    .. doctest::
        :hide:

        >>> assert np.isclose(np.sum(a), amplitude)

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from spectractor.extractor.psf import *
        Nx = 50
        Ny = 50
        yy, xx = np.mgrid[:Nx, :Ny]
        amplitude = 10
        a = evaluate_gauss2d(xx, yy, amplitude, Nx/2, Ny/2, sigma=1)
        im = plt.pcolor(xx, yy, a)
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(im, label="Gauss 2D")
        plt.show()

    """
    xc = (x - x_c) / sigma
    yc = (y - y_c) / sigma
    rr = xc * xc + yc * yc
    a = np.exp(-(rr / 2))
    norm = 2 * np.pi * sigma * sigma
    a *= amplitude / norm
    return a


@njit(["float32[:,:](int64[:,:], int64[:,:], float32, float32, float32, float32, boolean[:])"], fastmath=True, cache=True)
def evaluate_gauss2d_jacobian(x, y, amplitude, x_c, y_c, sigma, fixed):  # pragma: no cover
    r"""Compute a 2D Gaussian Jacobian, whose integral is normalised to unity.

    Parameters
    ----------
    x: array_like
        2D array of pixels :math:`x`, regularly spaced.
    y: array_like
        2D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    x_c: float
        X axis center  :math:`x_c` of the function.
    y_c: float
        Y axis center  :math:`y_c` of the function.
    sigma: float
        Width :math:`\sigma` of the Gaussian function.
    fixed: array_like
        Array of booleans, with True values for fixed parameters.

    Returns
    -------
    J: array_like
        2D array of the model Jacobian.

    Examples
    --------

    >>> Nx = 50
    >>> Ny = 50
    >>> yy, xx = np.mgrid[:Ny, :Nx]
    >>> amplitude = 10
    >>> a = evaluate_gauss2d(xx, yy, amplitude=amplitude, x_c=Nx/2, y_c=Ny/2, sigma=1)
    >>> J = evaluate_gauss2d_jacobian(xx, yy, amplitude=amplitude, x_c=Nx/2, y_c=Ny/2, sigma=1, fixed=np.array([False, False, True, False, False]))
    >>> J.shape
    (4, 2500)
    >>> J.dtype
    dtype('float32')
    >>> np.allclose(J[2], 0)
    True

    .. doctest::
        :hide:

        >>> assert np.allclose(J[0], a.ravel()/amplitude)

    """
    xc = (x - x_c).ravel() / sigma
    yc = (y - y_c).ravel() / sigma
    rr_ss = xc * xc + yc * yc
    psf = np.exp(-(rr_ss / 2))
    norm = amplitude / (2 * np.pi * sigma * sigma)
    J = np.zeros((4, x.size), dtype=np.float32)
    if not fixed[0]:
        J[0] = (norm / amplitude) * psf  # amplitude
    if not fixed[1]:
        J[1] = (norm / sigma) * xc * psf  # x_c
    if not fixed[2]:
        J[2] = (norm / sigma) * yc * psf  # y_c
    if not fixed[3]:
        J[3] = (norm / sigma) * rr_ss * psf - (2 * norm / sigma) * psf  # sigma
    return J



@njit(["float32[:,:](int64[:,:], int64[:,:], float32, float32, float32, float32, float32, float32, float32, boolean[:])"], fastmath=True, cache=True)
def evaluate_moffatgauss2d_jacobian(x, y, amplitude, x_c, y_c, gamma, alpha, eta_gauss, sigma, fixed):  # pragma: no cover
    r"""Compute a 2D Moffat Jacobian, whose integral is normalised to unity.

    Parameters
    ----------
    x: array_like
        2D array of pixels :math:`x`, regularly spaced.
    y: array_like
        2D array of pixels :math:`y`, regularly spaced.
    amplitude: float
        Integral :math:`A` of the function.
    x_c: float
        X axis center  :math:`x_c` of the function.
    y_c: float
        Y axis center  :math:`y_c` of the function.
    gamma: float
        Width  :math:`\gamma` of the function.
    alpha: float
        Exponent :math:`\alpha` of the Moffat function.
    eta_gauss: float
        Relative negative amplitude of the Gaussian function.
    sigma: float
        Width :math:`\sigma` of the Gaussian function.
    fixed: array_like
        Array of booleans, with True values for fixed parameters.


    Returns
    -------
    J: array_like
        2D array of the model Jacobian.

    Examples
    --------

    >>> Nx = 50
    >>> Ny = 50
    >>> yy, xx = np.mgrid[:Ny, :Nx]
    >>> amplitude = 10
    >>> a = evaluate_moffatgauss2d(xx, yy, amplitude=amplitude, x_c=Nx/2, y_c=Ny/2, gamma=5, alpha=2,
    ... eta_gauss=-0.1, sigma=1)
    >>> J = evaluate_moffatgauss2d_jacobian(xx, yy, amplitude=amplitude, x_c=Nx/2, y_c=Ny/2, gamma=5, alpha=2,
    ... eta_gauss=-0.1, sigma=1, fixed=np.array([False, False, True, False, False, False, False]))
    >>> J.shape
    (7, 2500)
    >>> J.dtype
    dtype('float32')
    >>> np.allclose(J[2], 0)
    True

    .. doctest::
        :hide:

        >>> assert np.allclose(J[0], a.ravel()/amplitude)

    """
    xc = (x - x_c).ravel()
    yc = (y - y_c).ravel()
    rr = xc * xc + yc * yc
    rr_gg = rr / (gamma * gamma)
    rr_ss = rr / (sigma * sigma)
    inv_moffat = 1 / (1 + rr_gg)
    psf_moffat = inv_moffat ** alpha
    dpsf_moffat = alpha * inv_moffat * psf_moffat
    psf_gauss = np.exp(-(rr_ss / 2))
    psf = psf_moffat + eta_gauss * psf_gauss
    norm = amplitude / ((np.pi * gamma * gamma) / (alpha - 1) + eta_gauss * 2 * np.pi * sigma * sigma)
    J = np.zeros((7, x.size), dtype=np.float32)
    if not fixed[0]:
        J[0] = (norm / amplitude) * psf  # amplitude
    if not fixed[1]:
        J[1] = (norm / (sigma * sigma)) * xc * eta_gauss * psf_gauss + (2 * norm / (gamma * gamma)) * xc * dpsf_moffat  # x_c
    if not fixed[2]:
        J[2] = (norm / (sigma * sigma)) * yc * eta_gauss * psf_gauss + (2 * norm / (gamma * gamma)) * yc * dpsf_moffat  # x_c
    if not fixed[3]:
        J[3] = (-2 * np.pi * gamma * norm * norm / (alpha-1) / amplitude) * psf + (2 * norm / gamma) * rr_gg * dpsf_moffat  # gamma
    if not fixed[4]:
        J[4] = (np.pi * gamma * gamma) * norm * norm / (amplitude * (alpha-1) * (alpha-1)) * psf - norm * psf_moffat * np.log(1 + rr_gg)  # alpha
    if not fixed[5]:
        J[5] = norm * psf_gauss - (2 * np.pi * sigma * sigma * norm * norm / amplitude) * psf
    if not fixed[6]:
        J[6] = (-4 * eta_gauss * np.pi * sigma * norm * norm / amplitude) * psf + (eta_gauss * norm / sigma) * rr_ss * psf_gauss
    return J


class PSF:
    """Generic PSF model class.

    The PSF models must contain at least the "amplitude", "x_c" and "y_c" parameters as the first three parameters
    (in this order) and "saturation" parameter as the last parameter. "amplitude", "x_c" and "y_c"
    stands respectively for the general amplitude of the model, the position along the dispersion axis and the
    transverse position with respect to the dispersion axis (assumed to be the X axis).
    Last "saturation" parameter must be express in the same units as the signal to model and as the "amplitude"
    parameter. The PSF models must be normalized to one in total flux divided by the first parameter (amplitude).
    Then the PSF model integral is equal to the "amplitude" parameter.

    """

    def __init__(self, clip=False):
        """
        Parameters
        ----------
        clip: bool, optional
            If True, PSF evaluation is clipped between 0 and saturation level (slower) (default: False)

        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.values_default = np.array([1, 0, 0, 1])
        self.params = FitParameters(values=self.values_default, labels=["amplitude", "x_c", "y_c", "saturation"],
                                    axis_names=["$A$", r"$x_c$", r"$y_c$", "saturation"])
        self.max_half_width = np.inf
        self.clip = clip

    def evaluate(self, pixels, values=None):  # pragma: no cover
        if values is not None:
            self.params.values = np.asarray(values).astype(float)
        if pixels.ndim == 3 and pixels.shape[0] == 2:
            return np.zeros_like(pixels[0], dtype="float32")
        elif pixels.ndim == 1:
            return np.zeros_like(pixels, dtype="float32")
        else:
            raise ValueError(f"Pixels array must have dimension 1 or shape=(2,Nx,Ny). Here pixels.ndim={pixels.shape}.")

    def jacobian(self, pixels, params, epsilon=None, model_input=None, analytical=True):  # pragma: no cover
        if epsilon is None and not analytical:
            raise ValueError(f"If analytical=False, must give epsilon values for numerical differentiation.")
        if params is not None:
            self.params.values = np.asarray(params).astype(float)
        if pixels.ndim == 3 and pixels.shape[0] == 2:
            return np.zeros((self.params.values.size, pixels[0].size), dtype="float32")
        elif pixels.ndim == 1:
            self.params.fixed[1] = True  # remove x_c
            return np.zeros((self.params.values.size, pixels.size), dtype="float32")
        else:
            raise ValueError(f"Pixels array must have dimension 1 or shape=(2,Nx,Ny). Here pixels.ndim={pixels.shape}.")

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
        >>> yy, xx = np.mgrid[:50, :60]
        >>> data = psf0.evaluate(np.array([xx, yy]), p0)
        >>> data = np.random.poisson(data)
        >>> data_errors = np.sqrt(data+1)

        Fit the data in 2D:

        >>> p = np.array([150000, 19, 31, 4.5, 2.5, -0.1, 3, 400000])
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
            >>> assert np.all(np.isclose(psf.params.values[1:3], p0[1:3], atol=1e-1))

        Fit the data in 1D:

        >>> data1d = data[:,int(p0[1])]
        >>> data1d_err = data_errors[:,int(p0[1])]
        >>> p = np.array([10000, 20, 32, 4, 3, -0.1, 2, 400000])
        >>> psf1d = MoffatGauss(p)
        >>> w = psf1d.fit_psf(data1d, data_errors=data1d_err, bgd_model_func=None)
        >>> w.plot_fit()

        ..  doctest::
            :hide:

            >>> assert w.model is not None
            >>> residuals = (w.data-w.model)/w.err
            >>> assert w.costs[-1] / w.pixels.size < 1.5
            >>> assert np.abs(np.mean(residuals)) < 0.3
            >>> assert np.std(residuals) < 1.5
            >>> assert np.all(np.isclose(w.params.values[2], p0[2], atol=1e-1))

        .. plot::

            import numpy as np
            import matplotlib.pyplot as plt
            from spectractor.extractor.psf import *
            p = np.array([200000, 20, 30, 5, 2, -0.1, 2, 400000])
            psf = MoffatGauss(p)
            yy, xx = np.mgrid[:50, :60]
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
        run_minimisation(w, method="newton", ftol=1 / w.pixels.size, xtol=1e-6, niter=50)
        self.params.values = np.copy(w.params.values)
        return w


class Moffat(PSF):

    def __init__(self, values=None, clip=False):
        PSF.__init__(self, clip=clip)
        self.values_default = np.array([1, 0, 0, 3, 2, 1]).astype(float)
        if values is None:
            values = np.copy(self.values_default)
        labels = ["amplitude", "x_c", "y_c", "gamma", "alpha", "saturation"]
        axis_names = ["$A$", r"$x_c$", r"$y_c$", r"$\gamma$", r"$\alpha$", "saturation"]
        bounds = [(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0.1, np.inf),
                                (1.1, 100), (0, np.inf)]
        self.params = FitParameters(values=values, labels=labels, axis_names=axis_names, bounds=bounds)

    def apply_max_width_to_bounds(self, max_half_width=None):
        if max_half_width is not None:
            self.max_half_width = max_half_width
        self.params.bounds[2] = (0, 2 * self.max_half_width)
        self.params.bounds[3] = (0.1, self.max_half_width)

    def evaluate(self, pixels, values=None):
        r"""Evaluate the Moffat function.

        The function is normalized to have an integral equal to amplitude parameter, with normalisation factor:

        .. math::

            f(y) \propto \frac{A \Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)},
            \quad \int_{y_{\text{min}}}^{y_{\text{max}}} f(y) \mathrm{d}y = A

        Parameters
        ----------
        pixels: list
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        values: array_like
            The parameter array. If None, the array used to instanciate the class is taken.
            If given, the class instance parameter array is updated.

        Returns
        -------
        output: array_like
            The PSF function evaluated.

        Examples
        --------
        >>> p = [2,20,30,4,2,10]
        >>> psf = Moffat(p, clip=True)
        >>> yy, xx = np.mgrid[:50, :60]
        >>> output = psf.evaluate(pixels=np.array([xx, yy]), values=p)

        ..  doctest::
            :hide:

            >>> assert np.sum(output) > 0

        >>> p = [2,20,30,4,2,10]
        >>> psf = Moffat(p, clip=True)
        >>> xx = np.arange(0, 50, 1)
        >>> output = psf.evaluate(pixels=xx, values=p)

        ..  doctest::
            :hide:

            >>> assert np.sum(output) > 0

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np
            from spectractor.extractor.psf import Moffat
            p = [2,20,30,4,2,10]
            psf = Moffat(p)
            yy, xx = np.mgrid[:50, :60]
            out = psf.evaluate(pixels=np.array([xx, yy]), values=p)
            fig = plt.figure(figsize=(5,5))
            plt.imshow(out, origin="lower")
            plt.xlabel("X [pixels]")
            plt.ylabel("Y [pixels]")
            plt.show()

        """
        if values is not None:
            self.params.values = np.asarray(values).astype(float)
        amplitude, x_c, y_c, gamma, alpha, saturation = self.params.values
        if pixels.ndim == 3 and pixels.shape[0] == 2:
            x, y = pixels  # .astype(np.float32)  # float32 to increase rapidity
            out = evaluate_moffat2d(x, y, amplitude, x_c, y_c, gamma, alpha)
            if self.clip:
                out = np.clip(out, 0, saturation)
            return out
        elif pixels.ndim == 1:
            y = np.array(pixels)
            if alpha > 0.5:
                norm = evaluate_moffat1d_normalisation(gamma, alpha)
                out = evaluate_moffat1d(y, amplitude, y_c, gamma, alpha, norm=norm)
            else:
                out = np.zeros_like(y)
            if self.clip:
                out = np.clip(out, 0, saturation)
            return out
        else:  # pragma: no cover
            raise ValueError(f"Pixels array must have dimension 1 or shape=(2,Nx,Ny). Here pixels.ndim={pixels.shape}.")

    def jacobian(self, pixels, params, epsilon=None, model_input=None, analytical=True):
        r"""Evaluate the PSF Moffat Jacobian.

        The function is normalized to have an integral equal to amplitude parameter, with normalisation factor:

        Parameters
        ----------
        pixels: array_like
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        params: array_like
            The parameter array. If None, the array used to instanciate the class is taken.
            If given, the class instance parameter array is updated.
        epsilon: array_like, optional
            The array of small steps to compute the partial derivatives of the model if analytical=False (default: None).
        model_input: array_like, optional
            A model input as a list with (x, model, model_err) to avoid an additional call to simulate() if analytical=False (default: None).
        analytical: bool, optional
            If True, use analytical derivatives to compute Jacobian operator. Otherwise use numerical differenciations
            with steps given by epsilon argument (default: True).

        Returns
        -------
        jacobian: array_like
            The PSF Jacobian.

        Examples
        --------
        >>> p = [2,20,30,4,2,10]
        >>> epsilon = [0.01] * len(p)
        >>> psf = Moffat(p, clip=True)
        >>> psf.params.fixed = [True, True, False, False, False, True]  # fix amplitude, x_c, saturation

        2D case

        >>> yy, xx = np.mgrid[:50, :60]
        >>> J_ana = psf.jacobian(pixels=np.array([xx, yy]), params=p, epsilon=epsilon, analytical=True)
        >>> J_num = psf.jacobian(pixels=np.array([xx, yy]), params=p, epsilon=epsilon, analytical=False)
        >>> np.allclose(J_num, J_ana, rtol=1e-4, atol=1e-4)
        True
        >>> np.allclose(J_ana[0:2], 0)
        True

        ..  doctest::
            :hide:

            >>> assert J_ana.shape == (len(p), xx.size)
            >>> assert J_num.shape == (len(p), xx.size)

        1D case

        >>> y = np.mgrid[:50]
        >>> J_ana = psf.jacobian(pixels=y, params=p, epsilon=epsilon, analytical=True)
        >>> J_num = psf.jacobian(pixels=y, params=p, epsilon=epsilon, analytical=False)
        >>> np.allclose(J_num, J_ana, rtol=1e-3, atol=1e-3)
        True
        >>> np.allclose(J_ana[0:2], 0)
        True

        ..  doctest::
            :hide:

            >>> assert J_ana.shape == (len(p), y.size)
            >>> assert J_num.shape == (len(p), y.size)
        """
        if epsilon is None and not analytical:
            raise ValueError(f"If analytical=False, must give epsilon values for numerical differentiation.")
        amplitude, x_c, y_c, gamma, alpha, saturation = self.params.values.astype(float)
        J = super().jacobian(pixels, params, epsilon=epsilon, model_input=model_input, analytical=analytical)
        if not analytical:
            if model_input is None:
                model = self.evaluate(pixels, values=params)
            else:
                x, model, model_err = model_input
            for ip, p in enumerate(params):
                if self.params.fixed[ip]:
                    continue
                tmp_p = np.copy(params).astype(float)
                if tmp_p[ip] + epsilon[ip] < self.params.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.params.bounds[ip][1]:
                    epsilon[ip] = - epsilon[ip]
                tmp_p[ip] += epsilon[ip]
                tmp_model = self.evaluate(pixels, values=tmp_p)
                J[ip] = (tmp_model.ravel() - model.ravel()) / epsilon[ip]
        else:
            fixed = np.array(self.params.fixed)
            if pixels.ndim == 1:
                norm = evaluate_moffat1d_normalisation(gamma, alpha)
                dnormda = evaluate_moffat1d_normalisation_dalpha(norm, alpha)
                J[:-1] = evaluate_moffat1d_jacobian(pixels, amplitude, y_c, gamma, alpha, norm, dnormda, fixed=fixed)  # [:-1] assumes saturation is fixed
            else:
                xx, yy = pixels
                if amplitude > 0:
                    J[:-1] = evaluate_moffat2d_jacobian(xx, yy, amplitude, x_c, y_c, gamma, alpha, fixed=fixed)  # [:-1] assumes saturation is fixed
        return J


class Gauss(PSF):

    def __init__(self, values=None, clip=False):
        PSF.__init__(self, clip=clip)
        self.values_default = np.array([1, 0, 0, 1, 1]).astype(float)
        if values is None:
            values = np.copy(self.values_default)
        labels = ["amplitude", "x_c", "y_c", "sigma", "saturation"]
        axis_names = ["$A$", r"$x_c$", r"$y_c$", r"$\sigma$", "saturation"]
        bounds = [(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (1, np.inf), (0, np.inf)]
        self.params = FitParameters(values=values, labels=labels, axis_names=axis_names, bounds=bounds)

    def apply_max_width_to_bounds(self, max_half_width=None):
        if max_half_width is not None:
            self.max_half_width = max_half_width
        self.params.bounds[2] = (0, 2 * self.max_half_width)
        self.params.bounds[3] = (1, self.max_half_width)

    def evaluate(self, pixels, values=None):
        r"""Evaluate the Gauss function.

        The function is normalized to have an integral equal to amplitude parameter, with normalisation factor:

        Parameters
        ----------
        pixels: list
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        values: array_like
            The parameter array. If None, the array used to instanciate the class is taken.
            If given, the class instance parameter array is updated.

        Returns
        -------
        output: array_like
            The PSF function evaluated.

        Examples
        --------
        >>> p = [2,20,30,2,10]
        >>> psf = Gauss(p, clip=True)
        >>> yy, xx = np.mgrid[:50, :60]
        >>> out = psf.evaluate(pixels=np.array([xx, yy]), values=p)

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np
            from spectractor.extractor.psf import Moffat
            p = [2,20,30,2,10]
            psf = Gauss(p)
            yy, xx = np.mgrid[:50, :60]
            out = psf.evaluate(pixels=np.array([xx, yy]), values=p)
            fig = plt.figure(figsize=(5,5))
            plt.imshow(out, origin="lower")
            plt.xlabel("X [pixels]")
            plt.ylabel("Y [pixels]")
            plt.show()

        """
        if values is not None:
            self.params.values = np.asarray(values).astype(float)
        amplitude, x_c, y_c, sigma, saturation = self.params.values
        if pixels.ndim == 3 and pixels.shape[0] == 2:
            x, y = pixels  # .astype(np.float32)  # float32 to increase rapidity
            out = evaluate_gauss2d(x, y, amplitude, x_c, y_c, sigma)
            if self.clip:
                out = np.clip(out, 0, saturation)
            return out
        elif pixels.ndim == 1:
            y = np.array(pixels)
            if amplitude > 0 and sigma > 0:
                out = evaluate_gauss1d(y, amplitude, y_c, sigma)
            else:
                out = np.zeros_like(y)
            if self.clip:
                out = np.clip(out, 0, saturation)
            return out
        else:  # pragma: no cover
            raise ValueError(f"Pixels array must have dimension 1 or shape=(2,Nx,Ny). Here pixels.ndim={pixels.shape}.")

    def jacobian(self, pixels, params, epsilon=None, model_input=None, analytical=True):
        r"""Evaluate the PSF Gauss Jacobian.

        The function is normalized to have an integral equal to amplitude parameter, with normalisation factor:

        Parameters
        ----------
        pixels: array_like
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        params: array_like
            The parameter array. If None, the array used to instanciate the class is taken.
            If given, the class instance parameter array is updated.
        epsilon: array_like, optional
            The array of small steps to compute the partial derivatives of the model if analytical=False (default: None).
        model_input: array_like, optional
            A model input as a list with (x, model, model_err) to avoid an additional call to simulate() if analytical=False (default: None).
        analytical: bool, optional
            If True, use analytical derivatives to compute Jacobian operator. Otherwise use numerical differenciations
            with steps given by epsilon argument (default: True).

        Returns
        -------
        jacobian: array_like
            The PSF Jacobian.

        Examples
        --------
        >>> p = [2,20,30,2,10]
        >>> epsilon = [0.001] * len(p)
        >>> psf = Gauss(p, clip=True)
        >>> psf.params.fixed = [True, True, False, False, True]  # fix amplitude, x_c, saturation

        2D case

        >>> yy, xx = np.mgrid[:50, :60]
        >>> J_ana = psf.jacobian(pixels=np.array([xx, yy]), params=p, epsilon=epsilon, analytical=True)
        >>> J_num = psf.jacobian(pixels=np.array([xx, yy]), params=p, epsilon=epsilon, analytical=False)
        >>> np.allclose(J_num, J_ana, rtol=1e-2, atol=1e-4)
        True
        >>> np.allclose(J_ana[0:2], 0)
        True

        ..  doctest::
            :hide:

            >>> assert J_ana.shape == (len(p), xx.size)
            >>> assert J_num.shape == (len(p), xx.size)

        1D case

        >>> y = np.mgrid[:50]
        >>> J_ana = psf.jacobian(pixels=y, params=p, epsilon=epsilon, analytical=True)
        >>> J_num = psf.jacobian(pixels=y, params=p, epsilon=epsilon, analytical=False)
        >>> np.allclose(J_num, J_ana, rtol=1e-2, atol=1e-4)
        True
        >>> np.allclose(J_ana[0:2], 0)
        True

        ..  doctest::
            :hide:

            >>> assert J_ana.shape == (len(p), y.size)
            >>> assert J_num.shape == (len(p), y.size)
        """
        if epsilon is None and not analytical:
            raise ValueError(f"If analytical=False, must give epsilon values for numerical differentiation.")
        amplitude, x_c, y_c, sigma, saturation = self.params.values.astype(float)
        J = super().jacobian(pixels, params, epsilon=epsilon, model_input=model_input, analytical=analytical)
        if not analytical:
            if model_input is None:
                model = self.evaluate(pixels, values=params)
            else:
                x, model, model_err = model_input
            for ip, p in enumerate(params):
                if self.params.fixed[ip]:
                    continue
                tmp_p = np.copy(params).astype(float)
                if tmp_p[ip] + epsilon[ip] < self.params.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.params.bounds[ip][1]:
                    epsilon[ip] = - epsilon[ip]
                tmp_p[ip] += epsilon[ip]
                tmp_model = self.evaluate(pixels, values=tmp_p)
                J[ip] = (tmp_model.ravel() - model.ravel()) / epsilon[ip]
        else:
            fixed = np.array(self.params.fixed)
            if pixels.ndim == 1:
                J[:-1] = evaluate_gauss1d_jacobian(pixels, amplitude, y_c, sigma, fixed=fixed)  # [:-1] assumes saturation is fixed
            else:
                xx, yy = pixels
                J[:-1] = evaluate_gauss2d_jacobian(xx, yy, amplitude, x_c, y_c, sigma, fixed=fixed)  # [:-1] assume saturation is fixed
        return J


class MoffatGauss(PSF):

    def __init__(self, values=None, clip=False):
        PSF.__init__(self, clip=clip)
        self.values_default = np.array([1, 0, 0, 3, 2, -0.5, 1, 1]).astype(float)
        if values is None:
            values = np.copy(self.values_default)
        labels = ["amplitude", "x_c", "y_c", "gamma", "alpha", "eta_gauss", "stddev", "saturation"]
        axis_names = ["$A$", r"$x_c$", r"$y_c$", r"$\gamma$", r"$\alpha$", r"$\eta$", r"$\sigma$", "saturation"]
        bounds = [(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0.1, np.inf), (1.1, 100),
                  (-1, -5e-3), (0.5, np.inf), (0, np.inf)]
        self.params = FitParameters(values=values, labels=labels, axis_names=axis_names, bounds=bounds)

    def apply_max_width_to_bounds(self, max_half_width=None):
        if max_half_width is not None:
            self.max_half_width = max_half_width
        self.params.bounds[2] = (0, 2 * self.max_half_width)
        self.params.bounds[3] = (0.1, self.max_half_width)
        self.params.bounds[6] = (0.5, self.max_half_width)

    def evaluate(self, pixels, values=None):
        r"""Evaluate the MoffatGauss function.

        The function is normalized to have an integral equal to amplitude parameter, with normalisation factor:

        .. math::

            f(y) \propto  \frac{A}{ \frac{\Gamma(\alpha)}{\gamma \sqrt{\pi} \Gamma(\alpha -1/2)}+\eta\sqrt{2\pi}\sigma},
            \quad \int_{y_{\text{min}}}^{y_{\text{max}}} f(y) \mathrm{d}y = A

        Parameters
        ----------
        pixels: list
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        values: array_like
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
        >>> yy, xx = np.mgrid[:50, :60]
        >>> out = psf.evaluate(pixels=np.array([xx, yy]), values=p)

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np
            from spectractor.extractor.psf import MoffatGauss
            p = [2,20,30,4,2,-0.5,1,10]
            psf = MoffatGauss(p)
            yy, xx = np.mgrid[:50, :60]
            out = psf.evaluate(pixels=np.array([xx, yy]), values=p)
            fig = plt.figure(figsize=(5,5))
            plt.imshow(out, origin="lower")
            plt.xlabel("X [pixels]")
            plt.ylabel("Y [pixels]")
            plt.show()

        """
        if values is not None:
            self.params.values = np.asarray(values).astype(float)
        amplitude, x_c, y_c, gamma, alpha, eta_gauss, stddev, saturation = self.params.values
        if pixels.ndim == 3 and pixels.shape[0] == 2:
            x, y = pixels  # .astype(np.float32)  # float32 to increase rapidity
            out = evaluate_moffatgauss2d(x, y, amplitude, x_c, y_c, gamma, alpha, eta_gauss, stddev)
            if self.clip:
                out = np.clip(out, 0, saturation)
            return out
        elif pixels.ndim == 1:
            y = np.array(pixels)
            if alpha > 0.5:
                norm = evaluate_moffat1d_normalisation(gamma, alpha)
                out = evaluate_moffatgauss1d(y, amplitude, y_c, gamma, alpha, eta_gauss, stddev, norm_moffat=norm)
                if self.clip:
                    out = np.clip(out, 0, saturation)
                return out
            else:
                return np.zeros_like(y)
        else:  # pragma: no cover
            raise ValueError(f"Pixels array must have dimension 1 or shape=(2,Nx,Ny). Here pixels.ndim={pixels.shape}.")

    def jacobian(self, pixels, params, epsilon=None, model_input=None, analytical=True):
        r"""Evaluate the PSF Moffat+Gauss Jacobian.

        The function is normalized to have an integral equal to amplitude parameter, with normalisation factor:

        Parameters
        ----------
        pixels: array_like
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        params: array_like
            The parameter array. If None, the array used to instanciate the class is taken.
            If given, the class instance parameter array is updated.
        epsilon: array_like, optional
            The array of small steps to compute the partial derivatives of the model if analytical=False (default: None).
        model_input: array_like, optional
            A model input as a list with (x, model, model_err) to avoid an additional call to simulate() if analytical=False (default: None).
        analytical: bool, optional
            If True, use analytical derivatives to compute Jacobian operator. Otherwise use numerical differenciations
            with steps given by epsilon argument (default: True).

        Returns
        -------
        jacobian: array_like
            The PSF Jacobian.

        Examples
        --------
        >>> p = [2,20,30,4,2,-0.5,1,10]
        >>> epsilon = [0.001] * len(p)
        >>> psf = MoffatGauss(p)
        >>> psf.params.fixed = [True, True, False, False, False, False, False, True]  # fix amplitude, x_c, saturation

        2D case

        >>> yy, xx = np.mgrid[:50, :60]
        >>> J_ana = psf.jacobian(pixels=np.array([xx, yy]), params=p, epsilon=epsilon, analytical=True)
        >>> J_num = psf.jacobian(pixels=np.array([xx, yy]), params=p, epsilon=epsilon, analytical=False)
        >>> np.allclose(J_num, J_ana, rtol=1e-4, atol=1e-4)
        True
        >>> np.allclose(J_ana[0:2], 0)
        True

        ..  doctest::
            :hide:

            >>> assert J_ana.shape == (len(p), xx.size)
            >>> assert J_num.shape == (len(p), xx.size)

        1D case

        >>> y = np.mgrid[:50]
        >>> J_ana = psf.jacobian(pixels=y, params=p, epsilon=epsilon, analytical=True)
        >>> J_num = psf.jacobian(pixels=y, params=p, epsilon=epsilon, analytical=False)
        >>> np.allclose(J_num, J_ana, rtol=1e-3, atol=1e-3)
        True
        >>> np.allclose(J_ana[0:2], 0)
        True

        ..  doctest::
            :hide:

            >>> assert J_ana.shape == (len(p), y.size)
            >>> assert J_num.shape == (len(p), y.size)
        """
        if epsilon is None and not analytical:
            raise ValueError(f"If analytical=False, must give epsilon values for numerical differentiation.")
        amplitude, x_c, y_c, gamma, alpha, eta_gauss, sigma, saturation = self.params.values.astype(float)
        J = super().jacobian(pixels, params, epsilon=epsilon, model_input=model_input, analytical=analytical)
        if not analytical:
            if model_input is None:
                model = self.evaluate(pixels, values=params)
            else:
                x, model, model_err = model_input
            for ip, p in enumerate(params):
                if self.params.fixed[ip]:
                    continue
                tmp_p = np.copy(params).astype(float)
                if tmp_p[ip] + epsilon[ip] < self.params.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.params.bounds[ip][1]:
                    epsilon[ip] = - epsilon[ip]
                tmp_p[ip] += epsilon[ip]
                tmp_model = self.evaluate(pixels, values=tmp_p)
                J[ip] = (tmp_model.ravel() - model.ravel()) / epsilon[ip]
        else:
            fixed = np.array(self.params.fixed)
            if pixels.ndim == 1:
                norm = evaluate_moffat1d_normalisation(gamma, alpha)
                dnormda = evaluate_moffat1d_normalisation_dalpha(norm, alpha)
                J[:-1] = evaluate_moffatgauss1d_jacobian(pixels, amplitude, y_c, gamma, alpha, eta_gauss, sigma, norm, dnormda, fixed=fixed)   # [:-1] assume saturation is fixed
            else:
                xx, yy = pixels
                J[:-1] = evaluate_moffatgauss2d_jacobian(xx, yy, amplitude, x_c, y_c, gamma, alpha, eta_gauss, sigma, fixed=fixed)  # [:-1] assume saturation is fixed
        return J


class Order0(PSF):

    def __init__(self, target, values=None, clip=False):
        PSF.__init__(self, clip=clip)
        self.values_default = np.array([1, 0, 0, 1, 1]).astype(float)
        if values is None:
            values = np.copy(self.values_default)
        labels = ["amplitude", "x_c", "y_c", "gamma", "saturation"]
        axis_names = ["$A$", r"$x_c$", r"$y_c$", r"$\gamma$", "saturation"]
        bounds = [(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0.5, 5), (0, np.inf)]
        self.params = FitParameters(values=values, labels=labels, axis_names=axis_names, bounds=bounds)
        self.psf_func = self.build_interpolated_functions(target=target)

    def build_interpolated_functions(self, target):
        """Interpolate the order 0 image and make 1D and 2D functions centered on its centroid, with varying width
        and normalized to get an integral equal to unity.

        Parameters
        ----------
        target: Target
            The target with a target.image attribute to interpolate.

        Returns
        -------
        func: Callable
            The 2D interpolated function centered in (target.image_x0, target.image_y0).
        """
        xx = np.arange(0, target.image.shape[1]) - target.image_x0
        yy = np.arange(0, target.image.shape[0]) - target.image_y0
        data = target.image / np.sum(target.image)
        tmp_func = RegularGridInterpolator((xx, yy), data, method="nearest", bounds_error=False, fill_value=None)

        def func(x, y, amplitude, x_c, y_c, gamma):
            return amplitude * tmp_func(((y - y_c)/gamma, (x - x_c)/gamma))

        return func

    def apply_max_width_to_bounds(self, max_half_width=None):
        if max_half_width is not None:
            self.max_half_width = max_half_width
        self.params.bounds[2] = (0, 2 * self.max_half_width)

    def evaluate(self, pixels, values=None):
        r"""Evaluate the Order 0 interpolated function.

        The function is normalized to have an integral equal to amplitude parameter.

        Parameters
        ----------
        pixels: list
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        values: array_like
            The parameter array. If None, the array used to instanciate the class is taken.
            If given, the class instance parameter array is updated.

        Returns
        -------
        output: array_like
            The PSF function evaluated.

        Examples
        --------
        >>> from spectractor.extractor.images import Image, find_target
        >>> im = Image('tests/data/reduc_20170605_028.fits', target_label="PNG321.0+3.9")
        >>> im.plot_image()
        >>> guess = [820, 580]
        >>> parameters.VERBOSE = True
        >>> parameters.DEBUG = True
        >>> x0, y0 = find_target(im, guess)

        >>> p = [1,40,50,1,1e20]
        >>> psf = Order0(target=im.target)

        2D evaluation:

        >>> yy, xx = np.mgrid[:80, :100]
        >>> out = psf.evaluate(pixels=np.array([xx, yy]), values=p)

        1D evaluation:

        >>> out = psf.evaluate(pixels=np.arange(100), values=p)

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np
            from spectractor.extractor.psf import Moffat, Order0
            from spectractor.extractor.images import Image, find_target
            im = Image('tests/data/reduc_20170605_028.fits', target_label="PNG321.0+3.9")
            im.plot_image()
            guess = [820, 580]
            parameters.VERBOSE = True
            parameters.DEBUG = True
            x0, y0 = find_target(im, guess)
            p = [1,40,50,1,1e20]
            psf = Order0(target=im.target, p=p)
            yy, xx = np.mgrid[:80, :100]
            out = psf.evaluate(pixels=np.array([xx, yy]), values=p)
            fig = plt.figure(figsize=(5,5))
            plt.imshow(out, origin="lower")
            plt.xlabel("X [pixels]")
            plt.ylabel("Y [pixels]")
            plt.grid()
            plt.show()

        """
        if values is not None:
            self.params.values = np.asarray(values).astype(float)
        amplitude, x_c, y_c, gamma, saturation = self.params.values
        if pixels.ndim == 3 and pixels.shape[0] == 2:
            x, y = pixels  # .astype(np.float32)  # float32 to increase rapidity
            out = self.psf_func(x, y, amplitude, x_c, y_c, gamma)
            if self.clip:
                out = np.clip(out, 0, saturation)
            return out
        elif pixels.ndim == 1:
            y = np.array(pixels)
            out = self.psf_func(x_c, y, amplitude, x_c, y_c, gamma).T[0]
            out *= amplitude / np.sum(out)
            if self.clip:
                out = np.clip(out, 0, saturation)
            return out
        else:  # pragma: no cover
            raise ValueError(f"Pixels array must have dimension 1 or shape=(2,Nx,Ny). Here pixels.ndim={pixels.shape}.")

    def jacobian(self, pixels, params, epsilon=None, model_input=None, analytical=False):
        r"""Evaluate the PSF Order0 Jacobian.
        Analytical Jacobian is not available.

        The function is normalized to have an integral equal to amplitude parameter, with normalisation factor:

        Parameters
        ----------
        pixels: array_like
            List containing the X abscisse 2D array and the Y abscisse 2D array.
        params: array_like
            The parameter array. If None, the array used to instanciate the class is taken.
            If given, the class instance parameter array is updated.
        epsilon: array_like, optional
            The array of small steps to compute the partial derivatives of the model if analytical=False (default: None).
        model_input: array_like, optional
            A model input as a list with (x, model, model_err) to avoid an additional call to simulate() if analytical=False (default: None).
        analytical: bool, optional
            If True, use analytical derivatives to compute Jacobian operator. Otherwise use numerical differenciations
            with steps given by epsilon argument (default: True).

        Returns
        -------
        jacobian: array_like
            The PSF Jacobian.

        Examples
        --------
        >>> from spectractor.extractor.images import Image, find_target
        >>> im = Image('tests/data/reduc_20170605_028.fits', target_label="PNG321.0+3.9")
        >>> im.plot_image()
        >>> guess = [820, 580]
        >>> parameters.VERBOSE = True
        >>> parameters.DEBUG = True
        >>> x0, y0 = find_target(im, guess)

        >>> p = [1,40,50,1,1e20]
        >>> psf = Order0(target=im.target, values=p)

        >>> epsilon = [0.01] * len(p)
        >>> psf.params.fixed[2] = True  # fix y_c
        >>> psf.params.fixed = [True, True, False, False, True]  # fix amplitude, x_c, saturation
        >>> yy, xx = np.mgrid[:50, :60]
        >>> J_num = psf.jacobian(pixels=np.array([xx, yy]), params=p, epsilon=epsilon, analytical=False)
        >>> J_num.shape
        (5, 3000)
        >>> np.sum(J_num[2])
        0.0
        >>> np.allclose(J_num[0:2], 0)
        True

        ..  doctest::
            :hide:

            >>> assert J_num.shape == (len(p), xx.size)
        """
        if epsilon is None and not analytical:
            raise ValueError(f"If analytical=False, must give epsilon values for numerical differentiation.")
        J = super().jacobian(pixels, params, epsilon=epsilon, model_input=model_input, analytical=analytical)
        if not analytical:
            if model_input is None:
                model = self.evaluate(pixels, values=params)
            else:
                x, model, model_err = model_input
            for ip, p in enumerate(params):
                if self.params.fixed[ip]:
                    continue
                tmp_p = np.copy(params).astype(float)
                if tmp_p[ip] + epsilon[ip] < self.params.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.params.bounds[ip][1]:
                    epsilon[ip] = - epsilon[ip]
                tmp_p[ip] += epsilon[ip]
                tmp_model = self.evaluate(pixels, values=tmp_p)
                J[ip] = (tmp_model.ravel() - model.ravel()) / epsilon[ip]
        else:
            raise ValueError(f"Analytical=True not allowed for Order0 PSF data driven model.")
        return J


class PSFFitWorkspace(FitWorkspace):
    """Generic PSF fitting workspace.

    """

    def __init__(self, psf, data, data_errors, bgd_model_func=None, jacobian_analytical=False, file_name="",
                 verbose=False, plot=False, live_fit=False, truth=None):
        """

        Parameters
        ----------
        psf: PSF
            PSF model instance.
        data: array_like
            The data array (background subtracted) of dimension 1 or 2.
        data_errors
        bgd_model_func
        file_name
        verbose
        plot
        live_fit
        truth

        Examples
        --------

        Build a mock spectrogram with random Poisson noise:

        >>> p = np.array([100, 50, 50, 3, 2, -0.1, 2, 200])
        >>> psf = MoffatGauss(p)
        >>> yy, xx = np.mgrid[:50, :60]
        >>> data = psf.evaluate(pixels=np.array([xx, yy]), values=p)
        >>> data_errors = np.sqrt(data+1)

        Fit the data:

        >>> w = PSFFitWorkspace(psf, data, data_errors, bgd_model_func=None, verbose=True)

        """
        params = copy.deepcopy(psf.params)
        params.fixed[-1] = True  # saturation
        FitWorkspace.__init__(self, params, file_name=file_name, verbose=verbose, plot=plot,
                              live_fit=live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        if data.shape != data_errors.shape:
            raise ValueError(f"Data and uncertainty arrays must have the same shapes. "
                             f"Here data.shape={data.shape} and data_errors.shape={data_errors.shape}.")
        self.psf = psf
        self.data = data
        self.err = data_errors
        self.bgd_model_func = bgd_model_func
        self.saturation = self.psf.params.values[-1]
        self.guess = np.copy(self.psf.params.values)
        self.jacobian_analytical = jacobian_analytical

        # prepare the fit
        if data.ndim == 2:
            self.Ny, self.Nx = self.data.shape
            self.psf.apply_max_width_to_bounds(self.Ny)
            yy, xx = np.mgrid[:self.Ny, :self.Nx]
            self.pixels = np.asarray([xx, yy], dtype=int)
            # flat data for fitworkspace
            self.data = self.data.flatten()
            self.err = self.err.flatten()
        elif data.ndim == 1:
            self.Ny = self.data.size
            self.Nx = 1
            self.psf.apply_max_width_to_bounds(self.Ny)
            self.pixels = np.arange(self.Ny, dtype=int)
            self.params.fixed[1] = True
        else:
            raise ValueError(f"Data array must have dimension 1 or 2. Here pixels.ndim={data.ndim}.")

        # update bounds
        total_flux = np.sum(data)
        self.params.bounds[0] = [0.1 * total_flux, 2 * total_flux]

        # error matrix
        # here image uncertainties are assumed to be uncorrelated
        # (which is not exactly true in rotated images)
        self.W = 1. / (self.err * self.err)

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
        >>> yy, xx = np.mgrid[:50, :60]
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
            >>> assert np.mean(np.abs(mod.reshape(data.shape)-data)/data_errors) < 1

        Fit the data in 1D:

        >>> data1d = data[:,int(p[1])]
        >>> data1d_err = data_errors[:,int(p[1])]
        >>> psf.params.values[0] = p[0] / 10.5
        >>> w = PSFFitWorkspace(psf, data1d, data1d_err, bgd_model_func=None, verbose=True)
        >>> x, mod, mod_err = w.simulate(*psf.params.values)
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
            yy, xx = np.mgrid[:50, :60]
            data = psf.evaluate(np.array([xx, yy]), p)
            data = np.random.poisson(data)
            data_errors = np.sqrt(data+1)
            data = np.random.poisson(data)
            data_errors = np.sqrt(data+1)
            w = PSFFitWorkspace(psf, data, data_errors, bgd_model_func=bgd_model_func, verbose=True)
            x, mod, mod_err = w.simulate(*p[:-1])
            w.plot_fit()

        """
        self.params.values = np.copy(psf_params)
        self.model = self.psf.evaluate(self.pixels, values=self.params.values).flatten()
        self.model_err = np.zeros_like(self.model)
        return self.pixels, self.model, self.model_err

    def plot_fit(self):
        if self.Nx == 1:
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
                    ax[0].plot(self.pixels, self.psf.evaluate(self.pixels, values=self.guess)
                               + self.bgd_model_func(self.pixels), 'k--', label="Guess")
                else:
                    ax[0].plot(self.pixels, self.psf.evaluate(self.pixels, values=self.guess),
                               'k--', label="Guess")
                self.psf.values = np.copy(self.params.values)
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
            for ip, p in enumerate(self.params.labels):
                txt += f'{p}: {self.params.values[ip]:.4g}\n'
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
        else:
            data = np.copy(self.data).reshape((self.Ny, self.Nx))
            model = np.copy(self.model).reshape((self.Ny, self.Nx))
            err = np.copy(self.err).reshape((self.Ny, self.Nx))
            gs_kw = dict(width_ratios=[3, 0.15], height_ratios=[1, 1, 1, 1])
            fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(5, 7), gridspec_kw=gs_kw)
            norm = np.nanmax(self.data)
            plot_image_simple(ax[0, 0], data=model / norm, aspect='auto', cax=ax[0, 1], vmin=0, vmax=1,
                              units='1/max(data)')
            ax[0, 0].set_title("Model", fontsize=10, loc='center', color='white', y=0.8)
            plot_image_simple(ax[1, 0], data=data / norm, title='Data', aspect='auto',
                              cax=ax[1, 1], vmin=0, vmax=1, units='1/max(data)')
            ax[1, 0].set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
            residuals = (data - model)
            # residuals_err = self.spectrum.spectrogram_err / self.model
            norm = err
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
            ax[3, 0].plot(np.arange(self.Nx), data.sum(axis=0), label='Data')
            ax[3, 0].plot(np.arange(self.Nx), model.sum(axis=0), label='Model')
            ax[3, 0].set_ylabel('Transverse sum')
            ax[3, 0].set_xlabel(r'X [pixels]')
            ax[3, 0].legend(fontsize=7)
            ax[3, 0].grid(True)
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
            figname = os.path.splitext(self.filename)[0] + "_bestfit.pdf"
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')
        if parameters.PdfPages:
            parameters.PdfPages.savefig()

    def jacobian(self, params, epsilon, model_input=None):
        """Compute the Jacobian matrix of a PSF model, with analytical derivatives if they exist,
        numerical derivatives otherwise.

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
        if model_input is None:
            model_input = self.simulate(params)
        return self.psf.jacobian(self.pixels, params=params, epsilon=epsilon, model_input=model_input,
                                 analytical=self.jacobian_analytical)

def load_PSF(psf_type=parameters.PSF_TYPE, target=None, clip=False):
    """Load the PSF model with a keyword.

    Parameters
    ----------
    psf_type: str
        PSF model keyword (default: parameters.PSF_TYPE).
    target: Target, optional
        The Target instance to make Order0 PSF model (default: None).
    clip: bool, optional
        If True, PSF evaluation is clipped between 0 and saturation level (slower) (default: False)

    Returns
    -------
    psf: PSF
        An instance of the selected PSF model.

    Examples
    --------
    >>> parameters.VERBOSE = False
    >>> load_PSF(psf_type="Gauss", clip=True)  # doctest: +ELLIPSIS
    <....Gauss object at ...>
    >>> load_PSF(psf_type="Moffat", clip=True)  # doctest: +ELLIPSIS
    <....Moffat object at ...>
    >>> load_PSF(psf_type="MoffatGauss", clip=False)  # doctest: +ELLIPSIS
    <....MoffatGauss object at ...>
    >>> from spectractor.extractor.spectrum import Spectrum
    >>> spec = Spectrum("./tests/data/reduc_20170530_134_spectrum.fits")  # doctest: +ELLIPSIS
    >>> load_PSF(psf_type="Order0", clip=True, target=spec.target)  # doctest: +ELLIPSIS
    <....Order0 object at ...>

    """
    if psf_type == "Moffat":
        psf = Moffat(clip=clip)
    elif psf_type == "MoffatGauss":
        psf = MoffatGauss(clip=clip)
    elif psf_type == "Gauss":
        psf = Gauss(clip=clip)
    elif psf_type == "Order0":
        if target is None:
            raise ValueError(f"A Target instance must be given when PSF_TYPE='Order0'. I got target={target}.")
        psf = Order0(target=target, clip=clip)
    else:
        raise ValueError(f"Unknown PSF_TYPE={psf_type}. Must be either Gauss, Moffat or MoffatGauss.")
    return psf


if __name__ == "__main__":
    import doctest

    doctest.testmod()

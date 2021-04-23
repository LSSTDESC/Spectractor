import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

from spectractor.fit.fitter import FitWorkspace
from spectractor import parameters
from spectractor.config import set_logger


def AP_wlz(w, l, zeta, wA, lA, zetaA):
    return np.sqrt((wA - w) * (wA - w) + (lA - l) * (lA - l) + (zetaA - zeta) * (zetaA - zeta))


def BP_wlz(w, l, zeta, wB, lB, zetaB):
    return np.sqrt((wB - w) * (wB - w) + (lB - l) * (lB - l) + (zetaB - zeta) * (zetaB - zeta))


def n(w, l, zeta, w0, l0, wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    r"""
    Compute the number of grooves between a point P(w,l,zeta) and the order 0 incident point
    on the disperser $S'_0(w'_0,l'_0,\zeta'_0)$.

    w0: float
        Position $w'_0$ of the order 0 along the w axis, origin at center of the hologram.
    l0: float
        Position $l'_0$ of the order 0 along the l axis, origin at center of the hologram.
    zetaA, zetaB, lB, lA, wB, wA: float
        Positions of the A and B monochromatic coherent sources with respect to the center of the hologram.
    """
    AP = AP_wlz(w, l, zeta=zeta, wA=wA, lA=lA, zetaA=zetaA)
    BP = BP_wlz(w, l, zeta=zeta, wB=wB, lB=lB, zetaB=zetaB)
    ASp0 = AP_wlz(w0, l0, zeta=0, wA=wA, lA=lA, zetaA=zetaA)
    BSp0 = BP_wlz(w0, l0, zeta=0, wB=wB, lB=lB, zetaB=zetaB)
    return ((BP - AP) - (BSp0 - ASp0)) / lambda_hologram


def dndw(w0, l0, wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    r"""
    Compute $\frac{\partial n(w,l)}{\partial w}(w'_0,l'_0)$.

    w0: float
        Position $w'_0$ of the order 0 along the w axis, origin at center of the hologram.
    l0: float
        Position $l'_0$ of the order 0 along the l axis, origin at center of the hologram.
    zetaA, zetaB, lB, lA, wB, wA: float
        Positions of the A and B monochromatic coherent sources with respect to the center of the hologram.
    """
    rA = AP_wlz(w0, l0, zeta=0, wA=wA, lA=lA, zetaA=zetaA)
    rB = BP_wlz(w0, l0, zeta=0, wB=wB, lB=lB, zetaB=zetaB)
    return (-(wA - w0) / rA + (wB - w0) / rB) / lambda_hologram


def Neff(w0, l0, wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    r"""
    Compute $N_\mathrm{eff}(w'_0,l'_0)$

    w0: float
        Position $w'_0$ of the order 0 along the w axis, origin at center of the hologram.
    l0: float
        Position $l'_0$ of the order 0 along the l axis, origin at center of the hologram.
    zetaA, zetaB, lB, lA, wB, wA: float
        Positions of the A and B monochromatic coherent sources with respect to the center of the hologram.
    """
    dndwS0 = dndw(w0, l0, wA=wA, wB=wB, lA=lA, lB=lB, zetaA=zetaA, zetaB=zetaB, lambda_hologram=lambda_hologram)
    dndlS0 = dndl(w0, l0, wA=wA, wB=wB, lA=lA, lB=lB, zetaA=zetaA, zetaB=zetaB, lambda_hologram=lambda_hologram)
    return dndwS0 * np.sqrt(1 + (dndlS0 / dndwS0) ** 2)


def dndl(w0, l0, wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    r"""
    Compute \frac{\partial n(w,l)}{\partial l}(w'_0,l'_0)$.

    w0: float
        Position $w'_0$ of the order 0 along the w axis, origin at center of the hologram.
    l0: float
        Position $l'_0$ of the order 0 along the l axis, origin at center of the hologram.
    zetaA, zetaB, lB, lA, wB, wA: float
        Positions of the A and B monochromatic coherent sources with respect to the center of the hologram.
    """
    rA = AP_wlz(w0, l0, zeta=0, wA=wA, lA=lA, zetaA=zetaA)
    rB = BP_wlz(w0, l0, zeta=0, wB=wB, lB=lB, zetaB=zetaB)
    return (-(lA - l0) / rA + (lB - l0) / rB) / lambda_hologram


def d2ndw2(w0, l0, wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    r"""
    Compute \frac{\partial^2 n(w,l)}{\partial w^2}(w'_0,l'_0)$.

    w0: float
        Position $w'_0$ of the order 0 along the w axis, origin at center of the hologram.
    l0: float
        Position $l'_0$ of the order 0 along the l axis, origin at center of the hologram.
    zetaA, zetaB, lB, lA, wB, wA: float
        Positions of the A and B monochromatic coherent sources with respect to the center of the hologram.
    """
    rA = AP_wlz(w0, l0, zeta=0, wA=wA, lA=lA, zetaA=zetaA)
    rB = BP_wlz(w0, l0, zeta=0, wB=wB, lB=lB, zetaB=zetaB)
    return ((zetaA ** 2 + (lA - l0) ** 2) / rA ** 3 - (zetaB ** 2 + (lB - l0) ** 2) / rB ** 3) / lambda_hologram


def d2ndl2(w0, l0, wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    r"""
    Compute \frac{\partial^2 n(w,l)}{\partial l^2}(w'_0,l'_0)$.

    w0: float
        Position $w'_0$ of the order 0 along the w axis, origin at center of the hologram.
    l0: float
        Position $l'_0$ of the order 0 along the l axis, origin at center of the hologram.
    zetaA, zetaB, lB, lA, wB, wA: float
        Positions of the A and B monochromatic coherent sources with respect to the center of the hologram.
    """
    rA = AP_wlz(w0, l0, zeta=0, wA=wA, lA=lA, zetaA=zetaA)
    rB = BP_wlz(w0, l0, zeta=0, wB=wB, lB=lB, zetaB=zetaB)
    return ((zetaA ** 2 + (wA - w0) ** 2) / rA ** 3 - (zetaB ** 2 + (wB - w0) ** 2) / rB ** 3) / lambda_hologram


def rotate_frames(w, l, zeta, w0, l0, wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    beta = -np.arctan(
        dndl(w0, l0, wA=wA, wB=wB, lA=lA, lB=lB, zetaA=zetaA, zetaB=zetaB, lambda_hologram=lambda_hologram) /
        dndw(w0, l0, wA=wA, wB=wB, lA=lA, lB=lB, zetaA=zetaA, zetaB=zetaB, lambda_hologram=lambda_hologram))
    u = np.cos(beta) * (w - w0) + np.sin(beta) * (l - l0)
    v = -np.sin(beta) * (w - w0) + np.cos(beta) * (l - l0)
    z = DCCD + zeta
    return u, v, z


def d2ndu2(w0, l0, wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    r"""
    Compute \frac{\partial^2 n(w,l)}{\partial w^2}(w'_0,l'_0)$.

    w0: float
        Position $w'_0$ of the order 0 along the w axis, origin at center of the hologram.
    l0: float
        Position $l'_0$ of the order 0 along the l axis, origin at center of the hologram.
    zetaA, zetaB, lB, lA, wB, wA: float
        Positions of the A and B monochromatic coherent sources with respect to the center of the hologram.
    """
    uA, vA, zA = rotate_frames(wA, lA, zetaA, w0, l0, wA=wA, wB=wB, lA=lA, lB=lB, zetaA=zetaA, zetaB=zetaB,
                               lambda_hologram=lambda_hologram)
    uB, vB, zB = rotate_frames(wB, lB, zetaB, w0, l0, wA=wA, wB=wB, lA=lA, lB=lB, zetaA=zetaA, zetaB=zetaB,
                               lambda_hologram=lambda_hologram)
    rA = np.sqrt(uA * uA + vA * vA + (zetaA - DCCD) * (zetaA - DCCD))
    rB = np.sqrt(uB * uB + vB * vB + (zetaB - DCCD) * (zetaB - DCCD))
    return (((zetaA - DCCD) * (zetaA - DCCD) + vA * vA) / rA ** 3 - (
            (zetaB - DCCD) * (zetaB - DCCD) + vB * vB) / rB ** 3) / lambda_hologram


def d2ndv2(w0, l0, wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    r"""
    Compute \frac{\partial^2 n(w,l)}{\partial l^2}(w0,l0)$.

    w0: float
        Position $w'_0$ of the order 0 along the w axis, origin at center of the hologram.
    l0: float
        Position $l'_0$ of the order 0 along the l axis, origin at center of the hologram.
    zA, zB, lB, lA, wB, wA: float
        Positions of the A and B monochromatic coherent sources with respect to the center of the hologram.
    """
    uA, vA, zA = rotate_frames(wA, lA, zetaA, w0, l0, wA=wA, wB=wB, lA=lA, lB=lB, zetaA=zetaA, zetaB=zetaB,
                               lambda_hologram=lambda_hologram)
    uB, vB, zB = rotate_frames(wB, lB, zetaB, w0, l0, wA=wA, wB=wB, lA=lA, lB=lB, zetaA=zetaA, zetaB=zetaB,
                               lambda_hologram=lambda_hologram)
    rA = np.sqrt(uA * uA + vA * vA + (zetaA - DCCD) * (zetaA - DCCD))
    rB = np.sqrt(uB * uB + vB * vB + (zetaB - DCCD) * (zetaB - DCCD))
    return (((zetaA - DCCD) * (zetaA - DCCD) + uA * uA) / rA ** 3 - (
            (zetaB - DCCD) * (zetaB - DCCD) + uB * uB) / rB ** 3) / lambda_hologram


def set_symmetric_lines(wA, wB, lA, lB, zetaA, zetaB, lambda_hologram):
    # hline = lambda w: (lB-lA)/(wB-wA) * w - (lB*wA-lA*wB)/(wB-wA)
    hline = lambda w: (lB - lA) / (wB - wA) * w + lA - wA * (lB - lA) / (wB - wA)

    # search for equality AC=BC on hline
    def func(w):
        return (wA - w) ** 2 + (lA - hline(w)) ** 2 + zetaA ** 2 - (wB - w) ** 2 - (lB - hline(w)) ** 2 - zetaB ** 2

    wC = newton(func, 0.5 * (wA + wB))
    center = [wC, hline(wC)]
    vline = lambda w: -(wB - wA) / (lB - lA) * w + center[1] + center[0] * (wB - wA) / (lB - lA)

    def func(w):
        return d2ndw2(w, hline(w), wA=wA, wB=wB, lA=lA, lB=lB, zetaA=zetaA, zetaB=zetaB,
                      lambda_hologram=lambda_hologram)

    wC2 = newton(func, 0.5 * (wA + wB))
    # center = [0.5 * (wA + wB), hline(0.5 * (wA + wB))]
    center2 = [wC2, hline(wC2)]
    vline2 = lambda w: -(wB - wA) / (lB - lA) * w + center2[1] + center2[0] * (wB - wA) / (lB - lA)
    return hline, vline, vline2


class HologramModel:

    def __init__(self, wA, lA, zetaA, wB, lB, zetaB, DCCD, DT=parameters.OBS_DIAMETER,
                 fT=parameters.OBS_FOCAL, lambda_hologram=parameters.HOLO_LAMBDA_R, tilt=parameters.HOLO_TILT):
        r"""

        Parameters
        ----------
        wA: float
            Position of source A along $w$ axis in mm.
        lA: float
            Position of source A along $l$ axis in mm.
        zetaA: float
            Position of source A along $\zeta$ axis in mm.
        wB: float
            Position of source B along $w$ axis in mm.
        lB: float
            Position of source B along $l$ axis in mm.
        zetaB: float
            Position of source B along $\zeta$ axis in mm.
        DCCD: float
            Distance between the disperser and the sensor in m.
        DT: float
            Diameter of the telescope in m.
        fT: float
            Focal of the telescope in mm.
        lambda_hologram: float
            Record wavelength of the hologram in mm.
        tilt: float
            Tilt angle around dispersion axis to avoid ghosts, in deg.

        Examples
        --------
        >>> holo = HologramModel(wA=-10, lA=0, zetaA=200, wB=10, lB=0, zetaB=200, DCCD=200, DT=1200, fT=21600,
        ... lambda_hologram=639e-6, tilt=1)
        >>> fig, ax = holo.plot()
        """
        self.wA, self.lA, self.zetaA = wA, lA, zetaA
        self.wB, self.lB, self.zetaB = wB, lB, zetaB
        self.DCCD = DCCD
        self.DT = DT
        self.fT = fT
        self.lambda_hologram = lambda_hologram
        self.alpha0 = 0
        self.beta0 = tilt

    @property
    def AB(self):
        return np.sqrt((self.wB - self.wA) ** 2 + (self.lB - self.lA) ** 2)

    @property
    def D(self):
        return self.DCCD * self.DT / self.fT  # diameter of the intersected beam by the grating

    def __str__(self):
        txt = "Hologram properties:\n"
        txt += "\tA: " + f"[{self.wA:.2f}, {self.lA:.2f}, {self.zetaA:.2f}]"
        txt += "\tB: " + f"[{self.wB:.2f}, {self.lB:.2f}, {self.zetaB:.2f}]\n"
        txt += "\tA'B'=" + f"{self.AB:.2f}mm" + "\tDzeta=" + f"{self.zetaA - self.zetaB:.2f}mm"
        txt += "\tDCCD=" + f"{self.DCCD:.2f}mm\n"
        txt += "\talpha0=" + f"{self.alpha0:.2f}deg" + "\tbeta0=" + f"{self.beta0:.2f}deg"
        return txt

    def Neff(self, w, l):
        return Neff(w, l, wA=self.wA, wB=self.wB, lA=self.lA, lB=self.lB, zetaA=self.zetaA, zetaB=self.zetaB,
                    lambda_hologram=self.lambda_hologram)

    def alpha(self, w, l):
        N = Neff(w, l, wA=self.wA, wB=self.wB, lA=self.lA, lB=self.lB, zetaA=self.zetaA, zetaB=self.zetaB,
                 lambda_hologram=self.lambda_hologram)
        alpha = np.arctan2(dndl(w, l, wA=self.wA, wB=self.wB, lA=self.lA, lB=self.lB,
                                zetaA=self.zetaA, zetaB=self.zetaB, lambda_hologram=self.lambda_hologram),
                           N) * 180 / np.pi
        return alpha

    def plot(self, doplot=True):
        ws = np.arange(-self.AB, self.AB, 0.5)
        ls = np.arange(-self.AB, self.AB, 0.5)
        ww, ll = np.meshgrid(ws, ls)
        NN = self.Neff(ww, ll)
        alpha = self.alpha(ww, ll)
        fig = plt.figure(figsize=(8.5, 7))
        im = plt.pcolor(ws, ls, alpha, cmap="bwr", shading="auto")
        plt.grid()
        CS = plt.contour(ws, ls, NN, 10, linewidths=0.5, colors='k')
        plt.clabel(CS, inline=1, fontsize=14, fmt='%.1f')
        labels = ['Lines per mm\n' + rf'with $D_{{\mathrm{{CCD}}}}={self.DCCD:.4g}$mm']
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
        cb = fig.colorbar(im, fraction=0.046, pad=0.08)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.set_label(r'Rotation angle $\alpha$ [$^\circ$]', fontsize=14)
        plt.gca().set_aspect("equal")
        if np.isclose(self.lB, self.lA, atol=1e-8):
            plt.axhline(0, color="red", linestyle="-.", label='Symmetric axes', lw=2, zorder=40)
            plt.axvline(0, color="red", linestyle="-.", zorder=40)
        else:
            hline, vline, vline2 = set_symmetric_lines(wA=self.wA, wB=self.wB, lA=self.lA, lB=self.lB,
                                                       zetaA=self.zetaA, zetaB=self.zetaB,
                                                       lambda_hologram=self.lambda_hologram)
            plt.plot(ws, hline(ws), color="red", linestyle="-.", label='Symmetric axes', lw=2, zorder=40)
            plt.plot(ws, vline(ws), color="red", linestyle="-.", lw=2, zorder=40)
            plt.plot(ws, vline2(ws), color="red", linestyle="-.", lw=2, zorder=40)
        plt.scatter(self.wA, self.lA, s=200, facecolor='red', label="Optical center (A')", zorder=42)
        plt.scatter(self.wB, self.lB, s=200, facecolor='black', label=r"Order +1 position at $\lambda_R$ (B')",
                    zorder=42)
        plt.scatter(0, 0, s=50, facecolor='blue', label="[A'B'] middle", zorder=42)
        plt.xlabel(r"$w$ [mm]", fontsize=14)
        plt.ylabel(r"$l$ [mm]", fontsize=14)
        plt.legend()
        plt.xlim(np.min(ws), np.max(ws))
        plt.ylim(np.min(ls), np.max(ls))
        fig.tight_layout()
        if doplot:
            plt.show()
        return fig, plt.gca()


def get_alpha0(xA, xB, yA, yB):
    return np.arctan((yB - yA) / (xB - xA)) * 180 / np.pi


def get_euler_angle_matrix(alpha0, beta0):
    r"""Return the rotation matrix to go from $(x,y,z)$ frame to $(w,l,\zeta)$ frame.

    See Also
    --------
    https://fr.wikipedia.org/wiki/Angles_d%27Euler

    Parameters
    ----------
    alpha0: float
        Dispersion axis angle $\alpha_0$ around $z$ to go from $x$ to $w$ axis, in deg.
    beta0: float
        Dispersion axis angle $\beta_0$ around $x$ to go from $z$ to $\zeta$ axis, in deg.

    Returns
    -------
    A: array
        Rotation matrix.

    """
    psi = alpha0 * np.pi / 180
    theta = np.arctan(np.tan(beta0 * np.pi / 180) / np.cos(alpha0 * np.pi / 180))
    # phi = 0 as u and w are parallel
    A = np.array([[np.cos(psi), -np.sin(psi) * np.cos(theta), np.sin(psi) * np.sin(theta)],
                  [np.sin(psi), np.cos(psi) * np.cos(theta), -np.cos(psi) * np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return A


def from_xyz_to_wlzeta_AB(xA, xB, yA, yB, zA, zB, DCCD, beta0):
    # rotations
    alpha0 = get_alpha0(xA, xB, yA, yB)
    A = get_euler_angle_matrix(alpha0, beta0)
    wA, lA, zetaA = A.T @ np.array([xA, yA, zA])
    wB, lB, zetaB = A.T @ np.array([xB, yB, zB])
    # translations
    xc, yc, zc = (wA + wB) / 2, (lA + lB) / 2, -DCCD + zetaA  # + zetaB) / 2
    wA, wB = xc - wA, xc - wB
    lA, lB = lA - yc, lB - yc
    zetaA, zetaB = zetaA - zc, zetaB - zc
    # wA, wB = xc - xA, xc - xB
    # lA, lB = yA - yc, yB - yc
    return xc, yc, zc, wA, lA, zetaA, wB, lB, zetaB


def from_xyz_to_wlzeta(x, y, z, xA, xB, yA, yB, zA, zB, DCCD, beta0):
    # rotations
    alpha0 = get_alpha0(xA, xB, yA, yB)
    A = get_euler_angle_matrix(alpha0, beta0)
    w, l, zeta = A.T @ np.array([x, y, z])
    # translations
    xc, yc, zc, wA, lA, zetaA, wB, lB, zetaB = from_xyz_to_wlzeta_AB(xA, xB, yA, yB, zA, zB, DCCD, beta0)
    w = xc - w
    l = l - yc
    zeta = zeta - zc
    return w, l, zeta


class HoloFitWorkspace(FitWorkspace):

    def __init__(self, hologram, theta_xy, theta, theta_err, Neff_xy, Neff, Neff_err, file_name="", nwalkers=18,
                 nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None, only_theta=False, only_Neff=False):
        """

        Parameters
        ----------
        hologram: HologramModel
        theta_xy
        theta
        theta_err
        Neff_xy
        Neff
        Neff_err
        file_name
        nwalkers
        nsteps
        burnin
        nbins
        verbose
        plot
        live_fit
        truth
        only_theta
        only_Neff
        """
        FitWorkspace.__init__(self, file_name, nwalkers, nsteps, burnin, nbins, verbose, plot, live_fit, truth=truth)
        self.my_logger = set_logger(self.__class__.__name__)
        self.theta_xy = theta_xy
        self.Neffs_xy = Neff_xy
        self.Neffs = Neff
        self.thetas = theta
        self.hologram = hologram
        if only_theta:
            self.xy = theta_xy
            self.data = theta
            self.err = theta_err
        elif only_Neff:
            self.xy = Neff_xy
            self.data = Neff
            self.err = Neff_err
        else:
            self.xy = np.array(list(theta_xy) + list(Neff_xy))
            self.data = np.array(list(theta) + list(Neff))
            self.err = np.array(list(theta_err) + list(Neff_err))
        self.xyz = np.hstack((self.xy, np.zeros((self.xy.shape[0], 1))))
        self.only_Neff = only_Neff
        self.only_theta = only_theta
        self.xA = 20
        self.xB = 0
        self.yA = 50
        self.yB = 50
        self.zA = 0
        self.zB = 0
        self.p = np.array([self.xA, self.xB, self.yA, self.yB, self.zA, self.zB])
        self.fixed = [False] * len(self.p)
        # self.fixed[-2:] = True, True
        self.fixed[-2] = True  # zA at 0 ie zetaA = DCCD
        if only_theta:
            self.fixed[0] = True
            self.fixed[-2:] = True, True
            # self.fixed[-1] = True
        if only_Neff:
            self.fixed[-2:] = True, True
        # self.fixed[-2:] = True, True
        self.input_labels = ["xA", "xB", "yA", "yB", "zA", "zB"]
        self.axis_names = ["$x_A$", "$x_B$", "$y_A$", "$y_B$", "$z_{A}$", "$z_{B}$"]
        self.bounds = np.array([(-10, 30), (-10, 30), (-60, 60), (-60, 60), (-10, 10), (-10, 10)])
        self.nwalkers = max(2 * self.ndim, nwalkers)

    def update_hologram(self, xA, xB, yA, yB, zA, zB):
        xc, yc, zc, wA, lA, zetaA, wB, lB, zetaB = from_xyz_to_wlzeta_AB(xA, xB, yA, yB, zA, zB, self.hologram.DCCD,
                                                                         self.hologram.beta0)
        self.hologram.wA, self.hologram.lA, self.hologram.zetaA = wA, lA, zetaA
        self.hologram.wB, self.hologram.lB, self.hologram.zetaB = wB, lB, zetaB
        self.hologram.alpha0 = get_alpha0(xA, xB, yA, yB)

    def simulate(self, xA, xB, yA, yB, zA, zB):
        npoints = self.xy.shape[0]
        Neffs = np.zeros(npoints)
        thetas = np.zeros(npoints)
        self.update_hologram(xA, xB, yA, yB, zA, zB)
        alpha0 = get_alpha0(xA, xB, yA, yB)
        for k in range(npoints):
            xpos, ypos, zpos = self.xyz[k]
            w, l, zeta = from_xyz_to_wlzeta(xpos, ypos, zpos, xA, xB, yA, yB, zA, zB, self.hologram.DCCD,
                                            self.hologram.beta0)
            Neffs[k] = self.hologram.Neff(w, l)
            alpha = self.hologram.alpha(w, l)
            thetas[k] = alpha - alpha0
        if self.only_theta:
            xy = self.theta_xy
            self.model = thetas
        elif self.only_Neff:
            xy = self.Neffs_xy
            self.model = Neffs
        else:
            xy = self.xy
            self.model = np.array(list(thetas[:len(self.theta_xy)]) + list(Neffs[len(self.theta_xy):]))
        self.model_err = np.zeros_like(self.model)
        return xy, self.model, self.model_err

    def plot_fit(self, output=""):
        fit_xA, fit_xB, fit_yA, fit_yB, fit_zA, fit_zB = self.p
        self.update_hologram(*self.p)
        alpha0 = get_alpha0(fit_xA, fit_xB, fit_yA, fit_yB)
        ws = np.arange(-self.hologram.AB, self.hologram.AB, 0.5)
        ls = np.arange(-self.hologram.AB, self.hologram.AB, 0.5)
        ww, ll = np.meshgrid(ws, ls)
        alpha = self.hologram.alpha(ww, ll)

        fig, ax = self.hologram.plot(doplot=False)
        www, lll, zzz = from_xyz_to_wlzeta(self.theta_xy.T[0], self.theta_xy.T[1], np.zeros(self.theta_xy.shape[0]),
                                           fit_xA, fit_xB, fit_yA, fit_yB, fit_zA, fit_zB, self.hologram.DCCD,
                                           self.hologram.beta0)
        plt.scatter(www, lll, c=self.thetas + alpha0, s=100, cmap="bwr", marker='o', edgecolors='k', vmin=np.min(alpha),
                    vmax=np.max(alpha))  # , vmin=-np.max(np.abs(alpha)), vmax=np.max(np.abs(alpha))) #
        # www, lll, zzz = from_xyz_to_wlzeta(self.xyz.T[0], self.xyz.T[1], self.xyz.T[2], fit_xA, fit_xB, fit_yA,
        #                                    fit_yB, fit_zA, fit_zB, self.hologram.DCCD)
        # sc2 = plt.scatter(www, lll, c=NN, s=100, cmap='rainbow', marker='o', edgecolors='k')
        # cb2 = fig.colorbar(sc2,fraction=0.046, pad=0.04)#, cax=cbar_ax)
        # cb2.formatter.set_powerlimits((0, 0))
        # cb2.update_ticks()
        # cb2.set_label('$N_{\mathrm{eff}}$ [lpmm]',fontsize=14)
        fig.tight_layout()
        if output != "":
            fig.savefig(output, dpi=200)
        plt.show()


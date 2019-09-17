import time
import matplotlib.pyplot as plt
import numpy as np

from spectractor import parameters
from spectractor.config import set_logger
from spectractor.tools import plot_image_simple, from_lambda_to_colormap
from spectractor.simulation.simulator import SimulatorInit, SpectrogramModel
from spectractor.simulation.atmosphere import Atmosphere, AtmosphereGrid
from spectractor.fit.fitter import FitWorkspace, run_minimisation, run_gradient_descent, save_gradient_descent

plot_counter = 0


class SpectrogramFitWorkspace(FitWorkspace):

    def __init__(self, file_name, atmgrid_file_name="", nwalkers=18, nsteps=1000, burnin=100, nbins=10,
                 verbose=0, plot=False, live_fit=False, truth=None):
        FitWorkspace.__init__(self, file_name, nwalkers, nsteps, burnin, nbins, verbose, plot,
                              live_fit, truth=truth)
        self.spectrum, self.telescope, self.disperser, self.target = SimulatorInit(file_name)
        self.airmass = self.spectrum.header['AIRMASS']
        self.pressure = self.spectrum.header['OUTPRESS']
        self.temperature = self.spectrum.header['OUTTEMP']
        self.my_logger = set_logger(self.__class__.__name__)
        if atmgrid_file_name == "":
            self.atmosphere = Atmosphere(self.airmass, self.pressure, self.temperature)
        else:
            self.use_grid = True
            self.atmosphere = AtmosphereGrid(file_name, atmgrid_file_name)
            if parameters.VERBOSE:
                self.my_logger.info(f'\n\tUse atmospheric grid models from file {atmgrid_file_name}. ')
        self.crop_spectrogram()
        self.lambdas = self.spectrum.lambdas
        self.data = self.spectrum.spectrogram
        self.err = self.spectrum.spectrogram_err
        self.A1 = 1.0
        self.A2 = 0.01
        self.ozone = 400.
        self.pwv = 3
        self.aerosols = 0.05
        self.D = self.spectrum.header['D2CCD']
        self.psf_poly_params = self.spectrum.chromatic_psf.from_table_to_poly_params()
        length = len(self.spectrum.chromatic_psf.table)
        self.psf_poly_params = self.psf_poly_params[length:-1]  # remove saturation (fixed parameter)
        self.psf_poly_params_labels = np.copy(self.spectrum.chromatic_psf.poly_params_labels[length:-1])
        self.psf_poly_params_names = np.copy(self.spectrum.chromatic_psf.poly_params_names[length:-1])
        self.psf_poly_params_bounds = self.spectrum.chromatic_psf.set_bounds(data=None)
        self.shift_x = self.spectrum.header['PIXSHIFT']
        self.shift_y = 0.
        self.angle = self.spectrum.rotation_angle
        self.saturation = self.spectrum.spectrogram_saturation
        self.p = np.array([self.A1, self.A2, self.ozone, self.pwv, self.aerosols,
                           self.D, self.shift_x, self.shift_y, self.angle])
        self.psf_params_start_index = self.p.size
        self.p = np.concatenate([self.p, self.psf_poly_params])
        self.input_labels = ["A1", "A2", "ozone [db]", "PWV [mm]", "VAOD", r"D_CCD [mm]",
                             r"shift_x [pix]", r"shift_y [pix]", r"angle [deg]"] + list(self.psf_poly_params_labels)
        self.axis_names = ["$A_1$", "$A_2$", "ozone [db]", "PWV [mm]", "VAOD", r"$D_{CCD}$ [mm]",
                           r"$\Delta_{\mathrm{x}}$ [pix]", r"$\Delta_{\mathrm{y}}$ [pix]",
                           r"$\theta$ [deg]"] + list(self.psf_poly_params_names)
        self.bounds = np.concatenate([np.array([(0, 2), (0, 0.5), (0, 800), (1, 10), (0, 1),
                                                (50, 60), (-3, 3), (-3, 3), (-90, 90)]),
                                      self.psf_poly_params_bounds[:-1]])  # remove saturation
        if atmgrid_file_name != "":
            self.bounds[2] = (min(self.atmosphere.OZ_Points), max(self.atmosphere.OZ_Points))
            self.bounds[3] = (min(self.atmosphere.PWV_Points), max(self.atmosphere.PWV_Points))
            self.bounds[4] = (min(self.atmosphere.AER_Points), max(self.atmosphere.AER_Points))
        self.nwalkers = max(2 * self.ndim, nwalkers)
        self.simulation = SpectrogramModel(self.spectrum, self.atmosphere, self.telescope, self.disperser,
                                           with_background=True, fast_sim=False)
        self.get_spectrogram_truth()

    def crop_spectrogram(self):
        bgd_width = parameters.PIXWIDTH_BACKGROUND + parameters.PIXDIST_BACKGROUND - parameters.PIXWIDTH_SIGNAL
        # spectrogram must have odd size in y for the fourier simulation
        yeven = 0
        if (self.spectrum.spectrogram_Ny - 2 * bgd_width) % 2 == 0:
            yeven = 1
        self.spectrum.spectrogram_ymax = self.spectrum.spectrogram_ymax - bgd_width + yeven
        self.spectrum.spectrogram_ymin += bgd_width
        self.spectrum.spectrogram_bgd = self.spectrum.spectrogram_bgd[bgd_width:-bgd_width + yeven, :]
        self.spectrum.spectrogram = self.spectrum.spectrogram[bgd_width:-bgd_width + yeven, :]
        self.spectrum.spectrogram_err = self.spectrum.spectrogram_err[bgd_width:-bgd_width + yeven, :]
        self.spectrum.spectrogram_y0 -= bgd_width
        self.spectrum.spectrogram_Ny, self.spectrum.spectrogram_Nx = self.spectrum.spectrogram.shape
        self.my_logger.debug(f'\n\tSize of the spectrogram region after cropping: '
                             f'({self.spectrum.spectrogram_Nx},{self.spectrum.spectrogram_Ny})')

    def get_spectrogram_truth(self):
        if 'A1' in list(self.spectrum.header.keys()):
            A1_truth = self.spectrum.header['A1']
            A2_truth = self.spectrum.header['A2']
            ozone_truth = self.spectrum.header['OZONE']
            pwv_truth = self.spectrum.header['PWV']
            aerosols_truth = self.spectrum.header['VAOD']
            D_truth = self.spectrum.header['D2CCD']
            # shift_x_truth = self.spectrum.header['X0SHIFT']
            # shift_y_truth = self.spectrum.header['Y0SHIFT']
            angle_truth = self.spectrum.header['ROTANGLE']
            self.truth = (A1_truth, A2_truth, ozone_truth, pwv_truth, aerosols_truth,
                          D_truth, angle_truth)
        else:
            self.truth = None
        self.truth = None

    def plot_spectrogram_comparison_simple(self, ax, title='', extent=None, dispersion=False):
        lambdas = self.spectrum.lambdas
        sub = np.where((lambdas > parameters.LAMBDA_MIN) & (lambdas < parameters.LAMBDA_MAX))[0]
        sub = np.where(sub < self.spectrum.spectrogram_Nx)[0]
        if extent is not None:
            sub = np.where((lambdas > extent[0]) & (lambdas < extent[1]))[0]
        if len(sub) > 0:
            norm = np.max(self.spectrum.spectrogram[:, sub])
            plot_image_simple(ax[0, 0], data=self.model[:, sub] / norm, aspect='auto', cax=ax[0, 1], vmin=0, vmax=1,
                              units='1/max(data)')
            if dispersion:
                x = self.spectrum.chromatic_psf.table['Dx'][sub[2:-3]] + self.spectrum.spectrogram_x0 - sub[0]
                y = np.ones_like(x)
                ax[0, 0].scatter(x, y, cmap=from_lambda_to_colormap(self.lambdas[sub[2:-3]]), edgecolors='None',
                                 c=self.lambdas[sub[2:-3]],
                                 label='', marker='o', s=10)
            # p0 = ax.plot(lambdas, self.model(lambdas), label='model')
            # # ax.plot(self.lambdas, self.model_noconv, label='before conv')
            if title != '':
                ax[0, 0].set_title(title, fontsize=10, loc='center', color='white', y=0.8)
            plot_image_simple(ax[1, 0], data=self.spectrum.spectrogram[:, sub] / norm, title='Data', aspect='auto',
                              cax=ax[1, 1], vmin=0, vmax=1, units='1/max(data)')
            ax[1, 0].set_title('Data', fontsize=10, loc='center', color='white', y=0.8)
            residuals = (self.spectrum.spectrogram - self.model)
            # residuals_err = self.spectrum.spectrogram_err / self.model
            norm = self.spectrum.spectrogram_err
            residuals /= norm
            std = float(np.std(residuals[:, sub]))
            plot_image_simple(ax[2, 0], data=residuals[:, sub], vmin=-5 * std, vmax=5 * std, title='(Data-Model)/Err',
                              aspect='auto', cax=ax[2, 1], units='', cmap="bwr")
            ax[2, 0].set_title('(Data-Model)/Err', fontsize=10, loc='center', color='black', y=0.8)
            ax[2, 0].text(0.05, 0.05, f'mean={np.mean(residuals[:, sub]):.3f}\nstd={np.std(residuals[:, sub]):.3f}',
                          horizontalalignment='left', verticalalignment='bottom',
                          color='black', transform=ax[2, 0].transAxes)
            ax[0, 0].set_xticks(ax[2, 0].get_xticks()[1:-1])
            ax[0, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[1, 1].get_yaxis().set_label_coords(3.5, 0.5)
            ax[2, 1].get_yaxis().set_label_coords(3.5, 0.5)
            # ax[0, 0].get_yaxis().set_label_coords(-0.15, 0.6)
            # ax[2, 0].get_yaxis().set_label_coords(-0.15, 0.5)
            # remove the underlying axes
            # for ax in ax[3, 1]:
            ax[3, 1].remove()
            ax[3, 0].plot(self.lambdas[sub], self.spectrum.spectrogram.sum(axis=0)[sub], label='Data')
            ax[3, 0].plot(self.lambdas[sub], self.model.sum(axis=0)[sub], label='Model')
            ax[3, 0].set_ylabel('Cross spectrum')
            ax[3, 0].set_xlabel(r'$\lambda$ [nm]')
            ax[3, 0].legend(fontsize=7)
            ax[3, 0].grid(True)

    def simulate(self, A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, angle, *psf_poly_params):
        global plot_counter
        self.simulation.fix_psf_cube = False
        if np.all(np.isclose(psf_poly_params, self.p[self.psf_params_start_index:], rtol=1e-6)):
            self.simulation.fix_psf_cube = True
        lambdas, model, model_err = \
            self.simulation.simulate(A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, angle, psf_poly_params)
        self.p = np.array([A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, angle] + list(psf_poly_params))
        self.lambdas = lambdas
        self.model = model
        self.model_err = model_err
        if self.live_fit and (plot_counter % 30) == 0:
            self.plot_fit()
        plot_counter += 1
        return lambdas, model, model_err

    def jacobian(self, params, epsilon, fixed_params=None):
        start = time.time()
        lambdas, model, model_err = self.simulate(*params)
        model = model.flatten()
        J = np.zeros((params.size, model.size))
        for ip, p in enumerate(params):
            if fixed_params[ip]:
                continue
            if ip < self.psf_params_start_index:
                self.simulation.fix_psf_cube = True
            else:
                self.simulation.fix_psf_cube = False
            tmp_p = np.copy(params)
            if tmp_p[ip] + epsilon[ip] < self.bounds[ip][0] or tmp_p[ip] + epsilon[ip] > self.bounds[ip][1]:
                epsilon[ip] = - epsilon[ip]
            tmp_p[ip] += epsilon[ip]
            tmp_lambdas, tmp_model, tmp_model_err = self.simulate(*tmp_p)
            J[ip] = (tmp_model.flatten() - model) / epsilon[ip]
            # print(ip, self.input_labels[ip], p, tmp_p[ip] + epsilon[ip], J[ip])
        # if False:
        #     plt.imshow(J, origin="lower", aspect="auto")
        #     plt.show()
        print(f"\tjacobian time computation = {time.time() - start:.1f}s")
        return J

    def plot_fit(self):
        """
        Examples
        --------
        >>> file_name = 'outputs/reduc_20170530_130_spectrum.fits'
        >>> atmgrid_filename = file_name.replace('sim', 'reduc').replace('spectrum', 'atmsim')
        >>> fit_workspace = SpectrogramFitWorkspace(file_name, atmgrid_filename=atmgrid_filename,
        ... nwalkers=28, nsteps=20000, burnin=10000, nbins=10, verbose=1, plot=True, live_fit=False)
        >>> A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, angle, *psf = fit_workspace.p
        >>> lambdas, model, model_err = fit_workspace.simulation.simulate(A1, A2, ozone, pwv, aerosols,
        ... D, shift_x, shift_y, shift_t, angle, psf)
        >>> fit_workspace.lambdas = lambdas
        >>> fit_workspace.model = model
        >>> fit_workspace.model_err = model_err
        >>> fit_workspace.plot_fit()
        """
        gs_kw = dict(width_ratios=[3, 0.15, 1, 0.15, 1, 0.15], height_ratios=[1, 1, 1, 1])
        fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(12, 8), constrained_layout=True, gridspec_kw=gs_kw)

        A1, A2, ozone, pwv, aerosols, D, shift_x, shift_y, shift_t, *psf = self.p
        plt.suptitle(f'A1={A1:.3f}, A2={A2:.3f}, PWV={pwv:.3f}, OZ={ozone:.3g}, VAOD={aerosols:.3f}, '
                     f'D={D:.2f}mm, shift_x={shift_x:.2f}pix, shift_y={shift_y:.2f}pix')
        # main plot
        self.plot_spectrogram_comparison_simple(ax[:, 0:2], title='Spectrogram model', dispersion=True)
        # zoom O2
        self.plot_spectrogram_comparison_simple(ax[:, 2:4], extent=[730, 800], title='Zoom $O_2$', dispersion=True)
        # zoom H2O
        self.plot_spectrogram_comparison_simple(ax[:, 4:6], extent=[870, 1000], title='Zoom $H_2 O$', dispersion=True)
        # fig.tight_layout()
        if self.live_fit:
            plt.draw()
            plt.pause(1e-8)
            plt.close()
        else:
            if parameters.DISPLAY and self.verbose:
                plt.show()
        if parameters.SAVE:
            figname = self.filename.replace(self.filename.split('.')[-1], "_bestfit.pdf")
            self.my_logger.info(f"\n\tSave figure {figname}.")
            fig.savefig(figname, dpi=100, bbox_inches='tight')


def lnprob_spectrogram(p):
    global fit_workspace
    lp = fit_workspace.lnprior(p)
    if not np.isfinite(lp):
        return -1e20
    return lp + fit_workspace.lnlike_spectrogram(p)


def plot_psf_poly_params(psf_poly_params):
    from spectractor.extractor.psf import PSF1D
    psf = PSF1D()
    truth_psf_poly_params = [0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955,
                             -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901,
                             0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673,
                             528.3594585697788, 628.4966480821147, 12.438043546369354, 499.99999999999835]

    x = np.linspace(-1, 1, 100)
    for i in range(5):
        plt.plot(x, np.polynomial.legendre.legval(x, truth_psf_poly_params[3 * i:3 * i + 3]),
                 label="truth " + psf.param_names[1 + i])
        plt.plot(x, np.polynomial.legendre.legval(x, psf_poly_params[3 * i:3 * i + 3]),
                 label="fit " + psf.param_names[1 + i])

        plt.legend()
        plt.show()


def run_spectrogram_minimisation(fit_workspace, method="newton"):
    my_logger = set_logger(__name__)
    bounds = fit_workspace.bounds

    nll = lambda params: -fit_workspace.lnlike(params)

    # sim_134
    # guess = fit_workspace.p
    # truth sim_134
    # guess = np.array([1., 0.05, 300, 5, 0.03, 55.45, 0.0, 0.0, -1.54, 0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955, -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901, 0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673, 528.3594585697788, 628.4966480821147, 12.438043546369354])
    guess = np.array(
        [1., 0.05, 300, 5, 0.03, 55.45, -0.275, 0.0, -1.54, -1.47570237e-01, -5.00195918e-01, 4.74296776e-01,
         2.85776501e+00, -1.86436219e+00, 1.83899390e+00, 1.89342052e+00,
         -9.43239034e-01, 1.06985560e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 1.44368271e+00, -9.95896258e-01, 1.59015965e+00])
    # 5.00000000e+02
    guess = fit_workspace.p
    if method != "newton":
        run_minimisation(fit_workspace, method=method)
    else:
        fit_workspace.simulation.fast_sim = True
        costs = np.array([fit_workspace.chisq(guess)])
        if parameters.DISPLAY:
            fit_workspace.plot_fit()
        params_table = np.array([guess])
        start = time.time()
        epsilon = 1e-4 * guess
        epsilon[epsilon == 0] = 1e-3
        epsilon[0] = 1e-3  # A1
        epsilon[1] = 1e-4  # A2
        epsilon[2] = 1  # ozone
        epsilon[3] = 0.01  # pwv
        epsilon[4] = 0.001  # aerosols
        epsilon[5] = 0.001  # DCCD
        epsilon[6] = 0.0005  # shift_x
        my_logger.info(f"\n\tStart guess: {guess}")

        # cancel the Gaussian part of the PSF
        # TODO: solve this Gaussian PSF part issue
        guess[-6:] = 0

        # fit trace
        fix = [True] * guess.size
        fix[0] = False  # A1
        fix[1] = False  # A2
        fix[6] = True  # x0
        fix[7] = True  # y0
        fix[8] = True  # angle
        fit_workspace.simulation.fast_sim = True
        fix[fit_workspace.psf_params_start_index:fit_workspace.psf_params_start_index + 3] = [False] * 3
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = run_gradient_descent(fit_workspace, guess, epsilon, params_table, costs,
                                                   fix=fix, xtol=1e-2, ftol=1e-2, niter=20)

        # fit PSF
        guess = np.array(fit_workspace.p)
        fix = [True] * guess.size
        fix[0] = False  # A1
        fix[1] = False  # A2
        fit_workspace.simulation.fast_sim = True
        fix[fit_workspace.psf_params_start_index:fit_workspace.psf_params_start_index + 9] = [False] * 9
        # fix[fit_workspace.psf_params_start_index:] = [False] * (guess.size - fit_workspace.psf_params_start_index)
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = run_gradient_descent(fit_workspace, guess, epsilon, params_table, costs,
                                                   fix=fix, xtol=1e-2, ftol=1e-2, niter=20)

        # fit dispersion
        guess = np.array(fit_workspace.p)
        fix = [True] * guess.size
        fix[0] = False
        fix[1] = False
        fix[5] = False  # DCCD
        fix[6] = False  # x0
        fit_workspace.simulation.fix_psf_cube = True
        fit_workspace.simulation.fast_sim = True
        params_table, costs = run_gradient_descent(fit_workspace, guess, epsilon, params_table, costs,
                                                   fix=fix, xtol=1e-3, ftol=1e-2, niter=10)

        # fit all except Gaussian part of the PSF
        # TODO: solve this Gaussian PSF part issue
        guess = np.array(fit_workspace.p)
        fit_workspace.simulation.fast_sim = False
        fix = [False] * guess.size
        fix[6] = False  # x0
        fix[7] = True  # y0
        fix[8] = True  # angle
        fix[-6:] = [True] * 6  # gaussian part
        parameters.SAVE = True
        fit_workspace.simulation.fix_psf_cube = False
        params_table, costs = run_gradient_descent(fit_workspace, guess, epsilon, params_table, costs,
                                                   fix=fix, xtol=1e-5, ftol=1e-5, niter=40)
        if fit_workspace.filename != "":
            fit_workspace.save_parameters_summary(header=fit_workspace.spectrum.date_obs)
            save_gradient_descent(fit_workspace, costs, params_table)
        print(f"Newton: total computation time: {time.time() - start}s")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from spectractor.config import load_config

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path', default=["tests/data/reduc_20170530_134_spectrum.fits"],
                        help="Input fits file name. It can be a list separated by spaces, or it can use * as wildcard.",
                        nargs='*')
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                        help="Write results in given output directory (default: ./outputs/).")
    parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    parser.add_argument("-c", "--config", dest="config", default="config/ctio.ini",
                        help="INI config file. (default: config.ctio.ini).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input

    load_config(args.config)

    # filename = 'outputs/reduc_20170530_130_spectrum.fits'
    # filename = 'outputs/sim_20170530_134_spectrum.fits'
    # 062
    filename = 'CTIODataJune2017_reduced_RG715_v2_prod6/data_30may17/sim_20170530_067_spectrum.fits'
    atmgrid_filename = filename.replace('sim', 'reduc').replace('spectrum', 'atmsim')

    w = SpectrogramFitWorkspace(filename, atmgrid_file_name=atmgrid_filename, nsteps=1000,
                                burnin=2, nbins=10, verbose=1, plot=True, live_fit=False)
    run_spectrogram_minimisation(w, method="newton")
    # run_emcee(w, ln=lnprob_spectrogram)
    # w.analyze_chains()

from astropy.coordinates import Angle
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from spectractor.extractor.targets import *
from spectractor.extractor.dispersers import *


class Image(object):

    def __init__(self, filename, target=""):
        """
        Args:
            target:
            filename (:obj:`str`): path to the image
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.filename = filename
        self.units = 'ADU'
        self.expo = -1
        self.airmass = None
        self.date_obs = None
        self.filter = None
        self.filters = None
        self.data = None
        self.data_rotated = None
        self.stat_errors = None
        self.stat_errors_rotated = None
        self.load_image(filename)
        # Load the target if given
        self.target = None
        self.target_pixcoords = None
        self.target_pixcoords_rotated = None
        if target != "":
            self.target = Target(target, verbose=parameters.VERBOSE)
            self.header['TARGET'] = self.target.label
            self.header.comments['TARGET'] = 'object targeted in the image'
            self.header['REDSHIFT'] = self.target.redshift
            self.header.comments['REDSHIFT'] = 'redshift of the target'
        self.err = None

    def load_image(self, file_name):
        """
        Args:
            file_name (:obj:`str`): path to the image
        """
        self.my_logger.info('\n\tLoading image %s...' % file_name)
        self.header, self.data = load_fits(file_name)
        extract_info_from_CTIO_header(self, self.header)
        self.header['LSHIFT'] = 0.
        self.header['D2CCD'] = DISTANCE2CCD
        IMSIZE = int(self.header['XLENGTH'])
        parameters.PIXEL2ARCSEC = float(self.header['XPIXSIZE'])
        if self.header['YLENGTH'] != IMSIZE:
            self.my_logger.warning('\n\tImage rectangular: X=%d pix, Y=%d pix' % (IMSIZE, self.header['YLENGTH']))
        if self.header['YPIXSIZE'] != parameters.PIXEL2ARCSEC:
            self.my_logger.warning('\n\tPixel size rectangular: X=%d arcsec, Y=%d arcsec' % (
                parameters.PIXEL2ARCSEC, self.header['YPIXSIZE']))
        self.coord = SkyCoord(self.header['RA'] + ' ' + self.header['DEC'], unit=(units.hourangle, units.deg),
                              obstime=self.header['DATE-OBS'])
        self.my_logger.info('\n\tImage loaded')
        # Load the disperser
        self.my_logger.info('\n\tLoading disperser %s...' % self.disperser)
        self.disperser = Hologram(self.disperser, data_dir=parameters.HOLO_DIR, verbose=parameters.VERBOSE)
        # compute CCD gain map
        self.build_gain_map()
        self.convert_to_ADU_rate_units()
        self.compute_statistical_error()
        self.compute_parallactic_angle()

    def save_image(self, output_file_name, overwrite=False):
        save_fits(output_file_name, self.header, self.data, overwrite=overwrite)
        self.my_logger.info('\n\tImage saved in %s' % output_file_name)

    def build_gain_map(self):
        l = IMSIZE
        self.gain = np.zeros_like(self.data)
        # ampli 11
        self.gain[0:l // 2, 0:l // 2] = self.header['GTGAIN11']
        # ampli 12
        self.gain[0:l // 2, l // 2:l] = self.header['GTGAIN12']
        # ampli 21
        self.gain[l // 2:l, 0:l] = self.header['GTGAIN21']
        # ampli 22
        self.gain[l // 2:l, l // 2:l] = self.header['GTGAIN22']

    def convert_to_ADU_rate_units(self):
        self.data /= self.expo
        self.units = 'ADU/s'

    def convert_to_ADU_units(self):
        self.data *= self.expo
        self.units = 'ADU'

    def compute_statistical_error(self):
        # removes the zeros and negative pixels first
        # set to minimum positive value
        data = np.copy(self.data)
        zeros = np.where(data <= 0)
        min_noz = np.min(data[np.where(data > 0)])
        data[zeros] = min_noz
        # compute poisson noise
        self.stat_errors = np.sqrt(data) / np.sqrt(self.gain * self.expo)

    def compute_parallactic_angle(self):
        """from A. Guyonnet."""
        latitude = parameters.OBS_LATITUDE.split()
        latitude = float(latitude[0]) - float(latitude[1]) / 60. - float(latitude[2]) / 3600.
        latitude = Angle(latitude, units.deg).radian
        ha = Angle(self.header['HA'], unit='hourangle').radian
        dec = Angle(self.header['DEC'], unit=units.deg).radian
        parallactic_angle = np.arctan(np.sin(ha) / (np.cos(dec) * np.tan(latitude) - np.sin(dec) * np.cos(ha)))
        self.parallactic_angle = parallactic_angle * 180 / np.pi
        self.header['PARANGLE'] = self.parallactic_angle
        self.header.comments['PARANGLE'] = 'parallactic angle in degree'
        return self.parallactic_angle

    def plot_image_simple(self, ax, data=None, scale="lin", title="", units="Image units", plot_stats=False,
                          target_pixcoords=None):
        if data is None: data = np.copy(self.data)
        if plot_stats: data = np.copy(self.stat_errors)
        if scale == "log" or scale == "log10":
            # removes the zeros and negative pixels first
            zeros = np.where(data <= 0)
            min_noz = np.min(data[np.where(data > 0)])
            data[zeros] = min_noz
            # apply log
            data = np.log10(data)
        im = ax.imshow(data, origin='lower', cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        cb = plt.colorbar(im, ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7, prune=None)
        cb.update_ticks()
        cb.set_label('%s (%s scale)' % (units, scale))  # ,fontsize=16)
        if title != "": ax.set_title(title)
        if target_pixcoords is not None:
            ax.scatter(target_pixcoords[0], target_pixcoords[1], marker='o', s=100, edgecolors='k', facecolors='none',
                       label='Target', linewidth=2)

    def plot_image(self, data=None, scale="lin", title="", units="Image units", plot_stats=False,
                   target_pixcoords=None):
        fig, ax = plt.subplots(1, 1, figsize=[9.3, 8])
        self.plot_image_simple(ax, data=data, scale=scale, title=title, units=units, plot_stats=plot_stats,
                               target_pixcoords=target_pixcoords)
        plt.legend()
        if DISPLAY: plt.show()


def find_target(image, guess, rotated=False):
    sub_image, x0, y0, Dx, Dy, sub_errors = find_target_init(image=image, guess=guess, rotated=rotated)
    # find the target
    saturated_pixels = np.where(sub_image >= image.saturation)
    if len(saturated_pixels[0]) > 100:
        avX, avY = find_target_1Dprofile(image, sub_image, guess, rotated)
    else:
        avX, avY = find_target_2Dprofile(image, sub_image, guess, rotated)
    # compute target position
    theX = x0 - Dx + avX
    theY = y0 - Dy + avY
    image.my_logger.info('\n\tX,Y target position in pixels: %.3f,%.3f' % (theX, theY))
    if rotated:
        image.target_pixcoords_rotated = [theX, theY]
    else:
        image.target_pixcoords = [theX, theY]
        image.header['TARGETX'] = theX
        image.header.comments['TARGETX'] = 'target position on X axis'
        image.header['TARGETY'] = theY
        image.header.comments['TARGETY'] = 'target position on Y axis'
    return [theX, theY]


def find_target_init(image, guess, rotated=False):
    x0 = guess[0]
    y0 = guess[1]
    Dx = parameters.XWINDOW
    Dy = parameters.YWINDOW
    sub_errors = None
    if rotated:
        angle = image.rotation_angle * np.pi / 180.
        rotmat = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        vec = np.array(image.target_pixcoords) - 0.5 * np.array(image.data.shape)
        guess2 = np.dot(rotmat, vec) + 0.5 * np.array(image.data_rotated.shape)
        x0 = int(guess2[0, 0])
        y0 = int(guess2[0, 1])
    if rotated:
        Dx = parameters.XWINDOW_ROT
        Dy = parameters.YWINDOW_ROT
        sub_image = np.copy(image.data_rotated[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
        sub_erros = np.copy(image.stat_errors[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
    else:
        sub_image = np.copy(image.data[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
        sub_errors = np.copy(image.stat_errors[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
    image.saturation = parameters.MAXADU / image.expo
    sub_image = clean_target_spikes(sub_image, image.saturation)
    return sub_image, x0, y0, Dx, Dy, sub_errors


def find_target_1Dprofile(image, sub_image, guess, rotated=False):
    """
    Find precisely the position of the targeted object.

    Args:
        sub_image:
        rotated:
        guess (:obj:`list`): [x,y] guessed position of th target
    """
    NY, NX = sub_image.shape
    X = np.arange(NX)
    Y = np.arange(NY)
    # Mask aigrette
    saturated_pixels = np.where(sub_image >= image.saturation)
    if parameters.DEBUG:
        image.my_logger.info('\n\t%d saturated pixels: set saturation level to %d %s.' % (
            len(saturated_pixels[0]), image.saturation, image.units))
    # compute profiles
    profile_X_raw = np.sum(sub_image, axis=0)
    profile_Y_raw = np.sum(sub_image, axis=1)
    # fit and subtract smooth polynomial background
    # with 3sigma rejection of outliers (star peaks)
    bkgd_X = fit_poly1d_outlier_removal(X, profile_X_raw, order=2)
    bkgd_Y = fit_poly1d_outlier_removal(Y, profile_Y_raw, order=2)
    profile_X = profile_X_raw - bkgd_X(X)  # np.min(profile_X)
    profile_Y = profile_Y_raw - bkgd_Y(Y)  # np.min(profile_Y)
    avX, sigX = weighted_avg_and_std(X, profile_X ** 4)
    avY, sigY = weighted_avg_and_std(Y, profile_Y ** 4)
    if profile_X[int(avX)] < 0.8 * np.max(profile_X):
        image.my_logger.warning('\n\tX position determination of the target probably wrong')
    if profile_Y[int(avY)] < 0.8 * np.max(profile_Y):
        image.my_logger.warning('\n\tY position determination of the target probably wrong')
    if parameters.DEBUG:
        profile_X_max = np.max(profile_X_raw) * 1.2
        profile_Y_max = np.max(profile_Y_raw) * 1.2

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        image.plot_image_simple(ax1, data=sub_image, scale="log", title="", units=image.units, plot_stats=False,
                                target_pixcoords=[avX, avY])
        ax1.legend(loc=1)

        ax2.plot(X, profile_X_raw, 'r-', lw=2)
        ax2.plot(X, bkgd_X(X), 'g--', lw=2, label='bkgd')
        # ax2.axvline(guess[0],color='y',linestyle='-',label='old',lw=2)
        ax2.axvline(avX, color='b', linestyle='-', label='new', lw=2)
        ax2.grid(True)
        ax2.set_xlabel('X [pixels]')
        ax2.legend(loc=1)

        ax3.plot(Y, profile_Y_raw, 'r-', lw=2)
        ax3.plot(Y, bkgd_Y(Y), 'g--', lw=2, label='bkgd')
        # ax3.axvline(guess[1],color='y',linestyle='-',label='old',lw=2)
        ax3.axvline(avY, color='b', linestyle='-', label='new', lw=2)
        ax3.grid(True)
        ax3.set_xlabel('Y [pixels]')
        ax3.legend(loc=1)
        f.tight_layout()
        plt.show()
    return avX, avY


def find_target_2Dprofile(image, sub_image, guess, rotated=False):
    """
    Find precisely the position of the targeted object.

    Args:
        sub_image:
        rotated:
        guess (:obj:`list`): [x,y] guessed position of th target
    """
    NY, NX = sub_image.shape
    XX = np.arange(NX)
    YY = np.arange(NY)
    Y, X = np.mgrid[:NY, :NX]
    # fit and subtract smooth polynomial background
    # with 3sigma rejection of outliers (star peaks)
    bkgd_2D = fit_poly2d_outlier_removal(X, Y, sub_image, order=2)
    sub_image_subtracted = sub_image - bkgd_2D(X, Y)
    # find a first guess of the target position
    # avX,sigX = weighted_avg_and_std(X,(sub_image_subtracted)**4)
    # avY,sigY = weighted_avg_and_std(Y,(sub_image_subtracted)**4)
    avX, sigX = weighted_avg_and_std(XX, np.sum(sub_image_subtracted, axis=0) ** 4)
    avY, sigY = weighted_avg_and_std(YY, np.sum(sub_image_subtracted, axis=1) ** 4)
    # fit a 2D star profile close to this position
    # guess = [np.max(sub_image_subtracted),avX,avY,1,1] #for Moffat2D
    # guess = [np.max(sub_image_subtracted),avX,avY,1,1,0] #for Gauss2D
    guess = [np.max(sub_image_subtracted), avX, avY, 2, image.saturation]
    mean_prior = 10  # in pixels
    # bounds = [ [0.5*np.max(sub_image_subtracted),avX-mean_prior,avY-mean_prior,0,-np.inf], [2*np.max(sub_image_subtracted),avX+mean_prior,avY+mean_prior,np.inf,np.inf] ] #for Moffat2D
    # bounds = [ [0.5*np.max(sub_image_subtracted),avX-mean_prior,avY-mean_prior,2,2,0], [np.inf,avX+mean_prior,avY+mean_prior,10,10,np.pi] ] #for Gauss2D
    bounds = [[0.5 * np.max(sub_image_subtracted), avX - mean_prior, avY - mean_prior, 2, 0.9 * image.saturation],
              [10 * image.saturation, avX + mean_prior, avY + mean_prior, 15, 1.1 * image.saturation]]
    saturated_pixels = np.where(sub_image >= image.saturation)
    if len(saturated_pixels[0]) > 0:
        # sub_image_subtracted[saturated_pixels] = np.nan
        if parameters.DEBUG:
            image.my_logger.info('\n\t%d saturated pixels: set saturation level to %d %s.' % (
                len(saturated_pixels[0]), image.saturation, image.units))
    # fit
    star2D = fit_star2d_outlier_removal(X, Y, sub_image_subtracted, guess=guess, bounds=bounds, sigma=3, niter=50)
    # compute target positions
    new_avX = star2D.x_mean.value
    new_avY = star2D.y_mean.value
    image.target_star2D = star2D
    image.target_bkgd2D = bkgd_2D
    ymax, xmax = np.unravel_index(sub_image_subtracted.argmax(), sub_image_subtracted.shape)
    dist = np.sqrt((new_avY - avY) ** 2 + (new_avX - avX) ** 2)
    if dist > mean_prior / 2:
        image.my_logger.warning(
            '\n\tX=%.2f,Y=%.2f target position determination probably wrong: %.1f  pixels from profile detection (%d,%d)' % (
                new_avX, new_avY, dist, avX, avY))
        # debugging plots
    if parameters.DEBUG:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        image.plot_image_simple(ax1, data=sub_image_subtracted, scale="lin", title="", units=image.units,
                                target_pixcoords=[new_avX, new_avY])
        # ax1.scatter([Dx],[Dy],marker='o',s=100,facecolors='none',edgecolors='w',label='old')
        ax1.legend(loc=1)

        image.plot_image_simple(ax2, data=bkgd_2D(X, Y) + star2D(X, Y), scale="lin", title="",
                                units='Background + Gauss (%s)' % image.units)
        ax2.legend(loc=1)

        image.plot_image_simple(ax3, data=sub_image_subtracted - star2D(X, Y), scale="lin", title="",
                                units='Background+Gauss subtracted image (%s)' % image.units,
                                target_pixcoords=[new_avX, new_avY])
        # ax3.scatter([guess[0]],[guess[1]],marker='o',s=100,facecolors='none',edgecolors='w',label='old')
        ax3.legend(loc=1)

        f.tight_layout()
        plt.show()
    return new_avX, new_avY


def compute_rotation_angle_hessian(image, deg_threshold=10, width_cut=YWINDOW, right_edge=IMSIZE - 200,
                                   margin_cut=12):
    x0, y0 = np.array(image.target_pixcoords).astype(int)
    # extract a region
    data = np.copy(image.data[y0 - width_cut:y0 + width_cut, 0:right_edge])
    lambda_plus, lambda_minus, theta = hessian_and_theta(data, margin_cut)
    # thresholds
    lambda_threshold = np.min(lambda_minus)
    mask = np.where(lambda_minus > lambda_threshold)
    theta_mask = np.copy(theta)
    theta_mask[mask] = np.nan
    minimum_pixels = 0.01 * 2 * width_cut * right_edge
    while len(theta_mask[~np.isnan(theta_mask)]) < minimum_pixels:
        lambda_threshold /= 2
        mask = np.where(lambda_minus > lambda_threshold)
        theta_mask = np.copy(theta)
        theta_mask[mask] = np.nan
        # print len(theta_mask[~np.isnan(theta_mask)]), lambda_threshold
    theta_guess = image.disperser.theta(image.target_pixcoords)
    mask2 = np.where(np.abs(theta - theta_guess) > deg_threshold)
    theta_mask[mask2] = np.nan
    theta_hist = []
    theta_hist = theta_mask[~np.isnan(theta_mask)].flatten()
    theta_median = np.median(theta_hist)
    theta_critical = 180. * np.arctan(20. / IMSIZE) / np.pi
    image.header['THETAFIT'] = theta_median
    image.header.comments['THETAFIT'] = '[USED] rotation angle from the Hessian analysis'
    image.header['THETAINT'] = theta_guess
    image.header.comments['THETAINT'] = 'rotation angle interp from disperser scan'
    if abs(theta_median - theta_guess) > theta_critical:
        image.my_logger.warning(
            '\n\tInterpolated angle and fitted angle disagrees with more than 20 pixels over {:d} pixels:  {:.2f} vs {:.2f}'.format(
                IMSIZE, theta_median, theta_guess))
    if parameters.DEBUG:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        xindex = np.arange(data.shape[1])
        x_new = np.linspace(xindex.min(), xindex.max(), 50)
        y_new = width_cut + (x_new - x0) * np.tan(theta_median * np.pi / 180.)
        ax1.imshow(theta_mask, origin='lower', cmap=cm.brg, aspect='auto', vmin=-deg_threshold, vmax=deg_threshold)
        # ax1.imshow(np.log10(data),origin='lower',cmap="jet",aspect='auto')
        ax1.plot(x_new, y_new, 'b-')
        ax1.set_ylim(0, 2 * width_cut)
        ax1.set_xlabel('X [pixels]')
        ax1.set_xlabel('Y [pixels]')
        ax1.grid(True)
        n, bins, patches = ax2.hist(theta_hist, bins=int(np.sqrt(len(theta_hist))))
        ax2.plot([theta_median, theta_median], [0, np.max(n)])
        ax2.set_xlabel("Rotation angles [degrees]")
        plt.show()
    return theta_median


def turn_image(image):
    image.rotation_angle = compute_rotation_angle_hessian(image)
    image.my_logger.info('\n\tRotate the image with angle theta=%.2f degree' % image.rotation_angle)
    image.data_rotated = np.copy(image.data)
    if not np.isnan(image.rotation_angle):
        image.data_rotated = ndimage.interpolation.rotate(image.data, image.rotation_angle,
                                                          prefilter=parameters.ROT_PREFILTER,
                                                          order=parameters.ROT_ORDER)
        image.stat_errors_rotated = ndimage.interpolation.rotate(image.stat_errors, image.rotation_angle,
                                                                 prefilter=parameters.ROT_PREFILTER,
                                                                 order=parameters.ROT_ORDER)
    if parameters.DEBUG:
        margin = 200
        y0 = int(image.target_pixcoords[1])
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=[8, 8])
        image.plot_image_simple(ax1, data=image.data[y0 - parameters.YWINDOW:y0 + parameters.YWINDOW, margin:-margin],
                                scale="log", title='Raw image (log10 scale)', units=image.units,
                                target_pixcoords=(image.target_pixcoords[0] - margin, parameters.YWINDOW))
        ax1.plot([0, image.data.shape[0] - 2 * margin], [parameters.YWINDOW, parameters.YWINDOW], 'k-')
        image.plot_image_simple(ax2, data=image.data_rotated[y0 - parameters.YWINDOW:y0 + parameters.YWINDOW,
                                          margin:-margin], scale="log", title='Turned image (log10 scale)',
                                units=image.units, target_pixcoords=image.target_pixcoords_rotated)
        ax2.plot([0, image.data_rotated.shape[0] - 2 * margin], [parameters.YWINDOW, parameters.YWINDOW], 'k-')
        plt.show()


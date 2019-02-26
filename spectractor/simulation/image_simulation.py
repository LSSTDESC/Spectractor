from spectractor.extractor.images import *
from spectractor.extractor.spectrum import *
from spectractor.simulation.simulator import *
from spectractor import parameters
import copy


# from astroquery.gaia import Gaia, TapPlus, GaiaClass
# Gaia = GaiaClass(TapPlus(url='http://gaia.ari.uni-heidelberg.de/tap'))


class StarModel:

    def __init__(self, pixcoords, model, amplitude, target=None):
        """ [x0, y0] coords in pixels, sigma width in pixels, A height in image units"""
        self.my_logger = set_logger(self.__class__.__name__)
        self.x0 = pixcoords[0]
        self.y0 = pixcoords[1]
        self.amplitude = amplitude
        self.target = target
        self.model = copy.deepcopy(model)
        self.model.x_mean = self.x0
        self.model.y_mean = self.y0
        self.model.amplitude = amplitude
        # to be realistic, usually fitted fwhm is too big, divide by 2
        self.fwhm = self.model.gamma / 2
        self.sigma = self.model.stddev / 2

    def plot_model(self):
        yy, xx = np.meshgrid(np.linspace(self.y0 - 10 * self.fwhm, self.y0 + 10 * self.fwhm, 50),
                             np.linspace(self.x0 - 10 * self.fwhm, self.x0 + 10 * self.fwhm, 50))
        star = self.model(xx, yy)
        fig, ax = plt.subplots(1, 1)
        im = plt.imshow(star, origin='lower', cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Star model: A=%.2f, fwhm=%.2f' % (self.amplitude, self.fwhm))
        cb = plt.colorbar(im, ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7, prune=None)
        cb.update_ticks()
        cb.set_label('Arbitrary units')  # ,fontsize=16)
        if parameters.DISPLAY: plt.show()


class StarFieldModel:

    def __init__(self, base_image, threshold=0):
        self.target = base_image.target
        self.field = None
        self.stars = []
        self.pixcoords = []
        x0, y0 = base_image.target_pixcoords
        '''
        result = Gaia.query_object_async(self.target.coord, 
            width=CCD_IMSIZE*CCD_PIXEL2ARCSEC*units.arcsec/np.cos(self.target.coord.dec.radian), 
            height=CCD_IMSIZE*CCD_PIXEL2ARCSEC*units.arcsec)
        if parameters.DEBUG:
            print result.pprint()
        for o in result:
            if o['dist'] < 0.005 : continue
            radec = SkyCoord(ra=float(o['ra']),dec=float(o['dec']), frame='icrs', unit='deg')
            y = int(y0 - (radec.dec.arcsec - self.target.coord.dec.arcsec)/CCD_PIXEL2ARCSEC)
            x = int(x0 - (radec.ra.arcsec - self.target.coord.ra.arcsec)*np.cos(radec.dec.radian)/CCD_PIXEL2ARCSEC)
            w = 10 # search windows in pixels
            if x<w or x>CCD_IMSIZE-w: continue
            if y<w or y>CCD_IMSIZE-w: continue
            sub =  base_image.data[y-w:y+w,x-w:x+w]
            A = np.max(sub) - np.min(sub)
            if A < threshold: continue
            self.stars.append( StarModel([x,y],base_image.target_star2D,A) )
            self.pixcoords.append([x,y])
        self.pixcoords = np.array(self.pixcoords).T
        np.savetxt('starfield.txt',self.pixcoords)
'''
        # mask background, faint stars, and saturated pixels
        image_thresholded = np.copy(base_image.data)
        self.saturation = 0.99 * parameters.CCD_MAXADU / base_image.expo
        self.saturated_pixels = np.where(image_thresholded > self.saturation)
        image_thresholded[self.saturated_pixels] = 0.
        image_thresholded -= threshold
        image_thresholded[np.where(image_thresholded < 0)] = 0.
        # mask order0 and spectrum
        margin = 30
        for y in range(int(y0) - 100, int(y0) + 100):
            for x in range(parameters.CCD_IMSIZE):
                u, v = pixel_rotation(x, y, base_image.disperser.theta([x0, y0]) * np.pi / 180., x0, y0)
                if margin > v > -margin:
                    image_thresholded[y, x] = 0.
        # look for local maxima and create stars
        peak_positions = detect_peaks(image_thresholded)
        for y in range(parameters.CCD_IMSIZE):
            for x in range(parameters.CCD_IMSIZE):
                if peak_positions[y, x]:
                    if np.sqrt((y - y0) ** 2 + (x - x0) ** 2) < 10 * base_image.target_star2D.gamma:
                        continue  # no double star
                    self.stars.append(StarModel([x, y], base_image.target_star2D, image_thresholded[y, x]))
                    self.pixcoords.append([x, y])
        self.pixcoords = np.array(self.pixcoords).T
        self.fwhm = base_image.target_star2D.gamma

    def model(self, x, y):
        if self.field is None:
            window = int(10 * self.fwhm)
            self.field = self.stars[0].model(x, y)
            for k in range(1, len(self.stars)):
                left = max(0, int(self.pixcoords[0][k]) - window)
                right = min(parameters.CCD_IMSIZE, int(self.pixcoords[0][k]) + window)
                low = max(0, int(self.pixcoords[1][k]) - window)
                up = min(parameters.CCD_IMSIZE, int(self.pixcoords[1][k]) + window)
                yy, xx = np.mgrid[low:up, left:right]
                self.field[low:up, left:right] += self.stars[k].model(xx, yy)
            self.field[self.saturated_pixels] += self.saturation
        return self.field

    def plot_model(self):
        yy, xx = np.mgrid[0:parameters.CCD_IMSIZE:1, 0:parameters.CCD_IMSIZE:1]
        starfield = self.model(xx, yy)
        fig, ax = plt.subplots(1, 1)
        im = plt.imshow(starfield, origin='lower', cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title(f'Star field model: fwhm={self.fwhm.value:.2f}')
        cb = plt.colorbar(im, ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7, prune=None)
        cb.update_ticks()
        cb.set_label('Arbitrary units')  # ,fontsize=16)
        if parameters.DISPLAY: plt.show()


class BackgroundModel:

    def __init__(self, level, frame=None):
        self.my_logger = set_logger(self.__class__.__name__)
        self.level = level
        if self.level <= 0:
            self.my_logger.warning('\n\tBackground level must be strictly positive.')
        self.frame = frame

    def model(self, x, y):
        """Background model for the image simulation.
        Args:
            x:
            y:

        Returns:

        """
        bkgd = self.level * np.ones_like(x)
        if self.frame is None:
            return bkgd
        else:
            xlim, ylim = self.frame
            bkgd[ylim:, :] = self.level / 100
            bkgd[:, xlim:] = self.level / 100
            kernel = np.outer(gaussian(parameters.CCD_IMSIZE, 50), gaussian(parameters.CCD_IMSIZE, 50))
            bkgd = fftconvolve(bkgd, kernel, mode='same')
            bkgd *= self.level / bkgd[parameters.CCD_IMSIZE // 2, parameters.CCD_IMSIZE // 2]
            return bkgd

    def plot_model(self):
        yy, xx = np.mgrid[0:parameters.CCD_IMSIZE:1, 0:parameters.CCD_IMSIZE:1]
        bkgd = self.model(xx, yy)
        fig, ax = plt.subplots(1, 1)
        im = plt.imshow(bkgd, origin='lower', cmap='jet')
        ax.grid(color='white', ls='solid')
        ax.grid(True)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Background model')
        cb = plt.colorbar(im, ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.locator = MaxNLocator(7, prune=None)
        cb.update_ticks()
        cb.set_label('Arbitrary units')  # ,fontsize=16)
        if parameters.DISPLAY: plt.show()


class SpectrumModel:

    def __init__(self, base_image, spectrumsim, sigma, A1=1, A2=0, reso=None, rotation=False):
        self.my_logger = set_logger(self.__class__.__name__)
        self.base_image = base_image
        self.spectrumsim = spectrumsim
        self.disperser = base_image.disperser
        self.sigma = sigma
        self.A1 = A1
        self.A2 = A2
        self.reso = reso
        self.rotation = rotation
        self.yprofile = models.Gaussian1D(1, 0, sigma)

    def model(self, x, y):
        self.true_lambdas = np.arange(parameters.LAMBDA_MIN, parameters.LAMBDA_MAX)
        self.true_spectrum = np.copy(self.spectrumsim.model(self.true_lambdas))
        x0, y0 = self.base_image.target_pixcoords
        if self.rotation:
            theta = self.disperser.theta(self.base_image.target_pixcoords) * np.pi / 180.
            u = np.cos(theta) * (x - x0) + np.sin(theta) * (y - y0)
            v = -np.sin(theta) * (x - x0) + np.cos(theta) * (y - y0)
            x = u + x0
            y = v + y0
        l = self.disperser.grating_pixel_to_lambda(x - x0, x0=self.base_image.target_pixcoords, order=1)
        amp = self.A1 * self.spectrumsim.model(l) * self.yprofile(y - y0) + self.A1 * self.A2 * self.spectrumsim.model(
            l / 2) * self.yprofile(y - y0)
        amp = amp * parameters.FLAM_TO_ADURATE * l * np.gradient(l, axis=1)
        if self.reso is not None:
            amp = fftconvolve_gaussian(amp, self.reso)
        return amp


class ImageModel(Image):

    def __init__(self, filename, target=None):
        self.my_logger = set_logger(self.__class__.__name__)
        Image.__init__(self, filename, target=target.label)
        self.true_lambdas = None
        self.true_spectrum = None

    def compute(self, star, background, spectrogram, starfield=None):
        yy, xx = np.mgrid[0:parameters.CCD_IMSIZE:1, 0:parameters.CCD_IMSIZE:1]
        self.data = star.model(xx, yy) + background.model(xx, yy)
        self.data[spectrogram.spectrogram_ymin:spectrogram.spectrogram_ymax,
        spectrogram.spectrogram_xmin:spectrogram.spectrogram_xmax] += (spectrogram.data - spectrogram.spectrogram_bgd)
        self.true_lambdas = spectrogram.lambdas
        self.true_spectrum = spectrogram.true_spectrum
        if starfield is not None:
            self.data += starfield.model(xx, yy)

    def add_poisson_noise(self):
        if self.units == 'ADU':
            self.my_logger.error('\n\tPoisson noise procedure has to be applied on map in ADU/s units')
        d = np.copy(self.data).astype(float)
        d *= self.expo * self.gain
        noisy = np.random.poisson(d).astype(float)
        self.data = noisy / (self.expo * self.gain)

    def save_image(self, output_filename, overwrite=False):
        hdu0 = fits.PrimaryHDU()
        hdu0.data = self.data
        hdu0.header = self.header
        hdu1 = fits.ImageHDU()
        hdu1.data = [self.true_lambdas, self.true_spectrum]
        hdulist = fits.HDUList([hdu0, hdu1])
        hdulist.writeto(output_filename, overwrite=overwrite)
        self.my_logger.info('\n\tImage saved in %s' % output_filename)

    def load_image(self, filename):
        super(ImageModel, self).load_image(filename)
        hdu_list = fits.open(filename)
        # self.true_lambdas, self.true_spectrum = hdu_list[1].data


def ImageSim(image_filename, spectrum_filename, outputdir, pwv=5, ozone=300, aerosols=0.03, A1=1, A2=0.05,
             psf_poly_params=None,
             with_rotation=True,
             with_stars=True):
    """ The basic use of the extractor consists first to define:
    - the path to the fits image from which to extract the image, 
    - the path of the output directory to save the extracted spectrum (created automatically if does not exist yet),
    - the rough position of the object in the image,
    - the name of the target (to search for the extra-atmospheric spectrum if available).
    Then different parameters and systematics can be set:
    - pwv: the pressure water vapor (in mm)
    - ozone: the ozone quantity (in XX)
    - aerosols: the vertical aerosol optical depth
    - A1: a global grey absorption parameter for the spectrum
    - A2: the relative amplitude of second order compared with first order
    - with_rotation: rotate the spectrum according to the disperser characteristics (True by default)
    - with_stars: include stars in the image field (True by default)
    """
    my_logger = set_logger(__name__)
    my_logger.info(f'\n\tStart IMAGE SIMULATOR')
    # Load reduced image
    spectrum, telescope, disperser, target = SimulatorInit(spectrum_filename)
    image = ImageModel(image_filename, target=target)
    guess = [spectrum.header['TARGETX'], spectrum.header['TARGETY']]
    if parameters.DEBUG:
        image.plot_image(scale='log10', target_pixcoords=guess)
    # Find the exact target position in the raw cut image: several methods
    my_logger.info('\n\tSearch for the target in the image...')
    target_pixcoords = find_target(image, guess)
    # Background model
    my_logger.info('\n\tBackground model...')
    background = BackgroundModel(level=image.target_bkgd2D(0, 0), frame=(1600, 1650))
    if parameters.DEBUG:
        background.plot_model()
    # Target model
    my_logger.info('\n\tStar model...')
    star = StarModel(target_pixcoords, image.target_star2D, image.target_star2D.amplitude.value, target=image.target)
    reso = star.sigma
    if parameters.DEBUG:
        star.plot_model()
    # Star field model
    starfield = None
    if with_stars:
        my_logger.info('\n\tStar field model...')
        starfield = StarFieldModel(image, threshold=0.01 * star.amplitude)
        if parameters.VERBOSE:
            image.plot_image(scale='log10', target_pixcoords=starfield.pixcoords)
            starfield.plot_model()
    # Spectrum model
    my_logger.info('\n\tSpectum model...')
    lambdas = spectrum.lambdas
    airmass = image.header['AIRMASS']
    pressure = image.header['OUTPRESS']
    temperature = image.header['OUTTEMP']
    telescope = TelescopeTransmission(image.filter)
    spectrogram = SpectrogramSimulatorCore(spectrum, telescope, disperser, airmass, pressure,
                                           temperature, pwv=pwv, ozone=ozone, aerosols=aerosols, A1=A1, A2=A2,
                                           D=spectrum.disperser.D, shift_x=0., shift_y=0., shift_t=0.,
                                           angle=spectrum.rotation_angle,
                                           psf_poly_params=psf_poly_params)
    # Image model
    my_logger.info('\n\tImage model...')
    image.compute(star, background, spectrogram, starfield=starfield)
    image.add_poisson_noise()
    image.convert_to_ADU_units()
    if parameters.VERBOSE:
        image.plot_image(scale="log", title="Image simulation", target_pixcoords=target_pixcoords, units=image.units)
    # Set output path
    ensure_dir(outputdir)
    output_filename = image_filename.split('/')[-1]
    output_filename = (output_filename.replace('reduc', 'sim')).replace('trim', 'sim')
    output_filename = os.path.join(outputdir, output_filename)
    # Save images and parameters
    image.header['A1'] = A1
    image.header['A2'] = A2
    image.header['OZONE'] = ozone
    image.header['PWV'] = pwv
    image.header['VAOD'] = aerosols
    image.header['reso'] = reso
    image.header['ROTATION'] = int(with_rotation)
    image.header['STARS'] = int(with_stars)
    image.save_image(output_filename, overwrite=True)
    return image


if __name__ == "__main__":
    import os
    from spectractor.logbook import LogBook
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                        help="Write results in given output directory (default: ./outputs/).")
    parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                        help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose
    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = ['CTIODataJune2017_reduced_RG715_v2/data_30may17/reduc_20170530_134.fits']
    spectrum_file_name = 'outputs/reduc_20170530_134_spectrum.fits'
    # guess = [720, 670]
    # hologramme HoloAmAg
    psf_poly_params = [0.11298966008548948, -0.396825836448203, 0.2060387678061209, 2.0649268678546955,
                       -1.3753936625491252, 0.9242067418613167, 1.6950153822467129, -0.6942452135351901,
                       0.3644178350759512, -0.0028059253333737044, -0.003111527339787137, -0.00347648933169673,
                       528.3594585697788, 628.4966480821147, 12.438043546369354, 499.99999999999835]
    # file_name="../CTIOAnaJun2017/ana_31may17/OverScanRemove/trim_images/trim_20170531_150.fits"
    # guess = [840, 530]
    # target = "HD205905"

    logbook = LogBook(logbook=args.logbook)
    for file_name in file_names:
        tag = file_name.split('/')[-1]
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue

        image = ImageSim(file_name, spectrum_file_name, args.output_directory, A2=0.01,
                         psf_poly_params=psf_poly_params, with_stars=False)

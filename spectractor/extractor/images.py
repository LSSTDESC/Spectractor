from astropy.coordinates import Angle
from matplotlib import cm

from spectractor.extractor.targets import *
from spectractor.extractor.psf import *
from spectractor.extractor.dispersers import *

import pandas as pd
import re


class Image(object):

    def __init__(self, file_name, target="", disperser_label="",logbook=""):
        """
        The image class contains all the features necessary to load an image and extract a spectrum.

        Parameters
        ----------
        file_name: str
            The file name where the image is.
        target: str, optional
            The target name, to be found in data bases.
        disperser_label: str, optional
            The disperser label to load its properties

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> assert im.file_name == 'tests/data/reduc_20170605_028.fits'
        >>> assert im.data is not None and np.mean(im.data) > 0
        >>> assert im.stat_errors is not None and np.mean(im.stat_errors) > 0
        >>> assert im.header is not None
        >>> assert im.gain is not None and np.mean(im.gain) > 0
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.file_name = file_name
        self.units = 'ADU'
        self.expo = -1
        self.airmass = None
        self.date_obs = None
        self.disperser = None
        self.disperser_label = disperser_label
        self.filter = None
        self.filters = None
        self.header = None
        self.data = None
        self.data_rotated = None
        self.gain = None

        self.logbook = logbook  # the logbook is a hook to retrieve information missing in the Header
        self.adurate = False  # to tell if the image is in ADU/sec or not

        self.stat_errors = None
        self.stat_errors_rotated = None
        self.rotation_angle = 0
        self.parallactic_angle = None
        self.saturation = None
        self.load_image(file_name)
        # Load the target if given
        self.target = None
        self.target_pixcoords = None
        self.target_pixcoords_rotated = None
        self.target_bkgd2D = None
        self.target_star2D = None
        if target != "":
            self.target = load_target(target, verbose=parameters.VERBOSE)
            self.header['TARGET'] = self.target.label
            self.header.comments['TARGET'] = 'object targeted in the image'
            self.header['REDSHIFT'] = self.target.redshift
            self.header.comments['REDSHIFT'] = 'redshift of the target'
        self.err = None




    def load_image(self, file_name):
        """
        Load the image and store some information from header in class attributes.
        Then load the target and disperser properties. Called when an Image instance is created.

        Parameters
        ----------
        file_name: str
            The fits file name.

        """
        if parameters.OBS_NAME == 'CTIO':
            load_CTIO_image(self)
        elif parameters.OBS_NAME == 'LPNHE':
            load_LPNHE_image(self)
        elif parameters.OBS_NAME == 'PICDUMIDI':
            load_PDM_image(self)


        # Load the disperser
        self.my_logger.info(f'\n\tLoading disperser {self.disperser_label}...')
        self.disperser = Hologram(self.disperser_label, D=parameters.DISTANCE2CCD,
                                  data_dir=parameters.HOLO_DIR, verbose=parameters.VERBOSE)
        self.convert_to_ADU_rate_units()
        self.compute_statistical_error()
        self.header["FILTER2"]=self.disperser_label

    def save_image(self, output_file_name, overwrite=False):
        """Save the image in a fits file.

        Parameters
        ----------
        output_file_name: str
            The output file name.
        overwrite: bool
            If True, overwrite the file if it exists previously (default: False).

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> im.save_image('tests/data/reduc_20170605_028_copy.fits', overwrite=True)
        """
        save_fits(output_file_name, self.header, self.data, overwrite=overwrite)
        self.my_logger.info(f'\n\tImage saved in {output_file_name}')

    def convert_to_ADU_rate_units(self):
        """Convert Image data from ADU to ADU/s units.

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> print(im.expo)
        600.0
        >>> data_before = np.copy(im.data)
        >>> im.convert_to_ADU_rate_units()
        >>> assert np.all(np.isclose(data_before, im.data * im.expo))
        """
        if not self.adurate:
            self.data = self.data.astype(np.float64) / self.expo
            self.units = 'ADU/s'
            self.adurate=True

    def convert_to_ADU_units(self):
        """Convert Image data from ADU/s to ADU units.

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> print(im.expo)
        600.0
        >>> data_before = np.copy(im.data)
        >>> im.convert_to_ADU_rate_units()
        >>> data_after = np.copy(im.data)
        >>> im.convert_to_ADU_units()
        >>> assert np.all(np.isclose(data_before, im.data))
        """
        if self.adurate==True:
            self.data *= self.expo
            self.units = 'ADU'
            self.adurate = False

    def compute_statistical_error(self):
        """Compute the image noise map from Image.data as np.sqrt(data) / np.sqrt(gain * expo).

        The read out noise is not included

        """
        # removes the zeros and negative pixels first
        # set to minimum positive value
        data = np.copy(self.data)
        zeros = np.where(data <= 0)
        min_noz = np.min(data[np.where(data > 0)])
        data[zeros] = min_noz
        # compute poisson noise
        #   TODO: add read out noise (add in square to electrons)
        self.stat_errors = np.sqrt(data) / np.sqrt(self.gain * self.expo)

    def compute_parallactic_angle(self):
        """Compute the parallactic angle.

        Script from A. Guyonnet.
        """
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

    def plot_image(self, ax=None, scale="lin", title="", units="Image units", plot_stats=False,
                   target_pixcoords=None, figsize=[9.3, 8], aspect=None, vmin=None, vmax=None,
                   cmap="jet", cax=None):
        """Plot image.

        Parameters
        ----------
        ax: Axes, optional
            Axes instance (default: None).
        scale: str
            Scaling of the image (choose between: lin, log or log10) (default: lin)
        title: str
            Title of the image (default: "")
        units: str
            Units of the image to be written in the color bar label (default: "Image units")
        cmap: colormap
            Color map label (default: None)
        target_pixcoords: array_like, optional
            2D array  giving the (x,y) coordinates of the targets on the image: add a scatter plot (default: None)
        vmin: float
            Minimum value of the image (default: None)
        vmax: float
            Maximum value of the image (default: None)
        aspect: str
            Aspect keyword to be passed to imshow (default: None)
        cax: Axes, optional
            Color bar axes if necessary (default: None).
        figsize: tuple
            Figure size (default: [9.3, 8]).
        plot_stats: bool
            If True, plot the uncertainty map instead of the image (default: False).

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> im.plot_image(target_pixcoords=[820, 580])
        >>> if parameters.DISPLAY: plt.show()
        """
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        data = np.copy(self.data)
        if plot_stats:
            data = np.copy(self.stat_errors)
        plot_image_simple(ax, data=data, scale=scale, title=title, units=units, cax=cax,
                          target_pixcoords=target_pixcoords, aspect=aspect, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.legend()
        if parameters.DISPLAY:
            plt.show()





def load_CTIO_image(image):
    """Specific routine to load CTIO fits files and load their data and properties for Spectractor.

    Parameters
    ----------
    image: Image
        The Image instance to fill with file data and header.
    """
    image.my_logger.info(f'\n\tLoading CTIO image {image.file_name}...')
    image.header, image.data = load_fits(image.file_name)
    extract_info_from_CTIO_header(image, image.header)
    image.header['LSHIFT'] = 0.
    image.header['D2CCD'] = parameters.DISTANCE2CCD
    parameters.CCD_IMSIZE = int(image.header['XLENGTH'])
    parameters.CCD_PIXEL2ARCSEC = float(image.header['XPIXSIZE'])
    if image.header['YLENGTH'] != parameters.CCD_IMSIZE:
        image.my_logger.warning(
            f'\n\tImage rectangular: X={parameters.CCD_IMSIZE:d} pix, Y={image.header["YLENGTH"]:d} pix')
    if image.header['YPIXSIZE'] != parameters.CCD_PIXEL2ARCSEC:
        image.my_logger.warning('\n\tPixel size rectangular: X=%d arcsec, Y=%d arcsec' % (
            parameters.CCD_PIXEL2ARCSEC, image.header['YPIXSIZE']))
    image.coord = SkyCoord(image.header['RA'] + ' ' + image.header['DEC'], unit=(units.hourangle, units.deg),
                          obstime=image.header['DATE-OBS'])
    image.my_logger.info('\n\tImage loaded')
    # compute CCD gain map
    build_CTIO_gain_map(image)
    image.compute_parallactic_angle()


def build_CTIO_gain_map(image):
    """Compute the CTIO gain map according to header GAIN values.

    Parameters
    ----------
    image: Image
        The Image instance to fill with file data and header.
    """
    size = parameters.CCD_IMSIZE
    image.gain = np.zeros_like(image.data)
    # ampli 11
    image.gain[0:size // 2, 0:size // 2] = image.header['GTGAIN11']
    # ampli 12
    image.gain[0:size // 2, size // 2:size] = image.header['GTGAIN12']
    # ampli 21
    image.gain[size // 2:size, 0:size] = image.header['GTGAIN21']
    # ampli 22
    image.gain[size // 2:size, size // 2:size] = image.header['GTGAIN22']


def load_LPNHE_image(image):
    """Specific routine to load LPNHE fits files and load their data and properties for Spectractor.

    Parameters
    ----------
    image: Image
        The Image instance to fill with file data and header.
    """
    image.my_logger.info(f'\n\tLoading LPNHE image {image.file_name}...')
    image.header, data1 = load_fits(image.file_name, 15)
    image.header, data2 = load_fits(image.file_name, 7)
    data1 = data1.astype(np.float64)
    data2 = data2.astype(np.float64)
    image.data = np.concatenate((data1[10:-10, 10:-10], data2[10:-10, 10:-10]))
    image.date_obs = image.header['DATE-OBS']
    image.expo = float(image.header['EXPTIME'])
    image.header['ROTANGLE'] = image.rotation_angle
    image.header['LSHIFT'] = 0.
    image.header['D2CCD'] = parameters.DISTANCE2CCD
    image.data = image.data.T
    image.my_logger.info('\n\tImage loaded')
    # compute CCD gain map
    image.gain = float(image.header['CCDGAIN']) * np.ones_like(image.data)
    parameters.CCD_IMSIZE = image.data.shape[1]

def load_PDM_image(image):
    """Specific routine to load Pic du Midi fits files and load their data and properties for Spectractor.

    Parameters
    ----------
    image: Image
        The Image instance to fill with file data and header.
    """
    image.my_logger.info(f'\n\tLoading Pic du Midi image {image.file_name}...')

    # Now implement this for PDM

    image.header, image.data = load_fits(image.file_name)

    extract_info_from_PDM_header(image, image.header)

    image.header['LSHIFT'] = 0.
    image.header['D2CCD'] = parameters.DISTANCE2CCD
    image.header['EXPTIME']=image.header['EXPOSURE']
    image.header['FILTERS'] = image.header['FILTER']+' '+ image.header['DISP']
    image.header['FILTER1'] = image.header['FILTER']
    image.header['FILTER2'] = image.header['DISP']
    image.header['TARGET'] = image.header['OBJECT']

    # Check if image already in ADU per second
    # later, there will be a header KEY to know if already
    image.adurate = True
    image.units = 'ADU/s'



    # cumpute CCD gain map
    image.gain = float(image.header['CCDGAIN']) * np.ones_like(image.data)
    if parameters.CCD_IMSIZE != image.data.shape[1]:
        image.my_logger.warning('\n\tCCD_IMSIZE: =%d , image size =%d ' % (
            parameters.CCD_IMSIZE, image.data.shape[1]))

    #image.data = image.data.T
    image.my_logger.info('\n\tImage loaded')

    # retrieve RA and DEC from logbook
    load_LogBook(image)

    image.my_logger.warning(f'\n\tload_PDM_image :: coord: ra = {image.header["RA"]} hour ...')
    image.my_logger.warning(f'\n\tload_PDM_image :: coord: dec = {image.header["DEC"]} dec ...')
    image.my_logger.warning(f'\n\tload_PDM_image :: date: dec = {image.header["DATE-OBS"]} ...')



    image.coord = SkyCoord(str(image.header['RA']) + ' ' + str(image.header['DEC']),
                           unit=(units.hourangle, units.deg),
                           obstime=image.header['DATE-OBS'])

    image.my_logger.info('\n\tImage loaded')
    # compute CCD gain map
    image.gain = float(image.header['CCDGAIN']) * np.ones_like(image.data)

    image.compute_parallactic_angle()
    image.my_logger.warning(f'\n\tload_PDM_image :: paralactic angle = {image.parallactic_angle}  ...')


def load_LogBook(image):
    image.my_logger.info(f'\n\tLoad Logbook  {image.logbook} for {image.file_name}...')

    basefilename = os.path.basename(image.file_name)
    dirfilename = os.path.dirname(image.file_name)

    if re.search(".*20190214.*",dirfilename):
        SEL_DATE="20190214"
    else:
        SEL_DATE = "20190215"


    FLAG_ROTATION=False

    if re.search(".*rot.*", basefilename):
        FLAG_ROTATION=True
        image.my_logger.warning(f'\n\tload_PDM_image :: Image has been rotated !!!')


    image.my_logger.info(f'\n\tLoad Logbook  : date identified  {SEL_DATE} for {image.file_name}...')

    # temporary decoding of info from filename

    if SEL_DATE == "20190214":
        if FLAG_ROTATION:
            SearchTagRe_date = '^T1M_([0-9]+)_.*_red_rot[.]fit$'
            SearchTagRe_time = '^T1M_[0-9]+_([0-9]+)_.*_red_rot[.]fit$'
            SearchTagRe_num = '^T1M_[0-9]+_[0-9]+_([0-9]+)_.*_red_rot[.]fit$'
            SearchTagRe_obj = '^T1M_[0-9]+_[0-9]+_[0-9]+_(.*)-.*_red_rot[.]fit$'
            SearchTagRe_disp = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*-(.*)_Filtre.*_red_rot[.]fit$'
            SearchTagRe_filt = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*-.*_(Filtre.*)[.][0-9]+_red_rot[.]fit$'
            SearchTagRe_evnum = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*-.*_Filtre.*[.]([0-9]+)_red_rot[.]fit$'
    else:
            SearchTagRe_date = '^T1M_([0-9]+)_.*_red[.]fit$'
            SearchTagRe_time = '^T1M_[0-9]+_([0-9]+)_.*_red[.]fit$'
            SearchTagRe_num = '^T1M_[0-9]+_[0-9]+_([0-9]+)_.*_red[.]fit$'
            SearchTagRe_obj = '^T1M_[0-9]+_[0-9]+_[0-9]+_(.*)-.*_red[.]fit$'
            SearchTagRe_disp = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*-(.*)_Filtre.*_red[.]fit$'
            SearchTagRe_filt = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*-.*_(Filtre.*)[.][0-9]+_red[.]fit$'
            SearchTagRe_evnum = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*-.*_Filtre.*[.]([0-9]+)_red[.]fit$'

    # 20190215
    # No disperser
    if SEL_DATE == "20190215":
        if FLAG_ROTATION:
            SearchTagRe_date = '^T1M_([0-9]+)_.*_red_rot[.]fit$'
            SearchTagRe_time = '^T1M_[0-9]+_([0-9]+)_.*_red_rot[.]fit$'
            SearchTagRe_num = '^T1M_[0-9]+_[0-9]+_([0-9]+)_.*_red_rot[.]fit$'
            SearchTagRe_obj = '^T1M_[0-9]+_[0-9]+_[0-9]+_(.*)_Filtre.*_red_rot[.]fit$'
            SearchTagRe_disp = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*_(.*)_Filtre.*_red_rot[.]fit$'
            SearchTagRe_filt = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*_(Filtre.*)[.][0-9]+_red_rot[.]fit$'
            SearchTagRe_evnum = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*_Filtre.*[.]([0-9]+)_red_rot[.]fit$'
        else:
            SearchTagRe_date = '^T1M_([0-9]+)_.*_red[.]fit$'
            SearchTagRe_time = '^T1M_[0-9]+_([0-9]+)_.*_red[.]fit$'
            SearchTagRe_num = '^T1M_[0-9]+_[0-9]+_([0-9]+)_.*_red[.]fit$'
            SearchTagRe_obj = '^T1M_[0-9]+_[0-9]+_[0-9]+_(.*)_Filtre.*_red[.]fit$'
            SearchTagRe_disp = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*_(.*)_Filtre.*_red[.]fit$'
            SearchTagRe_filt = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*_(Filtre.*)[.][0-9]+_red[.]fit$'
            SearchTagRe_evnum = '^T1M_[0-9]+_[0-9]+_[0-9]+_.*_Filtre.*[.]([0-9]+)_red[.]fit$'





    tag_date=re.findall(SearchTagRe_date, basefilename)[0]
    tag_time=re.findall(SearchTagRe_time, basefilename)[0]
    tag_num=re.findall(SearchTagRe_num, basefilename)[0]
    tag_obj=re.findall(SearchTagRe_obj, basefilename)[0]


    if SEL_DATE == "20190214":
        tag_disp=re.findall(SearchTagRe_disp, basefilename)[0]
    else:
        tag_disp="HOE"
    tag_filt=re.findall(SearchTagRe_filt, basefilename)[0]
    tag_evnum=re.findall(SearchTagRe_evnum, basefilename)[0]

    image.my_logger.warning(f'\n\tLoad Logbook  :  tag_date = {tag_date}...')
    image.my_logger.warning(f'\n\tLoad Logbook  :  tag_time = {tag_time}...')
    image.my_logger.warning(f'\n\tLoad Logbook  :  tag_num = {tag_num}...')
    image.my_logger.warning(f'\n\tLoad Logbook  :  tag_obj = {tag_obj}...')
    image.my_logger.warning(f'\n\tLoad Logbook  :  tag_disp = {tag_disp}...')
    image.my_logger.warning(f'\n\tLoad Logbook  :  tag_filt = {tag_filt}...')
    image.my_logger.warning(f'\n\tLoad Logbook  :  tag_evnum = {tag_evnum}...')

    df=pd.read_csv(image.logbook)
    df_sel=df[df['file']==basefilename]
    image.my_logger.warning(f'\n\tLoad Logbook  :  selected-row {df_sel.iloc[0]}...')


    ra = df_sel.loc[df_sel.index[0],"ra"]
    dec = df_sel.loc[df_sel.index[0], "dec"]
    airmass=df_sel.loc[df_sel.index[0], "airmass"]

    image.my_logger.warning(f'\n\tLoad Logbook  : RA  = {ra} ...')
    image.my_logger.warning(f'\n\tLoad Logbook  : DEC = {dec} ...')
    image.my_logger.warning(f'\n\tLoad Logbook  : airmass = {airmass} ...')

    image.header['HA'] = ra
    image.header['RA'] = ra
    image.header['DEC'] = dec
    image.header['AIRMASS'] = airmass

    # now need to load disperser and filter
    image.disperser=df_sel.loc[df_sel.index[0], "disperser"]
    image.disperser_label = df_sel.loc[df_sel.index[0], "disperser"]
    image.filter=df_sel.loc[df_sel.index[0], "filt"]

    image.my_logger.warning(f'\n\tLoad Logbook  : disperser = {image.disperser} ...')
    image.my_logger.warning(f'\n\tLoad Logbook  : filter = {image.filter} ...')

    image.my_logger.warning(f'\n\tLoad Logbook : DONE ')





def find_target(image, guess, rotated=False):
    """Find the target in the Image instance.

    The object is search in a windows of size defined by the XWINDOW and YWINDOW parameters,
    using two iterative fits of a PSF2D model.
    User must give a guess array in the raw image.

    Parameters
    ----------
    image: Image
        The Image instance.
    guess: array_like
        Two parameter array giving the estimated position of the target in the image.
    rotated: bool
        If True, the target is searched in the rotated image.

    Returns
    -------
    x0: float
        The x position of the target.
    y0: float
        The y position of the target.
    """
    Dx = parameters.XWINDOW
    Dy = parameters.YWINDOW
    theX, theY = guess
    if rotated:
        angle = image.rotation_angle * np.pi / 180.
        rotmat = np.matrix([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        vec = np.array(image.target_pixcoords) - 0.5 * np.array(image.data.shape[::-1])
        guess2 = np.dot(rotmat, vec) + 0.5 * np.array(image.data_rotated.shape[::-1])
        x0 = int(guess2[0, 0])
        y0 = int(guess2[0, 1])
        guess = [x0, y0]
        Dx = parameters.XWINDOW_ROT
        Dy = parameters.YWINDOW_ROT
    niter = 2
    for i in range(niter):
        sub_image, x0, y0, Dx, Dy, sub_errors = find_target_init(image=image, guess=guess, rotated=rotated,
                                                                 widths=[Dx, Dy])
        # find the target
        avX, avY = find_target_2Dprofile(image, sub_image, guess, sub_errors=sub_errors)
        # compute target position
        theX = x0 - Dx + avX
        theY = y0 - Dy + avY
        guess = [int(theX), int(theY)]
        Dx = Dx // (i + 2)
        Dy = Dy // (i + 2)
    image.my_logger.info(f'\n\tX,Y target position in pixels: {theX:.3f},{theY:.3f}')
    if rotated:
        image.target_pixcoords_rotated = [theX, theY]
    else:
        image.target_pixcoords = [theX, theY]
        image.header['TARGETX'] = theX
        image.header.comments['TARGETX'] = 'target position on X axis'
        image.header['TARGETY'] = theY
        image.header.comments['TARGETY'] = 'target position on Y axis'
    return [theX, theY]


def find_target_init(image, guess, rotated=False, widths=[parameters.XWINDOW, parameters.YWINDOW]):
    """Initialize the search of the target cropping the image and setting the saturation level.

    Parameters
    ----------
    image: Image
        The Image instance.
    guess: array_like
        Two parameter array giving the estimated position of the target in the image.
    rotated: bool
        If True, the target is searched in the rotated image.
    widths: array_like
        Two parameter array to define the width of the cropped image (default: [parameters.XWINDOW, parameters.YWINDOW])

    Returns
    -------
    sub_image: array_like
        The cropped image data array where the fit has to be performed.
    x0: float
        The x position of the target.
    y0: float
        The y position of the target.
    Dx: int
        The width of the cropped image.
    Dy: int
        The height of the cropped image.
    sub_errors: array_like
        The cropped image uncertainty array where the fit has to be performed.
    """
    x0 = int(guess[0])
    y0 = int(guess[1])
    Dx, Dy = widths
    if rotated:
        sub_image = np.copy(image.data_rotated[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
        sub_errors = np.copy(image.stat_errors[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
    else:
        sub_image = np.copy(image.data[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
        sub_errors = np.copy(image.stat_errors[y0 - Dy:y0 + Dy, x0 - Dx:x0 + Dx])
    image.saturation = parameters.CCD_MAXADU / image.expo
    # sub_image = clean_target_spikes(sub_image, image.saturation)
    return sub_image, x0, y0, Dx, Dy, sub_errors


def find_target_1Dprofile(image, sub_image, guess):
    """
    Find precisely the position of the targeted object fitting a PSF1D model
    on each projection of the image along x and y, using outlier removal.

    Parameters
    ----------
    image: Image
        The Image instance.
    sub_image: array_like
        The cropped image data array where the fit is performed.
    guess: array_like
        Two parameter array giving the estimated position of the target in the image.

    Examples
    --------
    >>> im = Image('tests/data/reduc_20170605_028.fits')
    >>> guess = [820, 580]

    ..plot:
        im.plot_image(target_pixcoords=[820, 580])

    >>> parameters.DEBUG = True
    >>> sub_image, x0, y0, Dx, Dy, sub_errors = find_target_init(im, guess, rotated=False)
    >>> x1, y1 = find_target_1Dprofile(im, sub_image, guess)
    >>> assert np.isclose(x1, np.argmax(np.nansum(sub_image, axis=0)), rtol=1e-2)
    >>> assert np.isclose(y1, np.argmax(np.nansum(sub_image, axis=1)), rtol=1e-2)
    """
    NY, NX = sub_image.shape
    X = np.arange(NX)
    Y = np.arange(NY)
    # Mask aigrette
    saturated_pixels = np.where(sub_image >= image.saturation)
    if parameters.DEBUG:
        image.my_logger.info('\n\t%d saturated pixels: set saturation level to %d %s.' % (
            len(saturated_pixels[0]), image.saturation, image.units))
        # sub_image[sub_image >= 0.5*image.saturation] = np.nan
    # compute profiles
    profile_X_raw = np.sum(sub_image, axis=0)
    profile_Y_raw = np.sum(sub_image, axis=1)
    # fit and subtract smooth polynomial background
    # with 3sigma rejection of outliers (star peaks)
    bkgd_X, outliers = fit_poly1d_outlier_removal(X, profile_X_raw, order=2)
    bkgd_Y, outliers = fit_poly1d_outlier_removal(Y, profile_Y_raw, order=2)
    profile_X = profile_X_raw - bkgd_X(X)  # np.min(profile_X)
    profile_Y = profile_Y_raw - bkgd_Y(Y)  # np.min(profile_Y)
    avX, sigX = weighted_avg_and_std(X, profile_X ** 4)
    avY, sigY = weighted_avg_and_std(Y, profile_Y ** 4)
    if profile_X[int(avX)] < 0.8 * np.max(profile_X):
        image.my_logger.warning('\n\tX position determination of the target probably wrong')
    if profile_Y[int(avY)] < 0.8 * np.max(profile_Y):
        image.my_logger.warning('\n\tY position determination of the target probably wrong')
    if parameters.DEBUG:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        plot_image_simple(ax1, data=sub_image, scale="log", title="", units=image.units, target_pixcoords=[avX, avY])
        ax1.legend(loc=1)
        ax2.plot(X, profile_X_raw, 'r-', lw=2)
        ax2.plot(X, bkgd_X(X), 'g--', lw=2, label='bkgd')
        ax2.axvline(avX, color='b', linestyle='-', label='new', lw=2)
        ax2.grid(True)
        ax2.set_xlabel('X [pixels]')
        ax2.legend(loc=1)
        ax3.plot(Y, profile_Y_raw, 'r-', lw=2)
        ax3.plot(Y, bkgd_Y(Y), 'g--', lw=2, label='bkgd')
        ax3.axvline(avY, color='b', linestyle='-', label='new', lw=2)
        ax3.grid(True)
        ax3.set_xlabel('Y [pixels]')
        ax3.legend(loc=1)
        f.tight_layout()
        if parameters.DISPLAY:
            plt.show()
    return avX, avY


def find_target_2Dprofile(image, sub_image, guess, sub_errors=None):
    """
    Find precisely the position of the targeted object fitting a PSF2D model.
    A polynomial 2D background is subtracted first. Saturated pixels are masked with np.nan values.

    THE ERROR ARRAY IS NOT USED FOR THE MOMENT.

    Parameters
    ----------
    image: Image
        The Image instance.
    sub_image: array_like
        The cropped image data array where the fit is performed.
    guess: array_like
        Two parameter array giving the estimated position of the target in the image.
    sub_image: array_like
        The cropped image data array where the fit is performed.

    Examples
    --------
    >>> im = Image('tests/data/reduc_20170605_028.fits')
    >>> guess = [820, 580]

    ..plot:
        im.plot_image(target_pixcoords=[820, 580])

    >>> parameters.DEBUG = True
    >>> sub_image, x0, y0, Dx, Dy, sub_errors = find_target_init(im, guess, rotated=False)
    >>> xmax = np.argmax(np.sum(sub_image, axis=0))
    >>> ymax = np.argmax(np.sum(sub_image, axis=1))
    >>> x1, y1 = find_target_2Dprofile(im, sub_image, guess)
    >>> assert np.isclose(x1, xmax, rtol=1e-2)
    >>> assert np.isclose(y1, ymax, rtol=1e-2)
    """
    # TODO: replace with minuit and test on image _133.fits or decrease mean_prior
    # fit and subtract smooth polynomial background
    # with 3sigma rejection of outliers (star peaks)
    NY, NX = sub_image.shape
    XX = np.arange(NX)
    YY = np.arange(NY)
    Y, X = np.mgrid[:NY, :NX]
    bkgd_2D = fit_poly2d_outlier_removal(X, Y, sub_image, order=2, sigma=3)
    image.target_bkgd2D = bkgd_2D
    sub_image_subtracted = sub_image - bkgd_2D(X, Y)
    # find a first guess of the target position
    avX, sigX = weighted_avg_and_std(XX, np.sum(sub_image_subtracted, axis=0) ** 4)
    avY, sigY = weighted_avg_and_std(YY, np.sum(sub_image_subtracted, axis=1) ** 4)
    # fit a 2D star profile close to this position
    # guess = [np.max(sub_image_subtracted),avX,avY,1,1] #for Moffat2D
    # guess = [np.max(sub_image_subtracted),avX-2,avY-2,2,2,0] #for Gauss2D
    guess = [np.max(sub_image_subtracted), avX, avY, 1, 1, 0.1, 2, image.saturation]
    if image.target_star2D is not None:
        guess = fitting._model_to_fit_params(image.target_star2D)[0]
        guess[1] = avX
        guess[2] = avY
    mean_prior = 10  # in pixels
    # bounds = [ [0.5*np.max(sub_image_subtracted),avX-mean_prior,avY-mean_prior,0,-np.inf],
    # [2*np.max(sub_image_subtracted),avX+mean_prior,avY+mean_prior,np.inf,np.inf] ] #for Moffat2D
    # bounds = [ [0.5*np.max(sub_image_subtracted),avX-mean_prior,avY-mean_prior,0,0,0],
    # [100*image.saturation,avX+mean_prior,avY+mean_prior,10,10,np.pi] ] #for Gauss2D
    # bounds = [[0.5 * np.max(sub_image_subtracted), avX - mean_prior, avY - mean_prior, 2, 0.9 * image.saturation],
    # [10 * image.saturation, avX + mean_prior, avY + mean_prior, 15, 1.1 * image.saturation]]
    bounds = [[0.1 * np.max(sub_image_subtracted), avX - mean_prior, avY - mean_prior, 1, 0, -100, 1,
               0.9 * image.saturation],
              [10 * image.saturation, avX + mean_prior, avY + mean_prior, 30, 10, 200, 15, 1.1 * image.saturation]]
    saturated_pixels = np.where(sub_image >= image.saturation)
    if len(saturated_pixels[0]) > 0:
        if parameters.DEBUG:
            image.my_logger.info(
                f'\n\t{len(saturated_pixels[0])} saturated pixels: set saturation level '
                f'to {image.saturation} {image.units}.')
        sub_image_subtracted[sub_image >= 0.9 * image.saturation] = np.nan
        sub_image[sub_image >= 0.9 * image.saturation] = np.nan
    # fit
    bounds = list(np.array(bounds).T)
    # star2D = fit_PSF2D(X, Y, sub_image_subtracted, guess=guess, bounds=bounds)
    star2D = fit_PSF2D_minuit(X, Y, sub_image_subtracted, guess=guess, bounds=bounds)
    new_avX = star2D.x_mean.value
    new_avY = star2D.y_mean.value
    image.target_star2D = star2D
    # check target positions
    dist = np.sqrt((new_avY - avY) ** 2 + (new_avX - avX) ** 2)
    if dist > mean_prior / 2:
        image.my_logger.warning(
            f'\n\tX={new_avX:.2f}, Y={new_avY:.2f} target position determination probably wrong: '
            f'{dist:.1f} pixels from profile detection ({avX:.2f}, {avY:.2f})')
    # debugging plots
    if parameters.DEBUG:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        vmin = 0
        vmax = float(np.nanmax(sub_image))
        plot_image_simple(ax1, data=sub_image, scale="lin", title="", units=image.units,
                          target_pixcoords=[new_avX, new_avY], vmin=vmin, vmax=vmax)
        ax1.legend(loc=1)

        plot_image_simple(ax2, data=star2D(X, Y) + bkgd_2D(X, Y), scale="lin", title="",
                          units=f'Background+Star2D ({image.units})', vmin=vmin, vmax=vmax)
        plot_image_simple(ax3, data=sub_image - star2D(X, Y) - bkgd_2D(X, Y), scale="lin", title="",
                          units=f'Background+Star2D subtracted image\n({image.units})',
                          target_pixcoords=[new_avX, new_avY], vmin=vmin, vmax=vmax)
        ax3.legend(loc=1)

        f.tight_layout()
        if parameters.DISPLAY:
            plt.show()
    return new_avX, new_avY


def compute_rotation_angle_hessian(image, deg_threshold=10, width_cut=parameters.YWINDOW,
                                   right_edge=parameters.CCD_IMSIZE - 200,
                                   margin_cut=12):
    """Compute the rotation angle in degree of a spectrogram with the Hessian of the image.
    Use the disperser rotation angle map as a prior and the target_pixcoords values to crop the image
    around the spectrogram.

    Parameters
    ----------
    image: Image
        The Image instance.
    deg_threshold: float
        Don't consider pixel with Hessian angle above deg_threshold or below -deg_threshold (default: 10).
    width_cut: int
        Half with of the image to consider in height (default: parameters.YWINDOW).
    right_edge: int
        Maximum pixel on the right edge (default: parameters.CCD_IMSIZE - 200).
    margin_cut: int
        After computing the Hessian, to avoid bad values on the edges the function cut on the
        edge of image margin_cut pixels.

    Returns
    -------
    theta: float
        The median value of the histogram of angles deduced with the Hessian of the pixels (in degree).

    Examples
    --------
    >>> im=Image('tests/data/reduc_20170605_028.fits', disperser_label='HoloPhAg')

    # Create of False spectrogram:
    >>> N = parameters.CCD_IMSIZE
    >>> im.data = np.ones((N, N))
    >>> slope = -0.1
    >>> y = lambda x: slope * (x - 0.5*N) + 0.5*N
    >>> for x in np.arange(N):
    ...     im.data[int(y(x)), x] = 10
    ...     im.data[int(y(x))+1, x] = 10

    ..plot:
        plt.imshow(im.data, origin='lower)
        plt.show()

    >>> im.target_pixcoords=(N//2, N//2)
    >>> parameters.DEBUG = True
    >>> theta = compute_rotation_angle_hessian(im)
    >>> assert np.isclose(theta, np.arctan(slope)*180/np.pi, rtol=1e-2)
    """

    image.my_logger.info(f'\n\t compute_rotation_angle_hessian, width_cut={width_cut}....,deg_threshold={deg_threshold}......')

    x0, y0 = np.array(image.target_pixcoords).astype(int)
    # extract a region
    #if parameters.OBS_NAME == 'PICDUMIDI':
    #    data = np.copy(image.data[y0 - width_cut:parameters.CCD_IMSIZE - width_cut, 0:right_edge])
    #else:
    #    data = np.copy(image.data[y0 - width_cut:y0 + width_cut, 0:right_edge])

    # For rotated image, this should work !
    data = np.copy(image.data[y0 - width_cut:y0 + width_cut, 0:right_edge])


    image.my_logger.warning(
        f'\n\tcompute_rotation_angle_hessian :: call hessian_and_theta with margin_cut = {margin_cut}...')

    lambda_plus, lambda_minus, theta = hessian_and_theta(data, margin_cut)

    #image.my_logger.warning(f'\n\tcompute_rotation_angle_hessian :: hessian_and_theta found  theta = {theta}, lambda_plus={lambda_plus}, lambda_minus={lambda_minus} ...')

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
    theta_mask = theta_mask[2:-2, 2:-2]
    theta_hist = theta_mask[~np.isnan(theta_mask)].flatten()
    if parameters.OBS_OBJECT_TYPE != 'STAR':
        pixels = np.where(~np.isnan(theta_mask))
        p = np.polyfit(pixels[1], pixels[0], deg=1)
        theta_median = np.arctan(p[0]) * 180 / np.pi
    else:
        theta_median = float(np.median(theta_hist))

    theta_critical = 180. * np.arctan(20. / parameters.CCD_IMSIZE) / np.pi
    image.header['THETAFIT'] = theta_median
    image.header.comments['THETAFIT'] = '[USED] rotation angle from the Hessian analysis'
    image.header['THETAINT'] = theta_guess
    image.header.comments['THETAINT'] = 'rotation angle interp from disperser scan'

    if abs(theta_median - theta_guess) > theta_critical:
        image.my_logger.warning(
            f'\n\tInterpolated angle and fitted angle disagrees with more than 20 pixels !!!! '
            f'over {parameters.CCD_IMSIZE:d} pixels: {theta_median:.2f} vs {theta_guess:.2f} !!!!')

    if parameters.DEBUG:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        xindex = np.arange(data.shape[1])
        x_new = np.linspace(xindex.min(), xindex.max(), 50)
        y_new = width_cut + (x_new - x0) * np.tan(theta_median * np.pi / 180.)
        ax1.imshow(theta_mask, origin='lower', cmap=cm.jet, aspect='auto', vmin=-deg_threshold, vmax=deg_threshold)
        ax1.plot(x_new, y_new, 'b-')
        ax1.set_ylim(0, 2 * width_cut)
        ax1.set_xlabel('X [pixels]')
        ax1.set_xlabel('Y [pixels]')
        ax1.grid(True)
        n, bins, patches = ax2.hist(theta_hist, bins=int(np.sqrt(len(theta_hist))))
        ax2.plot([theta_median, theta_median], [0, np.max(n)])
        ax2.set_xlabel("Rotation angles [degrees]")
        if parameters.DISPLAY:
            plt.show()
    return theta_median


def turn_image(image):
    """Compute the rotation angle using the Hessian algorithm and turn the image.

    The results are stored in Image.data_rotated and Image.stat_errors_rotated.

    Parameters
    ----------
    image: Image
        The Image instance.

    Examples
    --------
    >>> im=Image('tests/data/reduc_20170605_028.fits', disperser_label='HoloPhAg')

    # Create of False spectrogram:
    >>> N = parameters.CCD_IMSIZE
    >>> im.data = np.ones((N, N))
    >>> slope = -0.1
    >>> y = lambda x: slope * (x - 0.5*N) + 0.5*N
    >>> for x in np.arange(N):
    ...     im.data[int(y(x)), x] = 10
    ...     im.data[int(y(x))+1, x] = 10

    ..plot:
        plt.imshow(im.data, origin='lower)
        plt.show()

    >>> im.target_pixcoords=(N//2, N//2)
    >>> parameters.DEBUG = True
    >>> turn_image(im)
    >>> assert im.data_rotated is not None
    >>> assert np.isclose(im.rotation_angle, np.arctan(slope)*180/np.pi, rtol=1e-2)
    """

    image.my_logger.info(f'\n\tturn_image width_cut={parameters.YWINDOW}...')



    image.rotation_angle = compute_rotation_angle_hessian(image, width_cut=parameters.YWINDOW,
                                                          right_edge=parameters.CCD_IMSIZE - 200)

    image.header['ROTANGLE'] = image.rotation_angle
    image.my_logger.info(f'\n\tRotate the image with angle theta={image.rotation_angle:.2f} degree')
    image.data_rotated = np.copy(image.data)
    if not np.isnan(image.rotation_angle):
        image.data_rotated = ndimage.interpolation.rotate(image.data, image.rotation_angle,
                                                          prefilter=parameters.ROT_PREFILTER,
                                                          order=parameters.ROT_ORDER)
        image.stat_errors_rotated = np.sqrt(
            np.abs(ndimage.interpolation.rotate(image.stat_errors ** 2, image.rotation_angle,
                                                prefilter=parameters.ROT_PREFILTER,
                                                order=parameters.ROT_ORDER)))
    if parameters.DEBUG:
        margin = 100
        y0 = int(image.target_pixcoords[1])
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=[8, 8])
        plot_image_simple(ax1, data=image.data[max(0, y0 - 2 * parameters.YWINDOW):min(y0 + 2 * parameters.YWINDOW,
                                                                                       image.data.shape[0]),
                                    margin:-margin],
                          scale="log", title='Raw image (log10 scale)', units=image.units,
                          target_pixcoords=(image.target_pixcoords[0] - margin, 2 * parameters.YWINDOW), aspect='auto',cmap="jet")
        ax1.plot([0, image.data.shape[0] - 2 * margin], [parameters.YWINDOW, parameters.YWINDOW], 'k-')
        plot_image_simple(ax2, data=image.data_rotated[max(0, y0 - 2 * parameters.YWINDOW):
                                                       min(y0 + 2 * parameters.YWINDOW, image.data.shape[0]),
                                    margin:-margin], scale="log", title='Turned image (log10 scale)',
                          units=image.units, target_pixcoords=image.target_pixcoords_rotated, aspect='auto',cmap="jet")
        ax2.plot([0, image.data_rotated.shape[0] - 2 * margin], [2 * parameters.YWINDOW, 2 * parameters.YWINDOW], 'k-')
        f.tight_layout()
        if parameters.DISPLAY:
            plt.show()


if __name__ == "__main__":
    import doctest

    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")

    doctest.testmod()

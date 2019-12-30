import numpy as np
import matplotlib.pyplot as plt

import spectractor.parameters as parameters
from spectractor.tools import fit_poly1d_outlier_removal, fit_poly2d_outlier_removal

from astropy.stats import SigmaClip
from photutils import Background2D, SExtractorBackground
from scipy.signal import medfilt2d
from scipy.interpolate import interp2d


def remove_image_background_sextractor(data, sigma=3.0, box_size=(50, 50), filter_size=(3, 3), positive=False):
    sigma_clip = SigmaClip(sigma=sigma)
    bkg_estimator = SExtractorBackground()
    bkg = Background2D(data, box_size, filter_size=filter_size,
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data_wo_bkg = data - bkg.background
    if positive:
        data_wo_bkg -= np.min(data_wo_bkg)
    if parameters.DEBUG:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(bkg.background, origin='lower')
        ax[1].imshow(np.log10(1 + data_wo_bkg), origin='lower')
        plt.show()
    return data_wo_bkg


def extract_spectrogram_background_fit1D(data, err, deg=1, ws=(20, 30), pixel_step=1, sigma=5):
    """
    Fit a polynomial background slice per slice along the x axis,
    with outlier removal, on lateral bands defined by the ws parameter.

    Parameters
    ----------
    data: array
        The 2D array image. The transverse profile is fitted on the y direction for all pixels along the x direction.
    err: array
        The uncertainties related to the data array.
    deg: int
        Degree of the polynomial model for the background (default: 1).
    ws: list
        up/down region extension where the sky background is estimated with format [int, int] (default: [20,30])
    pixel_step: int, optional
        The step in pixels between the slices to be fitted (default: 1).
        The values for the skipped pixels are interpolated with splines from the fitted parameters.
    sigma: int
        Sigma for outlier rejection (default: 5).

    Returns
    -------
    bgd_model_func: callable
        A 2D function to model the extracted background

    Examples
    --------

    # Build a mock spectrogram with random Poisson noise:
    >>> from spectractor.extractor.psf import ChromaticPSF1D
    >>> from spectractor import parameters
    >>> parameters.DEBUG = True
    >>> s0 = ChromaticPSF1D(Nx=100, Ny=100, saturation=1000)
    >>> params = s0.generate_test_poly_params()
    >>> saturation = params[-1]
    >>> data = s0.evaluate(params)
    >>> bgd = 10*np.ones_like(data)
    >>> data += bgd
    >>> data = np.random.poisson(data)
    >>> data_errors = np.sqrt(data+1)

    # Fit the transverse profile:
    >>> bgd_model = extract_spectrogram_background_fit1D(data, data_errors, deg=1, ws=[30,50], sigma=5, pixel_step=1)
    """
    Ny, Nx = data.shape
    middle = Ny // 2
    index = np.arange(Ny)
    # Prepare the fit
    bgd_index = np.concatenate((np.arange(0, middle - ws[0]), np.arange(middle + ws[0], Ny))).astype(int)
    pixel_range = np.arange(0, Nx, pixel_step)
    bgd_model = np.zeros_like(data).astype(float)
    # Fit for pixels in pixel_range array
    for x in pixel_range:
        # fit the background with a polynomial function
        bgd = data[bgd_index, x]
        bgd_fit, outliers = fit_poly1d_outlier_removal(bgd_index, bgd, order=deg, sigma=sigma, niter=2)
        bgd_model[:, x] = bgd_fit(index)
    # Filter the background model
    bgd_model = medfilt2d(bgd_model, kernel_size=[3, 9])
    bgd_model_func = interp2d(np.arange(Nx), index, bgd_model, kind='linear', bounds_error=False, fill_value=None)
    if parameters.DEBUG:
        fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex='all')
        bgd_bands = np.copy(data).astype(float)
        bgd_bands[middle - ws[0]:middle + ws[0], :] = np.nan
        im = ax[0].imshow(bgd_bands, origin='lower', aspect="auto", vmin=0)
        c = plt.colorbar(im, ax=ax[0])
        c.set_label(f'Data units (lin scale)')
        ax[0].set_title(f'Data background: mean={np.nanmean(bgd_bands):.3f}, std={np.nanstd(bgd_bands):.3f}')
        ax[0].set_xlabel('X [pixels]')
        ax[0].set_ylabel('Y [pixels]')
        x = np.arange(Nx)
        y = np.arange(Ny)
        # noinspection PyTypeChecker
        b = bgd_model_func(x, y)
        im = ax[1].imshow(b, origin='lower', aspect="auto")
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Y [pixels]')
        c2 = plt.colorbar(im, ax=ax[1])
        c2.set_label(f'Data units (lin scale)')
        ax[1].set_title(f'Fitted background: mean={np.mean(b):.3f}, std={np.std(b):.3f}')
        res = (bgd_bands - b) / err
        im = ax[2].imshow(res, origin='lower', aspect="auto", vmin=-5, vmax=5)
        ax[2].set_xlabel('X [pixels]')
        ax[2].set_ylabel('Y [pixels]')
        c2 = plt.colorbar(im, ax=ax[2])
        c2.set_label(f'Data units (lin scale)')
        ax[2].set_title(f'Pull: mean={np.nanmean(res):.3f}, std={np.nanstd(res):.3f}')
        fig.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
    return bgd_model_func


def extract_spectrogram_background_sextractor(data, err, ws=(20, 30), mask_signal_region=True):
    """
    Use photutils library median filter to estimate background behgin the sources.

    Parameters
    ----------
    data: array
        The 2D array image. The transverse profile is fitted on the y direction for all pixels along the x direction.
    err: array
        The uncertainties related to the data array.
    ws: list
        up/down region extension where the sky background is estimated with format [int, int] (default: [20,30])
    mask_signal_region: bool
        If True, the signal region is masked with np.nan values (default: True)

    Returns
    -------
    bgd_model_func: callable
        A 2D function to model the extracted background

    Examples
    --------

    # Build a mock spectrogram with random Poisson noise:
    >>> from spectractor.extractor.psf import ChromaticPSF1D
    >>> from spectractor import parameters
    >>> parameters.DEBUG = True
    >>> s0 = ChromaticPSF1D(Nx=100, Ny=100, saturation=1000)
    >>> params = s0.generate_test_poly_params()
    >>> saturation = params[-1]
    >>> data = s0.evaluate(params)
    >>> bgd = 10*np.ones_like(data)
    >>> data += bgd
    >>> data = np.random.poisson(data)
    >>> data_errors = np.sqrt(data+1)

    # Fit the transverse profile:
    >>> bgd_model = extract_spectrogram_background_sextractor(data, data_errors, ws=[30,50])
    """
    Ny, Nx = data.shape
    middle = Ny // 2
    # Estimate the background in the two lateral bands together
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = SExtractorBackground()
    if mask_signal_region:
        bgd_bands = np.copy(data).astype(float)
        bgd_bands[middle - ws[0]:middle + ws[0], :] = np.nan
        mask = (np.isnan(bgd_bands))
    else:
        mask = None
    # windows size in x is set to only 6 pixels to be able to estimate rapid variations of the background on real data
    # filter window size is set to window // 2 so 3
    bkg = Background2D(data, ((ws[1] - ws[0]), (ws[1] - ws[0])),
                       filter_size=((ws[1] - ws[0]) // 2, (ws[1] - ws[0]) // 2),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                       mask=mask)
    bgd_model_func = interp2d(np.arange(Nx), np.arange(Ny), bkg.background, kind='linear', bounds_error=False,
                              fill_value=None)

    if parameters.DEBUG:
        fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex='all')
        bgd_bands = np.copy(data).astype(float)
        bgd_bands[middle - ws[0]:middle + ws[0], :] = np.nan
        im = ax[0].imshow(bgd_bands, origin='lower', aspect="auto", vmin=0)
        c = plt.colorbar(im, ax=ax[0])
        c.set_label(f'Data units (lin scale)')
        ax[0].set_title(f'Data background: mean={np.nanmean(bgd_bands):.3f}, std={np.nanstd(bgd_bands):.3f}')
        ax[0].set_xlabel('X [pixels]')
        ax[0].set_ylabel('Y [pixels]')
        bkg.plot_meshes(outlines=True, color='#1f77b4', axes=ax[0])
        b = bkg.background
        im = ax[1].imshow(b, origin='lower', aspect="auto")
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Y [pixels]')
        c2 = plt.colorbar(im, ax=ax[1])
        c2.set_label(f'Data units (lin scale)')
        ax[1].set_title(f'Fitted background: mean={np.mean(b):.3f}, std={np.std(b):.3f}')
        res = (bgd_bands - b) / err
        im = ax[2].imshow(res, origin='lower', aspect="auto", vmin=-5, vmax=5)
        ax[2].set_xlabel('X [pixels]')
        ax[2].set_ylabel('Y [pixels]')
        c2 = plt.colorbar(im, ax=ax[2])
        c2.set_label(f'Data units (lin scale)')
        ax[2].set_title(f'Pull: mean={np.nanmean(res):.3f}, std={np.nanstd(res):.3f}')
        fig.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
    return bgd_model_func


def extract_spectrogram_background_poly2D(data, deg=1, ws=(20, 30), pixel_step=1, sigma=5):
    """
    Fit a 2D polynomial background with outlier removal, on lateral bands defined by the ws parameter.

    Parameters
    ----------
    data: array
        The 2D array image. The transverse profile is fitted on the y direction for all pixels along the x direction.
    deg: int
        Degree of the polynomial model for the background (default: 1).
    ws: list
        up/down region extension where the sky background is estimated with format [int, int] (default: [20,30])
    pixel_step: int, optional
        The step in pixels between the slices to be fitted (default: 1).
        The values for the skipped pixels are interpolated with splines from the fitted parameters.
    sigma: int
        Sigma for outlier rejection (default: 5).

    Returns
    -------
    bgd_model_func: callable
        A 2D function to model the extracted background

    Examples
    --------

    # Build a mock spectrogram with random Poisson noise:
    >>> from spectractor.extractor.psf import ChromaticPSF1D
    >>> from spectractor import parameters
    >>> parameters.DEBUG = True
    >>> s0 = ChromaticPSF1D(Nx=80, Ny=100, saturation=1000)
    >>> params = s0.generate_test_poly_params()
    >>> saturation = params[-1]
    >>> data = s0.evaluate(params)
    >>> bgd = 10.*np.ones_like(data)
    >>> xx, yy = np.meshgrid(np.arange(s0.Nx), np.arange(s0.Ny))
    >>> bgd += 1000*np.exp(-((xx-20)**2+(yy-10)**2)/(2*2))
    >>> data += bgd
    >>> data = np.random.poisson(data)
    >>> data_errors = np.sqrt(data+1)

    # Fit the transverse profile:
    >>> bgd_model_func = extract_spectrogram_background_poly2D(data, deg=1, ws=[30,50], sigma=5, pixel_step=1)
    """
    Ny, Nx = data.shape
    middle = Ny // 2
    # Prepare the fit
    bgd_index = np.concatenate((np.arange(0, middle - ws[0]), np.arange(middle + ws[0], Ny))).astype(int)
    pixel_range = np.arange(0, Nx, pixel_step)
    # Concatenate the background lateral bounds
    bgd_bands = data[bgd_index, :]
    bgd_bands = bgd_bands[:, pixel_range]
    bgd_bands = bgd_bands.astype(float)
    # Fit a 1 degree 2D polynomial function with outlier removal
    xx, yy = np.meshgrid(pixel_range, bgd_index)
    bgd_model_func = fit_poly2d_outlier_removal(xx, yy, bgd_bands, order=deg, sigma=sigma, niter=20)
    bgd_model_func = interp2d(xx, yy, bgd_model_func(xx, yy), kind='linear', bounds_error=False,
                              fill_value=None)

    if parameters.DEBUG:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex='all')
        bgd_bands = np.copy(data).astype(float)
        bgd_bands[middle - ws[0]:middle + ws[0], :] = np.nan
        im = ax[0].imshow(bgd_bands, origin='lower', aspect="auto", vmin=0)
        c = plt.colorbar(im, ax=ax[0])
        c.set_label(f'Data units (lin scale)')
        ax[0].set_title(f'Data background: mean={np.nanmean(bgd_bands):.3f}, std={np.nanstd(bgd_bands):.3f}')
        ax[0].set_xlabel('X [pixels]')
        ax[0].set_ylabel('Y [pixels]')
        # noinspection PyTypeChecker
        b = bgd_model_func(np.arange(Nx), np.arange(Ny))
        im = ax[1].imshow(b, origin='lower', aspect="auto")
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Y [pixels]')
        c2 = plt.colorbar(im, ax=ax[1])
        c2.set_label(f'Data units (lin scale)')
        ax[1].set_title(f'Fitted background: mean={np.mean(b):.3f}, std={np.std(b):.3f}')
        fig.tight_layout()
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()
    return bgd_model_func


if __name__ == "__main__":
    import doctest

    doctest.testmod()

import matplotlib as mpl
mpl.use('Agg')  # must be run first! But therefore requires noqa E402 on all other imports

from spectractor import parameters  # noqa: E402
from spectractor.astrometry import Astrometry  # noqa: E402
from spectractor.logbook import LogBook  # noqa: E402
from spectractor.config import load_config  # noqa: E402
from spectractor.tools import set_wcs_output_directory, set_wcs_file_name  # noqa: E402
from spectractor.extractor.images import Image, find_target  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import numpy as np  # noqa: E402
import unittest  # noqa: E402


# TODO: DM-33441 Fix broken spectractor tests
@unittest.skip('Skipping test for LSST testing framework')
def test_astrometry():
    file_names = ['tests/data/reduc_20170530_134.fits']  # 'tests/data/reduc_20170605_028.fits']

    load_config('./config/ctio.ini')
    logbook = LogBook(logbook='./tests/data/ctiofulllogbook_jun2017_v5.csv')
    parameters.VERBOSE = True
    parameters.DEBUG = False

    radius = 500
    maxiter = 10

    for file_name in file_names:
        wcs_output_directory = set_wcs_output_directory(file_name)
        if os.path.isdir(wcs_output_directory):
            subprocess.check_output(f"rm -rf {wcs_output_directory}", shell=True)
        tag = file_name.split('/')[-1].replace('sim', 'reduc')
        disperser_label, target_label, xpos, ypos = logbook.search_for_image(tag)
        if target_label is None or xpos is None or ypos is None:
            continue
        im = Image(file_name, target_label=target_label, disperser_label=disperser_label, config="ctio.ini")
        a = Astrometry(im)
        extent = ((int(max(0, xpos - radius)), int(min(xpos + radius, parameters.CCD_IMSIZE))),
                  (int(max(0, ypos - radius)), int(min(ypos + radius, parameters.CCD_IMSIZE))))
        gaia_min_residuals = a.run_full_astrometry(extent=extent, maxiter=maxiter)
        # checks
        assert os.path.isdir(wcs_output_directory)
        assert os.path.isfile(set_wcs_file_name(file_name))
        assert a.data is not None
        assert np.sum(a.data) > 1e-10
        assert gaia_min_residuals < 0.8
        # assert np.all(np.abs([dra_median, ddec_median]) < 1e-3)
        if file_name == 'tests/data/reduc_20170605_028.fits':
            assert len(a.sources) > 200
            assert np.isclose(a.target_radec_position_after_pm.ra.value, 224.97283917)
            assert np.isclose(a.target_radec_position_after_pm.dec.value, -54.30209)
            a.my_logger.warning(f"{a.wcs.wcs.crval}")
            assert np.isclose(a.wcs.wcs.crval[0], 224.9718998, atol=0.03)
            assert np.isclose(a.wcs.wcs.crval[1], -54.28912925, atol=0.03)
        if file_name == 'tests/data/sim_20170530_134.fits':
            im = Image(file_name, target_label=target_label)
            parameters.SPECTRACTOR_FIT_TARGET_CENTROID = "WCS"
            x0_wcs, y0_wcs = find_target(im, guess=[xpos, ypos], rotated=False)
            parameters.SPECTRACTOR_FIT_TARGET_CENTROID = "fit"
            x0, y0 = find_target(im, guess=[xpos, ypos], rotated=False)
            im.my_logger.warning(f"\n\tTrue {target_label} position: "
                                 f"{np.array([float(im.header['X0_T']), float(im.header['Y0_T'])])}"
                                 f"\n\tFound {target_label} position with WCS: {np.array([x0_wcs, y0_wcs])}"
                                 f"\n\tFound {target_label} position with 2D fit: {np.array([x0, y0])}")
            assert np.abs(x0_wcs - float(im.header['X0_T'])) < 0.5
            assert np.abs(y0_wcs - float(im.header['Y0_T'])) < 1


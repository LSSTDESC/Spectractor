from numpy.testing import run_module_suite

from spectractor import parameters
from spectractor.astrometry import Astrometry
from spectractor.logbook import LogBook
from spectractor.config import load_config
from spectractor.tools import set_wcs_output_directory, set_wcs_file_name
from spectractor.extractor.images import Image, find_target
import os
import subprocess
import numpy as np


def test_astrometry():
    file_names = ['tests/data/reduc_20170530_134.fits']  # 'tests/data/reduc_20170605_028.fits']

    load_config('./config/ctio.ini')
    logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
    parameters.VERBOSE = True
    parameters.DEBUG = True

    radius = 500
    maxiter = 10

    for file_name in file_names:
        wcs_output_directory = set_wcs_output_directory(file_name)
        if os.path.isdir(wcs_output_directory):
            subprocess.check_output(f"rm -rf {wcs_output_directory}", shell=True)
        tag = file_name.split('/')[-1].replace('sim', 'reduc')
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue
        a = Astrometry(file_name, target, disperser_label)
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
            im = Image(file_name, target_label=target)
            x0_wcs, y0_wcs = find_target(im, guess=[xpos, ypos], rotated=False, use_wcs=True)
            x0, y0 = find_target(im, guess=[xpos, ypos], rotated=False, use_wcs=False)
            im.my_logger.warning(f"\n\tTrue {target} position: "
                                 f"{np.array([float(im.header['X0_T']), float(im.header['Y0_T'])])}"
                                 f"\n\tFound {target} position with WCS: {np.array([x0_wcs, y0_wcs])}"
                                 f"\n\tFound {target} position with 2D fit: {np.array([x0, y0])}")
            assert np.abs(x0_wcs - float(im.header['X0_T'])) < 0.5
            assert np.abs(y0_wcs - float(im.header['Y0_T'])) < 1


if __name__ == "__main__":
    run_module_suite()

from numpy.testing import run_module_suite

from spectractor import parameters
from spectractor.astrometry import Astrometry
from spectractor.logbook import LogBook
from spectractor.config import load_config
import os
import subprocess
import numpy as np


def test_astrometry():
    file_names = ['tests/data/reduc_20170605_028.fits']
    if os.path.isfile('./tests/data/reduc_20170605_028_new.fits'):
        os.remove('./tests/data/reduc_20170605_028_new.fits')
    if os.path.isdir('./tests/data/reduc_20170605_028_wcs'):
        subprocess.Popen(["rm", "-rf",  "./tests/data/reduc_20170605_028_wcs"])

    load_config('./config/ctio.ini')
    logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
    parameters.DEBUG = True

    radius = 500
    maxiter = 10

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue
        a = Astrometry(file_name, target, disperser_label)
        a.run_simple_astrometry(extent=((xpos - radius, xpos + radius), (ypos - radius, ypos + radius)))
        # iterate process until shift is below 1 mas on RA and DEC directions
        # or maximum iterations is reached
        dra, ddec = 0, 0
        for i in range(maxiter):
            dra, ddec = a.run_gaia_astrometry()
            if dra < 1e-3 and ddec < 1e-3:
                break
        if parameters.DEBUG:
            a.plot_sources_and_gaia_catalog(sources=a.sources, gaia_coord=a.gaia_matches, margin=200)
            a.plot_astrometry_shifts(vmax=3)
        print(dra, ddec)
        # checks
        assert os.path.isdir('./tests/data/reduc_20170605_028_wcs')
        assert os.path.isfile('./tests/data/reduc_20170605_028_new.fits')
        assert a.data is not None
        assert np.sum(a.data) > 1e-10
        assert len(a.sources) > 400
        assert np.isclose(a.target_coord_after_motion.ra.value, 224.97283917)
        assert np.isclose(a.target_coord_after_motion.dec.value, -54.30209)
        assert np.isclose(a.wcs.wcs.crval[0], 224.9718998)
        assert np.isclose(a.wcs.wcs.crval[1], -54.28912925)
        assert np.all(np.isclose([dra, ddec], (0.000903019299029, -9.00223510558e-10), rtol=1e-3))


if __name__ == "__main__":
    run_module_suite()

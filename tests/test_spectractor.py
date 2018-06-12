from numpy.testing import run_module_suite

from spectractor.spectractor import parameters, Spectractor
from spectractor.logbook import LogBook
import os

def test_spectractor():
    file_names = ['../data/data_05jun17/reduc_20170605_028.fits']

    logbook = LogBook(logbook='../ctiofulllogbook_jun2017_v5.csv')
    parameters.VERBOSE = True

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue
        spectrum = Spectractor(file_name, './outputs/', [xpos, ypos], target)
        assert spectrum.data is not None
        assert os.path.isfile('./outputs/'+tag.replace('.fits','_spectrum.fits')) is True

if __name__ == "__main__":
    run_module_suite()
from numpy.testing import run_module_suite

from spectractor import parameters
from spectractor.extractor.extractor import Spectractor
from spectractor.logbook import LogBook
import os


def test_logbook():
    logbook = LogBook('./ctiofulllogbook_jun2017_v5.csv')
    #target, xpos, ypos = logbook.search_for_image('reduc_20170529_085.fits')
    #assert xpos is None
    disperser_label, target, xpos, ypos = logbook.search_for_image('reduc_20170603_020.fits')
    assert target == "PKS1510-089"
    assert xpos == 830
    assert ypos == 590
    #logbook = LogBook('./ctiofulllogbook_jun2017_v5.csv')
    #logbook.plot_columns_vs_date(['T', 'seeing', 'W'])


def test_extractor():
    file_names = ['tests/data/reduc_20170530_134_spectrum.fits']

    logbook = LogBook(logbook='./ctiofulllogbook_jun2017_v5.csv')
    parameters.DEBUG = True

    for file_name in file_names:
        tag = file_name.split('/')[-1]
        disperser_label, target, xpos, ypos = logbook.search_for_image(tag)
        if target is None or xpos is None or ypos is None:
            continue
        spectrum = Spectractor(file_name, './outputs/', [xpos, ypos], target, disperser_label,
                               config='./config/ctio.ini',
                               line_detection=True, atmospheric_lines=True)
        assert spectrum.data is not None
        assert os.path.isfile('./outputs/' + tag.replace('.fits', '_spectrum.fits')) is True


if __name__ == "__main__":
    run_module_suite()

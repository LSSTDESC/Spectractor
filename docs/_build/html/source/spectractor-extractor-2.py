import matplotlib.pyplot as plt
import numpy as np
from spectractor.extractor.images import Image
im = Image('tests/data/reduc_20170605_028.fits')
im.plot_image(target_pixcoords=[820, 580])
import matplotlib as mpl
mpl.use('Agg')  # must be run first! But therefore requires noqa E402 on all other imports

import pytest  # noqa: E402
import numpy as np  # noqa: E402
from photutils.datasets import make_4gaussians_image  # noqa: E402
from spectractor.extractor.background import make_source_mask  # noqa: E402
from photutils.utils.exceptions import NoDetectionsWarning  # noqa: E402


def test_make_source_mask():
    data = make_4gaussians_image()
    mask1 = make_source_mask(data, 5, 10)
    mask2 = make_source_mask(data, 5, 10, dilate_size=20)
    assert np.count_nonzero(mask2) > np.count_nonzero(mask1)

    with pytest.warns(NoDetectionsWarning):
        mask = make_source_mask(data, 100, 100)
        assert np.count_nonzero(mask) == 0


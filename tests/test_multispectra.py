import matplotlib as mpl
mpl.use('Agg')  # must be run first! But therefore requires noqa E402 on all other imports

from spectractor.fit.fit_multispectra import _build_test_sample, MultiSpectraFitWorkspace, run_multispectra_minimisation
from spectractor import parameters
import numpy as np


NSPECTRA = 4
OZONE = 300
PWV = 5
AEROSOLS = 0.05
LOG10A = -2

def test_multispectra():
    spectra = _build_test_sample(NSPECTRA, aerosols=AEROSOLS, ozone=OZONE, pwv=PWV, angstrom_exponent=10**LOG10A)
    parameters.VERBOSE = True

    nsigma = 3
    labels = ["VAOD", "LOG10A", "OZONE", "PWV"]
    truth = [AEROSOLS, LOG10A, OZONE, PWV]
    indices = [0, 1, 2, 3]
    for method in ["noprior", "spectrum"]:
        w = MultiSpectraFitWorkspace("./tests/data/multispectra_test", spectra, bin_width=10, verbose=True, fixed_deltas=True,
                                     fixed_A1s=False, amplitude_priors_method=method, fit_angstrom_exponent=True)
        run_multispectra_minimisation(w, method="newton", verbose=True, sigma_clip=10)

        ipar = w.params.get_free_parameters()  # non fixed param indices
        cov_indices = [list(ipar).index(k) for k in indices]  # non fixed param indices in cov matrix

        k = 0
        for i, l in zip(indices, labels):
            icov = cov_indices[k]
            w.my_logger.info(f"Test {l} best-fit {w.params.values[i]:.3f}+/-{np.sqrt(w.params.cov[icov, icov]):.3f} "
                                f"vs {truth[i]:.3f} at {nsigma}sigma level: "
                                f"{np.abs(w.params.values[i] - truth[i]) / np.sqrt(w.params.cov[icov, icov]) < nsigma}")
            assert np.abs(w.params.values[i] - truth[i]) / np.sqrt(w.params.cov[icov, icov]) < nsigma
            k += 1
        assert np.all(np.isclose(w.A1s, 1, atol=5e-3))


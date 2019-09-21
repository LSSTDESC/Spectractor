import matplotlib.pyplot as plt
import numpy as np
from spectractor.tools import gauss, compute_fwhm
x = np.arange(0, 100, 1)
stddev = 4
middle = 40
psf = gauss(x, 1, middle, stddev)
fwhm, half, center, a, b = compute_fwhm(x, psf, full_output=True)
plt.figure()
plt.plot(x, psf, label="function")
plt.axvline(center, color="gray", label="center")
plt.axvline(a, color="k", label="edges at half max")
plt.axvline(b, color="k", label="edges at half max")
plt.axhline(half, color="r", label="half max")
plt.legend()
plt.title(f"FWHM={fwhm:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
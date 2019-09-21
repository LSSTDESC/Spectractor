import matplotlib.pyplot as plt
plt.figure
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
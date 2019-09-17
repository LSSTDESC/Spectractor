import spectractor.parameters as parameters
from spectractor.tools import multigauss_and_bgd
parameters.CALIB_BGD_NPARAMS = 4
x = np.arange(600., 800., 1)
p = [20, 1, -1, -1, 20, 650, 3, 40, 750, 5]
y = multigauss_and_bgd(x, *p)
plt.plot(x,y,'r-')
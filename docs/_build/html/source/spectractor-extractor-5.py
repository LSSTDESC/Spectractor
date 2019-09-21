from spectractor.extractor.images import Image, turn_image
import spectractor.parameters as parameters
im=Image('tests/data/reduc_20170605_028.fits', disperser_label='HoloPhAg')
N = parameters.CCD_IMSIZE
im.data = np.ones((N, N))
slope = -0.1
y = lambda x: slope * (x - 0.5*N) + 0.5*N
for x in np.arange(N):
    im.data[int(y(x)), x] = 10
    im.data[int(y(x))+1, x] = 10

im.target_pixcoords=(N//2, N//2)
turn_image(im)
plt.imshow(im.data_rotated, origin='lower')
plt.show()
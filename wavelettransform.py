import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt

image = cv2.imread('segmented_img2.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to float for more resolution for use with pywt
image = np.float32(image)
image /= 255

# ...
# Do your processing
# ...

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(image, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
#fig = plt.figure(figsize=(3, 3))
#ax = fig.add_subplot(1, 1, 1)
plt.imshow(HH, interpolation="nearest", cmap=plt.cm.gray)
plt.axis('off')
'''
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
'''

plt.savefig('foo.png', transparent = True, bbox_inches='tight', pad_inches = 0)
plt.show()
'''
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    #cv2.imshow("Wavelet result",a)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #cv2.imwrite("wavelet_result.tif", a)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
'''


# Convert back to uint8 OpenCV format
image *= 255
image = np.uint8(image)

#cv2.imshow('image', image)
cv2.waitKey(0)

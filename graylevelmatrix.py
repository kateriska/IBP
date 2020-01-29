import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

img = cv2.imread('segmented_img2.tif', 0)

#gray = color.rgb2gray(img)
image = img_as_ubyte(img)

bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
inds = np.digitize(image, bins)

max_value = inds.max() + 1
matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
print(matrix_coocurrence)
# Charasterictics of Gray Level Matrix:
# contrast ***
contrast = greycoprops(matrix_coocurrence, 'contrast')
print("Contrast:")
print(contrast)
# dissimalirity
dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
print("Dissimilarity:")
print(dissimilarity)
# homogeinity ***
homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
print("Homogeneity:")
print(homogeneity)
# ASM
asm = greycoprops(matrix_coocurrence, 'ASM')
print("ASM:")
print(asm)
# Energy ***
energy = greycoprops(matrix_coocurrence, 'energy')
print("Energy:")
print(energy)
# Correlation ***
correlation = greycoprops(matrix_coocurrence, 'correlation')
print("Correlation:")
print(correlation)

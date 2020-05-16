# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: WaveletShow.py - shows approximation, horizontal, vertical and diagonal detail of fingerprint after processing with Wavelet transform

import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import processedSegmentation

# function for showing horizontal, vertical, diagonal detail after processing with Wavelet transform and their approximation
def showWavelet(segmentation_type, input_img, wavelet_type):
    if (segmentation_type != "none"):
        img = cv2.imread(input_img, 0) # uint8 image in grayscale
    else:
        img = cv2.imread(input_img)

    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image

    # choose type of segmentation
    if (segmentation_type == "otsu"):
        segmented_img = processedSegmentation.imgSegmentation(img)
    elif (segmentation_type == "gauss"):
        segmented_img = processedSegmentation.adaptiveSegmentationGaussian(img)
    elif (segmentation_type == "mean"):
        segmented_img = processedSegmentation.adaptiveSegmentationMean(img)

    if (segmentation_type != "none"):
        cv2.imwrite('./processedImg/segmented_img.jpg', segmented_img)
        img = cv2.imread('./processedImg/segmented_img.jpg')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float32 for more resolution for use with pywt
    img = np.float32(img)
    img /= 255

    titles = ['Approximation', ' Horizontal detail (LH)',
          'Vertical detail (HL)', 'Diagonal detail (HH)']
    coeffs2 = pywt.dwt2(img, wavelet_type)
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(11, 11))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()
    return

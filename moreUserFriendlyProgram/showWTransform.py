# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: showWTransform.py - shows approximation, horizontal, vertical and diagonal detail of fingerprint

import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

#print(pywt.wavelist(kind='discrete'))
# function for segmentation of fingerprint with using Otsu tresholding
def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200);
    result_tresh = cv2.add(tresh_img, opening)
    result_orig = cv2.add(img, opening) # add mask with input image
    return result_orig

# function for segmentation of fingerprint with using adaptive Gaussian tresholding
def adaptiveSegmentationGaussian(img):
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,4)
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel) # use morphological operations
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200);
    result = cv2.add(img, opening) # add mask with input image
    return result

# function for segmentation of fingerprint with using adaptive Mean tresholding
def adaptiveSegmentationMean(img):
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,7)
    kernel = np.ones((24,24), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN,kernel) # use morphological operations
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200);
    result = cv2.add(img, opening) # add mask with input image
    return result

def showWavelet(segmentation_type, input_img, wavelet_type):
    if (segmentation_type != "none"):
        img = cv2.imread(input_img, 0) # uint8 image in grayscale
    else:
        img = cv2.imread(input_img)

    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image

    # choose type of segmentation
    if (segmentation_type == "otsu"):
        segmented_img = imgSegmentation(img)
    elif (segmentation_type == "gauss"):
        segmented_img = adaptiveSegmentationGaussian(img)
    elif (segmentation_type == "mean"):
        segmented_img = adaptiveSegmentationMean(img)

    if (segmentation_type != "none"):
        cv2.imwrite('segmented_img.jpg', segmented_img)
        img = cv2.imread('segmented_img.jpg')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float32 for more resolution for use with pywt
    img = np.float32(img)
    img /= 255

    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
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

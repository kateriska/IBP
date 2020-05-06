# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: processedSegmentation.py - process segmentation of fingerprint with using Otsu, Gaussian or Mean thresholding

import numpy as np
import cv2

# function for segmentation of fingerprint with using Otsu tresholding
def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200)
    result_tresh = cv2.add(tresh_img, opening)
    result_orig = cv2.add(img, opening) # add mask with input image
    return result_orig

# function for segmentation of fingerprint with using adaptive Gaussian tresholding
def adaptiveSegmentationGaussian(img):
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,4)
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel) # use morphological operations
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.Canny(opening, 100, 200)
    result = cv2.add(img, opening) # add mask with input image
    return result

# function for segmentation of fingerprint with using adaptive Mean tresholding
def adaptiveSegmentationMean(img):
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,7)
    kernel = np.ones((24,24), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN,kernel) # use morphological operations
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.Canny(opening, 100, 200)
    result = cv2.add(img, opening) # add mask with input image
    return result

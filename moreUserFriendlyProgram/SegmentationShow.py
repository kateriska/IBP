# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: SegmentationShow.py - shows segmentation of fingerprint with using of various methods of tresholding

import cv2
import numpy as np
from matplotlib import pyplot as plt

# function for segmentation of fingerprint with using Otsu tresholding
def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("Tresholded image", tresh_img) # show tresholded image

    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200);
    cv2.imshow("Segmented mask of image", opening) # show segmented mask of image
    result_orig = cv2.add(img, opening) # add mask with input image
    return result_orig

# function for segmentation of fingerprint with using adaptive Gaussian tresholding
def adaptiveSegmentationGaussian(img):
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,4)
    cv2.imshow("Tresholded image", th3) # show tresholded image
    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel) # use morphological operations
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.Canny(opening, 100, 200);
    cv2.imshow("Segmented mask of image", opening) # show segmented mask of image
    result = cv2.add(img, opening) # add mask with input image
    return result

# function for segmentation of fingerprint with using adaptive Mean tresholding
def adaptiveSegmentationMean(img):
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,7)
    cv2.imshow("Tresholded image", th2) # show tresholded image
    # noise removal
    kernel = np.ones((24,24), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN,kernel)  # use morphological operations
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.Canny(opening, 100, 200);
    cv2.imshow("Segmented mask of image", opening) # show segmented mask of image
    result = cv2.add(img, opening) # add mask with input image
    return result


def showSegmentation(segmentation_type, input_img):
    img = cv2.imread(input_img,0) # read grayscale image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    cv2.imshow("Normalized image", img) # show normalized image

    if (segmentation_type == "otsu"):
        segmented_img = imgSegmentation(img)
    elif (segmentation_type == "gauss"):
        segmented_img = adaptiveSegmentationGaussian(img)
    elif (segmentation_type == "mean"):
        segmented_img = adaptiveSegmentationMean(img)

    cv2.imshow("Segmented image", segmented_img) # show final segmented image
    cv2.waitKey(0)
    return

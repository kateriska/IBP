# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: SegmentationShow.py - shows segmentation of fingerprint with using of various methods of thresholding

import cv2
import numpy as np
from matplotlib import pyplot as plt

# function for segmentation of fingerprint with using Otsu thresholding
def imgSegmentation(img, results):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results.append(tresh_img)
    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations

    cv2.Canny(opening, 100, 200)
    results.append(opening)
    result_orig = cv2.add(img, opening) # add mask with input image
    results.append(result_orig)
    return result_orig

# function for segmentation of fingerprint with using adaptive Gaussian tresholding
def adaptiveSegmentationGaussian(img, results):
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,4)
    results.append(th3)
    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel) # use morphological operations

    cv2.Canny(opening, 100, 200)
    results.append(opening)
    result = cv2.add(img, opening) # add mask with input image
    results.append(result)
    return result

# function for segmentation of fingerprint with using adaptive Mean tresholding
def adaptiveSegmentationMean(img, results):
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,7)
    results.append(th2)
    # noise removal
    kernel = np.ones((24,24), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN,kernel)  # use morphological operations

    cv2.Canny(opening, 100, 200)
    results.append(opening)
    result = cv2.add(img, opening) # add mask with input image
    results.append(result)
    return result

# function for showing original normalized image, thresholded image, extracted mask and final segmented image
def showSegmentationProcess(results):
    titles = ['Original normalized image', 'Thresholded image',
          'Extracted mask', 'Final segmented image']

    fig = plt.figure(figsize=(11, 11))
    i = 0
    for a in results:
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(a, cmap='gray')
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        i = i + 1

    fig.tight_layout()
    plt.show()

# function for reading input image and processing segmentation with asked thresholding type
def showSegmentation(segmentation_type, input_img):
    img = cv2.imread(input_img,0) # read grayscale image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

    results = list()
    results.append(img)

    if (segmentation_type == "otsu"):
        segmented_img = imgSegmentation(img, results)
    elif (segmentation_type == "gauss"):
        segmented_img = adaptiveSegmentationGaussian(img, results)
    elif (segmentation_type == "mean"):
        segmented_img = adaptiveSegmentationMean(img, results)

    showSegmentationProcess(results)
    cv2.waitKey(0)
    return

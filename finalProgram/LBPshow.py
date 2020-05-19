# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: LBPshow.py - shows processed fingerprint with LBP

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import pywt
import processedSegmentation

# function for processing every neighbour pixel with LBP
def LBPprocesspixel(img, pix5, x, y):

    pixel_new_value = 0 # init the variable before try block

    try:
        if (img[x][y] >= pix5):
            pixel_new_value = 1
        else:
            pixel_new_value = 0

    except:
        pass

    return pixel_new_value

# function for processing LBP and gain new value for center pixel
def processLBP(img, x, y, lbp_values):
    # 3x3 window of pixels, where center pixel pix5 is on position [x,y]

    '''
            +------+------+------+
            | pix7 | pix8 | pix9 |
            +------+------+------+
    y-axis  | pix4 | pix5 | pix6 |
            +------+------+------+
            | pix1 | pix2 | pix3 |
            +------+------+------+

                    x-axis
    '''

    value_dec = 0 # init variable for computing the final new decimal value for center pixel

    pix5 = img[x][y] # center pixel on position [x,y]

    # process the all neighbour pixels and receive 8-bit binary code
    pix8 = LBPprocesspixel(img, pix5, x, y+1) # LSB
    pix7 = LBPprocesspixel(img, pix5, x-1, y+1)
    pix4 = LBPprocesspixel(img, pix5, x-1, y)
    pix1 = LBPprocesspixel(img, pix5, x-1, y-1)
    pix2 = LBPprocesspixel(img, pix5, x, y-1)
    pix3 = LBPprocesspixel(img, pix5, x+1, y-1)
    pix6 = LBPprocesspixel(img, pix5, x+1, y)
    pix9 = LBPprocesspixel(img, pix5, x+1, y+1) # MSB

    # compute new decimal value for center pixel - convert binary code to decimal number
    value_dec = (pix9 * 2 ** 7) + (pix6 * 2 ** 6) + (pix3 * 2 ** 5) + (pix2 * 2 ** 4) + (pix1 * 2 ** 3) + (pix4 * 2 ** 2) + (pix7 * 2 ** 1) + (pix8 * 2 ** 0)

    lbp_values.append(value_dec) # append new decimal value of pixel to array of whole processed lbp image
    return value_dec

def showLBP(segmentation_type, input_img):
    if (segmentation_type != "none"):
        img = cv2.imread(input_img, 0) # uint8 image in grayscale
    else:
        img = cv2.imread(input_img)

    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image

    if (segmentation_type == "otsu"):
        segmented_img = processedSegmentation.imgSegmentation(img)
    elif (segmentation_type == "gauss"):
        segmented_img = processedSegmentation.adaptiveSegmentationGaussian(img)
    elif (segmentation_type == "mean"):
        segmented_img = processedSegmentation.adaptiveSegmentationMean(img)

    if (segmentation_type != "none"):
        cv2.imwrite('./processedImg/segmented_img.jpg', segmented_img)
        img = cv2.imread('./processedImg/segmented_img.jpg')

    height, width, channel = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_image = np.zeros((height, width,3), np.uint8)

# processing LBP algorithm
    lbp_values = []
    for i in range(0, height):
        for j in range(0, width):
            lbp_image[i, j] = processLBP(img_gray, i, j, lbp_values)

    hist_lbp = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])

    # show LBP image and LBP histogram
    figure = plt.figure(figsize=(30, 30))
    image_plot = figure.add_subplot(1,2,1)
    image_plot.imshow(lbp_image)
    image_plot.set_xticks([])
    image_plot.set_yticks([])
    image_plot.set_title("LBP image", fontsize=10)
    current_plot = figure.add_subplot(1, 2, 2)
    current_plot.plot(hist_lbp, color = (0, 0, 0.2))

    current_plot.set_xlim([0,256])
    current_plot.set_ylim([0,6000])
    current_plot.set_title("LBP histogram", fontsize=10)
    current_plot.set_xlabel("Intensity")
    current_plot.set_ylabel("Count of pixels")
    ytick_list = [int(i) for i in current_plot.get_yticks()]
    current_plot.set_yticklabels(ytick_list,rotation = 90)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

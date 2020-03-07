import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import pywt

### IMAGE SEGMENTATION WITH MORPHOLOGY OPERATIONS
def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel)
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200);
    result_tresh = cv2.add(tresh_img, opening)
    result_orig = cv2.add(img, opening)
    return result_orig

def adaptiveSegmentationGaussian(img):
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,4)
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel)
    #cv2.imshow('Opening', opening)
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(opening, contours, -1, (0,255,0), 3)

    cv2.Canny(opening, 100, 200);
    #cv2.imshow('Opening with contours', opening)
    result = cv2.add(img, opening)
    #cv2.imshow('Img after noise removal', result)
    #cv2.imwrite("segment.tif", result)
    return result

def adaptiveSegmentationMean(img):
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,7)
    kernel = np.ones((24,24), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN,kernel)
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.Canny(opening, 100, 200);
    #cv2.imshow('Opening with contours', opening)
    result = cv2.add(img, opening)
    return result

### LBP ALGORITHM
def LBPprocesspixel(img, pix5, x, y):
    new_value = 0
    
    try:
        if (img[x][y] >= pix5):
            new_value = 1
        else:
            new_value = 0

    except:
        pass

    return new_value

def processLBP(img, x, y, lbp_values):
    '''
     pix7 | pix8 | pix9
    ----------------
     pix4 | pix5 | pix6
    ----------------
     pix1 | pix2 | pix3
    '''
    pix5 = img[x][y]
    pixel_new_value = []
    pixel_new_value.append(LBPprocesspixel(img, pix5, x, y+1))     #pix8
    pixel_new_value.append(LBPprocesspixel(img, pix5, x-1, y+1))   #pix7
    pixel_new_value.append(LBPprocesspixel(img, pix5, x-1, y))     #pix4
    pixel_new_value.append(LBPprocesspixel(img, pix5, x-1, y-1))   #pix1
    pixel_new_value.append(LBPprocesspixel(img, pix5, x, y-1))     #pix2
    pixel_new_value.append(LBPprocesspixel(img, pix5, x+1, y-1))   #pix3
    pixel_new_value.append(LBPprocesspixel(img, pix5, x+1, y))     #pix6
    pixel_new_value.append(LBPprocesspixel(img, pix5, x+1, y+1))   #pix9

    powers_bin = [1, 2, 4, 8, 16, 32, 64, 128]
    value_dec = 0
    for i in range(len(pixel_new_value)):
        value_dec = value_dec + (pixel_new_value[i] * powers_bin[i])

    #print(value_dec)
    lbp_values.append(value_dec)
    return value_dec

def showLBP(segmentation_type, input_img):
    if (segmentation_type != "none"):
        img = cv2.imread(input_img, 0) # uint8 image in grayscale
    else:
        img = cv2.imread(input_img)
#img = cv2.resize(img,(400,400)) # resize of image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
#segmented_img = adaptiveSegmentationMean(img)
    if (segmentation_type == "otsu"):
        segmented_img = imgSegmentation(img)
    elif (segmentation_type == "gauss"):
        segmented_img = adaptiveSegmentationGaussian(img)
    elif (segmentation_type == "mean"):
        segmented_img = adaptiveSegmentationMean(img)
#cv2.imshow('Segmented image', segmented_img)
    if (segmentation_type != "none"):
        cv2.imwrite('segmented_img.jpg', segmented_img)
        img = cv2.imread('segmented_img.jpg')

    height, width, channel = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_image = np.zeros((height, width,3), np.uint8)

# processing LBP algorithm
    lbp_values = []
    for i in range(0, height):
        for j in range(0, width):
            lbp_image[i, j] = processLBP(img_gray, i, j, lbp_values)

    cv2.imshow("lbp image", lbp_image)
    hist_lbp = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])


    figure = plt.figure()
    current_plot = figure.add_subplot(1, 1, 1)
    current_plot.plot(hist_lbp, color = (0, 0, 0.2))
#current_plot.set_xlim([0,260])
    current_plot.set_xlim([0,250])
    current_plot.set_ylim([0,6000])
    current_plot.set_title("LBP Histogram")
    current_plot.set_xlabel("Intensity")
    current_plot.set_ylabel("Count of pixels")
    ytick_list = [int(i) for i in current_plot.get_yticks()]
    current_plot.set_yticklabels(ytick_list,rotation = 90)
    plt.show()





    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

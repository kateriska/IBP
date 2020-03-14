import cv2
import numpy as np
from matplotlib import pyplot as plt



def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("Treshold img", tresh_img)

    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel)
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200);
    result_tresh = cv2.add(tresh_img, opening)
    cv2.imshow("Mask", result_tresh)
    result_orig = cv2.add(img, opening)
    return result_orig

def adaptiveSegmentationGaussian(img):
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,4)
    cv2.imshow("Treshold img", th3)
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
    cv2.imshow("Treshold img", th2)
    kernel = np.ones((24,24), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN,kernel)
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.Canny(opening, 100, 200);
    #cv2.imshow('Opening with contours', opening)
    result = cv2.add(img, opening)
    return result


def showSegmentation(segmentation_type, input_img):
    img = cv2.imread(input_img,0)
    cv2.imshow("Grayscale img", img)
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    cv2.imshow("Normalized img", img)
# converting to gray scale
    if (segmentation_type == "otsu"):
        segmented_img = imgSegmentation(img)
    elif (segmentation_type == "gauss"):
        segmented_img = adaptiveSegmentationGaussian(img)
    elif (segmentation_type == "mean"):
        segmented_img = adaptiveSegmentationMean(img)

    cv2.imshow("Segmented img", segmented_img)
    cv2.waitKey(0)
    return

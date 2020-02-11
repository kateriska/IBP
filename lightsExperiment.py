import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte, img_as_float

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


# IMAGES FOR TRAINING
path = '/home/katerina/Documents/IBP/trainingNew/*'

f = open("LightsTrained.csv","w+")
g = open("LightsresultTrained.csv","w+")
for file in glob.glob(path):
    img = cv2.imread(file, 0) # uint8 image in grayscale
    #img = cv2.resize(img,(400,400)) # resize of image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
    segmented_img = imgSegmentation(img)
    #cv2.imshow('Segmented image', segmented_img)
    cv2.imwrite('segmented_img.jpg', segmented_img)
    img_segmented = cv2.imread('segmented_img.jpg')

    image = img_as_float(img_segmented)
    pixel_mean = np.mean(image)
    pixel_std = np.std(image)
    pixel_average = np.average(image)
    pixel_var = np.var(image)
    # pixel intensity arithmetic mean
    print(np.mean(image))
    #  pixel intensity standard deviation
    print(np.std(image))
    print(np.average(image))

    print(np.var(image))

    file_substr = file.split('/')[-1]
    print(file_substr)
    # processing LBP algorithm
    hist = cv2.calcHist([img_segmented], [0], None, [256], [0, 256])
    histogram_list = list()
    histogram_list = hist
    sum_hist1 = 0
    sum_hist2 = 0
    sum_hist3 = 0
    sum_hist4 = 0
    # get values from histogram
    for i in range(0, 64):
        hist_value = histogram_list[i]
        hist_value_n = str(hist_value[0])
        sum_hist1 = sum_hist1 + float(hist_value_n)

    for i in range(64, 128):
        hist_value = histogram_list[i]
        hist_value_n = str(hist_value[0])
        sum_hist2 = sum_hist2 + float(hist_value_n)

    for i in range(128, 192):
        hist_value = histogram_list[i]
        hist_value_n = str(hist_value[0])
        sum_hist3 = sum_hist3 + float(hist_value_n)

    for i in range(192, 256):
        hist_value = histogram_list[i]
        hist_value_n = str(hist_value[0])
        sum_hist4 = sum_hist4 + float(hist_value_n)

    sum_hist1_div = sum_hist1 / 1000000
    sum_hist2_div = sum_hist2 / 1000000
    sum_hist3_div = sum_hist3 / 1000000
    sum_hist4_div = sum_hist4  / 1000000


    saved_str = str(pixel_mean) + ", " + str(pixel_std) + ", " + str(pixel_average) + ", " + str(pixel_var) + "\n"
    #print("[" + str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "],")
    f.write(saved_str)


    if ("fake" in file_substr):
        print("This is FAKE image.")
        g.write("0\n")
    elif ("Images" in file_substr):
        print("This is LIVE image.")
        g.write("1\n")

    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()


# IMAGES FOR TESTING
path_testing = '/home/katerina/Documents/IBP/testingNew/*'

h = open("LightsTested.csv","w+")
for file in glob.glob(path_testing):
    img = cv2.imread(file, 0) # uint8 image in grayscale
    #img = cv2.resize(img,(400,400)) # resize of image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
    segmented_img = imgSegmentation(img)
    #cv2.imshow('Segmented image', segmented_img)
    cv2.imwrite('segmented_img.jpg', segmented_img)
    img_segmented = cv2.imread('segmented_img.jpg')

    image = img_as_float(img_segmented)
    pixel_mean = np.mean(image)
    pixel_std = np.std(image)
    pixel_average = np.average(image)
    pixel_var = np.var(image)
    # pixel intensity arithmetic mean
    print(np.mean(image))
    #  pixel intensity standard deviation
    print(np.std(image))
    print(np.average(image))

    print(np.var(image))

    file_substr = file.split('/')[-1]
    print(file_substr)
    # processing LBP algorithm
    hist = cv2.calcHist([img_segmented], [0], None, [256], [0, 256])
    histogram_list = list()
    histogram_list = hist
    sum_hist1 = 0
    sum_hist2 = 0
    sum_hist3 = 0
    sum_hist4 = 0
    # get values from histogram
    for i in range(0, 64):
        hist_value = histogram_list[i]
        hist_value_n = str(hist_value[0])
        sum_hist1 = sum_hist1 + float(hist_value_n)

    for i in range(64, 128):
        hist_value = histogram_list[i]
        hist_value_n = str(hist_value[0])
        sum_hist2 = sum_hist2 + float(hist_value_n)

    for i in range(128, 192):
        hist_value = histogram_list[i]
        hist_value_n = str(hist_value[0])
        sum_hist3 = sum_hist3 + float(hist_value_n)

    for i in range(192, 256):
        hist_value = histogram_list[i]
        hist_value_n = str(hist_value[0])
        sum_hist4 = sum_hist4 + float(hist_value_n)

    sum_hist1_div = sum_hist1 / 1000000
    sum_hist2_div = sum_hist2 / 1000000
    sum_hist3_div = sum_hist3 / 1000000
    sum_hist4_div = sum_hist4  / 1000000


    saved_str = file_substr + ", " + str(pixel_mean) + ", " + str(pixel_std) + ", " + str(pixel_average) + ", " + str(pixel_var) + "\n"
    h.write(saved_str)

    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()



f.close()
g.close()
h.close()

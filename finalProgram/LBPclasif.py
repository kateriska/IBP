# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: LBPclasif.py - extraction of vectors based on LBP method

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import processedSegmentation

# function for processing pixel with LBP
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

# function for processing LBP
def processLBP(img, x, y, lbp_values):
    # 3x3 window of pixels
    '''
     pix7 | pix8 | pix9
    ----------------
     pix4 | pix5 | pix6
    ----------------
     pix1 | pix2 | pix3
    '''
    pix5 = img[x][y] # center pixel
    pixel_new_value = []
    pixel_new_value.append(LBPprocesspixel(img, pix5, x, y+1))     #pix8
    pixel_new_value.append(LBPprocesspixel(img, pix5, x-1, y+1))   #pix7
    pixel_new_value.append(LBPprocesspixel(img, pix5, x-1, y))     #pix4
    pixel_new_value.append(LBPprocesspixel(img, pix5, x-1, y-1))   #pix1
    pixel_new_value.append(LBPprocesspixel(img, pix5, x, y-1))     #pix2
    pixel_new_value.append(LBPprocesspixel(img, pix5, x+1, y-1))   #pix3
    pixel_new_value.append(LBPprocesspixel(img, pix5, x+1, y))     #pix6
    pixel_new_value.append(LBPprocesspixel(img, pix5, x+1, y+1))   #pix9

    powers_bin = [1, 2, 4, 8, 16, 32, 64, 128] # 2^0 - 2^7
    value_dec = 0
    for i in range(len(pixel_new_value)):
        value_dec = value_dec + (pixel_new_value[i] * powers_bin[i])

    #print(value_dec)
    lbp_values.append(value_dec)
    return value_dec

def vectorLBP(segmentation_type, color_type):
    # IMAGES FOR TRAINING

    # folders of trained and tested images based on chosen argument of user
    if (color_type == "b"): # blue images
        path_training = './dataset/blueTrain/*'
        path_testing = './dataset/blueTest/*'
    elif (color_type == "g"): # green images
        path_training = './dataset/greenTrain/*'
        path_testing = './dataset/greenTest/*'
    elif (color_type == "r"): # red images
        path_training = './dataset/redTrain/*'
        path_testing = './dataset/redTest/*'
    elif (color_type == "all"): # all images
        path_training = './dataset/allTrain/*'
        path_testing = './dataset/allTest/*'

    # csv files for trained images
    f = open("./csvFiles/LBPtrained.csv","w+")
    g = open("./csvFiles/LBPtrainedResult.csv","w+")
    for file in glob.glob(path_training):
        file_substr = file.split('/')[-1] # get name of processed file
        print(file_substr)

        img = cv2.imread(file, 0) # uint8 image in grayscale
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image

        # decide about type of segmentation
        if (segmentation_type == "otsu"):
            segmented_img = processedSegmentation.imgSegmentation(img)
        elif (segmentation_type == "gauss"):
            segmented_img = processedSegmentation.adaptiveSegmentationGaussian(img)
        elif (segmentation_type == "mean"):
            segmented_img = processedSegmentation.adaptiveSegmentationMean(img)

        # save, load segmented image and get their properties
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

        #cv2.imshow("lbp image", lbp_image)
        hist_lbp = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
        histogram_list = list()
        histogram_list = hist_lbp
        sum_hist1 = 0
        sum_hist2 = 0
        sum_hist3 = 0
        sum_hist4 = 0

        # get values from LBP histogram
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

        # transform to more proper range for classification
        sum_hist1_div = sum_hist1 / 1000000
        sum_hist2_div = sum_hist2 / 1000000
        sum_hist3_div = sum_hist3 / 1000000
        sum_hist4_div = sum_hist4 / 1000000

        img = cv2.imread('./processedImg/segmented_img.jpg', 0) # load input grayscale img

        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        # GLCM matrix characteristics - contrast, homogeinity, energy, correlation
        contrast = greycoprops(matrix_coocurrence, 'contrast')
        print("Contrast:")
        print(contrast)
        contrast_float = float(contrast)
        contrast_str = contrast_float / 10  # transform to more proper range for classification

        homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
        print("Homogeneity:")
        print(homogeneity)
        homogeneity_float = float(homogeneity)
        homogeneity_str = homogeneity_float / 10

        energy = greycoprops(matrix_coocurrence, 'energy')
        print("Energy:")
        print(energy)
        energy_float = float(energy)
        energy_str = energy_float / 10

        correlation = greycoprops(matrix_coocurrence, 'correlation')
        print("Correlation:")
        print(correlation)
        correlation_float = float(correlation)
        correlation_str = correlation_float / 10

        saved_str = str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + ", " + str(contrast_str) + ", " + str(homogeneity_str) + ", " + str(energy_str) + ", " + str(correlation_str) + "\n"
        f.write(saved_str) # write vector to file

        # save known result of image based on its file name
        if ("fake" in file_substr):
            print("This is known FAKE image for training")
            g.write("0\n")
        elif ("live" in file_substr):
            print("This is known LIVE image for training")
            g.write("1\n")

        print()



# IMAGES FOR TESTING

    h = open("./csvFiles/LBPtested.csv","w+") # csv file for tested images
    for file in glob.glob(path_testing):
        file_substr = file.split('/')[-1] # get name of processed file
        print(file_substr)

        img = cv2.imread(file, 0) # uint8 image in grayscale
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalized image

        # decide about type of segmentation
        if (segmentation_type == "otsu"):
            segmented_img = processedSegmentation.imgSegmentation(img)
        elif (segmentation_type == "gauss"):
            segmented_img = processedSegmentation.adaptiveSegmentationGaussian(img)
        elif (segmentation_type == "mean"):
            segmented_img = processedSegmentation.adaptiveSegmentationMean(img)

        # save, load segmented image and get their properties
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

        #cv2.imshow("lbp image", lbp_image)
        hist_lbp = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])

        histogram_list = list()
        histogram_list = hist_lbp
        sum_hist1 = 0
        sum_hist2 = 0
        sum_hist3 = 0
        sum_hist4 = 0

        # get values from LBP histogram
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

        # transform to more proper range for classification
        sum_hist1_div = sum_hist1 / 1000000
        sum_hist2_div = sum_hist2 / 1000000
        sum_hist3_div = sum_hist3 / 1000000
        sum_hist4_div = sum_hist4  / 1000000

        img = cv2.imread('./processedImg/segmented_img.jpg', 0)

        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        # GLCM matrix characteristics - contrast, homogeinity, energy, correlation
        contrast = greycoprops(matrix_coocurrence, 'contrast')
        print("Contrast:")
        print(contrast)
        contrast_float = float(contrast)
        contrast_str = contrast_float / 10 # transform to more proper range for classification

        homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
        print("Homogeneity:")
        print(homogeneity)
        homogeneity_float = float(homogeneity)
        homogeneity_str = homogeneity_float / 10

        energy = greycoprops(matrix_coocurrence, 'energy')
        print("Energy:")
        print(energy)
        energy_float = float(energy)
        energy_str = energy_float / 10

        correlation = greycoprops(matrix_coocurrence, 'correlation')
        print("Correlation:")
        print(correlation)
        correlation_float = float(correlation)
        correlation_str = correlation_float / 10

        saved_str = file_substr + ", " + str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + ", " + str(contrast_str) + ", " + str(homogeneity_str) + ", " + str(energy_str) + ", " + str(correlation_str) +"\n"
        print("This image will be used for liveness prediction")
        h.write(saved_str) # write vector to file
        print()

    # properly close all csv files
    f.close()
    g.close()
    h.close()


    return

# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: SobelLaplacianExperiment.py - extraction of vectors based on Sobel and Laplacian operator

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import pywt

# function for segmentation of fingerprint with using Otsu tresholding
def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel)  # use morphological operations
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200);
    result_tresh = cv2.add(tresh_img, opening)
    result_orig = cv2.add(img, opening) # add mask with input image
    return result_orig

def vectorSobelLaplacian(color_type):
    # IMAGES FOR TRAINING

    # folders of trained and tested images based on chosen argument of user
    if (color_type == "b"): # blue images
        path_training = '/home/katerina/Documents/FinalProgramIBP/blueTraining/*'
        path_testing = '/home/katerina/Documents/FinalProgramIBP/blueTesting/*'
    elif (color_type == "g"): # green images
        path_training = '/home/katerina/Documents/FinalProgramIBP/greenTraining/*'
        path_testing = '/home/katerina/Documents/FinalProgramIBP/greenTesting/*'
    elif (color_type == "r"): # red images
        path_training = '/home/katerina/Documents/FinalProgramIBP/redTraining/*'
        path_testing = '/home/katerina/Documents/FinalProgramIBP/redTesting/*'
    elif (color_type == "all"): # all images
        path_training = '/home/katerina/Documents/FinalProgramIBP/allTraining/*'
        path_testing = '/home/katerina/Documents/FinalProgramIBP/allTesting/*'

    # csv files for trained images
    f = open("SLTrained.csv","w+")
    g = open("SLTrainedResult.csv","w+")
    for file in glob.glob(path_training):
        img = cv2.imread(file, 0) # uint8 image in grayscale
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
        segmented_img = imgSegmentation(img)
        cv2.imwrite('segmented_img.jpg', segmented_img)

        image = cv2.imread('segmented_img.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(image,cv2.CV_64F) # get result of image processing with Laplacian operator
        cv2.imwrite("laplacian_img.jpg", laplacian)
        sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)  # get result of x-axis for image processed with Sobel operator
        cv2.imwrite("sobelx_img.jpg", sobelx)
        sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)  # get result of y-axis for image processed with Sobel operator
        cv2.imwrite("sobely_img.jpg", sobelx)

    # PROCESS LAPLACIAN IMAGE
        img = cv2.imread('laplacian_img.jpg',0)
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
        contrast_str = contrast_float / 10

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

    # PROCESS SOBEL ON X-AXIS IMAGE
        img = cv2.imread('sobelx_img.jpg',0)
        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        contrast2 = greycoprops(matrix_coocurrence, 'contrast')
        print("Contrast:")
        print(contrast2)
        contrast_float2 = float(contrast2)
        contrast_str2 = contrast_float2 / 10

        homogeneity2 = greycoprops(matrix_coocurrence, 'homogeneity')
        print("Homogeneity:")
        print(homogeneity2)
        homogeneity_float2 = float(homogeneity2)
        homogeneity_str2 = homogeneity_float2 / 10

        energy2 = greycoprops(matrix_coocurrence, 'energy')
        print("Energy:")
        print(energy2)
        energy_float2 = float(energy2)
        energy_str2 = energy_float2 / 10

        correlation2 = greycoprops(matrix_coocurrence, 'correlation')
        print("Correlation:")
        print(correlation2)
        correlation_float2 = float(correlation2)
        correlation_str2 = correlation_float2 / 10

    # PROCESS SOBEL ON Y-AXIS IMAGE
        img = cv2.imread('sobely_img.jpg',0)
        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        contrast3 = greycoprops(matrix_coocurrence, 'contrast')
        print("Contrast:")
        print(contrast3)
        contrast_float3 = float(contrast3)
        contrast_str3 = contrast_float3 / 10

        homogeneity3 = greycoprops(matrix_coocurrence, 'homogeneity')
        print("Homogeneity:")
        print(homogeneity3)
        homogeneity_float3 = float(homogeneity3)
        homogeneity_str3 = homogeneity_float3 / 10

        energy3 = greycoprops(matrix_coocurrence, 'energy')
        print("Energy:")
        print(energy3)
        energy_float3 = float(energy3)
        energy_str3 = energy_float3 / 10

        correlation3 = greycoprops(matrix_coocurrence, 'correlation')
        print("Correlation:")
        print(correlation3)
        correlation_float3 = float(correlation3)
        correlation_str3 = correlation_float3 / 10

        saved_str = (str(contrast_str) + ", " + str(homogeneity_str) + ", " + str(energy_str) + ", " + str(correlation_str) + ", "+ str(contrast_str2) + ", " + str(homogeneity_str2) + ", " + str(energy_str2) + ", " + str(correlation_str2)  + ", " + str(contrast_str3) + ", " + str(homogeneity_str3) + ", " + str(energy_str3) + ", " + str(correlation_str3) +   "\n" )
        print(saved_str)

        file_substr = file.split('/')[-1]

        f.write(saved_str) # write vector to file

        # save known result of image based on its file name
        if ("fake" in file_substr):
            print("This is FAKE image.")
            g.write("0\n")
        elif ("live" in file_substr):
            print("This is LIVE image.")
            g.write("1\n")



# IMAGES FOR TESTING
    h = open("SLTested.csv","w+") # csv file for tested images
    for file in glob.glob(path_testing):
        img = cv2.imread(file, 0) # uint8 image in grayscale
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
        segmented_img = imgSegmentation(img)
        cv2.imwrite('segmented_img.jpg', segmented_img)

        image = cv2.imread('segmented_img.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(image,cv2.CV_64F) # get result of image processing with Laplacian operator
        cv2.imwrite("laplacian_img.jpg", laplacian)
        sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)  # get result of x-axis for image processed with Sobel operator
        cv2.imwrite("sobelx_img.jpg", sobelx)
        sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)  # get result of y-axis for image processed with Sobel operator
        cv2.imwrite("sobely_img.jpg", sobelx)

    # PROCESS LAPLACIAN IMAGE
        img = cv2.imread('laplacian_img.jpg',0)
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
        contrast_str = contrast_float / 10

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

    # PROCESS SOBEL ON X-AXIS IMAGE
        img = cv2.imread('sobelx_img.jpg',0)
        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        contrast2 = greycoprops(matrix_coocurrence, 'contrast')
        print("Contrast:")
        print(contrast2)
        contrast_float2 = float(contrast2)
        contrast_str2 = contrast_float2 / 10

        homogeneity2 = greycoprops(matrix_coocurrence, 'homogeneity')
        print("Homogeneity:")
        print(homogeneity2)
        homogeneity_float2 = float(homogeneity2)
        homogeneity_str2 = homogeneity_float2 / 10

        energy2 = greycoprops(matrix_coocurrence, 'energy')
        print("Energy:")
        print(energy2)
        energy_float2 = float(energy2)
        energy_str2 = energy_float2 / 10

        correlation2 = greycoprops(matrix_coocurrence, 'correlation')
        print("Correlation:")
        print(correlation2)
        correlation_float2 = float(correlation2)
        correlation_str2 = correlation_float2 / 10

    # PROCESS SOBEL ON Y-AXIS IMAGE
        img = cv2.imread('sobely_img.jpg',0)
        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        contrast3 = greycoprops(matrix_coocurrence, 'contrast')
        print("Contrast:")
        print(contrast3)
        contrast_float3 = float(contrast3)
        contrast_str3 = contrast_float3 / 10

        homogeneity3 = greycoprops(matrix_coocurrence, 'homogeneity')
        print("Homogeneity:")
        print(homogeneity3)
        homogeneity_float3 = float(homogeneity3)
        homogeneity_str3 = homogeneity_float3 / 10

        energy3 = greycoprops(matrix_coocurrence, 'energy')
        print("Energy:")
        print(energy3)
        energy_float3 = float(energy3)
        energy_str3 = energy_float3 / 10

        correlation3 = greycoprops(matrix_coocurrence, 'correlation')
        print("Correlation:")
        print(correlation3)
        correlation_float3 = float(correlation3)
        correlation_str3 = correlation_float3 / 10


        file_substr = file.split('/')[-1]

        saved_str = (file_substr + ", " + str(contrast_str) + ", " + str(homogeneity_str) + ", " + str(energy_str) + ", " + str(correlation_str) + ", "+ str(contrast_str2) + ", " + str(homogeneity_str2) + ", " + str(energy_str2) + ", " + str(correlation_str2)  + ", " + str(contrast_str3) + ", " + str(homogeneity_str3) + ", " + str(energy_str3) + ", " + str(correlation_str3)  +   "\n" )
        print(saved_str)

        h.write(saved_str) # write vector to file

    # properly close all csv files
    f.close()
    g.close()
    h.close()

    return

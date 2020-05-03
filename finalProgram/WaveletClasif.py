# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: WaveletClasif.py - extraction of vectors based on Wavelet transform

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

def vectorWavelet(segmentation_type, color_type, wavelet_type):
# IMAGES FOR TRAINING

    # csv files for trained images
    f = open("./csvFiles/WaveletTrained.csv","w+")
    g = open("./csvFiles/WaveletTrainedResult.csv","w+")

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

    for file in glob.glob(path_training):
        img = cv2.imread(file, 0) # uint8 image in grayscale
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image

        if (segmentation_type == "otsu"):
            segmented_img = processedSegmentation.imgSegmentation(img)
        elif (segmentation_type == "gauss"):
            segmented_img = processedSegmentation.adaptiveSegmentationGaussian(img)
        elif (segmentation_type == "mean"):
            segmented_img = processedSegmentation.adaptiveSegmentationMean(img)

        cv2.imwrite('./processedImg/segmented_img.jpg', segmented_img)

        image = cv2.imread('./processedImg/segmented_img.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to float32 for more resolution for use with pywt
        image = np.float32(image)
        image /= 255

        coeffs2 = pywt.dwt2(image, wavelet_type)
        LL, (LH, HL, HH) = coeffs2

        # gaining LH, HL and HH image
        plt.imshow(LH, interpolation="nearest", cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig('./processedImg/LHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

        plt.imshow(HL, interpolation="nearest", cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig('./processedImg/HLimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

        plt.imshow(HH, interpolation="nearest", cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig('./processedImg/HHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    # PROCESS LH IMAGE - Horizontal detail
        img = cv2.imread('./processedImg/LHimg.png',0)
        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        # GLCM matrix characteristics - contrast, homogeinity, energy, correlation
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


    # PROCESS HL IMAGE - Vertical detail
        img = cv2.imread('./processedImg/HLimg.png',0)
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

    # PROCESS HH IMAGE - Diagonal detail
        img = cv2.imread('./processedImg/HHimg.png',0)
        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        contrast4 = greycoprops(matrix_coocurrence, 'contrast')
        print("Contrast:")
        print(contrast4)
        contrast_float4 = float(contrast4)
        contrast_str4 = contrast_float4 / 10

        homogeneity4 = greycoprops(matrix_coocurrence, 'homogeneity')
        print("Homogeneity:")
        print(homogeneity4)
        homogeneity_float4 = float(homogeneity4)
        homogeneity_str4 = homogeneity_float4 / 10

        energy4 = greycoprops(matrix_coocurrence, 'energy')
        print("Energy:")
        print(energy4)
        energy_float4 = float(energy4)
        energy_str4 = energy_float4 / 10

        correlation4 = greycoprops(matrix_coocurrence, 'correlation')
        print("Correlation:")
        print(correlation4)
        correlation_float4 = float(correlation4)
        correlation_str4 = correlation_float4 / 10

        saved_str = (str(contrast_str2) + ", " + str(homogeneity_str2) + ", " + str(energy_str2) + ", " + str(correlation_str2)  + ", " + str(contrast_str3) + ", " + str(homogeneity_str3) + ", " + str(energy_str3) + ", " + str(correlation_str3) + ", " +  str(contrast_str4) + ", " + str(homogeneity_str4) + ", " + str(energy_str4) + ", " + str(correlation_str4) +   "\n" )
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

        print()



# IMAGES FOR TESTING

    h = open("./csvFiles/WaveletTested.csv","w+") # csv file for tested images
    for file in glob.glob(path_testing):
        img = cv2.imread(file, 0) # uint8 image in grayscale
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image

        if (segmentation_type == "otsu"):
            segmented_img = processedSegmentation.imgSegmentation(img)
        elif (segmentation_type == "gauss"):
            segmented_img = processedSegmentation.adaptiveSegmentationGaussian(img)
        elif (segmentation_type == "mean"):
            segmented_img = processedSegmentation.adaptiveSegmentationMean(img)

        cv2.imwrite('./processedImg/segmented_img.jpg', segmented_img)
        image = cv2.imread('./processedImg/segmented_img.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to float for more resolution for use with pywt
        image = np.float32(image)
        image /= 255

        coeffs2 = pywt.dwt2(image, wavelet_type)
        LL, (LH, HL, HH) = coeffs2

        plt.imshow(LH, interpolation="nearest", cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig('./processedImg/LHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

        plt.imshow(HL, interpolation="nearest", cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig('./processedImg/HLimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

        plt.imshow(HH, interpolation="nearest", cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig('./processedImg/HHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    # PROCESS LH IMAGE
        img = cv2.imread('./processedImg/LHimg.png',0)
        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        # GLCM matrix characteristics - contrast, homogeinity, energy, correlation
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

    # PROCESS HL IMAGE
        img = cv2.imread('./processedImg/HLimg.png',0)
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

    # PROCESS HH IMAGE
        img = cv2.imread('./processedImg/HHimg.png',0)
        image = img_as_ubyte(img)

        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max() + 1
        matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

        contrast4 = greycoprops(matrix_coocurrence, 'contrast')
        print("Contrast:")
        print(contrast4)
        contrast_float4 = float(contrast4)
        contrast_str4 = contrast_float4 / 10

        homogeneity4 = greycoprops(matrix_coocurrence, 'homogeneity')
        print("Homogeneity:")
        print(homogeneity4)
        homogeneity_float4 = float(homogeneity4)
        homogeneity_str4 = homogeneity_float4 / 10

        energy4 = greycoprops(matrix_coocurrence, 'energy')
        print("Energy:")
        print(energy4)
        energy_float4 = float(energy4)
        energy_str4 = energy_float4 / 10

        correlation4 = greycoprops(matrix_coocurrence, 'correlation')
        print("Correlation:")
        print(correlation4)
        correlation_float4 = float(correlation4)
        correlation_str4 = correlation_float4 / 10

        file_substr = file.split('/')[-1]

        saved_str = (file_substr + ", " + str(contrast_str2) + ", " + str(homogeneity_str2) + ", " + str(energy_str2) + ", " + str(correlation_str2)  + ", " + str(contrast_str3) + ", " + str(homogeneity_str3) + ", " + str(energy_str3) + ", " + str(correlation_str3) + ", " +  str(contrast_str4) + ", " + str(homogeneity_str4) + ", " + str(energy_str4) + ", " + str(correlation_str4) +   "\n" )
        print(saved_str)

        h.write(saved_str) # write vector to file
        print()

    # properly close all csv files
    f.close()
    g.close()
    h.close()

    return

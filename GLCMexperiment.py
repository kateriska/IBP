import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

### IMAGE SEGMENTATION WITH MORPHOLOGY OPERATIONS
def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel)
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200);
    result_tresh = cv2.add(tresh_img, opening)
    result_orig = cv2.add(img, opening)
    return result_orig


# IMAGES FOR TRAINING
path = '/home/katerina/Documents/IBP/trainingGood/*'

f = open("GLCMTrained.csv","w+")
g = open("GLCMTrainedResult.csv","w+")
for file in glob.glob(path):
    img = cv2.imread(file, 0) # uint8 image in grayscale
    img = cv2.resize(img,(360,360)) # resize of image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
    segmented_img = imgSegmentation(img)
    #cv2.imshow('Segmented image', segmented_img)
    cv2.imwrite('segmented_img.tif', segmented_img)
    img = cv2.imread('segmented_img.tif', 0)

    image = img_as_ubyte(img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)
    #print(matrix_coocurrence)
    # Charasterictics of Gray Level Matrix:
    # contrast ***
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

    # Energy ***
    energy = greycoprops(matrix_coocurrence, 'energy')
    print("Energy:")
    print(energy)
    energy_float = float(energy)
    energy_str = energy_float / 10

    # Correlation ***
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    print("Correlation:")
    print(correlation)
    correlation_float = float(correlation)
    correlation_str = correlation_float / 10

    saved_str = (str(contrast_str) + ", " + str(homogeneity_str) + ", " + str(energy_str) + ", " + str(correlation_str) + "\n" )



    file_substr = file.split('/')[-1]

    #saved_str = str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "\n"
    #print("[" + str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "],")
    f.write(saved_str)

    #file_substr = file.split('/')[-1]
    #print(file_substr)
    if ("fake" in file_substr):
        print("This is FAKE image.")
        g.write("0\n")
    elif ("live" in file_substr):
        print("This is LIVE image.")
        g.write("1\n")

    #print(file_substr)



#print(weights)


    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()


# IMAGES FOR TESTING
path_testing = '/home/katerina/Documents/IBP/testingGood/*'

h = open("GLCMTested.csv","w+")
for file in glob.glob(path_testing):
    img = cv2.imread(file, 0) # uint8 image in grayscale
    img = cv2.resize(img,(360,360)) # resize of image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
    segmented_img = imgSegmentation(img)
    #cv2.imshow('Segmented image', segmented_img)
    cv2.imwrite('segmented_img.tif', segmented_img)
    img = cv2.imread('segmented_img.tif', 0)

    image = img_as_ubyte(img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)
    #print(matrix_coocurrence)
    # Charasterictics of Gray Level Matrix:
    # contrast ***
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    print("Contrast:")
    print(contrast)
    contrast_float = float(contrast)
    contrast_str = contrast_float / 10
    #contrast_str = str(contrast)
    #contrast_str = contrast_str[2:-2]
    # dissimalirity
    dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
    #print("Dissimilarity:")
    #print(dissimilarity)
    # homogeinity ***
    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    print("Homogeneity:")
    print(homogeneity)
    homogeneity_float = float(homogeneity)
    homogeneity_str = homogeneity_float / 10
    #homogeneity_str = str(homogeneity)
    #homogeneity_str = homogeneity_str[2:-2]
    # ASM
    asm = greycoprops(matrix_coocurrence, 'ASM')
    #print("ASM:")
    #print(asm)
    # Energy ***
    energy = greycoprops(matrix_coocurrence, 'energy')
    print("Energy:")
    print(energy)
    energy_float = float(energy)
    energy_str = energy_float / 10
    #energy_str = str(energy)
    #energy_str = energy_str[2:-2]
    # Correlation ***
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    print("Correlation:")
    print(correlation)
    correlation_float = float(correlation)
    correlation_str = correlation_float / 10
    #correlation_str = str(correlation)
    #correlation_str = correlation_str[2:-2]


    file_substr = file.split('/')[-1]

    #saved_str = str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "\n"
    #print("[" + str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "],")
    saved_str = (str(file_substr) + ", " + str(contrast_str) + ", " + str(homogeneity_str) + ", " + str(energy_str) + ", " + str(correlation_str) + "\n" )
    h.write(saved_str)


    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()



f.close()
g.close()
h.close()

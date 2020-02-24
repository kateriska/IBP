import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

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


### IMAGE SEGMENTATION WITH MORPHOLOGY OPERATIONS
def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow("tresh img", tresh_img)
    # noise removal
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel)
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.Canny(opening, 100, 200);
    result_tresh = cv2.add(tresh_img, opening)
    cv2.imshow("tresh result", result_tresh)
    result_orig = cv2.add(img, opening)
    return result_orig

### LBP ALGORITHM
def LBPprocesspixel(img, pix5, x, y):
    new_value = 0
    try:
        if img[x][y] >= pix5:
            new_value = 1
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

def vectorLBP(segmentation_type, color_type):
# IMAGES FOR TRAINING
    if (color_type == "b"):
        path_training = '/home/katerina/Documents/FinalProgramIBP/blueImagesTraining/*'
        path_testing = '/home/katerina/Documents/FinalProgramIBP/blueImagesTesting/*'
    elif (color_type == "g"):
        path_training = '/home/katerina/Documents/FinalProgramIBP/greenImagesTraining/*'
        path_testing = '/home/katerina/Documents/FinalProgramIBP/greenImagesTesting/*'
    elif (color_type == "r"):
        path_training = '/home/katerina/Documents/FinalProgramIBP/redImagesTraining/*'
        path_testing = '/home/katerina/Documents/FinalProgramIBP/redImagesTesting/*'
    elif (color_type == "all"):
        path_training = '/home/katerina/Documents/FinalProgramIBP/trainingNew/*'
        path_testing = '/home/katerina/Documents/FinalProgramIBP/testingNew/*'



    f = open("LBPGLCMTrained.csv","w+")
    g = open("LBPGLCMresultTrained.csv","w+")
    for file in glob.glob(path_training):
        img = cv2.imread(file, 0) # uint8 image in grayscale
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
        cv2.imwrite('segmented_img.jpg', segmented_img)
        img = cv2.imread('segmented_img.jpg')
        height, width, channel = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp_image = np.zeros((height, width,3), np.uint8)

        file_substr = file.split('/')[-1]
        print(file_substr)
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
        sum_hist4_div = sum_hist4 / 1000000

        img = cv2.imread('segmented_img.jpg', 0)

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


        saved_str = str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + ", " + str(contrast_str) + ", " + str(homogeneity_str) + ", " + str(energy_str) + ", " + str(correlation_str) + "\n"
    #print("[" + str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "],")
        f.write(saved_str)


        if ("fake" in file_substr):
            print("This is FAKE image.")
            g.write("0\n")
        elif ("Images" in file_substr):
            print("This is LIVE image.")
            g.write("1\n")

    #k = cv2.waitKey(1000)
    #destroy the window
    #cv2.destroyAllWindows()


# IMAGES FOR TESTING
    #path_testing = '/home/katerina/Documents/FinalProgramIBP/testingNew/*'

    h = open("LBPGLCMtested.csv","w+")
    for file in glob.glob(path_testing):
        img = cv2.imread(file, 0) # uint8 image in grayscale
    #img = cv2.resize(img,(360,360)) # resize of image
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
        #segmented_img = adaptiveSegmentationMean(img)
        if (segmentation_type == "otsu"):
            segmented_img = imgSegmentation(img)
        elif (segmentation_type == "gauss"):
            segmented_img = adaptiveSegmentationGaussian(img)
        elif (segmentation_type == "mean"):
            segmented_img = adaptiveSegmentationMean(img)
    #cv2.imshow('Segmented image', segmented_img)
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

        #cv2.imshow("lbp image", lbp_image)
        hist_lbp = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
#print(histogram_values)
        histogram_list = list()
        histogram_list = hist_lbp
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

        file_substr = file.split('/')[-1]

        img = cv2.imread('segmented_img.jpg', 0)

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



        saved_str = file_substr + ", " + str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + ", " + str(contrast_str) + ", " + str(homogeneity_str) + ", " + str(energy_str) + ", " + str(correlation_str) +"\n"
    #print("[" + str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "],")
        h.write(saved_str)

    #k = cv2.waitKey(1000)
    #destroy the window
    #cv2.destroyAllWindows()





    f.close()
    g.close()
    h.close()


    return

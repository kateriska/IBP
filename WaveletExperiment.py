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


# IMAGES FOR TRAINING
path = '/home/katerina/Documents/IBP/trainingNew/*'

f = open("WaveletTrained.csv","w+")
g = open("WaveleTrainedResult.csv","w+")
for file in glob.glob(path):
    img = cv2.imread(file, 0) # uint8 image in grayscale
    #img = cv2.resize(img,(360,360)) # resize of image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
    segmented_img = imgSegmentation(img)
    #cv2.imshow('Segmented image', segmented_img)
    cv2.imwrite('segmented_img.jpg', segmented_img)

    image = cv2.imread('segmented_img.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to float for more resolution for use with pywt
    image = np.float32(image)
    image /= 255

    # ...
    # Do your processing
    # ...

    # Wavelet transform of image, and plot approximation and details
    #titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(image, 'bior2.2')
    LL, (LH, HL, HH) = coeffs2
    #fig = plt.figure(figsize=(3, 3))
    #ax = fig.add_subplot(1, 1, 1)
    #plt.imshow(LL, interpolation="nearest", cmap=plt.cm.gray)
    #plt.axis('off')

    #plt.savefig('LLimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    plt.imshow(LH, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('LHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    plt.imshow(HL, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('HLimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    plt.imshow(HH, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('HHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)


    # PROCESS LL IMAGE
    '''
    img = cv2.imread('LLimg.png',0)
    image = img_as_ubyte(img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

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
    '''
    # PROCESS LH IMAGE - Horizontal detail
    img = cv2.imread('LHimg.png',0)
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


    # PROCESS HL IMAGE - Vertical detail
    img = cv2.imread('HLimg.png',0)
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
    img = cv2.imread('HHimg.png',0)
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

    #plt.show()
    # Convert back to uint8 OpenCV format
    #image *= 255
    #image = np.uint8(image)


    file_substr = file.split('/')[-1]

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
path_testing = '/home/katerina/Documents/IBP/testingNew/*'

h = open("WaveletTested.csv","w+")
for file in glob.glob(path_testing):
    img = cv2.imread(file, 0) # uint8 image in grayscale
    #img = cv2.resize(img,(360,360)) # resize of image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
    segmented_img = imgSegmentation(img)
    #cv2.imshow('Segmented image', segmented_img)
    cv2.imwrite('segmented_img.jpg', segmented_img)
    image = cv2.imread('segmented_img.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to float for more resolution for use with pywt
    image = np.float32(image)
    image /= 255

    # ...
    # Do your processing
    # ...

    # Wavelet transform of image, and plot approximation and details
    #titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(image, 'bior2.2')
    LL, (LH, HL, HH) = coeffs2

    #plt.imshow(LL, interpolation="nearest", cmap=plt.cm.gray)
    #plt.axis('off')

    #plt.savefig('LLimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    plt.imshow(LH, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('LHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    plt.imshow(HL, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('HLimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    plt.imshow(HH, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('HHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)


    # PROCESS LL IMAGE
    '''
    img = cv2.imread('LLimg.png',0)
    image = img_as_ubyte(img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

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
    '''

    # PROCESS LH IMAGE
    img = cv2.imread('LHimg.png',0)
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

    # PROCESS HL IMAGE
    img = cv2.imread('HLimg.png',0)
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
    img = cv2.imread('HHimg.png',0)
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

    #plt.show()

    # Convert back to uint8 OpenCV format
    #image *= 255
    #image = np.uint8(image)

    h.write(saved_str)

    #k = cv2.waitKey(1000)
    #destroy the window
    #cv2.destroyAllWindows()



f.close()
g.close()
h.close()

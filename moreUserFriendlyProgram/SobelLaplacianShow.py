import cv2
import numpy as np
from matplotlib import pyplot as plt

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

def showSobelLaplacian(segmentation_type, input_img):
    if (segmentation_type != "none"):
        img = cv2.imread(input_img, 0) # uint8 image in grayscale
    else:
        img = cv2.imread(input_img)

    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
# converting to gray scale
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    cv2.imwrite("laplacianShow.jpg", laplacian)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    cv2.imwrite("sobelxShow.jpg", sobelx)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    cv2.imwrite("sobelyShow.jpg", sobely)

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()

    return

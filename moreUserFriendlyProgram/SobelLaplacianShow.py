# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: SobelLaplacianShow.py - shows details of image processed by Sobel and Laplacian operator

import cv2
import numpy as np
from matplotlib import pyplot as plt
import processedSegmentation

def showSobelLaplacian(segmentation_type, input_img):
    if (segmentation_type != "none"):
        img = cv2.imread(input_img, 0) # uint8 image in grayscale
    else:
        img = cv2.imread(input_img)

    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

    if (segmentation_type == "otsu"):
        segmented_img = processedSegmentation.imgSegmentation(img)
    elif (segmentation_type == "gauss"):
        segmented_img = processedSegmentation.adaptiveSegmentationGaussian(img)
    elif (segmentation_type == "mean"):
        segmented_img = processedSegmentation.adaptiveSegmentationMean(img)

    if (segmentation_type != "none"):
        cv2.imwrite('./processedImg/segmented_img.jpg', segmented_img)
        img = cv2.imread('./processedImg/segmented_img.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(img,cv2.CV_64F) # processed with Laplacian operator
    cv2.imwrite("./processedImg/laplacian_img.jpg", laplacian)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # processed with Sobel on x-axis
    cv2.imwrite("./processedImg/sobelx_img.jpg", sobelx)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # processed with Sobel on y-axis
    cv2.imwrite("./processedImg/sobely_img.jpg", sobely)

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Applied Laplacian operator'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Applied Sobel operator of x axis'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Applied Sobel operator of y axis'), plt.xticks([]), plt.yticks([])

    plt.show()

    return

# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: SobelLaplacianShow.py - shows details of image processed by Sobel and Laplacian operator

import cv2
import numpy as np
from matplotlib import pyplot as plt
import processedSegmentation

def showSobelLaplacian(segmentation_type, input_img):
    
    img = cv2.imread(input_img, 0) # uint8 image in grayscale

    results = list()
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    results.append(img)
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
    laplacian_img = cv2.imread("./processedImg/laplacian_img.jpg", 0)
    results.append(laplacian_img)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # processed with Sobel on x-axis
    cv2.imwrite("./processedImg/sobelx_img.jpg", sobelx)
    sobelx_img = cv2.imread("./processedImg/sobelx_img.jpg", 0)
    results.append(sobelx_img)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # processed with Sobel on y-axis
    cv2.imwrite("./processedImg/sobely_img.jpg", sobely)
    sobely_img = cv2.imread("./processedImg/sobely_img.jpg", 0)
    results.append(sobely_img)

    titles = ['Original normalized image', 'Applied Laplacian operator',
          'Applied Sobel operator of x axis', 'Applied Sobel operator of y axis']


    fig = plt.figure(figsize=(11, 11))
    i = 0
    for a in results:
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(a, cmap='gray')
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        i = i + 1

    fig.tight_layout()
    plt.show()


    return

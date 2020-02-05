import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob

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


# IMAGES FOR TRAINING
path = '/home/katerina/Documents/IBP/trainingGood/*'

f = open("trainedValues.csv","w+")
g = open("resultTrainedImg.csv","w+")
for file in glob.glob(path):
    img = cv2.imread(file, 0) # uint8 image in grayscale
    img = cv2.resize(img,(360,360)) # resize of image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
    segmented_img = imgSegmentation(img)
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

    cv2.imshow("lbp image", lbp_image)
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

    #if (sum_hist1_div > 1 or sum_hist2_div > 1 or sum_hist3_div > 1 or sum_hist4_div > 1):
    #    continue

    saved_str = str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "\n"
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

    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()


# IMAGES FOR TESTING
path_testing = '/home/katerina/Documents/IBP/testingGood/*'

h = open("testedValues.csv","w+")
for file in glob.glob(path_testing):
    img = cv2.imread(file, 0) # uint8 image in grayscale
    img = cv2.resize(img,(360,360)) # resize of image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX) # normalize image
    segmented_img = imgSegmentation(img)
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

    cv2.imshow("lbp image", lbp_image)
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

    saved_str = file_substr + ", " + str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "\n"
    #print("[" + str(sum_hist1_div) + ", " + str(sum_hist2_div) + ", " + str(sum_hist3_div) + ", " + str(sum_hist4_div) + "],")
    h.write(saved_str)

    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()



f.close()
g.close()
h.close()

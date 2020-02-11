import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob
import skimage
import skimage.feature
from skimage.color import rgb2xyz, rgb2luv
from skimage import io, img_as_float
import skimage.measure

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
    img_read = cv2.imread("transparent_img.png")
    #result_color = cv2.add(img_read, opening)
    #cv2.imshow("Color segment", result_color)
    return result_orig

img_blue = cv2.imread("wavelengthPairs/f6b.jpg")
img_red = cv2.imread("wavelengthPairs/f6r.jpg")
img_green = cv2.imread("wavelengthPairs/f6g.jpg")

img_height, img_width = 480, 640
n_channels = 4
transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)

# Save the image for visualization
cv2.imwrite("./transparent_img.png", transparent_img)

img = cv2.addWeighted(img_blue, 0.5, img_red, 0.5, 0)
final_img = cv2.addWeighted(img, 0.5, img_green, 0.5, 0)

img_xyz = rgb2xyz(final_img)
img_luv = rgb2luv(final_img)
#cv2.rectangle(img_xyz, (250, 170), (300, 220), (255,0,0), 2)
#final_img = cv2.resize(final_img,(360,360)) # resize of image
cv2.imshow("result",final_img)
#img_xyz.convertTo(img_result, CV_8UC3, 255.0);

cv2.imwrite("mergedImg.jpg", final_img)
img_read = cv2.imread("mergedImg.jpg", 0)
img_segmented = imgSegmentation(img_read)
cv2.imshow("Segmented image", img_segmented)
image = img_as_float(img_segmented)
# pixel intensity arithmetic mean
print(np.mean(image))
#  pixel intensity standard deviation
print(np.std(image))
print(np.average(image))

print(np.var(image))

entropy = skimage.measure.shannon_entropy(img_segmented)
print(entropy)
#print(np.median(image))
hist = cv2.calcHist([img_segmented], [0], None, [256], [0, 256])
histogram_list = list()
histogram_list = hist
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

print(sum_hist1_div)
print(sum_hist2_div)
print(sum_hist3_div)
print(sum_hist4_div)
figure = plt.figure()
current_plot = figure.add_subplot(1, 1, 1)
current_plot.plot(hist, color = (0, 0, 0.2))
#current_plot.set_xlim([0,260])
current_plot.set_xlim([0,250])
current_plot.set_ylim([0,6000])
current_plot.set_title("Histogram")
current_plot.set_xlabel("Intensity")
current_plot.set_ylabel("Count of pixels")
ytick_list = [int(i) for i in current_plot.get_yticks()]
current_plot.set_yticklabels(ytick_list,rotation = 90)
plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([final_img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
#plt.show()
cv2.waitKey(0)

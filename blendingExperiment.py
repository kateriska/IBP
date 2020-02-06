import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import glob
import skimage
import skimage.feature
from skimage.color import rgb2xyz, rgb2luv

img_blue = cv2.imread("wavelengthPairs/pair5b.jpg")
img_red = cv2.imread("wavelengthPairs/pair5r.jpg")
img_green = cv2.imread("wavelengthPairs/pair5g.jpg")

img = cv2.addWeighted(img_blue, 0.5, img_red, 0.5, 0)
final_img = cv2.addWeighted(img, 0.5, img_green, 0.5, 0)

img_xyz = rgb2xyz(final_img)
img_luv = rgb2luv(final_img)
cv2.imshow("result",img_luv)
cv2.waitKey(0)

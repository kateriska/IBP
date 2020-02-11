import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading image
#img0 = cv2.imread('SanFrancisco.jpg',)
img0 = cv2.imread('segmented_img.jpg')

# converting to gray scale
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
#img = cv2.GaussianBlur(gray,(3,3),0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite("laplacian_img.jpg", laplacian)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
cv2.imwrite("sobelx_img.jpg", sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
cv2.imwrite("sobely_img.jpg", sobelx)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

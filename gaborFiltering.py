import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

orig_img = cv2.imread("segmented_img.jpg")  # uint8 image
img = np.float32(orig_img)
shape_img = img.shape

kernel_size = (21,21)
sigma = 8
lambda_v = 8
gamma = 0.05
psi = 0

theta1 = 0
theta2 = 45
theta3 = 90
theta4 = 135

enhanced = np.zeros(shape_img, dtype=np.float32)

kernel1 = cv2.getGaborKernel(kernel_size, sigma, theta1, lambda_v, gamma, psi, cv2.CV_32F);
kernel2 = cv2.getGaborKernel(kernel_size, sigma, theta2, lambda_v, gamma, psi, cv2.CV_32F);
kernel3 = cv2.getGaborKernel(kernel_size, sigma, theta3, lambda_v, gamma, psi, cv2.CV_32F);
kernel4 = cv2.getGaborKernel(kernel_size, sigma, theta4, lambda_v, gamma, psi, cv2.CV_32F);
gabor1 = cv2.filter2D(img, -1, kernel1);
gabor2 = cv2.filter2D(img, -1, kernel2);
gabor3 = cv2.filter2D(img, -1, kernel3);
gabor4 = cv2.filter2D(img, -1, kernel4);
cv2.imshow('gabor1', gabor1)
cv2.imshow('gabor2', gabor2)
cv2.imshow('gabor3', gabor3)
cv2.imshow('gabor4', gabor4)

cv2.waitKey(0)
cv2.destroyAllWindows()

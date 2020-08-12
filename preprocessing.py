import sys
import cv2
import numpy as np
import math
import pytesseract
import pyautogui
import matplotlib.pyplot as plt

imagepath = 'test3.jpg'

#read gray image
im = cv2.imread(imagepath,0) 

im = cv2.fastNlMeansDenoising(im,None,10,7,21)
median = cv2.medianBlur(im, 5)
kernel_size = 3
im = cv2.GaussianBlur(median,(kernel_size, kernel_size), 0)


kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
dilate=cv2.dilate(im,kernel)
erode=cv2.erode(im,kernel)
image=cv2.absdiff(dilate,erode)
image=cv2.bitwise_not(image)

ret, th = cv2.threshold( image, 254, 255, cv2.THRESH_BINARY )

ret,im = cv2.threshold(im, 130, 255,cv2.THRESH_BINARY)

cv2.imshow('result',im)

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        if th[i][j] == 0:
            im[i][j] = 255

#im = cv2.fastNlMeansDenoising(im,None,10,7,21)
#median = cv2.medianBlur(dst, 5)
#kernel_size = 3
#im = cv2.GaussianBlur(median,(kernel_size, kernel_size), 0)

#xgred=cv2.Sobel(im,cv2.CV_16SC1,1,0)
#ygred=cv2.Sobel(im,cv2.CV_16SC1,1,0)
#im=cv2.Canny(xgred,ygred,50,150)
im = cv2.Canny(im, 30, 100, apertureSize = 5)
im = cv2.dilate(im,cv2.getStructuringElement(cv2.MORPH_RECT,(25,25)))
im = cv2.bitwise_not(im)

cv2.namedWindow('result',cv2.WINDOW_NORMAL)
cv2.imshow('result',im)

cv2.waitKey(0)

text = pytesseract.image_to_string(im, config="--psm 13")
print(text)

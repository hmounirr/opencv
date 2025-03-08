import cv2 as cv
import numpy as np

img = cv.imread('images/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur the image 
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)
 
# Canny Edges
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# Thresholding
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

# Find contours
countours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(countours)} contours found!')

# Draw contours
cv.drawContours(blank, countours, -1, (0, 0, 255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)
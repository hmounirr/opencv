import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('images/corrosion.jpeg')

# Extract only the red channel (no merging needed)
r = img[:, :, 2]  
cv.imshow('Red Channel', r)
# Apply Gaussian Blur to reduce noise
r_blur = cv.GaussianBlur(r, (5, 5), 0)

# Apply Otsu’s Thresholding
_, thresh = cv.threshold(r_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

cv.imshow('Thresholded Rust', thresh)

# Get Contours
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print(f'{len(contours)} contours found!')

# Draw Contours on Original Image
img_contours = img.copy()
cv.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

cv.imshow('Detected Corrosion', img_contours)

cv.waitKey(0)
cv.destroyAllWindows()

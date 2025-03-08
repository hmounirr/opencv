import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('images/corrosion.jpeg')

# Resize for consistency (optional, adjust size as needed)
img = cv.resize(img, (800, 600))

# Convert to HSV (better for color segmentation)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# K-Means Clustering for better segmentation
Z = img.reshape((-1, 3))  # Convert image to a 2D array of pixels
Z = np.float32(Z)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3  # Number of clusters (tune this for better results)
_, labels, centers = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_img = centers[labels.flatten()].reshape(img.shape)

# Convert back to HSV to get color-based rust mask
hsv_segmented = cv.cvtColor(segmented_img, cv.COLOR_BGR2HSV)

# Define rust color range (fine-tune if necessary)
lower_rust = np.array([5, 40, 50])   
upper_rust = np.array([25, 255, 255])

# Create rust mask
rust_mask = cv.inRange(hsv_segmented, lower_rust, upper_rust)

# Apply Otsu's thresholding on grayscale for additional filtering
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, otsu_thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Combine rust mask & thresholded image
final_mask = cv.bitwise_and(rust_mask, otsu_thresh)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)
final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)

# Find contours of corroded areas
contours, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
output = img.copy()
cv.drawContours(output, contours, -1, (0, 0, 255), 2)

# Calculate total corroded area
corroded_area = sum(cv.contourArea(cnt) for cnt in contours)
print(f"Total Corroded Area: {corroded_area} pixels")

# Display results
cv.imshow('Original Image', img)
cv.imshow('Segmented Image (K-Means)', segmented_img)
cv.imshow('Rust Mask', rust_mask)
cv.imshow('Otsu Thresholding', otsu_thresh)
cv.imshow('Final Mask', final_mask)
cv.imshow('Detected Corrosion', output)

cv.waitKey(0)
cv.destroyAllWindows()

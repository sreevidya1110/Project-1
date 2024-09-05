import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "C:\Sreevidya\VIT\Project-1\images\img_1.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use binary thresholding to create a mask
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a mask to draw the regions
mask = np.zeros_like(gray)

# Loop through contours and draw only those within certain size ranges (to focus on head, arms, etc.)
for contour in contours:
    area = cv2.contourArea(contour)
    if 500 < area < 10000:  # Adjust this range based on the actual size of the regions
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

# Bitwise-and mask and the original image to extract the regions
extracted = cv2.bitwise_and(image, image, mask=mask)

# Display the extracted parts
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB))
plt.title("Extracted Head, Arms, Neck, and Legs")
plt.axis('off')
plt.show()

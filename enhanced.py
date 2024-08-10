import cv2
import numpy as np
import os
import csv
from glob import glob
import matplotlib.pyplot as plt

# Constants
image_folder = "C:\Sreevidya\VIT\Project-1\images"  # Update this to the path where your images are stored
output_csv = "C:\Sreevidya\VIT\Project-1\images\enhanced_dance_poses.csv"  # Update this to your desired output path

# Morphological kernel
kernel = np.ones((5, 5), np.uint8)

# Load image filenames
image_files = glob(os.path.join(image_folder, "*.png"))

# Grayscale Conversion using Luminosity Method
def grayscale_luminosity(image):
    return cv2.transform(image, np.array([[0.21, 0.72, 0.07]]))

# Adaptive Thresholding for Binarization
def adaptive_threshold(image):
    binarized_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binarized_image

# Enhancement function
def enhance_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Step a: Resize image
    resized_img = cv2.resize(img, (128, 128))

    # Step b: Crop image (optional - if needed)
    h, w = resized_img.shape[:2]
    crop_img = resized_img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]

    # Step c: Convert to grayscale using luminosity method
    gray_img = grayscale_luminosity(crop_img)
    gray_img = gray_img.astype(np.uint8)

    # Step d: Adjust contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray_img)

    # Step e: Binarization using adaptive thresholding method
    bin_img = adaptive_threshold(contrast_img)

    # Step f: Remove noise using Median Filter
    median_filtered_img = cv2.medianBlur(bin_img, 5)

    # Step g: Morphological operation (using closing)
    morph_img = cv2.morphologyEx(median_filtered_img, cv2.MORPH_CLOSE, kernel)

    # Step h: Apply normalization (Min-Max Normalization)
    norm_img = cv2.normalize(morph_img, None, 0, 255, cv2.NORM_MINMAX)

    return resized_img, norm_img

# Visualize original and enhanced images
def visualize_images(original, enhanced):
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Enhanced Image
    plt.subplot(1, 2, 2)
    plt.title("Enhanced Image")
    plt.imshow(enhanced, cmap="gray")
    plt.axis("off")

    plt.show()

# Create the CSV with the original and enhanced images
with open(output_csv, "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    header = ["Image"] + [f"Original_Pixel_{i}" for i in range(128*128*3)] + [f"Enhanced_Pixel_{i}" for i in range(128*128)]
    csvwriter.writerow(header)

    # Process each image
    for idx, image_path in enumerate(image_files):
        original_img, enhanced_img = enhance_image(image_path)
        original_pixels = original_img.flatten()
        enhanced_pixels = enhanced_img.flatten()

        # Visualize images
        visualize_images(original_img, enhanced_img)

        # Save data to CSV
        csvwriter.writerow([os.path.basename(image_path)] + original_pixels.tolist() + enhanced_pixels.tolist())

print(f"Original and enhanced images have been saved to {output_csv}")

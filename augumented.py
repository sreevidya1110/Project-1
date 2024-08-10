import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Define directories
original_folder = r'C:\Sreevidya\VIT\Project-1\images'
output_folders = {
    'crop': r'C:\Sreevidya\VIT\Project-1\augmented_images\crop',
    'shear': r'C:\Sreevidya\VIT\Project-1\augmented_images\shear',
    'grayscale': r'C:\Sreevidya\VIT\Project-1\augmented_images\grayscale',
    'hue': r'C:\Sreevidya\VIT\Project-1\augmented_images\hue',
    'saturation': r'C:\Sreevidya\VIT\Project-1\augmented_images\saturation',
    'brightness': r'C:\Sreevidya\VIT\Project-1\augmented_images\brightness',
    'exposure': r'C:\Sreevidya\VIT\Project-1\augmented_images\exposure',
    'blur': r'C:\Sreevidya\VIT\Project-1\augmented_images\blur',
    'noise': r'C:\Sreevidya\VIT\Project-1\augmented_images\noise',
    'cutout': r'C:\Sreevidya\VIT\Project-1\augmented_images\cutout',
    'mosaic': r'C:\Sreevidya\VIT\Project-1\augmented_images\mosaic'
}

# Create output folders if they do not exist
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Define augmentations
def augment_image(image_path, save_path, augmentation_type):
    img = Image.open(image_path)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    if augmentation_type == 'crop':
        # Crop augmentation example (simple center crop)
        width, height = img.size
        left = width / 4
        top = height / 4
        right = 3 * width / 4
        bottom = 3 * height / 4
        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(save_path)
    
    elif augmentation_type == 'shear':
        # Shear augmentation example (simple shear)
        rows, cols, _ = img_cv.shape
        M = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
        img_sheared = cv2.warpAffine(img_cv, M, (cols, rows))
        cv2.imwrite(save_path, img_sheared)
    
    elif augmentation_type == 'grayscale':
        img_grayscale = img.convert('L')
        img_grayscale.save(save_path)
    
    elif augmentation_type == 'hue':
        enhancer = ImageEnhance.Color(img)
        img_hue = enhancer.enhance(2)  # Adjust hue
        img_hue.save(save_path)
    
    elif augmentation_type == 'saturation':
        enhancer = ImageEnhance.Color(img)
        img_saturation = enhancer.enhance(2)  # Increase saturation
        img_saturation.save(save_path)
    
    elif augmentation_type == 'brightness':
        enhancer = ImageEnhance.Brightness(img)
        img_brightness = enhancer.enhance(2)  # Increase brightness
        img_brightness.save(save_path)
    
    elif augmentation_type == 'exposure':
        enhancer = ImageEnhance.Brightness(img)
        img_exposure = enhancer.enhance(2)  # Simulate exposure change
        img_exposure.save(save_path)
    
    elif augmentation_type == 'blur':
        img_blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
        cv2.imwrite(save_path, img_blurred)
    
    elif augmentation_type == 'noise':
        noise = np.random.normal(0, 25, img_cv.shape).astype(np.uint8)
        img_noisy = cv2.add(img_cv, noise)
        cv2.imwrite(save_path, img_noisy)
    
    elif augmentation_type == 'cutout':
        # Cutout augmentation example
        mask = np.ones(img_cv.shape, np.uint8) * 255
        x, y, w, h = 50, 50, 100, 100  # Define cutout area
        mask[y:y+h, x:x+w] = 0
        img_cutout = cv2.bitwise_and(img_cv, mask)
        cv2.imwrite(save_path, img_cutout)
    
    elif augmentation_type == 'mosaic':
        # Mosaic augmentation example (create a mosaic with 4 images)
        img_mosaic = np.concatenate([
            np.concatenate([img_cv, img_cv], axis=1),
            np.concatenate([img_cv, img_cv], axis=1)
        ], axis=0)
        cv2.imwrite(save_path, img_mosaic)

# Apply augmentations to all images
for filename in os.listdir(original_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        original_path = os.path.join(original_folder, filename)
        for aug_type, folder in output_folders.items():
            output_path = os.path.join(folder, filename)
            augment_image(original_path, output_path, aug_type)

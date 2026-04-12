import os
import cv2 as cv
import numpy as np

# Image Zooming Function
def zoom_image(img, scale_factor, method='nearest'):
    """
    Zooms an image by a given scale factor s in (0, 10].
    Handles both nearest-neighbor and bilinear interpolation.
    """
    if not (0 < scale_factor <= 10):
        raise ValueError("Scale factor 's' must be in the range (0, 10]")

    # Calculate new dimensions
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    dimensions = (new_width, new_height)

    if method == 'nearest':
        interp_flag = cv.INTER_NEAREST
    elif method == 'bilinear':
        interp_flag = cv.INTER_LINEAR
    else:
        raise ValueError("Method must be 'nearest' or 'bilinear'")

    # Resize image
    zoomed_img = cv.resize(img, dimensions, interpolation=interp_flag)
    
    return zoomed_img

# Normalized SSD Calculation
def calculate_normalized_ssd(img1, img2):
    """
    Computes the Normalized Sum of Squared Difference (SSD) between two images.
    Images are cast to float32 to prevent overflow during squaring.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match! {img1.shape} vs {img2.shape}")

    # Convert to float32
    i1 = img1.astype(np.float32)
    i2 = img2.astype(np.float32)

    # Calculate SSD
    squared_diff = np.sum((i1 - i2) ** 2)
    
    norm_factor = np.sqrt(np.sum(i1 ** 2) * np.sum(i2 ** 2))
    
    if norm_factor == 0:
        return 0.0

    normalized_ssd = squared_diff / norm_factor
    return normalized_ssd

# Testing the Algorithm
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    small_image_name = 'im01small.png' 
    large_image_name = 'im01.png' 
    
    small_image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images/a1q8images', small_image_name))
    large_image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images/a1q8images', large_image_name))

    img_small = cv.imread(small_image_path, cv.IMREAD_GRAYSCALE)
    img_large_original = cv.imread(large_image_path, cv.IMREAD_GRAYSCALE)

    if img_small is None or img_large_original is None:
        print("Error: Could not load the images. Please check the filenames in the script.")
    else:
        s = img_large_original.shape[1] / img_small.shape[1]
        print(f"Calculated Scale Factor (s): {s:.2f}")

        # Scale up using both methods
        zoomed_nearest = zoom_image(img_small, scale_factor=s, method='nearest')
        zoomed_bilinear = zoom_image(img_small, scale_factor=s, method='bilinear')

        nssd_nearest = calculate_normalized_ssd(img_large_original, zoomed_nearest)
        nssd_bilinear = calculate_normalized_ssd(img_large_original, zoomed_bilinear)

        # Report Results
        print("\n--- Normalized SSD Results ---")
        print(f"Nearest-Neighbor NSSD: {nssd_nearest:.6f}")
        print(f"Bilinear NSSD:         {nssd_bilinear:.6f}")
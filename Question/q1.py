import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Getting absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', 'runway.png'))

# Load the image
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image. Checked exact path: {image_path}")
else:
    print(f"Successfully loaded image from: {image_path}")
    # Normalize the image to the range [0, 1] for processing
    r = img.astype(np.float32) / 255.0


    # (a)
    gamma_a = 0.5
    # Formula: s = r^gamma
    s_a = np.power(r, gamma_a)
    # Convert back to 8-bit [0, 255]
    img_gamma_a = np.uint8(s_a * 255)


    # (b)
    gamma_b = 2.0
    s_b = np.power(r, gamma_b)
    img_gamma_b = np.uint8(s_b * 255)


    # (c)
    r1 = 0.2
    r2 = 0.8

    s_c = np.zeros_like(r)

    # Apply the piecewise linear conditions:
    # Condition 1: s(r) = 0 for r < r1 (already 0 due to zeros_like)
    
    # Condition 2: s(r) = (r - r1) / (r2 - r1) for r1 <= r <= r2
    mask_mid = (r >= r1) & (r <= r2)
    s_c[mask_mid] = (r[mask_mid] - r1) / (r2 - r1)
    
    # Condition 3: s(r) = 1 for r > r2
    mask_high = r > r2
    s_c[mask_high] = 1.0

    # Convert back to 8-bit [0, 255]
    img_contrast_stretch = np.uint8(s_c * 255)


    # Plotting the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_gamma_a, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Gamma = 0.5')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(img_gamma_b, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Gamma = 2.0')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_contrast_stretch, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Contrast Stretching')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
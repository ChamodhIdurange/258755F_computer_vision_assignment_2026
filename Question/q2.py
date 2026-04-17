import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Safely load the image using the absolute path method we set up earlier
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', 'highlights_and_shadows.jpg'))

img_bgr = cv.imread(image_path)

if img_bgr is None:
    print(f"Error: Could not load image from: {image_path}")
else:
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)

    # Split the channels into L, a, and b
    l, a, b = cv.split(img_lab)
    l_norm = l.astype(np.float32) / 255.0
    
    gamma = 0.5
    
    l_corrected_norm = np.power(l_norm, gamma)
    l_corrected = np.uint8(l_corrected_norm * 255)
    lab_corrected = cv.merge((l_corrected, a, b))

    img_corrected_rgb = cv.cvtColor(lab_corrected, cv.COLOR_LAB2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original Image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Corrected Image
    axes[0, 1].imshow(img_corrected_rgb)
    axes[0, 1].set_title(f'Gamma Corrected (Gamma = {gamma})')
    axes[0, 1].axis('off')

    # Original Histogram
    axes[1, 0].hist(l.ravel(), bins=256, range=[0, 256], color='gray')
    axes[1, 0].set_title('Histogram of Original L Channel')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')

    # Corrected Histogram
    axes[1, 1].hist(l_corrected.ravel(), bins=256, range=[0, 256], color='gray')
    axes[1, 1].set_title('Histogram of Corrected L Channel')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
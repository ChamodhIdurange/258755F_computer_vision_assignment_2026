import os
import cv2 as cv
import matplotlib.pyplot as plt

# Load Noisy Image
script_dir = os.path.dirname(os.path.abspath(__file__))

noisy_image_name = 'salt_and_pepper_noise.jpg' 

image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', noisy_image_name))

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from: {image_path}")
else:
    # Apply Gaussian smoothing
    gaussian_filtered = cv.GaussianBlur(img, (5, 5), sigmaX=0)

    # Apply Median filtering
    median_filtered = cv.medianBlur(img, 5)

    # Plotting the Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original (Salt & Pepper Noise)')
    axes[0].axis('off')
    
    axes[1].imshow(gaussian_filtered, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Gaussian Smoothing (5x5)')
    axes[1].axis('off')
    
    axes[2].imshow(median_filtered, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Median Filtering (k=5)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
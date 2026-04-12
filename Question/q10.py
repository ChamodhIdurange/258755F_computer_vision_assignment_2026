import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Manual Bilateral Filter
def manual_bilateral_filter(img, diameter, sigma_s, sigma_r):
    """
    Applies a bilateral filter to a grayscale image.
    :param img: Grayscale input image
    :param diameter: Diameter of each pixel neighborhood (must be odd)
    :param sigma_s: Spatial standard deviation
    :param sigma_r: Range (intensity) standard deviation
    """

    if diameter % 2 == 0:
        diameter += 1
        
    pad = diameter // 2
    
    # Pad the image to handle borders
    padded_img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REFLECT).astype(np.float32)
    output_img = np.zeros_like(img, dtype=np.float32)
    
    ax = np.arange(-pad, pad + 1)
    xx, yy = np.meshgrid(ax, ax)
    spatial_weights = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_s**2))
    
    print("Applying manual bilateral filter... (this might take a few seconds)")
    
    # Iterate over every pixel in the original image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i : i + diameter, j : j + diameter]
            
            # Calculating the intensity differences between center pixel and neighborhood
            center_pixel_intensity = img[i, j]
            intensity_diff = region - center_pixel_intensity
            
            intensity_weights = np.exp(-(intensity_diff**2) / (2.0 * sigma_r**2))
            
            # Combine spatial and intensity weights
            combined_weights = spatial_weights * intensity_weights
            
            combined_weights /= np.sum(combined_weights)
            
            output_img[i, j] = np.sum(region * combined_weights)
            
    return np.uint8(np.clip(output_img, 0, 255))

# Applying the Filters (
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    image_name = 'jeniffer.jpg' 
    image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', image_name))

    # Load image in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image from: {image_path}")
    else:

        # Filter Parameters
        d = 7           
        sigma_s = 15.0 
        sigma_r = 30.0 

        # Gaussian smoothing
        gaussian_cv = cv.GaussianBlur(img, (d, d), sigmaX=sigma_s)

        # Bilateral filtering
        bilateral_cv = cv.bilateralFilter(img, d, sigmaColor=sigma_r, sigmaSpace=sigma_s)

        # Manual Bilateral filtering
        bilateral_manual = manual_bilateral_filter(img, d, sigma_s, sigma_r)

        # Plotting the Comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gaussian_cv, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title(f'Gaussian Blur (OpenCV)\nKernel={d}, $\sigma_s$={sigma_s}')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(bilateral_cv, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title(f'Bilateral Filter (OpenCV)\n$d$={d}, $\sigma_s$={sigma_s}, $\sigma_r$={sigma_r}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(bilateral_manual, cmap='gray', vmin=0, vmax=255)
        axes[1, 1].set_title('Bilateral Filter (Manual implementation)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
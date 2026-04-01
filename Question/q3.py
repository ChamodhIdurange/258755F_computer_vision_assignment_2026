import os
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def manual_histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    cdf = hist.cumsum()
    
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_normalized, 0).astype('uint8')
    img_equalized = cdf_final[img]
    
    return img_equalized, hist, cdf_final


script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', 'runway.png'))

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from: {image_path}")
else:
    print(f"Successfully loaded image from: {image_path}")
    
    img_eq, original_hist, cdf = manual_histogram_equalization(img)
    eq_hist, _ = np.histogram(img_eq.flatten(), 256, [0, 256])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original Image
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Runway Image')
    axes[0, 0].axis('off')
    
    # Equalized Image
    axes[0, 1].imshow(img_eq, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Manually Equalized Image')
    axes[0, 1].axis('off')
    
    # Original Histogram & CDF
    cdf_scaled = cdf * (original_hist.max() / cdf.max()) 
    axes[1, 0].plot(cdf_scaled, color='blue', label='CDF')
    axes[1, 0].hist(img.flatten(), 256, [0, 256], color='gray', label='Histogram')
    axes[1, 0].set_title('Original Histogram & CDF')
    axes[1, 0].legend(loc='upper left')
    
    # Equalized Histogram & new CDF
    cdf_eq = eq_hist.cumsum()
    cdf_eq_scaled = cdf_eq * (eq_hist.max() / cdf_eq.max())
    axes[1, 1].plot(cdf_eq_scaled, color='blue', label='CDF')
    axes[1, 1].hist(img_eq.flatten(), 256, [0, 256], color='gray', label='Histogram')
    axes[1, 1].set_title('Equalized Histogram & CDF')
    axes[1, 1].legend(loc='upper left')
    
    plt.tight_layout()
    
    # 4. Save the figure
    output_path = os.path.join(script_dir, 'q3_output.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Saved the results to: {output_path}")
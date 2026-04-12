import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', 'women_standing_in_dark.png')) 

img_bgr = cv.imread(image_path)

if img_bgr is None:
    print(f"Error: Could not load image from: {image_path}")
else:
    # Convert to Grayscale
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # Useing Otsu thresholding
    ret_otsu, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Report the resulting threshold value
    print(f"Calculated Otsu Threshold Value: {ret_otsu}")

    # Carrying out histogram equalization ONLY for the foreground
    hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
    
    cdf = hist.cumsum()
    
    cdf_masked = np.ma.masked_equal(cdf, 0)
    
    # Normalize the CDF to 0-255
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_normalized, 0).astype('uint8')
    
    # Apply the equalization mapping to the entire grayscale image
    img_eq_full = cdf_final[gray]
    
    # Create the final results
    result_img = gray.copy()
    np.copyto(result_img, img_eq_full, where=(mask == 255))

    # Plotting the results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Color Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Grayscale Image')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title(f'Foreground Mask (Otsu Threshold: {ret_otsu})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(result_img, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Equalized Foreground')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
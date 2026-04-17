import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filter(img, d0=30, gamma_l=0.3, gamma_h=1.5):
    """
    Applies Homomorphic Filtering to a grayscale image.
    :param img: Input grayscale image
    :param d0: Cutoff frequency
    :param gamma_l: Low-frequency gain (controls illumination, < 1)
    :param gamma_h: High-frequency gain (controls reflectance, > 1)
    """
    # Convert to float32 and Log transformation
    img_float = np.float32(img)
    img_log = np.log1p(img_float) # log(1 + x) avoids log(0) error
    
    # FFT and shift to center
    fft_img = np.fft.fft2(img_log)
    fft_shift = np.fft.fftshift(fft_img)
    
    # Create the High-Frequency Emphasis Filter H(u,v)
    rows, cols = img.shape
    center_row, center_col = rows // 2, cols // 2
    
    u = np.arange(rows) - center_row
    v = np.arange(cols) - center_col
    U, V = np.meshgrid(v, u)
    
    # Distance squared from the center
    D_squared = U**2 + V**2
    
    # Gaussian High Pass Filter formula
    H = (gamma_h - gamma_l) * (1 - np.exp(-D_squared / (2 * d0**2))) + gamma_l
    
    # Apply filter in frequency domain
    filtered_fft_shift = fft_shift * H
    
    # Inverse FFT Shift and IFFT
    filtered_fft = np.fft.ifftshift(filtered_fft_shift)
    img_filtered_log = np.fft.ifft2(filtered_fft)
    
    # Exponential Transformation (and take real part)
    img_filtered = np.expm1(np.real(img_filtered_log)) # exp(x) - 1
    
    # Normalize back to 0-255
    img_normalized = cv.normalize(img_filtered, None, 0, 255, cv.NORM_MINMAX)
    
    return np.uint8(img_normalized)

# Run the Filter
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    image_name = 'highlights_and_shadows.jpg' 
    image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', image_name))

    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image from: {image_path}")
    else:
        filtered_img = homomorphic_filter(img, d0=30, gamma_l=0.5, gamma_h=2.0)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(filtered_img, cmap='gray')
        axes[1].set_title('Homomorphic Filtered Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
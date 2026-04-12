import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Helper Function
def get_gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    
    # Create a 2D coordinate grid
    xx, yy = np.meshgrid(ax, ax)
    
    # Compute the Gaussian function 
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # Normalize the kernel so the sum of all elements equals 1
    return kernel / np.sum(kernel)

# Compute normalized 5x5 kernel (sigma=2)
kernel_5x5 = get_gaussian_kernel(size=5, sigma=2)
print("Manually Computed 5x5 Gaussian Kernel (sigma=2):\n")
print(np.round(kernel_5x5, 4)) # Rounded for cleaner console output

# Visualize a 51x51 Gaussian kernel as 3D surface
kernel_51x51 = get_gaussian_kernel(size=51, sigma=8)

# Create coordinate grids for 3D plotting
x = np.arange(-25, 26)
y = np.arange(-25, 26)
X, Y = np.meshgrid(x, y)

# Set up the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, kernel_51x51, cmap='viridis', linewidth=0, antialiased=True)

ax.set_title('3D Surface Plot of 51x51 Gaussian Kernel')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Weight / Height')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Apply Smoothing
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', 'runway.png'))

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from: {image_path}")
else:
    #  Apply manual Gaussian smoothing using cv.filter2D and custom kernel
    smoothed_manual = cv.filter2D(img, -1, kernel_5x5)

    # Apply Gaussian smoothing using OpenCV's built-in function
    smoothed_cv = cv.GaussianBlur(img, (5, 5), sigmaX=2)

    # Plotting the image comparisons
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    axes2[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes2[0].set_title('Original Image')
    axes2[0].axis('off')
    
    axes2[1].imshow(smoothed_manual, cmap='gray', vmin=0, vmax=255)
    axes2[1].set_title('Manual Smoothing (5x5, sigma=2)')
    axes2[1].axis('off')
    
    axes2[2].imshow(smoothed_cv, cmap='gray', vmin=0, vmax=255)
    axes2[2].set_title('OpenCV cv.GaussianBlur (5x5, sigma=2)')
    axes2[2].axis('off')
    
    plt.tight_layout()
    plt.show()
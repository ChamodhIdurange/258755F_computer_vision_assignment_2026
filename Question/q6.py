import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Helper Function
def get_dog_kernels(size, sigma):
    """Generates 2D Derivative of Gaussian kernels for X and Y directions."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    # Base Gaussian function
    G = np.exp(-(xx**2 + yy**2) / (2. * sigma**2)) / (2 * np.pi * sigma**2)
    
    # Derivatives using the formulas derived in part (a)
    Gx = -(xx / sigma**2) * G
    Gy = -(yy / sigma**2) * G
    
    return Gx, Gy

# Compute 5x5 kernels for sigma=2
Gx_5x5, Gy_5x5 = get_dog_kernels(size=5, sigma=2)

print("5x5 Derivative of Gaussian Kernel (X-direction, sigma=2):\n")
print(np.round(Gx_5x5, 4))
print("\n5x5 Derivative of Gaussian Kernel (Y-direction, sigma=2):\n")
print(np.round(Gy_5x5, 4))

# Visualize a 51x51 DoG kernel
Gx_51x51, _ = get_dog_kernels(size=51, sigma=8)

x = np.arange(-25, 26)
y = np.arange(-25, 26)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Gx_51x51, cmap='coolwarm', linewidth=0, antialiased=True)

ax.set_title('3D Surface Plot of 51x51 DoG Kernel (X-Direction)')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Weight')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Apply kernels and compare with Sobel
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', 'runway.png'))

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from: {image_path}")
else:
    # Apply Manual DoG Kernels 
    grad_x_manual = cv.filter2D(img, cv.CV_64F, Gx_5x5)
    grad_y_manual = cv.filter2D(img, cv.CV_64F, Gy_5x5)
    
    # Calculate gradient magnitude for the manual DoG
    mag_manual = cv.magnitude(grad_x_manual, grad_y_manual)
    mag_manual_disp = cv.normalize(mag_manual, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Apply OpenCV Sobel
    grad_x_sobel = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    grad_y_sobel = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    
    mag_sobel = cv.magnitude(grad_x_sobel, grad_y_sobel)
    mag_sobel_disp = cv.normalize(mag_sobel, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    axes2[0].imshow(img, cmap='gray')
    axes2[0].set_title('Original Image')
    axes2[0].axis('off')
    
    axes2[1].imshow(mag_manual_disp, cmap='gray')
    axes2[1].set_title('Manual DoG Magnitude (5x5, sigma=2)')
    axes2[1].axis('off')
    
    axes2[2].imshow(mag_sobel_disp, cmap='gray')
    axes2[2].set_title('OpenCV Sobel Magnitude (3x3)')
    axes2[2].axis('off')
    
    plt.tight_layout()
    plt.show()
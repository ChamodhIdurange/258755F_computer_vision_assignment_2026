import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the Image
script_dir = os.path.dirname(os.path.abspath(__file__))

image_name = 'einstein.png' 

image_path = os.path.abspath(os.path.join(script_dir, '..', 'a1images', image_name))

img = cv.imread(image_path)

if img is None:
    print(f"Error: Could not load image from: {image_path}")
else:
    # Convert BGR to RGB for correct color plotting in matplotlib
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    sharpening_kernel = np.array([[ 0, -1,  0],
                                  [-1,  5, -1],
                                  [ 0, -1,  0]])

    # Apply the spatial filter
    sharpened_img = cv.filter2D(img_rgb, -1, sharpening_kernel)

    # Plotting the Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(sharpened_img)
    axes[1].set_title('Sharpened Image (3x3 Kernel)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
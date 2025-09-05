import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('a1images/sapphire.jpg', cv2.IMREAD_GRAYSCALE)  # Replace 'fig9.jpg' with actual path

# Otsu's thresholding for segmentation 
_, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert mask 
binary_mask = cv2.bitwise_not(binary_mask) if np.mean(image[binary_mask == 0]) > np.mean(image[binary_mask == 255]) else binary_mask

# Original and binary mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image (Fig. 9)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

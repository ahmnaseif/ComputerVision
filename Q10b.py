import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('a1images/sapphire.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary_mask = cv2.bitwise_not(binary_mask) if np.mean(image[binary_mask == 0]) > np.mean(image[binary_mask == 255]) else binary_mask

# Morphological closing
kernel = np.ones((5, 5), np.uint8) 
filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# Original mask and filled mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filled_mask, cmap='gray')
plt.title('Filled Mask')
plt.axis('off')

plt.tight_layout()
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

image_b = cv2.imread('a1images/ricesalted.png', cv2.IMREAD_GRAYSCALE)
preprocessed_b = cv2.medianBlur(image_b, 5)
_, thresholded = cv2.threshold(preprocessed_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological operations
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)

closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

# Results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(thresholded, cmap='gray')
plt.title('Otsu Thresholded Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(opened, cmap='gray')
plt.title('After Opening (Remove Small Objects)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(closed, cmap='gray')
plt.title('After Closing (Fill Holes)')
plt.axis('off')

plt.tight_layout()
plt.show()


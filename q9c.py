import cv2
import numpy as np
import matplotlib.pyplot as plt


image_b = cv2.imread('a1images/ricesalted.png', cv2.IMREAD_GRAYSCALE)  
preprocessed_b = cv2.medianBlur(image_b, 5)  

# Otsu's thresholding
_, thresholded = cv2.threshold(preprocessed_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Original, preprocessed, and thresholded images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_b, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(preprocessed_b, cmap='gray')
plt.title('Preprocessed Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(thresholded, cmap='gray')
plt.title('Otsu Thresholded Image')
plt.axis('off')

plt.tight_layout()
plt.show()


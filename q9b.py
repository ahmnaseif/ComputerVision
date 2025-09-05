import cv2
import numpy as np
import matplotlib.pyplot as plt

image_b = cv2.imread('a1images/ricesalted.png', cv2.IMREAD_GRAYSCALE)  

# Median filter to remove salt-and-pepper noise
preprocessed_b = cv2.medianBlur(image_b, 5)

# Original and preprocessed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_b, cmap='gray')
plt.title('Original Image 8b (Salt-and-Pepper Noise)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(preprocessed_b, cmap='gray')
plt.title('Preprocessed Image 8b')
plt.axis('off')

plt.tight_layout()
plt.show()

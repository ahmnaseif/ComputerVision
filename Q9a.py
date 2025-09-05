import cv2
import numpy as np
import matplotlib.pyplot as plt


image_a = cv2.imread('a1images/rice.png', cv2.IMREAD_GRAYSCALE)  

# Gaussian blur to remove Gaussian noise
preprocessed_a = cv2.GaussianBlur(image_a, (5, 5), 1.5)

# Original and preprocessed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_a, cmap='gray')
plt.title('Original Image 8a (Gaussian Noise)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(preprocessed_a, cmap='gray')
plt.title('Preprocessed Image 8a')
plt.axis('off')

plt.tight_layout()
plt.show()


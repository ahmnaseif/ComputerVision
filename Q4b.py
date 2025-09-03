import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('a1images/spider.png')  
# Convert from BGR to RGB 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

# Convert to HSV color space and split
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)

# Intensity transformation function
sigma = 70
a = 0.5  
transformed_saturation = s + a * 128 * np.exp(-((s.astype(np.float32) - 128) ** 2) / (2 * sigma ** 2))
transformed_saturation = np.clip(transformed_saturation, 0, 255).astype(np.uint8)

# Display the original and transformed saturation planes
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(s, cmap='gray')
plt.title('Original Saturation Plane')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(transformed_saturation, cmap='gray')
plt.title(f'Transformed Saturation Plane (a = {a})')
plt.axis('off')

plt.tight_layout()
plt.show()
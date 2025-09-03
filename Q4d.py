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

# Apply transformation 
sigma = 70
a = 0.7  
transformed_s = s.astype(np.float32) + a * 128 * np.exp(-((s.astype(np.float32) - 128) ** 2) / (2 * sigma ** 2))
transformed_s = np.clip(transformed_s, 0, 255).astype(np.uint8)


recombined_hsv = cv2.merge((h, transformed_s, v))
recombined_image = cv2.cvtColor(recombined_hsv, cv2.COLOR_HSV2RGB)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(recombined_image)
plt.title('Recombined Enhanced Image')
plt.axis('off')

plt.tight_layout()
plt.show()
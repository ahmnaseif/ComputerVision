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


enhanced_hsv = cv2.merge((h, transformed_s, v))
enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

# Plot the transformation function
x = np.arange(0, 256)
y = x + a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2))
y = np.clip(y, 0, 255)

# Display 
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(enhanced_image)
plt.title('Vibrance-Enhanced Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.plot(x, y, 'b-')
plt.title('Intensity Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)

plt.tight_layout()
plt.show()
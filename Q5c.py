import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('a1images/jeniffer.jpg')  
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)

# Create mask 
_, mask = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY_INV)

# Extract foreground 
foreground = cv2.bitwise_and(image, image, mask=mask)

# Compute histogram of the foreground 
hist_foreground = cv2.calcHist([v], [0], mask, [256], [0, 256])

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(foreground)
plt.title('Foreground Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.plot(hist_foreground, color='blue')
plt.title('Histogram of Foreground')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()
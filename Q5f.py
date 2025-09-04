import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('a1images/jeniffer.jpg') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)

# Create mask
_, mask = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY_INV)

# Extract foreground and compute histogram
foreground = cv2.bitwise_and(image, image, mask=mask)
hist_foreground = cv2.calcHist([v], [0], mask, [256], [0, 256])

# Compute cumulative sum and normalize
cumulative_hist = np.cumsum(hist_foreground)
cumulative_hist_normalized = (cumulative_hist - cumulative_hist.min()) * 255 / (cumulative_hist.max() - cumulative_hist.min())

# Equalize the foreground
equalized_v = cv2.LUT(v, cumulative_hist_normalized.astype(np.uint8))
hsv_equalized = cv2.merge((h, s, equalized_v))
equalized_foreground = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2RGB)

# Extract background
mask_inv = cv2.bitwise_not(mask)
background = cv2.bitwise_and(image, image, mask=mask_inv)

# Combine foreground and background
final_image = cv2.add(background, cv2.bitwise_and(equalized_foreground, equalized_foreground, mask=mask))

# Display all images
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(h, cmap='gray')
plt.title('Hue Plane')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(s, cmap='gray')
plt.title('Saturation Plane')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(v, cmap='gray')
plt.title('Value Plane')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(mask, cmap='gray')
plt.title('Foreground Mask')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(final_image)
plt.title('Final Image with Equalized Foreground')
plt.axis('off')

plt.tight_layout()
plt.show()
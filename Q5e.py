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

# Create lookup table for equalization
equalized_v = cv2.LUT(v, cumulative_hist_normalized.astype(np.uint8))

# Merge with original h and s planes 
hsv_equalized = cv2.merge((h, s, equalized_v))
equalized_foreground = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2RGB)

# Apply mask 
final_equalized = cv2.bitwise_and(equalized_foreground, equalized_foreground, mask=mask)

# Display the equalized foreground
plt.figure(figsize=(8, 4))
plt.imshow(final_equalized)
plt.title('Histogram-Equalized Foreground')
plt.axis('off')
plt.show()
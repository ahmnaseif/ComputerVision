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

# Compute cumulative sum
cumulative_hist = np.cumsum(hist_foreground)

# Normalize cumulative histogram to [0, 255]
cumulative_hist_normalized = (cumulative_hist - cumulative_hist.min()) * 255 / (cumulative_hist.max() - cumulative_hist.min())

# Display the cumulative histogram
plt.figure(figsize=(8, 4))
plt.plot(cumulative_hist_normalized, color='green')
plt.title('Cumulative Histogram of Foreground')
plt.xlabel('Intensity')
plt.ylabel('Cumulative Frequency')
plt.grid(True)
plt.show()
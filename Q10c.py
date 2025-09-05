import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('a1images/sapphire.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary_mask = cv2.bitwise_not(binary_mask) if np.mean(image[binary_mask == 0]) > np.mean(image[binary_mask == 255]) else binary_mask
kernel = np.ones((5, 5), np.uint8)
filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# Connected components with statistics
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)

# Extract areas
areas_pixels = stats[1:, cv2.CC_STAT_AREA]  # Areas of sapphires (skip background)

# Filled mask with labeled components
colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
for label in range(1, num_labels): 
    colored_labels[labels == label] = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(filled_mask, cmap='gray')
plt.title('Filled Mask')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(colored_labels)
plt.title('New')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Number of sapphires: {len(areas_pixels)}")
print(f"Areas in pixels: {areas_pixels}")
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_b = cv2.imread('a1images/ricesalted.png', cv2.IMREAD_GRAYSCALE)
preprocessed_b = cv2.medianBlur(image_b, 5)
_, thresholded = cv2.threshold(preprocessed_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

num_rice_grains = num_labels - 1

# Result
colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
for label in range(num_labels):
    colored_labels[labels == label] = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(closed, cmap='gray')
plt.title('Morphologically Processed Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(colored_labels)
plt.title(f'Connected Components (Rice Grains: {num_rice_grains})')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Number of rice grains: {num_rice_grains}")
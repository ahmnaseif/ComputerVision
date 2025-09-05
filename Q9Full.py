import cv2
import numpy as np
import matplotlib.pyplot as plt

img_gauss = cv2.imread('a1images/rice.png', cv2.IMREAD_GRAYSCALE)
img_sp = cv2.imread('a1images/ricesalted.png', cv2.IMREAD_GRAYSCALE)

# Gaussian blur to remove Gaussian noise
gauss_denoised = cv2.GaussianBlur(img_gauss, (5,5), 0)

# Otsu threshold
_, gauss_thresh = cv2.threshold(gauss_denoised, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
gauss_morph = cv2.morphologyEx(gauss_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
gauss_morph = cv2.morphologyEx(gauss_morph, cv2.MORPH_CLOSE, kernel, iterations=2)


num_labels_g, labels_g, stats_g, _ = cv2.connectedComponentsWithStats(gauss_morph, connectivity=8)
count_gauss = num_labels_g - 1


sp_denoised = cv2.medianBlur(img_sp, 3)

# Otsu threshold
_, sp_thresh = cv2.threshold(sp_denoised, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Morphology
sp_morph = cv2.morphologyEx(sp_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sp_morph = cv2.morphologyEx(sp_morph, cv2.MORPH_CLOSE, kernel, iterations=2)


num_labels_sp, labels_sp, stats_sp, _ = cv2.connectedComponentsWithStats(sp_morph, connectivity=8)
count_sp = num_labels_sp - 1

# Results
plt.figure(figsize=(12,8))

plt.subplot(2,3,1); plt.imshow(img_gauss, cmap='gray'); plt.title("Gaussian Noise Input"); plt.axis('off')
plt.subplot(2,3,2); plt.imshow(gauss_morph, cmap='gray'); plt.title(f"Processed (Count={count_gauss})"); plt.axis('off')

plt.subplot(2,3,4); plt.imshow(img_sp, cmap='gray'); plt.title("Salt & Pepper Noise Input"); plt.axis('off')
plt.subplot(2,3,5); plt.imshow(sp_morph, cmap='gray'); plt.title(f"Processed (Count={count_sp})"); plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Rice grains in Gaussian noise image: {count_gauss}")
print(f"Rice grains in Salt-and-Pepper noise image: {count_sp}")

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('a1images/daisy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


mask = np.zeros(img.shape[:2], np.uint8)
rect = (20, 20, img.shape[1]-40, img.shape[0]-40)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

# Blur background
blurred = cv2.GaussianBlur(img, (25, 25), 0)

# Combine
enhanced = blurred.copy()
enhanced[mask2 == 1] = img[mask2 == 1]

# Show results
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(mask2, cmap='gray'); plt.title("Segmentation Mask"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(enhanced); plt.title("Enhanced Image"); plt.axis("off")
plt.show()

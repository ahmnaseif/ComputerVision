import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('a1images/daisy.jpg')  
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize mask
mask = np.zeros(img.shape[:2], np.uint8)

# Rectangle around the flower 
rect = (20, 20, img.shape[1]-40, img.shape[0]-40)  

# GrabCut Models
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# GrabCut
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# GrabCut output to binary mask
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

foreground = img * mask2[:, :, np.newaxis]
background = img * (1 - mask2[:, :, np.newaxis])

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(mask2, cmap='gray')
plt.title('Segmentation Mask')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(foreground)
plt.title('Foreground Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(background)
plt.title('Background Image')
plt.axis('off')

plt.tight_layout()
plt.show()


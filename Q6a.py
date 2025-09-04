import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('a1images/einstein.png', cv2.IMREAD_GRAYSCALE)

# Sobel kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]], dtype=np.float32)

# Apply Sobel using filter2D
grad_x = cv2.filter2D(img, -1, sobel_x)
grad_y = cv2.filter2D(img, -1, sobel_y)

# Gradient magnitude
grad_mag = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(grad_x, cmap='gray'); plt.title("Sobel X")
plt.subplot(1,3,2); plt.imshow(grad_y, cmap='gray'); plt.title("Sobel Y")
plt.subplot(1,3,3); plt.imshow(grad_mag, cmap='gray'); plt.title("Gradient Magnitude")
plt.show()

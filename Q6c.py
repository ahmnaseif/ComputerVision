import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('a1images/einstein.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load 'fig6.jpg'. Please check the path.")

# vertical blur
h1 = np.array([[1], [2], [1]], dtype=np.float32) 
# horizontal diff  
h2 = np.array([[-1, 0, 1]], dtype=np.float32)      

# vertical diff
v1 = np.array([[-1], [0], [1]], dtype=np.float32)
# horizontal blur  
v2 = np.array([[1, 2, 1]], dtype=np.float32)       


# For X direction
tmp_x = cv2.filter2D(img, cv2.CV_32F, h1)   # vertical pass
grad_x_sep = cv2.filter2D(tmp_x, cv2.CV_32F, h2)  # horizontal pass

# For Y direction
tmp_y = cv2.filter2D(img, cv2.CV_32F, v2)   # horizontal pass
grad_y_sep = cv2.filter2D(tmp_y, cv2.CV_32F, v1)  # vertical pass

# Gradient magnitude
grad_mag_sep = cv2.magnitude(grad_x_sep, grad_y_sep)


plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(grad_x_sep, cmap='gray'); plt.title("Separable Sobel X"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(grad_y_sep, cmap='gray'); plt.title("Separable Sobel Y"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(grad_mag_sep, cmap='gray'); plt.title("Separable Gradient Magnitude"); plt.axis('off')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('a1images/einstein.png', cv2.IMREAD_GRAYSCALE)


# Sobel kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)


def convolve2d(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    # Pad with zeros
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros((h, w), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

# Apply Sobel using custom convolution
grad_x_custom = convolve2d(img, sobel_x)
grad_y_custom = convolve2d(img, sobel_y)

# Gradient magnitude
grad_mag_custom = np.sqrt(grad_x_custom**2 + grad_y_custom**2)


plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(grad_x_custom, cmap='gray'); plt.title("Custom Sobel X"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(grad_y_custom, cmap='gray'); plt.title("Custom Sobel Y"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(grad_mag_custom, cmap='gray'); plt.title("Custom Gradient Magnitude"); plt.axis('off')
plt.show()

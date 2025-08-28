
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale brain image
image = cv2.imread('a1images/brain_proton_density_slice.png', cv2.IMREAD_GRAYSCALE)

# Transformation Functions

def accentuate_white_matter(img):
    transformed = np.copy(img).astype(np.float32)
    transformed[transformed < 150] = 50
    transformed[(transformed >= 150) & (transformed <= 200)] = 100 + 3 * (transformed[(transformed >= 150) & (transformed <= 200)] - 150)
    transformed[transformed > 200] = 255
    return np.clip(transformed, 0, 255).astype(np.uint8)

def accentuate_gray_matter(img):
    transformed = np.copy(img).astype(np.float32)
    transformed[transformed < 100] = 50
    transformed[(transformed >= 100) & (transformed <= 150)] = 100 + 3 * (transformed[(transformed >= 100) & (transformed <= 150)] - 100)
    transformed[transformed > 150] = 200
    return np.clip(transformed, 0, 255).astype(np.uint8)

# Apply transformations
white_matter_img = accentuate_white_matter(image)
gray_matter_img = accentuate_gray_matter(image)

#Plot Transformation Functions
x = np.arange(256)

# White matter transformation 
y_white = np.piecewise(x,
    [x < 150, (x >= 150) & (x <= 200), x > 200],
    [lambda x: 50, lambda x: 100 + 3 * (x - 150), lambda x: 255])

# Gray matter transformation 
y_gray = np.piecewise(x,
    [x < 100, (x >= 100) & (x <= 150), x > 150],
    [lambda x: 50, lambda x: 100 + 3 * (x - 100), lambda x: 200])

# Display Everything 
plt.figure(figsize=(15, 10))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Brain Slice')
plt.axis('off')

# White matter enhanced image
plt.subplot(2, 3, 2)
plt.imshow(white_matter_img, cmap='gray')
plt.title('White Matter Enhanced')
plt.axis('off')

# Gray matter enhanced image
plt.subplot(2, 3, 3)
plt.imshow(gray_matter_img, cmap='gray')
plt.title('Gray Matter Enhanced')
plt.axis('off')

# White matter transformation plot
plt.subplot(2, 3, 4)
plt.plot(x, y_white, color='blue')
plt.title('White Matter Transformation Curve')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)

# Gray matter transformation plot
plt.subplot(2, 3, 5)
plt.plot(x, y_gray, color='orange')
plt.title('Gray Matter Transformation Curve')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)

# Histogram of original image
plt.subplot(2, 3, 6)
plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
plt.title('Original Image Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
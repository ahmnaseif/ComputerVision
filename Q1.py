import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('a1images/emma.jpg', cv2.IMREAD_GRAYSCALE)  


def apply_transformation(image):
    # Create a copy 
    transformed = np.copy(image).astype(np.float32)
    
    # Apply transformation
    transformed[transformed <= 50] = 2 * transformed[transformed <= 50]
    transformed[(transformed > 50) & (transformed <= 150)] = 100
    transformed[transformed > 150] = 100 + (155 / 105) * (transformed[transformed > 150] - 150)
    
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    
    return transformed

# Apply transformation
transformed_image = apply_transformation(image)

# Display the original and transformed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(transformed_image, cmap='gray')
plt.title('Transformed Image')
plt.axis('off')

plt.show()


# Save the transformed image 
cv2.imwrite('transformed_fig1b.jpg', transformed_image)
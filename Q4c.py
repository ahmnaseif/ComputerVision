import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('a1images/spider.png')  
# Convert from BGR to RGB 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

# Convert to HSV color space and split
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)

# Intensity transformation function
sigma = 70

# Apply transformation 
def preview_vibrance(a):
    transformed_s = s.astype(np.float32) + a * 128 * np.exp(-((s.astype(np.float32) - 128) ** 2) / (2 * sigma ** 2))
    transformed_s = np.clip(transformed_s, 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge((h, transformed_s, v))
    enhanced = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    return enhanced

a_values = [0.3, 0.5, 0.7, 0.9]
plt.figure(figsize=(15, 5))
for i, a in enumerate(a_values, 1):
    enhanced = preview_vibrance(a)
    plt.subplot(1, len(a_values) + 1, i)
    plt.imshow(enhanced)
    plt.title(f'a = {a}')
    plt.axis('off')

# Comparison
plt.subplot(1, len(a_values) + 1, len(a_values) + 1)
plt.imshow(image)
plt.title('Original')
plt.axis('off')

plt.tight_layout()
plt.show()

selected_a = 0.7
print(f"Visually pleasing value of a: {selected_a}")
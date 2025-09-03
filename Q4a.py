import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('a1images/spider.png')  
# Convert from BGR to RGB 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Split into hue, saturation, and value planes
hue, saturation, value = cv2.split(hsv_image)

# Display 
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(hue, cmap='gray')
plt.title('Hue Plane')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(saturation, cmap='gray')
plt.title('Saturation Plane')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(value, cmap='gray')
plt.title('Value Plane')
plt.axis('off')

plt.tight_layout()
plt.show()
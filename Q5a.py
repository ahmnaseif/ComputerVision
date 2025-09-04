import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('a1images/jeniffer.jpg')  
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  


hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)

# Display the planes in grayscale
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(h, cmap='gray')
plt.title('Hue Plane')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(s, cmap='gray')
plt.title('Saturation Plane')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(v, cmap='gray')
plt.title('Value Plane')
plt.axis('off')

plt.tight_layout()
plt.show()
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('a1images/jeniffer.jpg')  
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)

 # Invert to get foreground as white
_, mask = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY_INV) 

# Display the mask
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(v, cmap='gray')
plt.title('Value Plane')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('Foreground Mask')
plt.axis('off')

plt.tight_layout()
plt.show()
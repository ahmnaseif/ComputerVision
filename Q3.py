import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('a1images/highlights_and_shadows.jpg')  
# Convert from BGR to RGB 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

# Convert to L*a*b* color space
lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# Split into L, a, b channels
l_channel, a_channel, b_channel = cv2.split(lab_image)

# Apply gamma correction to L channel
gamma = 0.5  
l_corrected = np.power(l_channel / 255.0, gamma) * 255.0
l_corrected = l_corrected.astype(np.uint8)

# Merge the corrected L channel with original a and b channels
lab_corrected = cv2.merge((l_corrected, a_channel, b_channel))

# Convert back to RGB for display
corrected_image = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

# Compute histograms
hist_original = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
hist_corrected = cv2.calcHist([l_corrected], [0], None, [256], [0, 256])

# Plotting
plt.figure(figsize=(12, 5))

# Original and corrected images
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(corrected_image)
plt.title(f'Gamma Corrected Image (γ = {gamma})')
plt.axis('off')

plt.show()

# Plot histograms
plt.figure(figsize=(10, 5))
plt.plot(hist_original, color='blue', label='Original Histogram')
plt.plot(hist_corrected, color='orange', label=f'Corrected Histogram (γ = {gamma})')
plt.title('Histograms of L Channel')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
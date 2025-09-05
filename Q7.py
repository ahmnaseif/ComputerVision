import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def clamp(v, low, high):
    return max(low, min(high, v))

# Nearest-neighbor zoom
def zoom_nearest(img, s):
    h_in, w_in = img.shape[:2]
    h_out = int(np.round(h_in * s))
    w_out = int(np.round(w_in * s))
    if img.ndim == 2:
        out = np.zeros((h_out, w_out), dtype=img.dtype)
    else:
        out = np.zeros((h_out, w_out, img.shape[2]), dtype=img.dtype)

    for i_out in range(h_out):
        y = i_out / s
        i_in = int(np.round(y))
        i_in = clamp(i_in, 0, h_in - 1)
        for j_out in range(w_out):
            x = j_out / s
            j_in = int(np.round(x))
            j_in = clamp(j_in, 0, w_in - 1)
            out[i_out, j_out] = img[i_in, j_in]
    return out

# Bilinear zoom
def zoom_bilinear(img, s):
    h_in, w_in = img.shape[:2]
    h_out = int(np.round(h_in * s))
    w_out = int(np.round(w_in * s))

    is_uint8 = (img.dtype == np.uint8)
    dtype = np.float32 if is_uint8 else img.dtype

    if img.ndim == 2:
        out = np.zeros((h_out, w_out), dtype=dtype)
    else:
        out = np.zeros((h_out, w_out, img.shape[2]), dtype=dtype)

    for i_out in range(h_out):
        y = i_out / s
        i0 = int(np.floor(y))
        dy = y - i0
        i1 = clamp(i0 + 1, 0, h_in - 1)

        for j_out in range(w_out):
            x = j_out / s
            j0 = int(np.floor(x))
            dx = x - j0
            j1 = clamp(j0 + 1, 0, w_in - 1)

            if img.ndim == 2:
                I00 = float(img[i0, j0])
                I10 = float(img[i1, j0])
                I01 = float(img[i0, j1])
                I11 = float(img[i1, j1])
                val = (1 - dy) * (1 - dx) * I00 + dy * (1 - dx) * I10 + (1 - dy) * dx * I01 + dy * dx * I11
                out[i_out, j_out] = val
            else:
                for c in range(img.shape[2]):
                    I00 = float(img[i0, j0, c])
                    I10 = float(img[i1, j0, c])
                    I01 = float(img[i0, j1, c])
                    I11 = float(img[i1, j1, c])
                    val = (1 - dy) * (1 - dx) * I00 + dy * (1 - dx) * I10 + (1 - dy) * dx * I01 + dy * dx * I11
                    out[i_out, j_out, c] = val

    if is_uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# Normalized SSD
def normalized_ssd(img_ref, img_test):
    ref = img_ref.astype(np.float64) / 255.0 if img_ref.dtype == np.uint8 else img_ref
    test = img_test.astype(np.float64) / 255.0 if img_test.dtype == np.uint8 else img_test
    ssd = np.sum((ref - test) ** 2)
    n = np.prod(ref.shape)  
    return ssd / n


if __name__ == "__main__":
    
    small_path = "a1images/a1q8images/taylor_very_small.jpg"  
    orig_path = "a1images/a1q8images/taylor.jpg"  


    small = cv2.imread(small_path, cv2.IMREAD_UNCHANGED)
    orig = cv2.imread(orig_path, cv2.IMREAD_UNCHANGED)

    # Determine scale factor
    scale = orig.shape[1] / small.shape[1]  # Width-based scale
    
    # Grayscale 
    if len(small.shape) == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if len(orig.shape) == 3:
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # Apply zoom
    up_nn = zoom_nearest(small, scale)
    up_bl = zoom_bilinear(small, scale)

    # Resize 
    orig_resized = cv2.resize(orig, (up_nn.shape[1], up_nn.shape[0]), interpolation=cv2.INTER_AREA)

    # Compute SSD 
    nssd_nn = normalized_ssd(up_nn, orig_resized)
    nssd_bl = normalized_ssd(up_bl, orig_resized)

    print(f"Normalized SSD (Nearest Neighbor): {nssd_nn:.4f}")
    print(f"Normalized SSD (Bilinear): {nssd_bl:.4f}")

    # Convert to RGB for display 
    orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2RGB)
    nn_rgb = cv2.cvtColor(up_nn, cv2.COLOR_GRAY2RGB)
    bl_rgb = cv2.cvtColor(up_bl, cv2.COLOR_GRAY2RGB)

    # Show results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(orig_rgb); plt.title("Reference Original"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(nn_rgb); plt.title(f"Nearest Neighbor (s={scale:.1f})"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(bl_rgb); plt.title(f"Bilinear (s={scale:.1f})"); plt.axis('off')
    plt.tight_layout()
    plt.show()

import numpy as np
import cv2
import pywt
import os
from matplotlib import pyplot as plt

# Temporary path configuration, will change later based on the Watermarked image action / location in the storage
base_path = "./utils/"
watermarked_img_path = os.path.join(base_path, "watermarked_product_img", "watermarked_landscape1.JPG")
sideinfo_path = os.path.join(base_path, "side_info", "kitshop_logo_sideinfo.npz")
watermark_img_path = os.path.join(base_path, "watermark_images", "kitshop_logo.jpg")
output_wm_path = os.path.join(base_path, "extracted_watermark", "extracted_kitshop_logo.png")

block_size = 32
alpha = 0.05

# Load the watermarked image and convert to YUV color space
img = cv2.imread(watermarked_img_path)
if img is None:
    raise FileNotFoundError(f"Cannot load watermarked image: {watermarked_img_path}")
img = cv2.resize(img, (512, 512))
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y = np.float32(img_yuv[:, :, 0])

# Load the side info components for watermark extraction
data = np.load(sideinfo_path, allow_pickle=True)
Uwm = data["Uwm"]
Swm = data["Swm"]
Vwm = data["Vwm"]
blocks = data["blocks"]

# Extract the watermark from the high-textured blocks using DWT and SVD *Maybe the main problem is here*
wm_blocks = []

for energy, i, j in blocks:
    i, j = int(i), int(j)
    block = Y[i:i+block_size, j:j+block_size]
    if block.shape != (block_size, block_size):
        continue
    coeffs = pywt.dwt2(block, 'haar')
    LL, (LH, HL, HH) = coeffs
    U, S, V = np.linalg.svd(LL)
    Swm_extracted = (S - Swm[:len(S)]) / alpha
    wm_block = np.dot(Uwm[:, :len(S)], np.dot(np.diag(Swm_extracted), Vwm[:len(S), :]))
    wm_blocks.append(wm_block)

# Average the extracted watermark blocks the rebuild the watermark image
wm_avg = np.mean(wm_blocks, axis=0)
wm_avg = np.clip(wm_avg, 0, 255).astype(np.uint8)

# Save the extracted watermark image
os.makedirs(os.path.dirname(output_wm_path), exist_ok=True)
cv2.imwrite(output_wm_path, wm_avg)
print(f"Extracted watermark saved to: {output_wm_path}")

plt.imshow(wm_avg, cmap='gray')
plt.title("Extracted Watermark")
plt.axis("off")
plt.show()

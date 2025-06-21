
import numpy as np
import cv2
import pywt
import os
from scipy.ndimage import sobel

# Temporary path configuration, will change later based on the Input Image action
base_path = "./utils/"
original_img_path = os.path.join(base_path, "original_images", "landscape1.JPG")
watermark_img_path = os.path.join(base_path, "watermark_images", "kitshop_logo.jpg")
output_img_path = os.path.join(base_path, "watermarked_product_img", "watermarked_landscape1.JPG")
sideinfo_path = os.path.join(base_path, "side_info", "kitshop_logo_sideinfo.npz")

block_size = 32 # Size of the blocks to be divided using DWT
alpha = 0.05  # strength factor, tested with 0.05, 0.1, 0.2 but not good
num_blocks = 16  # number of high-texture blocks to embed watermark (tested with 16, 32, 64 but not good)

# Load the original color image and watermark
img = cv2.imread(original_img_path)
if img is None:
    raise FileNotFoundError(f"Cannot load image: {original_img_path}")
img = cv2.resize(img, (512, 512))
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y = np.float32(img_yuv[:, :, 0])  # luminance only

wm = cv2.imread(watermark_img_path, cv2.IMREAD_GRAYSCALE)
if wm is None:
    raise FileNotFoundError(f"Cannot load watermark: {watermark_img_path}")
wm = cv2.resize(wm, (block_size, block_size))
wm = np.float32(wm)

# Save the component-wise of the SVD algo for the watermark extraction
Uwm, Swm, Vwm = np.linalg.svd(wm)

# Detect the high-textured blocks in the input img using Sobel operator
sobel_energy = sobel(Y, axis=0)**2 + sobel(Y, axis=1)**2

h, w = Y.shape
blocks = []
for i in range(0, h, block_size):
    for j in range(0, w, block_size):
        block = sobel_energy[i:i+block_size, j:j+block_size]
        if block.shape == (block_size, block_size):
            energy = np.sum(block)
            blocks.append((energy, i, j))

# Select top N textured blocks
blocks.sort(reverse=True)
selected_blocks = blocks[:num_blocks]

# Embed the watermark into the selected blocks using DWT and SVD
Y_embed = Y.copy()

for energy, i, j in selected_blocks:
    block = Y[i:i+block_size, j:j+block_size]
    coeffs = pywt.dwt2(block, 'haar')

    #Tried HL band here but worse
    LL, (LH, HL, HH) = coeffs 

    #SVD on LL band *While using LL band, the overall watermarked image quality is better, less distortion
    U, S, V = np.linalg.svd(LL)
    k = S.shape[0]
    S_embed = S + alpha * Swm[:k]
    LL_wm = np.dot(U, np.dot(np.diag(S_embed), V))
    coeffs_wm = (LL_wm, (LH, HL, HH))
    block_wm = pywt.idwt2(coeffs_wm, 'haar')
    Y_embed[i:i+block_size, j:j+block_size] = block_wm

# Merge back Y color channel
img_yuv[:, :, 0] = np.clip(Y_embed, 0, 255)
img_output = cv2.cvtColor(img_yuv.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

# Save the watermarked image
os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
cv2.imwrite(output_img_path, img_output)

# Save side info for extraction, just adding Swm to try to improve extraction quality
os.makedirs(os.path.dirname(sideinfo_path), exist_ok=True)
np.savez(sideinfo_path, Uwm=Uwm, Swm=Swm, Vwm=Vwm, blocks=selected_blocks)


print(f"Watermarked image saved to: {output_img_path}")
print(f"Side info saved to: {sideinfo_path}")

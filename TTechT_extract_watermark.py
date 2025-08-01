
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import pywt
import os

# Load the img paths, final path will be changed later based on UI flow and the database location
original_image_path = "./utils/original_images/product9.jpg"
watermark_image_path = "./utils/watermark_images/kiet_logo.jpg"
# Watermarked image path and Attacked image path
#watermarked_image_path = "./utils/watermarked_product_img/watermarked_product5.jpg"
watermarked_image_path = "./utils/attacked_images/product9_attacked_scale.jpg"
extracted_watermark_path = "./utils/extracted_attacked_watermark/extracted_attacked_product9_scale.jpg"
# ****The scaling factor must be the same with the embedding
alpha = 0.6
wavelet_name = "haar"

# Load images
original_image = Image.open(original_image_path).convert("RGB")
watermark_image = Image.open(watermark_image_path).convert("RGB").resize(original_image.size)
watermarked_image = Image.open(watermarked_image_path).convert("RGB").resize(original_image.size)

orig_r, orig_g, orig_b = [np.float64(c) for c in original_image.split()]
watermark_r, watermark_g, watermark_b = [np.float64(c) for c in watermark_image.split()]
orig_r_modifier, orig_g_modifier, orig_b_modifier = [np.float64(c) for c in watermarked_image.split()]

def extract_watermark_channel(orig_modifier_channel, orig_channel, watermark_channel, alpha=0.6, cname=""):
    with tqdm(total=100,
              desc=f"Extracting {cname} channel",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:

        LL_modifier, (LH_modifier, HL_modifier, HH_modifier) = pywt.dwt2(orig_modifier_channel, wavelet_name)
        LL_orig, _ = pywt.dwt2(orig_channel, wavelet_name)
        LL_watermarked_orig, (LH_watermark, HL_watermark, HH_watermark) = pywt.dwt2(watermark_channel, wavelet_name)
        pbar.update(30)

        U_orig_mod, S_orig_mod, V_orig_mod = np.linalg.svd(LL_modifier, full_matrices=False)
        U_orig, S_orig, _    = np.linalg.svd(LL_orig, full_matrices=False)
        U_watermark, S_watermark, V_watermark = np.linalg.svd(LL_watermarked_orig, full_matrices=False)
        pbar.update(50)

        S_watermark_extract = (S_orig_mod - S_orig) / alpha
        LL_watermark_extract = U_watermark @ np.diag(S_watermark_extract) @ V_watermark
        pbar.update(10)

        watermark_coeffs_extract = (LL_watermark_extract, (LH_watermark, HL_watermark, HH_watermark))
        watermark_channel_extract = pywt.idwt2(watermark_coeffs_extract, wavelet_name)
        pbar.update(10)

    return watermark_channel_extract

watermark_r_extract = extract_watermark_channel(orig_r_modifier, orig_r, watermark_r, alpha, "Red")
watermark_g_extract = extract_watermark_channel(orig_g_modifier, orig_g, watermark_g, alpha, "Green")
watermark_b_extract = extract_watermark_channel(orig_b_modifier, orig_b, watermark_b, alpha, "Blue")

def to_uint8(mat):
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

r8, g8, b8 = map(to_uint8, (watermark_r_extract, watermark_g_extract, watermark_b_extract))
extracted_watermark_rgb = Image.merge("RGB", (Image.fromarray(r8), Image.fromarray(g8), Image.fromarray(b8)))

os.makedirs(os.path.dirname(extracted_watermark_path), exist_ok=True)
extracted_watermark_rgb.save(extracted_watermark_path)
extracted_watermark_rgb.show()
print(f"Extracted watermark saved to: {extracted_watermark_path}")

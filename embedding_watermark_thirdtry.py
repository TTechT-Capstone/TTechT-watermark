from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import pywt                    

# Load host & watermark images, the final path will be changed later 
original_image_path      = "./images/inputs/original_image.jpg"
watermark_image_path = "./images/inputs/watermark_image.jpg"

original_image  = Image.open(original_image_path).convert("RGB")
watermark_image = Image.open(watermark_image_path).convert("RGB")

# resize watermark img
watermark_image = watermark_image.resize(original_image.size)

# Split into channels → numpy float64
orig_r, orig_g, orig_b = [np.float64(c) for c in original_image.split()]
watermark_r, watermark_g, watermark_b = [np.float64(c) for c in wm_img_in.split()]

def embed_watermark(orig_channel, wm_channel, alpha=0.6, cname=""):
    """
    Embed watermark_channel into orig_channel (single color plane)
    using 1-level Haar DWT + SVD on the LL sub-band.
    """
    with tqdm(total=100,
              desc=f"Embedding in {cname} channel",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:

        # DWT
        LL_orig, (LH_orig, HL_orig, HH_orig) = pywt.dwt2(orig_channel, 'haar')
        LL_wm, (LH_wm, HL_wm, HH_wm) = pywt.dwt2(wm_channel,  'haar')
        pbar.update(25)

        # SVD on LL sub-bands
        U_orig, S_orig, V_orig = np.linalg.svd(LL_orig, full_matrices=False)
        U_wm, S_wm, V_wm = np.linalg.svd(LL_wm, full_matrices=False)
        pbar.update(25)

        # ---- Embed watermark singular values ----
        S_modifier = S_orig + alpha * S_wm
        LL_modifier = (U_orig @ np.diag(S_modifier) @ V_orig)
        pbar.update(25)

        # Inverse DWT
        coeffs_modifier = (LL_modifier, (LH_orig, HL_orig, HH_orig))
        watermarked_channel = pywt.idwt2(coeffs_modifier, 'haar')
        pbar.update(25)

    return watermarked_channel

alpha = 0.6   # Scaling factor
watermark_r_modifier = embed_watermark(orig_r, watermark_r, alpha, "Red")
watermark_g_modifier = embed_watermark(orig_g, watermark_g, alpha, "Green")
watermark_b_modifier = embed_watermark(orig_b, watermark_b, alpha, "Blue")


# Normalize back to 0-255 uint8 and save
def normalize_uint8(mat):
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

orig_r8 = norm_uint8(watermark_r_modifier)
orig_g8 = norm_uint8(watermark_g_modifier)
orig_b8 = norm_uint8(watermark_b_modifier)

watermarked_rgb = Image.merge("RGB",
                              (Image.fromarray(orig_r8),
                               Image.fromarray(orig_g8),
                               Image.fromarray(orig_b8)))
out_path = "./images/outputs/watermarked_product1.jpg"
watermarked_rgb.save(out_path)
watermarked_rgb.show()
print(f"Watermarked image saved to {out_path}")

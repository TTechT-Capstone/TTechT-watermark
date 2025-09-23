from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import pywt                    
import json, os
from pathlib import Path
# Load the img paths, final path will be changed later based on UI flow and the database location
original_image_path      = "./utils/original_images/product5.jpg"
watermark_image_path = "./utils/watermark_images/origity_logo.jpg"

original_image  = Image.open(original_image_path).convert("RGB")
watermark_image = Image.open(watermark_image_path).convert("RGB")

#Variables
wavelet_name = "haar"  # Wavelet type for DWT
# Resize watermark to match original image size
watermark_image = watermark_image.resize(original_image.size)

# Split the image into three color channels → numpy float64
orig_r, orig_g, orig_b = [np.float64(c) for c in original_image.split()]
watermark_r, watermark_g, watermark_b = [np.float64(c) for c in watermark_image.split()]

def embed_watermark(orig_channel, wm_channel, alpha=0.6, cname=""):
    """
    Embed the watermark imame (watermark_channel) into orig_channel (single color plane)
    Here we are using 1-level Haar DWT + SVD on the LL sub-band.
    """
    with tqdm(total=100,
              desc=f"Embedding in {cname} channel",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:

        # Implement DWT algorithm on the original and watermark channels
        LL_orig, (LH_orig, HL_orig, HH_orig) = pywt.dwt2(orig_channel, wavelet_name) # LL is the low-frequency sub-band -> Highest embedded quality
        LL_wm, (LH_wm, HL_wm, HH_wm) = pywt.dwt2(wm_channel, wavelet_name)
        pbar.update(25)

        # Implement SVD algorithm on LL sub-bands
        U_orig, S_orig, V_orig = np.linalg.svd(LL_orig, full_matrices=False)
        U_wm, S_wm, V_wm = np.linalg.svd(LL_wm, full_matrices=False)
        pbar.update(25)

        # Embedding the watermark
        S_modifier = S_orig + alpha * S_wm
        LL_modifier = (U_orig @ np.diag(S_modifier) @ V_orig)
        pbar.update(25)

        # Reconstruct the modified channel using Inverse DWT
        coeffs_modifier = (LL_modifier, (LH_orig, HL_orig, HH_orig))
        watermarked_channel = pywt.idwt2(coeffs_modifier, wavelet_name)
        pbar.update(25)

    return watermarked_channel, S_orig, LL_orig.shape

alpha = 0.6   # Scaling factor
watermark_r_modifier, S_val_R, LL_shape_R = embed_watermark(orig_r, watermark_r, alpha, "Red")
watermark_g_modifier, S_val_G, LL_shape_G = embed_watermark(orig_g, watermark_g, alpha, "Green")
watermark_b_modifier, S_val_B, LL_shape_B = embed_watermark(orig_b, watermark_b, alpha, "Blue")

# Normalize back to 0-255 uint8 and save
def normalize_uint8(mat):
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

orig_r8 = normalize_uint8(watermark_r_modifier)
orig_g8 = normalize_uint8(watermark_g_modifier)
orig_b8 = normalize_uint8(watermark_b_modifier)

watermarked_rgb = Image.merge("RGB",
                              (Image.fromarray(orig_r8),
                               Image.fromarray(orig_g8),
                               Image.fromarray(orig_b8)))
# Output path will be changed later based on the database location
out_path = "./utils/watermarked_product_img/5.jpg"

# Prevent save errors in API/server
os.makedirs(os.path.dirname(out_path), exist_ok=True)
# Save the watermarked image
watermarked_rgb.save(out_path)

# Remove this show line while integrate to website because it can hang Flask API runs headless
# watermarked_rgb.show()
print(f"Watermarked image saved to {out_path}")

# Dump side info for extraction in e-commerce website
meta = {
    "wm_params": { "alpha": float(alpha), "wavelet": wavelet_name, "channels": "RGB" },

    "canonical_size": list(watermarked_rgb.size),

    # Where this watermarked file was written (handy for logging/DB ingestion)
    "output_path": out_path,

    "ll_shapes": { "R": list(LL_shape_R), "G": list(LL_shape_G), "B": list(LL_shape_B) },
    "host_S": {
        "R": [float(x) for x in S_val_R.tolist()],
        "G": [float(x) for x in S_val_G.tolist()],
        "B": [float(x) for x in S_val_B.tolist()]
    },
    "watermark_ref": {
        "path": str(Path(watermark_image_path)),
        "resized_to": list(original_image.size)
    }
}
# Save metadata to a JSON file
meta_path = os.path.splitext(out_path)[0] + ".wm.json"
os.makedirs(os.path.dirname(meta_path), exist_ok=True)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[embed] Saved side-info JSON -> {meta_path}")

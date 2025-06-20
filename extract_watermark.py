import cv2
import numpy as np
import pywt
import os

def extract_watermark(user_id, product_img_filename):
    #Initialize File paths
    base_path = "./utils/"
    product_img_path = os.path.join(base_path, "original_images", product_img_filename)
    watermarked_img_path = os.path.join(base_path, "watermarked_product_img", f"watermarked_{product_img_filename}")
    extracted_watermark_path = os.path.join(base_path, "extracted_watermark", f"extracted_{user_id}.jpg")

    #Read original and watermarked images 
    original_image = cv2.imread(product_img_path)
    watermarked_image = cv2.imread(watermarked_img_path)

    original_image = cv2.resize(original_image, (512, 512)).astype(np.float32)
    watermarked_image = cv2.resize(watermarked_image, (512, 512)).astype(np.float32)

    alpha = 0.05  #Used the same watermark strength factor during embedding

    extracted_watermark = np.zeros((256, 256, 3), dtype=np.float32)
    #Separate each color channel and process them independently
    for ch in range(3):
        coeffs_orig = pywt.dwt2(original_image[:, :, ch], 'haar')
        LL_orig, (LH_orig, HL_orig, HH_orig) = coeffs_orig
        U_orig, S_orig, V_orig = np.linalg.svd(LL_orig)

        coeffs_wm = pywt.dwt2(watermarked_image[:, :, ch], 'haar')
        LL_wm, (LH_wm, HL_wm, HH_wm) = coeffs_wm
        U_wm, S_wm, V_wm = np.linalg.svd(LL_wm)

        # Recover watermark singular values
        Swm_extracted = (S_wm - S_orig) / alpha

        # Reconstruct watermark LL approximation
        LL_wm_recovered = np.dot(U_orig, np.dot(np.diag(Swm_extracted), V_orig))
        LL_wm_resized = cv2.resize(LL_wm_recovered, (256, 256))
        extracted_watermark[:, :, ch] = LL_wm_resized

    extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(extracted_watermark_path), exist_ok=True)
    cv2.imwrite(extracted_watermark_path, extracted_watermark)
    print(f"Extracted watermark saved: {extracted_watermark_path}")

    return extracted_watermark

#Test case:
extract_watermark(user_id="kitshop", product_img_filename="landscape1.jpg")

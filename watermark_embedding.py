import cv2
import numpy as np
import pywt
import os
import matplotlib.pyplot as plt

#Embedding digital watermark by using DWT and SVD
def embed_watermark(user_id, product_img_filename):
    #Initialize File paths
    base_path = "./utils/"
    product_img_path = os.path.join(base_path, "original_images", product_img_filename)
    watermark_img_path = os.path.join(base_path, "watermark_images", f"{user_id}.jpg")
    output_img_path = os.path.join(base_path, "watermarked_product_img", f"watermarked_{product_img_filename}")

    #Read the color of the original product image
    original_image = cv2.imread(product_img_path)
    original_image = cv2.resize(original_image, (512, 512))
    original_image = np.float32(original_image)

    #Read the shop's watermark image (size 256x256 perform better than 32x32 right now)
    watermark_image = cv2.imread(watermark_img_path)
    watermark_image = cv2.resize(watermark_image, (256, 256))
    watermark_image = np.float32(watermark_image)
    #Ensure watermark is in the same color space as original image
    alpha = 0.05  #Define watermark strength factor
    watermarked_image = np.zeros_like(original_image)

    #Now I will separate each color channel and process them independently
    for ch in range(3):
        #Apply DWT algorithm to product image channel
        coeffs = pywt.dwt2(original_image[:, :, ch], 'haar')
        LL, (LH, HL, HH) = coeffs
        Uc, Sc, Vc = np.linalg.svd(LL)

        #Apply SVD to watermark channel
        watermark_channel = watermark_image[:, :, ch]
        Uwm, Swm, Vwm = np.linalg.svd(watermark_channel)

        #Embed watermark singular values
        Sc_embed = Sc + alpha * Swm

        #Reconstruct watermarked image LL layer
        LL_new = np.dot(Uc, np.dot(np.diag(Sc_embed), Vc))
        coeffs_new = (LL_new, (LH, HL, HH))
        watermarked_channel = pywt.idwt2(coeffs_new, 'haar')
        watermarked_image[:, :, ch] = watermarked_channel

    #Save res
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    cv2.imwrite(output_img_path, watermarked_image)
    print(f"Watermarked image saved: {output_img_path}")

    return watermarked_image

#Test case:
embed_watermark(user_id="kitshop", product_img_filename="landscape1.jpg")

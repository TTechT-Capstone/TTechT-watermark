import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to calculate PSNR, PCC, MSE, SSIM
def calculate_psnr(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def peak_correlation_coefficient(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    numerator = np.sum((img1 - np.mean(img1)) * (img2 - np.mean(img2)))
    denominator = np.sqrt(np.sum((img1 - np.mean(img1)) ** 2) * np.sum((img2 - np.mean(img2)) ** 2))
    return numerator / denominator if denominator != 0 else 0

def mean_squared_error(img1, img2):
    return np.mean((img1 - img2) ** 2)

def compare_watermarks(original_path, extracted_path):
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    extracted = cv2.imread(extracted_path, cv2.IMREAD_GRAYSCALE)

    if original is None or extracted is None:
        raise FileNotFoundError("One or both image files could not be read. Please check the paths again.")

    if original.shape != extracted.shape:
        extracted = cv2.resize(extracted, (original.shape[1], original.shape[0]))

    pcc = peak_correlation_coefficient(original, extracted)
    mse_val = mean_squared_error(original, extracted)
    ssim_val = ssim(original, extracted)
    psnr_score = calculate_psnr(original, extracted)

    print("PCC:", pcc)
    print("MSE:", mse_val)
    print("SSIM:", ssim_val)
    print("PSNR:", psnr_score, "dB")
# The official path will be set later based on the database location
if __name__ == "__main__":
    original_path = "utils/watermark_images/kiet_logo.jpg"
    extracted_path = "utils/extracted_attacked_watermark/extracted_attacked_product9_scale.jpg"
    compare_watermarks(original_path, extracted_path)

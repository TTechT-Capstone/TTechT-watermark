import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import json, os, shutil, uuid, time
from typing import Optional, Dict


# Function to calculate the metrics to evaluate & detect
def calculate_psnr(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
# PCC will be the main metric used to evaluate unauthorized image usage
def pearson_correlation_coefficient(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    numerator   = np.sum((img1 - np.mean(img1)) * (img2 - np.mean(img2)))
    denominator = np.sqrt(np.sum((img1 - np.mean(img1)) ** 2) * np.sum((img2 - np.mean(img2)) ** 2))
    return float(numerator / denominator) if denominator != 0 else 0.0

def mean_squared_error(img1, img2):
    return float(np.mean((img1 - img2) ** 2))

# Compare watermarks using multiple metrics (for Test case)
def compare_watermarks(original_path, extracted_path):
    original  = cv2.imread(original_path,  cv2.IMREAD_GRAYSCALE)
    extracted = cv2.imread(extracted_path, cv2.IMREAD_GRAYSCALE)

    if original is None or extracted is None:
        raise FileNotFoundError("One or both image files could not be read. Please check the paths again.")

    if original.shape != extracted.shape:
        extracted = cv2.resize(extracted, (original.shape[1], original.shape[0]))

    pcc        = pearson_correlation_coefficient(original, extracted)
    mse_val    = mean_squared_error(original, extracted)
    ssim_val   = ssim(original, extracted)
    psnr_score = calculate_psnr(original, extracted)

    print("PCC:", pcc)
    # We use the absolute value of PCC for detection since there are some extracted watermark that have reversed colors
    print("|PCC|:", abs(pcc))
    print("MSE:", mse_val)
    print("SSIM:", ssim_val)
    print("PSNR:", psnr_score, "dB")


# Compute the metrics for API
def compute_metrics(original_path: str, extracted_path: str) -> Dict[str, float]:
    """
    Returns a dict with PCC, PCC absolute value, MSE, SSIM, PSNR.
    Resizes extracted to match original if needed.
    """
    original  = cv2.imread(original_path,  cv2.IMREAD_GRAYSCALE)
    extracted = cv2.imread(extracted_path, cv2.IMREAD_GRAYSCALE)
    if original is None or extracted is None:
        raise FileNotFoundError("One or both image files could not be read.")

    if original.shape != extracted.shape:
        extracted = cv2.resize(extracted, (original.shape[1], original.shape[0]))

    pcc_val  = pearson_correlation_coefficient(original, extracted)
    mse_val  = mean_squared_error(original, extracted)
    ssim_val = ssim(original, extracted)
    psnr_val = calculate_psnr(original, extracted)

    return {
        "pcc":     float(pcc_val),
        # This will be used to compare with the threshold
        "pcc_abs": float(abs(pcc_val)),   
        "mse":     float(mse_val),
        "ssim":    float(ssim_val),
        "psnr":    float(psnr_val)
    }
# Setting detection threshold
def is_match(metrics: Dict[str, float], pcc_threshold: float = 0.70, use_abs: bool = True) -> bool:
    """Convenience: decide using PCC or |PCC|."""
    key = "pcc_abs" if use_abs else "pcc"
    return float(metrics.get(key, 0.0)) >= float(pcc_threshold)


# Saving a detection record for Admin
def save_detection_record(
    original_logo_path: str,
    extracted_wm_path: str,
    metrics: Dict[str, float],
    matched_json_path: Optional[str] = None,
    suspect_image_path: Optional[str] = None,
    out_root: str = "./detections",
    pcc_threshold: float = 0.70
) -> Dict[str, object]:
    """
    Saves a folder with:
      - record.json (metrics + thresholds + paths + timestamps)
      - original_logo.(png/jpg)
      - extracted_wm.(png/jpg)
      - suspect.(png/jpg)          [optional if provided]
      - sideinfo.wm.json           [optional copy if provided]
    Returns a dict with 'record_dir', 'record_json', and 'record'.
    """
    rec_id  = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    rec_dir = os.path.join(out_root, rec_id)
    os.makedirs(rec_dir, exist_ok=True)

    def _copy(src: Optional[str], name: str) -> Optional[str]:
        if src and os.path.exists(src):
            ext = os.path.splitext(src)[1].lower()
            dst = os.path.join(rec_dir, name + ext)
            shutil.copy2(src, dst)
            return dst
        return None

    paths = {
        "original_logo": _copy(original_logo_path, "original_logo"),
        "extracted_wm":  _copy(extracted_wm_path,  "extracted_wm"),
        "suspect":       _copy(suspect_image_path, "suspect") if suspect_image_path else None,
        "sideinfo_json": _copy(matched_json_path,  "sideinfo") if matched_json_path else None,
    }

    record = {
        "id": rec_id,
        "created_at": int(time.time()),
        "metrics": metrics,
        # ** We have to clarified why we using absolute value for PCC 
        "thresholds": {"pcc_abs": float(pcc_threshold)},        
        "passed": bool(float(metrics.get("pcc_abs", 0.0)) >= float(pcc_threshold)),
        "paths": paths
    }

    rec_json = os.path.join(rec_dir, "record.json")
    with open(rec_json, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return {"record_dir": rec_dir, "record_json": rec_json, "record": record}


# Test case for command line interface
if __name__ == "__main__":
    # Final path will be changed based on the database location
    original_path  = "./utils/watermark_images/origity_logo.jpg"
    extracted_path = "./utils/extracted_watermark/extracted_5.jpg"

    # Console prints
    compare_watermarks(original_path, extracted_path)

    # Saving detection information as a record for admin
    try:
        m   = compute_metrics(original_path, extracted_path)
        rec = save_detection_record(
            original_logo_path=original_path,
            extracted_wm_path=extracted_path,
            metrics=m,
            matched_json_path=None,
            suspect_image_path=None,
            out_root="./detections",
            pcc_threshold=0.70
        )
        print("Saved detection record:", rec["record_json"])
    except Exception as e:
        print("Metric/record demo failed:", e)

# TTechT_extract_watermark.py  — hardened extractor (no detection here)
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import pywt
import os, json
from pathlib import Path
import glob

# Helpers
def to_uint8(mat):
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)
# Folder of published watermarked images and their .wm.json sidecars
SIDEINFO_DIR = "./utils/watermarked_product_img"

# Hamming distance threshold for pHash (tune 8–14)
PHASH_HAMMING_THRESHOLD = 12

def phash64_from_pil(img_pil):
    """Simple 64-bit pHash via DCT(32x32 gray)."""
    g = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(g))
    low = dct[:8, :8]
    med = np.median(low[1:].ravel())  # skip DC
    bits = (low > med).astype(np.uint8).ravel()[:64]
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"

def hamming64(h1, h2):
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")

def compute_image_phash(img_path):
    try:
        im = Image.open(img_path).convert("RGB")
        return phash64_from_pil(im)
    except Exception:
        return None

def load_sideinfo_candidates(dir_path):
    """Yield (json_path, meta_dict)."""
    for jp in glob.glob(os.path.join(dir_path, "*.wm.json")):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            yield jp, meta
        except Exception:
            continue

def find_best_sideinfo_for_suspect(suspect_image_path, dir_path=SIDEINFO_DIR, max_ham=PHASH_HAMMING_THRESHOLD):
    """
    Returns (best_json_path, best_meta) if a close match is found by pHash,
    else (None, None). Uses meta['output_path'] when present to hash the
    published watermarked image; otherwise derives image path from the json name.
    """
    try:
        suspect_hash = phash64_from_pil(Image.open(suspect_image_path).convert("RGB"))
    except Exception:
        return None, None

    best_json, best_meta, best_dist = None, None, 1_000

    for json_path, meta in load_sideinfo_candidates(dir_path):
        out_path = meta.get("output_path", "")
        if not out_path or not os.path.exists(out_path):
            # derive: change .wm.json -> .jpg/.png if possible
            stem = os.path.splitext(json_path)[0]
            for ext in (".jpg", ".jpeg", ".png"):
                cand = stem + ext
                if os.path.exists(cand):
                    out_path = cand
                    break
        if not out_path or not os.path.exists(out_path):
            continue

        cand_hash = compute_image_phash(out_path)
        if cand_hash is None:
            continue

        d = hamming64(suspect_hash, cand_hash)
        if d < best_dist:
            best_json, best_meta, best_dist = json_path, meta, d

    if best_json and best_dist <= max_ham:
        return best_json, best_meta
    return None, None


# Paths - adjust later on
suspect_image_path       = "./utils/watermarked_product_img/5.jpg"
sideinfo_json_path       = "./utils/watermarked_product_img/5.wm.json"
#Let the helper auto-pick if there is no known sideinfo
#sideinfo_json_path       = None
extracted_watermark_path = "./utils/extracted_watermark/extracted_5.jpg"

#admin 
def extract_channel(suspect_channel, watermark_channel, S_orig_saved, wavelet_name, alpha=0.6, cname=""):
    """
    Semi-blind extraction using saved S_orig (from .wm.json).
    Includes length guards so slight size differences never crash.
    """
    with tqdm(total=100,
              desc=f"Extracting {cname} channel",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:

        # DWTs
        LL_mod, (LHm, HLm, HHm)       = pywt.dwt2(suspect_channel,   wavelet_name)
        LL_wmref, (LH_wm, HL_wm, HH_wm) = pywt.dwt2(watermark_channel, wavelet_name)
        pbar.update(30)

        # SVDs
        U_mod, S_mod, V_mod = np.linalg.svd(LL_mod,   full_matrices=False)
        U_wm,  S_wm,  V_wm  = np.linalg.svd(LL_wmref, full_matrices=False)
        pbar.update(50)

        # Length guard (avoid crashes on off-by-1 sizes)
        n = min(len(S_mod), len(S_orig_saved), len(S_wm))
        if n == 0:
            # If nothing usable; return a zero channel to keep pipeline active
            return np.zeros_like(suspect_channel, dtype=np.float64)

        S_mod       = S_mod[:n]
        S_orig_used = S_orig_saved[:n]
        U_wm        = U_wm[:, :n]
        V_wm        = V_wm[:n, :]

        # Semi-blind extraction
        S_wm_est  = (S_mod - S_orig_used) / max(alpha, 1e-12)
        LL_wm_est = U_wm @ np.diag(S_wm_est) @ V_wm
        pbar.update(10)

        wm_coeffs      = (LL_wm_est, (LH_wm, HL_wm, HH_wm))
        wm_channel_est = pywt.idwt2(wm_coeffs, wavelet_name)
        pbar.update(10)

    return wm_channel_est


# Main extraction flow (robust, returns status)
def extract_from_suspect(suspect_path, sideinfo_path, out_path):
    """
    Returns dict with:
      - status: "ok_extracted" | "skip_no_sideinfo" | "skip_bad_meta"
      - reason: present for skip_* statuses
      - extracted_path, alpha, wavelet, canonical_size, sideinfo_used, watermark_logo (on success)
    """
    # Auto-pick a sideinfo if not provided
    if not sideinfo_path:
        auto_json, auto_meta = find_best_sideinfo_for_suspect(suspect_path, SIDEINFO_DIR, PHASH_HAMMING_THRESHOLD)
        if auto_json:
            sideinfo_path = auto_json

    # No side-info provided or file missing -> skip (proceed to EMBED in API)
    if not sideinfo_path or not os.path.exists(sideinfo_path):
        return {
            "status": "skip_no_sideinfo",
            "reason": "No .wm.json provided/found for this image. Proceed to embedding."
        }

    # Load side-info JSON written after embedding (with guards)
    try:
        with open(sideinfo_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        alpha        = float(meta["wm_params"]["alpha"])
        wavelet_name = meta["wm_params"]["wavelet"]
        canonical_wh = tuple(meta.get("canonical_size", [0, 0]))  # (W, H)
        S_R = np.array(meta["host_S"]["R"], dtype=np.float64)
        S_G = np.array(meta["host_S"]["G"], dtype=np.float64)
        S_B = np.array(meta["host_S"]["B"], dtype=np.float64)
        wm_logo_path = meta["watermark_ref"]["path"]
    except FileNotFoundError:
        return {"status": "skip_no_sideinfo", "reason": "Side-info file missing. Proceed to embedding."}
    except KeyError as e:
        return {"status": "skip_bad_meta", "reason": f"Missing key {e}. Proceed to embedding."}
    except Exception as e:
        return {"status": "skip_bad_meta", "reason": f"Unreadable side-info: {e}. Proceed to embedding."}

    # Validate watermark logo path
    if not os.path.exists(wm_logo_path):
        return {"status": "skip_bad_meta", "reason": "Watermark logo path invalid. Proceed to embedding."}

    # Load the images
    try:
        watermark_logo = Image.open(wm_logo_path).convert("RGB")
        suspect_img    = Image.open(suspect_path).convert("RGB")
    except Exception as e:
        return {"status": "skip_bad_meta", "reason": f"Image open failed: {e}. Proceed to embedding."}

    # Resize both to canonical size (from embed); fallback to suspect size if there is no canonical size
    if canonical_wh != (0, 0):
        watermark_logo = watermark_logo.resize(canonical_wh)
        suspect_img    = suspect_img.resize(canonical_wh)
    else:
        canonical_wh = suspect_img.size  

    # Split channels → float64
    wmr, wmg, wmb = [np.float64(c) for c in watermark_logo.split()]
    sur, sug, sub = [np.float64(c) for c in suspect_img.split()]

    # Extract each color channels
    ext_r = extract_channel(sur, wmr, S_R, wavelet_name, alpha, "Red")
    ext_g = extract_channel(sug, wmg, S_G, wavelet_name, alpha, "Green")
    ext_b = extract_channel(sub, wmb, S_B, wavelet_name, alpha, "Blue")

    # Save the extracted watermark image
    r8, g8, b8 = map(to_uint8, (ext_r, ext_g, ext_b))
    out_img = Image.merge("RGB", (Image.fromarray(r8), Image.fromarray(g8), Image.fromarray(b8)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_img.save(out_path)

    return {
        "status": "ok_extracted",
        "alpha": alpha,
        "wavelet": wavelet_name,
        "canonical_size": canonical_wh,
        "sideinfo_used": sideinfo_path,
        "watermark_logo": wm_logo_path,
        "extracted_path": out_path
    }

# Test case
if __name__ == "__main__":
    res = extract_from_suspect(suspect_image_path, sideinfo_json_path, extracted_watermark_path)
    print(res)

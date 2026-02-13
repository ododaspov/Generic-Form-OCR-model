"""
OCR Engine: Provides text recognition from image crops.
"""

import cv2
import torch

# Try EasyOCR first
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Fallback Tesseract
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

if not (EASYOCR_AVAILABLE or PYTESSERACT_AVAILABLE):
    raise RuntimeError("No OCR backend available. Install easyocr or pytesseract.")


# ------------------ OCR function ------------------
def recognize_text_from_crop(crop_img):
    """
    Recognize text from a cropped image region.
    Args:
        crop_img: np.ndarray, BGR or grayscale crop
    Returns:
        string: recognized text
    """
    # EasyOCR expects RGB
    if EASYOCR_AVAILABLE:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        if crop_img.ndim == 2:
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
        else:
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        result = reader.readtext(crop_rgb)
        if len(result) > 0:
            # Choose highest confidence
            result_sorted = sorted(result, key=lambda r: r[2], reverse=True)
            return result_sorted[0][1].strip()
        else:
            return ""

    # Tesseract fallback
    elif PYTESSERACT_AVAILABLE:
        gray = crop_img if crop_img.ndim == 2 else cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 6')
        return text.strip()

# ------------------ Optional preprocessing ------------------
def preprocess_for_ocr(crop_img):
    """
    Optional preprocessing to improve OCR results.
    """
    gray = crop_img if crop_img.ndim == 2 else cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return denoised

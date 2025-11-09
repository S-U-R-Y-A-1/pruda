# ocr_utils.py
from PIL import Image, ImageFilter, ImageOps
import pytesseract
import io

# If tesseract isn't on PATH, set this before calling any function:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image_for_ocr(img: Image.Image, grayscale=True, denoise=True, enlarge=False):
    """
    Basic preprocessing: grayscale, optional denoise (median filter), optional enlarge.
    Returns a PIL Image.
    """
    if grayscale:
        img = img.convert("L")
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    if enlarge:
        w, h = img.size
        img = img.resize((w * 2, h * 2), Image.LANCZOS)
    # increase contrast slightly
    img = ImageOps.autocontrast(img)
    return img

def image_to_text(img: Image.Image, config: str = "--psm 3"):
    """
    Run pytesseract OCR on a PIL image and return extracted text.
    psm options can be tuned (3 = fully automatic page segmentation).
    """
    pre = preprocess_image_for_ocr(img)
    text = pytesseract.image_to_string(pre, config=config)
    return text

def image_path_to_text(path: str, **kwargs):
    img = Image.open(path)
    return image_to_text(img, **kwargs)

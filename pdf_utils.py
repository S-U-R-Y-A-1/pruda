# pdf_utils.py
import os
from typing import List, Tuple, Optional
from PIL import Image
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

POPPLER_PATH = None  # set to the poppler bin path on Windows if needed

def pdf_selectable_text(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Try to extract selectable text per page using PyMuPDF (fast & preferable).
    Returns list of (page_number starting at 1, text) or empty list if not available.
    """
    if not PYMUPDF_AVAILABLE:
        return []
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        pages.append((i, text))
    doc.close()
    return pages

def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Convert each PDF page to a PIL Image using pdf2image.
    Requires poppler available on the system.
    """
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError("pdf2image not installed. pip install pdf2image and ensure poppler is available.")
    if POPPLER_PATH:
        pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
    else:
        pages = convert_from_path(pdf_path, dpi=dpi)
    return pages

def pdf_pages_text(pdf_path: str, use_selectable_first: bool = True) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number, text) for a PDF.
    Tries selectable text first (PyMuPDF). If pages lack text, falls back to OCR images (caller must OCR).
    """
    pages = []
    if use_selectable_first:
        sel = pdf_selectable_text(pdf_path)
        if sel:
            # return as-is; caller can detect empty page text and opt to OCR
            return sel
    # fallback to images (caller will OCR using ocr_utils)
    imgs = pdf_to_images(pdf_path)
    for i, img in enumerate(imgs, start=1):
        pages.append((i, ""))  # placeholder text empty -> caller should OCR
    return pages

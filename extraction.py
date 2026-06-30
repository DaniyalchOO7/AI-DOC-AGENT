"""
extraction.py
-------------
Pulls raw text out of uploaded PDF documents.
Tries pdfplumber first (fast, clean), falls back to OCR (pytesseract)
for scanned/image PDFs that have no embedded text layer.
"""

import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    """Pull raw text using pdfplumber — works for text-based PDFs."""
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chunks.append(text)
    result = "\n".join(chunks)
    if not result.strip():
        raise ValueError("No text layer found — falling back to OCR.")
    return result


def extract_text_with_ocr(file_path: str) -> str:
    """
    OCR fallback for scanned PDFs.
    Requires: pip install pytesseract pillow pdf2image
    Also requires Tesseract installed on your system:
        Windows: https://github.com/UB-Mannheim/tesseract/wiki
        Mac:     brew install tesseract
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        raise ValueError(
            "OCR dependencies not installed. Run: "
            "pip install pytesseract pillow pdf2image"
        )

    images = convert_from_path(file_path)
    text = "\n".join(pytesseract.image_to_string(img) for img in images)
    if not text.strip():
        raise ValueError("Could not extract text even with OCR.")
    return text


def extract_text(file_path: str) -> str:
    """
    Master extraction function — try pdfplumber, fall back to OCR.
    This is what app.py and main.py should call.
    """
    try:
        return extract_text_from_pdf(file_path)
    except ValueError:
        return extract_text_with_ocr(file_path)
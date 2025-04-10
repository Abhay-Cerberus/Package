import os
from pathlib import Path
from AiPackageWrapper.ocr_module import OCRProcessor

def test_extract():
    # Build the full path to Sample_photo.png in the tests folder
    img_path = Path(__file__).parent / "Sample_photo.png"
    assert img_path.exists(), f"Test image not found: {img_path}"

    text = OCRProcessor.extract_text(str(img_path))
    # Just check that the text extraction doesn't return an empty string
    assert text is not None, "OCR returned None"
    assert len(text.strip()) > 0, "OCR returned empty text"

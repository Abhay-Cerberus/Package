from pathlib import Path
from PIL import Image
from AiPackageWrapper.ocr_module import OCRProcessor
from unittest.mock import patch

@patch("pytesseract.image_to_string", return_value="Dummy OCR text")
def test_extract(mock_image_to_string):
    # Build the full path to Sample_photo.png in the tests folder
    img_path = Path(__file__).parent / "Sample_photo.png"
    
    # If the sample image doesn't exist, create a dummy image for testing.
    if not img_path.exists():
        img = Image.new("RGB", (100, 100), color="white")
        img.save(str(img_path))
    
    text = OCRProcessor.extract_text(str(img_path))
    # Verify that our OCR returns the dummy text.
    assert "Dummy OCR text" in text

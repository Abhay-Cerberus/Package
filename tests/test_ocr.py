from AiPackageWrapper.ocr_module import OCRProcessor

from unittest.mock import patch, MagicMock
import pytest

@patch("modules.ocr_module.Image.open")
@patch("modules.ocr_module.tr_ocr_model")
def test_extract(mock_model, mock_image):
    mock_model.return_value = "Detected text"
    mock_image.return_value = MagicMock()
    text = OCRProcessor.extract_text("fake_path.png")
    assert text == "Detected text"

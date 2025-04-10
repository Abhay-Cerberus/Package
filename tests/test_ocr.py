from AiPackageWrapper.ocr_module import OCRProcessor

def test_extract():
    result  = OCRProcessor.extract_text("./Sample_photo.png")
    result == "Hello Interns"
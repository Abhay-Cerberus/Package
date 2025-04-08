from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

if __name__=="__main__":
    image_path = input("Enter the full image path")
    extracted_text = extract_text(image_path)
    print("\n Extracted_Text: \n")
    print(extracted_text)
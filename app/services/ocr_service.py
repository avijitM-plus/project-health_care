import pytesseract
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

# Set tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRService:
    def extract_text_from_image(self, file_path: str) -> str:
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            logger.info(f"OCR extracted {len(text)} characters from {os.path.basename(file_path)}")
            return text
        except Exception as e:
            logger.error(f"OCR processing failed for {file_path}: {e}")
            return ""

ocr_service = OCRService()

import pdfplumber
import os
import logging

logger = logging.getLogger(__name__)


class ReportParser:
    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            logger.info(f"PDF parsed: {len(text)} characters from {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"PDF parsing failed for {file_path}: {e}")
        return text

report_parser = ReportParser()

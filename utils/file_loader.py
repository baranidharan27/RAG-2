# utils/file_loader.py

import aiofiles
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
import re
import logging

logger = logging.getLogger(__name__)

async def load_files(files):
    """
    Loads and extracts text from uploaded files.
    Supports PDF and DOCX formats.
    """
    try:
        contents = ""
        for file in files:
            filename = file.filename.lower()
            content = await file.read()
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(content)
                logger.info(f"Extracted text from PDF: {file.filename}")
            elif filename.endswith('.docx'):
                text = extract_text_from_docx(content)
                logger.info(f"Extracted text from DOCX: {file.filename}")
            else:
                # Assume plain text for other file types
                text = content.decode('utf-8', errors='ignore')
                logger.info(f"Extracted text from plain file: {file.filename}")
            contents += text + "\n"
        return contents
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        raise e

def extract_text_from_pdf(content):
    """
    Extracts text from a PDF file.
    """
    try:
        reader = PdfReader(BytesIO(content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(content):
    """
    Extracts text from a DOCX file.
    """
    try:
        document = Document(BytesIO(content))
        text = "\n".join([para.text for para in document.paragraphs])
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""

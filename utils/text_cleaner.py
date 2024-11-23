# utils/text_cleaner.py

import re
import logging

logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Cleans the input text by removing unwanted artifacts such as table references,
    page numbers, multiple newlines, and other non-textual elements.
    """
    try:
        # Remove headers/footers like "Page 1 of 10"
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Remove table and figure references
        text = re.sub(r'Table\s*\d+[^.\n]*\.?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Figure\s*\d+[^.\n]*\.?', '', text, flags=re.IGNORECASE)
        
        # Remove multiple newlines and excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Remove lines that start with numbers or bullet points
        text = re.sub(r'^\s*[\d\-\•]+\s+', '', text, flags=re.MULTILINE)
        
        # Additional cleaning rules as needed
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text.strip()  # Return the original text if cleaning fails

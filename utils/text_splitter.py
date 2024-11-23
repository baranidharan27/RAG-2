# utils/text_splitter.py

from utils.text_cleaner import clean_text
import logging

logger = logging.getLogger(__name__)

def split_text(text, chunk_size=400, overlap=50):
    """
    Splits the input text into chunks of specified size with a defined overlap.
    """
    try:
        cleaned_text = clean_text(text)
        words = cleaned_text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        logger.info(f"Split text into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise e

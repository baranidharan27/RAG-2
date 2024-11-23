# embeddings/embedding.py

import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv
from database.database import MongoDBClient
from utils.text_cleaner import clean_text
import logging

logger = logging.getLogger(__name__)

class EmbeddingHandler:
    def __init__(self):
        try:
            self.model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))
            self.embed_dim = int(os.getenv("EMBED_DIM"))
            self.db_client = MongoDBClient()
            self.index = faiss.IndexFlatL2(self.embed_dim)
            self._load_existing_embeddings()
            logger.info("EmbeddingHandler initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing EmbeddingHandler: {e}")
            raise e

    def _load_existing_embeddings(self):
        try:
            embeddings = self.db_client.get_embeddings()
            if embeddings:
                vectors = np.array([e['embedding'] for e in embeddings]).astype('float32')
                self.index.add(vectors)
                logger.info(f"Loaded {len(embeddings)} existing embeddings into FAISS index.")
            else:
                logger.info("No existing embeddings found in the database.")
        except Exception as e:
            logger.error(f"Error loading existing embeddings: {e}")
            raise e

    def generate_embeddings(self, text_chunks):
        try:
            # Clean text chunks before encoding
            cleaned_chunks = [clean_text(chunk) for chunk in text_chunks]
            embeddings = self.model.encode(cleaned_chunks)
            for chunk, embedding in zip(cleaned_chunks, embeddings):
                embedding_data = {
                    "text_chunk": chunk,
                    "embedding": embedding.tolist()
                }
                self.db_client.insert_embedding(embedding_data)
                self.index.add(np.array([embedding]).astype('float32'))
            logger.info(f"Generated and stored embeddings for {len(text_chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise e

    def search_similar(self, query_embedding, top_k=3):
        try:
            D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
            embeddings = self.db_client.get_embeddings()
            results = []
            for idx in I[0]:
                if idx < len(embeddings):
                    results.append(embeddings[idx]['text_chunk'])
            logger.info(f"Retrieved {len(results)} similar chunks from FAISS index.")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise e

    def close(self):
        try:
            self.db_client.close()
            logger.info("Closed EmbeddingHandler successfully.")
        except Exception as e:
            logger.error(f"Error closing EmbeddingHandler: {e}")
            raise e

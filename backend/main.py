# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from utils.file_loader import load_files
from utils.text_splitter import split_text
from embeddings.embedding import EmbeddingHandler
from llm_models.transformers_model import TransformersLLM
from llm_models.llama_cpp import LlamaCppLLM
from database.database import MongoDBClient
from utils.prompt_engineering import construct_prompt
from utils.text_cleaner import clean_text
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("backend/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
embedding_handler = EmbeddingHandler()
db_client = MongoDBClient()

class QueryRequest(BaseModel):
    session_id: str
    query: str
    llm_model: str
    knowledge_base: str

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        logger.info("Received upload request with %d files.", len(files))
        contents = await load_files(files)
        text_chunks = split_text(contents)
        logger.info("Generated %d text chunks.", len(text_chunks))
        embedding_handler.generate_embeddings(text_chunks)
        logger.info("Generated and stored embeddings.")
        return {"status": "Files uploaded and embeddings generated."}
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
def query(request: QueryRequest):
    try:
        logger.info("Received query: %s", request.query)
        
        # Generate query embedding
        query_embedding = embedding_handler.model.encode([request.query])[0]
        logger.info("Generated query embedding.")
        
        # Retrieve top 3 similar chunks
        relevant_chunks = embedding_handler.search_similar(query_embedding, top_k=3)
        logger.info("Retrieved top 3 similar chunks.")
        
        # Retrieve chat history from MongoDB
        chat_history = db_client.get_chat_history(request.session_id)
        logger.info("Retrieved chat history with %d messages.", len(chat_history))
        
        # Summarize chat history if it exceeds a certain length
        max_history_length = 5  # Adjust as needed
        if len(chat_history) > max_history_length:
            chat_history = chat_history[-max_history_length:]
            logger.info("Trimmed chat history to last %d messages.", max_history_length)
        
        # Initialize the selected LLM model
        if request.llm_model == 'transformers':
            llm = TransformersLLM()
            logger.info("Initialized TransformersLLM.")
            tokenizer = llm.tokenizer
        else:
            llm = LlamaCppLLM()
            logger.info("Initialized LlamaCppLLM.")
            tokenizer = None  # LlamaCpp may not use the same tokenizer
        
        # Construct the prompt with chat history and relevant chunks
        prompt = construct_prompt(
            chat_history,
            relevant_chunks,
            request.query,
            tokenizer=tokenizer,
            max_tokens=2048
        )
        logger.info("Constructed prompt.")
        
        # Generate response
        response = llm.generate(prompt)
        logger.info("Generated response from LLM.")
        
        # Insert the user query and assistant response into chat history
        db_client.insert_chat_history(request.session_id, {"user": request.query, "assistant": response})
        logger.info("Inserted chat history into MongoDB.")
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


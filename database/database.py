# database/database.py

import os
from pymongo import MongoClient, errors
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self):
        try:
            self.client = MongoClient(os.getenv("MONGODB_URI"))
            self.db = self.client[os.getenv("MONGODB_DB")]
            self.embedding_collection = self.db[os.getenv("MONGODB_COLLECTION")]
            self.chat_history_collection = self.db["chat_history"]
            logger.info("Connected to MongoDB successfully.")
        except errors.PyMongoError as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise e

    def insert_embedding(self, embedding_data):
        try:
            self.embedding_collection.insert_one(embedding_data)
            logger.info("Inserted embedding data into MongoDB.")
        except errors.PyMongoError as e:
            logger.error(f"Error inserting embedding data: {e}")
            raise e

    def get_embeddings(self):
        try:
            embeddings = list(self.embedding_collection.find())
            logger.info(f"Retrieved {len(embeddings)} embeddings from MongoDB.")
            return embeddings
        except errors.PyMongoError as e:
            logger.error(f"Error retrieving embeddings: {e}")
            raise e

    def insert_chat_history(self, session_id, message):
        try:
            self.chat_history_collection.update_one(
                {"session_id": session_id},
                {"$push": {"history": message}},
                upsert=True
            )
            logger.info(f"Inserted chat history for session_id: {session_id}")
        except errors.PyMongoError as e:
            logger.error(f"Error inserting chat history: {e}")
            raise e

    def get_chat_history(self, session_id):
        try:
            record = self.chat_history_collection.find_one({"session_id": session_id})
            history = record["history"] if record and "history" in record else []
            logger.info(f"Retrieved {len(history)} chat history records for session_id: {session_id}")
            return history
        except errors.PyMongoError as e:
            logger.error(f"Error retrieving chat history: {e}")
            raise e

    def close(self):
        try:
            self.client.close()
            logger.info("Closed MongoDB connection successfully.")
        except errors.PyMongoError as e:
            logger.error(f"Error closing MongoDB connection: {e}")
            raise e

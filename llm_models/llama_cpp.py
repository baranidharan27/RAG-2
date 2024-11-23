# llm_models/llama_cpp.py

import subprocess
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class LlamaCppLLM:
    def __init__(self, model_path=None):
        try:
            self.model_path = model_path or os.getenv("LLM_MODEL_PATH")
            if not self.model_path or not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Llama.cpp model file not found at {self.model_path}")
            logger.info(f"LlamaCppLLM initialized with model at {self.model_path}.")
        except Exception as e:
            logger.error(f"Error initializing LlamaCppLLM: {e}")
            raise e

    def generate(self, prompt):
        try:
            # Call the llama.cpp executable with the prompt
            # Ensure that 'llama.cpp' executable is in the PATH or provide the full path
            llama_executable = os.getenv("LLAMA_CPP_EXECUTABLE", "llama")  # Default to 'llama' if not set
            result = subprocess.run(
                [llama_executable, "-m", self.model_path, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=120  # Set a timeout as needed
            )
            if result.returncode != 0:
                logger.error(f"Llama.cpp error: {result.stderr}")
                raise Exception(f"Llama.cpp error: {result.stderr}")
            response = result.stdout.strip()
            logger.info("Response generated successfully by LlamaCppLLM.")
            return response
        except subprocess.TimeoutExpired:
            logger.error("Llama.cpp generation timed out.")
            raise Exception("Llama.cpp generation timed out.")
        except Exception as e:
            logger.error(f"Error generating response in LlamaCppLLM: {e}")
            raise e

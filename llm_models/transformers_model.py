# llm_models/transformers_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class TransformersLLM:
    def __init__(self, model_name=None, use_gpu=True):
        try:
            self.model_name = model_name or os.getenv("LLM_MODEL")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
            self.model.to(self.device)
            
            # Ensure pad_token_id is set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print("Pad token ID not set. Using eos_token_id as pad_token_id.")
            print("TransformersLLM initialized successfully.")
        except Exception as e:
            print(f"Error initializing TransformersLLM: {e}")
            raise e

    def generate(self, prompt):
        try:
            # Tokenize input with attention mask
            inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(self.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # Check input length against model's max_position_embeddings
            input_length = input_ids.shape[1]
            max_length = self.model.config.max_position_embeddings  # e.g., 2048 for many models
            
            if input_length > max_length:
                # Truncate the prompt to fit within the model's max_length
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]
                print(f"Truncated input to the last {max_length} tokens.")
            
            # Generate response with a reasonable max_new_tokens
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=110,  # Adjust as needed
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                top_k=50
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Response generated successfully.")
            return response
        except Exception as e:
            print(f"Error generating response in TransformersLLM: {e}")
            raise e

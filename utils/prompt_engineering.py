# utils/prompt_engineering.py

import logging

logger = logging.getLogger(__name__)

def construct_prompt(chat_history, context, query, tokenizer=None, max_tokens=2048):
    """
    Constructs a prompt for the LLM by combining chat history, context, and the current query.
    Optionally truncates the prompt to fit within a maximum token limit.
    """
    try:
        prompt = "You are an expert assistant.\n\n"
        
        # Add chat history if available
        if chat_history:
            history_text = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history])
            prompt += f"Chat History:\n{history_text}\n\n"
        
        # Add context if available
        if context:
            context_text = "\n".join(context)
            prompt += f"Context:\n{context_text}\n\n"
        
        # Add the current user query
        prompt += f"User: {query}\nAssistant:"
        
        # If tokenizer is provided, check and truncate prompt if necessary
        if tokenizer:
            encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
            token_count = len(encoded_prompt)
            logger.info(f"Constructed prompt token count: {token_count}")
            
            if token_count > max_tokens:
                # Truncate to fit within max_tokens
                truncated_ids = encoded_prompt[-max_tokens:]
                prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                logger.info(f"Truncated prompt to last {max_tokens} tokens.")
        
        return prompt
    except Exception as e:
        logger.error(f"Error constructing prompt: {e}")
        raise e

# test_llm.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_llm():
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"  # Replace with actual model if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    context_text = "palakurthi raja karthik is working at iisc bangalore"
    query = "where does palakurthi raja karthik works?"

    # Construct the prompt with clear instructions
    prompt = (
        f"Given the following context, answer the query:\n\n"
        f"Context: {context_text}\n\n"
        f"Query: {query}\n"
        "Answer:"
    )

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate the response
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,  # Increase the token limit if necessary
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        top_k=50
    )

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

if __name__ == "__main__":
    test_llm()


# frontend/app.py

import streamlit as st
import requests
from streamlit_chat import message
import uuid
import os

# Initialize environment variables if needed
from dotenv import load_dotenv
load_dotenv()

# Configure page
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

# Session State Initialization
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to handle login
def login(username, password):
    try:
        data = {
            "username": username,
            "password": password
        }
        response = requests.post("http://localhost:8000/login/", json=data)
        if response.status_code == 200:
            return response.json()['session_id']
        else:
            st.error("Invalid credentials. Please try again.")
            return None
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None

# Function to handle file upload
def upload_files(files):
    try:
        files_payload = [('files', (f.name, f.read(), f.type)) for f in files]
        headers = {'Session-ID': st.session_state['session_id']}
        response = requests.post("http://localhost:8000/upload/", files=files_payload, headers=headers)
        if response.status_code == 200:
            st.success("Files uploaded and processed successfully.")
        else:
            st.error(f"Error uploading files: {response.text}")
    except Exception as e:
        st.error(f"Exception during file upload: {e}")

# Function to handle query
def send_query(query, llm_model, knowledge_base):
    try:
        data = {
            "session_id": st.session_state['session_id'],
            "query": query,
            "llm_model": llm_model,
            "knowledge_base": knowledge_base
        }
        response = requests.post("http://localhost:8000/query/", json=data)
        if response.status_code == 200:
            answer = response.json()['response']
            st.session_state['chat_history'].append({"user": query, "assistant": answer})
        else:
            st.error(f"Error in query: {response.text}")
    except Exception as e:
        st.error(f"Exception during query: {e}")

# Login Page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        session_id = login(username, password)
        if session_id:
            st.session_state['session_id'] = session_id
            st.experimental_rerun()

# Chat Page
def chat_page():
    st.title("RAG Chatbot 💬")
    
    # Sidebar Options
    st.sidebar.title("Options")
    llm_model = st.sidebar.selectbox("Select LLM Model", ["transformers", "llama_cpp"])
    knowledge_base = st.sidebar.selectbox("Select Knowledge Base", ["knowledge_base-1"])
    
    # File Uploader
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    if st.button("Upload Files"):
        if uploaded_files:
            upload_files(uploaded_files)
        else:
            st.warning("Please upload at least one file.")
    
    # Chat Interface
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send", key="send_button"):
        if user_input.strip() != "":
            send_query(user_input, llm_model, knowledge_base)
            st.session_state['user_input'] = ""
        else:
            st.warning("Please enter a query before sending.")
    
    # Display Chat History
    for i, chat in enumerate(st.session_state['chat_history']):
        message(chat["user"], is_user=True, key=f"user_{i}")
        message(chat["assistant"], key=f"assistant_{i}")

# Main App Logic
def main():
    if not st.session_state['session_id']:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
str(uuid.uuid4())
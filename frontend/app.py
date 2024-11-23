# frontend/app.py

import streamlit as st
import requests
import uuid

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# Define login page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Implement your authentication logic here
        if username == "admin" and password == "password":  # Example credentials
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid credentials.")

# Define chat page
def chat_page():
    st.title("Chat Interface")

    # Upload file section
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    if st.button("Upload"):
        try:
            files = [('files', (f.name, f.read(), f.type)) for f in uploaded_files]
            response = requests.post("http://localhost:8000/upload/", files=files)
            if response.status_code == 200:
                st.success("Files uploaded and processed.")
            else:
                st.error(f"Error uploading files: {response.text}")
        except Exception as e:
            st.error(f"Exception during file upload: {e}")

    # Chat interface
    user_input = st.text_input("Enter your message:", key='user_input')

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            try:
                data = {
                    "session_id": st.session_state['session_id'],
                    "query": user_input,
                    "llm_model": "transformers",
                    "knowledge_base": "knowledge_base-1"
                }
                response = requests.post("http://localhost:8000/query/", json=data)
                if response.status_code == 200:
                    answer = response.json()['response']
                    st.session_state['chat_history'].append({"user": user_input, "assistant": answer})
                    # Clear the input field by resetting the widget's value
                    st.session_state['user_input'] = ''
                else:
                    st.error(f"Error in query: {response.text}")
            except Exception as e:
                st.error(f"Exception during query: {e}")

    # Display chat history
    for idx, chat in enumerate(st.session_state['chat_history']):
        st.write(f"**User:** {chat['user']}")
        st.write(f"**Assistant:** {chat['assistant']}")

# Main function
def main():
    if not st.session_state['logged_in']:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()

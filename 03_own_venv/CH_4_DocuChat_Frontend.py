import requests
import json
import os
import streamlit as st
import time

# BACKEND_URL = "https://didactic-space-waffle-gjrwp9rwp4r2w7jx-8000.app.github.dev"
BACKEND_URL = "http://localhost:8000"

def chat(user_input, data, session_id=None):
    url = BACKEND_URL + "/chat"
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    payload = {
        "user_input": user_input,
        "data_source": data,
        **({"session_id": session_id} if session_id else {})
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("response", {}).get("answer"), response_data.get("session_id")
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None, None

def upload_file(file_path):
    url = BACKEND_URL + "/uploadFile"
    headers = {"accept": "application/json"}
    
    try:
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            file_type = "application/pdf" if filename.endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            files = [("data_file", (filename, f, file_type))]
            try:
                response = requests.post(url, headers=headers, files=files, timeout=20)
            except requests.exceptions.Timeout:
                print("Timeout occurred")            

            response.raise_for_status()
            return response.json().get("file_path", "").split("/")[-1]
    except Exception as e:
        st.error(f"Upload Error: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Document Chat", page_icon="📕", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessionid" not in st.session_state:
    st.session_state.sessionid = None

data_file = st.file_uploader(
    label="Input file", accept_multiple_files=False, type=["docx", "csv", "txt", "pdf"]
)
st.divider()

if data_file is not None:
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, data_file.name)
    
    with open(file_path, "wb") as f:
        f.write(data_file.getbuffer())

    s3_upload_url = upload_file(file_path)
    
    if s3_upload_url:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("You can ask any question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                assistant_response, session_id = chat(
                    prompt, 
                    data=s3_upload_url, 
                    session_id=st.session_state.sessionid
                )
                
                if assistant_response:
                    st.session_state.sessionid = session_id
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in assistant_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

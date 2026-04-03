import streamlit as st
import requests

API_URL = "https://genai-rag-assistant-n5mg.onrender.com"

st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("AI Document Assistant")

st.sidebar.title("About")
st.sidebar.info("Upload a PDF and chat with it using AI")

# -------- Session State --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- Upload Section --------
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader("Choose file", type="pdf")

if uploaded_file:
    if st.sidebar.button("Upload"):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            st.sidebar.success("Uploaded successfully")
        else:
            st.sidebar.error("Upload failed")

# -------- Chat UI --------
st.subheader("Chat")

query = st.chat_input("Ask something about your document...")

if query:
    response = requests.get(f"{API_URL}/ask", params={"query": query})

    if response.status_code == 200:
        answer = response.json()["answer"]

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", answer))

    else:
        st.error("Error getting response")

# -------- Display Chat --------
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)

# -------- Clear Button --------
if st.sidebar.button("Clear Chat"):
    requests.post(f"{API_URL}/clear")
    st.session_state.chat_history = []
    st.sidebar.success("Chat cleared")
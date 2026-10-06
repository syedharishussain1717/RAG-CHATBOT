import os
import streamlit as st
from rag import read_file, chunk_text, build_index, answer

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 Chat with your Documents")

try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Secrets.")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("index", None)
st.session_state.setdefault("chunks", [])

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Process documents") and files:
        with st.spinner("Indexing..."):
            text = "\n".join(read_file(f, f.name) for f in files)
            chunks = chunk_text(text)
            if not chunks:
                st.error("No readable text found.")
            else:
                st.session_state.chunks = chunks
                st.session_state.index = build_index(chunks)
                st.session_state.messages = []
                st.success(f"Indexed {len(chunks)} chunks.")
    if st.button("Clear chat"):
        st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    if st.session_state.index is None:
        st.warning("Upload and process a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = answer(prompt, st.session_state.index, st.session_state.chunks,
                               API_KEY, st.session_state.messages[:-1])
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# 📄 RAG Chatbot: Chat with Your Documents

A simple Retrieval-Augmented Generation (RAG) chatbot. Upload PDF or TXT files and ask questions. Answers come only from your documents.

## Tech Stack
- **LLM:** Groq API (`openai/gpt-oss-120b`)
- **Embeddings:** Hugging Face `all-MiniLM-L6-v2` (runs locally)
- **Vector DB:** FAISS (`faiss-cpu`)
- **Frontend:** Streamlit

## How It Works
1. Uploaded documents are split into overlapping chunks.
2. Each chunk is converted to an embedding and stored in a FAISS index.
3. A question is embedded and the top 4 matching chunks are retrieved.
4. The chunks and question are sent to Groq, which answers using only that context.

## Project Structure
```
rag-chatbot/
├── app.py             # Streamlit UI
├── rag.py             # RAG logic (chunking, FAISS, Groq)
├── requirements.txt
└── .gitignore
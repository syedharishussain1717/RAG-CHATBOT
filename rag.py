import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a professional document assistant. Answer the user's question using ONLY the context provided below.

Rules:
1. If the answer is not in the context, say: "I couldn't find that information in the provided documents." Do not guess or use outside knowledge.
2. Be clear, concise, and well structured. Use bullet points for lists.
3. Quote or reference the relevant part of the context when it helps.
4. Keep a polite, professional tone.
5. If the question is ambiguous, ask one short clarifying question."""

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def read_file(file, name):
    if name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return file.read().decode("utf-8", errors="ignore")

def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    text = " ".join(text.split())
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def build_index(chunks):
    emb = get_embedder().encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    emb = np.array(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

def retrieve(query, index, chunks, k=4):
    q = get_embedder().encode([query], normalize_embeddings=True)
    _, ids = index.search(np.array(q, dtype="float32"), k)
    return [chunks[i] for i in ids[0] if i != -1]

def answer(query, index, chunks, api_key, history=None):
    context = "\n\n---\n\n".join(retrieve(query, index, chunks))
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"}]
    for m in (history or [])[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": query})
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0.2, max_tokens=2000
    )
    return resp.choices[0].message.content

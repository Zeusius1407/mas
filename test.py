import os
import faiss
import PyPDF2
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ---- Config ----
MODEL_PATH = "./models/llama-2-7b.Q4_K_M.gguf"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Fast & good enough

# ---- Load models ----
llm = Llama(model_path=MODEL_PATH, n_threads=6, n_ctx=16284)
embedder = SentenceTransformer(EMBED_MODEL)

# ---- PDF Loader ----
def load_pdf(path):
    reader = PyPDF2.PdfReader(path)
    return [page.extract_text() for page in reader.pages if page.extract_text()]

# ---- Chunking ----
def chunk_text(pages, min_len=50):
    chunks = []
    for text in pages:
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            if len(para.strip()) > min_len:
                chunks.append(para.strip())
    return chunks

# ---- Index Builder ----
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# ---- RAG Retrieval ----
def retrieve_chunks(query, chunks, index, k=5):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [chunks[i] for i in I[0]]

# ---- LLM Summarizer ----
def generate_summary(chunks, query=None):
    input_text = "\n".join(chunks[:5])
    prompt = f"""You are a helpful assistant. Summarize the following:

{input_text}

Summary:"""
    output = llm(prompt, max_tokens=250)
    return output["choices"][0]["text"].strip()


pdf_path = "SwiggyProspectus.pdf"
query = "Summarize the business strategy discussed in this document"

pages = load_pdf(pdf_path)
chunks = chunk_text(pages)
index, _ = build_faiss_index(chunks)

if query:
    top_chunks = retrieve_chunks(query, chunks, index)
else:
    top_chunks = chunks[:5]

summary = generate_summary(top_chunks)
print("Summary:\n", summary)

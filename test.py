import PyPDF2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict

# Configuration
MODEL_PATH = "./models/mistral-7b"  # Path to your local Mistral model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight embedding model
PDF_PATH = "large_document.pdf"
CHUNK_SIZE = 512  # Tokens per chunk
SUMMARY_LENGTH = 200  # Target summary length in words

# --- Step 1: Load Models ---
def load_models():
    # Load Mistral for generation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load embedding model
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    return tokenizer, model, embed_model

# --- Step 2: PDF Chunking ---
def chunk_pdf(pdf_path: str, chunk_size: int) -> List[Dict]:
    chunks = []
    with open(pdf_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        current_chunk = ""
        
        for page in pdf.pages:
            text = page.extract_text()
            words = text.split()
            
            for word in words:
                current_chunk += word + " "
                if len(current_chunk.split()) >= chunk_size:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": pdf.get_page_number(page) + 1
                    })
                    current_chunk = ""
    
    return chunks

# --- Step 3: Create Vector Store ---
def create_vector_store(chunks: List[str], embed_model):
    embeddings = embed_model.encode([chunk["text"] for chunk in chunks])
    return {
        "chunks": chunks,
        "embeddings": embeddings
    }

# --- Step 4: Retrieve Relevant Chunks ---
def retrieve_chunks(query: str, vector_store, embed_model, top_k: int = 3):
    query_embed = embed_model.encode([query])
    similarities = cosine_similarity(query_embed, vector_store["embeddings"])[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [vector_store["chunks"][i] for i in top_indices]

# --- Step 5: Generate Summary ---
def generate_summary(model, tokenizer, context: str, max_length: int):
    prompt = f"""
    Summarize the following document extract in about {max_length} words:
    {context}
    
    Summary:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.7,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Main Pipeline ---
def rag_summarize_pdf():
    # Load models
    tokenizer, model, embed_model = load_models()
    
    # Process PDF
    print("Chunking PDF...")
    chunks = chunk_pdf(PDF_PATH, CHUNK_SIZE)
    
    # Create vector store
    print("Creating vector store...")
    vector_store = create_vector_store(chunks, embed_model)
    
    # Retrieve relevant chunks for summary
    print("Retrieving key sections...")
    query = "What are the main points of this document?"
    relevant_chunks = retrieve_chunks(query, vector_store, embed_model)
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    
    # Generate summary
    print("Generating summary...")
    summary = generate_summary(model, tokenizer, context, SUMMARY_LENGTH)
    
    # Save results
    with open("summary.txt", "w") as f:
        f.write(f"Document: {PDF_PATH}\n")
        f.write(f"Summary length: {SUMMARY_LENGTH} words\n\n")
        f.write(summary)
    
    print(f"Summary saved to summary.txt")
    return summary

if __name__ == "__main__":
    rag_summarize_pdf()
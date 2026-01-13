import json
import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ===============================
# PATH HANDLING
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ENHANCED_FILE = os.path.join(DATA_DIR, "chunked_lic_documents_enhanced.json")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")

print(f"Loading enhanced chunks from: {ENHANCED_FILE}")

# ===============================
# LOAD ENHANCED CHUNKS
# ===============================

def load_chunks() -> List[dict]:
    with open(ENHANCED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ===============================
# BUILD VECTOR STORE
# ===============================

def build_vector_store(chunks: List[dict]):
    texts = []
    metadatas = []
    
    for c in chunks:
        # Handle both old format (text) and new format (content)
        text = c.get("text") or c.get("content", "")
        texts.append(text)
        
        metadata = {
            "chunk_id": c.get("chunk_id", "unknown"),
            "source": c.get("title") or c.get("source", "LIC Document"),
            "source_url": c.get("source_url", "https://licindia.in"),
        }
        
        # Add section if available
        if "section" in c:
            metadata["section"] = c["section"]
        
        # Add any additional metadata
        if "metadata" in c:
            metadata.update(c["metadata"])
        
        metadatas.append(metadata)

    print(f"Processing {len(texts)} chunks...")

    # Using HuggingFace embeddings (free, runs locally)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Generating embeddings... This may take 2-3 minutes.")
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    print(f"Saving to {FAISS_DIR}...")
    os.makedirs(FAISS_DIR, exist_ok=True)
    vector_store.save_local(FAISS_DIR)

    print(f"Vector store created with {len(texts)} embeddings")

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    build_vector_store(chunks)
    print("\nEmbeddings rebuilt successfully with enhanced data!")

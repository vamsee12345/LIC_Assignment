import json
import os
import pickle
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

# ===============================
# PATH HANDLING
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Use enhanced dataset with sample policy details
CHUNKED_FILE = os.path.join(DATA_DIR, "chunked_lic_documents_enhanced.json")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
BM25_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")

# ===============================
# LOAD CHUNKS
# ===============================

def load_chunks() -> List[dict]:
    with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ===============================
# BUILD VECTOR STORE
# ===============================

def build_vector_store(chunks: List[dict]):
    # Handle both 'text' and 'content' fields for compatibility
    texts = [c.get("text") or c.get("content", "") for c in chunks]

    metadatas = [{
        "chunk_id": c.get("chunk_id", ""),
        "title": c.get("title", c.get("source", "")),
        "section": c.get("section", ""),
        "source_url": c.get("source_url", "")
    } for c in chunks]

    # Using HuggingFace embeddings (free, runs locally)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    os.makedirs(FAISS_DIR, exist_ok=True)
    vector_store.save_local(FAISS_DIR)

    print(f"✅ Vector store created with {len(texts)} embeddings")
    
    # ===============================
    # BUILD BM25 INDEX (Keyword Search)
    # ===============================
    print("\n🔍 Building BM25 index for keyword search...")
    
    # Tokenize documents for BM25
    tokenized_corpus = [text.lower().split() for text in texts]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save BM25 index with metadata
    bm25_data = {
        'bm25': bm25,
        'texts': texts,
        'metadatas': metadatas
    }
    
    with open(BM25_FILE, 'wb') as f:
        pickle.dump(bm25_data, f)
    
    print(f"✅ BM25 index created with {len(texts)} documents")
    print(f"💾 Saved to: {BM25_FILE}")
    print("\n🎉 Hybrid search (FAISS + BM25) ready!")

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    chunks = load_chunks()
    build_vector_store(chunks)

import json
import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PARSED_FILE = os.path.join(DATA_DIR, "parsed_lic_documents.json")
CHUNKED_FILE = os.path.join(DATA_DIR, "chunked_lic_documents.json")


CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# ===============================
# LOAD & SAVE
# ===============================

def load_parsed_docs() -> List[Dict]:
    with open(PARSED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chunked_docs(data: List[Dict]):
    with open(CHUNKED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ===============================
# CHUNKING LOGIC
# ===============================

def chunk_documents(parsed_docs: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunked_docs = []

    for doc in parsed_docs:
        text_chunks = splitter.split_text(doc["text"])

        for idx, chunk in enumerate(text_chunks):
            chunked_docs.append({
                "chunk_id": f'{doc["title"].replace(" ", "_").lower()}_{doc["section"].lower()}_{idx}',
                "document_type": doc["document_type"],
                "title": doc["title"],
                "section": doc["section"],
                "source_url": doc["source_url"],
                "text": chunk
            })

    return chunked_docs


if __name__ == "__main__":
    parsed_docs = load_parsed_docs()
    chunked_docs = chunk_documents(parsed_docs)
    save_chunked_docs(chunked_docs)

    print(f"Created {len(chunked_docs)} chunks")
    print(f"Output saved to: {CHUNKED_FILE}")

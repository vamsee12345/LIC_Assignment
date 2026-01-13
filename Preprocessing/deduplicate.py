import json
import os
import hashlib
from typing import List, Dict

# ===============================
# PATHS
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CHUNKED_FILE = os.path.join(DATA_DIR, "chunked_lic_documents.json")
DEDUP_FILE = os.path.join(DATA_DIR, "deduped_chunks.json")

# ===============================
# HELPERS
# ===============================

def load_chunks() -> List[Dict]:
    with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chunks(chunks: List[Dict]):
    with open(DEDUP_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

def hash_text(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

# ===============================
# DEDUP LOGIC
# ===============================

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    seen_hashes = set()
    deduped = []

    for chunk in chunks:
        text_hash = hash_text(chunk["text"])

        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            deduped.append(chunk)

    return deduped

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    chunks = load_chunks()
    deduped_chunks = deduplicate_chunks(chunks)
    save_chunks(deduped_chunks)

    print(f"🔹 Original chunks: {len(chunks)}")
    print(f"✅ Deduplicated chunks: {len(deduped_chunks)}")

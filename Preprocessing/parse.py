import json
import os
import re
from typing import List, Dict


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_FILE = os.path.join(DATA_DIR, "raw_lic_documents.json")
PARSED_FILE = os.path.join(DATA_DIR, "parsed_lic_documents.json")

SECTION_KEYWORDS = {
    "Eligibility": ["eligibility", "eligible", "entry age"],
    "Benefits": ["benefit", "maturity", "survival", "death"],
    "Premium": ["premium", "payment", "mode", "payable"],
    "Claims": ["claim", "settlement", "documents required"],
}


def load_raw_documents() -> List[Dict]:
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parsed_documents(data: List[Dict]):
    with open(PARSED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def detect_section(text: str) -> str:
    text_lower = text.lower()
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return section
    return "Other"


def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n{2,}|\.\s+", text)
    return [p.strip() for p in paragraphs if len(p.strip()) > 50]



def parse_documents(raw_docs: List[Dict]) -> List[Dict]:
    parsed_docs = []

    for doc in raw_docs:
        content = doc["content"]
        paragraphs = split_into_paragraphs(content)

        for idx, para in enumerate(paragraphs):
            section = detect_section(para)

            parsed_docs.append({
                "document_type": doc["document_type"],
                "title": doc["title"],
                "section": section,
                "source_url": doc["source_url"],
                "paragraph_id": idx,
                "text": para
            })

    return parsed_docs


if __name__ == "__main__":
    raw_docs = load_raw_documents()
    parsed_docs = parse_documents(raw_docs)
    save_parsed_documents(parsed_docs)

    print(f"Parsed {len(parsed_docs)} sections")
    print(f" Output saved to: {PARSED_FILE}")

import os
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from rank_bm25 import BM25Okapi
import torch

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# ===============================
# PATHS
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
BM25_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")

# ===============================
# INITIALIZE MODEL 
# ===============================

print("Loading flan-t5-large model...")

# Load flan-t5-large model for better accuracy
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Create pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15
)

# Wrap in LangChain
_llm = HuggingFacePipeline(pipeline=pipe)

print("✓ flan-t5-large model loaded successfully!")

# ===============================
# LOAD BM25 INDEX
# ===============================

print("Loading BM25 index for hybrid search...")
with open(BM25_FILE, 'rb') as f:
    _bm25_data = pickle.load(f)
    _bm25 = _bm25_data['bm25']
    _bm25_texts = _bm25_data['texts']
    _bm25_metadatas = _bm25_data['metadatas']
print(f"BM25 index loaded with {len(_bm25_texts)} documents!")

# ===============================
# LOAD VECTOR STORE
# ===============================

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ===============================
# PROMPT
# ===============================

SYSTEM_PROMPT = """
You are an internal LIC knowledge assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not present, say:
  "Information not available in the provided documents."
- Be concise and factual.
- Cite the source URLs explicitly.
"""

# ===============================
# HYBRID SEARCH (FAISS + BM25)
# ===============================

def is_factual_query(question: str) -> bool:
    """
    Detect if query needs exact factual answers (codes, numbers, ages).
    These queries benefit more from BM25 keyword search.
    """
    factual_keywords = [
        'code', 'number', 'age', 'years', 'minimum', 'maximum',
        'how much', 'how many', 'what is the', 'specific',
        'exact', 'premium', 'sum assured', 'percentage', '%'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in factual_keywords)

def hybrid_search(vector_store, question: str, k: int = 4):
    """
    Adaptive hybrid search that adjusts weights based on query type.
    
    For factual queries (codes, numbers): Prioritize BM25 (keyword search)
    For conceptual queries (explanations): Balance FAISS + BM25
    """
    # Detect query type
    is_factual = is_factual_query(question)
    
    if is_factual:
        # For factual queries: Prioritize BM25 (better for exact matches)
        print("  → Factual query detected: Prioritizing BM25 keyword search...")
        bm25_k = k * 3  # More BM25 results
        faiss_k = k     # Fewer FAISS results
    else:
        # For conceptual queries: Balance both
        print("  → Conceptual query: Balancing FAISS + BM25...")
        bm25_k = k * 2
        faiss_k = k * 2
    
    # 1. Semantic search (FAISS) - good for understanding intent
    semantic_docs = vector_store.similarity_search(question, k=faiss_k)
    
    # 2. Keyword search (BM25) - good for exact matches
    tokenized_query = question.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    
    # Get top BM25 results
    top_bm25_indices = bm25_scores.argsort()[-bm25_k:][::-1]
    
    # 3. Combine results with priority based on query type
    seen_texts = set()
    combined_docs = []
    
    if is_factual:
        # For factual queries: BM25 results first (they have the exact numbers)
        for idx in top_bm25_indices:
            text = _bm25_texts[idx]
            if text not in seen_texts:
                seen_texts.add(text)
                from langchain_core.documents import Document
                doc = Document(
                    page_content=text,
                    metadata=_bm25_metadatas[idx]
                )
                combined_docs.append(doc)
        
        # Then add semantic results for context
        for doc in semantic_docs:
            text = doc.page_content
            if text not in seen_texts:
                seen_texts.add(text)
                combined_docs.append(doc)
    else:
        # For conceptual queries: FAISS first (better understanding)
        for doc in semantic_docs:
            text = doc.page_content
            if text not in seen_texts:
                seen_texts.add(text)
                combined_docs.append(doc)
        
        # Then add BM25 for exact term matches
        for idx in top_bm25_indices:
            text = _bm25_texts[idx]
            if text not in seen_texts:
                seen_texts.add(text)
                from langchain_core.documents import Document
                doc = Document(
                    page_content=text,
                    metadata=_bm25_metadatas[idx]
                )
                combined_docs.append(doc)
    
    # Return top k results
    return combined_docs[:k]

# ===============================
# RAG QA FUNCTION
# ===============================

def enhance_query(question: str) -> list:
    """
    Generate multiple query variations to improve retrieval.
    Returns list of queries to try.
    """
    queries = [question]  # Original query
    
    question_lower = question.lower()
    
    # For plan code queries, add variations
    if 'plan code' in question_lower or 'code number' in question_lower:
        # Extract plan name
        for plan in ['endowment', 'jeevan anand', 'jeevan umang', 'jeevan labh', 'tech term']:
            if plan in question_lower:
                queries.append(f"{plan} plan number")
                queries.append(f"LIC {plan} code")
                break
    
    # For eligibility queries
    if 'eligible' in question_lower or 'entry age' in question_lower:
        queries.append("minimum age maximum age eligibility")
    
    return queries

def answer_question(question: str, k: int = 4) -> str:  # Keep at 4 to avoid token limit
    vector_store = load_vector_store()

    # 1. Use HYBRID search (FAISS + BM25) for better retrieval
    print(f"Using hybrid search (semantic + keyword)...")
    docs = hybrid_search(vector_store, question, k=k)

    if not docs:
        return "Information not available in the provided documents."

    # 2. Build context smartly to stay within token limits
    context_parts = []
    sources = []
    
    # Calculate optimal chunk size to stay under 512 tokens
    # Rule: ~4 chars per token, need space for prompt (~150 tokens)
    # 512 - 150 = 362 tokens * 4 = ~1450 chars max for context
    # Divide by k documents = 1450/4 = 362 chars per doc
    max_chars_per_doc = 320  # Conservative to be safe
    
    for i, d in enumerate(docs, 1):
        text = d.page_content[:max_chars_per_doc]
        context_parts.append(f"Document {i}: {text}")
        
        # Collect source URLs
        if hasattr(d, 'metadata') and 'source_url' in d.metadata:
            sources.append(d.metadata['source_url'])
    
    context = "\n\n".join(context_parts)

    # 3. Concise prompt to save tokens while keeping instructions clear
    prompt = f"""Extract precise LIC policy information from context.

RULES:
1. Answer ONLY from context below
2. Extract EXACT details - don't paraphrase
3. If info missing: say "Information not available in the provided documents."
4. Table format "2 LIC's New Endowment Plan 714 512N277V03" → plan code is 714 (3-digit number after plan name)
5. For ages: mention minimum and maximum
6. Be concise

Context:
{context}

Question: {question}

Answer:"""

    try:
        # Model returns response object
        response = _llm.invoke(prompt)
        
        # Extract content from response
        if hasattr(response, 'content'):
            clean_response = response.content.strip()
        elif isinstance(response, str):
            clean_response = response.strip()
        else:
            clean_response = str(response).strip()
        
        # Post-process to improve answer quality
        clean_response = post_process_answer(clean_response, question, context)
        
        # If response is too short, return helpful message
        if len(clean_response) < 10:
            return "Information not available in the provided documents."
        
        # Add source citations
        if sources:
            unique_sources = list(set(sources))
            source_text = "\n\nSources:\n" + "\n".join(f"- {url}" for url in unique_sources)
            return clean_response + source_text
        
        return clean_response
        
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return f"Error: Unable to generate answer. {str(e)}"

def post_process_answer(answer: str, question: str, context: str) -> str:
    """
    Post-process answer to extract specific information better.
    """
    import re
    
    question_lower = question.lower()
    
    # For plan code queries, try to extract from context if answer is unclear
    if ('plan code' in question_lower or 'code number' in question_lower) and \
       ('not available' in answer.lower() or len(answer) < 20):
        
        # Try to find plan codes in context
        # Pattern: Plan name followed by 3-digit number
        patterns = [
            r'(?:plan code:|code:)\s*(\d{3})',  # "Plan Code: 714"
            r'(\d{3})\s+\d{3}[A-Z]\d+',  # "714 512N277V03"
        ]
        
        for plan in ['endowment plan', 'jeevan anand', 'jeevan umang', 'jeevan labh']:
            if plan in question_lower:
                # Look for this plan in context
                pattern = rf'{plan}.*?(\d{{3}})'
                match = re.search(pattern, context.lower())
                if match:
                    code = match.group(1)
                    return f"LIC's plan code is {code}"
    
    # For eligibility questions, ensure we mention both min and max
    if 'eligible' in question_lower or 'entry age' in question_lower:
        has_min = 'minimum' in answer.lower() or 'min' in answer.lower()
        has_max = 'maximum' in answer.lower() or 'max' in answer.lower()
        
        if not has_min or not has_max:
            # Try to extract from context
            min_pattern = r'minimum.*?age.*?(\d+)'
            max_pattern = r'maximum.*?age.*?(\d+)'
            
            min_match = re.search(min_pattern, context.lower())
            max_match = re.search(max_pattern, context.lower())
            
            if min_match and max_match:
                return f"Minimum entry age is {min_match.group(1)} and maximum entry age is {max_match.group(1)}"
    
    return answer

# ===============================
# CLI TEST
# ===============================

if __name__ == "__main__":
    while True:
        q = input("\nAsk LIC question (type 'exit' to quit): ")
        if q.lower() == "exit":
            break

        print("\nAnswer:")
        print(answer_question(q))

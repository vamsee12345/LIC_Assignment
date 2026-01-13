import sys
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path to import rag module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.qa import answer_question

app = FastAPI(
    title="GenAI Knowledge Assistant",
    description="RAG-based internal knowledge assistant for LIC",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# REQUEST / RESPONSE SCHEMA
# ===============================

class QueryRequest(BaseModel):
    question: str
    
    class Config:
        # Example for the API docs
        json_schema_extra = {
            "example": {
                "question": "What are endowment plans?"
            }
        }

class QueryResponse(BaseModel):
    answer: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Endowment plans are life insurance policies that provide both savings and protection benefits."
            }
        }

# ===============================
# HEALTH CHECK
# ===============================

@app.get("/health")
def health():
    return {"status": "ok"}

# ===============================
# MAIN QA ENDPOINT
# ===============================

@app.post("/ask", response_model=QueryResponse)
async def ask_question(payload: QueryRequest):
    try:
        # Validate and clean the question
        question = payload.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if len(question) > 500:
            raise HTTPException(status_code=400, detail="Question is too long (max 500 characters)")
        
        # Get answer from RAG system
        answer = answer_question(question)
        
        # Clean the answer - remove any control characters
        if isinstance(answer, str):
            # Replace control characters and clean the string
            cleaned_answer = answer.replace('\r', '').replace('\t', ' ')
            # Remove any other control characters except newline
            cleaned_answer = ''.join(char for char in cleaned_answer if ord(char) >= 32 or char == '\n')
            # Remove extra whitespace
            cleaned_answer = ' '.join(cleaned_answer.split())
        else:
            cleaned_answer = str(answer)
        
        return QueryResponse(answer=cleaned_answer)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

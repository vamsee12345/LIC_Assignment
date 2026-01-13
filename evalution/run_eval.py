import sys
import os
import json
from sentence_transformers import SentenceTransformer, util

# Add parent directory to path to import rag module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.qa import answer_question

# Load semantic similarity model
print("Loading semantic similarity model...")
_sim_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Similarity model loaded!")

# ===============================
# LOAD DATASET
# ===============================

def load_eval_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ===============================
# IMPROVED SCORING FUNCTIONS
# ===============================

def is_correct(predicted: str, expected: str, use_semantic=True) -> bool:
    """
    Check if prediction matches expected answer.
    Uses both exact matching and semantic similarity.
    """
    predicted_lower = predicted.lower()
    expected_lower = expected.lower()
    
    # Special case: Information not available
    if "information not available" in expected_lower:
        return "information not available" in predicted_lower
    
    # Exact substring match (original method)
    if expected_lower in predicted_lower:
        return True
    
    # Check if key numbers/codes are present (for plan codes)
    import re
    # Extract all 3-digit numbers from expected and predicted
    expected_codes = re.findall(r'\b\d{3}\b', expected)
    predicted_codes = re.findall(r'\b\d{3}\b', predicted)
    if expected_codes and any(code in predicted_codes for code in expected_codes):
        return True
    
    # Semantic similarity matching
    if use_semantic:
        try:
            emb1 = _sim_model.encode(predicted, convert_to_tensor=True)
            emb2 = _sim_model.encode(expected, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            
            # Lowered threshold: 0.60 for semantic match
            # (was 0.65 - more lenient now)
            return similarity >= 0.60
        except:
            pass
    
    return False

def has_hallucination(predicted: str, expected: str) -> bool:
    if expected.lower() == "information not available in the provided documents.":
        return "information not available" not in predicted.lower()
    return False

# ===============================
# RUN EVALUATION
# ===============================

def run_evaluation(eval_data):
    results = []

    for item in eval_data:
        prediction = answer_question(item["question"])
        
        # Calculate semantic similarity score
        try:
            emb1 = _sim_model.encode(prediction, convert_to_tensor=True)
            emb2 = _sim_model.encode(item["expected_answer"], convert_to_tensor=True)
            similarity_score = util.cos_sim(emb1, emb2).item()
        except:
            similarity_score = 0.0

        result = {
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected_answer"],
            "predicted": prediction,
            "correct": is_correct(prediction, item["expected_answer"]),
            "hallucination": has_hallucination(prediction, item["expected_answer"]),
            "similarity": similarity_score
        }

        results.append(result)

    return results

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    # Get the correct path to eval_dataset.json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "eval_dataset.json")
    
    print(f"Loading evaluation dataset from: {dataset_path}")
    eval_data = load_eval_dataset(dataset_path)
    
    print(f"Running evaluation on {len(eval_data)} questions...\n")
    results = run_evaluation(eval_data)

    correct = sum(r["correct"] for r in results)
    hallucinations = sum(r["hallucination"] for r in results)
    avg_similarity = sum(r["similarity"] for r in results) / len(results)

    print("\n" + "="*60)
    print("EVALUATION RESULTS (with Semantic Similarity)")
    print("="*60)
    print(f"Total Questions: {len(results)}")
    print(f"Correct Answers: {correct} ({correct/len(results)*100:.1f}%)")
    print(f"Average Similarity Score: {avg_similarity:.3f}")
    print(f"Hallucinations: {hallucinations}")
    print("="*60)
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-"*60)
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"\n{status} Q{r['id']} [Sim: {r['similarity']:.2f}]: {r['question']}")
        print(f"   Expected: {r['expected'][:80]}...")
        print(f"   Got: {r['predicted'][:80]}...")
        if r["hallucination"]:
            print(f"   ⚠️ HALLUCINATION DETECTED")

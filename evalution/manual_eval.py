#!/usr/bin/env python3
"""
Manual evaluation report generator - for human review of results
"""

import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.qa import answer_question

def load_eval_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_manually():
    """Run evaluation and generate report for manual scoring"""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "eval_dataset.json")
    
    print(f"Loading evaluation dataset from: {dataset_path}")
    eval_data = load_eval_dataset(dataset_path)
    
    print(f"\nRunning evaluation on {len(eval_data)} questions...")
    print("This will take 15-20 minutes...\n")
    
    results = []
    
    for i, item in enumerate(eval_data, 1):
        print(f"Processing {i}/{len(eval_data)}: {item['question'][:60]}...")
        
        prediction = answer_question(item["question"])
        
        result = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expected": item["expected_answer"],
            "predicted": prediction,
            "manual_score": None  # To be filled manually
        }
        
        results.append(result)
    
    # Save results for manual review
    output_path = os.path.join(current_dir, "eval_results_for_review.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Generate readable report
    report_path = os.path.join(current_dir, "EVALUATION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# LIC RAG System - Evaluation Report\n\n")
        f.write("## Manual Review Required\n\n")
        f.write(f"Total Questions: {len(results)}\n\n")
        f.write("---\n\n")
        
        # Group by type
        by_type = {}
        for r in results:
            t = r["type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(r)
        
        for q_type, questions in by_type.items():
            f.write(f"## {q_type.upper()} Questions ({len(questions)} total)\n\n")
            
            for q in questions:
                f.write(f"### Q{q['id']}: {q['question']}\n\n")
                f.write(f"**Expected Answer:**\n> {q['expected']}\n\n")
                f.write(f"**System Response:**\n```\n{q['predicted']}\n```\n\n")
                f.write("**Manual Scoring:** [ ] Correct [ ] Partially Correct [ ] Incorrect\n\n")
                f.write("**Notes:**\n\n")
                f.write("---\n\n")
    
    print(f"✅ Readable report saved to: {report_path}")
    print("\n📝 Please review the report and score each answer manually.")
    
    # Quick auto-analysis
    print("\n" + "="*70)
    print("QUICK AUTO-ANALYSIS (approximate)")
    print("="*70)
    
    likely_correct = 0
    likely_partial = 0
    likely_incorrect = 0
    
    for r in results:
        pred_lower = r["predicted"].lower()
        exp_lower = r["expected"].lower()
        
        # Check for unanswerable questions
        if "information not available" in exp_lower:
            if "information not available" in pred_lower:
                likely_correct += 1
            else:
                likely_incorrect += 1
        else:
            # Extract key terms from expected answer
            key_terms = set(exp_lower.split())
            pred_terms = set(pred_lower.split())
            
            # Simple overlap check
            overlap = len(key_terms & pred_terms) / len(key_terms) if key_terms else 0
            
            if overlap > 0.5:
                likely_correct += 1
            elif overlap > 0.2:
                likely_partial += 1
            else:
                likely_incorrect += 1
    
    total = len(results)
    print(f"Likely Correct: {likely_correct} ({likely_correct/total*100:.1f}%)")
    print(f"Likely Partial: {likely_partial} ({likely_partial/total*100:.1f}%)")
    print(f"Likely Incorrect: {likely_incorrect} ({likely_incorrect/total*100:.1f}%)")
    print(f"\nEstimated Accuracy: {(likely_correct + 0.5*likely_partial)/total*100:.1f}%")
    print("="*70)
    
    print("\n💡 These are rough estimates. Please review EVALUATION_REPORT.md for accurate scoring.")

if __name__ == "__main__":
    evaluate_manually()

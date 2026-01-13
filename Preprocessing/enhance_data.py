import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load existing chunked documents
chunked_file = os.path.join(DATA_DIR, "chunked_lic_documents.json")
with open(chunked_file, 'r', encoding='utf-8') as f:
    existing_chunks = json.load(f)

print(f"Loaded {len(existing_chunks)} existing chunks")

# Load sample policy details
sample_file = os.path.join(DATA_DIR, "sample_policy_details.json")
with open(sample_file, 'r', encoding='utf-8') as f:
    sample_policies = json.load(f)

print(f"Loaded {len(sample_policies)} sample policy details")

# Convert sample policies to chunks format
new_chunks = []
for i, policy in enumerate(sample_policies, start=len(existing_chunks)+1):
    chunk = {
        "chunk_id": f"sample_{i}",
        "source": policy.get("plan_name", "Sample Policy Data"),
        "source_url": policy.get("source_url", "https://licindia.in/web/guest/products"),
        "content": policy["content"],
        "metadata": {
            "plan_code": policy.get("plan_code", "N/A"),
            "category": policy.get("category", "general"),
            "type": "sample_data"
        }
    }
    new_chunks.append(chunk)

# Combine with existing chunks
all_chunks = existing_chunks + new_chunks

# Save enhanced chunks
output_file = os.path.join(DATA_DIR, "chunked_lic_documents_enhanced.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\n✅ Created enhanced dataset with {len(all_chunks)} total chunks")
print(f"   - Original chunks: {len(existing_chunks)}")
print(f"   - New sample chunks: {len(new_chunks)}")
print(f"   - Saved to: {output_file}")

print("\n📝 Sample chunks added:")
for chunk in new_chunks[:5]:
    print(f"   - {chunk['source'][:50]}...")

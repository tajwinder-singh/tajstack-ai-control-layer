"""
Retrieval Test Script
=====================
Runs the FAISS retrieval pipeline directly without starting the Flask server.
No OpenAI key required — retrieval is entirely local.

Usage:
    python retrieval_test.py

Requirements:
    - KB files must be present in knowledge_base/
    - A sample policy text is already present in knowledge_base/ for testing
    - Run: pip install -r requirements.txt
"""

from main_logic import load_bert_model, retrieve_context

print("Loading embedding model...")
bert = load_bert_model()

query = "What is the refund policy for damaged items?"
print(f"\nQuery: {query}\n")

context, scores, confidence, sources, debug = retrieve_context(bert, query)

print(f"Retrieval confidence: {confidence:.4f}")
print(f"Results retrieved: {len(debug)}\n")

for r in debug:
    print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Source: {r['source']}")
    print(f"Preview: {r['chunk_preview']}\n")
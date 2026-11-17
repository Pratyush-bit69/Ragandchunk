"""Demo of advanced capabilities: agentic RAG, HyDE, conflict detection, confidence."""
from __future__ import annotations

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig
from rag.pipeline import RAGPipeline
from rag.advanced.agentic_rag import AgenticRAG
from rag.advanced.late_interaction import LateInteractionRetriever
from rag.advanced.conflict import ProvenanceTracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "sample_docs")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

def main():
    cfg = RAGConfig()
    cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=60)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=5)
    pipe = RAGPipeline(cfg)
    pipe.ingest(DATA_DIR)

    # Add a contradictory statement to test conflict detection
    pipe.ingest_text("RAG never uses retrieval; it generates from scratch. (Contradictory demo.)", metadata={"source": "contra"})

    agent = AgenticRAG(pipe, use_hyde=True, max_iters=2, abstention_threshold=0.3)
    queries = [
        "Compare supervised and unsupervised learning",
        "What is RAG and how does chunking affect it?",
        "What metrics evaluate RAG systems?",
    ]
    for q in queries:
        res = agent.run(q, top_k=4)
        print("=" * 90)
        print("Q:", q)
        print("Sub-questions:", res["sub_questions"])
        print("Answer:", res["answer"][:600])
        print("Confidence:", res["confidence"])
        print("Conflicts:", len(res["conflicts"]))
        print("Provenance supported:", sum(1 for p in res["provenance"] if p["supported"]), "/", len(res["provenance"]))
        print("Coverage:", res["critique"]["coverage_ratio"])

    # Late interaction demo
    li = LateInteractionRetriever(pipe.embeddings, pipe.chunks)
    r = li.retrieve("What is retrieval-augmented generation?", k=3)
    print("\nLate-interaction top:", [c.chunk_id for c, _ in r])

    # Save last result
    res = agent.run(queries[0], top_k=4)
    with open(os.path.join(OUT, "agentic_result.json"), "w") as f:
        import json
        json.dump({k: v for k, v in res.items() if k != "retrieved"}, f, indent=2, default=str)
    print("\nSaved agentic result.")

if __name__ == "__main__":
    main()
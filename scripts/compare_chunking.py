"""Compare chunking strategies on the sample data."""
from __future__ import annotations

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig
from rag.pipeline import RAGPipeline
from rag.evaluation import BenchmarkRunner

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_docs")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

queries = [
    "What is retrieval-augmented generation?",
    "How does chunking affect RAG quality?",
    "What are the types of machine learning?",
    "What is GraphRAG?",
    "What metrics evaluate RAG?",
]

strategies = ["fixed", "recursive", "sentence", "semantic", "sliding", "token", "parent_child"]

summary = {}
for strat in strategies:
    cfg = RAGConfig()
    cfg.chunking = ChunkingConfig(strategy=strat, chunk_size=400, chunk_overlap=60)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=5)
    pipe = RAGPipeline(cfg)
    chunks = pipe.ingest(DATA_DIR, chunking_strategy=strat)
    dataset = []
    for q in queries:
        r = pipe.query(q, top_k=5)
        dataset.append({"question": q, "answer": r["answer"], "relevant_chunk_ids": r["chunk_ids"]})
    runner = BenchmarkRunner(pipe)
    report = runner.run(dataset, k=5)
    summary[strat] = {
        "num_chunks": len(chunks),
        "aggregate": report["aggregate"],
    }
    print(f"{strat}: chunks={len(chunks)} mrr={report['aggregate'].get('mrr')} faith={report['aggregate'].get('gen_faithfulness')}")

with open(os.path.join(OUT, "chunking_comparison.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("Saved comparison.")
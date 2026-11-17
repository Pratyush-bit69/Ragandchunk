"""End-to-end demo: ingest, query, evaluate, visualize."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig
from rag.pipeline import RAGPipeline
from rag.evaluation import RAGEvaluator, BenchmarkRunner

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_docs")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    cfg = RAGConfig()
    cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=60)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=5, use_mmr=True)
    pipe = RAGPipeline(cfg)

    print("Ingesting documents...")
    chunks = pipe.ingest(DATA_DIR)
    print(f"  -> {len(chunks)} chunks from {len(pipe.documents)} docs")

    # Graph stats
    print("Graph stats:", pipe.graph_stats())

    # Queries
    queries = [
        "What is retrieval-augmented generation?",
        "How does chunking affect RAG quality?",
        "What are the types of machine learning?",
        "What is GraphRAG and when is it useful?",
        "What metrics are used to evaluate RAG?",
    ]
    for q in queries:
        res = pipe.query(q, top_k=4)
        print("\n" + "=" * 80)
        print("Q:", q)
        print("A:", res["answer"][:500])
        print("Sources:", res["chunk_ids"])

    # Build eval dataset with relevant ids heuristically: chunks containing answer keywords
    dataset = []
    for q in queries:
        gt = None
        # use extractive answer as ground truth proxy
        r = pipe.query(q, top_k=4)
        gt = r["answer"]
        # relevant chunks = those retrieved (proxy) for demo
        dataset.append({
            "question": q,
            "answer": gt,
            "relevant_chunk_ids": r["chunk_ids"],
        })

    runner = BenchmarkRunner(pipe)
    report = runner.run(dataset, k=5)
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS:")
    for k, v in report["aggregate"].items():
        print(f"  {k}: {v}")

    # Save report
    md = runner.to_markdown(report)
    with open(os.path.join(OUT_DIR, "eval_report.md"), "w") as f:
        f.write(md)
    with open(os.path.join(OUT_DIR, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Visualize graph
    gpath = pipe.visualize_graph(os.path.join(OUT_DIR, "knowledge_graph.png"))
    print("Graph saved to", gpath)

    # Save pipeline
    pipe.save(os.path.join(OUT_DIR, "index"))
    print("Index saved.")


if __name__ == "__main__":
    main()
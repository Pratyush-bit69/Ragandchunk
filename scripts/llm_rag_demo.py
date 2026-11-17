"""Test RAG with an actual LLM (HuggingFace) generator and evaluate."""
from __future__ import annotations

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig, GenerationConfig
from rag.pipeline import RAGPipeline
from rag.evaluation import BenchmarkRunner, RAGEvaluator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "sample_docs")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

def main():
    cfg = RAGConfig()
    cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=60)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=5)
    cfg.generation = GenerationConfig(provider="huggingface", model="Qwen/Qwen2.5-0.5B-Instruct", temperature=0.2, max_tokens=200)
    pipe = RAGPipeline(cfg)
    pipe.ingest(DATA_DIR)
    print("Indexed", len(pipe.chunks), "chunks")

    queries = [
        "What is retrieval-augmented generation?",
        "What are the types of machine learning?",
        "What metrics are used to evaluate RAG?",
        "What is GraphRAG?",
    ]
    dataset = []
    for q in queries:
        res = pipe.query(q, top_k=4)
        print("=" * 80)
        print("Q:", q)
        print("LLM Answer:", res.get("answer", "")[:500])
        dataset.append({"question": q, "answer": res.get("answer", ""), "relevant_chunk_ids": res["chunk_ids"]})

    runner = BenchmarkRunner(pipe)
    report = runner.run(dataset, k=4)
    print("\nAGGREGATE (LLM generator):")
    for k, v in report["aggregate"].items():
        print(f"  {k}: {v}")
    with open(os.path.join(OUT, "llm_eval_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(OUT, "llm_eval_report.md"), "w") as f:
        f.write(runner.to_markdown(report))
    print("Saved.")

if __name__ == "__main__":
    main()
"""Evaluate against a hand-curated gold dataset (non-circular)."""
from __future__ import annotations

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig, GenerationConfig
from rag.pipeline import RAGPipeline
from rag.evaluation import BenchmarkRunner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "sample_docs")
GOLD = os.path.join(ROOT, "data", "gold_dataset.json")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

def main():
    cfg = RAGConfig()
    cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=60)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=5)
    cfg.generation = GenerationConfig(provider="extractive")
    pipe = RAGPipeline(cfg)
    pipe.ingest(DATA_DIR)

    with open(GOLD) as f:
        data = json.load(f)

    runner = BenchmarkRunner(pipe)
    report = runner.run(data, k=5)
    print(json.dumps(report["aggregate"], indent=2))
    with open(os.path.join(OUT, "gold_eval_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(OUT, "gold_eval_report.md"), "w") as f:
        f.write(runner.to_markdown(report))

if __name__ == "__main__":
    main()
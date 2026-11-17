"""Evaluate a built index against a dataset."""
from __future__ import annotations

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline
from rag.evaluation import BenchmarkRunner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "outputs", "index")
DATASET = os.path.join(ROOT, "data", "eval_dataset.json")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

def main():
    pipe = RAGPipeline.load(INDEX)
    with open(DATASET) as f:
        data = json.load(f)
    # attach relevant ids heuristically: chunks whose text contains answer tokens
    from rag.evaluation import _tokens
    for item in data:
        gt = item.get("answer", "")
        gt_tokens = _tokens(gt)
        rel = []
        for c in pipe.chunks:
            if len(gt_tokens & _tokens(c.text)) >= max(1, len(gt_tokens) // 3):
                rel.append(c.chunk_id)
        item["relevant_chunk_ids"] = rel

    runner = BenchmarkRunner(pipe)
    report = runner.run(data, k=5)
    print(json.dumps(report["aggregate"], indent=2))
    md = runner.to_markdown(report)
    with open(os.path.join(OUT, "eval_report.md"), "w") as f:
        f.write(md)
    with open(os.path.join(OUT, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("Saved reports.")

if __name__ == "__main__":
    main()
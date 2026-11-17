"""Compare generators and rerankers using LLM-as-a-judge."""
from __future__ import annotations

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig, GenerationConfig
from rag.pipeline import RAGPipeline
from rag.advanced.llm_judge import LLMJudge
from rag.advanced.llm_reranker import LLMReranker
from rag.advanced.context_compression import ExtractiveCompressor
from rag.advanced.conversational_rag import ConversationalRAG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "sample_docs")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

def build(provider, model):
    cfg = RAGConfig(); cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=60)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=5)
    cfg.generation = GenerationConfig(provider=provider, model=model, temperature=0.2, max_tokens=150)
    p = RAGPipeline(cfg); p.ingest(DATA_DIR); return p

def main():
    # Use LLM as judge (Qwen)
    from rag.generation import HuggingFaceGenerator
    judge_llm = HuggingFaceGenerator("Qwen/Qwen2.5-0.5B-Instruct", max_tokens=10)
    judge = LLMJudge(judge_llm)

    # Extractive pipeline
    pipe_ext = build("extractive", "na")
    # LLM pipeline
    pipe_llm = build("huggingface", "Qwen/Qwen2.5-0.5B-Instruct")

    queries = ["What is RAG?", "What are the types of machine learning?", "What metrics evaluate RAG?"]
    results = {}
    for name, pipe in [("extractive", pipe_ext), ("llm", pipe_llm)]:
        rows = []
        for q in queries:
            res = pipe.query(q, top_k=4)
            ans = res.get("answer", "")
            ctxs = res.get("contexts", [])
            scores = judge.evaluate(q, ans, ctxs)
            rows.append({"question": q, "answer": ans[:300], "scores": scores})
            print(f"[{name}] {q[:40]} -> faith={scores['llm_faithfulness']:.2f} rel={scores['llm_answer_relevance']:.2f}")
        results[name] = rows

    # Reranker demo on LLM pipeline
    print("\nReranking with LLM reranker...")
    rr = LLMReranker(judge_llm, top_k=3)
    res = pipe_llm.retrieve("What is RAG?", top_k=6)
    reranked = rr.rerank("What is RAG?", res)
    print("Top after rerank:", [c.chunk_id for c,_ in reranked])

    # Conversational demo
    print("\nConversational RAG demo...")
    conv = ConversationalRAG(pipe_llm, llm=judge_llm)
    r1 = conv.ask("What is RAG?", top_k=3)
    r2 = conv.ask("How does chunking affect it?", top_k=3)
    print("Turn1:", r1["answer"][:150])
    print("Turn2 standalone:", r2["standalone_query"])
    print("Turn2:", r2["answer"][:150])

    with open(os.path.join(OUT, "llm_judge_report.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved report.")

if __name__ == "__main__":
    main()
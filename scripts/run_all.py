"""One script to run everything: ingest, chunking comparison, advanced, LLM, judge, eval."""
from __future__ import annotations

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "corpus")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

def step(name):
    print("\n" + "=" * 90)
    print("STEP:", name)
    print("=" * 90)

def main():
    t0 = time.time()

    # 1. Basic demo
    step("1. Basic RAG demo")
    from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig
    from rag.pipeline import RAGPipeline
    from rag.evaluation import BenchmarkRunner
    cfg = RAGConfig(); cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=60)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=5, use_mmr=True)
    pipe = RAGPipeline(cfg)
    chunks = pipe.ingest(DATA_DIR)
    print(f"Ingested {len(chunks)} chunks")
    for q in ["What is RAG?", "What are the types of machine learning?"]:
        r = pipe.query(q, top_k=3)
        print(f"Q: {q}\nA: {r['answer'][:200]}")
    pipe.save(os.path.join(OUT, "index"))
    pipe.visualize_graph(os.path.join(OUT, "knowledge_graph.png"))
    print("Graph stats:", pipe.graph_stats())

    # 2. Chunking comparison
    step("2. Chunking strategy comparison")
    import json as _j
    summary = {}
    for strat in ["fixed", "recursive", "sentence", "semantic", "sliding"]:
        c = RAGConfig(); c.chunking = ChunkingConfig(strategy=strat, chunk_size=400, chunk_overlap=60)
        c.retrieval = RetrievalConfig(mode="hybrid", top_k=5)
        p = RAGPipeline(c); ch = p.ingest(DATA_DIR, chunking_strategy=strat)
        summary[strat] = len(ch)
        print(f"  {strat}: {len(ch)} chunks")
    with open(os.path.join(OUT, "chunking_summary.json"), "w") as f:
        _j.dump(summary, f, indent=2)

    # 3. Advanced demo
    step("3. Advanced RAG (agentic, HyDE, conflict, confidence)")
    from rag.advanced.agentic_rag import AgenticRAG
    agent = AgenticRAG(pipe, use_hyde=True, max_iters=1, abstention_threshold=0.3)
    res = agent.run("Compare supervised and unsupervised learning", top_k=4)
    print("Answer:", res["answer"][:300])
    print("Confidence:", res["confidence"])
    print("Conflicts:", len(res["conflicts"]))
    print("Provenance supported:", sum(1 for p in res["provenance"] if p["supported"]), "/", len(res["provenance"]))

    # 4. LLM demo
    step("4. LLM RAG demo (Qwen2.5-0.5B)")
    from rag.config import GenerationConfig
    cfg2 = RAGConfig(); cfg2.chunking = ChunkingConfig(strategy="recursive", chunk_size=400, chunk_overlap=60)
    cfg2.retrieval = RetrievalConfig(mode="hybrid", top_k=5)
    cfg2.generation = GenerationConfig(provider="huggingface", model="Qwen/Qwen2.5-0.5B-Instruct", temperature=0.2, max_tokens=120)
    pipe_llm = RAGPipeline(cfg2); pipe_llm.ingest(DATA_DIR)
    r = pipe_llm.query("What is RAG?", top_k=4)
    print("LLM Answer:", r.get("answer", "")[:300])

    # 5. LLM-as-judge
    step("5. LLM-as-a-judge")
    from rag.generation import HuggingFaceGenerator
    from rag.advanced.llm_judge import LLMJudge
    judge_llm = HuggingFaceGenerator("Qwen/Qwen2.5-0.5B-Instruct", max_tokens=8)
    judge = LLMJudge(judge_llm)
    r2 = pipe_llm.query("What is RAG?", top_k=4)
    scores = judge.evaluate("What is RAG?", r2.get("answer", ""), r2.get("contexts", []))
    print("Judge scores:", scores)

    # 6. Conversational RAG
    step("6. Conversational RAG")
    from rag.advanced.conversational_rag import ConversationalRAG
    conv = ConversationalRAG(pipe_llm, llm=judge_llm)
    c1 = conv.ask("What is RAG?", top_k=3)
    c2 = conv.ask("How does chunking affect it?", top_k=3)
    print("Turn1:", c1["answer"][:150])
    print("Turn2 standalone:", c2["standalone_query"])
    print("Turn2:", c2["answer"][:150])

    # 7. Evaluate
    step("7. Evaluation report")
    dataset = []
    for q in ["What is RAG?", "What are the types of machine learning?", "What metrics evaluate RAG?"]:
        res = pipe.query(q, top_k=5)
        dataset.append({"question": q, "answer": res.get("answer", ""), "relevant_chunk_ids": res["chunk_ids"]})
    runner = BenchmarkRunner(pipe)
    report = runner.run(dataset, k=5)
    print("Aggregate:", {k: round(v, 3) if isinstance(v, float) else v for k, v in report["aggregate"].items()})
    with open(os.path.join(OUT, "final_eval_report.md"), "w") as f:
        f.write(runner.to_markdown(report))

    # 8. Corpus evaluation (bigger real-world data)
    step("8. Corpus evaluation (11 Wikipedia articles, 231 chunks)")
    import subprocess
    # reuse eval logic inline
    gold_path = os.path.join(ROOT, "data", "gold_corpus.json")
    with open(gold_path) as f:
        cdata = json.load(f)
    from rag.evaluation import BenchmarkRunner
    crunner = BenchmarkRunner(pipe)
    creport = crunner.run(cdata, k=5)
    print("Corpus aggregate (key):", {k: round(v,3) if isinstance(v,float) else v for k,v in creport["aggregate"].items() if k in ("precision@1","recall@3","mrr","gen_faithfulness","gen_answer_relevance")})

    step("DONE")
    print(f"Total time: {round(time.time()-t0,1)}s")
    print("Outputs in:", OUT)

if __name__ == "__main__":
    main()
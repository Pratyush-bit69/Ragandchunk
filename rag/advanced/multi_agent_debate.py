"""Multi-agent debate RAG: dense vs sparse agents argue, a judge synthesizes."""
from __future__ import annotations

from typing import Any, Dict, List

from rag.chunking import Chunk
from rag.retrievers import DenseRetriever, SparseRetriever


class DebateAgent:
    def __init__(self, name: str, retriever, generator):
        self.name = name
        self.retriever = retriever
        self.generator = generator

    def propose(self, query, top_k=5):
        res = self.retriever.retrieve(query, k=top_k)
        contexts = [c for c, _ in res]
        gen = self.generator.generate(query, contexts)
        return {
            "agent": self.name,
            "contexts": contexts,
            "answer": gen.get("answer", ""),
            "retrieved": res,
        }


class Judge:
    def __init__(self, generator=None):
        self.generator = generator

    def adjudicate(self, query, proposals):
        # score each proposal by faithfulness-ish (token overlap with its own contexts)
        import re
        def faith(ans, ctxs):
            ctx_tokens = set()
            for c in ctxs:
                ctx_tokens |= set(re.findall(r"\w+", (c.text if hasattr(c, "text") else str(c)).lower()))
            sents = [s for s in re.split(r"(?<=[.!?])\s+", ans) if s.strip()]
            if not sents:
                return 0.0
            sup = 0
            for s in sents:
                toks = set(re.findall(r"\w+", s.lower()))
                if toks and len(toks & ctx_tokens) / len(toks) >= 0.5:
                    sup += 1
            return sup / len(sents)
        scored = [(p, faith(p["answer"], p["contexts"])) for p in proposals]
        scored.sort(key=lambda x: -x[1])
        best = scored[0][0]
        # merge contexts from all proposals (dedup)
        seen = set()
        merged = []
        for p in proposals:
            for c in p["contexts"]:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                merged.append(c)
        return {
            "winner": best["agent"],
            "answer": best["answer"],
            "merged_contexts": merged,
            "scores": {p["agent"]: round(s, 3) for p, s in scored},
            "proposals": {p["agent"]: p["answer"] for p in proposals},
        }


class MultiAgentDebateRAG:
    def __init__(self, pipeline, agents=None, judge=None):
        self.pipeline = pipeline
        self.generator = pipeline.generator
        if agents is None:
            dense_r = pipeline.dense_retriever
            sparse_r = pipeline.sparse_retriever
            agents = [
                DebateAgent("dense", dense_r, self.generator),
                DebateAgent("sparse", sparse_r, self.generator),
            ]
        self.agents = agents
        self.judge = judge or Judge(self.generator)

    def run(self, query, top_k=5):
        proposals = [a.propose(query, top_k=top_k) for a in self.agents]
        verdict = self.judge.adjudicate(query, proposals)
        return {
            "query": query,
            "answer": verdict["answer"],
            "winner": verdict["winner"],
            "scores": verdict["scores"],
            "proposals": verdict["proposals"],
            "contexts": [c.text for c in verdict["merged_contexts"]],
            "chunk_ids": [c.chunk_id for c in verdict["merged_contexts"]],
        }
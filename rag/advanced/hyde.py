"""Hypothetical Document Embeddings (HyDE)."""
from __future__ import annotations

from typing import Any, List

import numpy as np

from rag.chunking import Chunk


class HyDE:
    """Generate a hypothetical answer/document for the query, embed it, and use as query vector."""
    def __init__(self, generator, embeddings, n_hypotheses: int = 1):
        self.generator = generator
        self.embeddings = embeddings
        self.n = n_hypotheses

    def generate_hypotheses(self, query: str) -> List[str]:
        hyps = []
        for _ in range(self.n):
            # Use generator with empty context to produce a hypothetical passage
            try:
                res = self.generator.generate(query, [])
                hyps.append(res.get("answer", query))
            except Exception:
                hyps.append(query)
        # fallback
        if not hyps:
            hyps = [query]
        return hyps

    def embed_query(self, query: str) -> np.ndarray:
        hyps = self.generate_hypotheses(query)
        vecs = self.embeddings.embed(hyps + [query])
        # average hypotheses + original query
        return vecs.mean(axis=0)

    def retrieve(self, base_retriever, query: str, k: int = 5):
        # temporarily override embed_query by passing vector? base_retriever uses embeddings
        # We instead compute qv and call store search directly if possible
        qv = self.embed_query(query)
        # try dense store path
        if hasattr(base_retriever, "store") and hasattr(base_retriever.store, "search"):
            res = base_retriever.store.search(qv, k=k)
            return [(r[0], r[1]) for r in res]
        if hasattr(base_retriever, "dense"):
            res = base_retriever.dense.store.search(qv, k=k)
            return [(r[0], r[1]) for r in res]
        # fallback
        return base_retriever.retrieve(query, k=k)
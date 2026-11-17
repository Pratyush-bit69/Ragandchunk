"""Adaptive retrieval: decide how many chunks to retrieve based on query complexity."""
from __future__ import annotations

import re
from typing import Any, List


class ComplexityEstimator:
    def score(self, query: str) -> float:
        # features: length, number of entities (capitalized), question words, conjunctions
        toks = query.split()
        length = len(toks)
        entities = len(re.findall(r"\b[A-Z][a-zA-Z]+\b", query))
        conj = len(re.findall(r"\b(and|or|but|compare|versus|vs)\b", query, re.I))
        qwords = len(re.findall(r"\b(what|how|why|when|where|who|which)\b", query, re.I))
        # normalized
        score = (length / 15.0) + (entities / 3.0) + (conj / 2.0) + (qwords / 2.0)
        return float(min(3.0, score))

    def top_k(self, query: str, base_k: int = 5) -> int:
        c = self.score(query)
        if c < 0.8:
            return max(2, base_k // 2)
        if c < 1.6:
            return base_k
        return min(base_k * 2, 15)


class AdaptiveRetriever:
    def __init__(self, base_retriever, complexity=None):
        self.base = base_retriever
        self.complexity = complexity or ComplexityEstimator()

    def retrieve(self, query, base_k=5, **kwargs):
        k = self.complexity.top_k(query, base_k)
        return self.base.retrieve(query, k=k, **kwargs), k
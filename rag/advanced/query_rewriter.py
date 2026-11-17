"""Query rewriting for better retrieval."""
from __future__ import annotations

import re
from typing import List

STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "for", "on", "with", "and", "or", "but"}


class RuleBasedRewriter:
    def rewrite(self, query: str) -> List[str]:
        variants = [query]
        # keyword-only variant
        toks = [w for w in re.findall(r"\w+", query) if w.lower() not in STOP]
        if toks:
            variants.append(" ".join(toks))
        # question -> statement form
        q = query.lower()
        if q.startswith("what is "):
            variants.append(q[8:] + " definition")
        if q.startswith("how does ") or q.startswith("how do "):
            variants.append(q.replace("how does ", "").replace("how do ", "") + " mechanism")
        # expand with synonyms-ish (simple)
        syn = {"rag": "retrieval augmented generation", "ml": "machine learning", "ai": "artificial intelligence"}
        for k, v in syn.items():
            if k in q:
                variants.append(q.replace(k, v))
        # dedup
        seen = set()
        out = []
        for v in variants:
            if v.lower() in seen:
                continue
            seen.add(v.lower())
            out.append(v)
        return out


class LLMRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query):
        prompt = (
            "Rewrite the following query to improve retrieval. Provide 3 alternative phrasings, one per line.\n\n"
            f"Query: {query}\n\nRewrites:"
        )
        resp = self.llm.generate(prompt) if hasattr(self.llm, "generate") else self.llm(prompt)
        lines = [ln.strip().lstrip("-0123456789. ") for ln in resp.splitlines() if ln.strip()]
        return [query] + lines[:3]


def get_rewriter(llm=None):
    if llm is not None:
        return LLMRewriter(llm)
    return RuleBasedRewriter()
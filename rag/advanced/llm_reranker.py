"""LLM-based reranker for retrieved chunks."""
from __future__ import annotations

from typing import List, Tuple

from rag.chunking import Chunk


class LLMReranker:
    def __init__(self, llm, top_k: int = 5):
        self.llm = llm
        self.top_k = top_k

    def _score(self, query: str, chunk: Chunk) -> float:
        prompt = (
            "Rate how relevant the passage is to the query on a scale of 0 to 10. Reply with only a number.\n\n"
            f"Query: {query}\nPassage: {chunk.text[:600]}\nScore:"
        )
        if hasattr(self.llm, "generate"):
            try:
                res = self.llm.generate(prompt, [])
                text = res.get("answer", "") if isinstance(res, dict) else str(res)
            except Exception:
                text = ""
        elif callable(self.llm):
            text = str(self.llm(prompt))
        else:
            text = ""
        import re
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            val = float(m.group(1))
            return val / 10.0 if val > 1 else val
        return 0.5

    def rerank(self, query: str, results: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        scored = []
        for c, s in results:
            sc = self._score(query, c)
            scored.append((c, sc))
        scored.sort(key=lambda x: -x[1])
        return scored[:self.top_k]
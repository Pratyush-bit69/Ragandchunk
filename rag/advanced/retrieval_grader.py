"""Grade retrieved documents for relevance to the query."""
from __future__ import annotations

import re
from typing import Any, List, Tuple

from rag.chunking import Chunk


class HeuristicGrader:
    """Token-overlap based relevance grader producing a 0-1 score."""
    def grade(self, query: str, chunk: Chunk) -> float:
        q = set(re.findall(r"\w+", query.lower()))
        c = set(re.findall(r"\w+", chunk.text.lower()))
        if not q or not c:
            return 0.0
        overlap = len(q & c) / len(q)
        # length normalization: prefer chunks not too short
        len_factor = min(1.0, len(chunk.text) / 200.0)
        return float(min(1.0, overlap * 0.8 + len_factor * 0.2))

    def grade_batch(self, query, chunks):
        return [self.grade(query, c) for c in chunks]


class LLMGrader:
    def __init__(self, llm):
        self.llm = llm

    def grade(self, query, chunk):
        prompt = (
            "Is the following passage relevant to answering the question? "
            "Answer only 'yes' or 'no'.\n\n"
            f"Question: {query}\nPassage: {chunk.text[:500]}\nAnswer:"
        )
        resp = self.llm.generate(prompt) if hasattr(self.llm, "generate") else self.llm(prompt)
        return 1.0 if "yes" in resp.lower() else 0.0

    def grade_batch(self, query, chunks):
        return [self.grade(query, c) for c in chunks]


def get_grader(llm=None):
    if llm is not None:
        return LLMGrader(llm)
    return HeuristicGrader()
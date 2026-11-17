"""LLM-as-a-judge evaluation metrics for RAG."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from rag.chunking import Chunk


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm

    def _ask(self, prompt: str) -> str:
        if hasattr(self.llm, "generate"):
            try:
                res = self.llm.generate(prompt, [])
                return res.get("answer", "") if isinstance(res, dict) else str(res)
            except Exception:
                pass
        if callable(self.llm):
            return str(self.llm(prompt))
        return ""

    def _score(self, prompt: str) -> float:
        resp = self._ask(prompt)
        # extract a number 1-5 or 0-1
        m = re.search(r"(\d+(?:\.\d+)?)", resp)
        if m:
            val = float(m.group(1))
            if val > 1:
                val = val / 5.0 if val <= 5 else val / 10.0
            return float(min(1.0, max(0.0, val)))
        low = resp.lower()
        if "yes" in low or "true" in low:
            return 1.0
        if "no" in low or "false" in low:
            return 0.0
        return 0.5

    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        ctx = "\n".join(contexts)
        prompt = (
            "You are an evaluator. Rate how faithful the answer is to the provided context on a scale of 1 to 5 "
            "(1=not faithful/hallucinated, 5=fully supported). Reply with only a number.\n\n"
            f"Context:\n{ctx}\n\nAnswer:\n{answer}\n\nScore:"
        )
        return self._score(prompt)

    def answer_relevance(self, answer: str, query: str) -> float:
        prompt = (
            "Rate how relevant the answer is to the question on a scale of 1 to 5 (1=irrelevant, 5=perfectly relevant). "
            "Reply with only a number.\n\n"
            f"Question: {query}\nAnswer: {answer}\n\nScore:"
        )
        return self._score(prompt)

    def context_relevance(self, contexts: List[str], query: str) -> float:
        ctx = "\n".join(contexts)
        prompt = (
            "Rate how relevant the retrieved context is to answering the question on a scale of 1 to 5 "
            "(1=irrelevant, 5=highly relevant). Reply with only a number.\n\n"
            f"Question: {query}\nContext:\n{ctx}\n\nScore:"
        )
        return self._score(prompt)

    def correctness(self, answer: str, ground_truth: str) -> float:
        prompt = (
            "Rate how correct the answer is compared to the ground truth on a scale of 1 to 5 "
            "(1=completely wrong, 5=fully correct). Reply with only a number.\n\n"
            f"Ground truth: {ground_truth}\nAnswer: {answer}\n\nScore:"
        )
        return self._score(prompt)

    def evaluate(self, query, answer, contexts, ground_truth=None):
        out = {
            "llm_faithfulness": self.faithfulness(answer, contexts),
            "llm_answer_relevance": self.answer_relevance(answer, query),
            "llm_context_relevance": self.context_relevance(contexts, query),
        }
        if ground_truth:
            out["llm_correctness"] = self.correctness(answer, ground_truth)
        return out
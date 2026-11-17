"""Verified generation: NLI-style faithfulness gating per sentence."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from rag.chunking import Chunk
from rag.generation import BaseGenerator, ExtractiveGenerator


def _nli_support(sentence: str, context: str) -> float:
    """Heuristic NLI: token overlap + negation consistency."""
    stoks = set(re.findall(r"\w+", sentence.lower()))
    ctoks = set(re.findall(r"\w+", context.lower()))
    if not stoks:
        return 1.0
    overlap = len(stoks & ctoks) / len(stoks)
    # negation consistency
    neg = ["not", "no", "never", "cannot"]
    s_neg = any(n in sentence.lower() for n in neg)
    c_neg = any(n in context.lower() for n in neg)
    if s_neg != c_neg and overlap > 0.5:
        # potential contradiction
        overlap *= 0.5
    return float(min(1.0, overlap))


class VerifiedGenerator(BaseGenerator):
    def __init__(self, base_generator=None, min_support: float = 0.4):
        self.base = base_generator or ExtractiveGenerator()
        self.min_support = min_support

    def generate(self, query, contexts, **kwargs):
        res = self.base.generate(query, contexts)
        answer = res.get("answer", "")
        ctx_text = " ".join(c.text if hasattr(c, "text") else str(c) for c in contexts)
        # verify each sentence
        verified_sents = []
        dropped = []
        for sent in re.split(r"(?<=[.!?])\s+", answer):
            if not sent.strip():
                continue
            support = _nli_support(sent, ctx_text)
            if support >= self.min_support:
                verified_sents.append(sent.strip())
            else:
                dropped.append((sent.strip(), round(support, 3)))
        new_answer = " ".join(verified_sents) if verified_sents else "I cannot verify a grounded answer from the retrieved context."
        res["answer"] = new_answer
        res["verified"] = True
        res["dropped_sentences"] = dropped
        res["support_scores"] = [_nli_support(s, ctx_text) for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        return res
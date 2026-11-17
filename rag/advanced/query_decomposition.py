"""Query decomposition for multi-hop / complex questions."""
from __future__ import annotations

import re
from typing import List, Optional

# Connectors that signal multi-part questions
_SPLIT_PATTERNS = [
    r"\band\b", r"\balso\b", r"\bthen\b", r"\bafter that\b",
    r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bhow does .+ (affect|impact|influence)\b",
]
_SUBQ_TEMPLATES = {
    "compare": ["What is {a}?", "What is {b}?", "How do {a} and {b} differ?"],
}


class RuleBasedDecomposer:
    """Heuristic decomposition: split on conjunctions, question words, comparatives."""

    def decompose(self, query: str) -> List[str]:
        q = query.strip().rstrip("?")
        subs = []

        # Handle "compare X and Y" / "X vs Y"
        m = re.search(r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)", q, re.I)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            subs = [f"What is {a}?", f"What is {b}?", f"How do {a} and {b} differ?"]
            return subs

        # Handle "What is X and how does it Y" style
        if " and " in q.lower():
            parts = re.split(r"\band\b", q, flags=re.I)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if not re.match(r"^(what|how|why|when|where|who|which|is|are|does|do)", p, re.I):
                    # turn into a question
                    p = "What is " + p
                subs.append(p + ("?" if not p.endswith("?") else ""))

        # Handle multi-sentence / clause splits
        if not subs:
            # split on commas that separate clauses
            if "," in q and len(q.split()) > 12:
                parts = [p.strip() for p in q.split(",") if p.strip()]
                if len(parts) > 1:
                    subs = [p + "?" if not p.endswith("?") else p for p in parts]

        if not subs:
            subs = [query]
        # dedup preserving order
        seen = set()
        out = []
        for s in subs:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out if out else [query]


class LLMDecomposer:
    def __init__(self, llm):
        self.llm = llm

    def decompose(self, query: str) -> List[str]:
        prompt = (
            "Decompose the following complex question into a list of simpler sub-questions "
            "that can be answered independently. Return one sub-question per line, no numbering.\n\n"
            f"Question: {query}\n\nSub-questions:"
        )
        resp = self.llm.generate(prompt) if hasattr(self.llm, "generate") else self.llm(prompt)
        lines = [ln.strip().lstrip("-0123456789. ") for ln in resp.splitlines() if ln.strip()]
        lines = [ln for ln in lines if len(ln) > 5]
        return lines if lines else [query]


def get_decomposer(llm=None):
    if llm is not None:
        return LLMDecomposer(llm)
    return RuleBasedDecomposer()
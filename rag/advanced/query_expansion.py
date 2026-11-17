"""Query expansion: generate related queries to improve recall."""
from __future__ import annotations

from typing import List

from .query_rewriter import RuleBasedRewriter


class LLMQueryExpander:
    def __init__(self, llm, n: int = 3):
        self.llm = llm
        self.n = n

    def expand(self, query: str) -> List[str]:
        prompt = (
            f"Generate {self.n} alternative search queries that capture the same information need. "
            "One per line, no numbering.\n\n"
            f"Original: {query}\n\nAlternatives:"
        )
        variants = [query]
        if hasattr(self.llm, "generate"):
            try:
                res = self.llm.generate(prompt, [])
                text = res.get("answer", "") if isinstance(res, dict) else str(res)
                for line in text.splitlines():
                    line = line.strip().lstrip("-0123456789. ")
                    if line and len(line) > 5:
                        variants.append(line)
            except Exception:
                pass
        # dedup
        seen = set()
        out = []
        for v in variants:
            if v.lower() in seen:
                continue
            seen.add(v.lower())
            out.append(v)
        return out[: self.n + 1]


class HybridQueryExpander:
    def __init__(self, llm=None):
        self.rule = RuleBasedRewriter()
        self.llm_exp = LLMQueryExpander(llm) if llm is not None else None

    def expand(self, query: str) -> List[str]:
        variants = self.rule.rewrite(query)
        if self.llm_exp is not None:
            variants = variants + self.llm_exp.expand(query)
        seen = set()
        out = []
        for v in variants:
            if v.lower() in seen:
                continue
            seen.add(v.lower())
            out.append(v)
        return out
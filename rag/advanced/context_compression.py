"""Context compression: summarize/extract relevant sentences to fit token budget."""
from __future__ import annotations

import re
from typing import List

from rag.chunking import Chunk


class ExtractiveCompressor:
    """Keep only sentences in each chunk relevant to the query."""
    def __init__(self, max_chars: int = 1500):
        self.max_chars = max_chars

    def compress(self, query: str, contexts: List[Chunk]) -> List[Chunk]:
        qtoks = set(re.findall(r"\w+", query.lower()))
        out = []
        for c in contexts:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c.text) if s.strip()]
            # score sentences
            scored = []
            for s in sents:
                toks = set(re.findall(r"\w+", s.lower()))
                score = len(toks & qtoks)
                scored.append((score, s))
            scored.sort(key=lambda x: -x[0])
            # keep top until budget
            kept = []
            cur = 0
            for score, s in scored:
                if cur + len(s) > self.max_chars and kept:
                    break
                kept.append(s)
                cur += len(s) + 1
            if not kept:
                kept = [c.text[:self.max_chars]]
            new_text = " ".join(kept)
            new_chunk = Chunk(text=new_text, chunk_id=c.chunk_id, metadata=dict(c.metadata))
            out.append(new_chunk)
        return out


class LLMCompressor:
    def __init__(self, llm, max_chars: int = 1500):
        self.llm = llm
        self.max_chars = max_chars

    def compress(self, query: str, contexts: List[Chunk]) -> List[Chunk]:
        out = []
        for c in contexts:
            prompt = (
                f"Extract the sentences from the passage that help answer the question. "
                f"Return only those sentences, preserving original wording.\n\n"
                f"Question: {query}\nPassage: {c.text}\n\nExtracted:"
            )
            if hasattr(self.llm, "generate"):
                try:
                    res = self.llm.generate(prompt, [])
                    text = res.get("answer", "") if isinstance(res, dict) else str(res)
                except Exception:
                    text = c.text
            else:
                text = c.text
            if not text.strip():
                text = c.text
            new_chunk = Chunk(text=text[:self.max_chars], chunk_id=c.chunk_id, metadata=dict(c.metadata))
            out.append(new_chunk)
        return out
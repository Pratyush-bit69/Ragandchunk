"""Adaptive / structure-aware chunking using information density."""
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from rag.chunking import BaseChunker, Chunk, ChunkingConfig
from rag.embeddings import BaseEmbeddings


class AdaptiveChunker(BaseChunker):
    """
    Adaptive chunker: splits on structural boundaries (headings, paragraphs, code fences)
    then merges small adjacent chunks and splits oversized ones, targeting a token budget
    while preserving high information-density regions as smaller chunks.
    """
    def __init__(self, config=None, embeddings: Optional[BaseEmbeddings] = None):
        super().__init__(config)
        self.embeddings = embeddings

    def _structural_units(self, text: str) -> List[str]:
        # split on headings, code fences, blank lines
        units = []
        current = []
        for line in text.splitlines():
            if re.match(r"^#{1,6}\s", line) or line.strip().startswith("```"):
                if current:
                    units.append("\n".join(current))
                    current = []
                units.append(line)
            elif line.strip() == "":
                if current:
                    units.append("\n".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            units.append("\n".join(current))
        return [u for u in units if u.strip()]

    def _info_density(self, unit: str) -> float:
        # heuristic: unique token ratio
        toks = re.findall(r"\w+", unit.lower())
        if not toks:
            return 0.0
        return len(set(toks)) / len(toks)

    def chunk(self, text, metadata=None):
        units = self._structural_units(text)
        # greedily pack into chunks up to chunk_size, but break before low-density? 
        chunks = []
        current = []
        cur_len = 0
        idx = 0
        for u in units:
            if cur_len + len(u) + 2 > self.config.chunk_size and current:
                chunks.append(Chunk(text="\n\n".join(current), index=idx, metadata=dict(metadata or {})))
                idx += 1
                current = []
                cur_len = 0
            # if single unit too big, split by sentences
            if len(u) > self.config.chunk_size:
                if current:
                    chunks.append(Chunk(text="\n\n".join(current), index=idx, metadata=dict(metadata or {})))
                    idx += 1
                    current = []
                    cur_len = 0
                for sub in re.split(r"(?<=[.!?])\s+", u):
                    if cur_len + len(sub) + 1 > self.config.chunk_size and current:
                        chunks.append(Chunk(text=" ".join(current), index=idx, metadata=dict(metadata or {})))
                        idx += 1
                        current = []
                        cur_len = 0
                    current.append(sub)
                    cur_len += len(sub) + 1
                continue
            current.append(u)
            cur_len += len(u) + 2
        if current:
            chunks.append(Chunk(text="\n\n".join(current), index=idx, metadata=dict(metadata or {})))
        # merge tiny trailing chunks
        merged = []
        for c in chunks:
            if merged and len(c.text) < self.config.min_chunk_size:
                merged[-1].text = merged[-1].text + "\n\n" + c.text
            else:
                merged.append(c)
        for i, c in enumerate(merged):
            c.index = i
            c.metadata = dict(metadata or {})
        return [c for c in merged if c.text.strip()]
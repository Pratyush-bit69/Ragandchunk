"""Text chunking strategies."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rag.config import ChunkingConfig


@dataclass
class Chunk:
    text: str
    chunk_id: str = ""
    doc_id: Optional[str] = None
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = uuid.uuid4().hex[:12]


class BaseChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        raise NotImplementedError

    def chunk_documents(self, docs) -> List[Chunk]:
        out = []
        for d in docs:
            meta = dict(d.metadata) if hasattr(d, "metadata") else {}
            if hasattr(d, "doc_id"):
                meta["doc_id"] = d.doc_id
            chunks = self.chunk(d.content, metadata=meta)
            for c in chunks:
                c.doc_id = getattr(d, "doc_id", None)
            out.extend(chunks)
        return out


class FixedSizeChunker(BaseChunker):
    def chunk(self, text, metadata=None):
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks = []
        start = 0
        idx = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            piece = text[start:end]
            if len(piece.strip()) >= self.config.min_chunk_size or start == 0:
                chunks.append(Chunk(text=piece, index=idx, metadata=dict(metadata or {})))
                idx += 1
            if end >= n:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return chunks


class RecursiveCharacterChunker(BaseChunker):
    def __init__(self, config=None, separators=None):
        super().__init__(config)
        self.separators = separators or self.config.separators

    def _split(self, text, separators):
        if not separators:
            return [text]
        sep = separators[0]
        rest = separators[1:]
        if sep == "":
            return [text[i:i + self.config.chunk_size] for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap)]
        parts = text.split(sep)
        chunks = []
        current = ""
        for part in parts:
            candidate = part if not current else current + sep + part
            if len(candidate) <= self.config.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.config.chunk_size:
                    chunks.extend(self._split(part, rest))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        # merge small chunks with overlap handling
        merged = []
        for c in chunks:
            if merged and len(merged[-1]) + len(c) + 1 <= self.config.chunk_size:
                merged[-1] = merged[-1] + sep + c if sep else merged[-1] + c
            else:
                merged.append(c)
        # apply overlap
        if self.config.chunk_overlap > 0 and len(merged) > 1:
            out = [merged[0]]
            for i in range(1, len(merged)):
                prev = merged[i - 1]
                tail = prev[-self.config.chunk_overlap:]
                out.append(tail + merged[i])
            merged = out
        return [m for m in merged if len(m.strip()) >= self.config.min_chunk_size or True]

    def chunk(self, text, metadata=None):
        pieces = self._split(text, self.separators)
        return [Chunk(text=p, index=i, metadata=dict(metadata or {})) for i, p in enumerate(pieces) if p.strip()]


class SentenceChunker(BaseChunker):
    def _sentences(self, text):
        # simple sentence splitting
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p.strip()]

    def chunk(self, text, metadata=None):
        sents = self._sentences(text)
        chunks = []
        current = []
        cur_len = 0
        idx = 0
        for s in sents:
            if cur_len + len(s) + 1 > self.config.chunk_size and current:
                chunks.append(Chunk(text=" ".join(current), index=idx, metadata=dict(metadata or {})))
                idx += 1
                # overlap by sentences
                if self.config.chunk_overlap > 0:
                    # keep last sentence(s) roughly
                    current = current[-1:]
                    cur_len = sum(len(x) for x in current)
                else:
                    current = []
                    cur_len = 0
            current.append(s)
            cur_len += len(s) + 1
        if current:
            chunks.append(Chunk(text=" ".join(current), index=idx, metadata=dict(metadata or {})))
        return [c for c in chunks if len(c.text.strip()) >= self.config.min_chunk_size or len(chunks) == 1]


class ParagraphChunker(BaseChunker):
    def chunk(self, text, metadata=None):
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks = []
        current = []
        cur_len = 0
        idx = 0
        for p in paras:
            if cur_len + len(p) + 2 > self.config.chunk_size and current:
                chunks.append(Chunk(text="\n\n".join(current), index=idx, metadata=dict(metadata or {})))
                idx += 1
                current = []
                cur_len = 0
            current.append(p)
            cur_len += len(p) + 2
        if current:
            chunks.append(Chunk(text="\n\n".join(current), index=idx, metadata=dict(metadata or {})))
        return chunks


class SlidingWindowChunker(BaseChunker):
    def chunk(self, text, metadata=None):
        size = self.config.chunk_size
        step = max(1, size - self.config.chunk_overlap)
        chunks = []
        idx = 0
        for start in range(0, len(text), step):
            piece = text[start:start + size]
            if not piece.strip():
                continue
            if len(piece.strip()) < self.config.min_chunk_size and start + size < len(text):
                continue
            chunks.append(Chunk(text=piece, index=idx, metadata=dict(metadata or {})))
            idx += 1
            if start + size >= len(text):
                break
        return chunks


class TokenChunker(BaseChunker):
    def __init__(self, config=None, encoding_name="cl100k_base"):
        super().__init__(config)
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.enc = None

    def chunk(self, text, metadata=None):
        if self.enc is None:
            # fallback to char-based
            return FixedSizeChunker(self.config).chunk(text, metadata)
        toks = self.enc.encode(text)
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks = []
        idx = 0
        start = 0
        n = len(toks)
        while start < n:
            end = min(start + size, n)
            piece = self.enc.decode(toks[start:end])
            chunks.append(Chunk(text=piece, index=idx, metadata=dict(metadata or {})))
            idx += 1
            if end >= n:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return chunks


class SemanticChunker(BaseChunker):
    """Embedding-based semantic chunking: split where adjacent sentence similarity drops."""
    def __init__(self, config=None, embeddings=None, breakpoint_percentile=70):
        super().__init__(config)
        self.embeddings = embeddings
        self.breakpoint_percentile = breakpoint_percentile

    def _sentences(self, text):
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p.strip()]

    def chunk(self, text, metadata=None):
        sents = self._sentences(text)
        if len(sents) <= 1:
            return [Chunk(text=text, index=0, metadata=dict(metadata or {}))]
        if self.embeddings is None:
            # fallback
            return SentenceChunker(self.config).chunk(text, metadata)
        vecs = self.embeddings.embed(sents)
        # cosine similarity between adjacent
        import numpy as np
        sims = []
        for i in range(len(vecs) - 1):
            a, b = vecs[i], vecs[i + 1]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            sims.append(float(np.dot(a, b) / denom))
        if not sims:
            return [Chunk(text=text, index=0, metadata=dict(metadata or {}))]
        threshold = float(np.percentile(sims, 100 - self.breakpoint_percentile))
        # breakpoints where sim < threshold
        groups = []
        current = [sents[0]]
        for i, s in enumerate(sents[1:], start=1):
            if sims[i - 1] < threshold:
                groups.append(current)
                current = [s]
            else:
                current.append(s)
        if current:
            groups.append(current)
        chunks = []
        # enforce max size by merging/splitting
        for i, g in enumerate(groups):
            t = " ".join(g)
            if len(t) > self.config.chunk_size:
                # split with sentence chunker
                sub = SentenceChunker(self.config).chunk(t, metadata)
                chunks.extend(sub)
            else:
                chunks.append(Chunk(text=t, index=i, metadata=dict(metadata or {})))
        # reindex
        for i, c in enumerate(chunks):
            c.index = i
            c.metadata = dict(metadata or {})
        return [c for c in chunks if c.text.strip()]


class ParentChildChunker(BaseChunker):
    """Creates small child chunks for retrieval and keeps parent chunks for context."""
    def __init__(self, config=None, child_size=None, parent_size=None):
        super().__init__(config)
        self.child_size = child_size or max(128, config.chunk_size // 4 if config else 128)
        self.parent_size = parent_size or (config.chunk_size if config else 512)

    def chunk(self, text, metadata=None):
        # parent chunks
        pcfg = ChunkingConfig(chunk_size=self.parent_size, chunk_overlap=self.config.chunk_overlap, min_chunk_size=self.config.min_chunk_size, separators=self.config.separators)
        parents = RecursiveCharacterChunker(pcfg).chunk(text, metadata)
        children = []
        for pi, p in enumerate(parents):
            ccfg = ChunkingConfig(chunk_size=self.child_size, chunk_overlap=max(0, self.child_size // 8), min_chunk_size=10, separators=self.config.separators)
            cs = RecursiveCharacterChunker(ccfg).chunk(p.text, metadata)
            for c in cs:
                c.parent_id = p.chunk_id
                c.metadata = dict(metadata or {})
                c.metadata["parent_chunk_id"] = p.chunk_id
                children.append(c)
        # store parents in metadata of first? We return children; parents accessible via chunker.parents (sidecar)
        self.last_parents = parents
        return children


class CodeChunker(BaseChunker):
    """Chunk code by top-level definitions."""
    def chunk(self, text, metadata=None):
        # split on def/class boundaries
        lines = text.splitlines()
        chunks = []
        current = []
        idx = 0
        for line in lines:
            if re.match(r"^(def |class |function |export |const |let |var )", line) and current:
                chunks.append(Chunk(text="\n".join(current), index=idx, metadata=dict(metadata or {})))
                idx += 1
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append(Chunk(text="\n".join(current), index=idx, metadata=dict(metadata or {})))
        # fallback if nothing split
        if len(chunks) <= 1 and len(text) > self.config.chunk_size:
            return FixedSizeChunker(self.config).chunk(text, metadata)
        return chunks


CHUNKERS = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveCharacterChunker,
    "sentence": SentenceChunker,
    "paragraph": ParagraphChunker,
    "sliding": SlidingWindowChunker,
    "token": TokenChunker,
    "semantic": SemanticChunker,
    "parent_child": ParentChildChunker,
    "code": CodeChunker,
}


def get_chunker(strategy: str, config: Optional[ChunkingConfig] = None, **kwargs) -> BaseChunker:
    cls = CHUNKERS.get(strategy, RecursiveCharacterChunker)
    return cls(config=config, **kwargs)
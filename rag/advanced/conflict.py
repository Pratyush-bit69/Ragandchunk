"""Detect and resolve conflicting information across retrieved chunks."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from rag.chunking import Chunk


NEGATION = ["not", "no", "never", "cannot", "can't", "won't", "isn't", "aren't", "does not", "do not", "did not"]
CONTRAST = ["however", "but", "although", "whereas", "on the other hand", "in contrast", "conversely"]


class ConflictDetector:
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold

    def _sentences(self, text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _token_set(self, s):
        return set(re.findall(r"\w+", s.lower()))

    def detect(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Find pairs of chunks that discuss the same topic but assert conflicting facts."""
        conflicts = []
        # group sentences by shared subject keywords
        sents = []
        for ci, c in enumerate(chunks):
            for s in self._sentences(c.text):
                sents.append((ci, c.chunk_id, s))

        for i in range(len(sents)):
            for j in range(i + 1, len(sents)):
                ci, idi, si = sents[i]
                cj, idj, sj = sents[j]
                if idi == idj:
                    continue
                ti, tj = self._token_set(si), self._token_set(sj)
                if not ti or not tj:
                    continue
                overlap = len(ti & tj) / max(1, min(len(ti), len(tj)))
                if overlap < self.similarity_threshold:
                    continue
                # check negation/contrast asymmetry
                neg_i = any(n in si.lower() for n in NEGATION)
                neg_j = any(n in sj.lower() for n in NEGATION)
                contrast = any(cw in (si + " " + sj).lower() for cw in CONTRAST)
                if neg_i != neg_j or contrast:
                    conflicts.append({
                        "chunk_a": idi,
                        "chunk_b": idj,
                        "sentence_a": si,
                        "sentence_b": sj,
                        "overlap": overlap,
                        "type": "negation" if neg_i != neg_j else "contrast",
                    })
        # dedup by pair
        seen = set()
        out = []
        for cf in conflicts:
            key = tuple(sorted([cf["chunk_a"], cf["chunk_b"]]))
            if key in seen:
                continue
            seen.add(key)
            out.append(cf)
        return out

    def summarize(self, chunks, conflicts):
        if not conflicts:
            return "No conflicts detected among retrieved sources."
        lines = [f"Detected {len(conflicts)} potential conflict(s):"]
        for cf in conflicts:
            lines.append(f"- [{cf['type']}] {cf['sentence_a'][:120]}  <>  {cf['sentence_b'][:120]}")
        return "\n".join(lines)


class ProvenanceTracker:
    """Track which source supports each claim (sentence-level)."""
    def __init__(self):
        pass

    def attribute(self, answer: str, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        out = []
        for sent in re.split(r"(?<=[.!?])\s+", answer):
            if not sent.strip():
                continue
            best = None
            best_score = 0.0
            stoks = set(re.findall(r"\w+", sent.lower()))
            for c in chunks:
                ctoks = set(re.findall(r"\w+", c.text.lower()))
                if not ctoks:
                    continue
                score = len(stoks & ctoks) / max(1, len(stoks))
                if score > best_score:
                    best_score = score
                    best = c
            out.append({
                "sentence": sent.strip(),
                "source_chunk": best.chunk_id if best else None,
                "support_score": round(best_score, 3),
                "supported": best_score >= 0.4,
            })
        return out
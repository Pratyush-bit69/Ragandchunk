"""Self-consistency RAG: sample multiple retrieval/generation paths and vote."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

from rag.chunking import Chunk


class SelfConsistencyRAG:
    def __init__(self, pipeline, n_samples=5, top_k=5):
        self.pipeline = pipeline
        self.n = n_samples
        self.top_k = top_k

    def _sample_contexts(self, retrieved, frac=0.8):
        import random
        random.seed()
        k = max(1, int(len(retrieved) * frac))
        return [c for c, _ in (retrieved[:k] if len(retrieved) <= k else __import__("random").sample(retrieved, k))]

    def run(self, query):
        answers = []
        all_contexts = []
        retrieved = self.pipeline.retrieve(query, top_k=self.top_k * 2)
        for i in range(self.n):
            ctxs = self._sample_contexts(retrieved)
            if not ctxs:
                ctxs = [c for c, _ in retrieved[:self.top_k]]
            gen = self.pipeline.generator.generate(query, ctxs)
            answers.append(gen.get("answer", ""))
            all_contexts.append(ctxs)
        # vote at sentence level: pick sentences appearing in majority
        sent_counter = Counter()
        sent_to_ctx = {}
        for ans in answers:
            for sent in re.split(r"(?<=[.!?])\s+", ans):
                s = sent.strip()
                if s:
                    sent_counter[s] += 1
                    sent_to_ctx[s] = None
        threshold = self.n // 2 + 1
        chosen = [s for s, cnt in sent_counter.items() if cnt >= threshold]
        if not chosen:
            # fallback: most common full answer
            if answers:
                chosen = [Counter(answers).most_common(1)[0][0]]
        final_answer = " ".join(chosen)
        # contexts = union of contexts from answers that contributed
        merged_ids = set()
        merged_ctxs = []
        for ctxs in all_contexts:
            for c in ctxs:
                if c.chunk_id not in merged_ids:
                    merged_ids.add(c.chunk_id)
                    merged_ctxs.append(c)
        return {
            "query": query,
            "answer": final_answer,
            "sample_answers": answers,
            "contexts": [c.text for c in merged_ctxs[:self.top_k]],
            "chunk_ids": [c.chunk_id for c in merged_ctxs[:self.top_k]],
            "agreement": round(len(chosen) / max(1, len(sent_counter)), 3),
        }
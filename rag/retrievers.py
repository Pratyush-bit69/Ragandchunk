"""Retrievers: dense, sparse, hybrid, rerank, parent-doc, multi-query."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rag.chunking import Chunk
from rag.embeddings import BaseEmbeddings
from rag.vectorstore import InMemoryVectorStore


class BaseRetriever:
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        raise NotImplementedError


class DenseRetriever(BaseRetriever):
    def __init__(self, store, embeddings: BaseEmbeddings):
        self.store = store
        self.embeddings = embeddings

    def retrieve(self, query, k=5, filter_fn=None):
        qv = self.embeddings.embed_query(query)
        res = self.store.search(qv, k=k, filter_fn=filter_fn)
        return [(r[0], r[1]) for r in res]


class SparseRetriever(BaseRetriever):
    """BM25-based sparse retriever."""
    def __init__(self, chunks: List[Chunk]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.tokenized = [self._tok(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    @staticmethod
    def _tok(text):
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query, k=5, filter_fn=None):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(self._tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if filter_fn is not None:
                # need metadata; chunks store metadata
                pass
            out.append((self.chunks[i], float(scores[i])))
            if len(out) >= k:
                break
        if filter_fn is not None:
            out = [r for r in out if filter_fn(r[0].metadata)]
        return out[:k]


class HybridRetriever(BaseRetriever):
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, alpha: float = 0.5):
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(self, query, k=5, filter_fn=None):
        # fetch more from each
        fetch = max(k * 4, 20)
        dres = self.dense.retrieve(query, k=fetch, filter_fn=filter_fn)
        sres = self.sparse.retrieve(query, k=fetch, filter_fn=filter_fn)
        # normalize scores
        def norm(res):
            if not res:
                return res
            scs = [r[1] for r in res]
            mn, mx = min(scs), max(scs)
            rng = mx - mn if mx != mn else 1.0
            return [(c, (s - mn) / rng) for c, s in res]
        dres = norm(dres)
        sres = norm(sres)
        # rrf-style fusion with alpha weighting
        scores = {}
        chunkmap = {}
        for c, s in dres:
            scores[c.chunk_id] = scores.get(c.chunk_id, 0) + self.alpha * s
            chunkmap[c.chunk_id] = c
        for c, s in sres:
            scores[c.chunk_id] = scores.get(c.chunk_id, 0) + (1 - self.alpha) * s
            chunkmap[c.chunk_id] = c
        order = sorted(scores, key=lambda x: -scores[x])
        return [(chunkmap[cid], scores[cid]) for cid in order[:k]]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query, results, top_k=None):
        if not results:
            return results
        pairs = [(query, r[0].text) for r in results]
        scores = self.model.predict(pairs)
        order = sorted(range(len(results)), key=lambda i: -float(scores[i]))
        out = [(results[i][0], float(scores[i])) for i in order]
        if top_k:
            out = out[:top_k]
        return out


class RerankingRetriever(BaseRetriever):
    def __init__(self, base: BaseRetriever, reranker: CrossEncoderReranker, top_k=None):
        self.base = base
        self.reranker = reranker
        self.top_k = top_k

    def retrieve(self, query, k=5, filter_fn=None):
        fetch = max(k * 4, 20)
        res = self.base.retrieve(query, k=fetch, filter_fn=filter_fn)
        return self.reranker.rerank(query, res, top_k=k)


class MultiQueryRetriever(BaseRetriever):
    def __init__(self, base: BaseRetriever, llm=None, n=3):
        self.base = base
        self.llm = llm
        self.n = n

    def _variants(self, query):
        # simple rule-based variants
        variants = [query]
        # keyword extraction-ish
        words = query.split()
        if len(words) > 3:
            variants.append(" ".join(words[:len(words)//2]))
            variants.append(" ".join(words[len(words)//2:]))
        return list(dict.fromkeys(variants))[:self.n]

    def retrieve(self, query, k=5, filter_fn=None):
        seen = set()
        merged = []
        for q in self._variants(query):
            for c, s in self.base.retrieve(q, k=k, filter_fn=filter_fn):
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                merged.append((c, s))
        # sort by score desc
        merged.sort(key=lambda x: -x[1])
        return merged[:k]


class ParentDocumentRetriever(BaseRetriever):
    def __init__(self, child_retriever: BaseRetriever, parent_store: Dict[str, Chunk]):
        self.child_retriever = child_retriever
        self.parent_store = parent_store

    def retrieve(self, query, k=5, filter_fn=None):
        child_res = self.child_retriever.retrieve(query, k=k * 2, filter_fn=filter_fn)
        seen_parents = set()
        out = []
        for c, s in child_res:
            pid = c.metadata.get("parent_chunk_id") or c.parent_id
            if pid and pid in self.parent_store:
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)
                out.append((self.parent_store[pid], s))
            else:
                out.append((c, s))
            if len(out) >= k:
                break
        return out[:k]


class MMRRetriever(BaseRetriever):
    def __init__(self, base: BaseRetriever, embeddings: BaseEmbeddings, lambda_mult: float = 0.5):
        self.base = base
        self.embeddings = embeddings
        self.lambda_mult = lambda_mult

    def retrieve(self, query, k=5, filter_fn=None):
        fetch = max(k * 4, 20)
        res = self.base.retrieve(query, k=fetch, filter_fn=filter_fn)
        if not res:
            return res
        qv = self.embeddings.embed_query(query)
        selected = []
        selected_vecs = []
        candidates = list(res)
        cand_vecs = [self.embeddings.embed([c.text])[0] for c, _ in candidates]
        # first pick highest score
        # greedy mmr
        remaining = list(range(len(candidates)))
        while len(selected) < k and remaining:
            best_i = None
            best_score = -1e9
            for i in remaining:
                sim_to_query = float(np.dot(cand_vecs[i], qv) / ((np.linalg.norm(cand_vecs[i]) * np.linalg.norm(qv)) or 1.0))
                if selected_vecs:
                    max_sim = max(float(np.dot(cand_vecs[i], sv) / ((np.linalg.norm(cand_vecs[i]) * np.linalg.norm(sv)) or 1.0)) for sv in selected_vecs)
                else:
                    max_sim = 0.0
                mmr = self.lambda_mult * sim_to_query - (1 - self.lambda_mult) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_i = i
            selected.append(candidates[best_i])
            selected_vecs.append(cand_vecs[best_i])
            remaining.remove(best_i)
        return selected
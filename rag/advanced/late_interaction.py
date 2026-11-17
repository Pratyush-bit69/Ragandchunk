"""Late-interaction (ColBERT-style) retriever using MaxSim over token embeddings."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rag.chunking import Chunk
from rag.embeddings import BaseEmbeddings


class LateInteractionRetriever:
    """
    Approximates ColBERT: encode query and document into token-level vectors,
    score = sum over query tokens of max over doc tokens cosine.
    Uses a sentence-transformer with token embeddings (model.encode with output_value).
    Falls back to word-level hashed vectors if model lacks token output.
    """
    def __init__(self, embeddings: BaseEmbeddings, chunks: List[Chunk]):
        self.embeddings = embeddings
        self.chunks = chunks
        # Precompute per-chunk token vectors
        self.doc_token_vecs: List[np.ndarray] = []
        for c in chunks:
            self.doc_token_vecs.append(self._tokenize_encode(c.text))

    def _tokenize_encode(self, text: str) -> np.ndarray:
        # Try sentence-transformers token embeddings
        st = getattr(self.embeddings, "model", None)
        if st is not None and hasattr(st, "encode"):
            try:
                # sentence-transformers supports output_value='token_embeddings'
                out = st.encode(text, output_value="token_embeddings", convert_to_numpy=True)
                arr = np.asarray(out)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    # normalize
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    return (arr / norms).astype(np.float32)
            except Exception:
                pass
        # Fallback: one vector per word using hashing embeddings
        words = text.split()
        if not words:
            return np.zeros((1, 64), dtype=np.float32)
        # use hashing vectorizer per word
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
            hv = HashingVectorizer(n_features=64, alternate_sign=False, norm="l2")
            mat = hv.transform(words).toarray().astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms
        except Exception:
            return np.zeros((1, 64), dtype=np.float32)

    def _query_tokens(self, query: str) -> np.ndarray:
        return self._tokenize_encode(query)

    def score(self, qvecs: np.ndarray, dvecs: np.ndarray) -> float:
        if qvecs.shape[0] == 0 or dvecs.shape[0] == 0:
            return 0.0
        # MaxSim: for each query token, max cosine over doc tokens; sum
        sim = qvecs @ dvecs.T  # (Q, D)
        maxsim = sim.max(axis=1)
        return float(maxsim.sum())

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        qvecs = self._query_tokens(query)
        scored = []
        for c, dvecs in zip(self.chunks, self.doc_token_vecs):
            scored.append((c, self.score(qvecs, dvecs)))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
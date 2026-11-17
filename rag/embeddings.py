"""Embedding providers."""
from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Sequence

import numpy as np

from rag.config import EmbeddingConfig


class BaseEmbeddings:
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self.embed(["probe"]).shape[1]


class SentenceTransformerEmbeddings(BaseEmbeddings):
    def __init__(self, model: str = "all-MiniLM-L6-v2", batch_size: int = 32, normalize: bool = True, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model, device=device)

    def embed(self, texts):
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=self.normalize, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class TFIDFEmbeddings(BaseEmbeddings):
    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self._fitted = False
        self._corpus: List[str] = []

    def fit(self, corpus: Sequence[str]):
        self.vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, texts):
        if not self._fitted:
            # fit on provided texts as fallback
            self.fit(list(texts))
        mat = self.vectorizer.transform(list(texts))
        return mat.toarray().astype(np.float32)


class HashingEmbeddings(BaseEmbeddings):
    def __init__(self, dim: int = 384, ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vectorizer = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2", ngram_range=ngram_range)
        self._dim = dim

    def embed(self, texts):
        mat = self.vectorizer.transform(list(texts))
        return mat.toarray().astype(np.float32)

    @property
    def dim(self):
        return self._dim


class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def embed(self, texts):
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        vecs = [d.embedding for d in resp.data]
        return np.asarray(vecs, dtype=np.float32)


class CachedEmbeddings(BaseEmbeddings):
    def __init__(self, base: BaseEmbeddings, cache_path: Optional[str] = None):
        self.base = base
        self.cache = {}
        self.cache_path = cache_path
        if cache_path and os.path.exists(cache_path):
            import json
            try:
                with open(cache_path) as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _key(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed(self, texts):
        out = [None] * len(texts)
        to_compute = []
        idxs = []
        for i, t in enumerate(texts):
            k = self._key(t)
            if k in self.cache:
                out[i] = np.asarray(self.cache[k], dtype=np.float32)
            else:
                to_compute.append(t)
                idxs.append(i)
        if to_compute:
            vecs = self.base.embed(to_compute)
            for j, i in enumerate(idxs):
                k = self._key(texts[i])
                self.cache[k] = vecs[j].tolist()
                out[i] = vecs[j]
        if self.cache_path:
            import json
            try:
                with open(self.cache_path, "w") as f:
                    json.dump(self.cache, f)
            except Exception:
                pass
        return np.asarray(out, dtype=np.float32)


def get_embeddings(config: EmbeddingConfig) -> BaseEmbeddings:
    prov = config.provider
    if prov == "sentence-transformers":
        return SentenceTransformerEmbeddings(config.model, config.batch_size, config.normalize)
    if prov == "tfidf":
        return TFIDFEmbeddings()
    if prov == "hashing":
        return HashingEmbeddings()
    if prov == "openai":
        return OpenAIEmbeddings(config.model)
    raise ValueError(f"Unknown embedding provider: {prov}")
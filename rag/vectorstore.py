"""Vector stores."""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rag.chunking import Chunk


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class InMemoryVectorStore:
    def __init__(self):
        self.ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        for c, e in zip(chunks, embeddings):
            self.ids.append(c.chunk_id)
            self.chunks.append(c)
            self.metadatas.append(dict(c.metadata))
        arr = np.asarray(embeddings, dtype=np.float32)
        if self.vectors is None:
            self.vectors = arr
        else:
            self.vectors = np.vstack([self.vectors, arr])

    def search(self, query_vec, k=5, filter_fn=None):
        if self.vectors is None or len(self.ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        # cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1) * (np.linalg.norm(q) or 1.0)
        sims = (self.vectors @ q) / np.where(norms == 0, 1.0, norms)
        order = np.argsort(-sims)
        results = []
        for idx in order:
            if filter_fn is not None and not filter_fn(self.metadatas[idx]):
                continue
            results.append((self.chunks[idx], float(sims[idx]), self.metadatas[idx]))
            if len(results) >= k:
                break
        return results

    def delete(self, ids):
        keep = [i for i, cid in enumerate(self.ids) if cid not in set(ids)]
        self.ids = [self.ids[i] for i in keep]
        self.chunks = [self.chunks[i] for i in keep]
        self.metadatas = [self.metadatas[i] for i in keep]
        if self.vectors is not None:
            self.vectors = self.vectors[keep] if keep else None

    def persist(self, path):
        os.makedirs(path, exist_ok=True)
        data = {
            "ids": self.ids,
            "chunks": [c.__dict__ for c in self.chunks],
            "metadatas": self.metadatas,
        }
        with open(os.path.join(path, "store.json"), "w") as f:
            json.dump(data, f)
        if self.vectors is not None:
            np.save(os.path.join(path, "vectors.npy"), self.vectors)

    @classmethod
    def load(cls, path):
        store = cls()
        with open(os.path.join(path, "store.json")) as f:
            data = json.load(f)
        from rag.chunking import Chunk
        store.ids = data["ids"]
        store.chunks = [Chunk(**c) for c in data["chunks"]]
        store.metadatas = data["metadatas"]
        vp = os.path.join(path, "vectors.npy")
        if os.path.exists(vp):
            store.vectors = np.load(vp)
        return store

    def __len__(self):
        return len(self.ids)


class ChromaVectorStore:
    def __init__(self, collection_name="rag_collection", persist_dir=None):
        import chromadb
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, chunks, embeddings):
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [dict(c.metadata) for c in chunks]
        self.collection.add(ids=ids, documents=docs, embeddings=[e.tolist() for e in embeddings], metadatas=metas)

    def search(self, query_vec, k=5, filter_fn=None):
        where = None
        res = self.collection.query(query_embeddings=[query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)], n_results=k)
        out = []
        for i in range(len(res["ids"][0])):
            cid = res["ids"][0][i]
            doc = res["documents"][0][i]
            dist = res["distances"][0][i]
            meta = res["metadatas"][0][i] or {}
            # reconstruct chunk
            from rag.chunking import Chunk
            chunk = Chunk(text=doc, chunk_id=cid, metadata=meta)
            out.append((chunk, 1.0 - float(dist), meta))
        if filter_fn:
            out = [r for r in out if filter_fn(r[2])]
        return out

    def __len__(self):
        return self.collection.count()


def get_vectorstore(backend="memory", **kwargs):
    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "chroma":
        return ChromaVectorStore(**kwargs)
    raise ValueError(f"Unknown vectorstore backend: {backend}")
"""End-to-end RAG pipeline."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from rag.config import RAGConfig
from rag.loaders import Document, load_document
from rag.chunking import Chunk, get_chunker, ChunkingConfig
from rag.embeddings import get_embeddings, BaseEmbeddings
from rag.vectorstore import get_vectorstore
from rag.retrievers import (
    DenseRetriever, SparseRetriever, HybridRetriever, RerankingRetriever,
    MultiQueryRetriever, ParentDocumentRetriever, MMRRetriever,
)
from rag.graph import KnowledgeGraph, GraphBuilder, GraphRetriever, GraphRAGPipeline, RuleBasedEntityExtractor
from rag.generation import get_generator


class RAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.embeddings: BaseEmbeddings = get_embeddings(self.config.embedding)
        ck = {}
        if self.config.chunking.strategy == "semantic":
            ck["embeddings"] = self.embeddings
        self.chunker = get_chunker(self.config.chunking.strategy, self.config.chunking, **ck)
        self.vectorstore = get_vectorstore("memory")
        self.sparse_retriever: Optional[SparseRetriever] = None
        self.dense_retriever: Optional[DenseRetriever] = None
        self.retriever = None
        self.generator = get_generator(self.config.generation)
        self.kg: Optional[KnowledgeGraph] = None
        self.graph_retriever: Optional[GraphRetriever] = None
        self.chunks: List[Chunk] = []
        self.documents: List[Document] = []
        self._parent_store: Dict[str, Chunk] = {}

    # ---------- ingestion ----------
    def ingest(self, source, chunking_strategy: Optional[str] = None) -> List[Chunk]:
        if isinstance(source, (str, os.PathLike)):
            docs = load_document(source)
        elif isinstance(source, list):
            docs = source
        else:
            docs = [source]
        self.documents.extend(docs)
        if chunking_strategy:
            kwargs = {}
            if chunking_strategy == "semantic":
                kwargs["embeddings"] = self.embeddings
            self.chunker = get_chunker(chunking_strategy, self.config.chunking, **kwargs)
        chunks = self.chunker.chunk_documents(docs)
        # parent-child handling
        if self.config.chunking.strategy == "parent_child" or chunking_strategy == "parent_child":
            parents = getattr(self.chunker, "last_parents", [])
            for p in parents:
                self._parent_store[p.chunk_id] = p
        self.chunks.extend(chunks)
        # embed
        texts = [c.text for c in chunks]
        if texts:
            if hasattr(self.embeddings, "fit") and not getattr(self.embeddings, "_fitted", True):
                all_texts = [c.text for c in self.chunks]
                self.embeddings.fit(all_texts)
            vecs = self.embeddings.embed(texts)
            self.vectorstore.add(chunks, vecs)
        # sparse retriever rebuild
        self.sparse_retriever = SparseRetriever(self.chunks)
        # dense retriever
        self.dense_retriever = DenseRetriever(self.vectorstore, self.embeddings)
        # graph
        if self.config.graph.enabled:
            builder = GraphBuilder()
            kg = builder.build_from_chunks(chunks)
            if self.kg is None:
                self.kg = kg
            else:
                # merge
                for n, e in kg.entities.items():
                    self.kg.add_entity(e)
                for r in kg.relations:
                    self.kg.add_relation(r)
                for cid, ents in kg.chunk_entities.items():
                    self.kg.chunk_entities[cid].extend(ents)
            self.graph_retriever = GraphRetriever(self.kg, self.chunks, self.embeddings, hops=self.config.graph.max_hops)
        self._build_retriever()
        return chunks

    def ingest_text(self, text: str, metadata=None) -> List[Chunk]:
        from rag.loaders import Document
        doc = Document(content=text, metadata=metadata or {})
        return self.ingest([doc])

    def _build_retriever(self):
        mode = self.config.retrieval.mode
        if mode == "dense":
            base = self.dense_retriever
        elif mode == "sparse":
            base = self.sparse_retriever
        elif mode == "hybrid":
            base = HybridRetriever(self.dense_retriever, self.sparse_retriever, self.config.retrieval.hybrid_alpha)
        else:
            base = self.dense_retriever
        # graph rag wrapper
        if self.config.graph.enabled and self.graph_retriever is not None:
            base = GraphRAGPipeline(base, self.graph_retriever, alpha=0.5)
        # parent doc
        if self._parent_store:
            base = ParentDocumentRetriever(base, self._parent_store)
        # mmr
        if self.config.retrieval.use_mmr:
            base = MMRRetriever(base, self.embeddings, self.config.retrieval.mmr_lambda)
        # rerank
        if self.config.retrieval.rerank:
            try:
                from rag.retrievers import CrossEncoderReranker, RerankingRetriever
                rr = CrossEncoderReranker(self.config.retrieval.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
                base = RerankingRetriever(base, rr, top_k=self.config.retrieval.top_k)
            except Exception as e:
                print(f"[pipeline] rerank unavailable: {e}")
        # multi-query
        # optional: wrap
        self.retriever = base

    # ---------- retrieval ----------
    def retrieve(self, query: str, top_k: Optional[int] = None, filter_fn=None) -> List[Tuple[Chunk, float]]:
        k = top_k or self.config.retrieval.top_k
        if self.retriever is None:
            return []
        # fetch_k
        fetch = max(k, self.config.retrieval.fetch_k)
        # Most retrievers accept k; graphrag accepts k
        res = self.retriever.retrieve(query, k=fetch, filter_fn=filter_fn)
        return res[:k]

    # ---------- generation ----------
    def query(self, query: str, top_k: Optional[int] = None, generate: bool = True) -> Dict[str, Any]:
        retrieved = self.retrieve(query, top_k=top_k)
        contexts = [c for c, _ in retrieved]
        out = {
            "query": query,
            "retrieved": retrieved,
            "contexts": [c.text for c in contexts],
            "chunk_ids": [c.chunk_id for c in contexts],
        }
        if generate:
            gen = self.generator.generate(query, contexts)
            out.update(gen)
        return out

    # ---------- persistence ----------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.vectorstore.persist(path)
        import json
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump([c.__dict__ for c in self.chunks], f)
        with open(os.path.join(path, "config.json"), "w") as f:
            from dataclasses import asdict
            json.dump(asdict(self.config), f, default=str)

    @classmethod
    def load(cls, path: str):
        from rag.vectorstore import InMemoryVectorStore
        pipe = cls()
        pipe.vectorstore = InMemoryVectorStore.load(path)
        import json
        with open(os.path.join(path, "chunks.json")) as f:
            from rag.chunking import Chunk
            pipe.chunks = [Chunk(**c) for c in json.load(f)]
        pipe.sparse_retriever = SparseRetriever(pipe.chunks)
        pipe.dense_retriever = DenseRetriever(pipe.vectorstore, pipe.embeddings)
        pipe._build_retriever()
        return pipe

    # ---------- graph helpers ----------
    def graph_stats(self):
        return self.kg.stats() if self.kg else {}

    def visualize_graph(self, path: Optional[str] = None, max_nodes=50):
        if self.kg is None:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        g = self.kg.graph
        if g.number_of_nodes() > max_nodes:
            # take top-degree nodes
            nodes = sorted(g.degree, key=lambda x: -x[1])[:max_nodes]
            nodes = [n for n, _ in nodes]
            g = g.subgraph(nodes)
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(g, seed=42)
        nx.draw(g, pos, with_labels=True, node_size=300, font_size=7, arrows=True)
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150)
            plt.close()
            return path
        else:
            out = "/tmp/opencode/graph.png"
            plt.savefig(out, dpi=150)
            plt.close()
            return out
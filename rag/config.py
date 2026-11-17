"""Configuration for the RAG system."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers"  # sentence-transformers | tfidf | hashing | openai
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True
    cache_dir: Optional[str] = None


@dataclass
class ChunkingConfig:
    strategy: str = "recursive"  # fixed | recursive | sentence | semantic | sliding | parent_child | token
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    respect_sentences: bool = True


@dataclass
class RetrievalConfig:
    top_k: int = 5
    fetch_k: int = 20
    mode: str = "hybrid"  # dense | sparse | hybrid
    hybrid_alpha: float = 0.5  # weight for dense in hybrid
    rerank: bool = False
    rerank_model: Optional[str] = None
    use_mmr: bool = False
    mmr_lambda: float = 0.5


@dataclass
class GenerationConfig:
    provider: str = "extractive"  # extractive | openai | anthropic
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 512
    system_prompt: Optional[str] = None


@dataclass
class GraphConfig:
    enabled: bool = True
    max_hops: int = 2
    community_detection: bool = True
    entity_extraction: str = "rule"  # rule | llm


@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "precision_at_k", "recall_at_k", "mrr", "ndcg", "faithfulness",
        "answer_relevance", "context_precision", "context_recall",
    ])
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    persist_dir: Optional[str] = None
    collection_name: str = "rag_collection"

    @classmethod
    def from_env(cls) -> "RAGConfig":
        cfg = cls()
        cfg.embedding.provider = os.getenv("RAG_EMBEDDING_PROVIDER", cfg.embedding.provider)
        cfg.embedding.model = os.getenv("RAG_EMBEDDING_MODEL", cfg.embedding.model)
        cfg.generation.provider = os.getenv("RAG_GENERATION_PROVIDER", cfg.generation.provider)
        cfg.generation.model = os.getenv("RAG_GENERATION_MODEL", cfg.generation.model)
        cfg.persist_dir = os.getenv("RAG_PERSIST_DIR", cfg.persist_dir)
        return cfg
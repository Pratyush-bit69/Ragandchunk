"""Knowledge graph and GraphRAG."""
from __future__ import annotations

import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from rag.chunking import Chunk
from rag.embeddings import BaseEmbeddings


@dataclass
class Entity:
    name: str
    type: str = "ENTITY"
    description: str = ""
    source_chunks: List[str] = field(default_factory=list)


@dataclass
class Relation:
    source: str
    target: str
    relation: str = "related_to"
    weight: float = 1.0
    source_chunks: List[str] = field(default_factory=list)


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.chunk_entities: Dict[str, List[str]] = defaultdict(list)  # chunk_id -> entity names

    def add_entity(self, entity: Entity):
        self.entities[entity.name] = entity
        if not self.graph.has_node(entity.name):
            self.graph.add_node(entity.name, type=entity.type, description=entity.description)

    def add_relation(self, rel: Relation):
        self.relations.append(rel)
        if not self.graph.has_node(rel.source):
            self.graph.add_node(rel.source, type="ENTITY")
        if not self.graph.has_node(rel.target):
            self.graph.add_node(rel.target, type="ENTITY")
        self.graph.add_edge(rel.source, rel.target, relation=rel.relation, weight=rel.weight)

    def neighbors(self, entity: str, hops: int = 1) -> Set[str]:
        if entity not in self.graph:
            return set()
        # undirected traversal
        ug = self.graph.to_undirected()
        nodes = set()
        frontier = {entity}
        visited = {entity}
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                for nb in ug.neighbors(n):
                    if nb not in visited:
                        visited.add(nb)
                        nxt.add(nb)
            nodes |= nxt
            frontier = nxt
            if not frontier:
                break
        return nodes

    def subgraph_for_entities(self, entities: List[str], hops: int = 1) -> nx.Graph:
        nodes = set(entities)
        for e in entities:
            nodes |= self.neighbors(e, hops=hops)
        return self.graph.subgraph(nodes).copy()

    def to_text(self, nodes: Optional[Set[str]] = None) -> str:
        lines = []
        if nodes is None:
            nodes = set(self.graph.nodes)
        for u, v, data in self.graph.edges(data=True):
            if u in nodes and v in nodes:
                lines.append(f"{u} --[{data.get('relation','related')}]--> {v}")
        return "\n".join(lines)

    def communities(self):
        ug = self.graph.to_undirected()
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(ug))
            return [list(c) for c in comms]
        except Exception:
            # fallback: connected components
            return [list(c) for c in nx.connected_components(ug)]

    def stats(self):
        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relations": self.graph.number_of_edges(),
            "num_components": nx.number_weakly_connected_components(self.graph),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0.0,
        }


class RuleBasedEntityExtractor:
    """Extract entities and relations using patterns (no LLM)."""
    ENTITY_PATTERNS = [
        (r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", "PROPER"),
    ]
    STOPWORDS = {"The", "This", "That", "These", "Those", "A", "An", "In", "On", "At", "For", "With", "And", "Or", "But"}

    def extract_entities(self, text: str) -> List[Entity]:
        ents = set()
        for m in re.finditer(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b", text):
            name = m.group(1).strip()
            if name in self.STOPWORDS:
                continue
            if len(name) < 3:
                continue
            ents.add(name)
        # also quoted terms
        for m in re.finditer(r'"([^"]{3,40})"', text):
            ents.add(m.group(1).strip())
        # numbers as entities? skip
        return [Entity(name=e, type="CONCEPT") for e in ents]

    def extract_relations(self, text: str, entities: List[str]) -> List[Relation]:
        rels = []
        sents = re.split(r"(?<=[.!?])\s+", text)
        for s in sents:
            found = [e for e in entities if e in s]
            # co-occurrence relation
            for i in range(len(found)):
                for j in range(i + 1, len(found)):
                    rels.append(Relation(source=found[i], target=found[j], relation="co_occurs", weight=1.0))
        return rels


class LLMEntityExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, text: str):
        prompt = (
            "Extract entities and relations from the text. "
            "Return JSON with keys 'entities' (list of {name,type}) and 'relations' (list of {source,target,relation}).\n\n"
            f"Text:\n{text}\n\nJSON:"
        )
        resp = self.llm.generate(prompt)
        import json
        try:
            data = json.loads(resp)
        except Exception:
            # try to find json
            m = re.search(r"\{.*\}", resp, re.S)
            data = json.loads(m.group(0)) if m else {"entities": [], "relations": []}
        entities = [Entity(name=e.get("name"), type=e.get("type", "ENTITY")) for e in data.get("entities", [])]
        relations = [Relation(source=r.get("source"), target=r.get("target"), relation=r.get("relation", "related_to")) for r in data.get("relations", [])]
        return entities, relations


class GraphBuilder:
    def __init__(self, extractor=None):
        self.extractor = extractor or RuleBasedEntityExtractor()

    def build_from_chunks(self, chunks: List[Chunk], embeddings: Optional[BaseEmbeddings] = None) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        for c in chunks:
            if isinstance(self.extractor, RuleBasedEntityExtractor):
                ents = self.extractor.extract_entities(c.text)
                rels = self.extractor.extract_relations(c.text, [e.name for e in ents])
            else:
                ents, rels = self.extractor.extract(c.text)
            for e in ents:
                e.source_chunks.append(c.chunk_id)
                kg.add_entity(e)
                kg.chunk_entities[c.chunk_id].append(e.name)
            for r in rels:
                r.source_chunks.append(c.chunk_id)
                kg.add_relation(r)
        return kg

    def build_from_documents(self, docs, chunker=None):
        from rag.chunking import get_chunker, ChunkingConfig
        from rag.loaders import Document
        chunks = []
        for d in docs:
            text = d.content if hasattr(d, "content") else str(d)
            ch = (chunker or get_chunker("recursive", ChunkingConfig())).chunk(text, metadata=getattr(d, "metadata", {}))
            chunks.extend(ch)
        return self.build_from_chunks(chunks)


class GraphRetriever:
    """Retrieve chunks using graph traversal from query entities."""
    def __init__(self, kg: KnowledgeGraph, chunk_store: List[Chunk], embeddings: Optional[BaseEmbeddings] = None, hops: int = 2):
        self.kg = kg
        self.chunk_store = {c.chunk_id: c for c in chunk_store}
        self.embeddings = embeddings
        self.hops = hops
        # entity -> chunks index
        self.entity_chunks = defaultdict(set)
        for cid, ents in kg.chunk_entities.items():
            for e in ents:
                self.entity_chunks[e].add(cid)

    def retrieve(self, query: str, k: int = 5, filter_fn=None) -> List[Tuple[Chunk, float]]:
        # find seed entities in query
        extractor = RuleBasedEntityExtractor()
        seed_ents = [e.name for e in extractor.extract_entities(query)]
        # also fuzzy match against graph nodes
        q_lower = query.lower()
        for node in self.kg.graph.nodes:
            if node.lower() in q_lower and node not in seed_ents:
                seed_ents.append(node)
        # traverse
        relevant_chunks = set()
        for e in seed_ents:
            if e in self.kg.graph:
                nodes = self.kg.neighbors(e, hops=self.hops) | {e}
                for n in nodes:
                    for cid in self.entity_chunks.get(n, ()):
                        relevant_chunks.add(cid)
        # if none, fall back to all chunks via similarity? return empty
        out = []
        for cid in relevant_chunks:
            if cid in self.chunk_store:
                out.append((self.chunk_store[cid], 1.0))
        # rank by embedding similarity if available
        if self.embeddings is not None and out:
            qv = self.embeddings.embed_query(query)
            scored = []
            for c, _ in out:
                cv = self.embeddings.embed([c.text])[0]
                denom = (np.linalg.norm(cv) * np.linalg.norm(qv)) or 1.0
                scored.append((c, float(np.dot(cv, qv) / denom)))
            scored.sort(key=lambda x: -x[1])
            out = scored
        return out[:k]


class GraphRAGPipeline:
    """Combines graph retrieval with vector retrieval."""
    def __init__(self, vector_retriever, graph_retriever, alpha=0.5):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.alpha = alpha

    def retrieve(self, query, k=5, filter_fn=None):
        vres = self.vector_retriever.retrieve(query, k=k, filter_fn=filter_fn)
        gres = self.graph_retriever.retrieve(query, k=k)
        # merge
        scores = {}
        cmap = {}
        for c, s in vres:
            scores[c.chunk_id] = scores.get(c.chunk_id, 0) + self.alpha * (s if s is not None else 1.0)
            cmap[c.chunk_id] = c
        for c, s in gres:
            scores[c.chunk_id] = scores.get(c.chunk_id, 0) + (1 - self.alpha) * (s if s is not None else 1.0)
            cmap[c.chunk_id] = c
        order = sorted(scores, key=lambda x: -scores[x])
        return [(cmap[cid], scores[cid]) for cid in order[:k]]
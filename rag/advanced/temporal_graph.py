"""Temporal knowledge graph: facts with validity intervals and time-aware retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from rag.graph import KnowledgeGraph, Entity, Relation


@dataclass
class TemporalFact:
    subject: str
    predicate: str
    obj: str
    start: Optional[str] = None  # ISO date
    end: Optional[str] = None
    source_chunk: Optional[str] = None
    confidence: float = 1.0


class TemporalKnowledgeGraph(KnowledgeGraph):
    def __init__(self):
        super().__init__()
        self.facts: List[TemporalFact] = []
        self.graph = nx.MultiDiGraph()

    def add_fact(self, fact: TemporalFact):
        self.facts.append(fact)
        if not self.graph.has_node(fact.subject):
            self.graph.add_node(fact.subject, type="ENTITY")
        if not self.graph.has_node(fact.obj):
            self.graph.add_node(fact.obj, type="ENTITY")
        self.graph.add_edge(
            fact.subject, fact.obj,
            relation=fact.predicate,
            start=fact.start, end=fact.end,
            source=fact.source_chunk, confidence=fact.confidence,
        )

    def facts_at(self, time: str) -> List[TemporalFact]:
        out = []
        for f in self.facts:
            if f.start and time < f.start:
                continue
            if f.end and time > f.end:
                continue
            out.append(f)
        return out

    def query_entity_timeline(self, entity: str) -> List[TemporalFact]:
        return [f for f in self.facts if f.subject == entity or f.obj == entity]

    def latest_fact(self, subject: str, predicate: str) -> Optional[TemporalFact]:
        cands = [f for f in self.facts if f.subject == subject and f.predicate == predicate]
        if not cands:
            return None
        # sort by start desc
        def key(f):
            return f.start or ""
        cands.sort(key=key, reverse=True)
        return cands[0]

    @staticmethod
    def parse_date(text: str) -> Optional[str]:
        # find ISO-like or year
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if m:
            return m.group(1)
        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            return m.group(0) + "-01-01"
        return None


class TemporalExtractor:
    """Extract (subject, predicate, object, time) triples with simple patterns."""
    PATTERNS = [
        re.compile(r"(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(became|was|is|appointed|elected|founded|launched|released|started|ended)\s+(.+?)(?:\s+in\s+(\d{4}))?[.\n]", re.I),
    ]

    def extract(self, text: str, chunk_id: Optional[str] = None) -> List[TemporalFact]:
        facts = []
        for pat in self.PATTERNS:
            for m in pat.finditer(text):
                subj = m.group(1).strip()
                pred = m.group(2).strip().lower()
                obj = m.group(3).strip().rstrip(",;:")
                year = m.group(4)
                start = (year + "-01-01") if year else None
                facts.append(TemporalFact(subject=subj, predicate=pred, obj=obj, start=start, source_chunk=chunk_id))
        return facts
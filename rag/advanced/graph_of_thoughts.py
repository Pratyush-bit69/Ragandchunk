"""Graph-of-Thoughts RAG: organize reasoning as a graph of thoughts with aggregation."""
from __future__ import annotations

from typing import Any, Dict, List

from rag.chunking import Chunk


class Thought:
    def __init__(self, content: str, thought_type: str = "intermediate", score: float = 0.0):
        self.content = content
        self.type = thought_type
        self.score = score
        self.children: List["Thought"] = []
        self.parents: List["Thought"] = []

    def add_child(self, t):
        self.children.append(t)
        t.parents.append(self)


class GraphOfThoughts:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.thoughts: List[Thought] = []

    def generate_thoughts(self, query, sub_questions):
        thoughts = []
        # root
        root = Thought(query, "question")
        thoughts.append(root)
        for sq in sub_questions:
            t = Thought(sq, "sub_question")
            root.add_child(t)
            # retrieve + generate for sub-question
            res = self.pipeline.query(sq, top_k=3)
            ans = res.get("answer", "")
            ans_t = Thought(ans, "answer", score=0.5)
            t.add_child(ans_t)
            thoughts.append(t)
            thoughts.append(ans_t)
        # aggregation thought
        agg_content = " ".join(th.content for th in thoughts if th.type == "answer")
        agg = Thought(agg_content, "aggregation")
        root.add_child(agg)
        thoughts.append(agg)
        self.thoughts = thoughts
        return thoughts

    def aggregate(self):
        answers = [t.content for t in self.thoughts if t.type == "answer"]
        return " ".join(answers)

    def to_dict(self):
        return [{"type": t.type, "content": t.content[:200], "children": len(t.children)} for t in self.thoughts]
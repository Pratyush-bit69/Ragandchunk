"""Conversational RAG with multi-turn memory and query condensation."""
from __future__ import annotations

from typing import Any, Dict, List

from rag.chunking import Chunk


class ConversationalRAG:
    def __init__(self, pipeline, llm=None, memory_turns: int = 5):
        self.pipeline = pipeline
        self.llm = llm
        self.memory: List[Dict[str, str]] = []
        self.memory_turns = memory_turns

    def _condense_query(self, query: str) -> str:
        if not self.memory or self.llm is None:
            return query
        history = "\n".join(f"User: {m['q']}\nAssistant: {m['a']}" for m in self.memory[-self.memory_turns:])
        prompt = (
            "Given the conversation history and a follow-up question, rewrite the follow-up as a standalone question.\n\n"
            f"History:\n{history}\n\nFollow-up: {query}\n\nStandalone question:"
        )
        if hasattr(self.llm, "generate"):
            try:
                res = self.llm.generate(prompt, [])
                text = res.get("answer", "") if isinstance(res, dict) else str(res)
                if text.strip():
                    return text.strip()
            except Exception:
                pass
        return query

    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        standalone = self._condense_query(query)
        res = self.pipeline.query(standalone, top_k=top_k)
        answer = res.get("answer", "")
        self.memory.append({"q": query, "a": answer, "standalone": standalone})
        if len(self.memory) > self.memory_turns * 2:
            self.memory = self.memory[-self.memory_turns * 2:]
        return {
            "query": query,
            "standalone_query": standalone,
            "answer": answer,
            "contexts": res.get("contexts", []),
            "chunk_ids": res.get("chunk_ids", []),
            "retrieved": res.get("retrieved", []),
        }

    def reset(self):
        self.memory = []

    def history(self):
        return list(self.memory)
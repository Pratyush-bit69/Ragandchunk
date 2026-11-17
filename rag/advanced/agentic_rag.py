"""Agentic RAG: plan -> retrieve -> grade -> rewrite -> synthesize -> self-critique."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.chunking import Chunk
from rag.pipeline import RAGPipeline
from .query_decomposition import get_decomposer
from .hyde import HyDE
from .retrieval_grader import get_grader
from .query_rewriter import get_rewriter
from .conflict import ConflictDetector, ProvenanceTracker
from .confidence import ConfidenceEstimator


class AgenticRAG:
    def __init__(self, pipeline: RAGPipeline, llm=None, use_hyde=False, max_iters: int = 2, abstention_threshold=0.35):
        self.pipeline = pipeline
        self.llm = llm
        self.use_hyde = use_hyde
        self.max_iters = max_iters
        self.decomposer = get_decomposer(llm)
        self.grader = get_grader(llm)
        self.rewriter = get_rewriter(llm)
        self.conflict = ConflictDetector()
        self.provenance = ProvenanceTracker()
        self.confidence = ConfidenceEstimator(abstention_threshold)
        self.hyde = HyDE(pipeline.generator, pipeline.embeddings) if use_hyde else None

    def _retrieve(self, query, top_k=5):
        if self.use_hyde and self.hyde is not None:
            qv = self.hyde.embed_query(query)
            res = self.pipeline.vectorstore.search(qv, k=top_k)
            return [(r[0], r[1]) for r in res]
        return self.pipeline.retrieve(query, top_k=top_k)

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        trace = []
        # 1. Decompose
        subqs = self.decomposer.decompose(query)
        trace.append({"step": "decompose", "sub_questions": subqs})

        # 2. Retrieve for each sub-question (and original)
        all_chunks: List[Chunk] = []
        score_map = {}
        queries_to_run = [query] + [sq for sq in subqs if sq.lower() != query.lower()]
        for q in queries_to_run:
            # try rewrites
            variants = self.rewriter.rewrite(q)
            for v in variants[:2]:
                res = self._retrieve(v, top_k=top_k)
                for c, s in res:
                    if c.chunk_id not in score_map or s > score_map[c.chunk_id]:
                        score_map[c.chunk_id] = s
                    if c.chunk_id not in {x.chunk_id for x in all_chunks}:
                        all_chunks.append(c)
        trace.append({"step": "retrieve", "num_chunks": len(all_chunks)})

        # 3. Grade relevance
        grade_map = {}
        self._grade_map = grade_map
        graded = []
        for c in all_chunks:
            g = self.grader.grade(query, c)
            grade_map[c.chunk_id] = g
            graded.append((c, g))
        # filter low-grade
        relevant = [c for c, g in graded if g >= 0.15]
        if not relevant:
            relevant = [c for c, _ in graded[:top_k]]
        # sort by grade then retrieval score
        relevant_sorted = sorted(relevant, key=lambda c: (grade_map.get(c.chunk_id, 0), score_map.get(c.chunk_id, 0)), reverse=True)
        relevant_sorted = relevant_sorted[:top_k * 2]
        trace.append({"step": "grade", "kept": len(relevant_sorted)})

        # 4. Iterative refinement: if avg grade low, rewrite and retrieve more
        for it in range(self.max_iters):
            avg_grade = float(sum(self._grade(c) for c in relevant_sorted) / max(1, len(relevant_sorted)))
            if avg_grade >= 0.3:
                break
            # rewrite original query
            new_qs = self.rewriter.rewrite(query)
            added = False
            for v in new_qs:
                res = self._retrieve(v, top_k=top_k)
                for c, s in res:
                    if c.chunk_id not in self._grade_map:
                        self._grade_map[c.chunk_id] = self.grader.grade(query, c)
                    if c.chunk_id not in {x.chunk_id for x in relevant_sorted}:
                        relevant_sorted.append(c)
                        added = True
            if not added:
                break
            relevant_sorted = relevant_sorted[:top_k * 2]
            trace.append({"step": f"refine_iter_{it}", "chunks": len(relevant_sorted)})

        contexts = relevant_sorted[:top_k]
        retrieval_scores = [score_map.get(c.chunk_id, 0.0) for c in contexts]

        # 5. Conflict detection
        conflicts = self.conflict.detect(contexts)
        conflict_summary = self.conflict.summarize(contexts, conflicts)

        # 6. Generate
        gen = self.pipeline.generator.generate(query, contexts)
        answer = gen.get("answer", "")

        # 7. Provenance
        prov = self.provenance.attribute(answer, contexts)

        # 8. Confidence & abstention
        conf = self.confidence.score(query, answer, contexts, retrieval_scores)
        if conf["abstain"]:
            answer = f"I'm not confident enough to answer reliably. (reason: {conf['reason']})\nPartial: " + answer

        # 9. Self-critique (optional simple): check if answer addresses subqs
        critique = self._critique(query, answer, subqs)

        return {
            "query": query,
            "sub_questions": subqs,
            "answer": answer,
            "contexts": [c.text for c in contexts],
            "chunk_ids": [c.chunk_id for c in contexts],
            "retrieved": [(c, score_map.get(c.chunk_id, 0.0)) for c in contexts],
            "conflicts": conflicts,
            "conflict_summary": conflict_summary,
            "provenance": prov,
            "confidence": conf,
            "critique": critique,
            "trace": trace,
        }

    def _grade(self, chunk):
        return self._grade_map.get(getattr(chunk, "chunk_id", None), 0.5)

    def _critique(self, query, answer, subqs):
        # simple coverage check: do sub-question keywords appear in answer?
        ans_l = answer.lower()
        covered = 0
        for sq in subqs:
            toks = [w for w in sq.lower().split() if len(w) > 4]
            if any(t in ans_l for t in toks):
                covered += 1
        return {
            "sub_questions": len(subqs),
            "covered": covered,
            "coverage_ratio": round(covered / max(1, len(subqs)), 3),
        }
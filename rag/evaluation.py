"""Evaluation metrics for RAG: retrieval + generation."""
from __future__ import annotations

import re
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rag.chunking import Chunk


# ---------- Retrieval metrics ----------

def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set, k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for rid in top if rid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for rid in top if rid in relevant_ids)
    return hits / len(relevant_ids)


def hit_rate(retrieved_ids: Sequence[str], relevant_ids: set, k: int) -> float:
    top = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top) else 0.0


def mrr(retrieved_ids: Sequence[str], relevant_ids: set) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def average_precision(retrieved_ids: Sequence[str], relevant_ids: set) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    s = 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            hits += 1
            s += hits / i
    return s / min(len(relevant_ids), len(retrieved_ids)) if relevant_ids else 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set, k: int) -> float:
    top = retrieved_ids[:k]
    dcg = 0.0
    for i, rid in enumerate(top, start=1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    # ideal
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(retrieved_ids: Sequence[str], relevant_ids: set, k_values: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    out = {}
    for k in k_values:
        out[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        out[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        out[f"hit_rate@{k}"] = hit_rate(retrieved_ids, relevant_ids, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    out["mrr"] = mrr(retrieved_ids, relevant_ids)
    out["map"] = average_precision(retrieved_ids, relevant_ids)
    return out


# ---------- Text similarity helpers ----------

def _tokens(s):
    return set(re.findall(r"\w+", s.lower()))


def token_f1(pred: str, gold: str) -> float:
    p = _tokens(pred)
    g = _tokens(gold)
    if not p or not g:
        return 0.0
    inter = p & g
    prec = len(inter) / len(p)
    rec = len(inter) / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def rouge_l(pred: str, gold: str) -> float:
    # simple LCS-based rouge-L
    p = pred.split()
    g = gold.split()
    if not p or not g:
        return 0.0
    # LCS length
    m, n = len(p), len(g)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i-1].lower() == g[j-1].lower():
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m if m else 0
    rec = lcs / n if n else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------- Generation / RAG metrics ----------

def faithfulness(answer: str, contexts: List[str]) -> float:
    """Fraction of answer sentences supported by context (token overlap heuristic)."""
    ctx_tokens = set()
    for c in contexts:
        ctx_tokens |= _tokens(c)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sents:
        return 0.0
    supported = 0
    for s in sents:
        toks = _tokens(s)
        if not toks:
            continue
        overlap = len(toks & ctx_tokens) / len(toks)
        if overlap >= 0.5:
            supported += 1
    return supported / len(sents)


def answer_relevance(answer: str, query: str) -> float:
    """Token overlap between answer and query (heuristic)."""
    a = _tokens(answer)
    q = _tokens(query)
    if not a or not q:
        return 0.0
    return len(a & q) / len(q)


def context_precision(contexts: List[str], relevant_contexts: Optional[List[str]] = None, query: Optional[str] = None) -> float:
    """Precision of retrieved contexts w.r.t. relevant contexts (by id/text). If relevant_contexts given, compute hit ratio."""
    if relevant_contexts is not None:
        rel_set = set(relevant_contexts)
        if not contexts:
            return 0.0
        hits = sum(1 for c in contexts if c in rel_set)
        return hits / len(contexts)
    # fallback: query-context relevance (avg token overlap)
    if query is None:
        return 0.0
    q = _tokens(query)
    scores = []
    for c in contexts:
        toks = _tokens(c)
        if not toks or not q:
            scores.append(0.0)
        else:
            scores.append(len(toks & q) / len(q))
    return float(np.mean(scores)) if scores else 0.0


def context_recall(contexts: List[str], ground_truth: str) -> float:
    """Recall of ground truth tokens covered by contexts."""
    gt = _tokens(ground_truth)
    if not gt:
        return 0.0
    ctx = set()
    for c in contexts:
        ctx |= _tokens(c)
    return len(gt & ctx) / len(gt)


def context_relevance(contexts: List[str], query: str) -> float:
    return context_precision(contexts, query=query)


class RAGEvaluator:
    def __init__(self, k_values=(1, 3, 5, 10)):
        self.k_values = k_values

    def evaluate_retrieval(self, retrieved_ids, relevant_ids, **kw):
        return retrieval_metrics(retrieved_ids, set(relevant_ids), self.k_values)

    def evaluate_generation(self, answer, query, contexts, ground_truth=None):
        out = {
            "faithfulness": faithfulness(answer, contexts),
            "answer_relevance": answer_relevance(answer, query),
            "context_precision": context_precision(contexts, query=query),
            "context_recall": context_recall(contexts, ground_truth) if ground_truth else None,
            "token_f1": token_f1(answer, ground_truth) if ground_truth else None,
            "rouge_l": rouge_l(answer, ground_truth) if ground_truth else None,
            "exact_match": exact_match(answer, ground_truth) if ground_truth else None,
        }
        return {k: v for k, v in out.items() if v is not None}

    def evaluate_end_to_end(self, query, answer, contexts, retrieved_ids, relevant_ids, ground_truth=None):
        out = {}
        out.update(self.evaluate_retrieval(retrieved_ids, relevant_ids))
        out.update({f"gen_{k}": v for k, v in self.evaluate_generation(answer, query, contexts, ground_truth).items()})
        return out


class BenchmarkRunner:
    """Run evaluation over a dataset of QA pairs."""
    def __init__(self, pipeline, evaluator=None):
        self.pipeline = pipeline
        self.evaluator = evaluator or RAGEvaluator()

    def run(self, dataset: List[Dict[str, Any]], k=5) -> Dict[str, Any]:
        results = []
        for item in dataset:
            query = item["question"]
            relevant = item.get("relevant_chunk_ids", []) or item.get("relevant_ids", [])
            ground_truth = item.get("answer") or item.get("ground_truth")
            gen = self.pipeline.query(query, top_k=k)
            retrieved_ids = [c.chunk_id for c, _ in gen.get("retrieved", [])]
            answer = gen.get("answer", "")
            contexts = gen.get("contexts", [])
            metrics = self.evaluator.evaluate_end_to_end(
                query, answer, contexts, retrieved_ids, set(relevant), ground_truth
            )
            results.append({"question": query, "answer": answer, "metrics": metrics})
        # aggregate
        agg = {}
        if results:
            keys = results[0]["metrics"].keys()
            for key in keys:
                vals = [r["metrics"].get(key) for r in results if r["metrics"].get(key) is not None]
                agg[key] = float(np.mean(vals)) if vals else None
        return {"per_query": results, "aggregate": agg, "n": len(results)}

    def to_markdown(self, report) -> str:
        lines = ["# RAG Evaluation Report", "", f"Queries: {report['n']}", "", "## Aggregate Metrics", ""]
        for k, v in report["aggregate"].items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines.append("")
        lines.append("## Per Query")
        for r in report["per_query"]:
            lines.append(f"### {r['question']}")
            lines.append(f"Answer: {r['answer'][:300]}")
            lines.append(f"Metrics: {r['metrics']}")
            lines.append("")
        return "\n".join(lines)
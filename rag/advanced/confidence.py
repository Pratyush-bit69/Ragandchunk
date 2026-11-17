"""Confidence estimation and abstention for RAG answers."""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

from rag.chunking import Chunk


class ConfidenceEstimator:
    def __init__(self, abstention_threshold: float = 0.35):
        self.threshold = abstention_threshold

    def score(self, query: str, answer: str, contexts: List[Chunk], retrieval_scores: List[float]) -> Dict[str, Any]:
        # retrieval confidence: mean top scores
        if retrieval_scores:
            ret_conf = float(np.mean(retrieval_scores[:3]))
        else:
            ret_conf = 0.0
        # context coverage: fraction of answer tokens in contexts
        ctx_tokens = set()
        for c in contexts:
            ctx_tokens |= set(re.findall(r"\w+", c.text.lower()))
        ans_tokens = set(re.findall(r"\w+", answer.lower()))
        coverage = len(ans_tokens & ctx_tokens) / max(1, len(ans_tokens)) if ans_tokens else 0.0
        # agreement: std of retrieval scores (lower std -> more agreement)
        if len(retrieval_scores) > 1:
            agreement = 1.0 - min(1.0, float(np.std(retrieval_scores)))
        else:
            agreement = ret_conf
        # length sanity: very short answers less confident
        length_factor = min(1.0, len(answer.split()) / 8.0)
        overall = float(np.clip(0.4 * ret_conf + 0.3 * coverage + 0.2 * agreement + 0.1 * length_factor, 0.0, 1.0))
        abstain = overall < self.threshold or coverage < 0.2
        return {
            "overall": round(overall, 3),
            "retrieval_confidence": round(ret_conf, 3),
            "context_coverage": round(coverage, 3),
            "agreement": round(agreement, 3),
            "abstain": abstain,
            "reason": self._reason(overall, coverage, ret_conf),
        }

    def _reason(self, overall, coverage, ret_conf):
        if ret_conf < 0.2:
            return "low retrieval confidence"
        if coverage < 0.2:
            return "answer not grounded in context"
        if overall < self.threshold:
            return "overall confidence below threshold"
        return "ok"

    def calibrate_threshold(self, scores: List[float], labels: List[bool]) -> float:
        """Pick threshold maximizing F1 for abstain decisions (labels True=should abstain)."""
        best_t, best_f1 = self.threshold, 0.0
        for t in [i / 100 for i in range(5, 95)]:
            tp = fp = fn = tn = 0
            for s, lab in zip(scores, labels):
                pred = s < t
                if pred and lab: tp += 1
                if pred and not lab: fp += 1
                if (not pred) and lab: fn += 1
                if (not pred) and not lab: tn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = best_t
        return best_t
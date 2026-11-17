"""Evaluation metrics for complex RAG scenarios: multi-hop, conflict, abstention, provenance."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from rag.evaluation import _tokens


def multihop_accuracy(predicted_answer: str, gold_facts: List[str]) -> float:
    """Fraction of required facts (sub-answers) present in predicted answer."""
    if not gold_facts:
        return 0.0
    ans = predicted_answer.lower()
    hits = sum(1 for f in gold_facts if any(tok in ans for tok in _tokens(f)) and len(_tokens(f) & _tokens(predicted_answer)) >= max(1, len(_tokens(f)) // 2))
    return hits / len(gold_facts)


def reasoning_chain_score(predicted_chain: List[str], gold_chain: List[str]) -> float:
    """Overlap between predicted reasoning steps and gold steps."""
    if not gold_chain:
        return 0.0
    pred_set = set(s.lower() for s in predicted_chain)
    gold_set = set(s.lower() for s in gold_chain)
    if not gold_set:
        return 0.0
    return len(pred_set & gold_set) / len(gold_set)


def conflict_resolution_score(answer: str, conflicts: List[dict], preferred_source: str = None) -> float:
    """Did the answer acknowledge conflicts? Heuristic: mentions 'however/although/conflicting' or cites both."""
    if not conflicts:
        return 1.0
    al = answer.lower()
    ack = any(w in al for w in ["however", "although", "conflicting", "disagree", "contrast", "both", "some sources"])
    return 1.0 if ack else 0.0


def abstention_quality(predictions: List[Dict[str, Any]], labels: List[bool]) -> Dict[str, float]:
    """Evaluate abstention decisions. labels True = should abstain (unanswerable)."""
    tp = fp = fn = tn = 0
    for p, lab in zip(predictions, labels):
        abstained = p.get("abstained", False)
        if abstained and lab: tp += 1
        if abstained and not lab: fp += 1
        if (not abstained) and lab: fn += 1
        if (not abstained) and not lab: tn += 1
    total = len(labels) or 1
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def provenance_score(provenance: List[Dict[str, Any]]) -> float:
    """Fraction of answer sentences supported by a source with score >= 0.4."""
    if not provenance:
        return 0.0
    supported = sum(1 for p in provenance if p.get("supported"))
    return supported / len(provenance)


def citation_coverage(answer: str, contexts: List[str]) -> float:
    """Fraction of contexts cited/used in answer (token overlap >= 0.3)."""
    if not contexts:
        return 0.0
    used = 0
    for c in contexts:
        ctoks = _tokens(c)
        atoks = _tokens(answer)
        if ctoks and len(ctoks & atoks) / len(ctoks) >= 0.3:
            used += 1
    return used / len(contexts)


def evaluate_complex(result: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    if "gold_facts" in gold:
        out["multihop_accuracy"] = multihop_accuracy(result.get("answer", ""), gold["gold_facts"])
    if "gold_chain" in gold:
        out["chain_score"] = reasoning_chain_score(result.get("trace", []).__class__ and [], gold["gold_chain"])
    if result.get("conflicts"):
        out["conflict_ack"] = conflict_resolution_score(result.get("answer", ""), result["conflicts"])
    if "provenance" in result:
        out["provenance_score"] = provenance_score(result["provenance"])
    out["citation_coverage"] = citation_coverage(result.get("answer", ""), result.get("contexts", []))
    if "confidence" in result:
        out["confidence"] = result["confidence"].get("overall", 0.0)
        out["abstained"] = 1.0 if result["confidence"].get("abstain") else 0.0
    return out
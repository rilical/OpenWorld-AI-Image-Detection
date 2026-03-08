"""Selective classification and abstention metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from .bootstrap import bootstrap_ci


def risk_coverage(confidence: Iterable[float], correct_mask: Iterable[bool]) -> Dict[str, List[float]]:
    """Compute risk-coverage curve and AURC.

    confidence: model confidence for each sample.
    correct_mask: boolean mask for correctness on all samples (including abstained).
    """
    conf = np.asarray(confidence)
    correct = np.asarray(correct_mask).astype(float)
    n = len(conf)
    if n == 0:
        return {"coverage": [0.0], "risk": [1.0], "aurc": 1.0}

    order = np.argsort(-conf)
    conf_sorted = conf[order]
    correct_sorted = correct[order]

    coverages = [0.0]
    risks = [1.0]
    cum_correct = 0.0
    for i in range(1, n + 1):
        cum_correct = correct_sorted[:i].sum()
        coverage = i / n
        risk = 1.0 - (cum_correct / i)
        coverages.append(float(coverage))
        risks.append(float(risk))

    # Trapezoidal rule for area under risk-coverage curve.
    aurc_val = float(np.trapezoid(risks, coverages))
    return {"coverage": coverages, "risk": risks, "aurc": aurc_val}


def aurc(confidence: Iterable[float], correct_mask: Iterable[bool]) -> float:
    return float(risk_coverage(confidence, correct_mask)["aurc"])


def coverage(answered_mask: Iterable[bool]) -> float:
    ans = np.asarray(answered_mask).astype(bool)
    if ans.size == 0:
        return 0.0
    return float(ans.mean())


def abstain_rate(answered_mask: Iterable[bool]) -> float:
    return 1.0 - coverage(answered_mask)


def selective_accuracy(correct_mask: Iterable[bool], answered_mask: Iterable[bool]) -> float:
    corr = np.asarray(correct_mask).astype(bool)
    ans = np.asarray(answered_mask).astype(bool)
    if ans.sum() == 0:
        return 0.0
    return float((corr & ans).sum() / ans.sum())


def worst_group_selective_accuracy(
    correct_mask: Iterable[bool],
    answered_mask: Iterable[bool],
    group_ids: Iterable[str],
) -> Dict[str, Any]:
    """Selective accuracy per group; returns worst group and per-group breakdown."""
    corr = np.asarray(correct_mask).astype(bool)
    ans = np.asarray(answered_mask).astype(bool)
    groups = np.asarray(group_ids)

    per_group: Dict[str, float] = {}
    for g in np.unique(groups):
        mask = groups == g
        g_ans = ans[mask]
        g_corr = corr[mask]
        if g_ans.sum() == 0:
            per_group[str(g)] = 0.0
        else:
            per_group[str(g)] = float((g_corr & g_ans).sum() / g_ans.sum())

    if not per_group:
        return {"worst": 0.0, "worst_group": "", "per_group": {}}

    worst_group = min(per_group, key=per_group.get)
    return {
        "worst": per_group[worst_group],
        "worst_group": worst_group,
        "per_group": per_group,
    }


__all__ = [
    "risk_coverage",
    "aurc",
    "coverage",
    "abstain_rate",
    "selective_accuracy",
    "worst_group_selective_accuracy",
    "bootstrap_ci",
]

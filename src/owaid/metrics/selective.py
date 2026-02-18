"""Selective classification and abstention metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


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

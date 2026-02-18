"""Binary classification metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn import metrics as skm


def auroc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Area under the ROC curve for class-1 (AI) score."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    if len(np.unique(y)) <= 1:
        return 0.0
    return float(skm.roc_auc_score(y, s))


def tpr_at_fpr(y_true: Iterable[int], y_score: Iterable[float], fpr: float = 0.01) -> float:
    """TPR at a fixed FPR threshold."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    fpr_curve, tpr_curve, _ = skm.roc_curve(y, s)
    # Include all points where FPR <= target then take max TPR achievable.
    mask = fpr_curve <= fpr
    if not mask.any():
        return 0.0
    return float(tpr_curve[mask].max())

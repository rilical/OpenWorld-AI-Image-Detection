"""Binary classification metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
from sklearn import metrics as skm


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Simple classification accuracy."""
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if y.size == 0:
        return 0.0
    return float((y == pred).mean())


def auroc_metadata(y_true: Iterable[int], y_score: Iterable[float]) -> Dict[str, Any]:
    """Return AUROC plus metadata for undefined edge cases."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    if len(np.unique(y)) <= 1:
        return {
            "value": None,
            "defined": False,
            "reason": "AUROC is undefined when only one class is present in y_true.",
        }
    return {"value": float(skm.roc_auc_score(y, s)), "defined": True, "reason": None}


def auroc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Area under the ROC curve for class-1 (AI) score."""
    meta = auroc_metadata(y_true, y_score)
    return float(meta["value"] or 0.0)


def tpr_at_fpr(y_true: Iterable[int], y_score: Iterable[float], fpr: float = 0.01) -> float:
    """TPR at a fixed FPR threshold."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    if len(np.unique(y)) <= 1:
        return 0.0
    fpr_curve, tpr_curve, _ = skm.roc_curve(y, s)
    # Include all points where FPR <= target then take max TPR achievable.
    mask = fpr_curve <= fpr
    if not mask.any():
        return 0.0
    return float(tpr_curve[mask].max())


def binary_confusion_rates(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float | None]:
    """Return confusion-matrix-derived rates when both classes are present."""
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if y.size == 0 or len(np.unique(y)) <= 1:
        return {
            "tpr": None,
            "tnr": None,
            "fpr": None,
            "fnr": None,
            "precision": None,
            "accuracy": accuracy(y, pred) if y.size else 0.0,
        }

    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "tnr": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "accuracy": accuracy(y, pred),
    }

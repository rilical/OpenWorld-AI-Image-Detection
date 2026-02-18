"""Split conformal prediction helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import numpy as np
import torch


def _scores_from_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return 1.0 - probs[np.arange(len(labels)), labels]


def build_split_conformal(
    probs: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    alpha: float,
    method: str = "split",
) -> Dict[str, Any]:
    """Compute split conformal quantile thresholds.

    Args:
      probs: Calibrated probabilities, shape [N, 2].
      labels: True labels, shape [N].
      alpha: Desired miscoverage.
      method: "split" or "mondrian".
    """
    if method not in {"split", "mondrian"}:
        raise ValueError(f"Unknown conformal method: {method}")

    probs_np = probs.detach().cpu().numpy() if torch.is_tensor(probs) else np.asarray(probs)
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    out: Dict[str, Any] = {
        "alpha": float(alpha),
        "method": method,
        "n": int(len(labels_np)),
    }

    if method == "split":
        scores = _scores_from_probs(probs_np, labels_np)
        n = len(scores)
        k = int(math.ceil((n + 1) * (1.0 - alpha)))
        k = max(1, min(k, n))
        qhat = float(np.partition(scores, k - 1)[k - 1])
        out.update({"qhat": qhat})
    else:
        class_qhat: Dict[str, float] = {}
        for c in np.unique(labels_np):
            idx = np.where(labels_np == c)[0]
            if len(idx) == 0:
                continue
            c_scores = _scores_from_probs(probs_np[idx], labels_np[idx])
            n = len(c_scores)
            k = int(math.ceil((n + 1) * (1.0 - alpha)))
            k = max(1, min(k, n))
            class_qhat[str(int(c))] = float(np.partition(c_scores, k - 1)[k - 1])
        out.update({"class_qhat": class_qhat, "qhat": float(max(class_qhat.values(), default=1.0))})

    return out


def prediction_set_from_probs(probs: np.ndarray | torch.Tensor, conformal: Dict[str, Any]):
    probs_np = probs.detach().cpu().numpy() if torch.is_tensor(probs) else np.asarray(probs)
    method = conformal.get("method", "split")
    qhat = conformal["qhat"]
    class_qhat = conformal.get("class_qhat", {})

    sets = []
    for row in probs_np:
        included = []
        for c in range(row.shape[-1]):
            nc = 1.0 - row[c]
            threshold = qhat
            if method == "mondrian" and str(c) in class_qhat:
                threshold = class_qhat[str(c)]
            if nc <= threshold:
                included.append(c)
        sets.append(included)
    return sets

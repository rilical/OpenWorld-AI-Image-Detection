"""Tri-state abstention policy utilities."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

LABEL_NAME = {0: "Real", 1: "AI", -1: "Abstain"}


def _to_tensor(x: Any, device=None) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device) if device is not None else x
    return torch.as_tensor(x, device=device)


def predict_with_abstention(
    logits: torch.Tensor,
    temperature: float | Mapping[str, float] | None = None,
    conformal: Mapping[str, Any] | None = None,
    classes: Sequence[int] = (0, 1),
) -> Dict[str, Any]:
    """Produce logits, calibrated probs, prediction set, and tri-state output."""
    temp = 1.0
    if isinstance(temperature, Mapping):
        temp = float(temperature.get("temperature", 1.0))
    elif temperature is not None:
        temp = float(temperature)
    temp = max(temp, 1e-8)

    probs = F.softmax(logits / temp, dim=-1)
    confidence, pred = probs.max(dim=-1)

    if conformal is None:
        decision = pred
        abstained = torch.zeros_like(pred, dtype=torch.bool)
        pred_sets = [[int(c)] for c in pred.tolist()]
    else:
        qhat = conformal.get("qhat")
        class_qhat = conformal.get("class_qhat") or {}
        method = conformal.get("method", "split")

        pred_sets = []
        decision = []
        abstained = []
        for i in range(probs.size(0)):
            included = []
            for c in classes:
                nc = 1.0 - probs[i, int(c)]
                threshold = qhat
                if method == "mondrian":
                    key = str(int(c))
                    if key in class_qhat:
                        threshold = float(class_qhat[key])
                if nc <= threshold:
                    included.append(int(c))

            if len(included) == 1:
                d = included[0]
                pred_sets.append(included)
                decision.append(d)
                abstained.append(False)
            else:
                pred_sets.append(included)
                decision.append(-1)
                abstained.append(True)

        pred_sets = [list(v) for v in pred_sets]
        decision = torch.tensor(decision, device=logits.device)
        abstained = torch.tensor(abstained, device=logits.device, dtype=torch.bool)

    return {
        "logits": logits,
        "probs": probs,
        "predictions": decision,
        "prediction_set": pred_sets,
        "confidence": confidence,
        "abstained": abstained,
        "labels": [LABEL_NAME[int(d.item())] for d in (decision if torch.is_tensor(decision) else torch.as_tensor(decision))],
    }


def predict_with_threshold_abstention(
    logits: torch.Tensor,
    temperature: float | Mapping[str, float] | None = None,
    tau: float = 0.9,
) -> Dict[str, Any]:
    """Threshold baseline: abstain when max softmax probability < tau."""
    temp = 1.0
    if isinstance(temperature, Mapping):
        temp = float(temperature.get("temperature", 1.0))
    elif temperature is not None:
        temp = float(temperature)
    temp = max(temp, 1e-8)

    probs = F.softmax(logits / temp, dim=-1)
    confidence, pred = probs.max(dim=-1)

    abstained = confidence < tau
    decision = pred.clone()
    decision[abstained] = -1

    pred_sets = []
    for i in range(probs.size(0)):
        if abstained[i]:
            pred_sets.append([0, 1])
        else:
            pred_sets.append([int(pred[i].item())])

    return {
        "logits": logits,
        "probs": probs,
        "predictions": decision,
        "prediction_set": pred_sets,
        "confidence": confidence,
        "abstained": abstained,
        "labels": [LABEL_NAME[int(d.item())] for d in decision],
    }


def sweep_tau(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float | Mapping[str, float] | None = None,
    n_steps: int = 50,
) -> list[Dict[str, float]]:
    """Evaluate threshold abstention over a grid of tau values.

    Returns list of dicts with keys: tau, coverage, selective_accuracy, abstain_rate.
    """
    results = []
    for tau in torch.linspace(0.5, 0.99, n_steps).tolist():
        out = predict_with_threshold_abstention(logits, temperature=temperature, tau=tau)
        answered = ~out["abstained"]
        n_answered = int(answered.sum().item())
        n_total = len(labels)
        cov = n_answered / n_total if n_total > 0 else 0.0
        if n_answered > 0:
            correct = (out["predictions"][answered] == labels[answered]).float()
            sel_acc = float(correct.mean().item())
        else:
            sel_acc = 0.0
        results.append({
            "tau": tau,
            "coverage": cov,
            "selective_accuracy": sel_acc,
            "abstain_rate": 1.0 - cov,
        })
    return results

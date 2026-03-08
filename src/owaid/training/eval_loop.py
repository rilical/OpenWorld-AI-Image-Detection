"""Evaluation loop including abstention metrics and optional predictions export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import pandas as pd

from ..models.abstention import predict_with_abstention
from ..metrics.classification import auroc, tpr_at_fpr
from ..metrics.calibration_metrics import ece
from ..metrics.selective import risk_coverage, abstain_rate, selective_accuracy, coverage


def evaluate_model(
    model,
    loader,
    device: str,
    temperature: float | Dict[str, Any] | None = None,
    conformal: Dict[str, Any] | None = None,
    save_predictions_path: str | None = None,
) -> Dict[str, Any]:
    model.eval()

    all_logits = []
    all_labels = []
    all_meta = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        out = model(images)
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels.cpu())
        all_meta.extend(batch.get("meta", [{}] * len(labels)))

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    abst = predict_with_abstention(logits, temperature=temperature, conformal=conformal)
    # probs = abst["probs"].numpy()
    # predictions = abst["predictions"].numpy()
    # confidence = abst["confidence"].numpy()
    probs = abst["probs"].detach().cpu().numpy() # new for debugging
    predictions = abst["predictions"].detach().cpu().numpy() # new for debugging 
    confidence = abst["confidence"].detach().cpu().numpy() # new for debugging

    # answer_mask = ~abst["abstained"].numpy()
    # y_true = labels.numpy()
    answer_mask = ~abst["abstained"].detach().cpu().numpy() # new for debugging
    y_true = labels.detach().cpu().numpy() # new for debugging 

    y_score = probs[:, 1]
    metrics = {
        "auroc": auroc(y_true, y_score),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, y_score, fpr=0.01),
        "coverage": float(coverage(answer_mask)),
        "abstain_rate": float(abstain_rate(answer_mask)),
    }
    ece_value, ece_payload = ece(probs, y_true)
    metrics["ece"] = ece_value
    metrics.update({"reliability": ece_payload})

    correct_mask = (predictions == y_true)
    rc = risk_coverage(confidence, correct_mask)
    metrics["aurc"] = rc["aurc"]
    metrics["risk_coverage"] = rc
    metrics["selective_accuracy"] = float(selective_accuracy(correct_mask, answer_mask))

    if save_predictions_path:
        df = pd.DataFrame(
            {
                "label": y_true,
                "pred": predictions,
                "prob_ai": probs[:, 1],
                "confidence": confidence,
                "abstained": ~answer_mask,
                "prediction_set": [str(p) for p in abst["prediction_set"]],
            }
        )
        df["meta"] = all_meta
        Path(save_predictions_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(save_predictions_path)
        except Exception:
            fallback = str(save_predictions_path).replace(".parquet", ".csv")
            df.to_csv(fallback, index=False)

    return metrics

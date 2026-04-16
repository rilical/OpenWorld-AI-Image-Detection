"""Evaluation loop including abstention metrics and prediction export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..metrics import (
    auroc,
    auroc_metadata,
    binary_confusion_rates,
    bootstrap_ci,
    ece,
    tpr_at_fpr,
)
from ..metrics.calibration_metrics import empirical_conformal_coverage
from ..metrics.selective import (
    abstain_rate,
    coverage,
    risk_coverage,
    selective_accuracy,
    worst_group_selective_accuracy,
)
from ..models.abstention import predict_with_abstention, predict_with_threshold_abstention


def _normalize_meta(meta: Any, index: int) -> Dict[str, Any]:
    if isinstance(meta, dict):
        return meta
    return {"id": f"sample:{index}", "raw_meta": meta}


def _extract_group_ids(meta_list: List[Dict[str, Any]]) -> np.ndarray | None:
    group_ids = []
    for meta in meta_list:
        group = meta.get("generator") or meta.get("group") or meta.get("source")
        if group is None:
            return None
        group_ids.append(str(group))
    return np.asarray(group_ids) if group_ids else None


def _confidence_summary(confidence: np.ndarray) -> Dict[str, float]:
    if confidence.size == 0:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}
    return {
        "mean": float(confidence.mean()),
        "median": float(np.median(confidence)),
        "p05": float(np.percentile(confidence, 5)),
        "p95": float(np.percentile(confidence, 95)),
    }


def _per_generator_breakdown(
    group_ids: np.ndarray,
    labels: np.ndarray,
    y_score: np.ndarray,
    predictions: np.ndarray,
    answered_mask: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    breakdown: Dict[str, Dict[str, Any]] = {}
    correct_mask = predictions == labels
    for group in np.unique(group_ids):
        mask = group_ids == group
        if mask.sum() == 0:
            continue
        group_labels = labels[mask]
        group_scores = y_score[mask]
        group_answered = answered_mask[mask]
        group_correct = correct_mask[mask]
        meta = auroc_metadata(group_labels, group_scores)
        breakdown[str(group)] = {
            "n": int(mask.sum()),
            "coverage": float(coverage(group_answered)),
            "selective_accuracy": float(selective_accuracy(group_correct, group_answered)),
            "auroc": meta["value"],
            "auroc_defined": bool(meta["defined"]),
        }
    return breakdown


def _paired_attack_summary(meta_list: List[Dict[str, Any]], predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any] | None:
    by_pair: Dict[str, Dict[str, bool]] = {}
    for meta, pred, label in zip(meta_list, predictions, labels):
        pair_id = meta.get("pair_id") or meta.get("source_id")
        variant = meta.get("variant")
        if variant is None and meta.get("is_adversarial") is not None:
            variant = "adversarial" if meta.get("is_adversarial") else "clean"
        if pair_id is None or variant is None:
            return None
        bucket = by_pair.setdefault(str(pair_id), {})
        bucket[str(variant)] = bool(pred == label)

    if not by_pair:
        return None

    changed = 0
    total = 0
    for variants in by_pair.values():
        if "clean" in variants and "adversarial" in variants:
            total += 1
            if variants["clean"] and not variants["adversarial"]:
                changed += 1
    if total == 0:
        return None
    return {"paired_attack_success_rate": changed / total, "paired_examples": total}


def _flatten_prediction_rows(
    labels: np.ndarray,
    predictions: np.ndarray,
    probs: np.ndarray,
    confidence: np.ndarray,
    pred_sets: List[List[int]],
    answered_mask: np.ndarray,
    meta_list: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for idx, meta in enumerate(meta_list):
        row = {
            "sample_id": meta.get("id", f"sample:{idx}"),
            "label": int(labels[idx]),
            "pred": int(predictions[idx]),
            "prob_real": float(probs[idx, 0]),
            "prob_ai": float(probs[idx, 1]),
            "confidence": float(confidence[idx]),
            "abstained": bool(not answered_mask[idx]),
            "prediction_set": json.dumps(pred_sets[idx]),
            "source_dataset": meta.get("source_dataset"),
            "split": meta.get("split"),
            "generator": meta.get("generator"),
            "path": meta.get("path"),
            "real_only": bool(meta.get("real_only", False)),
            "meta_json": json.dumps(meta, sort_keys=True),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_model(
    model=None,
    loader=None,
    device: str = "cpu",
    temperature: float | Dict[str, Any] | None = None,
    conformal: Dict[str, Any] | None = None,
    save_predictions_path: str | None = None,
    abstention_method: str = "conformal",
    tau: float = 0.9,
    dataset_name: str | None = None,
    predictor=None,
) -> Dict[str, Any]:
    """Run evaluation with configurable abstention policy."""
    if loader is None:
        raise ValueError("evaluate_model requires a dataloader")
    if predictor is None and model is None:
        raise ValueError("evaluate_model requires either model or predictor")

    eval_model = predictor.model if predictor is not None else model
    eval_model.eval()

    logits_list = []
    labels_list = []
    meta_list: List[Dict[str, Any]] = []

    with torch.no_grad():
        n_total = len(loader) if hasattr(loader.dataset, "__len__") else None
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"eval [{dataset_name or ''}]", leave=False, total=n_total)):
            if batch is None:
                continue
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            print(f"[eval] batch {batch_idx} | samples so far: {batch_idx * images.shape[0] + images.shape[0]}", flush=True)
            out = eval_model(images)
            logits_list.append(out["logits"].detach().cpu())
            labels_list.append(labels.detach().cpu())
            batch_meta = batch.get("meta", [{}] * len(labels))
            meta_list.extend(_normalize_meta(meta, batch_idx * len(labels) + i) for i, meta in enumerate(batch_meta))

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    if predictor is not None:
        abst = predictor.apply_conformal_if_available(
            logits.to(device),
            abstention_method=abstention_method,
            tau=tau,
        )
        logits = logits.cpu()
    elif abstention_method == "threshold":
        abst = predict_with_threshold_abstention(logits, temperature=temperature, tau=tau)
    elif abstention_method == "forced":
        abst = predict_with_abstention(logits, temperature=temperature, conformal=None)
    else:
        abst = predict_with_abstention(logits, temperature=temperature, conformal=conformal)

    probs = abst["probs"].detach().cpu().numpy()
    predictions = abst["predictions"].detach().cpu().numpy()
    confidence = abst["confidence"].detach().cpu().numpy()
    pred_sets = abst["prediction_set"]
    answered_mask = (~abst["abstained"]).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    y_score = probs[:, 1]
    correct_mask = predictions == y_true
    dataset_key = (dataset_name or "").lower()

    metrics: Dict[str, Any] = {
        "dataset": dataset_name,
        "abstention_method": abstention_method,
        "n_samples": int(len(y_true)),
        "coverage": float(coverage(answered_mask)),
        "abstain_rate": float(abstain_rate(answered_mask)),
        "selective_accuracy": float(selective_accuracy(correct_mask, answered_mask)),
        "accuracy_non_abstained": float(selective_accuracy(correct_mask, answered_mask)),
        "confidence_summary": _confidence_summary(confidence),
    }
    if abstention_method == "threshold":
        metrics["tau"] = float(tau)

    auroc_info = auroc_metadata(y_true, y_score)
    metrics["auroc"] = auroc_info["value"]
    if not auroc_info["defined"]:
        metrics["auroc_note"] = auroc_info["reason"]
        metrics["tpr_at_1pct_fpr"] = None
        metrics["tpr_at_1pct_fpr_note"] = "Undefined when only one class is present in y_true."
    else:
        metrics["tpr_at_1pct_fpr"] = float(tpr_at_fpr(y_true, y_score, fpr=0.01))

    if np.unique(y_true).size > 1:
        ece_value, ece_payload = ece(probs, y_true)
        metrics["ece"] = ece_value
        metrics["reliability"] = ece_payload
    else:
        metrics["ece"] = None
        metrics["ece_note"] = "ECE is omitted for single-class evaluation because confidence bins are not class-balanced."
        metrics["reliability"] = None

    answered_predictions = predictions[answered_mask]
    answered_labels = y_true[answered_mask]
    metrics["answered_confusion_rates"] = binary_confusion_rates(answered_labels, answered_predictions)

    rc = risk_coverage(confidence, np.where(answered_mask, correct_mask, False))
    metrics["aurc"] = rc["aurc"]
    metrics["risk_coverage"] = rc

    group_ids = _extract_group_ids(meta_list)
    metrics["conformal_coverage"] = empirical_conformal_coverage(pred_sets, y_true, group_ids=group_ids)
    if group_ids is not None:
        metrics["worst_group_selective_accuracy"] = worst_group_selective_accuracy(correct_mask, answered_mask, group_ids)
        metrics["per_generator"] = _per_generator_breakdown(group_ids, y_true, y_score, predictions, answered_mask)

    if auroc_info["defined"]:
        metrics["bootstrap_auroc"] = bootstrap_ci(lambda yt, ys: auroc(yt, ys), y_true, y_score)
    else:
        metrics["bootstrap_auroc"] = None
    metrics["bootstrap_selective_accuracy"] = bootstrap_ci(
        lambda corr, ans: selective_accuracy(corr, ans),
        correct_mask,
        answered_mask,
    )

    if dataset_key == "aria" or np.unique(y_true).size == 1 and np.all(y_true == 0):
        ai_mask = predictions == 1
        metrics["false_positive_rate"] = float(ai_mask.mean())
        metrics["ai_prediction_rate"] = float(ai_mask.mean())
        metrics["selective_false_positive_rate"] = (
            float(ai_mask[answered_mask].mean()) if answered_mask.any() else None
        )

    if dataset_key == "raid":
        metrics["attack_success_proxy"] = float(np.mean((predictions != y_true) | (~answered_mask)))
        metrics["attack_success_proxy_note"] = (
            "Proxy defined as the rate of incorrect predictions or abstentions on evaluated RAID samples."
        )
        paired = _paired_attack_summary(meta_list, predictions, y_true)
        if paired is not None:
            metrics["paired_summary"] = paired
        else:
            metrics["paired_summary"] = None
            metrics["paired_summary_note"] = "No clean/adversarial pairing metadata was available."

    if save_predictions_path:
        df = _flatten_prediction_rows(y_true, predictions, probs, confidence, pred_sets, answered_mask, meta_list)
        target = Path(save_predictions_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(target)
        except Exception:
            df.to_csv(target.with_suffix(".csv"), index=False)

    return metrics

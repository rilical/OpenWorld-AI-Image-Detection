"""Shared helpers for command-line scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from owaid.data import build_eval_dataloader
from owaid.inference import (
    Predictor,
    build_model_from_config,
    load_calibration_artifacts,
    load_checkpoint,
)
from owaid.training import evaluate_model
from owaid.utils.config import deep_update, load_yaml, merge_cli_overrides
from owaid.utils.logging import JsonlLogger, write_json
from owaid.utils.paths import make_run_dir


def load_config_with_overrides(
    config_path: str,
    run_dir: str | None = None,
    overrides=None,
) -> Dict[str, Any]:
    """Load config, then optionally merge a run's resolved config and CLI overrides."""
    cfg = load_yaml(config_path)

    if run_dir:
        run_cfg_path = Path(run_dir) / "config.yaml"
        if run_cfg_path.exists():
            with run_cfg_path.open("r", encoding="utf-8") as f:
                run_cfg = yaml.safe_load(f) or {}
            cfg = deep_update(cfg, run_cfg)

    if overrides:
        cfg = merge_cli_overrides(cfg, overrides)
    return cfg


def _load_model(cfg: Dict[str, Any], ckpt_path: str | None, device: str):
    model = build_model_from_config(cfg)
    if ckpt_path and Path(ckpt_path).exists():
        model, _ = load_checkpoint(ckpt_path, model=model, device=device)
    return model.to(device)


def _build_predictor(cfg: Dict[str, Any], run_dir: str, device: str, temperature, conformal) -> Predictor:
    ckpt = Path(run_dir) / "checkpoints" / "best.pt"
    if not ckpt.exists():
        ckpt = Path(run_dir) / "checkpoints" / "last.pt"
    model = _load_model(cfg, str(ckpt), device)
    return Predictor(
        model=model,
        transform=lambda image: image,
        temperature=temperature,
        conformal=conformal,
        device=device,
    )


def evaluate_in_run(
    cfg: Dict[str, Any],
    run_dir: str,
    dataset_name: str,
    artifact_name: str,
    device: str,
    save_predictions: bool = True,
    evaluation_mode: str = "all",
    tau: float = 0.9,
):
    """Evaluate one dataset under one or more abstention modes and persist artifacts."""
    loader = build_eval_dataloader(cfg, dataset_name)
    calibration = load_calibration_artifacts(run_dir)
    temperature_artifact = calibration["temperature"]
    conformal_artifact = calibration["conformal"]

    mode_specs = {
        "forced": {
            "predictor": _build_predictor(cfg, run_dir, device, temperature=None, conformal=None),
            "abstention_method": "forced",
        },
        "temperature": {
            "predictor": _build_predictor(cfg, run_dir, device, temperature=temperature_artifact, conformal=None),
            "abstention_method": "forced",
        },
        "threshold": {
            "predictor": _build_predictor(cfg, run_dir, device, temperature=temperature_artifact, conformal=None),
            "abstention_method": "threshold",
        },
        "conformal": {
            "predictor": _build_predictor(cfg, run_dir, device, temperature=temperature_artifact, conformal=conformal_artifact),
            "abstention_method": "conformal",
        },
    }
    selected_modes = list(mode_specs) if evaluation_mode == "all" else [evaluation_mode]

    eval_dir = Path(run_dir) / "eval" / artifact_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, Any] = {}
    eval_logger = JsonlLogger(str(Path(run_dir) / "logs" / "eval.jsonl"))

    for mode in selected_modes:
        spec = mode_specs[mode]
        preds_path = None
        if save_predictions:
            preds_path = str(eval_dir / f"predictions_{mode}.parquet")

        metrics = evaluate_model(
            loader=loader,
            device=device,
            save_predictions_path=preds_path,
            abstention_method=spec["abstention_method"],
            tau=tau,
            dataset_name=artifact_name,
            predictor=spec["predictor"],
        )
        metrics["evaluation_mode"] = mode
        all_metrics[mode] = metrics
        eval_logger.log({"dataset": dataset_name, "artifact": artifact_name, "mode": mode, "metrics": metrics})

    write_json(str(eval_dir / "metrics.json"), all_metrics)
    return all_metrics


def ensure_run_dir(cfg: Dict[str, Any], run: str | None = None) -> str:
    """Return the provided run directory or create a new one from config."""
    if run:
        return run
    return make_run_dir(cfg)

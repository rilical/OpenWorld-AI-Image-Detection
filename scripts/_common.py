"""Shared helpers for command-line scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from owaid.data import build_eval_dataloader
from owaid.models import CLIPBinaryDetector, ClipDIREFusionDetector
from owaid.training import load_checkpoint
from owaid.utils.config import load_yaml, merge_cli_overrides
from owaid.utils.logging import JsonlLogger, read_json, write_json
from owaid.utils.paths import make_run_dir


def load_config_with_overrides(config_path: str, run_dir: str | None = None, overrides=None) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    if overrides:
        cfg = merge_cli_overrides(cfg, overrides)

    if run_dir:
        run_cfg_path = Path(run_dir) / "config.yaml"
        if run_cfg_path.exists():
            with run_cfg_path.open("r", encoding="utf-8") as f:
                run_cfg = yaml.safe_load(f)
            cfg.update(run_cfg)

    return cfg


def _load_model(cfg: Dict[str, Any], ckpt_path: str | None, device: str):
    model_cfg = cfg.get("model", {})
    clip_cfg = model_cfg.get("clip", {})
    dire_cfg = model_cfg.get("dire", {})
    head_cfg = model_cfg.get("head", {})
    mtype = model_cfg.get("type", "clip_baseline")
    hidden = head_cfg.get("hidden_dims", [512, 256])
    dropout = float(head_cfg.get("dropout", 0.1))

    if mtype == "clip_dire_fusion":
        model = ClipDIREFusionDetector(
            model_name=clip_cfg.get("model_name", "ViT-B-32"),
            pretrained=clip_cfg.get("pretrained", "openai"),
            freeze=bool(clip_cfg.get("freeze", True)),
            unfreeze_last_n=clip_cfg.get("unfreeze_last_n"),
            head_hidden_dims=hidden,
            dropout=dropout,
            cache_dir=dire_cfg.get("cache_dir"),
        )
    else:
        model = CLIPBinaryDetector(
            model_name=clip_cfg.get("model_name", "ViT-B-32"),
            pretrained=clip_cfg.get("pretrained", "openai"),
            freeze=bool(clip_cfg.get("freeze", True)),
            unfreeze_last_n=clip_cfg.get("unfreeze_last_n"),
            head_hidden_dims=hidden,
            dropout=dropout,
        )

    if ckpt_path and Path(ckpt_path).exists():
        _ = load_checkpoint(ckpt_path, model)

    return model.to(device)


def load_calibration_artifacts(run_dir: str):
    temp_path = Path(run_dir) / "calibration" / "temperature.json"
    conf_path = Path(run_dir) / "calibration" / "conformal.json"

    temperature = None
    conformal = None
    if temp_path.exists():
        temperature = read_json(str(temp_path)).get("temperature")
    if conf_path.exists():
        conformal = read_json(str(conf_path))
    return temperature, conformal


def evaluate_in_run(
    cfg: Dict[str, Any],
    run_dir: str,
    dataset_name: str,
    artifact_name: str,
    device: str,
    save_predictions: bool = True,
):
    from owaid.training import evaluate_model

    ckpt = Path(run_dir) / "checkpoints" / "best.pt"
    if not ckpt.exists():
        ckpt = Path(run_dir) / "checkpoints" / "last.pt"

    model = _load_model(cfg, str(ckpt), device)
    loader = build_eval_dataloader(cfg, dataset_name)
    temperature, conformal = load_calibration_artifacts(run_dir)

    preds_path = None
    if save_predictions:
        preds_path = str(Path(run_dir) / "eval" / artifact_name / "predictions.parquet")

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        temperature=temperature,
        conformal=conformal,
        save_predictions_path=preds_path,
    )

    out = Path(run_dir) / "eval" / artifact_name / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(out), metrics)
    eval_logger = JsonlLogger(str(Path(run_dir) / "logs" / "eval.jsonl"))
    eval_logger.log(
        {
            "dataset": dataset_name,
            "artifact": artifact_name,
            "metrics": metrics,
        }
    )
    return metrics


def ensure_run_dir(cfg: Dict[str, Any], run: str | None = None) -> str:
    if run:
        return run
    return make_run_dir(cfg)

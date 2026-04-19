"""Artifact-loading helpers for shared inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

from ..models import (
    CLIPBinaryDetector,
    ClipDIREFusionDetector,
    DomainAdversarialWrapper,
    SGFNet,
)
from ..training import load_checkpoint as training_load_checkpoint
from ..utils.logging import read_json


def load_run_config(run_dir: str) -> Dict[str, Any]:
    """Load the resolved config stored under a run directory."""
    cfg_path = Path(run_dir) / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Run config must be a mapping: {cfg_path}")
    return cfg


def resolve_checkpoint_path(run_dir_or_checkpoint: str) -> Path:
    """Resolve a run directory or explicit checkpoint path to a checkpoint file."""
    path = Path(run_dir_or_checkpoint)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    best = path / "checkpoints" / "best.pt"
    last = path / "checkpoints" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(
        f"No checkpoint found under {path}. Expected {best} or {last}."
    )


def _build_base_model(cfg: Dict[str, Any]):
    """Construct the configured detector model without any DAT wrapping."""
    model_cfg = cfg.get("model", {})
    clip_cfg = model_cfg.get("clip", {})
    head_cfg = model_cfg.get("head", {})
    dire_cfg = model_cfg.get("dire", {})
    hidden = head_cfg.get("hidden_dims", [512, 256])
    dropout = float(head_cfg.get("dropout", 0.1))

    model_type = model_cfg.get("type", "clip_baseline")

    if model_type == "sgf_net":
        sgf_cfg = model_cfg.get("sgf", {})
        return SGFNet(
            model_name=clip_cfg.get("model_name", "ViT-B-32"),
            pretrained=clip_cfg.get("pretrained", "openai"),
            freeze_clip=bool(clip_cfg.get("freeze", True)),
            spectral_dim=int(sgf_cfg.get("spectral_dim", 128)),
            pixel_dim=int(sgf_cfg.get("pixel_dim", 128)),
            fused_dim=int(sgf_cfg.get("fused_dim", 256)),
            head_hidden_dims=sgf_cfg.get("head_hidden_dims", [128]),
            dropout=dropout,
        )

    if model_type == "clip_dire_fusion" and bool(dire_cfg.get("enabled", True)):
        return ClipDIREFusionDetector(
            model_name=clip_cfg.get("model_name", "ViT-B-32"),
            pretrained=clip_cfg.get("pretrained", "openai"),
            freeze=bool(clip_cfg.get("freeze", True)),
            unfreeze_last_n=clip_cfg.get("unfreeze_last_n"),
            head_hidden_dims=hidden,
            dropout=dropout,
            cache_dir=dire_cfg.get("cache_dir"),
        )

    return CLIPBinaryDetector(
        model_name=clip_cfg.get("model_name", "ViT-B-32"),
        pretrained=clip_cfg.get("pretrained", "openai"),
        freeze=bool(clip_cfg.get("freeze", True)),
        unfreeze_last_n=clip_cfg.get("unfreeze_last_n"),
        head_hidden_dims=hidden,
        dropout=dropout,
    )


def build_model_from_config(cfg: Dict[str, Any]):
    """Construct the configured detector model, optionally wrapped for DAT."""
    base_model = _build_base_model(cfg)

    train_cfg = cfg.get("train", {})
    dat_cfg = train_cfg.get("domain_adversarial", {}) if isinstance(train_cfg, dict) else {}
    if isinstance(dat_cfg, dict) and bool(dat_cfg.get("enabled", False)):
        with torch.no_grad():
            was_training = base_model.training
            base_model.eval()
            probe = torch.zeros(1, 3, 224, 224)
            out = base_model(probe, return_features=True)
            if was_training:
                base_model.train()
        if not isinstance(out, dict) or "features" not in out:
            raise ValueError(
                "build_model_from_config: configured model "
                f"{type(base_model).__name__} does not expose 'features' with "
                "return_features=True; cannot enable DAT."
            )
        feature_dim = int(out["features"].shape[-1])
        return DomainAdversarialWrapper(
            base_model=base_model,
            feature_dim=feature_dim,
            hidden_dims=dat_cfg.get("hidden_dims", [128]),
            num_domains=int(dat_cfg.get("num_domains", 2)),
            dropout=float(dat_cfg.get("dropout", 0.1)),
            max_lambda=float(dat_cfg.get("max_lambda", 0.5)),
        )

    return base_model


def load_checkpoint(
    run_dir_or_checkpoint: str,
    model=None,
    optimizer=None,
    device: str = "cpu",
) -> Tuple[Any, Dict[str, Any]]:
    """Load a checkpoint from a run directory or explicit checkpoint path."""
    ckpt_path = resolve_checkpoint_path(run_dir_or_checkpoint)
    if model is None:
        run_dir = ckpt_path.parent.parent
        model = build_model_from_config(load_run_config(str(run_dir)))
    state = training_load_checkpoint(str(ckpt_path), model, optimizer=optimizer)
    model.to(device)
    return model, state


def load_calibration_artifacts(run_dir: str) -> Dict[str, Any]:
    """Load available calibration artifacts from a run directory."""
    run_path = Path(run_dir)
    artifacts: Dict[str, Any] = {"temperature": None, "conformal": None}
    temp_path = run_path / "calibration" / "temperature.json"
    conf_path = run_path / "calibration" / "conformal.json"
    if temp_path.exists():
        artifacts["temperature"] = read_json(str(temp_path))
    if conf_path.exists():
        artifacts["conformal"] = read_json(str(conf_path))
    return artifacts

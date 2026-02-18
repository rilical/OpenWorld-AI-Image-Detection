"""Train CLIP + residual fusion detector."""

from __future__ import annotations

import argparse
from datetime import datetime
import platform
import socket
import subprocess
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

from owaid.data import build_commfor_dataloaders
from owaid.models import ClipDIREFusionDetector
from owaid.training import run_training
from owaid.utils.config import load_yaml, merge_cli_overrides
from owaid.utils.logging import JsonlLogger, write_json, write_yaml
from owaid.utils.paths import make_run_dir
from owaid.utils.seed import set_seed


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _build_meta(cfg):
    import torch

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_hash": _git_hash(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": cfg.get("seed", 0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--opts", nargs="*")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = merge_cli_overrides(cfg, args.opts)
    set_seed(int(cfg.get("seed", 0)), deterministic=bool(cfg.get("deterministic", True)))

    run_dir = make_run_dir(cfg)
    cfg.setdefault("output", {})["run_dir"] = run_dir
    write_yaml(str(Path(run_dir) / "config.yaml"), cfg)
    write_json(str(Path(run_dir) / "meta.json"), _build_meta(cfg))

    dataloaders = build_commfor_dataloaders(cfg)
    model_cfg = cfg.get("model", {})
    clip_cfg = model_cfg.get("clip", {})
    dire_cfg = model_cfg.get("dire", {})
    hidden = model_cfg.get("head", {}).get("hidden_dims", [512, 256])
    dropout = float(model_cfg.get("head", {}).get("dropout", 0.1))

    model = ClipDIREFusionDetector(
        model_name=clip_cfg.get("model_name", "ViT-B-32"),
        pretrained=clip_cfg.get("pretrained", "openai"),
        freeze=bool(clip_cfg.get("freeze", True)),
        unfreeze_last_n=clip_cfg.get("unfreeze_last_n"),
        head_hidden_dims=hidden,
        dropout=dropout,
        cache_dir=dire_cfg.get("cache_dir"),
    ).to(args.device)

    logger = JsonlLogger(str(Path(run_dir) / "logs" / "train.jsonl"))
    run_training(cfg, model, dataloaders["train"], dataloaders.get("val"), run_dir, args.device, logger)


if __name__ == "__main__":
    main()

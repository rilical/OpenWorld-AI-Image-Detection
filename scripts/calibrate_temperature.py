"""Fit temperature on held-out calibration split."""

from __future__ import annotations

import os
os.environ.setdefault(
    "HF_HOME",
    "/ocean/projects/cis250202p/gyar/personal/dl/OpenWorld-AI-Image-Detection/.cache/huggingface",
)

import argparse
from datetime import datetime
import platform
import socket
import subprocess
import torch

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

from owaid.calibration import fit_temperature
from owaid.data import build_commfor_dataloaders
from owaid.utils.logging import write_json, write_yaml
from owaid.utils.seed import set_seed
from scripts._common import ensure_run_dir, _load_model, load_config_with_overrides


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--opts", nargs="*")
    return p.parse_args()


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _build_meta(cfg):
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_hash": _git_hash(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "seed": cfg.get("seed", 0),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config_with_overrides(args.config, overrides=args.opts)
    set_seed(int(cfg.get("seed", 0)), deterministic=bool(cfg.get("deterministic", True)))
    run_dir = ensure_run_dir(cfg, args.run)
    cfg.setdefault("output", {})["run_dir"] = run_dir
    write_yaml(str(Path(run_dir) / "config.yaml"), cfg)
    model = _load_model(cfg, args.ckpt, args.device)
    cfg.setdefault("output", {})["run_dir"] = run_dir
    write_json(str(Path(run_dir) / "meta.json"), _build_meta(cfg) | {"torch_version": torch.__version__})
    dataloaders = build_commfor_dataloaders(cfg)
    report = fit_temperature(model, dataloaders["calibration"], device=args.device)

    write_json(f"{run_dir}/calibration/temperature.json", report)


if __name__ == "__main__":
    main()

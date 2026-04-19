"""Build split-conformal thresholds."""

from __future__ import annotations

import argparse
import torch

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

from owaid.calibration import build_split_conformal
from owaid.data import build_commfor_dataloaders
from owaid.utils.logging import read_json, write_json
from owaid.utils.seed import set_seed
from scripts._common import ensure_run_dir, _load_model, load_config_with_overrides


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--temperature", required=True)
    p.add_argument("--run", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--opts", nargs="*")
    return p.parse_args()


def _collect_probs(model, loader, device, temperature):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            out = model(images)
            probs = torch.softmax(out["logits"] / temperature, dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_probs), torch.cat(all_labels)


def main() -> None:
    args = parse_args()
    cfg = load_config_with_overrides(args.config, run_dir=args.run, overrides=args.opts)
    set_seed(int(cfg.get("seed", 0)), deterministic=bool(cfg.get("deterministic", True)))
    run_dir = ensure_run_dir(cfg, args.run)
    cfg.setdefault("output", {})["run_dir"] = run_dir

    model = _load_model(cfg, args.ckpt, args.device)
    loader = build_commfor_dataloaders(cfg)["calibration"]

    temperature = float(read_json(args.temperature)["temperature"])
    probs, labels = _collect_probs(model, loader, args.device, temperature)
    report = build_split_conformal(
        probs=probs,
        labels=labels,
        alpha=cfg.get("calibration", {}).get("conformal_alpha", 0.05),
        method=cfg.get("calibration", {}).get("conformal_method", "split"),
    )
    write_json(f"{run_dir}/calibration/conformal.json", report)


if __name__ == "__main__":
    main()

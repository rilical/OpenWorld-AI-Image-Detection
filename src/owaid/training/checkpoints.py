"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str, model, optimizer, epoch: int, global_step: int, cfg: dict):
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "cfg": cfg,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "cfg": ckpt.get("cfg"),
    }

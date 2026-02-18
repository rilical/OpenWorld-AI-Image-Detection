"""Minimal local demo interface for tri-state prediction.

This module intentionally keeps inference logic separate from training code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from PIL import Image
from owaid.models import CLIPBinaryDetector, ClipDIREFusionDetector
from owaid.models.abstention import predict_with_abstention
from owaid.training import load_checkpoint
from owaid.utils.logging import read_json


def _load_model(run_dir: str, device: str):
    cfg_path = Path(run_dir) / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    clip_cfg = model_cfg.get("clip", {})
    head_cfg = model_cfg.get("head", {})
    dire_cfg = model_cfg.get("dire", {})
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

    ckpt = Path(run_dir) / "checkpoints" / "best.pt"
    if ckpt.exists():
        load_checkpoint(str(ckpt), model)
    model.to(device).eval()

    temp = None
    conformal = None
    tpath = Path(run_dir) / "calibration" / "temperature.json"
    cpath = Path(run_dir) / "calibration" / "conformal.json"
    if tpath.exists():
        temp = read_json(str(tpath)).get("temperature")
    if cpath.exists():
        conformal = read_json(str(cpath))
    return model, temp, conformal


def run_inference(image_path: str, run_dir: str, device: str = "cpu"):
    model, temp, conformal = _load_model(run_dir, device)
    img = Image.open(image_path).convert("RGB")
    # Use model's preprocessing assumptions: expect normalized tensor.
    # For bootstrap, rely on simple ToTensor + CLIP normalization.
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        result = predict_with_abstention(out["logits"], temperature=temp, conformal=conformal)

    return {
        "label": result["labels"][0],
        "confidence": float(result["confidence"][0].item()),
        "prediction_set": result["prediction_set"][0],
        "abstained": bool(result["abstained"][0].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = run_inference(args.image, args.run, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Predict a single image from a saved run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

from PIL import Image

from owaid.inference import load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single image from a saved run")
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = load_run(args.run, device=args.device)
    image = Image.open(args.image).convert("RGB")
    result = predictor.predict_pil(image)
    output = {
        "tri_state_label": result["tri_state_label"],
        "pred_label": result["pred_label"],
        "confidence": result["confidence"],
        "prediction_set": result["prediction_set"],
        "abstained": result["abstained"],
        "conformal_applied": predictor.conformal is not None,
        "residual_available": predictor.render_residual_heatmap(image) is not None,
        "probs": result["probs"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

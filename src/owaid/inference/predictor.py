"""Unified inference API for open-world AI image detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from ..data.transforms import build_clip_transform
from ..models import ClipDIREFusionDetector
from ..models.abstention import predict_with_abstention, predict_with_threshold_abstention
from .io import build_model_from_config, load_calibration_artifacts, load_checkpoint, load_run_config


class Predictor:
    """Wrap a trained model plus optional calibration artifacts."""

    def __init__(
        self,
        model: torch.nn.Module,
        transform,
        temperature: float | dict | None = None,
        conformal: dict | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.transform = transform
        self.temperature = temperature
        self.conformal = conformal
        self.device = device

    def apply_temperature_if_available(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities for logits."""
        if isinstance(self.temperature, dict):
            temperature = float(self.temperature.get("temperature", 1.0))
        elif self.temperature is None:
            temperature = 1.0
        else:
            temperature = float(self.temperature)
        temperature = max(temperature, 1e-8)
        return torch.softmax(logits / temperature, dim=-1)

    def apply_conformal_if_available(
        self,
        logits: torch.Tensor,
        abstention_method: str = "conformal",
        tau: float | None = None,
    ) -> Dict[str, Any]:
        """Apply the configured abstention policy to a logits tensor."""
        if abstention_method == "threshold":
            return predict_with_threshold_abstention(
                logits,
                temperature=self.temperature,
                tau=float(tau or 0.9),
            )
        if abstention_method == "forced":
            return predict_with_abstention(logits, temperature=self.temperature, conformal=None)
        return predict_with_abstention(
            logits,
            temperature=self.temperature,
            conformal=self.conformal,
        )

    def _normalize_batch_result(self, logits: torch.Tensor, result: Dict[str, Any]) -> Dict[str, Any]:
        tri_state = result["labels"]
        predictions = result["predictions"].detach().cpu().tolist()
        probs = result["probs"].detach().cpu()
        confidence = result["confidence"].detach().cpu().tolist()
        abstained = result["abstained"].detach().cpu().tolist()
        return {
            "logits": logits.detach().cpu(),
            "probs": probs,
            "pred_label": predictions,
            "prediction_set": result["prediction_set"],
            "tri_state_label": tri_state,
            "abstained": abstained,
            "confidence": confidence,
            "labels": tri_state,
        }

    def predict_pil(self, image: Image.Image) -> Dict[str, Any]:
        """Single PIL image -> normalized prediction dict."""
        batch = self.predict_batch(self.transform(image).unsqueeze(0))
        return {
            "logits": batch["logits"][0].tolist(),
            "probs": batch["probs"][0].tolist(),
            "pred_label": batch["pred_label"][0],
            "prediction_set": batch["prediction_set"][0],
            "tri_state_label": batch["tri_state_label"][0],
            "abstained": bool(batch["abstained"][0]),
            "confidence": float(batch["confidence"][0]),
            "label": batch["tri_state_label"][0],
        }

    def predict_batch(
        self,
        images: torch.Tensor,
        abstention_method: str = "conformal",
        tau: float | None = None,
    ) -> Dict[str, Any]:
        """Batch tensor [B,3,H,W] -> normalized batched predictions."""
        images = images.to(self.device)
        with torch.no_grad():
            out = self.model(images)
            result = self.apply_conformal_if_available(
                out["logits"],
                abstention_method=abstention_method,
                tau=tau,
            )
        return self._normalize_batch_result(out["logits"], result)

    def predict_set(self, image: Image.Image) -> List[int]:
        """Single image -> conformal prediction set."""
        return self.predict_pil(image)["prediction_set"]

    def predict_tri_state(self, image: Image.Image) -> str:
        """Single image -> 'AI' | 'Real' | 'Abstain'."""
        return self.predict_pil(image)["label"]

    def predict_proba(self, image_or_batch: Image.Image | torch.Tensor) -> Any:
        """Return calibrated probabilities for a single image or a tensor batch."""
        if isinstance(image_or_batch, Image.Image):
            tensor = self.transform(image_or_batch).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
            return self.apply_temperature_if_available(out["logits"])[0].detach().cpu().tolist()

        if torch.is_tensor(image_or_batch):
            with torch.no_grad():
                out = self.model(image_or_batch.to(self.device))
            return self.apply_temperature_if_available(out["logits"]).detach().cpu()

        raise TypeError(f"Unsupported input type for predict_proba: {type(image_or_batch)!r}")

    def render_residual_heatmap(self, image: Image.Image) -> Optional[Image.Image]:
        """If DIRE model, return residual heatmap overlay. None for baseline."""
        if not isinstance(self.model, ClipDIREFusionDetector):
            return None
        return None


def load_run(run_dir: str, device: str = "cpu") -> Predictor:
    """Load model + calibration artifacts from a completed run directory."""
    run_path = Path(run_dir)
    cfg = load_run_config(str(run_path))
    model = build_model_from_config(cfg)
    model, _ = load_checkpoint(str(run_path), model=model, device=device)
    model.eval()
    artifacts = load_calibration_artifacts(str(run_path))
    transform = build_clip_transform(cfg, train=False)

    return Predictor(
        model=model,
        transform=transform,
        temperature=artifacts["temperature"],
        conformal=artifacts["conformal"],
        device=device,
    )

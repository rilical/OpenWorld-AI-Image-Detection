"""Temperature scaling for classification confidence calibration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from ..utils.logging import read_json, write_json


def _collect_logits_labels(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list = []
    label_list = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            out = model(images)
            logits_list.append(out["logits"].detach().cpu())
            label_list.append(labels.detach().cpu())
    return torch.cat(logits_list), torch.cat(label_list)


def _nll(logits: torch.Tensor, labels: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits / temperature, labels, reduction="mean")


def fit_temperature(model: Any, loader, device: str = "cpu") -> Dict[str, float]:
    """Fit scalar temperature via NLL minimization on a calibration loader."""
    logits, labels = _collect_logits_labels(model, loader, device)
    logits = logits.to(device)
    labels = labels.to(device)

    temperature = torch.ones(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)

    with torch.no_grad():
        nll_before = float(_nll(logits, labels, temperature).item())

    for _ in range(10):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            nll_loss = _nll(logits, labels, temperature)
            nll_loss.backward()
            return nll_loss

        optimizer.step(closure)

    with torch.no_grad():
        temperature.clamp_(min=1e-6)
        nll_after = float(_nll(logits, labels, temperature).item())

    return {
        "temperature": float(temperature.item()),
        "n": int(labels.numel()),
        "nll_before": nll_before,
        "nll_after": nll_after,
    }


def apply_temperature(logits: torch.Tensor, temperature: float | Dict[str, float]) -> torch.Tensor:
    """Apply a scalar temperature to logits."""
    if isinstance(temperature, dict):
        value = float(temperature.get("temperature", 1.0))
    else:
        value = float(temperature)
    value = max(value, 1e-8)
    return logits / value


def save_temperature_artifact(path: str, artifact: Dict[str, float]) -> None:
    """Persist a temperature artifact as JSON."""
    write_json(path, artifact)


def load_temperature_artifact(path: str) -> Dict[str, float]:
    """Load a temperature artifact from JSON."""
    data = read_json(path)
    if not isinstance(data, dict) or "temperature" not in data:
        raise ValueError(f"Invalid temperature artifact: {path}")
    return data


if __name__ == "__main__":
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.9]])
    scaled = apply_temperature(logits, 2.0)
    print({"input_shape": list(logits.shape), "scaled_shape": list(scaled.shape)})

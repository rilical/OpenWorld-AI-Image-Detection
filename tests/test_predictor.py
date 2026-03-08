from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from owaid.inference.predictor import Predictor, load_run


class DummyModel(torch.nn.Module):
    def forward(self, images):
        batch = images.shape[0]
        logits = torch.tensor([[0.2, 1.0]] * batch, dtype=torch.float32)
        return {"logits": logits}


def test_predictor_normalizes_single_image():
    predictor = Predictor(
        model=DummyModel(),
        transform=lambda image: torch.zeros(3, 4, 4),
        temperature={"temperature": 2.0},
        conformal={"qhat": 0.6, "method": "split"},
        device="cpu",
    )
    result = predictor.predict_pil(Image.new("RGB", (4, 4), color="white"))
    assert set(result) >= {
        "logits",
        "probs",
        "pred_label",
        "prediction_set",
        "tri_state_label",
        "abstained",
        "confidence",
    }


def test_load_run_uses_saved_artifacts(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "calibration").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "model:",
                "  type: clip_baseline",
                "  clip:",
                "    model_name: ViT-B-32",
                "    pretrained: openai",
                "    freeze: true",
                "  head:",
                "    hidden_dims: [4]",
                "    dropout: 0.0",
                "data:",
                "  img_size: 4",
            ]
        ),
        encoding="utf-8",
    )
    torch.save({"model_state_dict": {}, "epoch": 0, "global_step": 0}, run_dir / "checkpoints" / "best.pt")
    (run_dir / "calibration" / "temperature.json").write_text(json.dumps({"temperature": 1.5}), encoding="utf-8")
    (run_dir / "calibration" / "conformal.json").write_text(json.dumps({"qhat": 0.7, "method": "split"}), encoding="utf-8")

    monkeypatch.setattr("owaid.inference.io.build_model_from_config", lambda cfg: DummyModel())
    monkeypatch.setattr("owaid.inference.predictor.build_model_from_config", lambda cfg: DummyModel())
    monkeypatch.setattr("owaid.inference.io.load_checkpoint", lambda path, model=None, optimizer=None, device="cpu": (model or DummyModel(), {}))
    monkeypatch.setattr("owaid.inference.predictor.load_checkpoint", lambda path, model=None, optimizer=None, device="cpu": (model or DummyModel(), {}))

    predictor = load_run(str(run_dir), device="cpu")
    result = predictor.predict_pil(Image.new("RGB", (4, 4), color="white"))
    assert predictor.temperature == {"temperature": 1.5}
    assert predictor.conformal == {"qhat": 0.7, "method": "split"}
    assert result["tri_state_label"] in {"AI", "Real", "Abstain"}

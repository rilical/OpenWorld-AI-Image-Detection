"""CLIP encoder + small MLP head for binary AI/Real classification."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except Exception:  # pragma: no cover - optional dependency fallback
    open_clip = None


class CLIPBinaryDetector(nn.Module):
    """Binary CLIP detector with optional encoder freezing."""

    def __init__(
        self,
        model_name: str | Any = "ViT-B-32",
        pretrained: str = "openai",
        freeze: bool = True,
        unfreeze_last_n: int | None = None,
        head_hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Create model.

        Parameters
        ----------
        model_name:
            Either a string backbone name or a config-like object.
        pretrained:
            OpenCLIP pretrained tag when using string-based construction.
        freeze:
            If True (default), encoder is frozen.
        unfreeze_last_n:
            If provided with ``freeze=True``, unfreeze last N transformer blocks.
        head_hidden_dims:
            Iterable of hidden dimensions for the classification head.
        """
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "open_clip package not installed. Install open-clip-torch before "
                "constructing CLIPBinaryDetector."
            )

        if not isinstance(model_name, str):
            cfg = model_name
            if isinstance(cfg, dict):
                cfg_dict = cfg
            else:
                cfg_dict = vars(cfg)

            if isinstance(cfg_dict, dict):
                model_cfg = cfg_dict.get("model", cfg_dict)
                if isinstance(model_cfg, dict):
                    model_cfg = model_cfg.get("clip", model_cfg.get("backbone", model_cfg))
                if isinstance(model_cfg, dict):
                    model_name = str(model_cfg.get("model_name", "ViT-B-32"))
                    pretrained = str(model_cfg.get("pretrained", "openai"))
                    freeze = bool(model_cfg.get("freeze", freeze))
                    unfreeze_last_n = model_cfg.get("unfreeze_last_n", unfreeze_last_n)

        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.freeze = bool(freeze)
        self.unfreeze_last_n = unfreeze_last_n

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self.clip_model.eval()

        with torch.no_grad():
            probe = torch.zeros(1, 3, 224, 224)
            embedding = self.encode(probe)
        feature_dim = int(embedding.shape[-1])

        self._configure_encoder_freeze()

        dims = list(head_hidden_dims or [512, 256])
        head_layers: list[nn.Module] = []
        in_dim = feature_dim
        for hidden in dims:
            head_layers.extend([nn.Linear(in_dim, int(hidden)), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            in_dim = int(hidden)
        head_layers.append(nn.Linear(in_dim, 2))
        self.head = nn.Sequential(*head_layers)

    def _extract_transformer_blocks(self):
        visual = getattr(self.clip_model, "visual", None)
        if visual is None:
            return []

        transformer = getattr(visual, "transformer", None)
        if transformer is not None and hasattr(transformer, "resblocks"):
            return list(transformer.resblocks)

        if hasattr(visual, "resblocks"):
            return list(visual.resblocks)

        return []

    def _configure_encoder_freeze(self) -> None:
        for param in self.clip_model.parameters():
            param.requires_grad = not self.freeze

        if self.freeze and self.unfreeze_last_n:
            n = int(self.unfreeze_last_n)
            if n > 0:
                blocks = self._extract_transformer_blocks()
                if blocks:
                    for block in blocks[-n:]:
                        for p in block.parameters():
                            p.requires_grad = True
                else:
                    # Deterministic fallback if block-level access is unavailable.
                    params = list(self.clip_model.parameters())
                    for p in params[-n:]:
                        p.requires_grad = True

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return image embeddings for input images."""
        requires_grad = any(p.requires_grad for p in self.clip_model.parameters())
        with torch.set_grad_enabled(requires_grad):
            features = self.clip_model.encode_image(images)
        if features.ndim > 2:
            features = features[:, 0]
        return features

    def forward(self, images: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """Run classifier.

        Returns
        -------
        dict
            ``{"logits": ..., "probs": ..., "features": ...?}``
        """
        if images.ndim != 4:
            raise ValueError("images must be BCHW tensor")

        features = self.encode(images)
        logits = self.head(features)
        out = {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
        }
        if return_features:
            out["features"] = features
        return out


if __name__ == "__main__":
    if open_clip is None:
        print("open_clip not installed; skipping clip detector smoke test")
    else:
        model = CLIPBinaryDetector(model_name="ViT-B-32", pretrained="openai")
        x = torch.randn(1, 3, 224, 224)
        out = model(x, return_features=True)
        assert out["logits"].shape == (1, 2)
        assert out["probs"].shape == (1, 2)
        assert out["features"].shape[0] == 1
        print("smoke-ok")

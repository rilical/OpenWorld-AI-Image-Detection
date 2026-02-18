"""Fusion head for CLIP + residual branch."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .clip_detector import CLIPBinaryDetector
from .dire_residual import ResidualEncoder


class ClipDIREFusionDetector(nn.Module):
    """Concatenate CLIP embedding and residual embedding and classify."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        freeze: bool = True,
        unfreeze_last_n: int | None = None,
        head_hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.1,
        residual_dim: int = 128,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone = CLIPBinaryDetector(
            model_name=model_name,
            pretrained=pretrained,
            freeze=freeze,
            unfreeze_last_n=unfreeze_last_n,
            head_hidden_dims=(),
            dropout=dropout,
        )
        self.residual_encoder = ResidualEncoder(in_channels=3, out_dim=residual_dim, cache_dir=cache_dir)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            clip_dim = self.backbone.encode(dummy).shape[-1]
        in_dim = clip_dim + residual_dim
        dims = list(head_hidden_dims or [256, 128])

        layers = []
        cur = in_dim
        for dim in dims:
            layers.extend([nn.Linear(cur, dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            cur = dim
        layers.append(nn.Linear(cur, 2))
        self.head = nn.Sequential(*layers)

    def forward(self, images, return_features: bool = False, sample_ids=None):
        clip_features = self.backbone.encode(images)
        residual_features = self.residual_encoder(images, sample_ids=sample_ids)
        fused = torch.cat([clip_features, residual_features.to(clip_features.device)], dim=-1)
        logits = self.head(fused)
        out = {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
        }
        if return_features:
            out["features"] = fused
        return out

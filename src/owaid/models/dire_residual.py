"""Residual branch utilities for DIRE-style residual encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEncoder(nn.Module):
    """Lightweight residual encoder with optional filesystem cache."""

    def __init__(self, in_channels: int = 3, out_dim: int = 128, cache_dir: str | None = None) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, out_dim),
        )
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _reconstruct_like(images: torch.Tensor) -> torch.Tensor:
        # A lightweight reconstruction approximation to avoid external diffusion dependency.
        return F.avg_pool2d(images, kernel_size=3, stride=1, padding=1)

    def forward(self, images: torch.Tensor, sample_ids: list[str] | None = None) -> torch.Tensor:
        # Offline cache: if sample_ids provided and cache exists, reuse features.
        if sample_ids is not None and self.cache_dir:
            cached = []
            to_compute = []
            for idx, sample_id in enumerate(sample_ids):
                path = Path(self.cache_dir) / f"{sample_id}.pt"
                if path.exists():
                    cached.append((idx, torch.load(path, map_location=images.device)))
                else:
                    to_compute.append(idx)

            features = torch.empty((images.size(0), 128), device=images.device, dtype=images.dtype)
            if to_compute:
                residual = torch.abs(images - self._reconstruct_like(images))
                computed = self.net(residual[to_compute])
                for local_idx, value in zip(to_compute, computed):
                    global_idx = local_idx
                    features[global_idx] = value
                    torch.save(value.cpu(), Path(self.cache_dir) / f"{sample_ids[global_idx]}.pt")
            for global_idx, value in cached:
                features[global_idx] = value.to(images.device)
            return features

        residual = torch.abs(images - self._reconstruct_like(images))
        return self.net(residual)


class ResidualEncoderNoCache(ResidualEncoder):
    pass

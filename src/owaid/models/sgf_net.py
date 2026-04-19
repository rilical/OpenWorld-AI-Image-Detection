"""SGF-Net: Spectral-Gated Forensic Fusion Network.

A three-branch architecture that fuses CLIP semantic features, FFT spectral
forensics, and NPR/SRM pixel-level structural artifacts through a learned
spectral gating mechanism for open-world AI image detection.

All SGF-specific components live in this file:
  * ``SpectralBranch`` — 2D FFT log-magnitude → CNN; exposes ``spectral_stats``
    (low/mid/high band [mean, std, kurtosis]) for gating.
  * ``PixelForensicBranch`` — NPR (4 directional neighbor diffs, Tan et al.
    CVPR 2024) + SRM high-pass filters → CNN; exposes ``pixel_stats``
    (per-direction [mean, std]) for gating.
  * ``SpectralGatingNetwork`` — tiny MLP over spectral+pixel stats → softmax
    gate weights over the three branches.
  * ``SGFNet`` — assembles the three branches, the gate, and a classifier head.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_detector import CLIPBinaryDetector


# ----------------------------------------------------------------------------
# Spectral branch
# ----------------------------------------------------------------------------

class SpectralBranch(nn.Module):
    """Extract spectral forensic features via 2D FFT + lightweight CNN.

    Real images follow a characteristic 1/f spectral falloff. Diffusion models
    produce anomalous spectral peaks, and GANs create periodic grid artifacts
    in the frequency domain. This branch learns to detect those deviations
    from a log-magnitude spectrum.
    """

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    @staticmethod
    def _to_log_spectrum(images: torch.Tensor) -> torch.Tensor:
        """Centered log-magnitude spectrum per RGB channel, shape (B, 3, H, W)."""
        freq = torch.fft.fft2(images, norm="ortho")
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        magnitude = freq.abs()
        return torch.log1p(magnitude)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spectrum = self._to_log_spectrum(images)
        return self.cnn(spectrum)

    def spectral_stats(self, images: torch.Tensor) -> torch.Tensor:
        """Lightweight spectral statistics for gating, shape (B, 9).

        Per image: [mean, std, kurtosis] for [low, mid, high] frequency bands.
        """
        spectrum = self._to_log_spectrum(images)
        spec_avg = spectrum.mean(dim=1)
        B, H, W = spec_avg.shape
        cy, cx = H // 2, W // 2

        y = torch.arange(H, device=images.device).float() - cy
        x = torch.arange(W, device=images.device).float() - cx
        dist = (y[:, None] ** 2 + x[None, :] ** 2).sqrt()
        max_r = dist.max()

        low = dist <= max_r / 3
        mid = (dist > max_r / 3) & (dist <= 2 * max_r / 3)
        high = dist > 2 * max_r / 3

        stats = []
        for mask in [low, mid, high]:
            vals = spec_avg[:, mask]
            mean = vals.mean(dim=1)
            std = vals.std(dim=1)
            centered = vals - mean.unsqueeze(1)
            var = (centered ** 2).mean(dim=1).clamp(min=1e-8)
            kurt = (centered ** 4).mean(dim=1) / (var ** 2) - 3.0
            stats.extend([mean, std, kurt])

        return torch.stack(stats, dim=1)


# ----------------------------------------------------------------------------
# Pixel forensic branch (NPR + SRM)
# ----------------------------------------------------------------------------

_SRM_FILTER_1 = torch.tensor(
    [[ 0, -1,  0],
     [-1,  4, -1],
     [ 0, -1,  0]], dtype=torch.float32
)

_SRM_FILTER_2 = torch.tensor(
    [[-1,  2, -1],
     [ 2, -4,  2],
     [-1,  2, -1]], dtype=torch.float32
)

_SRM_FILTER_3 = torch.tensor(
    [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]], dtype=torch.float32
)


class PixelForensicBranch(nn.Module):
    """Extract pixel-level forensic features via NPR + SRM filters + CNN.

    NPR (Neighboring Pixel Relationships, Tan et al. CVPR 2024) exposes
    upsampling artifacts from generative models. SRM high-pass filters extract
    noise residuals that reveal processing-pipeline fingerprints. Together
    they capture fine-grained local artifacts that CLIP's semantic features
    miss.
    """

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = out_dim

        self.register_buffer(
            "_srm_filters",
            torch.stack([_SRM_FILTER_1, _SRM_FILTER_2, _SRM_FILTER_3]).unsqueeze(1),
        )

        # 7 channels in: 4 NPR directions + 3 SRM responses
        self.cnn = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    @staticmethod
    def _compute_npr(images: torch.Tensor) -> torch.Tensor:
        """4-channel NPR: absolute differences between neighboring pixels, (B, 4, H, W)."""
        gray = images.mean(dim=1, keepdim=True)

        h_diff = torch.abs(gray[:, :, :, :-1] - gray[:, :, :, 1:])
        h_diff = F.pad(h_diff, (0, 1))

        v_diff = torch.abs(gray[:, :, :-1, :] - gray[:, :, 1:, :])
        v_diff = F.pad(v_diff, (0, 0, 0, 1))

        dr_diff = torch.abs(gray[:, :, :-1, :-1] - gray[:, :, 1:, 1:])
        dr_diff = F.pad(dr_diff, (0, 1, 0, 1))

        dl_diff = torch.abs(gray[:, :, :-1, 1:] - gray[:, :, 1:, :-1])
        dl_diff = F.pad(dl_diff, (1, 0, 0, 1))

        return torch.cat([h_diff, v_diff, dr_diff, dl_diff], dim=1)

    def _compute_srm(self, images: torch.Tensor) -> torch.Tensor:
        """Apply 3 SRM high-pass filters to the grayscale of ``images``, (B, 3, H, W)."""
        gray = images.mean(dim=1, keepdim=True)
        return F.conv2d(gray, self._srm_filters, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        npr = self._compute_npr(images)
        srm = self._compute_srm(images)
        combined = torch.cat([npr, srm], dim=1)
        return self.cnn(combined)

    def pixel_stats(self, images: torch.Tensor) -> torch.Tensor:
        """Lightweight pixel-artifact statistics for gating, shape (B, 8).

        Per image: [mean, std] for each of the 4 NPR directions.
        """
        npr = self._compute_npr(images)
        B = npr.shape[0]
        npr_flat = npr.view(B, 4, -1)
        means = npr_flat.mean(dim=2)
        stds = npr_flat.std(dim=2)
        return torch.cat([means, stds], dim=1)


# ----------------------------------------------------------------------------
# Spectral gating network
# ----------------------------------------------------------------------------

class SpectralGatingNetwork(nn.Module):
    """Per-image softmax gate over N branches, driven by spectral+pixel stats.

    Lets the model learn that diffusion-generated images are best detected via
    spectral anomalies while GAN images are best caught via pixel artifacts,
    and route accordingly.
    """

    def __init__(
        self,
        spectral_stats_dim: int = 9,
        pixel_stats_dim: int = 8,
        n_branches: int = 3,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        input_dim = spectral_stats_dim + pixel_stats_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_branches),
        )
        # Initialize near-uniform to avoid early branch collapse
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.xavier_uniform_(self.gate[-1].weight, gain=0.1)

    def forward(
        self,
        spectral_stats: torch.Tensor,
        pixel_stats: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([spectral_stats, pixel_stats], dim=1)
        return torch.softmax(self.gate(combined), dim=1)


# ----------------------------------------------------------------------------
# SGF-Net
# ----------------------------------------------------------------------------

class SGFNet(nn.Module):
    """Spectral-Gated Forensic Fusion Network.

    Three parallel branches extract complementary forensic signals:
      1. Semantic (frozen CLIP ViT encoder)
      2. Spectral (FFT log-magnitude → CNN)
      3. Pixel forensic (NPR + SRM → CNN)

    A spectral gating network dynamically weights each branch per-image
    based on spectral and structural statistics, then a classification head
    produces binary AI/Real logits.
    """

    def __init__(
        self,
        # CLIP backbone config
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        freeze_clip: bool = True,
        # Branch output dimensions
        spectral_dim: int = 128,
        pixel_dim: int = 128,
        fused_dim: int = 256,
        # Classification head
        head_hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Branch 1: CLIP semantic encoder (frozen by default)
        self.clip_backbone = CLIPBinaryDetector(
            model_name=model_name,
            pretrained=pretrained,
            freeze=freeze_clip,
            head_hidden_dims=(),  # encoder-only
            dropout=dropout,
        )
        with torch.no_grad():
            probe = torch.zeros(1, 3, 224, 224)
            clip_dim = self.clip_backbone.encode(probe).shape[-1]
        self.clip_dim = clip_dim

        # Branch 2: Spectral forensics
        self.spectral_branch = SpectralBranch(out_dim=spectral_dim)

        # Branch 3: Pixel forensics (NPR + SRM)
        self.pixel_branch = PixelForensicBranch(out_dim=pixel_dim)

        # Branch projection layers → common fused_dim
        self.proj_clip = nn.Linear(clip_dim, fused_dim)
        self.proj_spectral = nn.Linear(spectral_dim, fused_dim)
        self.proj_pixel = nn.Linear(pixel_dim, fused_dim)

        # Spectral gating network
        self.gating = SpectralGatingNetwork(
            spectral_stats_dim=9,
            pixel_stats_dim=8,
            n_branches=3,
        )

        # Classification head
        dims = list(head_hidden_dims or [128])
        head_layers: list[nn.Module] = []
        in_dim = fused_dim
        for hidden in dims:
            head_layers.extend([
                nn.Linear(in_dim, int(hidden)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = int(hidden)
        head_layers.append(nn.Linear(in_dim, 2))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        if images.ndim != 4:
            raise ValueError("images must be BCHW tensor")

        # Branch embeddings
        z_clip = self.clip_backbone.encode(images)
        z_spec = self.spectral_branch(images)
        z_pixel = self.pixel_branch(images)

        # Gating statistics
        spec_stats = self.spectral_branch.spectral_stats(images)
        pixel_stats = self.pixel_branch.pixel_stats(images)
        gate = self.gating(spec_stats, pixel_stats)

        # Project each branch to common dimension and apply gating
        p_clip = self.proj_clip(z_clip)
        p_spec = self.proj_spectral(z_spec)
        p_pixel = self.proj_pixel(z_pixel)

        z_fused = (
            gate[:, 0:1] * p_clip
            + gate[:, 1:2] * p_spec
            + gate[:, 2:3] * p_pixel
        )

        logits = self.head(z_fused)
        out: Dict[str, Any] = {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
        }
        if return_features:
            out["features"] = z_fused
            out["gate_weights"] = gate
            out["z_clip"] = z_clip
            out["z_spectral"] = z_spec
            out["z_pixel"] = z_pixel

        return out


__all__ = ["SGFNet", "SpectralBranch", "PixelForensicBranch", "SpectralGatingNetwork"]

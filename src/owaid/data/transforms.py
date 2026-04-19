"""Image preprocessing and optional corruption utilities."""

from __future__ import annotations

import io
import random
from typing import Any, Callable, Dict

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import Tensor
from torchvision import transforms

try:
    import open_clip
except Exception:  # pragma: no cover
    open_clip = None


DEFAULT_NORMALIZATION = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])


def _to_tensor(sample: Any) -> Tensor:
    """Convert common input formats to a float CHW tensor."""
    if torch.is_tensor(sample):
        if sample.ndim != 3:
            raise ValueError("Tensor images must be CHW")
        return sample.float()

    if isinstance(sample, Image.Image):
        return transforms.ToTensor()(sample)

    if isinstance(sample, np.ndarray):
        if sample.ndim == 3 and sample.shape[0] == 3:
            tensor = torch.from_numpy(sample).float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            return tensor

        if sample.ndim == 3 and sample.shape[-1] == 3:
            tensor = torch.from_numpy(sample).permute(2, 0, 1).float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            return tensor

        return transforms.ToTensor()(Image.fromarray(sample))

    raise TypeError(f"Unsupported image type: {type(sample)!r}")


def _jpeg_quality_transform(quality: int | None) -> Callable[[Image.Image], Image.Image] | None:
    """Create a transform that re-encodes a PIL image as JPEG."""
    if quality is None:
        return None

    def _fn(img: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        q = int(np.clip(quality, 1, 100))
        img.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")

    return _fn


class RandomJPEGCompression:
    """Randomly re-encode a PIL image as JPEG at a random quality.

    With probability ``p``, the image is re-encoded at a quality drawn
    uniformly from ``[quality_range[0], quality_range[1]]``. Uses Python's
    ``random`` module (consistent with torchvision transforms) so global
    seeding via ``set_seed`` carries over.
    """

    def __init__(self, p: float = 0.7, quality_range: tuple[int, int] = (60, 95)) -> None:
        lo, hi = int(quality_range[0]), int(quality_range[1])
        if lo > hi:
            lo, hi = hi, lo
        self.p = float(p)
        self.quality_lo = max(1, min(100, lo))
        self.quality_hi = max(1, min(100, hi))

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        quality = random.randint(self.quality_lo, self.quality_hi)
        buffer = io.BytesIO()
        rgb = img if img.mode == "RGB" else img.convert("RGB")
        rgb.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"{self.__class__.__name__}(p={self.p}, "
            f"quality_range=({self.quality_lo}, {self.quality_hi}))"
        )


def _gaussian_blur_transform(sigma: float | None) -> Callable[[Image.Image], Image.Image] | None:
    """Create a PIL gaussian blur transform."""
    if sigma is None:
        return None

    return lambda img: img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def _get_model_norm(cfg_dict: Dict[str, Any]) -> tuple[list[float], list[float]]:
    """Read OpenCLIP normalization from the selected model when available."""
    if open_clip is None:
        return DEFAULT_NORMALIZATION

    model_cfg = cfg_dict.get("model", cfg_dict)
    if not isinstance(model_cfg, dict):
        return DEFAULT_NORMALIZATION

    clip_cfg = model_cfg.get("clip", model_cfg.get("backbone", {}))
    if not isinstance(clip_cfg, dict):
        clip_cfg = {}

    model_name = str(clip_cfg.get("model_name", "ViT-B-32"))
    pretrained = str(clip_cfg.get("pretrained", "openai"))

    try:
        _, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
    except Exception:
        return DEFAULT_NORMALIZATION

    for t in preprocess.transforms:
        if isinstance(t, transforms.Normalize):
            return (list(t.mean), list(t.std))
    return DEFAULT_NORMALIZATION


def build_clip_transform(cfg: Any, train: bool) -> Callable[[Any], Tensor]:
    """Build a CLIP-compatible transform pipeline.

    When ``train=True`` and ``data.transforms.tier1.enabled`` is set, a
    "Tier-1 bias-removal" augmentation pipeline is prepended before
    ``ToTensor``/``Normalize``:

        RandomJPEGCompression -> RandomResizedCrop -> ColorJitter

    Tier-1 replaces the terminal ``Resize((img_size, img_size))`` with
    ``RandomResizedCrop(img_size, scale=tier1.resized_crop_scale)``. The
    eval path (``train=False``) and the default (tier1 absent / disabled)
    are unchanged.

    Examples
    --------
    >>> cfg = {"data": {"img_size": 224, "transforms": {"resize_shorter": 224}}, "deterministic": True}
    >>> tx = build_clip_transform(cfg, train=False)
    >>> hasattr(tx, "__call__")
    True
    >>> tier1_cfg = {
    ...     "data": {
    ...         "img_size": 224,
    ...         "transforms": {"tier1": {"enabled": True}},
    ...     }
    ... }
    >>> tx_train = build_clip_transform(tier1_cfg, train=True)
    >>> hasattr(tx_train, "__call__")
    True
    """
    cfg_dict = cfg if isinstance(cfg, dict) else vars(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict) if isinstance(cfg_dict, dict) else {}
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    img_size = int(data_cfg.get("img_size", 224))
    transform_cfg = data_cfg.get("transforms", {}) if isinstance(data_cfg, dict) else {}
    if transform_cfg is None:
        transform_cfg = {}

    deterministic = bool(cfg_dict.get("deterministic", False))
    use_corruptions = bool(transform_cfg.get("use_corruptions", False))
    resize_shorter = transform_cfg.get("resize_shorter")
    center_crop = bool(transform_cfg.get("center_crop", False))
    jpeg_quality = transform_cfg.get("jpeg_quality")
    blur_sigma = transform_cfg.get("blur_sigma")

    tier1_cfg = transform_cfg.get("tier1", {}) if isinstance(transform_cfg, dict) else {}
    if not isinstance(tier1_cfg, dict):
        tier1_cfg = {}
    tier1_enabled = bool(tier1_cfg.get("enabled", False)) and bool(train)

    mean, std = _get_model_norm(cfg_dict)

    steps: list[Callable[[Any], Any]] = []

    # Tier-1 runs BEFORE the terminal resize and operates on PIL images.
    # Order: RandomJPEGCompression -> RandomResizedCrop -> ColorJitter.
    # When active, RandomResizedCrop replaces the terminal Resize(img_size).
    if tier1_enabled:
        jpeg_prob = float(tier1_cfg.get("jpeg_prob", 0.7))
        jpeg_quality_range = tier1_cfg.get("jpeg_quality_range", [60, 95])
        if not isinstance(jpeg_quality_range, (list, tuple)) or len(jpeg_quality_range) != 2:
            jpeg_quality_range = [60, 95]
        steps.append(
            RandomJPEGCompression(
                p=jpeg_prob,
                quality_range=(int(jpeg_quality_range[0]), int(jpeg_quality_range[1])),
            )
        )

        resized_crop_scale = tier1_cfg.get("resized_crop_scale", [0.7, 1.0])
        if not isinstance(resized_crop_scale, (list, tuple)) or len(resized_crop_scale) != 2:
            resized_crop_scale = [0.7, 1.0]
        steps.append(
            transforms.RandomResizedCrop(
                img_size,
                scale=(float(resized_crop_scale[0]), float(resized_crop_scale[1])),
            )
        )

        cj = float(tier1_cfg.get("color_jitter", 0.1))
        steps.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    elif resize_shorter is not None:
        side = int(resize_shorter)
        steps.append(transforms.Resize((side, side)))
        if center_crop:
            steps.append(transforms.CenterCrop(img_size))
        else:
            steps.append(transforms.Resize((img_size, img_size)))
    else:
        steps.append(transforms.Resize((img_size, img_size)))

    if use_corruptions:
        quality = jpeg_quality
        sigma = blur_sigma

        if train and not deterministic:
            if quality is not None:
                q = int(quality)
                # draw from [max(1, int(0.5*q)), q] with global RNG.
                quality = int(np.random.randint(max(1, int(0.5 * q)), q + 1))
            if sigma is not None:
                sigma = float(np.random.uniform(0.0, float(sigma)))

        quality_fn = _jpeg_quality_transform(None if quality is None else int(quality))
        if quality_fn is not None:
            steps.append(transforms.Lambda(quality_fn))

        blur_fn = _gaussian_blur_transform(None if sigma is None else float(sigma))
        if blur_fn is not None:
            steps.append(transforms.Lambda(blur_fn))

    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(steps)


if __name__ == "__main__":
    sample = Image.fromarray(np.random.default_rng(0).integers(0, 256, size=(256, 256, 3), dtype=np.uint8))
    cfg = {
        "data": {
            "img_size": 224,
            "transforms": {
                "resize_shorter": 224,
                "center_crop": True,
                "use_corruptions": True,
                "jpeg_quality": 90,
                "blur_sigma": 1.5,
            },
        },
        "deterministic": True,
    }

    for train in [False, True]:
        tx = build_clip_transform(cfg, train=train)
        out = tx(sample)
        print(
            f"train={train} shape={tuple(out.shape)} min={out.min():.4f} max={out.max():.4f}"
            f" mean={out.mean():.4f} std={out.std(unbiased=False):.4f}"
        )

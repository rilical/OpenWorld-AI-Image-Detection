"""Domain-Adversarial Training (DAT) building blocks.

Provides a gradient reversal layer (GRL) and a generic wrapper that attaches a
domain-classifier head to any detector whose ``forward(images, return_features=True)``
returns a dict with a ``"features"`` key.

This module is designed to be strictly opt-in; the wrapper preserves the base
model's task outputs unchanged and only appends ``"domain_logits"`` to the dict.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer. Forward is identity; backward multiplies grad by -lambda."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to ``x`` scaled by ``lambda_``."""
    return GradReverse.apply(x, lambda_)


class DomainAdversarialWrapper(nn.Module):
    """Wrap a base detector so it also emits domain logits via a gradient-reversed head.

    The base model must accept ``forward(images, return_features=True)`` and return a
    dict containing ``"features"``. The wrapper:
      - Delegates the task forward to the base model.
      - Passes base_features through a gradient-reversal layer then a small MLP
        to produce ``domain_logits``.

    The current lambda is stored as ``self._current_lambda`` (a python float, set
    externally by the training loop via :meth:`set_lambda`). ``self.max_lambda`` is
    kept for introspection.
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        hidden_dims: Iterable[int] = (128,),
        num_domains: int = 2,
        dropout: float = 0.1,
        max_lambda: float = 0.5,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.feature_dim = int(feature_dim)
        self.num_domains = int(num_domains)
        self.max_lambda = float(max_lambda)
        self._current_lambda: float = 0.0

        # Sanity-check: confirm the base model actually exposes "features" when asked.
        # This catches mis-wired models early with a helpful error message.
        try:
            with torch.no_grad():
                was_training = self.base.training
                self.base.eval()
                probe = torch.zeros(1, 3, 224, 224)
                probe_out = self.base(probe, return_features=True)
                if was_training:
                    self.base.train()
        except Exception as exc:
            raise ValueError(
                "DomainAdversarialWrapper: base model "
                f"{type(base_model).__name__} failed the return_features=True probe: "
                f"{exc}"
            ) from exc

        if not isinstance(probe_out, dict) or "features" not in probe_out:
            raise ValueError(
                "DomainAdversarialWrapper: base model "
                f"{type(base_model).__name__} does not expose a 'features' key when "
                "called with return_features=True. DAT requires a feature-emitting "
                "detector."
            )

        probed_dim = int(probe_out["features"].shape[-1])
        if probed_dim != self.feature_dim:
            raise ValueError(
                "DomainAdversarialWrapper: feature_dim mismatch for base model "
                f"{type(base_model).__name__}: provided {self.feature_dim}, probed "
                f"{probed_dim}."
            )

        dims = list(hidden_dims)
        layers: list[nn.Module] = []
        in_dim = self.feature_dim
        for h in dims:
            layers.extend(
                [
                    nn.Linear(in_dim, int(h)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, self.num_domains))
        self.domain_head = nn.Sequential(*layers)

    def set_lambda(self, value: float) -> None:
        """Set the current gradient-reversal strength."""
        self._current_lambda = float(value)

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        """Run the base model and append domain logits via GRL + MLP.

        Returns a merged dict: all keys from the base model's output plus
        ``"domain_logits"``. ``"features"`` is kept only when ``return_features=True``
        to avoid unnecessary memory churn during ordinary training.
        """
        base_out = self.base(images, return_features=True)
        if not isinstance(base_out, dict) or "features" not in base_out:
            raise ValueError(
                "DomainAdversarialWrapper: base model "
                f"{type(self.base).__name__} returned no 'features' during forward."
            )

        feat = base_out["features"]
        feat_rev = grad_reverse(feat, self._current_lambda)
        domain_logits = self.domain_head(feat_rev)

        merged: Dict[str, Any] = dict(base_out)
        merged["domain_logits"] = domain_logits
        if not return_features:
            merged.pop("features", None)
        return merged


if __name__ == "__main__":  # pragma: no cover - smoke test
    torch.manual_seed(0)

    class _Dummy(nn.Module):
        def __init__(self, dim: int = 16) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(3, dim)
            self.head = nn.Linear(dim, 2)

        def forward(self, x, return_features: bool = False):
            feat = self.proj(self.pool(x).flatten(1))
            logits = self.head(feat)
            out = {"logits": logits, "probs": torch.softmax(logits, dim=-1)}
            if return_features:
                out["features"] = feat
            return out

    base = _Dummy(16)
    wrapper = DomainAdversarialWrapper(base, feature_dim=16, hidden_dims=(8,))
    wrapper.set_lambda(0.25)
    x = torch.randn(2, 3, 224, 224)
    out = wrapper(x)
    assert out["logits"].shape == (2, 2)
    assert out["domain_logits"].shape == (2, 2)
    assert "features" not in out
    out_feat = wrapper(x, return_features=True)
    assert "features" in out_feat
    print("domain-adversarial smoke-ok")

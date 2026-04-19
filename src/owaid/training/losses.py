"""Custom loss functions for confidence-calibrated training.

ConfidenceSeparationLoss encourages the model to output high confidence
when correct and low confidence when wrong, improving downstream abstention
without requiring any changes to the abstention mechanism itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceSeparationLoss(nn.Module):
    """Auxiliary loss: BCE(max_softmax_confidence, correctness).

    Trains the model so that its softmax confidence is predictive of
    whether it will be correct, shaping the confidence landscape for
    better downstream threshold and conformal abstention.
    """

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute confidence separation loss.

        Args:
            logits: (B, 2) raw logits.
            labels: (B,) ground truth labels.

        Returns:
            Scalar loss.
        """
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values          # (B,)
        predictions = probs.argmax(dim=-1)              # (B,)
        correct = (predictions == labels).float()       # (B,)

        # BCE: encourage confidence ~ correctness
        # Clamp to avoid log(0)
        conf_clamped = confidence.clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(conf_clamped, correct)
        return loss


class SGFNetLoss(nn.Module):
    """Combined loss for SGF-Net training.

    L_total = L_CE + lambda_conf * L_confidence_separation
    """

    def __init__(self, lambda_conf: float = 0.3) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.conf_loss = ConfidenceSeparationLoss()
        self.lambda_conf = lambda_conf

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels) + self.lambda_conf * self.conf_loss(logits, labels)

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, positive_weight: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("positive_weight", torch.tensor([positive_weight]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.positive_weight
        )
        probability = logits.sigmoid()
        dimensions = (1, 2, 3)
        intersection = (probability * target).sum(dimensions)
        denominator = probability.sum(dimensions) + target.sum(dimensions)
        dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
        return 0.5 * bce + 0.5 * dice_loss

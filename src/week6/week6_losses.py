"""Loss functions for Week 6 rare-class damage segmentation experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Multiclass Dice loss, ignoring background by default."""

    def __init__(self, include_background: bool = False, eps: float = 1e-7) -> None:
        super().__init__()
        self.include_background = include_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probabilities * target_one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Multiclass focal loss for class imbalance."""

    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32) if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets.long(), weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1.0 - pt).pow(self.gamma) * ce_loss
        if self.reduction == "sum":
            return focal.sum()
        if self.reduction == "none":
            return focal
        return focal.mean()


class TverskyLoss(nn.Module):
    """Tversky loss; higher beta penalizes false negatives more strongly."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        include_background: bool = False,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.include_background = include_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        true_positive = (probabilities * target_one_hot).sum(dim=dims)
        false_positive = (probabilities * (1.0 - target_one_hot)).sum(dim=dims)
        false_negative = ((1.0 - probabilities) * target_one_hot).sum(dim=dims)
        score = (true_positive + self.eps) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + self.eps
        )
        if not self.include_background:
            score = score[1:]
        return 1.0 - score.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for rare and hard damage classes."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75) -> None:
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.tversky(logits, targets).pow(self.gamma)


class CombinedLoss(nn.Module):
    """Weighted sum of multiple loss modules."""

    def __init__(self, losses: list[nn.Module], weights: list[float] | None = None) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)
        if len(self.losses) != len(self.weights):
            raise ValueError("losses and weights must have the same length")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = logits.new_tensor(0.0)
        for loss_fn, weight in zip(self.losses, self.weights):
            total = total + weight * loss_fn(logits, targets)
        return total


def build_loss(name: str, class_weights: list[float] | torch.Tensor | None = None) -> nn.Module:
    """Factory used by experiment scripts."""
    normalized = name.lower().replace("-", "_")
    weights = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    if normalized in {"cross_entropy", "ce", "weighted_cross_entropy", "weighted_ce"}:
        return nn.CrossEntropyLoss(weight=weights)
    if normalized == "dice":
        return DiceLoss()
    if normalized == "focal_loss" or normalized == "focal":
        return FocalLoss(alpha=weights)
    if normalized == "tversky_loss" or normalized == "tversky":
        return TverskyLoss()
    if normalized == "focal_tversky":
        return FocalTverskyLoss()
    if normalized in {"cross_entropy_dice", "ce_dice", "combined"}:
        return CombinedLoss([nn.CrossEntropyLoss(weight=weights), DiceLoss()], [1.0, 1.0])
    if normalized in {"weighted_cross_entropy_dice", "weighted_ce_dice", "ce_dice_0_7_0_3"}:
        return CombinedLoss([nn.CrossEntropyLoss(weight=weights), DiceLoss()], [0.7, 0.3])
    raise ValueError(f"Unknown Week 6 loss: {name}")

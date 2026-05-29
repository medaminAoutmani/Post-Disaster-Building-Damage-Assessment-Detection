"""Ordinal and calibration-oriented losses and metrics for Week 13."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def labels_to_coral_levels(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Convert ordinal labels into cumulative CORAL targets."""
    thresholds = torch.arange(1, num_classes, device=labels.device).view(1, -1)
    return (labels.view(-1, 1) >= thresholds).float()


def coral_logits_to_classes(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Decode cumulative ordinal logits into class IDs."""
    return (torch.sigmoid(logits) >= threshold).long().sum(dim=1)


def coral_logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Approximate four-class probabilities from monotonic cumulative probabilities."""
    cumulative = torch.sigmoid(logits)
    p0 = 1.0 - cumulative[:, 0]
    p1 = cumulative[:, 0] - cumulative[:, 1]
    p2 = cumulative[:, 1] - cumulative[:, 2]
    p3 = cumulative[:, 2]
    probs = torch.stack([p0, p1, p2, p3], dim=1).clamp_min(0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-7)


class CoralLoss(nn.Module):
    """COnsistent RAnk Logits loss for ordered damage labels."""

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        levels = labels_to_coral_levels(labels, num_classes=logits.shape[1] + 1)
        return F.binary_cross_entropy_with_logits(logits, levels, pos_weight=self.pos_weight)


class EarthMoverDistanceLoss(nn.Module):
    """Distance-aware loss that penalizes ordinally distant mistakes more strongly."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target = F.one_hot(labels, num_classes=logits.shape[1]).float()
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(target, dim=1)
        return torch.mean(torch.square(pred_cdf - target_cdf))


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    """Compute multiclass expected calibration error."""
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels).float()
    ece = probs.new_tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            gap = torch.abs(confidences[mask].mean() - correct[mask].mean())
            ece += mask.float().mean() * gap
    return float(ece.item())


def mean_severity_distance(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Average absolute ordinal class distance."""
    return float(torch.abs(predictions.long() - labels.long()).float().mean().item())


def damaged_auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC for damaged-vs-no-damage detection using rank statistics."""
    scores = 1.0 - probs[:, 0]
    targets = (labels > 0).long()
    positives = targets.sum()
    negatives = targets.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, targets.numel() + 1, device=scores.device).float()
    positive_rank_sum = ranks[targets == 1].sum()
    auc = (positive_rank_sum - positives.float() * (positives.float() + 1.0) / 2.0) / (positives.float() * negatives.float())
    return float(auc.item())

"""Week 9 multi-task metrics."""

from __future__ import annotations

import torch

from week6.week6_metrics import confusion_matrix_from_logits, metrics_from_confusion_matrix
from week9_losses import damage_confusion_excluding_background


def binary_confusion_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=1)
    targets = targets.long()
    valid = (targets >= 0) & (targets <= 1)
    bins = 2 * targets[valid].reshape(-1) + predictions[valid].reshape(-1)
    return torch.bincount(bins, minlength=4).reshape(2, 2)


def binary_metrics(confusion: torch.Tensor, prefix: str, eps: float = 1e-7) -> dict[str, float]:
    confusion = confusion.float()
    true_positive = confusion[1, 1]
    false_positive = confusion[0, 1]
    false_negative = confusion[1, 0]
    dice = (2.0 * true_positive + eps) / (2.0 * true_positive + false_positive + false_negative + eps)
    iou = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)
    accuracy = (confusion.diag().sum() + eps) / (confusion.sum() + eps)
    return {
        f"{prefix}_dice": float(dice.item()),
        f"{prefix}_iou": float(iou.item()),
        f"{prefix}_pixel_accuracy": float(accuracy.item()),
    }


def multitask_metrics(
    pre_confusion: torch.Tensor,
    post_confusion: torch.Tensor,
    damage_confusion: torch.Tensor,
    damage_building_confusion: torch.Tensor,
) -> dict[str, float]:
    metrics = {}
    metrics.update(binary_metrics(pre_confusion, "pre_building"))
    metrics.update(binary_metrics(post_confusion, "post_building"))
    metrics.update({f"damage_{key}": value for key, value in metrics_from_confusion_matrix(damage_confusion).items()})
    building_damage_metrics = metrics_from_confusion_matrix(damage_building_confusion)
    metrics.update({f"building_only_damage_{key}": value for key, value in building_damage_metrics.items()})
    return metrics


__all__ = [
    "binary_confusion_from_logits",
    "confusion_matrix_from_logits",
    "damage_confusion_excluding_background",
    "multitask_metrics",
]

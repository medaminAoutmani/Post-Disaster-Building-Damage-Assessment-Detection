"""Week 7 metric aliases and temporal helpers."""

from __future__ import annotations

import torch

from week6.week6_metrics import (
    CLASS_NAMES,
    batch_boundary_iou,
    confusion_matrix_from_logits,
    metrics_from_confusion_matrix,
    save_confusion_matrix_csv,
    save_per_class_metrics,
)


def temporal_change_energy(pre_image: torch.Tensor, post_image: torch.Tensor) -> torch.Tensor:
    """Mean absolute normalized RGB change per sample."""
    return torch.abs(post_image - pre_image).mean(dim=(1, 2, 3))

__all__ = [
    "CLASS_NAMES",
    "batch_boundary_iou",
    "confusion_matrix_from_logits",
    "metrics_from_confusion_matrix",
    "save_confusion_matrix_csv",
    "save_per_class_metrics",
    "temporal_change_energy",
]


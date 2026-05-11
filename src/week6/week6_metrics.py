"""Week 6 segmentation metrics and confusion-matrix helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.nn import functional as F


CLASS_NAMES = ["background", "no_damage", "minor_damage", "major_damage", "destroyed"]


def confusion_matrix_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=1)
    valid = (targets >= 0) & (targets < num_classes)
    bins = num_classes * targets[valid].reshape(-1) + predictions[valid].reshape(-1)
    return torch.bincount(bins, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(confusion: torch.Tensor, eps: float = 1e-7) -> dict[str, float]:
    confusion = confusion.float()
    true_positive = torch.diag(confusion)
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    dice = (2.0 * true_positive + eps) / (2.0 * true_positive + false_positive + false_negative + eps)
    iou = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)
    precision = (true_positive + eps) / (true_positive + false_positive + eps)
    recall = (true_positive + eps) / (true_positive + false_negative + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    pixel_accuracy = (true_positive.sum() + eps) / (confusion.sum() + eps)
    class_frequency = confusion.sum(dim=1) / (confusion.sum() + eps)
    frequency_weighted_iou = (class_frequency[1:] * iou[1:]).sum()
    rare_class_recall = recall[2:].mean()

    metrics = {
        "mean_dice": float(dice[1:].mean().item()),
        "mean_iou": float(iou[1:].mean().item()),
        "frequency_weighted_iou": float(frequency_weighted_iou.item()),
        "rare_class_recall": float(rare_class_recall.item()),
        "pixel_accuracy": float(pixel_accuracy.item()),
        "macro_f1": float(f1[1:].mean().item()),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f"dice_{class_name}"] = float(dice[index].item())
        metrics[f"iou_{class_name}"] = float(iou[index].item())
        metrics[f"f1_{class_name}"] = float(f1[index].item())
    return metrics


def dice_score(logits: torch.Tensor, targets: torch.Tensor, class_index: int, eps: float = 1e-7) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=1)
    pred_mask = predictions == class_index
    target_mask = targets == class_index
    intersection = (pred_mask & target_mask).sum()
    denominator = pred_mask.sum() + target_mask.sum()
    return (2.0 * intersection.float() + eps) / (denominator.float() + eps)


def mean_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 5, eps: float = 1e-7) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=1)
    values = []
    for class_index in range(1, num_classes):
        pred_mask = predictions == class_index
        target_mask = targets == class_index
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        values.append((intersection + eps) / (union + eps))
    return torch.stack(values).mean()


def boundary_iou(logits: torch.Tensor, targets: torch.Tensor, class_index: int, kernel_size: int = 3) -> torch.Tensor:
    """Approximate boundary IoU using max-pool morphological edges."""
    predictions = (torch.argmax(logits, dim=1) == class_index).float().unsqueeze(1)
    target = (targets == class_index).float().unsqueeze(1)
    padding = kernel_size // 2
    pred_edge = F.max_pool2d(predictions, kernel_size, stride=1, padding=padding) - predictions
    target_edge = F.max_pool2d(target, kernel_size, stride=1, padding=padding) - target
    intersection = ((pred_edge > 0) & (target_edge > 0)).sum().float()
    union = ((pred_edge > 0) | (target_edge > 0)).sum().float()
    return (intersection + 1e-7) / (union + 1e-7)


def batch_boundary_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 5) -> dict[str, float]:
    values = {}
    for class_index, class_name in enumerate(CLASS_NAMES[:num_classes]):
        if class_index == 0:
            continue
        values[f"boundary_iou_{class_name}"] = float(boundary_iou(logits, targets, class_index).item())
    values["mean_boundary_iou"] = float(sum(values.values()) / max(1, len(values)))
    return values


def precision_recall_points(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_indices: list[int] | None = None,
    thresholds: list[float] | None = None,
) -> list[dict[str, float | str]]:
    """Return thresholded precision/recall rows for selected rare classes."""
    class_indices = class_indices or [2, 3, 4]
    thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    probabilities = torch.softmax(logits.detach().cpu(), dim=1)
    targets = targets.detach().cpu()
    rows: list[dict[str, float | str]] = []
    for class_index in class_indices:
        class_name = CLASS_NAMES[class_index]
        truth = targets == class_index
        scores = probabilities[:, class_index] 
        for threshold in thresholds:
            predicted = scores >= threshold
            true_positive = (predicted & truth).sum().float()
            false_positive = (predicted & ~truth).sum().float()
            false_negative = (~predicted & truth).sum().float()
            precision = (true_positive + 1e-7) / (true_positive + false_positive + 1e-7)
            recall = (true_positive + 1e-7) / (true_positive + false_negative + 1e-7)
            rows.append(
                {
                    "class": class_name,
                    "threshold": threshold,
                    "precision": float(precision.item()),
                    "recall": float(recall.item()),
                }
            )
    return rows


def save_confusion_matrix_csv(confusion: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth/prediction", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, confusion.tolist()):
            writer.writerow([class_name, *row])


def save_per_class_metrics(metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "dice", "iou", "f1"])
        for class_name in CLASS_NAMES:
            writer.writerow(
                [
                    class_name,
                    metrics.get(f"dice_{class_name}", 0.0),
                    metrics.get(f"iou_{class_name}", 0.0),
                    metrics.get(f"f1_{class_name}", 0.0),
                ]
            )


def save_precision_recall_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "threshold", "precision", "recall"])
        writer.writeheader()
        writer.writerows(rows)

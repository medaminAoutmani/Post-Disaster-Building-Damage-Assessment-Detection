"""Train the Week 5 multiclass xBD damage segmentation model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from week2_dataset import XBDChangeDataset, read_split_file
from week3_dataset_statistics import collect_dataset_statistics, save_metrics_csv
from week3_train import limit_dataloader, save_history_csv, save_json, save_training_config_yaml
from week5_model import DamageResNet34UNet


CLASS_NAMES = ["background", "no_damage", "minor_damage", "major_damage", "destroyed"]
CLASS_COLORS_RGB = np.array(
    [
        [0, 0, 0],
        [0, 180, 0],
        [255, 230, 0],
        [255, 140, 0],
        [220, 0, 0],
    ],
    dtype=np.uint8,
)
DEFAULT_CLASS_WEIGHTS = [1.0, 0.2, 1.5, 1.7, 1.8]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIX_CHANNEL_MEAN = IMAGENET_MEAN + IMAGENET_MEAN
SIX_CHANNEL_STD = IMAGENET_STD + IMAGENET_STD


def get_week5_transforms(image_size: int = 512, train: bool = True) -> A.Compose:
    """Build transforms with ImageNet normalization for the pretrained encoder."""
    transforms: list[A.BasicTransform] = [A.Resize(image_size, image_size)]
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.OneOf([A.Blur(blur_limit=3, p=1.0), A.GaussNoise(p=1.0)], p=0.25),
            ]
        )
    transforms.extend([A.Normalize(mean=SIX_CHANNEL_MEAN, std=SIX_CHANNEL_STD), ToTensorV2()])
    return A.Compose(transforms)


def build_week5_dataloaders(
    data_dir: Path,
    split_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create multiclass DataLoaders."""
    train_ids = read_split_file(split_dir / "train.txt")
    val_ids = read_split_file(split_dir / "val.txt")
    test_ids = read_split_file(split_dir / "test.txt")

    train_dataset = XBDChangeDataset(
        data_dir,
        train_ids,
        split="train",
        transform=get_week5_transforms(image_size, train=True),
        target_mode="multiclass",
        filter_empty=True,
    )
    val_dataset = XBDChangeDataset(
        data_dir,
        val_ids,
        split="train",
        transform=get_week5_transforms(image_size, train=False),
        target_mode="multiclass",
        filter_empty=True,
    )
    test_dataset = XBDChangeDataset(
        data_dir,
        test_ids,
        split="train",
        transform=get_week5_transforms(image_size, train=False),
        target_mode="multiclass",
        filter_empty=True,
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def build_week5_overfit_dataloaders(
    data_dir: Path,
    split_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    sample_count: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation loaders over the same tiny clean sample set."""
    train_ids = read_split_file(split_dir / "train.txt")
    dataset = XBDChangeDataset(
        data_dir,
        train_ids,
        split="train",
        transform=get_week5_transforms(image_size, train=False),
        target_mode="multiclass",
        filter_empty=True,
    )
    subset = Subset(dataset, list(range(min(sample_count, len(dataset)))))
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


class CrossEntropyDiceLoss(nn.Module):
    """Weighted cross entropy plus multiclass Dice loss."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = nn.functional.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probabilities * target_one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.eps) / (denominator + self.eps)
        dice_loss = 1.0 - dice_per_class[1:].mean()
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def confusion_matrix_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Compute a multiclass confusion matrix where rows are truth and columns are predictions."""
    predictions = torch.argmax(logits, dim=1)
    valid = (targets >= 0) & (targets < num_classes)
    bins = num_classes * targets[valid].reshape(-1) + predictions[valid].reshape(-1)
    return torch.bincount(bins, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(confusion: torch.Tensor, eps: float = 1e-7) -> dict[str, float]:
    """Compute per-class Dice/IoU plus mean Dice, pixel accuracy, and macro F1."""
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

    metrics = {
        "mean_dice": float(dice[1:].mean().item()),
        "mean_iou": float(iou[1:].mean().item()),
        "pixel_accuracy": float(pixel_accuracy.item()),
        "macro_f1": float(f1[1:].mean().item()),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f"dice_{class_name}"] = float(dice[index].item())
        metrics[f"iou_{class_name}"] = float(iou[index].item())
    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], torch.Tensor]:
    """Train for one epoch and return loss, metrics, and confusion matrix."""
    model.train()
    total_loss = 0.0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        confusion += confusion_matrix_from_logits(logits.detach().cpu(), masks.detach().cpu())

    return total_loss / len(dataloader), metrics_from_confusion_matrix(confusion), confusion


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], torch.Tensor]:
    """Evaluate and return loss, metrics, and confusion matrix."""
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()
        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += float(loss.item())
        confusion += confusion_matrix_from_logits(logits.cpu(), masks.cpu())

    return total_loss / len(dataloader), metrics_from_confusion_matrix(confusion), confusion


def tensor_image_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized 6-channel tensor to a displayable post-disaster RGB image."""
    image = image_tensor.detach().cpu().numpy()
    post_image = np.transpose(image[3:6], (1, 2, 0))
    post_image = post_image * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
    return np.clip(post_image * 255.0, 0, 255).astype(np.uint8)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Map class-id mask to RGB colors."""
    return CLASS_COLORS_RGB[np.clip(mask, 0, len(CLASS_NAMES) - 1)]


def create_multiclass_overlay(input_image: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay non-background predicted damage classes on the post-disaster image."""
    overlay = input_image.copy()
    color_mask = mask_to_color(pred_mask)
    foreground = pred_mask > 0
    overlay[foreground] = ((1.0 - alpha) * overlay[foreground] + alpha * color_mask[foreground]).astype(np.uint8)
    return overlay


def per_sample_mean_dice(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Compute mean foreground Dice for one prediction."""
    scores = []
    for class_id in range(1, len(CLASS_NAMES)):
        pred = pred_mask == class_id
        target = target_mask == class_id
        intersection = np.logical_and(pred, target).sum()
        denominator = pred.sum() + target.sum()
        scores.append((2.0 * intersection + eps) / (denominator + eps))
    return float(np.mean(scores))


def save_prediction_examples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    predictions_dir: Path,
    visualizations_dir: Path,
    max_examples: int = 8,
) -> list[dict[str, str | float]]:
    """Save multiclass prediction panels and return per-sample records."""
    predictions_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    for directory in [
        predictions_dir / "best_examples",
        predictions_dir / "difficult_scenes",
        predictions_dir / "failure_cases",
        visualizations_dir / "overlays",
        visualizations_dir / "masks",
        visualizations_dir / "comparisons",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0
    records: list[dict[str, str | float]] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            masks = batch["mask"].cpu().numpy().astype(np.uint8)
            sample_ids = batch["sample_id"]

            for index, sample_id in enumerate(sample_ids):
                input_image = tensor_image_to_uint8(batch["image"][index])
                gt_mask = masks[index]
                pred_mask = predictions[index]
                dice = per_sample_mean_dice(pred_mask, gt_mask)
                if dice >= 0.55:
                    category = "best_examples"
                elif dice >= 0.25:
                    category = "difficult_scenes"
                else:
                    category = "failure_cases"

                gt_color = mask_to_color(gt_mask)
                pred_color = mask_to_color(pred_mask)
                overlay = create_multiclass_overlay(input_image, pred_mask)
                panel = np.concatenate([input_image, gt_color, pred_color, overlay], axis=1)

                category_dir = predictions_dir / category
                cv2.imwrite(str(category_dir / f"{sample_id}_input.png"), cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(category_dir / f"{sample_id}_gt_mask_color.png"), cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(category_dir / f"{sample_id}_pred_mask_color.png"), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(category_dir / f"{sample_id}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(category_dir / f"{sample_id}_input_gt_pred_overlay.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                cv2.imwrite(
                    str(visualizations_dir / "comparisons" / f"{sample_id}_input_gt_pred_overlay.png"),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(str(visualizations_dir / "masks" / f"{sample_id}_gt_mask_color.png"), cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(visualizations_dir / "masks" / f"{sample_id}_pred_mask_color.png"), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(visualizations_dir / "overlays" / f"{sample_id}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                records.append({"sample_id": sample_id, "mean_dice": dice, "category": category})

                saved += 1
                if saved >= max_examples:
                    return records

    return records


def save_prediction_records(records: list[dict[str, str | float]], output_path: Path) -> None:
    """Save qualitative result categories and per-sample mean Dice scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "mean_dice", "category"])
        writer.writeheader()
        writer.writerows(records)


def save_confusion_matrix_csv(confusion: torch.Tensor, output_path: Path) -> None:
    """Save a confusion matrix CSV where rows are truth and columns are predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, confusion.tolist()):
            writer.writerow([class_name, *row])


def save_confusion_matrix_plot(confusion: torch.Tensor, output_path: Path) -> None:
    """Save a normalized confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion.float().numpy()
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, np.maximum(row_sums, 1.0))

    plt.figure(figsize=(8, 6))
    plt.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Row-normalized frequency")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Week 5 multiclass confusion matrix")
    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            plt.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_week5_training_curves(history: list[dict[str, float | int]], output_dir: Path) -> None:
    """Save loss, mean Dice, and major/destroyed Dice curves."""
    if not history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(row["train_loss"]) for row in history], label="Train loss")
    plt.plot(epochs, [float(row["val_loss"]) for row in history], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(row["train_dice"]) for row in history], label="Train mean Dice")
    plt.plot(epochs, [float(row["val_dice"]) for row in history], label="Val mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Mean foreground Dice")
    plt.title("Training and validation mean Dice")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_dice_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(row["val_dice_major_damage"]) for row in history], label="Major damage")
    plt.plot(epochs, [float(row["val_dice_destroyed"]) for row in history], label="Destroyed")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Scientifically important damage-class Dice")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "major_destroyed_dice_curve.png", dpi=200)
    plt.close()


def save_failure_analysis_template(output_path: Path) -> None:
    """Create a reusable Week 5 failure-analysis note file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """# Week 5 Failure Case Analysis

Use the color prediction panels in `results/week5/predictions/`, the per-class metrics in
`results/week5/metrics/final_metrics.json`, and the confusion matrices in
`results/week5/confusion_matrices/`.

## Tiny Buildings

- Observation:
- Typical failure:

## Smoke and Haze

- Observation:
- False positives:

## Flood Reflections

- Observation:
- Confused classes:

## Dense Urban Areas

- Observation:
- Merged buildings:

## Major vs Destroyed

- Observation:
- Common confusion:

## Research Notes

- Compare baseline multiclass, class-weighted loss, frozen encoder, unfrozen encoder, and different encoder/decoder learning rates.
""",
        encoding="utf-8",
    )


def build_optimizer(
    model: nn.Module,
    encoder_lr: float,
    decoder_lr: float,
) -> torch.optim.Optimizer:
    """Use separate learning rates for encoder and decoder parameters."""
    encoder_params = []
    decoder_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("stem", "encoder")):
            encoder_params.append(parameter)
        else:
            decoder_params.append(parameter)
    parameter_groups = []
    if encoder_params:
        parameter_groups.append({"params": encoder_params, "lr": encoder_lr})
    if decoder_params:
        parameter_groups.append({"params": decoder_params, "lr": decoder_lr})
    return torch.optim.AdamW(parameter_groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Week 5 multiclass xBD damage segmentation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Use only this many training samples.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Use only this many validation samples.")
    parser.add_argument("--overfit-samples", type=int, default=None, help="Debug by training/validating on N clean samples.")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "week5")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--num-predictions", type=int, default=18, help="Validation predictions to save when validation improves.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained ResNet34 weights.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only the decoder and segmentation head.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weighting in CrossEntropyLoss.")
    parser.add_argument("--class-weights", type=float, nargs=5, default=DEFAULT_CLASS_WEIGHTS)
    args = parser.parse_args()

    args.checkpoint_dir = args.checkpoint_dir or args.results_dir / "checkpoints"
    args.prediction_dir = args.prediction_dir or args.results_dir / "predictions"
    metrics_dir = args.results_dir / "metrics"
    config_dir = args.results_dir / "config"
    visualizations_dir = args.results_dir / "visualizations"
    confusion_dir = args.results_dir / "confusion_matrices"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.overfit_samples is not None and args.overfit_samples > 0:
        train_loader, val_loader = build_week5_overfit_dataloaders(
            args.data_dir,
            args.split_dir,
            args.image_size,
            args.batch_size,
            args.num_workers,
            args.overfit_samples,
        )
    else:
        train_loader, val_loader, _ = build_week5_dataloaders(
            args.data_dir,
            args.split_dir,
            args.image_size,
            args.batch_size,
            args.num_workers,
        )
        train_loader = limit_dataloader(train_loader, args.max_train_samples, shuffle=True)
        val_loader = limit_dataloader(val_loader, args.max_val_samples, shuffle=False)

    model = DamageResNet34UNet(pretrained=not args.no_pretrained, freeze_encoder=args.freeze_encoder).to(device)
    optimizer = build_optimizer(model, args.encoder_lr, args.decoder_lr)
    class_weights = None if args.no_class_weights else torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    criterion = CrossEntropyDiceLoss(class_weights=class_weights)

    for directory in [args.checkpoint_dir, args.prediction_dir, metrics_dir, config_dir, visualizations_dir, confusion_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "data_dir": str(args.data_dir),
        "split_dir": str(args.split_dir),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "encoder_learning_rate": args.encoder_lr,
        "decoder_learning_rate": args.decoder_lr,
        "loss": "CrossEntropy + multiclass Dice",
        "class_names": CLASS_NAMES,
        "class_weights": None if args.no_class_weights else args.class_weights,
        "model": "DamageResNet34UNet",
        "encoder": "ResNet34",
        "output_channels": 5,
        "pretrained": not args.no_pretrained,
        "freeze_encoder": args.freeze_encoder,
        "normalization": "ImageNet mean/std duplicated for pre and post RGB",
        "num_workers": args.num_workers,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "overfit_samples": args.overfit_samples,
        "target_mode": "multiclass",
        "device": str(device),
    }
    save_json(config, config_dir / "training_config.json")
    save_training_config_yaml(config, config_dir / "training_config.yaml")
    train_sample_counts, train_class_counts = collect_dataset_statistics(args.data_dir, "train")
    save_metrics_csv(train_sample_counts, train_class_counts, metrics_dir / "dataset_statistics.csv")
    save_failure_analysis_template(args.results_dir / "failure_analysis.md")

    best_val_dice = -1.0
    best_metrics: dict[str, float | int] = {}
    best_confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    history: list[dict[str, float | int]] = []
    print(f"device={device}")
    print(f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)}")
    print(f"model=DamageResNet34UNet classes={len(CLASS_NAMES)} pretrained={not args.no_pretrained}")
    if args.overfit_samples is not None and args.overfit_samples > 0:
        print("overfit_test=true multiclass_target=high_mean_foreground_dice")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, val_confusion = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_mean_dice={train_metrics['mean_dice']:.4f} "
            f"val_loss={val_loss:.4f} val_mean_dice={val_metrics['mean_dice']:.4f} "
            f"val_major_dice={val_metrics['dice_major_damage']:.4f} "
            f"val_destroyed_dice={val_metrics['dice_destroyed']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_pixel_acc={val_metrics['pixel_accuracy']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_metrics["mean_dice"],
            "val_dice": val_metrics["mean_dice"],
            "train_mean_iou": train_metrics["mean_iou"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_dice_no_damage": val_metrics["dice_no_damage"],
            "val_dice_minor_damage": val_metrics["dice_minor_damage"],
            "val_dice_major_damage": val_metrics["dice_major_damage"],
            "val_dice_destroyed": val_metrics["dice_destroyed"],
        }
        history.append(epoch_record)
        save_history_csv(history, metrics_dir / "training_log.csv")
        plot_week5_training_curves(history, visualizations_dir)

        if val_metrics["mean_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_dice"]
            best_confusion = val_confusion
            best_metrics = {
                "best_epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            save_json(best_metrics, metrics_dir / "final_metrics.json")
            save_confusion_matrix_csv(best_confusion, confusion_dir / "confusion_matrix.csv")
            save_confusion_matrix_plot(best_confusion, confusion_dir / "confusion_matrix.png")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "class_names": CLASS_NAMES,
                    "model": "DamageResNet34UNet",
                    "pretrained": not args.no_pretrained,
                },
                args.checkpoint_dir / "week5_resnet34_unet_multiclass_best.pt",
            )
            records = save_prediction_examples(
                model,
                val_loader,
                device,
                args.prediction_dir,
                visualizations_dir,
                max_examples=args.num_predictions,
            )
            save_prediction_records(records, metrics_dir / "prediction_records.csv")

    save_history_csv(history, metrics_dir / "training_log.csv")
    plot_week5_training_curves(history, visualizations_dir)
    if best_metrics:
        save_json(best_metrics, metrics_dir / "final_metrics.json")
        save_confusion_matrix_csv(best_confusion, confusion_dir / "confusion_matrix.csv")
        save_confusion_matrix_plot(best_confusion, confusion_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()

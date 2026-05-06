"""Train the Week 3 binary U-Net segmentation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from week2_dataset import XBDChangeDataset, build_dataloaders, get_transforms, read_split_file
from week3_model import UNet


class BCEDiceLoss(nn.Module):
    """BCE for pixel classification plus Dice loss for mask overlap."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-7) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def limit_dataloader(dataloader: DataLoader, max_samples: int | None, shuffle: bool) -> DataLoader:
    """Return a DataLoader backed by only the first max_samples items."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataloader.dataset):
        return dataloader

    subset = Subset(dataloader.dataset, list(range(max_samples)))
    return DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=shuffle,
        num_workers=dataloader.num_workers,
        pin_memory=getattr(dataloader, "pin_memory", False),
    )


def binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> dict[str, float]:
    """Compute Dice, IoU, precision, recall, and F1 from raw logits."""
    predictions = (torch.sigmoid(logits) > 0.5).float()
    true_positive = (predictions * targets).sum(dim=(1, 2, 3))
    false_positive = (predictions * (1.0 - targets)).sum(dim=(1, 2, 3))
    false_negative = ((1.0 - predictions) * targets).sum(dim=(1, 2, 3))

    dice = (2.0 * true_positive + eps) / (2.0 * true_positive + false_positive + false_negative + eps)
    iou = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)
    precision = (true_positive + eps) / (true_positive + false_positive + eps)
    recall = (true_positive + eps) / (true_positive + false_negative + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(dice.mean().item()),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Train for one epoch and return average loss and segmentation metrics."""
    model.train()
    total_loss = 0.0
    total_metrics: dict[str, float] = {}

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        metrics = binary_metrics_from_logits(logits.detach(), masks)
        for metric_name, value in metrics.items():
            total_metrics[metric_name] = total_metrics.get(metric_name, 0.0) + value

    return total_loss / len(dataloader), {name: value / len(dataloader) for name, value in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate and return average loss and segmentation metrics."""
    model.eval()
    total_loss = 0.0
    total_metrics: dict[str, float] = {}

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += float(loss.item())
        metrics = binary_metrics_from_logits(logits, masks)
        for metric_name, value in metrics.items():
            total_metrics[metric_name] = total_metrics.get(metric_name, 0.0) + value

    return total_loss / len(dataloader), {name: value / len(dataloader) for name, value in total_metrics.items()}


def tensor_image_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a 6-channel pre/post tensor to a displayable post-disaster RGB image."""
    image = image_tensor.detach().cpu().numpy()
    post_image = np.transpose(image[3:6], (1, 2, 0))
    return np.clip(post_image * 255.0, 0, 255).astype(np.uint8)


def save_prediction_examples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_examples: int = 8,
) -> None:
    """Save input, ground-truth, and prediction panels from validation samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            logits = model(images)
            predictions = (torch.sigmoid(logits) > 0.5).float().cpu()
            masks = batch["mask"].cpu()
            sample_ids = batch["sample_id"]

            for index, sample_id in enumerate(sample_ids):
                input_image = tensor_image_to_uint8(batch["image"][index])
                gt_mask = (masks[index, 0].numpy() * 255).astype(np.uint8)
                pred_mask = (predictions[index, 0].numpy() * 255).astype(np.uint8)

                panel = np.concatenate(
                    [
                        input_image,
                        cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB),
                        cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB),
                    ],
                    axis=1,
                )
                cv2.imwrite(str(output_dir / f"{sample_id}_input_gt_pred.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_dir / f"{sample_id}_input.png"), cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_dir / f"{sample_id}_gt_mask.png"), gt_mask)
                cv2.imwrite(str(output_dir / f"{sample_id}_pred_mask.png"), pred_mask)

                saved += 1
                if saved >= max_examples:
                    return


def build_overfit_dataloaders(
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
        transform=get_transforms(image_size, train=False),
        target_mode="binary",
        filter_empty=True,
    )
    subset = Subset(dataset, list(range(min(sample_count, len(dataset)))))
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Week 3 binary U-Net segmentation baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Use only this many training samples.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Use only this many validation samples.")
    parser.add_argument("--overfit-samples", type=int, default=None, help="Debug by training/validating on N clean samples.")
    parser.add_argument("--small-model", action="store_true", help="Use a smaller U-Net for faster CPU experiments.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs") / "checkpoints")
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs") / "predictions")
    parser.add_argument("--num-predictions", type=int, default=8, help="Validation predictions to save when validation improves.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.overfit_samples is not None and args.overfit_samples > 0:
        train_loader, val_loader = build_overfit_dataloaders(
            args.data_dir,
            args.split_dir,
            args.image_size,
            args.batch_size,
            args.num_workers,
            args.overfit_samples,
        )
    else:
        train_loader, val_loader, _ = build_dataloaders(
            data_dir=args.data_dir,
            split_dir=args.split_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_mode="binary",
        )
        train_loader = limit_dataloader(train_loader, args.max_train_samples, shuffle=True)
        val_loader = limit_dataloader(val_loader, args.max_val_samples, shuffle=False)

    features = (16, 32, 64) if args.small_model else (32, 64, 128, 256)
    model = UNet(in_channels=6, out_channels=1, features=features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = BCEDiceLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0
    print(f"device={device}")
    print(f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)}")
    if args.overfit_samples is not None and args.overfit_samples > 0:
        print("overfit_test=true target_dice=>0.9000")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_dice={train_metrics['dice']:.4f} train_iou={train_metrics['iou']:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                args.checkpoint_dir / "week3_unet_binary_best.pt",
            )
            save_prediction_examples(
                model,
                val_loader,
                device,
                args.prediction_dir / f"epoch_{epoch:03d}",
                max_examples=args.num_predictions,
            )


if __name__ == "__main__":
    main()

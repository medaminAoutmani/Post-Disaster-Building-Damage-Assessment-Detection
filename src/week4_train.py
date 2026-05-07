"""Train the Week 4 ResNet34-encoder U-Net segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")

from week2_dataset import XBDChangeDataset, build_dataloaders, read_split_file
from week3_dataset_statistics import collect_dataset_statistics, save_metrics_csv
from week3_train import (
    BCEDiceLoss,
    binary_metrics_from_logits,
    evaluate,
    limit_dataloader,
    per_sample_dice_from_logits,
    plot_training_curves,
    save_history_csv,
    save_json,
    save_prediction_records,
    save_training_config_yaml,
    train_one_epoch,
)
from week4_model import ResNet34UNet


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIX_CHANNEL_MEAN = IMAGENET_MEAN + IMAGENET_MEAN
SIX_CHANNEL_STD = IMAGENET_STD + IMAGENET_STD


def get_week4_transforms(image_size: int = 512, train: bool = True) -> A.Compose:
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


def build_week4_dataloaders(
    data_dir: Path,
    split_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create Week 4 DataLoaders with pretrained-encoder normalization."""
    train_ids = read_split_file(split_dir / "train.txt")
    val_ids = read_split_file(split_dir / "val.txt")
    test_ids = read_split_file(split_dir / "test.txt")

    train_dataset = XBDChangeDataset(
        data_dir,
        train_ids,
        split="train",
        transform=get_week4_transforms(image_size, train=True),
        target_mode="binary",
        filter_empty=True,
    )
    val_dataset = XBDChangeDataset(
        data_dir,
        val_ids,
        split="train",
        transform=get_week4_transforms(image_size, train=False),
        target_mode="binary",
        filter_empty=True,
    )
    test_dataset = XBDChangeDataset(
        data_dir,
        test_ids,
        split="train",
        transform=get_week4_transforms(image_size, train=False),
        target_mode="binary",
        filter_empty=True,
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def build_week4_overfit_dataloaders(
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
        transform=get_week4_transforms(image_size, train=False),
        target_mode="binary",
        filter_empty=True,
    )
    subset = Subset(dataset, list(range(min(sample_count, len(dataset)))))
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def tensor_image_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized 6-channel tensor to a displayable post-disaster RGB image."""
    image = image_tensor.detach().cpu().numpy()
    post_image = np.transpose(image[3:6], (1, 2, 0))
    post_image = post_image * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
    return np.clip(post_image * 255.0, 0, 255).astype(np.uint8)


def create_prediction_overlay(input_image: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Overlay predicted building pixels on the post-disaster image."""
    overlay = input_image.copy()
    mask_bool = pred_mask > 0
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    return overlay


def save_prediction_examples(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    predictions_dir: Path,
    visualizations_dir: Path,
    max_examples: int = 8,
) -> list[dict[str, str | float]]:
    """Save denormalized Week 4 prediction panels and return per-sample records."""
    predictions_dir.mkdir(parents=True, exist_ok=True)
    category_dirs = {
        "best_examples": predictions_dir / "best_examples",
        "difficult_scenes": predictions_dir / "difficult_scenes",
        "failure_cases": predictions_dir / "failure_cases",
    }
    visualization_dirs = {
        "overlays": visualizations_dir / "overlays",
        "masks": visualizations_dir / "masks",
        "comparisons": visualizations_dir / "comparisons",
    }
    for category_dir in category_dirs.values():
        category_dir.mkdir(parents=True, exist_ok=True)
    for visualization_dir in visualization_dirs.values():
        visualization_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0
    records: list[dict[str, str | float]] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            logits = model(images)
            predictions = (torch.sigmoid(logits) > 0.5).float().cpu()
            masks = batch["mask"].cpu()
            sample_dice = per_sample_dice_from_logits(logits.cpu(), masks)
            sample_ids = batch["sample_id"]

            for index, sample_id in enumerate(sample_ids):
                input_image = tensor_image_to_uint8(batch["image"][index])
                gt_mask = (masks[index, 0].numpy() * 255).astype(np.uint8)
                pred_mask = (predictions[index, 0].numpy() * 255).astype(np.uint8)
                dice = float(sample_dice[index].item())
                if dice >= 0.70:
                    category = "best_examples"
                elif dice >= 0.30:
                    category = "difficult_scenes"
                else:
                    category = "failure_cases"

                panel = np.concatenate(
                    [
                        input_image,
                        cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB),
                        cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB),
                    ],
                    axis=1,
                )
                category_dir = category_dirs[category]
                cv2.imwrite(str(category_dir / f"{sample_id}_input_gt_pred.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(category_dir / f"{sample_id}_input.png"), cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(category_dir / f"{sample_id}_gt_mask.png"), gt_mask)
                cv2.imwrite(str(category_dir / f"{sample_id}_pred_mask.png"), pred_mask)
                overlay = create_prediction_overlay(input_image, pred_mask)
                cv2.imwrite(
                    str(visualization_dirs["comparisons"] / f"{sample_id}_input_gt_pred.png"),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(str(visualization_dirs["masks"] / f"{sample_id}_gt_mask.png"), gt_mask)
                cv2.imwrite(str(visualization_dirs["masks"] / f"{sample_id}_pred_mask.png"), pred_mask)
                cv2.imwrite(
                    str(visualization_dirs["overlays"] / f"{sample_id}_prediction_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
                records.append({"sample_id": sample_id, "dice": dice, "category": category})

                saved += 1
                if saved >= max_examples:
                    return records

    return records


def save_failure_analysis_template(output_path: Path) -> None:
    """Create a reusable qualitative failure-analysis note file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """# Week 4 Failure Case Analysis

Use the prediction panels in `results/week4/predictions/` and the per-sample CSV in
`results/week4/metrics/prediction_records.csv` to compare the pretrained encoder against Week 3.

## Best Predictions

- Samples where building footprints align well with the ground truth:
- Improvements compared with Week 3:

## Difficult Scenes

- Tiny buildings:
- Shadows:
- Smoke or haze:
- Flood reflections:
- Dense urban regions:
- Partial/unclear building boundaries:

## Failure Cases

- False positives:
- False negatives:
- Missed destroyed buildings:
- Mask shifted or fragmented:
- Empty or near-empty predictions:

## Research Directions

- Fine-tune encoder learning rate separately from decoder learning rate.
- Test ResNet50 or EfficientNet encoders.
- Move from binary building segmentation to multiclass damage segmentation.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Week 4 ResNet34-encoder U-Net segmentation model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Use only this many training samples.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Use only this many validation samples.")
    parser.add_argument("--overfit-samples", type=int, default=None, help="Debug by training/validating on N clean samples.")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "week4")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--num-predictions", type=int, default=18, help="Validation predictions to save when validation improves.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained ResNet34 weights.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only the decoder and segmentation head.")
    args = parser.parse_args()

    args.checkpoint_dir = args.checkpoint_dir or args.results_dir / "checkpoints"
    args.prediction_dir = args.prediction_dir or args.results_dir / "predictions"
    metrics_dir = args.results_dir / "metrics"
    config_dir = args.results_dir / "config"
    visualizations_dir = args.results_dir / "visualizations"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.overfit_samples is not None and args.overfit_samples > 0:
        train_loader, val_loader = build_week4_overfit_dataloaders(
            args.data_dir,
            args.split_dir,
            args.image_size,
            args.batch_size,
            args.num_workers,
            args.overfit_samples,
        )
    else:
        train_loader, val_loader, _ = build_week4_dataloaders(
            args.data_dir,
            args.split_dir,
            args.image_size,
            args.batch_size,
            args.num_workers,
        )
        train_loader = limit_dataloader(train_loader, args.max_train_samples, shuffle=True)
        val_loader = limit_dataloader(val_loader, args.max_val_samples, shuffle=False)

    model = ResNet34UNet(out_channels=1, pretrained=not args.no_pretrained, freeze_encoder=args.freeze_encoder).to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    criterion = BCEDiceLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "data_dir": str(args.data_dir),
        "split_dir": str(args.split_dir),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "loss": "BCE + Dice",
        "model": "ResNet34UNet",
        "encoder": "ResNet34",
        "pretrained": not args.no_pretrained,
        "freeze_encoder": args.freeze_encoder,
        "normalization": "ImageNet mean/std duplicated for pre and post RGB",
        "num_workers": args.num_workers,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "overfit_samples": args.overfit_samples,
        "target_mode": "binary",
        "device": str(device),
    }
    save_json(config, config_dir / "training_config.json")
    save_training_config_yaml(config, config_dir / "training_config.yaml")
    train_sample_counts, train_class_counts = collect_dataset_statistics(args.data_dir, "train")
    save_metrics_csv(train_sample_counts, train_class_counts, metrics_dir / "dataset_statistics.csv")
    save_failure_analysis_template(args.results_dir / "failure_analysis.md")

    best_val_dice = -1.0
    best_metrics: dict[str, float | int] = {}
    history: list[dict[str, float | int]] = []
    print(f"device={device}")
    print(f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)}")
    print(f"model=ResNet34UNet pretrained={not args.no_pretrained} freeze_encoder={args.freeze_encoder}")
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

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_metrics["dice"],
            "val_dice": val_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_record)
        save_history_csv(history, metrics_dir / "training_log.csv")
        plot_training_curves(history, visualizations_dir)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_metrics = {
                "best_epoch": epoch,
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
            save_json(best_metrics, metrics_dir / "final_metrics.json")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "model": "ResNet34UNet",
                    "pretrained": not args.no_pretrained,
                },
                args.checkpoint_dir / "week4_resnet34_unet_binary_best.pt",
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
    plot_training_curves(history, visualizations_dir)
    if best_metrics:
        save_json(best_metrics, metrics_dir / "final_metrics.json")


if __name__ == "__main__":
    main()

"""Standalone Week 6 inference and qualitative export pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import DataLoader

from week2_dataset import XBDChangeDataset, read_split_file
from week6_augmentations import get_week6_transforms
from week6_experiment_runner import build_model
from week6_metrics import CLASS_NAMES, confusion_matrix_from_logits, metrics_from_confusion_matrix, save_confusion_matrix_csv, save_per_class_metrics
from week6_utils import get_device, load_checkpoint, save_json
from week6_visualization import save_confidence_map, save_error_heatmap, save_overlay, save_prediction_panel, tensor_to_rgb_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 6 checkpoint inference and export predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Override model name if it is not stored in the checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week6" / "inference_exports")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-limit", type=int, default=50)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or checkpoint.get("model", "attention_unet")
    model = build_model(model_name, pretrained=not args.no_pretrained).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    split_ids = read_split_file(args.split_dir / f"{args.split}.txt")
    if args.max_samples is not None:
        split_ids = split_ids[: args.max_samples]
    dataset = XBDChangeDataset(
        args.data_dir,
        split_ids,
        split="train",
        transform=get_week6_transforms(args.image_size, train=False, advanced=False),
        target_mode="multiclass",
        filter_empty=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.output_dir / "panels"
    confidence_dir = args.output_dir / "confidence_maps"
    error_dir = args.output_dir / "error_heatmaps"
    metrics_dir = args.output_dir / "metrics"
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    records = []
    saved = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).long()
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            confusion += confusion_matrix_from_logits(logits.cpu(), masks.cpu(), len(CLASS_NAMES))
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_name = str(sample_id).replace("/", "_").replace("\\", "_")
                records.append({"sample_id": sample_name})
                if saved >= args.save_limit:
                    continue
                pre_image, post_image = tensor_to_rgb_pair(images[index].cpu())
                target = masks[index].cpu().numpy()
                prediction = predictions[index].cpu().numpy()
                save_prediction_panel(pre_image, post_image, target, prediction, panel_dir / f"{sample_name}.png")
                save_confidence_map(logits[index].cpu().unsqueeze(0), confidence_dir / f"{sample_name}.png")
                save_error_heatmap(target, prediction, error_dir / f"{sample_name}.png")
                save_overlay(post_image, prediction, args.output_dir / "overlays" / f"{sample_name}.png")
                saved += 1

    metrics = metrics_from_confusion_matrix(confusion)
    save_json({"checkpoint": str(args.checkpoint), "model": model_name, **metrics}, metrics_dir / "final_metrics.json")
    save_per_class_metrics(metrics, metrics_dir / "per_class_metrics.csv")
    save_confusion_matrix_csv(confusion, metrics_dir / "confusion_matrix.csv")
    with (metrics_dir / "prediction_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Inference exports saved to {args.output_dir}")


if __name__ == "__main__":
    main()

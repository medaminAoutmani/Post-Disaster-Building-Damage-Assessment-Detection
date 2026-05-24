"""Standalone Week 7 Siamese inference pipeline."""

from __future__ import annotations

import argparse
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

from week2_dataset import read_split_file
from week6.week6_augmentations import get_week6_transforms
from week6.week6_utils import get_device, load_checkpoint, save_json
from week6.week6_visualization import save_confidence_map, save_error_heatmap, save_overlay, save_prediction_panel, tensor_to_rgb_pair
from week7_dataset import XBDTemporalDamageDataset
from week7_experiment_runner import build_model
from week7_metrics import CLASS_NAMES, confusion_matrix_from_logits, metrics_from_confusion_matrix, save_confusion_matrix_csv, save_per_class_metrics
from week7_visualization import save_temporal_difference_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 7 Siamese checkpoint inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="siamese_resnet50_unet")
    parser.add_argument("--fusion", default="concat")
    parser.add_argument("--attention", default="no_attention")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--xbd-split", default="train")
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week7" / "inference_exports")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model = build_model(args.model, args.fusion, args.attention, pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    ids = read_split_file(args.split_dir / f"{args.split}.txt")
    dataset = XBDTemporalDamageDataset(
        args.data_dir,
        ids,
        split=args.xbd_split,
        transform=get_week6_transforms(args.image_size, train=False),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    saved = 0
    with torch.no_grad():
        for batch in loader:
            pre = batch["pre_image"].to(device)
            post = batch["post_image"].to(device)
            masks = batch["mask"].to(device).long()
            logits = model(pre, post)
            predictions = torch.argmax(logits, dim=1)
            confusion += confusion_matrix_from_logits(logits.cpu(), masks.cpu(), len(CLASS_NAMES))
            for index, sample_id in enumerate(batch["sample_id"]):
                if saved >= args.save_limit:
                    continue
                name = str(sample_id).replace("/", "_").replace("\\", "_")
                pre_rgb, post_rgb = tensor_to_rgb_pair(batch["image"][index].cpu())
                target = masks[index].cpu().numpy()
                prediction = predictions[index].cpu().numpy()
                save_prediction_panel(pre_rgb, post_rgb, target, prediction, args.output_dir / "panels" / f"{name}.png")
                save_overlay(post_rgb, prediction, args.output_dir / "overlays" / f"{name}.png")
                save_confidence_map(logits[index].cpu().unsqueeze(0), args.output_dir / "confidence_maps" / f"{name}.png")
                save_error_heatmap(target, prediction, args.output_dir / "error_heatmaps" / f"{name}.png")
                save_temporal_difference_map(pre_rgb, post_rgb, args.output_dir / "temporal_difference_maps" / f"{name}.png")
                saved += 1
    metrics = metrics_from_confusion_matrix(confusion)
    save_json(metrics, args.output_dir / "metrics" / "final_metrics.json")
    save_per_class_metrics(metrics, args.output_dir / "metrics" / "per_class_metrics.csv")
    save_confusion_matrix_csv(confusion, args.output_dir / "metrics" / "confusion_matrix.csv")
    print(f"Week 7 inference exports saved to {args.output_dir}")


if __name__ == "__main__":
    main()


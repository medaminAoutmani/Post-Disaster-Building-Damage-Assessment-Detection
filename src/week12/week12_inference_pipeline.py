"""Run the Week 12 hierarchical inference pipeline on object-level building crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
for path in [CURRENT_DIR, WEEK11_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset
from week11_train_classifier import confusion_matrix, metrics_from_confusion, save_confusion_csv, save_confusion_plot, save_json
from week12_hierarchical_model import STAGE1_CLASS_NAMES, STAGE2_CLASS_NAMES
from week12_model_backbones import ObjectDamageRepresentationModel


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> ObjectDamageRepresentationModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ObjectDamageRepresentationModel(
        backbone=checkpoint.get("backbone", "resnet34"),
        fusion=checkpoint.get("fusion", "gated"),
        embedding_dim=int(checkpoint.get("embedding_dim", 256)),
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Week 12: hierarchical object-level inference.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week12_hierarchical_pipeline")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--damaged-threshold", type=float, default=0.35, help="Lower values prioritize damaged recall.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BuildingDamageDataset(args.dataset_root, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    stage1 = load_model(args.stage1_checkpoint, len(STAGE1_CLASS_NAMES), device)
    stage2 = load_model(args.stage2_checkpoint, len(STAGE2_CLASS_NAMES), device)

    all_predictions = []
    all_labels = []
    rows = []
    for batch in loader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        diff = batch["diff"].to(device)
        labels = batch["label"].long()
        stage1_probs = torch.softmax(stage1(pre, post, diff), dim=1).cpu()
        damaged_mask = stage1_probs[:, 1] >= args.damaged_threshold
        predictions = torch.zeros(len(labels), dtype=torch.long)
        if damaged_mask.any():
            damaged_indices = damaged_mask.nonzero(as_tuple=False).flatten()
            stage2_logits = stage2(pre[damaged_indices].to(device), post[damaged_indices].to(device), diff[damaged_indices].to(device))
            stage2_predictions = torch.argmax(stage2_logits.cpu(), dim=1) + 1
            predictions[damaged_indices.cpu()] = stage2_predictions
        all_predictions.append(predictions)
        all_labels.append(labels)
        for i in range(len(labels)):
            rows.append(
                {
                    "metadata_path": batch["metadata_path"][i],
                    "true_label": int(labels[i].item()),
                    "true_class": CLASS_NAMES[int(labels[i].item())],
                    "predicted_label": int(predictions[i].item()),
                    "predicted_class": CLASS_NAMES[int(predictions[i].item())],
                    "stage1_no_damage_prob": float(stage1_probs[i, 0].item()),
                    "stage1_damaged_prob": float(stage1_probs[i, 1].item()),
                }
            )

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    confusion = confusion_matrix(predictions, labels, len(CLASS_NAMES))
    metrics = metrics_from_confusion(confusion)
    save_json(
        {
            "split": args.split,
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "stage2_checkpoint": str(args.stage2_checkpoint),
            "damaged_threshold": args.damaged_threshold,
            **metrics,
        },
        args.output_dir / "metrics" / "pipeline_metrics.json",
    )
    save_confusion_csv(confusion, args.output_dir / "confusion_matrices" / "confusion_matrix.csv")
    save_confusion_plot(confusion, args.output_dir / "confusion_matrices" / "confusion_matrix.png")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

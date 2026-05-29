"""Apply topology-guided no_damage/minor_damage correction to CNN predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
WEEK12_DIR = CURRENT_DIR.parent / "week12"
for path in [WEEK11_DIR, WEEK12_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset
from week11_train_classifier import confusion_matrix, metrics_from_confusion, save_confusion_csv, save_confusion_plot, save_json
from week12_model_backbones import ObjectDamageRepresentationModel
from week13_topology_features import TOPOLOGY_FEATURE_NAMES, extract_topology_signature


def load_checkpoint(path: Path, device: torch.device) -> ObjectDamageRepresentationModel:
    checkpoint = torch.load(path, map_location="cpu")
    model = ObjectDamageRepresentationModel(
        backbone=checkpoint.get("backbone", "convnext_tiny"),
        fusion=checkpoint.get("fusion", "gated"),
        embedding_dim=int(checkpoint.get("embedding_dim", 256)),
        num_classes=len(CLASS_NAMES),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def topology_score(sample_dir: Path, config: dict, thresholds: int) -> float:
    features = extract_topology_signature(sample_dir, thresholds=thresholds)
    vector = np.asarray([features[name] for name in TOPOLOGY_FEATURE_NAMES], dtype=np.float32)
    mean = np.asarray(config["normalization_mean"], dtype=np.float32)
    std = np.asarray(config["normalization_std"], dtype=np.float32)
    no_proto = np.asarray(config["no_damage_prototype"], dtype=np.float32)
    minor_proto = np.asarray(config["minor_damage_prototype"], dtype=np.float32)
    normalized = (vector - mean) / std
    return float(np.linalg.norm(normalized - no_proto) - np.linalg.norm(normalized - minor_proto))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Week 13: hybrid CNN + TDA correction.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings_week8_extra")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold-json", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week13_topology" / "hybrid")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--thresholds", type=int, default=16)
    parser.add_argument("--ambiguity-margin", type=float, default=0.20)
    args = parser.parse_args()

    config = json.loads(args.threshold_json.read_text(encoding="utf-8"))
    threshold = float(config["topology_distance_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BuildingDamageDataset(args.dataset_root, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_checkpoint(args.checkpoint, device)

    baseline_confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    hybrid_confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    corrections = []
    for batch in dataloader:
        logits = model(batch["pre"].to(device), batch["post"].to(device), batch["diff"].to(device))
        probs = F.softmax(logits, dim=1).cpu()
        baseline_predictions = torch.argmax(probs, dim=1)
        hybrid_predictions = baseline_predictions.clone()
        labels = batch["label"].long()
        for index in range(len(labels)):
            prediction = int(baseline_predictions[index].item())
            no_minor_gap = abs(float(probs[index, 0].item()) - float(probs[index, 1].item()))
            if prediction in {0, 1} and no_minor_gap <= args.ambiguity_margin:
                sample_dir = Path(str(batch["metadata_path"][index])).parent
                score = topology_score(sample_dir, config, thresholds=args.thresholds)
                corrected = 1 if score >= threshold else 0
                hybrid_predictions[index] = corrected
                if corrected != prediction:
                    corrections.append(
                        {
                            "metadata_path": str(batch["metadata_path"][index]),
                            "label": int(labels[index].item()),
                            "baseline_prediction": prediction,
                            "hybrid_prediction": corrected,
                            "topology_score": score,
                            "threshold": threshold,
                            "p_no_damage": float(probs[index, 0].item()),
                            "p_minor_damage": float(probs[index, 1].item()),
                        }
                    )
        baseline_confusion += confusion_matrix(baseline_predictions, labels, len(CLASS_NAMES))
        hybrid_confusion += confusion_matrix(hybrid_predictions, labels, len(CLASS_NAMES))

    baseline_metrics = metrics_from_confusion(baseline_confusion)
    hybrid_metrics = metrics_from_confusion(hybrid_confusion)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "baseline_metrics": baseline_metrics,
            "hybrid_metrics": hybrid_metrics,
            "topology_distance_threshold": threshold,
            "ambiguity_margin": args.ambiguity_margin,
            "num_corrections": len(corrections),
        },
        args.output_dir / "metrics" / "hybrid_metrics.json",
    )
    save_confusion_csv(baseline_confusion, args.output_dir / "confusion_matrices" / "baseline_confusion_matrix.csv")
    save_confusion_csv(hybrid_confusion, args.output_dir / "confusion_matrices" / "hybrid_confusion_matrix.csv")
    save_confusion_plot(hybrid_confusion, args.output_dir / "confusion_matrices" / "hybrid_confusion_matrix.png")
    with (args.output_dir / "corrections.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["metadata_path", "label", "baseline_prediction", "hybrid_prediction", "topology_score", "threshold", "p_no_damage", "p_minor_damage"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corrections)


if __name__ == "__main__":
    main()

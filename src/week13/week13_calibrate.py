"""Post-training calibration and threshold search for Week 13 damage models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
WEEK12_DIR = CURRENT_DIR.parent / "week12"
for path in [CURRENT_DIR, WEEK11_DIR, WEEK12_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset
from week11_train_classifier import confusion_matrix, metrics_from_confusion, save_confusion_csv, save_confusion_plot, save_json
from week12_model_backbones import ObjectDamageRepresentationModel
from week13_losses import (
    coral_logits_to_class_probs,
    damaged_auroc,
    expected_calibration_error,
    mean_severity_distance,
)
from week13_models import MultiTaskDamageModel


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loss_type = checkpoint.get("loss_type", "ce")
    backbone = checkpoint.get("backbone", "convnext_tiny")
    fusion = checkpoint.get("fusion", "gated")
    embedding_dim = int(checkpoint.get("embedding_dim", 256))
    dropout = float(checkpoint.get("dropout", 0.3))
    if loss_type == "multitask":
        model: nn.Module = MultiTaskDamageModel(backbone=backbone, fusion=fusion, embedding_dim=embedding_dim, dropout=dropout)
    else:
        output_dim = 3 if loss_type == "coral" else 1 if loss_type == "regression" else len(CLASS_NAMES)
        model = ObjectDamageRepresentationModel(backbone=backbone, fusion=fusion, embedding_dim=embedding_dim, num_classes=output_dim, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def collect_probs(model: nn.Module, dataloader: DataLoader, device: torch.device, loss_type: str, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    all_probs = []
    all_labels = []
    for batch in dataloader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        diff = batch["diff"].to(device)
        labels = batch["label"].long()
        if loss_type == "multitask":
            logits = model(pre, post, diff)["logits"] / temperature
            probs = F.softmax(logits, dim=1)
        elif loss_type == "coral":
            logits = model(pre, post, diff) / temperature
            probs = coral_logits_to_class_probs(logits)
        elif loss_type == "regression":
            scores = torch.round(model(pre, post, diff).squeeze(1)).long().clamp(0, len(CLASS_NAMES) - 1)
            probs = F.one_hot(scores, num_classes=len(CLASS_NAMES)).float()
        else:
            logits = model(pre, post, diff) / temperature
            probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)


def optimize_temperature_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=80)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp_min(0.05), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(0.05, 10.0).item())


@torch.no_grad()
def collect_logits_for_temperature(model: nn.Module, dataloader: DataLoader, device: torch.device, loss_type: str) -> tuple[torch.Tensor | None, torch.Tensor]:
    if loss_type in {"coral", "regression"}:
        _, labels = collect_probs(model, dataloader, device, loss_type)
        return None, labels
    all_logits = []
    all_labels = []
    for batch in dataloader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        diff = batch["diff"].to(device)
        logits = model(pre, post, diff)["logits"] if loss_type == "multitask" else model(pre, post, diff)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].long())
    return torch.cat(all_logits), torch.cat(all_labels)


def predict_with_thresholds(probs: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    eligible = probs >= thresholds.view(1, -1)
    masked = probs.masked_fill(~eligible, -1.0)
    threshold_predictions = torch.argmax(masked, dim=1)
    fallback = torch.argmax(probs, dim=1)
    has_match = eligible.any(dim=1)
    return torch.where(has_match, threshold_predictions, fallback)


def metrics_for_predictions(probs: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    confusion = confusion_matrix(predictions, labels, len(CLASS_NAMES))
    metrics = metrics_from_confusion(confusion)
    metrics["ece"] = expected_calibration_error(probs, labels)
    metrics["mean_severity_distance"] = mean_severity_distance(predictions, labels)
    metrics["damaged_auroc"] = damaged_auroc(probs, labels)
    return metrics, confusion


def greedy_threshold_search(probs: torch.Tensor, labels: torch.Tensor, steps: int = 15) -> torch.Tensor:
    thresholds = torch.full((len(CLASS_NAMES),), 0.5)
    grid = torch.linspace(0.05, 0.95, steps)
    best_predictions = predict_with_thresholds(probs, thresholds)
    best_metrics, _ = metrics_for_predictions(probs, labels, best_predictions)
    best_score = best_metrics["macro_f1"]
    for _ in range(3):
        changed = False
        for class_index in range(len(CLASS_NAMES)):
            local_best_value = thresholds[class_index]
            local_best_score = best_score
            for value in grid:
                candidate = thresholds.clone()
                candidate[class_index] = value
                predictions = predict_with_thresholds(probs, candidate)
                metrics, _ = metrics_for_predictions(probs, labels, predictions)
                score = metrics["macro_f1"]
                if score > local_best_score:
                    local_best_score = score
                    local_best_value = value
            if local_best_score > best_score:
                thresholds[class_index] = local_best_value
                best_score = local_best_score
                changed = True
        if not changed:
            break
    return thresholds


def save_reliability_diagram(probs: torch.Tensor, labels: torch.Tensor, path: Path, bins: int = 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels).float()
    centers = []
    accuracies = []
    confidence_means = []
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (confidences > lower) & (confidences <= upper)
        centers.append(float((lower + upper) / 2.0))
        accuracies.append(float(correct[mask].mean().item()) if mask.any() else 0.0)
        confidence_means.append(float(confidences[mask].mean().item()) if mask.any() else 0.0)
    plt.figure(figsize=(6, 6))
    plt.bar(centers, accuracies, width=1.0 / bins, alpha=0.75, label="accuracy")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="perfect calibration")
    plt.plot(centers, confidence_means, color="#d62728", marker="o", linewidth=1.5, label="confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Week 13 reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_thresholds(thresholds: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_name", "threshold"])
        for class_name, threshold in zip(CLASS_NAMES, thresholds.tolist()):
            writer.writerow([class_name, threshold])


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 13: calibrate object-level damage checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week13_calibration")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-steps", type=int, default=19)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    loss_type = checkpoint.get("loss_type", "ce")
    dataset = BuildingDamageDataset(args.dataset_root, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logits, labels_for_temp = collect_logits_for_temperature(model, dataloader, device, loss_type)
    temperature = 1.0 if logits is None else optimize_temperature_from_logits(logits, labels_for_temp)
    probs, labels = collect_probs(model, dataloader, device, loss_type, temperature=temperature)

    argmax_predictions = torch.argmax(probs, dim=1)
    argmax_metrics, argmax_confusion = metrics_for_predictions(probs, labels, argmax_predictions)
    thresholds = greedy_threshold_search(probs, labels, steps=args.threshold_steps)
    threshold_predictions = predict_with_thresholds(probs, thresholds)
    threshold_metrics, threshold_confusion = metrics_for_predictions(probs, labels, threshold_predictions)

    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "loss_type": loss_type,
            "temperature": temperature,
            "argmax_metrics": argmax_metrics,
            "threshold_metrics": threshold_metrics,
            "thresholds": {name: float(thresholds[i].item()) for i, name in enumerate(CLASS_NAMES)},
        },
        args.output_dir / "metrics" / "calibration_metrics.json",
    )
    save_confusion_csv(argmax_confusion, args.output_dir / "confusion_matrices" / "argmax_confusion_matrix.csv")
    save_confusion_csv(threshold_confusion, args.output_dir / "confusion_matrices" / "threshold_confusion_matrix.csv")
    save_confusion_plot(threshold_confusion, args.output_dir / "confusion_matrices" / "threshold_confusion_matrix.png")
    save_thresholds(thresholds, args.output_dir / "metrics" / "classwise_thresholds.csv")
    save_reliability_diagram(probs, labels, args.output_dir / "visualizations" / "reliability_diagram.png")

    print(json.dumps({"temperature": temperature, "argmax_metrics": argmax_metrics, "threshold_metrics": threshold_metrics}, indent=2))


if __name__ == "__main__":
    main()

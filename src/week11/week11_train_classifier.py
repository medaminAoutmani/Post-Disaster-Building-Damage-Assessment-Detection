"""Train the Week 11 building-level Siamese damage classifier."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset
from week11_model import SiameseBuildingClassifier


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute a confusion matrix where rows are truth and columns are predictions."""
    bins = num_classes * targets.reshape(-1) + predictions.reshape(-1)
    return torch.bincount(bins, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: torch.Tensor, eps: float = 1e-7) -> dict[str, float]:
    """Compute accuracy, macro/weighted F1, and per-class precision/recall/F1."""
    matrix = confusion.float()
    true_positive = torch.diag(matrix)
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = true_positive / predicted.clamp_min(eps)
    recall = true_positive / support.clamp_min(eps)
    f1 = (2.0 * true_positive) / (support + predicted).clamp_min(eps)
    total = matrix.sum().clamp_min(eps)

    metrics = {
        "accuracy": float(true_positive.sum().div(total).item()),
        "macro_f1": float(f1.mean().item()),
        "weighted_f1": float((f1 * support).sum().div(total).item()),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f"precision_{class_name}"] = float(precision[index].item())
        metrics[f"recall_{class_name}"] = float(recall[index].item())
        metrics[f"f1_{class_name}"] = float(f1[index].item())
        metrics[f"support_{class_name}"] = float(support[index].item())
        metrics[f"predicted_{class_name}"] = float(predicted[index].item())
    return metrics


def class_counts(dataset: BuildingDamageDataset) -> torch.Tensor:
    """Count samples per class in a Week 11 dataset."""
    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.float32)
    for sample in dataset.samples:
        counts[int(sample["label"])] += 1.0
    return counts


def build_loss_weights(counts: torch.Tensor, mode: str, beta: float = 0.999) -> torch.Tensor | None:
    """Build class weights for CrossEntropyLoss."""
    if mode == "none":
        return None
    if mode == "inverse":
        weights = counts.sum() / counts.clamp_min(1.0)
    elif mode == "effective":
        effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / effective_num.clamp_min(1e-7)
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    return weights / weights.mean().clamp_min(1e-7)


def build_weighted_sampler(dataset: BuildingDamageDataset, counts: torch.Tensor) -> WeightedRandomSampler:
    """Sample minority classes more often during training."""
    class_weights = 1.0 / counts.clamp_min(1.0)
    sample_weights = torch.tensor([float(class_weights[int(sample["label"])]) for sample in dataset.samples])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], torch.Tensor]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    for batch in dataloader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        diff = batch["diff"].to(device)
        labels = batch["label"].to(device).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(pre, post, diff)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        predictions = torch.argmax(logits.detach().cpu(), dim=1)
        confusion += confusion_matrix(predictions, labels.detach().cpu(), len(CLASS_NAMES))
    return total_loss / len(dataloader), metrics_from_confusion(confusion), confusion


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], torch.Tensor]:
    """Evaluate the classifier."""
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    for batch in dataloader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        diff = batch["diff"].to(device)
        labels = batch["label"].to(device).long()
        logits = model(pre, post, diff)
        loss = criterion(logits, labels)

        total_loss += float(loss.item())
        predictions = torch.argmax(logits.cpu(), dim=1)
        confusion += confusion_matrix(predictions, labels.cpu(), len(CLASS_NAMES))
    return total_loss / len(dataloader), metrics_from_confusion(confusion), confusion


def save_json(data: dict, path: Path) -> None:
    """Save JSON data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_history(history: list[dict[str, float | int]], path: Path) -> None:
    """Save epoch metrics to CSV."""
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_confusion_csv(confusion: torch.Tensor, path: Path) -> None:
    """Save confusion matrix CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, confusion.tolist()):
            writer.writerow([class_name, *row])


def save_confusion_plot(confusion: torch.Tensor, path: Path) -> None:
    """Save a normalized confusion matrix plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion.float().numpy()
    normalized = np.divide(matrix, np.maximum(matrix.sum(axis=1, keepdims=True), 1.0))
    plt.figure(figsize=(7, 6))
    plt.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Row-normalized frequency")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Week 11 building-level confusion matrix")
    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            plt.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_history(history: list[dict[str, float | int]], output_dir: Path) -> None:
    """Plot loss and F1 curves."""
    if not history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(row["train_loss"]) for row in history], label="Train loss")
    plt.plot(epochs, [float(row["val_loss"]) for row in history], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy")
    plt.title("Week 11 classifier loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(row["val_macro_f1"]) for row in history], label="Macro F1")
    plt.plot(epochs, [float(row["val_recall_minor_damage"]) for row in history], label="Minor recall")
    plt.plot(epochs, [float(row["val_recall_major_damage"]) for row in history], label="Major recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Week 11 key validation metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "key_metrics_curve.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 11: train building-level damage classifier.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "week11")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet ResNet18 weights if available.")
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "inverse", "effective"],
        default="none",
        help="Class weighting for CrossEntropyLoss. Use 'effective' first for imbalanced Week 11 crops.",
    )
    parser.add_argument("--weighted-sampler", action="store_true", help="Use inverse-frequency weighted sampling.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = BuildingDamageDataset(args.dataset_root, "train")
    val_dataset = BuildingDamageDataset(args.dataset_root, "val")
    train_counts = class_counts(train_dataset)
    val_counts = class_counts(val_dataset)
    train_sampler = build_weighted_sampler(train_dataset, train_counts) if args.weighted_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SiameseBuildingClassifier(pretrained=args.pretrained).to(device)
    loss_weights = build_loss_weights(train_counts, args.class_weight_mode)
    criterion = nn.CrossEntropyLoss(weight=None if loss_weights is None else loss_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_dir = args.results_dir / "metrics"
    confusion_dir = args.results_dir / "confusion_matrices"
    checkpoint_dir = args.results_dir / "checkpoints"
    visualizations_dir = args.results_dir / "visualizations"
    for directory in [metrics_dir, confusion_dir, checkpoint_dir, visualizations_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset_root": str(args.dataset_root),
        "model": "SiameseBuildingClassifier",
        "encoder": "ResNet18",
        "fusion": "concat(pre, post, diff, abs(pre-post)) embeddings",
        "loss": "CrossEntropyLoss",
        "class_weight_mode": args.class_weight_mode,
        "loss_weights": None if loss_weights is None else [float(value) for value in loss_weights.tolist()],
        "weighted_sampler": args.weighted_sampler,
        "class_names": CLASS_NAMES,
        "train_class_counts": {class_name: int(train_counts[index].item()) for index, class_name in enumerate(CLASS_NAMES)},
        "val_class_counts": {class_name: int(val_counts[index].item()) for index, class_name in enumerate(CLASS_NAMES)},
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pretrained": args.pretrained,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    save_json(config, args.results_dir / "config" / "training_config.json")

    best_macro_f1 = -1.0
    best_metrics: dict[str, float | int] = {}
    best_confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    history: list[dict[str, float | int]] = []
    print(f"device={device} train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    print(f"train_class_counts={config['train_class_counts']}")
    print(f"val_class_counts={config['val_class_counts']}")
    print(f"class_weight_mode={args.class_weight_mode} weighted_sampler={args.weighted_sampler}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, val_confusion = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"minor_recall={val_metrics['recall_minor_damage']:.4f} "
            f"major_recall={val_metrics['recall_major_damage']:.4f}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        save_history(history, metrics_dir / "training_log.csv")
        plot_history(history, visualizations_dir)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_confusion = val_confusion
            best_metrics = {"best_epoch": epoch, "val_loss": val_loss, **val_metrics}
            save_json(best_metrics, metrics_dir / "final_metrics.json")
            save_confusion_csv(best_confusion, confusion_dir / "confusion_matrix.csv")
            save_confusion_plot(best_confusion, confusion_dir / "confusion_matrix.png")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "class_names": CLASS_NAMES,
                },
                checkpoint_dir / "week11_siamese_resnet18_best.pt",
            )

    if best_metrics:
        save_json(best_metrics, metrics_dir / "final_metrics.json")
        save_confusion_csv(best_confusion, confusion_dir / "confusion_matrix.csv")
        save_confusion_plot(best_confusion, confusion_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()

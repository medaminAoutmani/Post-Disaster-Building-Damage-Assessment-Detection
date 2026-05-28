"""Train Week 12 hierarchical stage 1: no_damage vs damaged."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
for path in [CURRENT_DIR, WEEK11_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import BuildingDamageDataset
from week11_train_classifier import confusion_matrix, save_json
from week12_hierarchical_model import STAGE1_CLASS_NAMES, Stage1DamageDataset
from week12_model_backbones import BACKBONE_NAMES, FUSION_NAMES, ObjectDamageRepresentationModel


def metrics_from_confusion_local(confusion: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    matrix = confusion.float()
    tp = torch.diag(matrix)
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    f1 = (2.0 * tp) / (support + predicted).clamp_min(1e-7)
    precision = tp / predicted.clamp_min(1e-7)
    recall = tp / support.clamp_min(1e-7)
    metrics = {
        "accuracy": float(tp.sum().div(matrix.sum().clamp_min(1e-7)).item()),
        "macro_f1": float(f1.mean().item()),
    }
    for index, name in enumerate(class_names):
        metrics[f"precision_{name}"] = float(precision[index].item())
        metrics[f"recall_{name}"] = float(recall[index].item())
        metrics[f"f1_{name}"] = float(f1[index].item())
        metrics[f"support_{name}"] = float(support[index].item())
        metrics[f"predicted_{name}"] = float(predicted[index].item())
    return metrics


def class_counts(dataset: Stage1DamageDataset, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for sample in dataset.samples:
        counts[0 if int(sample["label"]) == 0 else 1] += 1.0
    return counts


def weighted_sampler(dataset: Stage1DamageDataset, counts: torch.Tensor) -> WeightedRandomSampler:
    weights = 1.0 / counts.clamp_min(1.0)
    sample_weights = torch.tensor([float(weights[0 if int(sample["label"]) == 0 else 1]) for sample in dataset.samples])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def save_confusion_csv_local(confusion: torch.Tensor, class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("actual/predicted," + ",".join(class_names) + "\n")
        for class_name, row in zip(class_names, confusion.tolist()):
            file.write(class_name + "," + ",".join(str(value) for value in row) + "\n")


def save_confusion_plot_local(confusion: torch.Tensor, class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion.float().numpy()
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1.0)
    plt.figure(figsize=(6, 5))
    plt.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Row-normalized frequency")
    plt.xticks(range(len(class_names)), class_names, rotation=25, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            plt.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_epoch(
    model: ObjectDamageRepresentationModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float], torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((len(STAGE1_CLASS_NAMES), len(STAGE1_CLASS_NAMES)), dtype=torch.long)
    total_loss = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            labels = batch["label"].to(device).long()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch["pre"].to(device), batch["post"].to(device), batch["diff"].to(device))
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            confusion += confusion_matrix(torch.argmax(logits.detach().cpu(), dim=1), labels.cpu(), len(STAGE1_CLASS_NAMES))
    return total_loss / max(len(loader), 1), metrics_from_confusion_local(confusion, STAGE1_CLASS_NAMES), confusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 12: train hierarchical stage 1 no_damage vs damaged.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "week12_hierarchical_stage1")
    parser.add_argument("--backbone", choices=BACKBONE_NAMES, default="resnet34")
    parser.add_argument("--fusion", choices=FUSION_NAMES, default="gated")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--damaged-weight", type=float, default=2.0, help="Extra CE weight for damaged recall.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Stage1DamageDataset(BuildingDamageDataset(args.dataset_root, "train", augment=args.augment_train))
    val_dataset = Stage1DamageDataset(BuildingDamageDataset(args.dataset_root, "val"))
    counts = class_counts(train_dataset, len(STAGE1_CLASS_NAMES))
    sampler = weighted_sampler(train_dataset, counts) if args.weighted_sampler else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = ObjectDamageRepresentationModel(args.backbone, len(STAGE1_CLASS_NAMES), args.embedding_dim, args.fusion, args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.damaged_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_recall = -1.0
    for directory in [args.results_dir / "metrics", args.results_dir / "confusion_matrices", args.results_dir / "checkpoints"]:
        directory.mkdir(parents=True, exist_ok=True)
    save_json(
        {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "class_names": STAGE1_CLASS_NAMES, "train_class_counts": counts.tolist()},
        args.results_dir / "config" / "training_config.json",
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, metrics, confusion = run_epoch(model, val_loader, criterion, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} damaged_recall={metrics['recall_damaged']:.4f}")
        if metrics["recall_damaged"] > best_recall:
            best_recall = metrics["recall_damaged"]
            save_json({"best_epoch": epoch, "val_loss": val_loss, **metrics}, args.results_dir / "metrics" / "final_metrics.json")
            save_confusion_csv_local(confusion, STAGE1_CLASS_NAMES, args.results_dir / "confusion_matrices" / "confusion_matrix.csv")
            save_confusion_plot_local(confusion, STAGE1_CLASS_NAMES, args.results_dir / "confusion_matrices" / "confusion_matrix.png")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "fusion": args.fusion,
                    "embedding_dim": args.embedding_dim,
                    "class_names": STAGE1_CLASS_NAMES,
                },
                args.results_dir / "checkpoints" / "week12_stage1_best.pt",
            )


if __name__ == "__main__":
    main()

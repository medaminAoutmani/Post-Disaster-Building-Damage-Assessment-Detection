"""Train Week 12 hierarchical stage 2: minor vs major vs destroyed."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
for path in [CURRENT_DIR, WEEK11_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import BuildingDamageDataset
from week11_train_classifier import confusion_matrix, save_json
from week12_hierarchical_model import STAGE2_CLASS_NAMES, Stage2DamageDataset
from week12_model_backbones import BACKBONE_NAMES, FUSION_NAMES, ObjectDamageRepresentationModel
from week12_train_stage1 import metrics_from_confusion_local, save_confusion_csv_local, save_confusion_plot_local


def class_counts(dataset: Stage2DamageDataset) -> torch.Tensor:
    counts = torch.zeros(len(STAGE2_CLASS_NAMES), dtype=torch.float32)
    for sample in dataset.samples:
        counts[int(sample["label"]) - 1] += 1.0
    return counts


def weighted_sampler(dataset: Stage2DamageDataset, counts: torch.Tensor) -> WeightedRandomSampler:
    class_weights = 1.0 / counts.clamp_min(1.0)
    sample_weights = torch.tensor([float(class_weights[int(dataset[index]["label"])]) for index in range(len(dataset))])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def run_epoch(
    model: ObjectDamageRepresentationModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float], torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((len(STAGE2_CLASS_NAMES), len(STAGE2_CLASS_NAMES)), dtype=torch.long)
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
            confusion += confusion_matrix(torch.argmax(logits.detach().cpu(), dim=1), labels.cpu(), len(STAGE2_CLASS_NAMES))
    return total_loss / max(len(loader), 1), metrics_from_confusion_local(confusion, STAGE2_CLASS_NAMES), confusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 12: train hierarchical stage 2 damaged-class classifier.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "week12_hierarchical_stage2")
    parser.add_argument("--backbone", choices=BACKBONE_NAMES, default="resnet34")
    parser.add_argument("--fusion", choices=FUSION_NAMES, default="gated")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Stage2DamageDataset(BuildingDamageDataset(args.dataset_root, "train", augment=args.augment_train))
    val_dataset = Stage2DamageDataset(BuildingDamageDataset(args.dataset_root, "val"))
    counts = class_counts(train_dataset)
    sampler = weighted_sampler(train_dataset, counts) if args.weighted_sampler else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = ObjectDamageRepresentationModel(args.backbone, len(STAGE2_CLASS_NAMES), args.embedding_dim, args.fusion, args.pretrained).to(device)
    weights = (counts.sum() / counts.clamp_min(1.0))
    criterion = nn.CrossEntropyLoss(weight=(weights / weights.mean()).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro_f1 = -1.0
    for directory in [args.results_dir / "metrics", args.results_dir / "confusion_matrices", args.results_dir / "checkpoints"]:
        directory.mkdir(parents=True, exist_ok=True)
    save_json(
        {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "class_names": STAGE2_CLASS_NAMES, "train_class_counts": counts.tolist()},
        args.results_dir / "config" / "training_config.json",
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, metrics, confusion = run_epoch(model, val_loader, criterion, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} macro_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            save_json({"best_epoch": epoch, "val_loss": val_loss, **metrics}, args.results_dir / "metrics" / "final_metrics.json")
            save_confusion_csv_local(confusion, STAGE2_CLASS_NAMES, args.results_dir / "confusion_matrices" / "confusion_matrix.csv")
            save_confusion_plot_local(confusion, STAGE2_CLASS_NAMES, args.results_dir / "confusion_matrices" / "confusion_matrix.png")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "fusion": args.fusion,
                    "embedding_dim": args.embedding_dim,
                    "class_names": STAGE2_CLASS_NAMES,
                },
                args.results_dir / "checkpoints" / "week12_stage2_best.pt",
            )


if __name__ == "__main__":
    main()

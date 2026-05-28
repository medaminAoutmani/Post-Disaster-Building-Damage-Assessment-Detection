"""Train Week 12 object-level representation models with CE, ArcFace, or SupCon."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
for path in [CURRENT_DIR, WEEK11_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset
from week11_train_classifier import (
    build_loss_weights,
    build_weighted_sampler,
    cap_dataset_per_class,
    class_counts,
    confusion_matrix,
    metrics_from_confusion,
    save_confusion_csv,
    save_confusion_plot,
    save_json,
)
from week12_model_backbones import BACKBONE_NAMES, FUSION_NAMES, ArcMarginProduct, ObjectDamageRepresentationModel, SupConLoss


def save_history(history: list[dict[str, float | int]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def forward_loss(
    model: ObjectDamageRepresentationModel,
    batch: dict,
    criterion: nn.Module,
    device: torch.device,
    loss_type: str,
    arc_head: ArcMarginProduct | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["label"].to(device).long()
    pre = batch["pre"].to(device)
    post = batch["post"].to(device)
    diff = batch["diff"].to(device)
    logits, embeddings = model(pre, post, diff, return_embedding=True)
    if loss_type == "arcface":
        if arc_head is None:
            raise ValueError("ArcFace loss requires arc_head.")
        margin_logits = arc_head(embeddings, labels)
        loss = criterion(margin_logits, labels)
        logits = arc_head.cosine_logits(embeddings)
    elif loss_type == "supcon":
        loss = criterion(embeddings, labels)
    else:
        loss = criterion(logits, labels)
    return loss, logits


def run_epoch(
    model: ObjectDamageRepresentationModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    loss_type: str,
    optimizer: torch.optim.Optimizer | None = None,
    arc_head: ArcMarginProduct | None = None,
) -> tuple[float, dict[str, float], torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    if arc_head is not None:
        arc_head.train(training)
    total_loss = 0.0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in dataloader:
            labels = batch["label"].long()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, logits = forward_loss(model, batch, criterion, device, loss_type, arc_head)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            predictions = torch.argmax(logits.detach().cpu(), dim=1)
            confusion += confusion_matrix(predictions, labels, len(CLASS_NAMES))
    return total_loss / max(len(dataloader), 1), metrics_from_confusion(confusion), confusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 12: train embedding-centric object-level damage model.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "week12_backbone")
    parser.add_argument("--backbone", choices=BACKBONE_NAMES, default="resnet34")
    parser.add_argument("--fusion", choices=FUSION_NAMES, default="concat")
    parser.add_argument("--loss-type", choices=["ce", "arcface", "supcon"], default="ce")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--class-weight-mode", choices=["none", "inverse", "effective"], default="effective")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--max-train-per-class", type=int, nargs=4, default=None, metavar=("NO_DAMAGE", "MINOR", "MAJOR", "DESTROYED"))
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BuildingDamageDataset(args.dataset_root, "train", augment=args.augment_train)
    val_dataset = BuildingDamageDataset(args.dataset_root, "val")
    if args.max_train_per_class is not None:
        caps = [None if value < 0 else value for value in args.max_train_per_class]
        cap_dataset_per_class(train_dataset, caps, args.seed)
    train_counts = class_counts(train_dataset)
    val_counts = class_counts(val_dataset)
    sampler = build_weighted_sampler(train_dataset, train_counts) if args.weighted_sampler else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ObjectDamageRepresentationModel(
        backbone=args.backbone,
        num_classes=len(CLASS_NAMES),
        embedding_dim=args.embedding_dim,
        fusion=args.fusion,
        pretrained=args.pretrained,
    ).to(device)
    arc_head = None
    if args.loss_type == "arcface":
        arc_head = ArcMarginProduct(args.embedding_dim, len(CLASS_NAMES), scale=args.arcface_scale, margin=args.arcface_margin).to(device)
        criterion: nn.Module = nn.CrossEntropyLoss()
        parameters = list(model.parameters()) + list(arc_head.parameters())
    elif args.loss_type == "supcon":
        criterion = SupConLoss(temperature=args.supcon_temperature)
        parameters = list(model.parameters())
    else:
        loss_weights = build_loss_weights(train_counts, args.class_weight_mode)
        criterion = nn.CrossEntropyLoss(weight=None if loss_weights is None else loss_weights.to(device))
        parameters = list(model.parameters())

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    metrics_dir = args.results_dir / "metrics"
    confusion_dir = args.results_dir / "confusion_matrices"
    checkpoint_dir = args.results_dir / "checkpoints"
    for directory in [metrics_dir, confusion_dir, checkpoint_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "week": 12,
        "goal": "embedding-centric object-level damage representation learning",
        "dataset_root": str(args.dataset_root),
        "backbone": args.backbone,
        "fusion": args.fusion,
        "loss_type": args.loss_type,
        "embedding_dim": args.embedding_dim,
        "class_weight_mode": args.class_weight_mode,
        "weighted_sampler": args.weighted_sampler,
        "augment_train": args.augment_train,
        "max_train_per_class": args.max_train_per_class,
        "class_names": CLASS_NAMES,
        "train_class_counts": {name: int(train_counts[i].item()) for i, name in enumerate(CLASS_NAMES)},
        "val_class_counts": {name: int(val_counts[i].item()) for i, name in enumerate(CLASS_NAMES)},
        "device": str(device),
    }
    save_json(config, args.results_dir / "config" / "training_config.json")

    best_macro_f1 = -1.0
    best_confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    history: list[dict[str, float | int]] = []
    print(f"device={device} backbone={args.backbone} fusion={args.fusion} loss_type={args.loss_type}")
    print(f"train_class_counts={config['train_class_counts']}")
    print(f"val_class_counts={config['val_class_counts']}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = run_epoch(model, train_loader, criterion, device, args.loss_type, optimizer, arc_head)
        eval_loss_type = "arcface" if args.loss_type == "arcface" else "ce"
        eval_criterion = criterion if args.loss_type != "supcon" else nn.CrossEntropyLoss()
        eval_arc_head = arc_head if args.loss_type == "arcface" else None
        val_loss, val_metrics, val_confusion = run_epoch(model, val_loader, eval_criterion, device, eval_loss_type, None, eval_arc_head)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} minor_f1={val_metrics['f1_minor_damage']:.4f} "
            f"major_f1={val_metrics['f1_major_damage']:.4f}"
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
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_confusion = val_confusion
            save_json({"best_epoch": epoch, "val_loss": val_loss, **val_metrics}, metrics_dir / "final_metrics.json")
            save_confusion_csv(best_confusion, confusion_dir / "confusion_matrix.csv")
            save_confusion_plot(best_confusion, confusion_dir / "confusion_matrix.png")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arc_head_state_dict": None if arc_head is None else arc_head.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "class_names": CLASS_NAMES,
                    "backbone": args.backbone,
                    "fusion": args.fusion,
                    "loss_type": args.loss_type,
                    "embedding_dim": args.embedding_dim,
                    "arcface_scale": args.arcface_scale,
                    "arcface_margin": args.arcface_margin,
                },
                checkpoint_dir / f"week12_{args.backbone}_{args.fusion}_{args.loss_type}_best.pt",
            )


if __name__ == "__main__":
    main()

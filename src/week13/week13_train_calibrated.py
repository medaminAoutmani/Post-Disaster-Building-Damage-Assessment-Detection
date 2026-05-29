"""Train Week 13 calibration-aware ordinal object-level damage models."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
WEEK12_DIR = CURRENT_DIR.parent / "week12"
for path in [CURRENT_DIR, WEEK11_DIR, WEEK12_DIR]:
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
from week12_model_backbones import BACKBONE_NAMES, FUSION_NAMES, ObjectDamageRepresentationModel
from week13_losses import (
    CoralLoss,
    EarthMoverDistanceLoss,
    coral_logits_to_class_probs,
    coral_logits_to_classes,
    damaged_auroc,
    expected_calibration_error,
    mean_severity_distance,
)
from week13_models import MultiTaskDamageModel


LOSS_TYPES = ["ce", "label_smoothing", "emd", "coral", "regression", "multitask"]


def save_history(history: list[dict[str, float | int]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.loss_type == "multitask":
        return MultiTaskDamageModel(
            backbone=args.backbone,
            embedding_dim=args.embedding_dim,
            fusion=args.fusion,
            pretrained=args.pretrained,
            dropout=args.dropout,
        )
    output_dim = 3 if args.loss_type == "coral" else 1 if args.loss_type == "regression" else len(CLASS_NAMES)
    return ObjectDamageRepresentationModel(
        backbone=args.backbone,
        num_classes=output_dim,
        embedding_dim=args.embedding_dim,
        fusion=args.fusion,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )


def model_outputs(model: nn.Module, batch: dict, device: torch.device, loss_type: str) -> dict[str, torch.Tensor]:
    pre = batch["pre"].to(device)
    post = batch["post"].to(device)
    diff = batch["diff"].to(device)
    if loss_type == "multitask":
        return model(pre, post, diff)
    logits = model(pre, post, diff)
    return {"logits": logits.squeeze(1) if loss_type == "regression" else logits}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_type: str,
    criterion: nn.Module,
    multitask_presence_weight: float,
    multitask_regression_weight: float,
) -> torch.Tensor:
    if loss_type == "regression":
        return criterion(outputs["logits"], labels.float())
    if loss_type == "multitask":
        class_loss = criterion(outputs["logits"], labels)
        presence_target = (labels > 0).float()
        severity_target = labels.float()
        presence_loss = F.binary_cross_entropy_with_logits(outputs["presence_logit"], presence_target)
        severity_loss = F.smooth_l1_loss(outputs["severity"], severity_target)
        return class_loss + multitask_presence_weight * presence_loss + multitask_regression_weight * severity_loss
    return criterion(outputs["logits"], labels)


def predictions_and_probs(outputs: dict[str, torch.Tensor], loss_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_type == "coral":
        probs = coral_logits_to_class_probs(outputs["logits"])
        return coral_logits_to_classes(outputs["logits"]), probs
    if loss_type == "regression":
        predictions = torch.round(outputs["logits"]).long().clamp(0, len(CLASS_NAMES) - 1)
        probs = F.one_hot(predictions, num_classes=len(CLASS_NAMES)).float()
        return predictions, probs
    probs = F.softmax(outputs["logits"], dim=1)
    return torch.argmax(probs, dim=1), probs


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    loss_type: str,
    optimizer: torch.optim.Optimizer | None,
    multitask_presence_weight: float,
    multitask_regression_weight: float,
) -> tuple[float, dict[str, float], torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in dataloader:
            labels = batch["label"].to(device).long()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model_outputs(model, batch, device, loss_type)
            loss = compute_loss(outputs, labels, loss_type, criterion, multitask_presence_weight, multitask_regression_weight)
            if training:
                loss.backward()
                optimizer.step()
            predictions, probs = predictions_and_probs({key: value.detach() for key, value in outputs.items()}, loss_type)
            total_loss += float(loss.item())
            confusion += confusion_matrix(predictions.cpu(), labels.cpu(), len(CLASS_NAMES))
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    metrics = metrics_from_confusion(confusion)
    probs_cpu = torch.cat(all_probs)
    labels_cpu = torch.cat(all_labels)
    predictions_cpu = torch.cat(all_predictions)
    metrics["ece"] = expected_calibration_error(probs_cpu, labels_cpu)
    metrics["mean_severity_distance"] = mean_severity_distance(predictions_cpu, labels_cpu)
    metrics["damaged_auroc"] = damaged_auroc(probs_cpu, labels_cpu)
    return total_loss / max(len(dataloader), 1), metrics, confusion


def build_criterion(args: argparse.Namespace, train_counts: torch.Tensor) -> nn.Module:
    if args.loss_type == "coral":
        return CoralLoss()
    if args.loss_type == "emd":
        return EarthMoverDistanceLoss()
    if args.loss_type == "regression":
        return nn.SmoothL1Loss()
    loss_weights = build_loss_weights(train_counts, args.class_weight_mode)
    if args.loss_type == "label_smoothing":
        return nn.CrossEntropyLoss(
            weight=None if loss_weights is None else loss_weights,
            label_smoothing=args.label_smoothing,
        )
    return nn.CrossEntropyLoss(weight=None if loss_weights is None else loss_weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 13: train calibration-aware ordinal damage model.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "week13_calibrated")
    parser.add_argument("--backbone", choices=BACKBONE_NAMES, default="convnext_tiny")
    parser.add_argument("--fusion", choices=FUSION_NAMES, default="gated")
    parser.add_argument("--loss-type", choices=LOSS_TYPES, default="coral")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
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
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--multitask-presence-weight", type=float, default=0.5)
    parser.add_argument("--multitask-regression-weight", type=float, default=0.2)
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

    model = build_model(args).to(device)
    criterion = build_criterion(args, train_counts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_dir = args.results_dir / "metrics"
    confusion_dir = args.results_dir / "confusion_matrices"
    checkpoint_dir = args.results_dir / "checkpoints"
    for directory in [metrics_dir, confusion_dir, checkpoint_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "week": 13,
            "goal": "calibration-aware ordinal semantic damage learning",
            "dataset_root": str(args.dataset_root),
            "backbone": args.backbone,
            "fusion": args.fusion,
            "loss_type": args.loss_type,
            "embedding_dim": args.embedding_dim,
            "class_weight_mode": args.class_weight_mode,
            "weighted_sampler": args.weighted_sampler,
            "augment_train": args.augment_train,
            "class_names": CLASS_NAMES,
            "train_class_counts": {name: int(train_counts[i].item()) for i, name in enumerate(CLASS_NAMES)},
            "val_class_counts": {name: int(val_counts[i].item()) for i, name in enumerate(CLASS_NAMES)},
            "device": str(device),
        },
        args.results_dir / "config" / "training_config.json",
    )

    best_macro_f1 = -1.0
    history: list[dict[str, float | int]] = []
    print(f"device={device} backbone={args.backbone} fusion={args.fusion} loss_type={args.loss_type}")
    print(f"train_class_counts={ {name: int(train_counts[i].item()) for i, name in enumerate(CLASS_NAMES)} }")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.loss_type,
            optimizer,
            args.multitask_presence_weight,
            args.multitask_regression_weight,
        )
        val_loss, val_metrics, val_confusion = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.loss_type,
            None,
            args.multitask_presence_weight,
            args.multitask_regression_weight,
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} minor_f1={val_metrics['f1_minor_damage']:.4f} "
            f"ece={val_metrics['ece']:.4f} severity_dist={val_metrics['mean_severity_distance']:.4f}"
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
            save_json({"best_epoch": epoch, "val_loss": val_loss, **val_metrics}, metrics_dir / "final_metrics.json")
            save_confusion_csv(val_confusion, confusion_dir / "confusion_matrix.csv")
            save_confusion_plot(val_confusion, confusion_dir / "confusion_matrix.png")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "class_names": CLASS_NAMES,
                    "backbone": args.backbone,
                    "fusion": args.fusion,
                    "loss_type": args.loss_type,
                    "embedding_dim": args.embedding_dim,
                    "dropout": args.dropout,
                },
                checkpoint_dir / f"week13_{args.backbone}_{args.fusion}_{args.loss_type}_best.pt",
            )


if __name__ == "__main__":
    main()

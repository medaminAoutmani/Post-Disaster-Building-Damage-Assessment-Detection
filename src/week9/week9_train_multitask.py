"""Train Week 9 multi-task Siamese damage segmentation models."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
WEEK7_DIR = SRC_DIR / "week7"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(WEEK7_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK7_DIR))

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from week2_dataset import read_split_file
from week6.week6_augmentations import get_week6_transforms
from week6.week6_sampler import build_weighted_sampler
from week6.week6_scheduler import build_scheduler
from week6.week6_utils import collect_environment_metadata, ensure_experiment_dirs, fix_seed, get_device, save_checkpoint, save_json, setup_file_logger
from week7_callbacks import EarlyStopping
from week7_metrics import CLASS_NAMES, save_confusion_matrix_csv, save_per_class_metrics
from week9_dataset import XBDMultiTaskCombinedDataset, XBDMultiTaskSampleDataset
from week9_losses import MultiTaskDamageLoss, damage_confusion_excluding_background
from week9_metrics import binary_confusion_from_logits, confusion_matrix_from_logits, multitask_metrics
from week9_model_multitask_siamese import MultiTaskSiameseResNet50UNet

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore[assignment]


MOROCCO_ADAPTATION_KEYWORDS = ["earthquake", "flood", "flooding", "wildfire", "fire"]
EXPERIMENTS = {
    "multitask_difference": {"fusion": "difference", "attention": "no_attention"},
    "multitask_cbam_difference": {"fusion": "difference", "attention": "cbam"},
    "multitask_nonlocal_difference": {"fusion": "difference", "attention": "non_local"},
}


def filter_ids(sample_ids: list[str], keywords: list[str] | None) -> list[str]:
    if not keywords:
        return sample_ids
    lowered = [keyword.lower() for keyword in keywords]
    return [sample_id for sample_id in sample_ids if any(keyword in sample_id.lower() for keyword in lowered)]


def collect_extra_sample_ids(extra_data_dir: Path, split: str) -> list[str]:
    labels_dir = extra_data_dir / split / "labels"
    if not labels_dir.exists():
        return []
    return sorted(path.name.replace("_post_disaster.json", "") for path in labels_dir.glob("*_post_disaster.json"))


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    keywords = MOROCCO_ADAPTATION_KEYWORDS if args.morocco_adaptation else args.disaster_keywords
    train_ids = filter_ids(read_split_file(args.split_dir / "train.txt"), keywords)
    val_ids = filter_ids(read_split_file(args.split_dir / "val.txt"), keywords)
    if args.max_train_samples:
        train_ids = train_ids[: args.max_train_samples]
    if args.max_val_samples:
        val_ids = val_ids[: args.max_val_samples]

    train_datasets = [
        XBDMultiTaskSampleDataset(
            args.data_dir,
            train_ids,
            split=args.xbd_split,
            transform=get_week6_transforms(args.image_size, train=True),
        )
    ]
    extra_ids = collect_extra_sample_ids(args.extra_data_dir, args.xbd_split) if args.use_week8_extra else []
    if args.max_extra_samples:
        extra_ids = extra_ids[: args.max_extra_samples]
    if extra_ids:
        train_datasets.append(
            XBDMultiTaskSampleDataset(
                args.extra_data_dir,
                extra_ids,
                split=args.xbd_split,
                transform=get_week6_transforms(args.image_size, train=True),
            )
        )
    train_dataset = XBDMultiTaskCombinedDataset(train_datasets)
    val_dataset = XBDMultiTaskSampleDataset(
        args.data_dir,
        val_ids,
        split=args.xbd_split,
        transform=get_week6_transforms(args.image_size, train=False),
    )
    sampler = build_weighted_sampler(train_dataset, args.damage_class_weights) if args.sampler == "weighted" else None
    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    )


def _trainable_parameters(modules: list[nn.Module], seen: set[int]) -> list[nn.Parameter]:
    parameters = []
    for module in modules:
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter.requires_grad and parameter_id not in seen:
                parameters.append(parameter)
                seen.add(parameter_id)
    return parameters


def build_optimizer(model: nn.Module, encoder_lr: float, fusion_lr: float, decoder_lr: float) -> torch.optim.Optimizer:
    seen: set[int] = set()
    encoder_parameters = _trainable_parameters([model.encoder], seen)
    fusion_parameters = _trainable_parameters([model.fusions, model.bottleneck_attention], seen)
    decoder_modules = [
        model.decoder4,
        model.decoder3,
        model.decoder2,
        model.decoder1,
        model.final_upsample,
        model.pre_building_head,
        model.post_building_head,
        model.damage_head,
    ]
    decoder_parameters = _trainable_parameters(decoder_modules, seen)
    groups = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": encoder_lr, "name": "encoder"})
    if fusion_parameters:
        groups.append({"params": fusion_parameters, "lr": fusion_lr, "name": "fusion_attention"})
    if decoder_parameters:
        groups.append({"params": decoder_parameters, "lr": decoder_lr, "name": "decoder_heads"})
    return torch.optim.AdamW(groups)


def run_epoch(model, dataloader, criterion, device, optimizer=None, scaler=None, use_amp: bool = False, grad_clip_norm: float = 1.0):
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "pre_building_loss": 0.0, "post_building_loss": 0.0, "damage_loss": 0.0}
    pre_confusion = torch.zeros((2, 2), dtype=torch.long)
    post_confusion = torch.zeros((2, 2), dtype=torch.long)
    damage_confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    building_damage_confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in dataloader:
            tensor_batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                outputs = model(tensor_batch["pre_image"], tensor_batch["post_image"])
                loss, loss_parts = criterion(outputs, tensor_batch)
            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            totals["loss"] += float(loss.detach().item())
            for key in ["pre_building_loss", "post_building_loss", "damage_loss"]:
                totals[key] += loss_parts[key]
            pre_confusion += binary_confusion_from_logits(outputs["pre_building_logits"].detach().cpu(), tensor_batch["pre_building_mask"].detach().cpu())
            post_confusion += binary_confusion_from_logits(outputs["post_building_logits"].detach().cpu(), tensor_batch["post_building_mask"].detach().cpu())
            damage_confusion += confusion_matrix_from_logits(outputs["damage_logits"].detach().cpu(), tensor_batch["damage_mask"].detach().cpu(), len(CLASS_NAMES))
            building_damage_confusion += damage_confusion_excluding_background(outputs["damage_logits"].detach().cpu(), tensor_batch["damage_mask"].detach().cpu(), len(CLASS_NAMES))

    denominator = max(1, len(dataloader))
    losses = {key: value / denominator for key, value in totals.items()}
    metrics = multitask_metrics(pre_confusion, post_confusion, damage_confusion, building_damage_confusion)
    return losses, metrics, damage_confusion


def save_history(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_experiment(args: argparse.Namespace) -> Path:
    preset = EXPERIMENTS[args.experiment]
    fusion = args.fusion or preset["fusion"]
    attention = args.attention or preset["attention"]
    experiment_name = args.experiment_name or f"experiment_{args.experiment}"
    fix_seed(args.seed)
    device = get_device()
    experiment_dir = ensure_experiment_dirs(args.results_root, experiment_name)
    logger = setup_file_logger(experiment_dir / "logs" / "training_console.log")
    train_loader, val_loader = build_dataloaders(args)
    model = MultiTaskSiameseResNet50UNet(
        fusion_strategy=fusion,
        attention_type=attention,
        pretrained=not args.no_pretrained,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    optimizer = build_optimizer(model, args.encoder_lr, args.fusion_lr, args.decoder_lr)
    criterion = MultiTaskDamageLoss(
        damage_loss_name=args.damage_loss,
        damage_class_weights=args.damage_class_weights,
        building_class_weights=args.building_class_weights,
        lambda_pre=args.lambda_pre,
        lambda_post=args.lambda_post,
        lambda_damage=args.lambda_damage,
    ).to(device)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(str(experiment_dir / "logs" / "tensorboard")) if SummaryWriter is not None and not args.no_tensorboard else None
    save_json(
        {
            "experiment": args.experiment,
            "fusion": fusion,
            "attention": attention,
            "epochs": args.epochs,
            "scheduler": args.scheduler,
            "encoder_lr": args.encoder_lr,
            "fusion_lr": args.fusion_lr,
            "decoder_lr": args.decoder_lr,
            "damage_loss": args.damage_loss,
            "damage_class_weights": args.damage_class_weights,
            "building_class_weights": args.building_class_weights,
            "lambda_pre": args.lambda_pre,
            "lambda_post": args.lambda_post,
            "lambda_damage": args.lambda_damage,
            "grad_clip_norm": args.grad_clip_norm,
            "use_week8_extra": args.use_week8_extra,
            "environment": collect_environment_metadata(model),
        },
        experiment_dir / "config" / "training_config.json",
    )

    early_stopping = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    history = []
    best_metrics = {}
    best_confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_losses, train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, args.grad_clip_norm)
        val_losses, val_metrics, val_confusion = run_epoch(model, val_loader, criterion, device, use_amp=use_amp, grad_clip_norm=args.grad_clip_norm)
        score = val_metrics["damage_mean_dice"]
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(score)
            else:
                scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_losses["loss"],
            "val_loss": val_losses["loss"],
            "train_damage_dice": train_metrics["damage_mean_dice"],
            "val_damage_dice": val_metrics["damage_mean_dice"],
            "val_building_only_damage_dice": val_metrics["building_only_damage_mean_dice"],
            "val_rare_class_recall": val_metrics["damage_rare_class_recall"],
            "val_pre_building_dice": val_metrics["pre_building_dice"],
            "val_post_building_dice": val_metrics["post_building_dice"],
        }
        history.append(record)
        save_history(history, experiment_dir / "metrics" / "training_log.csv")
        if writer is not None:
            for key, value in record.items():
                if key != "epoch":
                    writer.add_scalar(key, float(value), epoch)
        logger.info(
            "epoch=%s train_loss=%.4f val_loss=%.4f damage_dice=%.4f building_damage_dice=%.4f rare_recall=%.4f",
            epoch,
            train_losses["loss"],
            val_losses["loss"],
            val_metrics["damage_mean_dice"],
            val_metrics["building_only_damage_mean_dice"],
            val_metrics["damage_rare_class_recall"],
        )

        if score >= early_stopping.best_score + args.early_stopping_min_delta:
            best_metrics = {
                "best_epoch": epoch,
                "training_time_seconds": time.perf_counter() - start,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            best_confusion = val_confusion
            save_json(best_metrics, experiment_dir / "metrics" / "final_metrics.json")
            save_per_class_metrics(
                {key.replace("damage_", ""): value for key, value in val_metrics.items() if key.startswith("damage_")},
                experiment_dir / "metrics" / "damage_per_class_metrics.csv",
            )
            save_confusion_matrix_csv(best_confusion, experiment_dir / "confusion_matrices" / "damage_confusion_matrix.csv")
            save_checkpoint(model, experiment_dir / "checkpoints" / "best_model.pt", epoch, val_metrics, {"model": "multitask_siamese_resnet50_unet", "fusion": fusion, "attention": attention})
        save_checkpoint(model, experiment_dir / "checkpoints" / "last_model.pt", epoch, val_metrics)
        if early_stopping.step(score):
            break

    if writer is not None:
        writer.close()
    return experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 9 multi-task Siamese experiments.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="multitask_cbam_difference")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--attention", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--extra-data-dir", type=Path, default=Path("data") / "week8_extra")
    parser.add_argument("--use-week8-extra", action="store_true")
    parser.add_argument("--xbd-split", default="train")
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--results-root", type=Path, default=Path("results") / "week9")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--encoder-lr", type=float, default=0.0001)
    parser.add_argument("--fusion-lr", type=float, default=0.0003)
    parser.add_argument("--decoder-lr", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-extra-samples", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--damage-loss", default="cross_entropy_dice")
    parser.add_argument("--sampler", choices=["shuffle", "weighted"], default="weighted")
    parser.add_argument("--scheduler", default="warmup_cosine")
    parser.add_argument("--damage-class-weights", type=float, nargs=5, default=[1.0, 2.0, 8.0, 12.0, 12.0])
    parser.add_argument("--building-class-weights", type=float, nargs=2, default=[1.0, 10.0])
    parser.add_argument("--lambda-pre", type=float, default=1.0)
    parser.add_argument("--lambda-post", type=float, default=1.0)
    parser.add_argument("--lambda-damage", type=float, default=3.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--morocco-adaptation", action="store_true")
    parser.add_argument("--disaster-keywords", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = train_experiment(args)
    print(f"Finished {experiment_dir}")


if __name__ == "__main__":
    main()

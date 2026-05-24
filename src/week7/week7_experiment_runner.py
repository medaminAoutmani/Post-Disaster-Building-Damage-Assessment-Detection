"""Week 7 temporal Siamese experiment runner."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from week2_dataset import read_split_file
from week6.week6_augmentations import get_week6_transforms
from week6.week6_losses import build_loss
from week6.week6_sampler import build_weighted_sampler
from week6.week6_scheduler import build_scheduler
from week6.week6_utils import collect_environment_metadata, ensure_experiment_dirs, fix_seed, get_device, save_checkpoint, save_json, setup_file_logger
from week7_analysis import summarize_week7_experiments
from week7_callbacks import EarlyStopping
from week7_dataset import XBDTemporalDamageDataset
from week7_failure_analysis import write_failure_analysis_template
from week7_metrics import CLASS_NAMES, confusion_matrix_from_logits, metrics_from_confusion_matrix, save_confusion_matrix_csv, save_per_class_metrics
from week7_model_siamese_attention_unet import SiameseAttentionUNet
from week7_model_siamese_deeplab import SiameseDeepLabDamage
from week7_model_siamese_resnet50_unet import SiameseResNet50UNet
from week7_utils import create_week7_results_tree

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore[assignment]


EXPERIMENTS = {
    "siamese_concat": {"fusion": "concat", "attention": "no_attention", "model": "siamese_resnet50_unet"},
    "siamese_difference": {"fusion": "difference", "attention": "no_attention", "model": "siamese_resnet50_unet"},
    "siamese_concat_difference": {"fusion": "concat_difference", "attention": "no_attention", "model": "siamese_resnet50_unet"},
    "siamese_gated": {"fusion": "gated_fusion", "attention": "no_attention", "model": "siamese_resnet50_unet"},
    "siamese_bottleneck_attention": {"fusion": "concat", "attention": "bottleneck_attention", "model": "siamese_attention_unet"},
    "siamese_cbam": {"fusion": "concat", "attention": "cbam", "model": "siamese_attention_unet"},
    "siamese_nonlocal": {"fusion": "concat", "attention": "non_local", "model": "siamese_attention_unet"},
}

MOROCCO_ADAPTATION_KEYWORDS = ["earthquake", "flood", "flooding", "wildfire", "fire"]


def filter_ids(sample_ids: list[str], keywords: list[str] | None) -> list[str]:
    if not keywords:
        return sample_ids
    lowered = [keyword.lower() for keyword in keywords]
    return [sample_id for sample_id in sample_ids if any(keyword in sample_id.lower() for keyword in lowered)]


def build_model(name: str, fusion: str, attention: str, pretrained: bool) -> nn.Module:
    if name == "siamese_attention_unet":
        return SiameseAttentionUNet(fusion_strategy=fusion, attention_type=attention, pretrained=pretrained)
    if name == "siamese_deeplab":
        return SiameseDeepLabDamage(fusion_strategy=fusion, pretrained=pretrained)
    return SiameseResNet50UNet(fusion_strategy=fusion, attention_type=attention, pretrained=pretrained)


def build_dataloaders(args: argparse.Namespace, sampler_name: str) -> tuple[DataLoader, DataLoader]:
    keywords = MOROCCO_ADAPTATION_KEYWORDS if args.morocco_adaptation else args.disaster_keywords
    train_ids = filter_ids(read_split_file(args.split_dir / "train.txt"), keywords)
    val_ids = filter_ids(read_split_file(args.split_dir / "val.txt"), keywords)
    if args.max_train_samples:
        train_ids = train_ids[: args.max_train_samples]
    if args.max_val_samples:
        val_ids = val_ids[: args.max_val_samples]
    train_dataset = XBDTemporalDamageDataset(
        args.data_dir,
        train_ids,
        split=args.xbd_split,
        transform=get_week6_transforms(args.image_size, train=True),
    )
    val_dataset = XBDTemporalDamageDataset(
        args.data_dir,
        val_ids,
        split=args.xbd_split,
        transform=get_week6_transforms(args.image_size, train=False),
    )
    sampler = build_weighted_sampler(train_dataset, args.class_weights) if sampler_name == "weighted" else None
    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    )


def run_epoch(model, dataloader, criterion, device, optimizer=None, scaler=None, use_amp: bool = False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in dataloader:
            pre = batch["pre_image"].to(device)
            post = batch["post_image"].to(device)
            masks = batch["mask"].to(device).long()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                logits = model(pre, post)
                loss = criterion(logits, masks)
            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.item())
            confusion += confusion_matrix_from_logits(logits.detach().cpu(), masks.detach().cpu(), len(CLASS_NAMES))
    return total_loss / max(1, len(dataloader)), metrics_from_confusion_matrix(confusion), confusion


def save_history(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


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
    encoder_modules = [module for name, module in model.named_children() if name.startswith("encoder")]
    fusion_modules = [module for name, module in model.named_children() if name in {"fusions", "bottleneck_attention"}]
    decoder_modules = [module for name, module in model.named_children() if name not in {"encoder_pre", "encoder_post", "fusions", "bottleneck_attention"}]

    parameter_groups = []
    encoder_parameters = _trainable_parameters(encoder_modules, seen)
    fusion_parameters = _trainable_parameters(fusion_modules, seen)
    decoder_parameters = _trainable_parameters(decoder_modules, seen)
    remaining_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in seen]

    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": encoder_lr, "name": "encoder"})
    if fusion_parameters:
        parameter_groups.append({"params": fusion_parameters, "lr": fusion_lr, "name": "fusion_attention"})
    if decoder_parameters:
        parameter_groups.append({"params": decoder_parameters, "lr": decoder_lr, "name": "decoder"})
    if remaining_parameters:
        parameter_groups.append({"params": remaining_parameters, "lr": decoder_lr, "name": "remaining"})
    return torch.optim.AdamW(parameter_groups)


def train_experiment(args: argparse.Namespace) -> Path:
    preset = EXPERIMENTS[args.experiment]
    fusion = args.fusion or preset["fusion"]
    attention = args.attention or preset["attention"]
    model_name = args.model or preset["model"]
    experiment_name = args.experiment_name or f"experiment_{args.experiment}"
    fix_seed(args.seed)
    device = get_device()
    experiment_dir = ensure_experiment_dirs(args.results_root, experiment_name)
    write_failure_analysis_template(experiment_dir / "analysis" / "failure_analysis.md")
    logger = setup_file_logger(experiment_dir / "logs" / "training_console.log")
    train_loader, val_loader = build_dataloaders(args, args.sampler)
    model = build_model(model_name, fusion, attention, pretrained=not args.no_pretrained).to(device)
    encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
    fusion_lr = args.fusion_lr if args.fusion_lr is not None else args.lr
    decoder_lr = args.decoder_lr if args.decoder_lr is not None else args.lr
    optimizer = build_optimizer(model, encoder_lr, fusion_lr, decoder_lr)
    criterion = build_loss(args.loss, args.class_weights).to(device)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(str(experiment_dir / "logs" / "tensorboard")) if SummaryWriter is not None and not args.no_tensorboard else None
    save_json(
        {
            "experiment": args.experiment,
            "model": model_name,
            "fusion": fusion,
            "attention": attention,
            "loss": args.loss,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "lr": args.lr,
            "encoder_lr": encoder_lr,
            "fusion_lr": fusion_lr,
            "decoder_lr": decoder_lr,
            "epochs": args.epochs,
            "morocco_adaptation": args.morocco_adaptation,
            "environment": collect_environment_metadata(model),
        },
        experiment_dir / "config" / "training_config.json",
    )
    early_stopping = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    start = time.perf_counter()
    history = []
    best_metrics = {}
    best_confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_metrics, val_confusion = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_metrics["mean_dice"])
            else:
                scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_metrics["mean_dice"],
            "val_dice": val_metrics["mean_dice"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_rare_class_recall": val_metrics.get("rare_class_recall", 0.0),
        }
        history.append(record)
        save_history(history, experiment_dir / "metrics" / "training_log.csv")
        if writer is not None:
            for key, value in record.items():
                if key != "epoch":
                    writer.add_scalar(key, float(value), epoch)
        logger.info("epoch=%s train_loss=%.4f val_loss=%.4f val_dice=%.4f", epoch, train_loss, val_loss, val_metrics["mean_dice"])
        if val_metrics["mean_dice"] >= early_stopping.best_score + args.early_stopping_min_delta:
            best_metrics = {
                "best_epoch": epoch,
                "training_time_seconds": time.perf_counter() - start,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            best_confusion = val_confusion
            save_json(best_metrics, experiment_dir / "metrics" / "final_metrics.json")
            save_per_class_metrics(val_metrics, experiment_dir / "metrics" / "per_class_metrics.csv")
            save_confusion_matrix_csv(best_confusion, experiment_dir / "confusion_matrices" / "confusion_matrix.csv")
            save_checkpoint(model, experiment_dir / "checkpoints" / "best_model.pt", epoch, val_metrics, {"model": model_name, "fusion": fusion, "attention": attention})
        save_checkpoint(model, experiment_dir / "checkpoints" / "last_model.pt", epoch, val_metrics)
        if early_stopping.step(val_metrics["mean_dice"]):
            break
    if writer is not None:
        writer.close()
    if best_metrics:
        save_json(best_metrics, experiment_dir / "metrics" / "final_metrics.json")
        save_confusion_matrix_csv(best_confusion, experiment_dir / "confusion_matrices" / "confusion_matrix.csv")
    summarize_week7_experiments(args.results_root, args.results_root / "comparative_analysis" / "week7_summary.csv")
    return experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 7 temporal Siamese experiments.")
    parser.add_argument("--scaffold-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="siamese_concat")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--attention", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--xbd-split", default="train")
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--results-root", type=Path, default=Path("results") / "week7")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--fusion-lr", type=float, default=None)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--loss", default="cross_entropy_dice")
    parser.add_argument("--sampler", choices=["shuffle", "weighted"], default="shuffle")
    parser.add_argument("--scheduler", default="reduce_on_plateau")
    parser.add_argument("--class-weights", type=float, nargs=5, default=[1.0, 0.2, 1.5, 1.7, 1.8])
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--morocco-adaptation", action="store_true")
    parser.add_argument("--disaster-keywords", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_week7_results_tree(args.results_root)
    if args.scaffold_only:
        print(f"Created Week 7 results scaffold at {args.results_root}")
        return
    if args.summarize_only:
        output = args.results_root / "comparative_analysis" / "week7_summary.csv"
        summarize_week7_experiments(args.results_root, output)
        print(f"Updated {output}")
        return
    experiment_dir = train_experiment(args)
    print(f"Finished {experiment_dir}")


if __name__ == "__main__":
    main()

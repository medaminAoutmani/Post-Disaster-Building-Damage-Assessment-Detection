"""Week 6 experiment runner and result-tree orchestrator."""

from __future__ import annotations

import argparse
import csv
import json
import random
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

from week2_dataset import XBDChangeDataset, read_split_file
from week3_train import limit_dataloader
from week5_train import DEFAULT_CLASS_WEIGHTS, build_optimizer
from week6_augmentations import get_week6_transforms
from week6_analysis import load_final_metrics, summarize_experiments, write_analysis_template
from week6_losses import build_loss
from week6_metrics import (
    CLASS_NAMES,
    batch_boundary_iou,
    confusion_matrix_from_logits,
    metrics_from_confusion_matrix,
    precision_recall_points,
    save_confusion_matrix_csv,
    save_per_class_metrics,
    save_precision_recall_csv,
)
from week6_model_attention_unet import AttentionUNet
from week6_model_deeplabv3plus import DeepLabV3Damage
from week6_model_resnet50_unet import ResNet50UNet
from week6_model_unetplusplus import UNetPlusPlus
from week6_sampler import build_weighted_sampler
from week6_scheduler import build_scheduler
from week6_utils import (
    ExperimentConfig,
    collect_environment_metadata,
    create_week6_results_tree,
    ensure_experiment_dirs,
    fix_seed,
    get_device,
    save_checkpoint,
    save_experiment_config,
    save_json,
    setup_file_logger,
)
from week6_visualization import save_confidence_map, save_error_heatmap, save_overlay, save_prediction_panel, tensor_to_rgb_pair

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore[assignment]

try:
    import optuna
except Exception:
    optuna = None  # type: ignore[assignment]


EXPERIMENTS = {
    "baseline": {"model": "resnet34_unet", "loss": "cross_entropy_dice", "sampler": "shuffle"},
    "focal_loss": {"model": "resnet34_unet", "loss": "focal", "sampler": "shuffle"},
    "tversky_loss": {"model": "resnet34_unet", "loss": "tversky", "sampler": "shuffle"},
    "focal_tversky": {"model": "resnet34_unet", "loss": "focal_tversky", "sampler": "shuffle"},
    "weighted_sampler": {"model": "resnet34_unet", "loss": "cross_entropy_dice", "sampler": "weighted"},
    "attention_unet": {"model": "attention_unet", "loss": "focal", "sampler": "weighted"},
    "unetplusplus": {"model": "unetplusplus", "loss": "focal_tversky", "sampler": "weighted"},
    "deeplabv3": {"model": "deeplabv3", "loss": "cross_entropy_dice", "sampler": "weighted"},
    "resnet50": {"model": "resnet50_unet", "loss": "tversky", "sampler": "weighted"},
}

MOROCCO_ADAPTATION_KEYWORDS = ["earthquake", "flood", "flooding", "wildfire", "fire"]


def build_model(name: str, pretrained: bool = True, freeze_encoder: bool = False) -> nn.Module:
    normalized = name.lower()
    if normalized == "attention_unet":
        return AttentionUNet(out_channels=len(CLASS_NAMES))
    if normalized == "unetplusplus":
        return UNetPlusPlus(out_channels=len(CLASS_NAMES))
    if normalized == "deeplabv3":
        return DeepLabV3Damage(out_channels=len(CLASS_NAMES), pretrained=pretrained, freeze_backbone=freeze_encoder)
    if normalized == "resnet50_unet":
        return ResNet50UNet(out_channels=len(CLASS_NAMES), pretrained=pretrained, freeze_encoder=freeze_encoder)
    if normalized == "resnet34_unet":
        from week5_model import DamageResNet34UNet

        return DamageResNet34UNet(pretrained=pretrained, freeze_encoder=freeze_encoder)
    raise ValueError(f"Unknown model: {name}")


def filter_sample_ids_by_disaster(sample_ids: list[str], keywords: list[str] | None) -> list[str]:
    if not keywords:
        return sample_ids
    lowered_keywords = [keyword.lower() for keyword in keywords]
    return [sample_id for sample_id in sample_ids if any(keyword in sample_id.lower() for keyword in lowered_keywords)]


def build_dataloaders(args: argparse.Namespace, sampler_name: str) -> tuple[DataLoader, DataLoader]:
    train_ids = read_split_file(args.split_dir / "train.txt")
    val_ids = read_split_file(args.split_dir / "val.txt")
    disaster_keywords = MOROCCO_ADAPTATION_KEYWORDS if args.morocco_adaptation else args.disaster_keywords
    train_ids = filter_sample_ids_by_disaster(train_ids, disaster_keywords)
    val_ids = filter_sample_ids_by_disaster(val_ids, disaster_keywords)
    if not train_ids or not val_ids:
        raise ValueError(f"No train/val samples remain after disaster filtering: {disaster_keywords}")
    train_dataset = XBDChangeDataset(
        args.data_dir,
        train_ids,
        split="train",
        transform=get_week6_transforms(args.image_size, train=True, advanced=not args.no_advanced_augmentations),
        target_mode="multiclass",
        filter_empty=True,
    )
    val_dataset = XBDChangeDataset(
        args.data_dir,
        val_ids,
        split="train",
        transform=get_week6_transforms(args.image_size, train=False, advanced=False),
        target_mode="multiclass",
        filter_empty=True,
    )
    sampler = None
    shuffle = True
    if sampler_name == "weighted":
        sampler = build_weighted_sampler(train_dataset, args.class_weights)
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return (
        limit_dataloader(train_loader, args.max_train_samples, shuffle=shuffle and sampler is None),
        limit_dataloader(val_loader, args.max_val_samples, shuffle=False),
    )


def _sample_dice(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=1)
    scores = []
    for index in range(predictions.shape[0]):
        class_scores = []
        for class_index in range(1, len(CLASS_NAMES)):
            pred_mask = predictions[index] == class_index
            true_mask = masks[index] == class_index
            intersection = (pred_mask & true_mask).sum().float()
            denominator = pred_mask.sum().float() + true_mask.sum().float()
            class_scores.append((2.0 * intersection + 1e-7) / (denominator + 1e-7))
        scores.append(torch.stack(class_scores).mean())
    return torch.stack(scores)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    collect_examples: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    boundary_totals: dict[str, float] = {}
    boundary_batches = 0
    pr_rows: list[dict[str, float | str]] = []
    examples: list[dict[str, object]] = []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).long()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                logits = model(images)
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
            if not is_train:
                batch_boundary = batch_boundary_iou(logits.detach().cpu(), masks.detach().cpu(), len(CLASS_NAMES))
                for key, value in batch_boundary.items():
                    boundary_totals[key] = boundary_totals.get(key, 0.0) + value
                boundary_batches += 1
                pr_rows.extend(precision_recall_points(logits.detach().cpu(), masks.detach().cpu()))
                if collect_examples:
                    scores = _sample_dice(logits.detach().cpu(), masks.detach().cpu())
                    predictions = torch.argmax(logits.detach().cpu(), dim=1)
                    for item_index, score in enumerate(scores):
                        sample_id = batch.get("sample_id", [f"sample_{len(examples)}"])[item_index]
                        examples.append(
                            {
                                "sample_id": str(sample_id),
                                "score": float(score.item()),
                                "image": images[item_index].detach().cpu(),
                                "target": masks[item_index].detach().cpu(),
                                "prediction": predictions[item_index],
                                "logits": logits[item_index].detach().cpu(),
                            }
                        )
    metrics = metrics_from_confusion_matrix(confusion)
    if boundary_batches:
        metrics.update({key: value / boundary_batches for key, value in boundary_totals.items()})
    return total_loss / max(1, len(dataloader)), metrics, confusion, examples, pr_rows


def save_history_csv(history: list[dict[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_validation_examples(examples: list[dict[str, object]], experiment_dir: Path, epoch: int, max_each: int = 6) -> None:
    if not examples:
        return
    sorted_examples = sorted(examples, key=lambda item: float(item["score"]))
    groups = {
        "failure_cases": sorted_examples[:max_each],
        "best_examples": sorted_examples[-max_each:],
        "difficult_scenes": random.sample(examples, k=min(max_each, len(examples))),
    }
    for group_name, group_examples in groups.items():
        for item in group_examples:
            sample_id = str(item["sample_id"]).replace("/", "_").replace("\\", "_")
            pre_image, post_image = tensor_to_rgb_pair(item["image"])  # type: ignore[arg-type]
            target = item["target"].numpy()  # type: ignore[union-attr]
            prediction = item["prediction"].numpy()  # type: ignore[union-attr]
            stem = f"epoch_{epoch:03d}_{sample_id}_dice_{float(item['score']):.3f}"
            save_prediction_panel(
                pre_image,
                post_image,
                target,
                prediction,
                experiment_dir / "predictions" / group_name / f"{stem}.png",
            )
            save_confidence_map(item["logits"].unsqueeze(0), experiment_dir / "visualizations" / "confidence_maps" / f"{stem}.png")  # type: ignore[union-attr]
            save_error_heatmap(target, prediction, experiment_dir / "visualizations" / "error_heatmaps" / f"{stem}.png")
            save_overlay(post_image, prediction, experiment_dir / "visualizations" / "overlays" / f"{stem}.png")


def aggregate_precision_recall_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, float], list[dict[str, float | str]]] = {}
    for row in rows:
        key = (str(row["class"]), float(row["threshold"]))
        grouped.setdefault(key, []).append(row)
    aggregated = []
    for (class_name, threshold), group in sorted(grouped.items()):
        aggregated.append(
            {
                "class": class_name,
                "threshold": threshold,
                "precision": sum(float(row["precision"]) for row in group) / len(group),
                "recall": sum(float(row["recall"]) for row in group) / len(group),
            }
        )
    return aggregated


def save_multi_seed_summary(results_root: Path, experiment_prefix: str, seeds: list[int]) -> None:
    rows = []
    for seed in seeds:
        metrics = load_final_metrics(results_root / f"{experiment_prefix}_seed_{seed}")
        if metrics:
            rows.append(metrics)
    if not rows:
        return
    keys = ["val_mean_dice", "val_mean_iou", "val_macro_f1", "val_rare_class_recall", "val_dice_minor_damage", "val_dice_major_damage", "val_dice_destroyed"]
    summary = {"experiment_prefix": experiment_prefix, "seeds": seeds}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = variance ** 0.5
    save_json(summary, results_root / "comparative_analysis" / f"{experiment_prefix}_multi_seed_summary.json")


def clone_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def save_optuna_trials_csv(study, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
            **trial.params,
            **trial.user_attrs,
        }
        rows.append(row)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_optuna_hpo(args: argparse.Namespace) -> None:
    if optuna is None:
        raise ImportError("Optuna is not installed. Install requirements.txt or run: pip install optuna")

    hpo_dir = args.results_root / "hyperparameter_optimization" / args.optuna_study_name
    hpo_dir.mkdir(parents=True, exist_ok=True)
    storage = args.optuna_storage or f"sqlite:///{(hpo_dir / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.optuna_study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial) -> float:
        trial_args = clone_args(args)
        trial_args.optuna_study = False
        trial_args.seeds = None
        trial_args.k_folds = 0
        trial_args.epochs = args.optuna_epochs
        trial_args.max_train_samples = args.optuna_max_train_samples
        trial_args.max_val_samples = args.optuna_max_val_samples
        trial_args.no_tensorboard = True
        trial_args.num_visual_examples = 0
        trial_args.visualize_every = args.optuna_epochs + 1
        trial_args.early_stopping_patience = min(args.early_stopping_patience, 3)
        trial_args.encoder_lr = trial.suggest_float("encoder_lr", 1e-5, 3e-4, log=True)
        trial_args.decoder_lr = trial.suggest_float("decoder_lr", 3e-5, 1e-3, log=True)
        trial_args.loss = trial.suggest_categorical("loss", ["cross_entropy_dice", "focal", "tversky", "focal_tversky"])
        trial_args.sampler = trial.suggest_categorical("sampler", ["shuffle", "weighted"])
        trial_args.scheduler = trial.suggest_categorical("scheduler", ["reduce_on_plateau", "cosine", "warmup_cosine"])
        trial_args.batch_size = trial.suggest_categorical("batch_size", args.optuna_batch_sizes)

        minor_weight = trial.suggest_float("minor_class_weight", 1.0, 2.5)
        major_weight = trial.suggest_float("major_class_weight", 1.2, 3.0)
        destroyed_weight = trial.suggest_float("destroyed_class_weight", 1.2, 3.2)
        trial_args.class_weights = [1.0, 0.2, minor_weight, major_weight, destroyed_weight]
        trial_args.experiment_name = f"experiment_{args.experiment}_optuna_trial_{trial.number:03d}"

        experiment_dir = train_experiment(trial_args)
        metrics = load_final_metrics(experiment_dir)
        mean_dice = float(metrics.get("val_mean_dice", 0.0))
        rare_recall = float(metrics.get("val_rare_class_recall", 0.0))
        objective_value = mean_dice + args.optuna_rare_recall_weight * rare_recall

        trial.set_user_attr("experiment_dir", str(experiment_dir))
        trial.set_user_attr("val_mean_dice", mean_dice)
        trial.set_user_attr("val_rare_class_recall", rare_recall)
        trial.set_user_attr("val_macro_f1", float(metrics.get("val_macro_f1", 0.0)))
        trial.set_user_attr("objective", objective_value)
        save_optuna_trials_csv(study, hpo_dir / "optuna_trials.csv")
        return objective_value

    study.optimize(objective, n_trials=args.optuna_trials, gc_after_trial=True)
    save_optuna_trials_csv(study, hpo_dir / "optuna_trials.csv")
    best_summary = {
        "study_name": args.optuna_study_name,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "hpo_epochs": args.optuna_epochs,
        "hpo_max_train_samples": args.optuna_max_train_samples,
        "hpo_max_val_samples": args.optuna_max_val_samples,
        "objective": f"val_mean_dice + {args.optuna_rare_recall_weight} * val_rare_class_recall",
    }
    (hpo_dir / "best_trial.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    summarize_experiments(args.results_root, args.results_root / "comparative_analysis" / "architecture_comparison.csv")
    print(f"Optuna study complete. Best trial: {study.best_trial.number} value={study.best_value:.4f}")
    print(f"Best params saved to {hpo_dir / 'best_trial.json'}")


def make_kfold_split_dirs(args: argparse.Namespace) -> list[Path]:
    """Create fold-specific split folders from the current train+val IDs."""
    all_ids = read_split_file(args.split_dir / "train.txt") + read_split_file(args.split_dir / "val.txt")
    disaster_keywords = MOROCCO_ADAPTATION_KEYWORDS if args.morocco_adaptation else args.disaster_keywords
    all_ids = filter_sample_ids_by_disaster(all_ids, disaster_keywords)
    if not all_ids:
        raise ValueError(f"No samples remain after disaster filtering: {disaster_keywords}")
    test_ids = read_split_file(args.split_dir / "test.txt") if (args.split_dir / "test.txt").exists() else []
    rng = random.Random(args.seed)
    rng.shuffle(all_ids)
    fold_count = max(2, args.k_folds)
    fold_dirs = []
    for fold_index in range(fold_count):
        val_ids = all_ids[fold_index::fold_count]
        train_ids = [sample_id for sample_id in all_ids if sample_id not in set(val_ids)]
        fold_dir = args.results_root / "cross_validation" / f"fold_{fold_index + 1}" / "splits"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
        (fold_dir / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
        (fold_dir / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
        fold_dirs.append(fold_dir)
    return fold_dirs


def train_experiment(args: argparse.Namespace) -> Path:
    preset = EXPERIMENTS.get(args.experiment, {})
    model_name = args.model or preset.get("model", "attention_unet")
    loss_name = args.loss or preset.get("loss", "focal")
    sampler_name = args.sampler or preset.get("sampler", "weighted")
    experiment_name = args.experiment_name or f"experiment_{args.experiment}"

    fix_seed(args.seed)
    start_time = time.perf_counter()
    experiment_dir = ensure_experiment_dirs(args.results_root, experiment_name)
    write_analysis_template(experiment_dir)
    logger = setup_file_logger(experiment_dir / "logs" / "training_console.log")
    device = get_device()
    config = ExperimentConfig(
        experiment_name=experiment_name,
        model_name=model_name,
        loss_name=loss_name,
        sampler_name=sampler_name,
        scheduler_name=args.scheduler,
        data_dir=str(args.data_dir),
        split_dir=str(args.split_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        freeze_encoder=args.freeze_encoder,
        amp=args.amp,
        tensorboard=not args.no_tensorboard,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        advanced_augmentations=not args.no_advanced_augmentations,
        visualize_every=args.visualize_every,
        disaster_keywords=MOROCCO_ADAPTATION_KEYWORDS if args.morocco_adaptation else args.disaster_keywords,
    )
    save_experiment_config(config, experiment_dir / "config")

    train_loader, val_loader = build_dataloaders(args, sampler_name)
    model = build_model(model_name, pretrained=not args.no_pretrained, freeze_encoder=args.freeze_encoder).to(device)
    optimizer = build_optimizer(model, args.encoder_lr, args.decoder_lr)
    criterion = build_loss(loss_name, args.class_weights).to(device)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(str(experiment_dir / "logs" / "tensorboard")) if SummaryWriter is not None and not args.no_tensorboard else None
    save_json(collect_environment_metadata(model), experiment_dir / "config" / "environment_metadata.json")

    disaster_keywords = MOROCCO_ADAPTATION_KEYWORDS if args.morocco_adaptation else args.disaster_keywords
    logger.info("device=%s experiment=%s model=%s loss=%s sampler=%s", device, experiment_name, model_name, loss_name, sampler_name)
    logger.info("disaster_keywords=%s", disaster_keywords)
    logger.info("amp=%s tensorboard=%s early_stopping_patience=%s", use_amp, writer is not None, args.early_stopping_patience)
    logger.info("train_samples=%s val_samples=%s", len(train_loader.dataset), len(val_loader.dataset))

    best_val_dice = -1.0
    patience_counter = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_metrics, val_confusion, val_examples, pr_rows = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            collect_examples=epoch % max(1, args.visualize_every) == 0 or epoch == args.epochs,
        )
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
            "train_mean_iou": train_metrics["mean_iou"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_frequency_weighted_iou": val_metrics["frequency_weighted_iou"],
            "val_rare_class_recall": val_metrics["rare_class_recall"],
            "val_mean_boundary_iou": val_metrics.get("mean_boundary_iou", 0.0),
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_dice_no_damage": val_metrics["dice_no_damage"],
            "val_dice_minor_damage": val_metrics["dice_minor_damage"],
            "val_dice_major_damage": val_metrics["dice_major_damage"],
            "val_dice_destroyed": val_metrics["dice_destroyed"],
        }
        history.append(record)
        save_history_csv(history, experiment_dir / "metrics" / "training_log.csv")
        if writer is not None:
            for key, value in record.items():
                if key != "epoch":
                    writer.add_scalar(key, float(value), epoch)
            for index, group in enumerate(optimizer.param_groups):
                writer.add_scalar(f"lr/group_{index}", float(group["lr"]), epoch)
        logger.info(
            "epoch=%s train_loss=%.4f val_loss=%.4f val_dice=%.4f boundary_iou=%.4f rare_recall=%.4f minor=%.4f major=%.4f destroyed=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["mean_dice"],
            val_metrics.get("mean_boundary_iou", 0.0),
            val_metrics["rare_class_recall"],
            val_metrics["dice_minor_damage"],
            val_metrics["dice_major_damage"],
            val_metrics["dice_destroyed"],
        )

        improved = val_metrics["mean_dice"] > best_val_dice + args.early_stopping_min_delta
        if improved:
            best_val_dice = val_metrics["mean_dice"]
            patience_counter = 0
            final_metrics = {
                "best_epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            save_json(final_metrics, experiment_dir / "metrics" / "final_metrics.json")
            save_per_class_metrics(val_metrics, experiment_dir / "metrics" / "per_class_metrics.csv")
            save_precision_recall_csv(aggregate_precision_recall_rows(pr_rows), experiment_dir / "metrics" / "precision_recall_curves.csv")
            save_confusion_matrix_csv(val_confusion, experiment_dir / "confusion_matrices" / "confusion_matrix.csv")
            save_validation_examples(val_examples, experiment_dir, epoch, max_each=args.num_visual_examples)
            save_checkpoint(
                model,
                experiment_dir / "checkpoints" / "best_model.pt",
                epoch,
                val_metrics,
                {"class_names": CLASS_NAMES, "model": model_name, "experiment": experiment_name},
            )
        else:
            patience_counter += 1
        save_checkpoint(model, experiment_dir / "checkpoints" / "last_model.pt", epoch, val_metrics)
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            logger.info("early_stopping=true epoch=%s best_val_dice=%.4f", epoch, best_val_dice)
            break

    training_time_seconds = time.perf_counter() - start_time
    metadata = collect_environment_metadata(model)
    metadata["training_time_seconds"] = training_time_seconds
    metadata["best_val_dice"] = best_val_dice
    save_json(metadata, experiment_dir / "config" / "environment_metadata.json")
    if writer is not None:
        writer.close()
    summarize_experiments(args.results_root, args.results_root / "comparative_analysis" / "architecture_comparison.csv")
    return experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or scaffold Week 6 damage-segmentation experiments.")
    parser.add_argument("--scaffold-only", action="store_true", help="Only create results/week6 folders.")
    parser.add_argument("--summarize-only", action="store_true", help="Only refresh comparative analysis CSV.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="attention_unet")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--loss", default=None)
    parser.add_argument("--sampler", choices=["shuffle", "weighted"], default=None)
    parser.add_argument("--scheduler", default="reduce_on_plateau")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--results-root", type=Path, default=Path("results") / "week6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Run the same experiment for multiple seeds.")
    parser.add_argument("--k-folds", type=int, default=0, help="Run k-fold validation using temporary Week 6 split folders.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--class-weights", type=float, nargs=5, default=DEFAULT_CLASS_WEIGHTS)
    parser.add_argument("--amp", action="store_true", default=True, help="Use CUDA mixed precision when available.")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable mixed precision.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--visualize-every", type=int, default=1)
    parser.add_argument("--num-visual-examples", type=int, default=6)
    parser.add_argument("--no-advanced-augmentations", action="store_true")
    parser.add_argument("--disaster-keywords", nargs="*", default=None, help="Only use sample IDs containing these keywords.")
    parser.add_argument(
        "--morocco-adaptation",
        action="store_true",
        help="Shortcut for earthquake/flooding/wildfire/fire samples relevant to Morocco adaptation.",
    )
    parser.add_argument("--optuna-study", action="store_true", help="Run a cheap Optuna HPO study instead of one normal experiment.")
    parser.add_argument("--optuna-study-name", default="morocco_week6_hpo")
    parser.add_argument("--optuna-storage", default=None, help="Optuna storage URL. Defaults to a local SQLite DB in results/week6.")
    parser.add_argument("--optuna-trials", type=int, default=12)
    parser.add_argument("--optuna-epochs", type=int, default=6)
    parser.add_argument("--optuna-max-train-samples", type=int, default=2000)
    parser.add_argument("--optuna-max-val-samples", type=int, default=500)
    parser.add_argument("--optuna-batch-sizes", type=int, nargs="*", default=[2, 4])
    parser.add_argument("--optuna-rare-recall-weight", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_week6_results_tree(args.results_root)
    if args.scaffold_only:
        print(f"Created Week 6 results scaffold at {args.results_root}")
        return
    if args.summarize_only:
        output = args.results_root / "comparative_analysis" / "architecture_comparison.csv"
        summarize_experiments(args.results_root, output)
        print(f"Updated {output}")
        return
    if args.optuna_study:
        run_optuna_hpo(args)
        return
    if args.k_folds and args.k_folds > 1:
        original_split_dir = args.split_dir
        original_experiment_name = args.experiment_name
        seeds = args.seeds or [args.seed]
        fold_dirs = make_kfold_split_dirs(args)
        for fold_number, fold_split_dir in enumerate(fold_dirs, start=1):
            args.split_dir = fold_split_dir
            for seed in seeds:
                args.seed = seed
                args.experiment_name = f"{original_experiment_name or 'experiment_' + args.experiment}_fold_{fold_number}_seed_{seed}"
                experiment_dir = train_experiment(args)
                print(f"Finished {experiment_dir}")
        args.split_dir = original_split_dir
        summarize_experiments(args.results_root, args.results_root / "comparative_analysis" / "cross_validation_summary.csv")
        return
    if args.seeds:
        base_experiment_name = args.experiment_name
        for seed in args.seeds:
            args.seed = seed
            args.experiment_name = f"{base_experiment_name or 'experiment_' + args.experiment}_seed_{seed}"
            experiment_dir = train_experiment(args)
            print(f"Finished {experiment_dir}")
        save_multi_seed_summary(args.results_root, base_experiment_name or "experiment_" + args.experiment, args.seeds)
        return
    experiment_dir = train_experiment(args)
    print(f"Finished {experiment_dir}")


if __name__ == "__main__":
    main()

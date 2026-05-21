"""Reusable Week 6 experiment utilities."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


EXPERIMENT_SUBDIRS = [
    "checkpoints",
    "config",
    "confusion_matrices",
    "metrics",
    "predictions/best_examples",
    "predictions/failure_cases",
    "predictions/difficult_scenes",
    "visualizations/confidence_maps",
    "visualizations/error_heatmaps",
    "visualizations/overlays",
    "logs",
    "logs/tensorboard",
    "analysis",
]


@dataclass
class ExperimentConfig:
    """Serializable metadata shared by Week 6 experiments."""

    experiment_name: str
    model_name: str
    loss_name: str = "cross_entropy_dice"
    sampler_name: str = "shuffle"
    scheduler_name: str = "reduce_on_plateau"
    data_dir: str = "data"
    split_dir: str = "splits"
    image_size: int = 512
    batch_size: int = 4
    epochs: int = 20
    encoder_lr: float = 1e-4
    decoder_lr: float = 3e-4
    num_workers: int = 0
    seed: int = 42
    pretrained: bool = True
    freeze_encoder: bool = False
    amp: bool = True
    tensorboard: bool = True
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    advanced_augmentations: bool = True
    visualize_every: int = 1
    disaster_keywords: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fix_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Make training runs easier to reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_experiment_dirs(results_root: Path, experiment_name: str) -> Path:
    """Create a research-grade isolated experiment folder."""
    experiment_dir = results_root / experiment_name
    for relative in EXPERIMENT_SUBDIRS:
        directory = experiment_dir / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)
    return experiment_dir


def save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_yaml_like(data: dict[str, Any], output_path: Path) -> None:
    """Write a lightweight YAML file without adding a dependency."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{key}: {value}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_experiment_config(config: ExperimentConfig | dict[str, Any], config_dir: Path) -> None:
    data = config.to_dict() if isinstance(config, ExperimentConfig) else dict(config)
    save_json(data, config_dir / "training_config.json")
    save_yaml_like(data, config_dir / "training_config.yaml")


def setup_file_logger(log_path: Path) -> logging.Logger:
    """Create a console/file logger for an experiment."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_checkpoint(
    model: nn.Module,
    output_path: Path,
    epoch: int,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device | None = None) -> dict[str, Any]:
    target_device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=target_device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def collect_environment_metadata(model: nn.Module | None = None) -> dict[str, Any]:
    """Capture reproducibility metadata for the experiment folder."""
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"
    metadata: dict[str, Any] = {
        "git_commit": git_commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "created_unix_time": time.time(),
    }
    if torch.cuda.is_available():
        metadata["gpu_count"] = torch.cuda.device_count()
        metadata["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if model is not None:
        metadata["parameter_count"] = count_parameters(model)
        metadata["trainable_parameter_count"] = count_trainable_parameters(model)
    return metadata


def create_week6_results_tree(results_root: Path = Path("results") / "week6") -> None:
    """Create the recommended Week 6 results tree and placeholder files."""
    experiments = [
        "experiment_baseline",
        "experiment_focal_loss",
        "experiment_tversky_loss",
        "experiment_focal_tversky",
        "experiment_weighted_sampler",
        "experiment_attention_unet",
        "experiment_unetplusplus",
        "experiment_deeplabv3",
        "experiment_resnet50",
    ]
    for experiment_name in experiments:
        ensure_experiment_dirs(results_root, experiment_name)

    for folder in [
        "ablation_studies/ablation_plots",
        "comparative_analysis",
        "final_model/inference_examples",
        "paper_figures",
    ]:
        directory = results_root / folder
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)

"""Visualization helpers for qualitative Week 6 damage analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_COLORS_RGB = np.array(
    [
        [0, 0, 0],
        [0, 180, 0],
        [255, 230, 0],
        [255, 140, 0],
        [220, 0, 0],
    ],
    dtype=np.uint8,
)

SIX_CHANNEL_MEAN = np.array((0.485, 0.456, 0.406, 0.485, 0.456, 0.406), dtype=np.float32)
SIX_CHANNEL_STD = np.array((0.229, 0.224, 0.225, 0.229, 0.224, 0.225), dtype=np.float32)


def tensor_to_rgb_pair(image_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert a normalized 6-channel tensor into pre/post uint8 RGB images."""
    array = image_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    array = (array * SIX_CHANNEL_STD + SIX_CHANNEL_MEAN)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array[:, :, :3], array[:, :, 3:6]


def colorize_mask(mask: np.ndarray, colors: np.ndarray = CLASS_COLORS_RGB) -> np.ndarray:
    clipped = np.clip(mask.astype(np.int64), 0, len(colors) - 1)
    return colors[clipped]


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = np.clip(image_rgb, 0, 255).astype(np.uint8)
    colored = colorize_mask(mask)
    foreground = mask > 0
    overlay = image.copy()
    overlay[foreground] = ((1.0 - alpha) * image[foreground] + alpha * colored[foreground]).astype(np.uint8)
    return overlay


def save_overlay(image_rgb: np.ndarray, mask: np.ndarray, output_path: Path, alpha: float = 0.45) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, make_overlay(image_rgb, mask, alpha=alpha))


def save_prediction_panel(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [
        ("pre", pre_image),
        ("post", post_image),
        ("truth", colorize_mask(target)),
        ("prediction", colorize_mask(prediction)),
        ("error", (target != prediction).astype(np.uint8) * 255),
    ]
    fig, axes = plt.subplots(1, len(images), figsize=(16, 4))
    for axis, (title, image) in zip(axes, images):
        axis.imshow(image, cmap="gray" if title == "error" else None)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def save_confidence_map(logits: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    confidence = torch.softmax(logits.detach().cpu(), dim=1).max(dim=1).values.squeeze().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_error_heatmap(target: np.ndarray, prediction: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error = (target != prediction).astype(np.float32)
    plt.figure(figsize=(5, 5))
    plt.imshow(error, cmap="magma", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

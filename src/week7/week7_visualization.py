"""Week 7 temporal and attention visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from week6.week6_visualization import save_confidence_map, save_error_heatmap, save_overlay, save_prediction_panel


def save_temporal_difference_map(pre_image: np.ndarray, post_image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diff = np.abs(post_image.astype(np.float32) - pre_image.astype(np.float32)).mean(axis=2)
    plt.figure(figsize=(5, 5))
    plt.imshow(diff, cmap="magma")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_attention_heatmap(attention: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = attention.detach().cpu().float()
    while heatmap.ndim > 2:
        heatmap = heatmap.mean(dim=0)
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap.numpy(), cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

__all__ = [
    "save_attention_heatmap",
    "save_confidence_map",
    "save_error_heatmap",
    "save_overlay",
    "save_prediction_panel",
    "save_temporal_difference_map",
]


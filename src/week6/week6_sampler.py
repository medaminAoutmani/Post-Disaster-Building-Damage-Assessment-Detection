"""Sampling strategies for Week 6 rare-class learning experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


def _sample_mask(sample: object) -> torch.Tensor:
    if isinstance(sample, dict) and "mask" in sample:
        mask = sample["mask"]
    elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
        mask = sample[1]
    else:
        raise ValueError("Dataset samples must expose a mask in dict['mask'] or tuple[1].")
    return torch.as_tensor(mask)


def compute_class_aware_sample_weights(
    dataset: Dataset,
    class_weights: Sequence[float],
    max_samples: int | None = None,
) -> torch.Tensor:
    """Give higher probability to samples containing rare/high-value classes."""
    weights = []
    limit = min(len(dataset), max_samples or len(dataset))
    class_weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
    for index in range(limit):
        mask = _sample_mask(dataset[index]).long()
        present = torch.unique(mask)
        valid = present[(present >= 0) & (present < len(class_weight_tensor))]
        weights.append(float(class_weight_tensor[valid].max().item()) if len(valid) else 1.0)
    return torch.as_tensor(weights, dtype=torch.double)


def build_weighted_sampler(
    dataset: Dataset,
    class_weights: Sequence[float],
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    weights = compute_class_aware_sample_weights(dataset, class_weights)
    return WeightedRandomSampler(weights=weights, num_samples=num_samples or len(weights), replacement=True)


class HardExampleSampler(Sampler[int]):
    """Sampler that prioritizes sample indices with high recorded validation loss."""

    def __init__(self, losses: dict[int, float], dataset_size: int, top_fraction: float = 0.3) -> None:
        self.dataset_size = dataset_size
        self.top_fraction = top_fraction
        self.losses = losses

    def __iter__(self):
        ranked = sorted(self.losses.items(), key=lambda item: item[1], reverse=True)
        hard_count = max(1, int(self.dataset_size * self.top_fraction))
        hard_indices = [index for index, _ in ranked[:hard_count]]
        remaining = [index for index in range(self.dataset_size) if index not in set(hard_indices)]
        order = hard_indices + remaining
        return iter(order)

    def __len__(self) -> int:
        return self.dataset_size


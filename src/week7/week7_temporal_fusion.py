"""Temporal feature fusion modules for Siamese damage segmentation."""

from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Learn a per-pixel gate between pre- and post-disaster features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, pre_features: torch.Tensor, post_features: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([pre_features, post_features], dim=1))
        return gate * post_features + (1.0 - gate) * pre_features


class TemporalFusion(nn.Module):
    """Fuse paired temporal features with a named strategy."""

    def __init__(self, channels: int, strategy: str = "concat") -> None:
        super().__init__()
        self.strategy = strategy
        self.gated = GatedFusion(channels) if strategy == "gated_fusion" else None
        if strategy == "concat":
            in_channels = channels * 2
        elif strategy == "difference":
            in_channels = channels
        elif strategy == "concat_difference":
            in_channels = channels * 3
        elif strategy == "gated_fusion":
            in_channels = channels
        else:
            raise ValueError(f"Unknown temporal fusion strategy: {strategy}")
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre_features: torch.Tensor, post_features: torch.Tensor) -> torch.Tensor:
        if pre_features.shape[2:] != post_features.shape[2:]:
            post_features = nn.functional.interpolate(
                post_features,
                size=pre_features.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        difference = torch.abs(post_features - pre_features)
        if self.strategy == "concat":
            fused = torch.cat([pre_features, post_features], dim=1)
        elif self.strategy == "difference":
            fused = difference
        elif self.strategy == "concat_difference":
            fused = torch.cat([pre_features, post_features, difference], dim=1)
        elif self.strategy == "gated_fusion" and self.gated is not None:
            fused = self.gated(pre_features, post_features)
        else:
            raise ValueError(f"Unknown temporal fusion strategy: {self.strategy}")
        return self.project(fused)


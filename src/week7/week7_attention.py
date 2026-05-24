"""Attention blocks for Week 7 temporal Siamese models."""

from __future__ import annotations

import torch
from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.layers(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.shared(nn.functional.adaptive_avg_pool2d(x, 1))
        maximum = self.shared(nn.functional.adaptive_max_pool2d(x, 1))
        return x * self.sigmoid(avg + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        maximum = x.max(dim=1, keepdim=True).values
        attention = self.sigmoid(self.conv(torch.cat([avg, maximum], dim=1)))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class NonLocalBlock(nn.Module):
    """Compact non-local self-attention block for long-range context."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        inter_channels = max(channels // 2, 8)
        self.theta = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.g = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.out = nn.Sequential(nn.Conv2d(inter_channels, channels, kernel_size=1), nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        theta = self.theta(x).reshape(batch, -1, height * width).transpose(1, 2)
        phi = self.phi(x).reshape(batch, -1, height * width)
        attention = torch.softmax(torch.bmm(theta, phi), dim=-1)
        g = self.g(x).reshape(batch, -1, height * width).transpose(1, 2)
        y = torch.bmm(attention, g).transpose(1, 2).reshape(batch, -1, height, width)
        return x + self.out(y)


class IdentityAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_attention(name: str, channels: int) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized in {"none", "no_attention", "identity"}:
        return IdentityAttention()
    if normalized in {"se", "bottleneck_attention"}:
        return SEBlock(channels)
    if normalized == "cbam":
        return CBAM(channels)
    if normalized in {"non_local", "nonlocal", "non_local_block"}:
        return NonLocalBlock(channels)
    raise ValueError(f"Unknown attention type: {name}")


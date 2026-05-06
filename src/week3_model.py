"""Improved U-Net model definition for Week 3 binary segmentation.

Week 2 keeps the first baseline model in ``week2_model.py``. This module keeps
the Week 3 training path separate so model experiments can evolve without
overwriting the original baseline.
"""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two convolution blocks used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    """Week 3 U-Net. Default input is pre RGB + post RGB = 6 channels."""

    def __init__(self, in_channels: int = 6, out_channels: int = 1, features: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        channels = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(channels, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
            channels = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip = skip_connections[index // 2]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat((skip, x), dim=1)
            x = self.ups[index + 1](x)

        return self.final_conv(x)

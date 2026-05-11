"""Attention U-Net for Week 6 multiclass damage segmentation."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AttentionGate(nn.Module):
    """Attention gate for filtering skip features."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_projection = nn.Sequential(nn.Conv2d(gate_channels, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.skip_projection = nn.Sequential(nn.Conv2d(skip_channels, inter_channels, 1), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[2:] != skip.shape[2:]:
            gate = nn.functional.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.relu(self.gate_projection(gate) + self.skip_projection(skip)))
        return skip * attention


class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 16))
        self.refine = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        attended_skip = self.attention(x, skip)
        return self.refine(torch.cat([x, attended_skip], dim=1))


class AttentionUNet(nn.Module):
    """Scratch Attention U-Net accepting 6-channel pre/post inputs."""

    def __init__(self, in_channels: int = 6, out_channels: int = 5, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.dec4 = AttentionDecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = AttentionDecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = AttentionDecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = AttentionDecoderBlock(c * 2, c, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.dec4(x, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        if x.shape[2:] != input_size:
            x = nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.head(x)


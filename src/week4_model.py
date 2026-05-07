"""Week 4 U-Net with a pretrained ResNet34 encoder.

This model keeps the Week 3 binary segmentation task but replaces the scratch
encoder with a ResNet34 feature extractor. The first ResNet convolution is
adapted from 3 RGB channels to the project's 6-channel pre/post input.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision.models import resnet34

try:
    from torchvision.models import ResNet34_Weights
except ImportError:  # Older torchvision compatibility.
    ResNet34_Weights = None  # type: ignore[assignment]


class DecoderBlock(nn.Module):
    """Upsample, concatenate a skip feature, then refine with two convolutions."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))


def _load_resnet34(pretrained: bool) -> nn.Module:
    """Load ResNet34 while supporting both old and new torchvision APIs."""
    if not pretrained:
        return resnet34(weights=None) if ResNet34_Weights is not None else resnet34(pretrained=False)

    try:
        if ResNet34_Weights is not None:
            return resnet34(weights=ResNet34_Weights.DEFAULT)
        return resnet34(pretrained=True)
    except Exception as exc:
        warnings.warn(
            f"Could not load pretrained ResNet34 weights ({exc}). Falling back to random initialization.",
            stacklevel=2,
        )
        return resnet34(weights=None) if ResNet34_Weights is not None else resnet34(pretrained=False)


def _adapt_first_conv_to_six_channels(conv: nn.Conv2d) -> nn.Conv2d:
    """Convert ResNet's 3-channel stem convolution into a 6-channel one."""
    adapted = nn.Conv2d(
        in_channels=6,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )

    with torch.no_grad():
        if conv.weight.shape[1] == 3:
            adapted.weight[:, :3] = conv.weight / 2.0
            adapted.weight[:, 3:] = conv.weight / 2.0
        else:
            nn.init.kaiming_normal_(adapted.weight, mode="fan_out", nonlinearity="relu")
        if conv.bias is not None and adapted.bias is not None:
            adapted.bias.copy_(conv.bias)

    return adapted


class ResNet34UNet(nn.Module):
    """Binary segmentation U-Net using ResNet34 encoder features."""

    def __init__(self, out_channels: int = 1, pretrained: bool = True, freeze_encoder: bool = False) -> None:
        super().__init__()
        encoder = _load_resnet34(pretrained)
        encoder.conv1 = _adapt_first_conv_to_six_channels(encoder.conv1)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        if freeze_encoder:
            for parameter in (
                list(self.stem.parameters())
                + list(self.encoder1.parameters())
                + list(self.encoder2.parameters())
                + list(self.encoder3.parameters())
                + list(self.encoder4.parameters())
            ):
                parameter.requires_grad = False

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        stem = self.stem(x)
        pooled = self.pool(stem)
        enc1 = self.encoder1(pooled)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        x = self.decoder4(enc4, enc3)
        x = self.decoder3(x, enc2)
        x = self.decoder2(x, enc1)
        x = self.decoder1(x, stem)
        x = self.final_upsample(x)

        if x.shape[2:] != input_size:
            x = nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.final_conv(x)

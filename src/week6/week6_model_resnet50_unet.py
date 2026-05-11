"""ResNet50 U-Net upgrade for Week 6 experiments."""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision.models import resnet50

try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None  # type: ignore[assignment]

from week4_model import DecoderBlock, _adapt_first_conv_to_six_channels


def _load_resnet50(pretrained: bool) -> nn.Module:
    if not pretrained:
        return resnet50(weights=None) if ResNet50_Weights is not None else resnet50(pretrained=False)
    try:
        if ResNet50_Weights is not None:
            return resnet50(weights=ResNet50_Weights.DEFAULT)
        return resnet50(pretrained=True)
    except Exception as exc:
        warnings.warn(f"Could not load pretrained ResNet50 weights ({exc}). Using random init.", stacklevel=2)
        return resnet50(weights=None) if ResNet50_Weights is not None else resnet50(pretrained=False)


class ResNet50UNet(nn.Module):
    """U-Net decoder on top of a 6-channel ResNet50 encoder."""

    def __init__(self, out_channels: int = 5, pretrained: bool = True, freeze_encoder: bool = False) -> None:
        super().__init__()
        encoder = _load_resnet50(pretrained)
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

        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
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


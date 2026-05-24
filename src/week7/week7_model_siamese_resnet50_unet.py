"""Siamese ResNet50 U-Net for temporal damage segmentation."""

from __future__ import annotations

import copy
import warnings

import torch
from torch import nn
from torchvision.models import resnet50

try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None  # type: ignore[assignment]

from week4_model import DecoderBlock
from week7_attention import build_attention
from week7_temporal_fusion import TemporalFusion


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


class ResNet50Encoder(nn.Module):
    """Return ResNet50 feature pyramid for one RGB temporal image."""

    def __init__(self, pretrained: bool = True, freeze: bool = False) -> None:
        super().__init__()
        encoder = _load_resnet50(pretrained)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(x)
        pooled = self.pool(stem)
        enc1 = self.encoder1(pooled)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        return {"stem": stem, "enc1": enc1, "enc2": enc2, "enc3": enc3, "enc4": enc4}


class SiameseResNet50UNet(nn.Module):
    """Two temporal ResNet50 encoders with feature-level fusion and U-Net decoder."""

    FEATURE_CHANNELS = {"stem": 64, "enc1": 256, "enc2": 512, "enc3": 1024, "enc4": 2048}

    def __init__(
        self,
        out_channels: int = 5,
        fusion_strategy: str = "concat",
        attention_type: str = "no_attention",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        share_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_pre = ResNet50Encoder(pretrained=pretrained, freeze=freeze_encoder)
        self.encoder_post = self.encoder_pre if share_encoder else copy.deepcopy(self.encoder_pre)
        self.fusions = nn.ModuleDict(
            {
                name: TemporalFusion(channels, strategy=fusion_strategy)
                for name, channels in self.FEATURE_CHANNELS.items()
            }
        )
        self.bottleneck_attention = build_attention(attention_type, self.FEATURE_CHANNELS["enc4"])
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

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor) -> torch.Tensor:
        input_size = pre_image.shape[2:]
        pre = self.encoder_pre(pre_image)
        post = self.encoder_post(post_image)
        fused = {name: self.fusions[name](pre[name], post[name]) for name in self.FEATURE_CHANNELS}
        x = self.bottleneck_attention(fused["enc4"])
        x = self.decoder4(x, fused["enc3"])
        x = self.decoder3(x, fused["enc2"])
        x = self.decoder2(x, fused["enc1"])
        x = self.decoder1(x, fused["stem"])
        x = self.final_upsample(x)
        if x.shape[2:] != input_size:
            x = nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.final_conv(x)


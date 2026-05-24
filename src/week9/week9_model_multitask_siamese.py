"""Week 9 multi-task Siamese ResNet50 U-Net."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
WEEK7_DIR = SRC_DIR / "week7"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(WEEK7_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK7_DIR))

from week4_model import DecoderBlock
from week7_attention import build_attention
from week7_model_siamese_resnet50_unet import ResNet50Encoder
from week7_temporal_fusion import TemporalFusion


class MultiTaskSiameseResNet50UNet(nn.Module):
    """Shared-encoder Siamese U-Net with building and damage heads."""

    FEATURE_CHANNELS = {"stem": 64, "enc1": 256, "enc2": 512, "enc3": 1024, "enc4": 2048}

    def __init__(
        self,
        fusion_strategy: str = "difference",
        attention_type: str = "cbam",
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained, freeze=freeze_encoder)
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
        self.pre_building_head = nn.Conv2d(32, 2, kernel_size=1)
        self.post_building_head = nn.Conv2d(32, 2, kernel_size=1)
        self.damage_head = nn.Conv2d(32, 5, kernel_size=1)

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = pre_image.shape[2:]
        pre = self.encoder(pre_image)
        post = self.encoder(post_image)
        fused = {name: self.fusions[name](pre[name], post[name]) for name in self.FEATURE_CHANNELS}
        x = self.bottleneck_attention(fused["enc4"])
        x = self.decoder4(x, fused["enc3"])
        x = self.decoder3(x, fused["enc2"])
        x = self.decoder2(x, fused["enc1"])
        x = self.decoder1(x, fused["stem"])
        x = self.final_upsample(x)
        if x.shape[2:] != input_size:
            x = nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return {
            "pre_building_logits": self.pre_building_head(x),
            "post_building_logits": self.post_building_head(x),
            "damage_logits": self.damage_head(x),
        }

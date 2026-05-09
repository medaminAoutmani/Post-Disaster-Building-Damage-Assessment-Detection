"""Week 5 multiclass damage segmentation model."""

from __future__ import annotations

from week4_model import ResNet34UNet


class DamageResNet34UNet(ResNet34UNet):
    """ResNet34-encoder U-Net for 5-class xBD damage segmentation."""

    def __init__(self, pretrained: bool = True, freeze_encoder: bool = False) -> None:
        super().__init__(out_channels=5, pretrained=pretrained, freeze_encoder=freeze_encoder)

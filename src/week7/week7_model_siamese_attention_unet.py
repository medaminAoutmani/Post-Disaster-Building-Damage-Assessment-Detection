"""Named Siamese attention U-Net variants for Week 7."""

from __future__ import annotations

from week7_model_siamese_resnet50_unet import SiameseResNet50UNet


class SiameseAttentionUNet(SiameseResNet50UNet):
    def __init__(self, out_channels: int = 5, fusion_strategy: str = "concat", attention_type: str = "cbam", pretrained: bool = True) -> None:
        super().__init__(
            out_channels=out_channels,
            fusion_strategy=fusion_strategy,
            attention_type=attention_type,
            pretrained=pretrained,
        )


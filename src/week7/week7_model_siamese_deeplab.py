"""Siamese DeepLab-style placeholder backed by the Week 7 Siamese decoder."""

from __future__ import annotations

from week7_model_siamese_resnet50_unet import SiameseResNet50UNet


class SiameseDeepLabDamage(SiameseResNet50UNet):
    """Practical Week 7 DeepLab-style experiment alias.

    The project can later replace this with an ASPP decoder. For now it keeps
    the same Siamese temporal encoder/fusion contract as the other Week 7
    models so experiments remain comparable.
    """

    def __init__(self, out_channels: int = 5, fusion_strategy: str = "concat", pretrained: bool = True) -> None:
        super().__init__(
            out_channels=out_channels,
            fusion_strategy=fusion_strategy,
            attention_type="bottleneck_attention",
            pretrained=pretrained,
        )


"""DeepLabV3 model for Week 6 large-context experiments.

Torchvision provides DeepLabV3, not DeepLabV3+. This module keeps the original
filename for project continuity, but the exported model name is intentionally
DeepLabV3Damage for research-report correctness.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50

try:
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
except ImportError:
    DeepLabV3_ResNet50_Weights = None  # type: ignore[assignment]

from week4_model import _adapt_first_conv_to_six_channels


def _random_deeplab() -> nn.Module:
    try:
        return deeplabv3_resnet50(weights=None, weights_backbone=None)
    except TypeError:
        return deeplabv3_resnet50(pretrained=False)


def _load_deeplab(pretrained: bool) -> nn.Module:
    if not pretrained:
        return _random_deeplab()
    try:
        if DeepLabV3_ResNet50_Weights is not None:
            return deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        return deeplabv3_resnet50(pretrained=True)
    except Exception as exc:
        warnings.warn(f"Could not load pretrained DeepLabV3 weights ({exc}). Using random init.", stacklevel=2)
        return _random_deeplab()


class DeepLabV3Damage(nn.Module):
    """DeepLabV3 wrapper adapted to 6-channel input and 5 damage classes."""

    def __init__(self, out_channels: int = 5, pretrained: bool = True, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.model = _load_deeplab(pretrained)
        self.model.backbone.conv1 = _adapt_first_conv_to_six_channels(self.model.backbone.conv1)
        classifier_in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(classifier_in_channels, out_channels, kernel_size=1)
        if freeze_backbone:
            for parameter in self.model.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]

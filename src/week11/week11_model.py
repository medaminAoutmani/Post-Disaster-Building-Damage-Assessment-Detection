"""Simple Siamese CNN baseline for Week 11 object-level damage classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_resnet18_encoder(pretrained: bool = False) -> tuple[nn.Module, int]:
    """Create a ResNet18 image encoder without its classifier head."""
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dim


class SiameseBuildingClassifier(nn.Module):
    """Three-branch ResNet18 classifier using pre, post, and diff crop features."""

    def __init__(self, num_classes: int = 4, pretrained: bool = False, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder, feature_dim = build_resnet18_encoder(pretrained=pretrained)
        fused_dim = feature_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        diff_features = self.encoder(diff)
        fused = torch.cat([pre_features, post_features, diff_features, torch.abs(pre_features - post_features)], dim=1)
        return self.classifier(fused)


class SiameseBuildingFeatureClassifier(nn.Module):
    """Siamese ResNet18 classifier fused with handcrafted morphology/TDA features."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 4,
        pretrained: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder, cnn_feature_dim = build_resnet18_encoder(pretrained=pretrained)
        fused_cnn_dim = cnn_feature_dim * 4
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_cnn_dim + 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        diff: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        diff_features = self.encoder(diff)
        cnn_fused = torch.cat(
            [pre_features, post_features, diff_features, torch.abs(pre_features - post_features)],
            dim=1,
        )
        handcrafted = self.feature_encoder(features)
        return self.classifier(torch.cat([cnn_fused, handcrafted], dim=1))

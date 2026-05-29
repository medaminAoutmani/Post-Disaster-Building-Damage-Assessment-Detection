"""Week 13 model heads built on the Week 12 object representation backbone."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CURRENT_DIR = Path(__file__).resolve().parent
WEEK12_DIR = CURRENT_DIR.parent / "week12"
if str(WEEK12_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK12_DIR))

from week12_model_backbones import FUSION_NAMES, build_image_encoder, TemporalFusion


class MultiTaskDamageModel(nn.Module):
    """Shared ordinal representation with class, damaged-presence, and severity heads."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        embedding_dim: int = 256,
        fusion: str = "gated",
        pretrained: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.encoder, feature_dim = build_image_encoder(backbone, pretrained=pretrained)
        self.fusion = TemporalFusion(feature_dim, mode=fusion, dropout=dropout)
        self.embedding_head = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, 4)
        self.presence_head = nn.Linear(embedding_dim, 1)
        self.severity_head = nn.Linear(embedding_dim, 1)

    def encode(self, pre: torch.Tensor, post: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        diff_features = self.encoder(diff)
        embedding = self.embedding_head(self.fusion(pre_features, post_features, diff_features))
        return F.normalize(embedding, dim=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor, diff: torch.Tensor, return_embedding: bool = False) -> dict[str, torch.Tensor]:
        embedding = self.encode(pre, post, diff)
        outputs = {
            "logits": self.classifier(embedding),
            "presence_logit": self.presence_head(embedding).squeeze(1),
            "severity": self.severity_head(embedding).squeeze(1),
        }
        if return_embedding:
            outputs["embedding"] = embedding
        return outputs

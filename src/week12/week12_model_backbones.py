"""Backbones, fusion blocks, and metric-learning heads for Week 12."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


BACKBONE_NAMES = ["resnet18", "resnet34", "efficientnet_b0", "convnext_tiny"]
FUSION_NAMES = ["concat", "gated", "cross_attention"]


def build_image_encoder(backbone: str = "resnet34", pretrained: bool = False) -> tuple[nn.Module, int]:
    """Create an image encoder that returns one feature vector per crop."""
    backbone = backbone.lower()
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    if backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        feature_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        return model, feature_dim
    raise ValueError(f"Unknown Week 12 backbone: {backbone}")


class TemporalFusion(nn.Module):
    """Fuse pre-disaster, post-disaster, and difference embeddings."""

    def __init__(self, feature_dim: int, mode: str = "concat", num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if mode not in FUSION_NAMES:
            raise ValueError(f"Unknown fusion mode {mode}. Expected one of {FUSION_NAMES}.")
        self.mode = mode
        self.feature_dim = feature_dim
        if mode == "gated":
            self.gate = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.Sigmoid())
            self.output_dim = feature_dim * 3
        elif mode == "cross_attention":
            heads = max(1, min(num_heads, feature_dim))
            while feature_dim % heads != 0 and heads > 1:
                heads -= 1
            self.attention = nn.MultiheadAttention(feature_dim, num_heads=heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(feature_dim)
            self.output_dim = feature_dim * 4
        else:
            self.output_dim = feature_dim * 4

    def forward(self, pre: torch.Tensor, post: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(pre - post)
        if self.mode == "gated":
            gate = self.gate(torch.cat([pre, post], dim=1))
            changed = gate * post + (1.0 - gate) * pre
            return torch.cat([changed, diff, delta], dim=1)
        if self.mode == "cross_attention":
            tokens = torch.stack([pre, post, diff, delta], dim=1)
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            return self.norm(tokens + attended).flatten(1)
        return torch.cat([pre, post, diff, delta], dim=1)


class ObjectDamageRepresentationModel(nn.Module):
    """Shared-backbone model that exposes normalized building embeddings and logits."""

    def __init__(
        self,
        backbone: str = "resnet34",
        num_classes: int = 4,
        embedding_dim: int = 256,
        fusion: str = "concat",
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
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, pre: torch.Tensor, post: torch.Tensor, diff: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        diff_features = self.encoder(diff)
        embedding = self.embedding_head(self.fusion(pre_features, post_features, diff_features))
        return F.normalize(embedding, dim=1) if normalize else embedding

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        diff: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(pre, post, diff, normalize=True)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


class ArcMarginProduct(nn.Module):
    """ArcFace angular-margin classifier head."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.3,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return self.scale * ((one_hot * phi) + ((1.0 - one_hot) * cosine))

    def cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return inference logits without applying a label-dependent margin."""
        return self.scale * F.linear(F.normalize(embeddings), F.normalize(self.weight))


class SupConLoss(nn.Module):
    """Supervised contrastive loss for label-aware embedding compactness."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)
        logits = torch.div(torch.matmul(embeddings, embeddings.T), self.temperature)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=embeddings.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positives = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives.clamp_min(1.0)
        valid = positives > 0
        if not torch.any(valid):
            return embeddings.new_tensor(0.0, requires_grad=True)
        return -mean_log_prob_pos[valid].mean()

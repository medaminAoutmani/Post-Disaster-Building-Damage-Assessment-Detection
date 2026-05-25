"""Week 9 multi-task losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from week6.week6_losses import build_loss


class BinaryDiceFromLogitsLoss(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        targets = targets.float()
        dims = (0, 1, 2)
        intersection = (probabilities * targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
        return 1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)


class BuildingCEDiceLoss(nn.Module):
    """Binary building segmentation loss using 2-logit CE plus foreground Dice."""

    def __init__(self, class_weights: list[float] | None = None) -> None:
        super().__init__()
        weight = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)
        self.dice = BinaryDiceFromLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, targets.long()) + self.dice(logits, targets)


class BuildingMaskedDamageCEDiceLoss(nn.Module):
    """Damage CE+Dice computed only on ground-truth building pixels."""

    def __init__(self, class_weights: list[float] | None = None, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer(
            "class_weights",
            torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        valid = targets > 0
        if not bool(valid.any()):
            return logits.new_tensor(0.0)

        ce_map = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        ce_loss = ce_map[valid].mean()

        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        probabilities = probabilities * valid_mask
        target_one_hot = target_one_hot * valid_mask
        dims = (0, 2, 3)
        intersection = (probabilities * target_one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        dice_loss = 1.0 - dice[1:].mean()
        return ce_loss + dice_loss


def build_damage_loss(name: str, class_weights: list[float] | None = None) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized in {"building_masked_ce_dice", "masked_ce_dice", "building_masked_cross_entropy_dice"}:
        return BuildingMaskedDamageCEDiceLoss(class_weights)
    return build_loss(name, class_weights)


class MultiTaskDamageLoss(nn.Module):
    def __init__(
        self,
        damage_loss_name: str = "cross_entropy_dice",
        damage_class_weights: list[float] | None = None,
        building_class_weights: list[float] | None = None,
        lambda_pre: float = 1.0,
        lambda_post: float = 1.0,
        lambda_damage: float = 3.0,
    ) -> None:
        super().__init__()
        self.pre_loss = BuildingCEDiceLoss(building_class_weights)
        self.post_loss = BuildingCEDiceLoss(building_class_weights)
        self.damage_loss = build_damage_loss(damage_loss_name, damage_class_weights)
        self.lambda_pre = lambda_pre
        self.lambda_post = lambda_post
        self.lambda_damage = lambda_damage

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        pre_loss = self.pre_loss(outputs["pre_building_logits"], batch["pre_building_mask"])
        post_loss = self.post_loss(outputs["post_building_logits"], batch["post_building_mask"])
        damage_loss = self.damage_loss(outputs["damage_logits"], batch["damage_mask"])
        total = self.lambda_pre * pre_loss + self.lambda_post * post_loss + self.lambda_damage * damage_loss
        return total, {
            "pre_building_loss": float(pre_loss.detach().item()),
            "post_building_loss": float(post_loss.detach().item()),
            "damage_loss": float(damage_loss.detach().item()),
            "total_loss": float(total.detach().item()),
        }


def damage_confusion_excluding_background(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=1)
    valid = (targets > 0) & (targets < num_classes)
    if not bool(valid.any()):
        return torch.zeros((num_classes, num_classes), dtype=torch.long)
    bins = num_classes * targets[valid].reshape(-1) + predictions[valid].reshape(-1)
    return torch.bincount(bins, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

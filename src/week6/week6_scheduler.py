"""Learning-rate schedulers for Week 6 experiments."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau


def build_scheduler(name: str, optimizer: Optimizer, epochs: int):
    normalized = name.lower().replace("-", "_")
    if normalized in {"none", "constant"}:
        return None
    if normalized in {"cosine", "cosine_annealing"}:
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if normalized in {"reduce_on_plateau", "plateau"}:
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    if normalized == "warmup_cosine":
        return build_warmup_cosine_scheduler(optimizer, warmup_epochs=max(1, epochs // 10), total_epochs=epochs)
    raise ValueError(f"Unknown scheduler: {name}")


def build_warmup_cosine_scheduler(optimizer: Optimizer, warmup_epochs: int, total_epochs: int) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


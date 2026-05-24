"""Callbacks for Week 7 training loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 8
    min_delta: float = 1e-4
    best_score: float = -1.0
    bad_epochs: int = 0

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience


@dataclass
class LRMonitor:
    def values(self, optimizer) -> dict[str, float]:
        return {f"lr_group_{index}": float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}


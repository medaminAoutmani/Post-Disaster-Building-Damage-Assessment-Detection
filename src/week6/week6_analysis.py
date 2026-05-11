"""Research analysis helpers for Week 6 experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_final_metrics(experiment_dir: Path) -> dict[str, float]:
    metrics_path = experiment_dir / "metrics" / "final_metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def summarize_experiments(results_root: Path, output_csv: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for experiment_dir in sorted(results_root.glob("experiment_*")):
        metrics = load_final_metrics(experiment_dir)
        if not metrics:
            continue
        rows.append(
            {
                "experiment": experiment_dir.name,
                "mean_dice": float(metrics.get("val_mean_dice", metrics.get("mean_dice", 0.0))),
                "minor_dice": float(metrics.get("val_dice_minor_damage", metrics.get("dice_minor_damage", 0.0))),
                "major_dice": float(metrics.get("val_dice_major_damage", metrics.get("dice_major_damage", 0.0))),
                "destroyed_dice": float(metrics.get("val_dice_destroyed", metrics.get("dice_destroyed", 0.0))),
                "macro_f1": float(metrics.get("val_macro_f1", metrics.get("macro_f1", 0.0))),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def class_imbalance_summary(class_counts: dict[str, int]) -> dict[str, float]:
    total = float(sum(class_counts.values()))
    if total == 0.0:
        return {key: 0.0 for key in class_counts}
    return {key: value / total for key, value in class_counts.items()}


def confusion_matrix_percentages(confusion: np.ndarray) -> np.ndarray:
    row_sums = confusion.sum(axis=1, keepdims=True)
    return np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums != 0)


def write_analysis_template(experiment_dir: Path) -> None:
    analysis_dir = experiment_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    templates = {
        "failure_analysis.md": "# Failure Analysis\n\n## Strong Cases\n\n- \n\n## Failure Cases\n\n- \n\n## Next Action\n\n- \n",
        "class_imbalance_analysis.md": "# Class Imbalance Analysis\n\n## Observations\n\n- \n\n## Mitigation\n\n- \n",
        "model_comparison.md": "# Model Comparison\n\n## Baseline Comparison\n\n- \n\n## Interpretation\n\n- \n",
    }
    for filename, content in templates.items():
        path = analysis_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")


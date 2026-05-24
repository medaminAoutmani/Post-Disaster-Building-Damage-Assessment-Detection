"""Week 7 research analysis helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_metrics(experiment_dir: Path) -> dict:
    metrics_path = experiment_dir / "metrics" / "final_metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}


def summarize_week7_experiments(results_root: Path, output_csv: Path) -> list[dict]:
    rows = []
    for experiment_dir in sorted(results_root.glob("experiment_*")):
        metrics = load_metrics(experiment_dir)
        if not metrics:
            continue
        rows.append(
            {
                "experiment": experiment_dir.name,
                "best_epoch": metrics.get("best_epoch", ""),
                "val_mean_dice": metrics.get("val_mean_dice", metrics.get("mean_dice", 0.0)),
                "val_rare_class_recall": metrics.get("val_rare_class_recall", 0.0),
                "val_macro_f1": metrics.get("val_macro_f1", metrics.get("macro_f1", 0.0)),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


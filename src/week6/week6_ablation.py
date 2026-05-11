"""Ablation study utilities for Week 6 research reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_ablation_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation_summary.json"
    csv_path = output_dir / "ablation_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def build_ablation_row(
    experiment: str,
    mean_dice: float,
    minor_dice: float,
    major_dice: float | None = None,
    destroyed_dice: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "mean_dice": mean_dice,
        "minor_dice": minor_dice,
        "major_dice": major_dice,
        "destroyed_dice": destroyed_dice,
        "notes": notes,
    }


def compare_to_baseline(rows: list[dict[str, Any]], baseline_name: str = "baseline") -> list[dict[str, Any]]:
    baseline = next((row for row in rows if row.get("experiment") == baseline_name), None)
    if baseline is None:
        return rows
    baseline_dice = float(baseline.get("mean_dice", 0.0))
    compared = []
    for row in rows:
        updated = dict(row)
        updated["delta_mean_dice"] = float(row.get("mean_dice", 0.0)) - baseline_dice
        compared.append(updated)
    return compared


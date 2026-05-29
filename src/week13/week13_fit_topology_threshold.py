"""Fit a topology threshold for true no_damage vs true minor_damage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from week13_topology_features import (
    TOPOLOGY_FEATURE_NAMES,
    feature_matrix,
    load_feature_rows,
    save_json,
    save_topology_csv,
)


def normalize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-7, 1.0, std)
    return (matrix - mean) / std, mean, std


def score_samples(matrix: np.ndarray, no_proto: np.ndarray, minor_proto: np.ndarray) -> np.ndarray:
    distance_to_no = np.linalg.norm(matrix - no_proto[None, :], axis=1)
    distance_to_minor = np.linalg.norm(matrix - minor_proto[None, :], axis=1)
    return distance_to_no - distance_to_minor


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.unique(scores)
    best = {"threshold": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        tp = float(((predictions == 1) & (labels == 1)).sum())
        fp = float(((predictions == 1) & (labels == 0)).sum())
        fn = float(((predictions == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-7)
        if f1 > best["f1"]:
            best = {"threshold": float(threshold), "f1": f1, "precision": precision, "recall": recall}
    return best["threshold"], best


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 13: fit topology no/minor threshold.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings_week8_extra")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--topology-csv", type=Path, default=Path("results") / "week13_topology" / "topology_features.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week13_topology" / "threshold")
    parser.add_argument("--thresholds", type=int, default=16)
    args = parser.parse_args()

    if not args.topology_csv.exists():
        save_topology_csv(args.dataset_root, args.split, args.topology_csv, thresholds=args.thresholds, classes={0, 1})

    rows = [row for row in load_feature_rows(args.topology_csv) if int(row["label"]) in {0, 1}]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    matrix_raw = feature_matrix(rows)
    matrix, mean, std = normalize(matrix_raw)
    no_proto = matrix[labels == 0].mean(axis=0)
    minor_proto = matrix[labels == 1].mean(axis=0)
    scores = score_samples(matrix, no_proto, minor_proto)
    threshold, metrics = best_threshold(scores, labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "feature_names": TOPOLOGY_FEATURE_NAMES,
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
            "no_damage_prototype": no_proto.tolist(),
            "minor_damage_prototype": minor_proto.tolist(),
            "topology_distance_threshold": threshold,
            "metrics": metrics,
            "score_definition": "distance_to_no_damage_prototype - distance_to_minor_damage_prototype",
        },
        args.output_dir / "topology_threshold.json",
    )
    with (args.output_dir / "topology_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metadata_path", "label", "score"])
        for row, score in zip(rows, scores.tolist()):
            writer.writerow([row["metadata_path"], row["label"], score])


if __name__ == "__main__":
    main()

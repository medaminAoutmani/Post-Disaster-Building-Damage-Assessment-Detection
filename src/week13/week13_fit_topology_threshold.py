"""Fit all-class topology prototypes for damage-class validation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from week13_topology_features import (
    CLASS_NAMES,
    TOPOLOGY_FEATURE_NAMES,
    feature_matrix,
    load_feature_rows,
    save_json,
    save_topology_csv,
)


def normalize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matrix) == 0:
        raise ValueError("Cannot normalize an empty topology feature matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Topology feature matrix contains NaN or infinite values. Rebuild the topology CSV from image crops.")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-7, 1.0, std)
    return (matrix - mean) / std, mean, std


def score_samples(matrix: np.ndarray, no_proto: np.ndarray, minor_proto: np.ndarray) -> np.ndarray:
    if not np.isfinite(no_proto).all() or not np.isfinite(minor_proto).all():
        raise ValueError("Cannot compute no/minor topology scores because one prototype contains NaN or infinite values.")
    distance_to_no = np.linalg.norm(matrix - no_proto[None, :], axis=1)
    distance_to_minor = np.linalg.norm(matrix - minor_proto[None, :], axis=1)
    return distance_to_no - distance_to_minor


def prototype_distances(matrix: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    if not np.isfinite(prototypes).all():
        raise ValueError("Cannot compute topology prototype distances because at least one prototype contains NaN or infinite values.")
    return np.linalg.norm(matrix[:, None, :] - prototypes[None, :, :], axis=2)


def confusion_matrix_np(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for prediction, label in zip(predictions.tolist(), labels.tolist()):
        matrix[int(label), int(prediction)] += 1
    return matrix


def metrics_from_confusion_np(matrix: np.ndarray) -> dict[str, float]:
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = true_positive / np.maximum(predicted, 1.0)
    recall = true_positive / np.maximum(support, 1.0)
    f1 = 2.0 * true_positive / np.maximum(support + predicted, 1.0)
    total = max(float(matrix.sum()), 1.0)
    metrics = {
        "accuracy": float(true_positive.sum() / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / total),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f"precision_{class_name}"] = float(precision[index])
        metrics[f"recall_{class_name}"] = float(recall[index])
        metrics[f"f1_{class_name}"] = float(f1[index])
        metrics[f"support_{class_name}"] = float(support[index])
        metrics[f"predicted_{class_name}"] = float(predicted[index])
    return metrics


def save_confusion_csv(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name, *row])


def class_counts(rows: list[dict[str, str]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        label = int(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def print_class_counts(counts: dict[int, int], selected_classes: set[int]) -> None:
    print("Topology feature rows by class:")
    for class_index in sorted(selected_classes):
        print(f"  {CLASS_NAMES[class_index]}: {counts.get(class_index, 0)}")


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    if not np.isfinite(scores).all():
        raise ValueError("Cannot fit topology threshold because scores contain NaN or infinite values.")
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
    parser = argparse.ArgumentParser(description="Week 13: fit all-class topology prototypes.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "week11_buildings_week8_extra")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--topology-csv", type=Path, default=Path("results") / "week13_topology" / "topology_features.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week13_topology" / "threshold")
    parser.add_argument("--thresholds", type=int, default=16)
    parser.add_argument("--rebuild-topology-csv", action="store_true", help="Regenerate topology features even if --topology-csv already exists.")
    parser.add_argument(
        "--allow-missing-classes",
        action="store_true",
        help="Fit prototypes for available classes only. By default, missing requested classes are an error.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class names or ids to include. Default: all damage classes.",
    )
    args = parser.parse_args()

    if args.classes:
        selected_classes = set()
        for value in args.classes:
            if value.isdigit():
                selected_classes.add(int(value))
            else:
                selected_classes.add(CLASS_NAMES.index(value))
    else:
        selected_classes = set(range(len(CLASS_NAMES)))

    if args.rebuild_topology_csv or not args.topology_csv.exists():
        save_topology_csv(args.dataset_root, args.split, args.topology_csv, thresholds=args.thresholds, classes=selected_classes)

    rows = [row for row in load_feature_rows(args.topology_csv) if int(row["label"]) in selected_classes]
    available_classes = {int(row["label"]) for row in rows}
    if not selected_classes.issubset(available_classes):
        missing = sorted(selected_classes - available_classes)
        print(
            "Existing topology CSV is missing requested classes: "
            + ", ".join(CLASS_NAMES[index] for index in missing)
            + ". Regenerating it from the dataset split.",
            file=sys.stderr,
        )
        save_topology_csv(args.dataset_root, args.split, args.topology_csv, thresholds=args.thresholds, classes=selected_classes)
        rows = [row for row in load_feature_rows(args.topology_csv) if int(row["label"]) in selected_classes]
        available_classes = {int(row["label"]) for row in rows}
    counts = class_counts(rows)
    print_class_counts(counts, selected_classes)
    missing_after_rebuild = sorted(selected_classes - available_classes)
    if missing_after_rebuild:
        message = (
            "Dataset split has no topology rows for requested classes: "
            + ", ".join(CLASS_NAMES[index] for index in missing_after_rebuild)
            + ". Use --split train for prototype fitting, choose a dataset split that contains all classes, "
            + "or pass --allow-missing-classes to fit only available prototypes."
        )
        if not args.allow_missing_classes:
            raise ValueError(message)
        print(f"Warning: {message}", file=sys.stderr)
    if len(available_classes) < 2:
        raise ValueError("Topology prototype validation requires at least two classes with samples.")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    matrix_raw = feature_matrix(rows)
    matrix, mean, std = normalize(matrix_raw)
    class_indices = sorted(available_classes)
    prototypes = np.asarray([matrix[labels == class_index].mean(axis=0) for class_index in class_indices], dtype=np.float32)
    if not np.isfinite(prototypes).all():
        raise ValueError(
            "Fitted topology prototypes contain NaN or infinite values. "
            "Check class counts above and rebuild features with --split train --rebuild-topology-csv."
        )
    distances = prototype_distances(matrix, prototypes)
    nearest_positions = np.argmin(distances, axis=1)
    topology_predictions = np.asarray([class_indices[position] for position in nearest_positions], dtype=np.int64)
    confusion = confusion_matrix_np(topology_predictions, labels, len(CLASS_NAMES))
    metrics = metrics_from_confusion_np(confusion)

    threshold = None
    binary_metrics = {}
    scores = np.zeros(len(rows), dtype=np.float32)
    if {0, 1}.issubset(available_classes):
        no_proto = prototypes[class_indices.index(0)]
        minor_proto = prototypes[class_indices.index(1)]
        binary_rows = np.isin(labels, [0, 1])
        scores = score_samples(matrix, no_proto, minor_proto)
        threshold, binary_metrics = best_threshold(scores[binary_rows], labels[binary_rows])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "mode": "all_class_topology_prototype_validation",
            "feature_names": TOPOLOGY_FEATURE_NAMES,
            "class_names": CLASS_NAMES,
            "prototype_class_indices": class_indices,
            "class_prototypes": {CLASS_NAMES[class_index]: prototypes[position].tolist() for position, class_index in enumerate(class_indices)},
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
            "topology_validation_metrics": metrics,
            "topology_distance_threshold": threshold,
            "metrics": metrics,
            "legacy_no_minor_metrics": binary_metrics,
            "score_definition": "nearest normalized topology prototype among all damage classes",
        },
        args.output_dir / "topology_threshold.json",
    )
    save_confusion_csv(confusion, args.output_dir / "topology_confusion_matrix.csv")
    with (args.output_dir / "topology_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metadata_path", "label", "topology_prediction", "topology_prediction_name", "nearest_distance", "legacy_no_minor_score"])
        for row, prediction, nearest_distance, score in zip(rows, topology_predictions.tolist(), distances.min(axis=1).tolist(), scores.tolist()):
            writer.writerow([row["metadata_path"], row["label"], prediction, CLASS_NAMES[prediction], nearest_distance, score])


if __name__ == "__main__":
    main()

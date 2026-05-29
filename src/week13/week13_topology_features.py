"""Topology signatures for Week 13 no-damage/minor-damage calibration."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
WEEK11_DIR = CURRENT_DIR.parent / "week11"
if str(WEEK11_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK11_DIR))

from week11_dataset import CLASS_NAMES, BuildingDamageDataset


TOPOLOGY_FEATURE_NAMES = [
    "building_components_mean",
    "building_components_max",
    "building_holes_mean",
    "building_holes_max",
    "edge_components_mean",
    "edge_components_max",
    "edge_holes_mean",
    "edge_holes_max",
    "diff_components_mean",
    "diff_components_max",
    "diff_holes_mean",
    "diff_holes_max",
    "building_area_ratio",
    "edge_area_ratio",
    "diff_area_ratio",
    "edge_to_building_ratio",
    "diff_to_building_ratio",
    "diff_to_edge_ratio",
    "betti0_wasserstein_edge_diff",
    "betti1_wasserstein_edge_diff",
    "betti0_bottleneck_edge_diff",
    "betti1_bottleneck_edge_diff",
]


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def otsu_mask(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def load_or_estimate_building_mask(sample_dir: Path) -> np.ndarray:
    for name in ["mask.png", "building_mask.png", "post_mask.png"]:
        candidate = sample_dir / name
        if candidate.exists():
            return load_grayscale(candidate)
    post = load_grayscale(sample_dir / "post.png")
    return otsu_mask(post)


def edge_mask(sample_dir: Path) -> np.ndarray:
    post = load_grayscale(sample_dir / "post.png")
    return cv2.Canny(post, 80, 160)


def difference_mask(sample_dir: Path) -> np.ndarray:
    diff = load_grayscale(sample_dir / "diff.png")
    return otsu_mask(diff)


def component_and_hole_counts(mask: np.ndarray) -> tuple[int, int]:
    binary = (mask > 0).astype(np.uint8)
    components = max(cv2.connectedComponents(binary, connectivity=8)[0] - 1, 0)
    inverse = 1 - binary
    labels_count, labels = cv2.connectedComponents(inverse, connectivity=8)
    border_labels = set(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    holes = sum(1 for label in range(1, labels_count) if label not in border_labels)
    return components, holes


def betti_curve(image: np.ndarray, thresholds: int = 16) -> tuple[np.ndarray, np.ndarray]:
    betti0 = []
    betti1 = []
    for threshold in np.linspace(0, 255, thresholds):
        mask = np.where(image >= threshold, 255, 0).astype(np.uint8)
        components, holes = component_and_hole_counts(mask)
        betti0.append(float(components))
        betti1.append(float(holes))
    return np.asarray(betti0, dtype=np.float32), np.asarray(betti1, dtype=np.float32)


def curve_to_diagram(curve: np.ndarray) -> np.ndarray:
    """Represent curve changes as a lightweight persistence-style diagram."""
    pairs = []
    for index in range(1, len(curve)):
        change = abs(float(curve[index] - curve[index - 1]))
        for _ in range(int(round(change))):
            pairs.append((float(index - 1), float(index)))
    if not pairs:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pairs, dtype=np.float32)


def pairwise_linf(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    return np.max(np.abs(left[:, None, :] - right[None, :, :]), axis=2)


def bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 and len(right) == 0:
        return 0.0
    if len(left) == 0 or len(right) == 0:
        return float(max(len(left), len(right)))
    distances = pairwise_linf(left, right)
    return float(np.min(distances, axis=1).max())


def wasserstein_distance(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 and len(right) == 0:
        return 0.0
    if len(left) == 0 or len(right) == 0:
        return float(max(len(left), len(right)))
    distances = pairwise_linf(left, right)
    return float(np.min(distances, axis=1).sum() / max(len(left), 1))


def summarize_curves(prefix: str, betti0: np.ndarray, betti1: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_components_mean": float(betti0.mean()),
        f"{prefix}_components_max": float(betti0.max()),
        f"{prefix}_holes_mean": float(betti1.mean()),
        f"{prefix}_holes_max": float(betti1.max()),
    }


def extract_topology_signature(sample_dir: Path, thresholds: int = 16) -> dict[str, float]:
    building = load_or_estimate_building_mask(sample_dir)
    edges = edge_mask(sample_dir)
    diff = difference_mask(sample_dir)

    building_b0, building_b1 = betti_curve(building, thresholds)
    edge_b0, edge_b1 = betti_curve(edges, thresholds)
    diff_b0, diff_b1 = betti_curve(diff, thresholds)
    edge_d0 = curve_to_diagram(edge_b0)
    edge_d1 = curve_to_diagram(edge_b1)
    diff_d0 = curve_to_diagram(diff_b0)
    diff_d1 = curve_to_diagram(diff_b1)

    building_area = float((building > 0).mean())
    edge_area = float((edges > 0).mean())
    diff_area = float((diff > 0).mean())
    eps = 1e-7
    features = {}
    features.update(summarize_curves("building", building_b0, building_b1))
    features.update(summarize_curves("edge", edge_b0, edge_b1))
    features.update(summarize_curves("diff", diff_b0, diff_b1))
    features.update(
        {
            "building_area_ratio": building_area,
            "edge_area_ratio": edge_area,
            "diff_area_ratio": diff_area,
            "edge_to_building_ratio": edge_area / max(building_area, eps),
            "diff_to_building_ratio": diff_area / max(building_area, eps),
            "diff_to_edge_ratio": diff_area / max(edge_area, eps),
            "betti0_wasserstein_edge_diff": wasserstein_distance(edge_d0, diff_d0),
            "betti1_wasserstein_edge_diff": wasserstein_distance(edge_d1, diff_d1),
            "betti0_bottleneck_edge_diff": bottleneck_distance(edge_d0, diff_d0),
            "betti1_bottleneck_edge_diff": bottleneck_distance(edge_d1, diff_d1),
        }
    )
    return features


def save_topology_csv(dataset_root: Path, split: str, output_path: Path, thresholds: int = 16, classes: set[int] | None = None) -> None:
    dataset = BuildingDamageDataset(dataset_root, split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metadata_path", "sample_dir", "label", "class_name", *TOPOLOGY_FEATURE_NAMES])
        writer.writeheader()
        for sample in dataset.samples:
            label = int(sample["label"])
            if classes is not None and label not in classes:
                continue
            sample_dir = Path(sample["sample_dir"])
            features = extract_topology_signature(sample_dir, thresholds)
            writer.writerow(
                {
                    "metadata_path": str(sample_dir / "metadata.json"),
                    "sample_dir": str(sample_dir),
                    "label": label,
                    "class_name": CLASS_NAMES[label],
                    **features,
                }
            )


def load_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in TOPOLOGY_FEATURE_NAMES] for row in rows], dtype=np.float32)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

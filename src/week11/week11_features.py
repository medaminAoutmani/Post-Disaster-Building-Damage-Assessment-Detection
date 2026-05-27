"""Handcrafted morphology, change, and lightweight topology features for Week 11."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


FEATURE_NAMES = [
    "mask_area_ratio",
    "metadata_area_log",
    "perimeter_log",
    "compactness",
    "bbox_aspect_ratio",
    "bbox_area_ratio",
    "extent",
    "solidity",
    "contour_count",
    "hole_count",
    "euler_number",
    "contour_fragmentation",
    "largest_component_ratio",
    "distance_mean",
    "distance_std",
    "distance_max",
    "edge_density_pre",
    "edge_density_post",
    "edge_density_delta",
    "diff_mean",
    "diff_std",
    "diff_p90",
    "diff_p95",
    "diff_high_ratio",
    "diff_component_count",
    "diff_largest_component_ratio",
    "diff_euler_number",
    "diff_contour_fragmentation",
    "mask_ph_dim0_count",
    "mask_ph_dim0_entropy",
    "mask_ph_dim0_mean_lifetime",
    "mask_ph_dim1_count",
    "mask_ph_dim1_entropy",
    "diff_ph_dim0_count",
    "diff_ph_dim0_entropy",
    "diff_ph_dim1_count",
    "diff_ph_dim1_entropy",
]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely for sparse masks and empty contours."""
    return default if abs(denominator) < 1e-7 else numerator / denominator


def load_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale uint8."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def connected_component_stats(binary: np.ndarray) -> tuple[int, float]:
    """Return foreground component count and largest-component area ratio."""
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    foreground_count = max(0, component_count - 1)
    total_area = float(binary.sum())
    if foreground_count == 0 or total_area <= 0:
        return 0, 0.0
    largest = float(stats[1:, cv2.CC_STAT_AREA].max())
    return foreground_count, safe_divide(largest, total_area)


def hole_count(binary: np.ndarray) -> int:
    """Count holes in a binary foreground mask using contour hierarchy."""
    contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return int(sum(1 for item in hierarchy[0] if item[3] != -1))


def contour_fragmentation(binary: np.ndarray) -> float:
    """Approximate boundary fragmentation by comparing contour perimeter to convex hull perimeter."""
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    hull_perimeter = 0.0
    for contour in contours:
        hull = cv2.convexHull(contour)
        hull_perimeter += float(cv2.arcLength(hull, True))
    return safe_divide(perimeter, hull_perimeter)


def persistence_entropy(lifetimes: np.ndarray) -> float:
    """Compute persistence entropy from finite lifetimes."""
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 1e-7)]
    total = float(lifetimes.sum())
    if lifetimes.size == 0 or total <= 0.0:
        return 0.0
    probabilities = lifetimes / total
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def gudhi_persistence_features(binary: np.ndarray) -> list[float]:
    """Extract optional cubical-complex persistence features with Gudhi.

    Returns zeros when Gudhi is not installed so the feature dimension stays
    stable across machines.
    """
    try:
        import gudhi as gd
    except ImportError:
        return [0.0] * 5

    if int(binary.sum()) == 0:
        return [0.0] * 5

    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    filtration = -distance.astype(np.float64)
    complex_ = gd.CubicalComplex(top_dimensional_cells=filtration)
    persistence = complex_.persistence()
    lifetimes_by_dim = {0: [], 1: []}
    for dimension, (birth, death) in persistence:
        if dimension not in lifetimes_by_dim or not np.isfinite(death):
            continue
        lifetime = death - birth
        if lifetime > 1e-7:
            lifetimes_by_dim[dimension].append(lifetime)

    dim0 = np.asarray(lifetimes_by_dim[0], dtype=np.float64)
    dim1 = np.asarray(lifetimes_by_dim[1], dtype=np.float64)
    return [
        float(dim0.size),
        persistence_entropy(dim0),
        float(dim0.mean()) if dim0.size else 0.0,
        float(dim1.size),
        persistence_entropy(dim1),
    ]


def edge_density(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute Canny edge density, optionally inside a binary mask."""
    edges = cv2.Canny(gray, threshold1=50, threshold2=150) > 0
    if mask is None:
        return float(edges.mean())
    valid = mask > 0
    return safe_divide(float(np.logical_and(edges, valid).sum()), float(valid.sum()))


def extract_feature_vector(sample_dir: Path) -> np.ndarray:
    """Extract morphology, change, and lightweight topology features for one building crop."""
    sample_dir = Path(sample_dir)
    pre_gray = load_gray(sample_dir / "pre.png")
    post_gray = load_gray(sample_dir / "post.png")
    diff_gray = load_gray(sample_dir / "diff.png")
    mask = (load_gray(sample_dir / "mask.png") > 0).astype(np.uint8)
    with (sample_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    image_area = float(mask.shape[0] * mask.shape[1])
    mask_area = float(mask.sum())
    metadata_area = float(metadata.get("area", 0.0))
    perimeter = float(metadata.get("perimeter", 0.0))
    bbox = metadata.get("bbox", [0, 0, 0, 0])
    bbox_width = float(max(1.0, bbox[2] - bbox[0]))
    bbox_height = float(max(1.0, bbox[3] - bbox[1]))
    bbox_area = bbox_width * bbox_height
    compactness = safe_divide(4.0 * np.pi * metadata_area, perimeter * perimeter)
    extent = safe_divide(mask_area, bbox_area)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    hull_area = 0.0
    for contour in contours:
        hull_area += float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = safe_divide(mask_area, hull_area)

    component_count, largest_component_ratio = connected_component_stats(mask)
    holes = hole_count(mask)
    euler_number = component_count - holes
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    valid_distances = distances[mask > 0]
    if valid_distances.size == 0:
        valid_distances = np.asarray([0.0], dtype=np.float32)

    diff_inside = diff_gray[mask > 0] if mask_area > 0 else diff_gray.reshape(-1)
    diff_threshold = max(25.0, float(np.percentile(diff_inside, 85)) if diff_inside.size else 25.0)
    diff_binary = ((diff_gray >= diff_threshold) & (mask > 0)).astype(np.uint8)
    diff_component_count, diff_largest_component_ratio = connected_component_stats(diff_binary)
    diff_holes = hole_count(diff_binary)
    diff_euler_number = diff_component_count - diff_holes
    mask_persistence = gudhi_persistence_features(mask)
    diff_persistence = gudhi_persistence_features(diff_binary)

    features = np.asarray(
        [
            safe_divide(mask_area, image_area),
            float(np.log1p(metadata_area)),
            float(np.log1p(perimeter)),
            compactness,
            safe_divide(bbox_width, bbox_height, default=1.0),
            safe_divide(bbox_area, image_area),
            extent,
            solidity,
            float(contour_count),
            float(holes),
            float(euler_number),
            contour_fragmentation(mask),
            largest_component_ratio,
            float(valid_distances.mean()),
            float(valid_distances.std()),
            float(valid_distances.max()),
            edge_density(pre_gray, mask),
            edge_density(post_gray, mask),
            edge_density(post_gray, mask) - edge_density(pre_gray, mask),
            float(diff_inside.mean()) / 255.0,
            float(diff_inside.std()) / 255.0,
            float(np.percentile(diff_inside, 90)) / 255.0,
            float(np.percentile(diff_inside, 95)) / 255.0,
            safe_divide(float((diff_inside > diff_threshold).sum()), float(diff_inside.size)),
            float(diff_component_count),
            diff_largest_component_ratio,
            float(diff_euler_number),
            contour_fragmentation(diff_binary),
            *mask_persistence,
            *diff_persistence[:2],
            *diff_persistence[3:5],
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

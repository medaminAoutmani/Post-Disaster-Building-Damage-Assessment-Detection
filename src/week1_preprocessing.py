"""Week 1 xBD preprocessing utilities.

Loads pre/post disaster images, parses xBD polygon labels, draws damage overlays,
computes class counts, and rasterizes polygons into segmentation masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.wkt import loads


DAMAGE_CLASS_IDS = {
    "background": 0,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}

# OpenCV uses BGR order.
DAMAGE_COLORS_BGR = {
    "no-damage": (0, 255, 0),       # green
    "minor-damage": (0, 255, 255),  # yellow
    "major-damage": (0, 165, 255),  # orange
    "destroyed": (0, 0, 255),       # red
    "un-classified": (180, 180, 180),
}


def warn(message: str) -> None:
    """Emit a warning with a consistent Week 1 prefix."""
    warnings.warn(f"[week1] {message}", stacklevel=2)


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Load an image with OpenCV and convert BGR to RGB for matplotlib."""
    if not image_path.exists():
        warn(f"Missing image: {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def display_pre_post(pre_image: np.ndarray, post_image: np.ndarray) -> None:
    """Display pre- and post-disaster images side by side."""
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(pre_image)
    axes[0].set_title("Pre-disaster")
    axes[0].axis("off")
    axes[1].imshow(post_image)
    axes[1].set_title("Post-disaster")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def read_label_features(label_path: Path) -> list[dict]:
    """Read xBD JSON labels and return pixel-space building features."""
    if not label_path.exists():
        warn(f"Missing label file: {label_path}")
        raise FileNotFoundError(f"Could not read label file: {label_path}")

    with label_path.open("r", encoding="utf-8") as file:
        label_data = json.load(file)
    features = label_data.get("features", {}).get("xy", [])
    if not features:
        warn(f"No pixel-space features found in: {label_path}")
    return features


def geometry_to_arrays(geometry: Polygon | MultiPolygon, warn_invalid: bool = True) -> list[np.ndarray]:
    """Convert a Shapely polygon or multipolygon to OpenCV-ready point arrays."""
    if geometry.is_empty:
        if warn_invalid:
            warn("Invalid polygon skipped: empty geometry")
        return []

    polygons: Iterable[Polygon]
    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = geometry.geoms
    else:
        if warn_invalid:
            warn(f"Invalid polygon skipped: unsupported geometry type {geometry.geom_type}")
        return []

    arrays: list[np.ndarray] = []
    for polygon in polygons:
        if not polygon.is_valid:
            if warn_invalid:
                warn("Invalid polygon skipped: Shapely geometry is not valid")
            continue
        points = np.asarray(polygon.exterior.coords, dtype=np.float32)
        if len(points) >= 3:
            arrays.append(np.rint(points).astype(np.int32).reshape((-1, 1, 2)))
        else:
            if warn_invalid:
                warn("Invalid polygon skipped: fewer than 3 points")
    return arrays


def parse_polygons(features: list[dict], warn_invalid: bool = True) -> list[tuple[str, list[np.ndarray]]]:
    """Extract subtype and polygon coordinates from xBD features."""
    parsed: list[tuple[str, list[np.ndarray]]] = []
    for feature in features:
        subtype = feature.get("properties", {}).get("subtype", "un-classified")
        wkt_text = feature.get("wkt")
        if not wkt_text:
            if warn_invalid:
                warn("Invalid polygon skipped: missing WKT")
            continue

        try:
            geometry = loads(wkt_text)
        except Exception as error:
            if warn_invalid:
                warn(f"Invalid polygon skipped: WKT parse failed ({error})")
            continue

        polygon_arrays = geometry_to_arrays(geometry, warn_invalid=warn_invalid)
        if polygon_arrays:
            parsed.append((subtype, polygon_arrays))
        elif warn_invalid:
            uid = feature.get("properties", {}).get("uid", "unknown")
            warn(f"Invalid polygon skipped for uid={uid}")
    return parsed


def draw_polygons(
    image_rgb: np.ndarray,
    polygons: list[tuple[str, list[np.ndarray]]],
    fill_alpha: float = 0.35,
    border_thickness: int = 2,
) -> np.ndarray:
    """Draw transparent colored building fills plus outlines on a copy of the image."""
    if not 0.0 <= fill_alpha <= 1.0:
        warn(f"Overlay alpha should be between 0 and 1; clamping {fill_alpha}")
        fill_alpha = min(max(fill_alpha, 0.0), 1.0)

    image_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    fill_layer = image_bgr.copy()

    for subtype, polygon_arrays in polygons:
        color = DAMAGE_COLORS_BGR.get(subtype, DAMAGE_COLORS_BGR["un-classified"])
        cv2.fillPoly(fill_layer, polygon_arrays, color=color)

    overlay_bgr = cv2.addWeighted(fill_layer, fill_alpha, image_bgr, 1.0 - fill_alpha, 0)

    for subtype, polygon_arrays in polygons:
        color = DAMAGE_COLORS_BGR.get(subtype, DAMAGE_COLORS_BGR["un-classified"])
        cv2.polylines(overlay_bgr, polygon_arrays, isClosed=True, color=color, thickness=border_thickness)

    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def create_damage_mask(
    image_shape: tuple[int, int] | tuple[int, int, int],
    polygons: list[tuple[str, list[np.ndarray]]],
    warn_empty: bool = True,
) -> np.ndarray:
    """Rasterize damage polygons into a single-channel class-id mask."""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for subtype, polygon_arrays in polygons:
        class_id = DAMAGE_CLASS_IDS.get(subtype)
        if class_id is None:
            warn(f"Unknown damage subtype skipped in mask: {subtype}")
            continue
        cv2.fillPoly(mask, polygon_arrays, color=class_id)

    if warn_empty and not np.any(mask):
        warn("Empty mask generated: no labeled building pixels were rasterized")

    return mask


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert a grayscale class-id mask into an RGB visualization."""
    colors_rgb = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (255, 255, 0),
        3: (255, 165, 0),
        4: (255, 0, 0),
    }
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in colors_rgb.items():
        color_mask[mask == class_id] = color
    return color_mask


def damage_statistics(features: list[dict]) -> Counter:
    """Count buildings per damage subtype."""
    counts = Counter()
    for feature in features:
        subtype = feature.get("properties", {}).get("subtype", "un-classified")
        counts[subtype] += 1
    return counts


def polygon_bounding_boxes(polygons: list[tuple[str, list[np.ndarray]]]) -> list[dict]:
    """Extract bounding boxes from parsed polygon coordinates."""
    boxes: list[dict] = []
    for subtype, polygon_arrays in polygons:
        for polygon_array in polygon_arrays:
            points = polygon_array.reshape(-1, 2)
            xmin = int(points[:, 0].min())
            ymin = int(points[:, 1].min())
            xmax = int(points[:, 0].max())
            ymax = int(points[:, 1].max())
            boxes.append(
                {
                    "damage_class": subtype,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )
    return boxes


def save_bounding_boxes_csv(boxes: list[dict], output_path: Path) -> None:
    """Save polygon bounding boxes for object detection experiments."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["damage_class", "xmin", "ymin", "xmax", "ymax"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(boxes)


def image_pair_paths(data_dir: Path, sample_id: str, split: str = "train") -> tuple[Path, Path, Path]:
    """Return pre image, post image, and post label paths for a sample id."""
    images_dir = data_dir / split / "images"
    labels_dir = data_dir / split / "labels"
    pre_image = images_dir / f"{sample_id}_pre_disaster.png"
    post_image = images_dir / f"{sample_id}_post_disaster.png"
    post_label = labels_dir / f"{sample_id}_post_disaster.json"
    return pre_image, post_image, post_label


def check_sample_files(pre_path: Path, post_path: Path, label_path: Path) -> bool:
    """Warn about missing files and return whether all expected files exist."""
    paths = [pre_path, post_path, label_path]
    missing = [path for path in paths if not path.exists()]
    for path in missing:
        warn(f"Missing expected file: {path}")
    return not missing


def first_sample_id(data_dir: Path, split: str = "train") -> str:
    """Find the first sample id that has post-disaster labels."""
    labels_dir = data_dir / split / "labels"
    first_label = next(labels_dir.glob("*_post_disaster.json"), None)
    if first_label is None:
        raise FileNotFoundError(f"No post-disaster labels found in {labels_dir}")
    return first_label.stem.replace("_post_disaster", "")


def save_statistics_csv(counts: Counter, output_path: Path) -> None:
    """Save class counts as a CSV table."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["damage_class", "count"])
        for subtype in ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]:
            writer.writerow([subtype, counts.get(subtype, 0)])


def dataset_damage_statistics(data_dir: Path, split: str = "train", check_files: bool = True) -> Counter:
    """Count buildings per damage subtype across all post-disaster labels."""
    labels_dir = data_dir / split / "labels"
    counts = Counter()
    label_paths = sorted(labels_dir.glob("*_post_disaster.json"))

    if not label_paths:
        warn(f"No post-disaster labels found for dataset statistics: {labels_dir}")
        return counts

    for label_path in label_paths:
        sample_id = label_path.stem.replace("_post_disaster", "")
        if check_files:
            pre_path, post_path, _ = image_pair_paths(data_dir, sample_id, split)
            check_sample_files(pre_path, post_path, label_path)
        try:
            counts.update(damage_statistics(read_label_features(label_path)))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            warn(f"Skipping label during dataset statistics: {label_path} ({error})")

    return counts


def save_visualization_examples(
    sample_id: str,
    pre_image: np.ndarray,
    post_image: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Save original images, polygon overlay, grayscale mask, and color mask."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{sample_id}_pre_original.png"), cv2.cvtColor(pre_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{sample_id}_post_original.png"), cv2.cvtColor(post_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{sample_id}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{sample_id}_mask.png"), mask)
    cv2.imwrite(str(output_dir / f"{sample_id}_mask_color.png"), cv2.cvtColor(mask_to_color(mask), cv2.COLOR_RGB2BGR))


def run_sample(
    data_dir: Path,
    output_dir: Path,
    sample_id: str | None = None,
    show: bool = False,
    split: str = "train",
    overlay_alpha: float = 0.35,
) -> Counter:
    """Run the full Week 1 preprocessing pipeline for one sample."""
    sample_id = sample_id or first_sample_id(data_dir, split)
    pre_path, post_path, label_path = image_pair_paths(data_dir, sample_id, split)
    check_sample_files(pre_path, post_path, label_path)

    pre_image = load_image_rgb(pre_path)
    post_image = load_image_rgb(post_path)
    features = read_label_features(label_path)
    polygons = parse_polygons(features)
    overlay = draw_polygons(post_image, polygons, fill_alpha=overlay_alpha)
    mask = create_damage_mask(post_image.shape, polygons)
    counts = damage_statistics(features)
    boxes = polygon_bounding_boxes(polygons)

    save_visualization_examples(sample_id, pre_image, post_image, overlay, mask, output_dir)
    save_statistics_csv(counts, output_dir / f"{sample_id}_damage_statistics.csv")
    save_bounding_boxes_csv(boxes, output_dir / f"{sample_id}_bounding_boxes.csv")

    if show:
        display_pre_post(pre_image, post_image)
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(post_image)
        axes[0].set_title("Post-disaster")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("Damage polygons")
        axes[1].axis("off")
        axes[2].imshow(mask_to_color(mask))
        axes[2].set_title("Segmentation mask")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 1 xBD visualization and mask generation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "visualizations")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample-id", type=str, default=None, help="Example: guatemala-volcano_00000000")
    parser.add_argument("--overlay-alpha", type=float, default=0.35, help="Transparent polygon fill strength.")
    parser.add_argument("--dataset-stats", action="store_true", help="Also compute full dataset damage statistics.")
    parser.add_argument("--only-dataset-stats", action="store_true", help="Skip sample outputs and only save dataset stats.")
    parser.add_argument("--show", action="store_true", help="Display figures with matplotlib.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_stats or args.only_dataset_stats:
        dataset_counts = dataset_damage_statistics(args.data_dir, args.split)
        save_statistics_csv(dataset_counts, args.output_dir / f"{args.split}_dataset_damage_statistics.csv")
        print("Dataset Damage Class\tTotal Buildings")
        for subtype, count in dataset_counts.most_common():
            print(f"{subtype}\t{count}")

    if not args.only_dataset_stats:
        counts = run_sample(
            args.data_dir,
            args.output_dir,
            args.sample_id,
            args.show,
            args.split,
            args.overlay_alpha,
        )
        print("Sample Damage Class\tCount")
        for subtype, count in counts.most_common():
            print(f"{subtype}\t{count}")


if __name__ == "__main__":
    main()

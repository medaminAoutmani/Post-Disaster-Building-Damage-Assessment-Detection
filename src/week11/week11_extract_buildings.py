"""Create object-level xBD building crops for Week 11 classification.

The extractor uses ground-truth xBD polygon labels as the reliable first source
of building instances. Each connected component becomes one object-level sample
with pre/post/difference crops and a metadata JSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from week1_preprocessing import image_pair_paths, load_image_rgb, parse_polygons
from week2_dataset import read_label_features_quiet, read_split_file


DAMAGE_LABELS = ["no_damage", "minor_damage", "major_damage", "destroyed"]
XBD_TO_FOLDER = {
    "no-damage": "no_damage",
    "minor-damage": "minor_damage",
    "major-damage": "major_damage",
    "destroyed": "destroyed",
}


def extract_disaster_type(sample_id: str) -> str:
    """Infer the xBD disaster name from a sample id."""
    parts = sample_id.rsplit("_", maxsplit=1)
    return parts[0] if len(parts) == 2 else sample_id


def square_bbox_from_component(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    """Return a padded square crop box clipped to image bounds."""
    side = max(width, height) + 2 * padding
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    xmin = int(round(center_x - side / 2.0))
    ymin = int(round(center_y - side / 2.0))
    xmax = xmin + side
    ymax = ymin + side

    if xmin < 0:
        xmax -= xmin
        xmin = 0
    if ymin < 0:
        ymax -= ymin
        ymin = 0
    if xmax > image_width:
        xmin -= xmax - image_width
        xmax = image_width
    if ymax > image_height:
        ymin -= ymax - image_height
        ymax = image_height

    return max(0, xmin), max(0, ymin), min(image_width, xmax), min(image_height, ymax)


def resize_rgb(image: np.ndarray, crop_size: int) -> np.ndarray:
    """Resize an RGB crop to a fixed square size."""
    return cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, crop_size: int) -> np.ndarray:
    """Resize a binary mask crop without interpolation artifacts."""
    return cv2.resize(mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """Save an RGB image with OpenCV."""
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def component_metadata(
    sample_id: str,
    building_id: str,
    damage_class: str,
    component_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
    polygon: list[list[int]],
) -> dict:
    """Build metadata for one object-level building sample."""
    ys, xs = np.where(component_mask > 0)
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    centroid = [float(xs.mean()), float(ys.mean())] if len(xs) else [0.0, 0.0]
    return {
        "building_id": building_id,
        "sample_id": sample_id,
        "damage_class": damage_class,
        "damage_class_id": DAMAGE_LABELS.index(damage_class),
        "disaster_type": extract_disaster_type(sample_id),
        "area": int(component_mask.sum()),
        "perimeter": perimeter,
        "centroid": centroid,
        "bbox": list(bbox),
        "crop_bbox": list(crop_bbox),
        "polygon": polygon,
        "source": "xbd_gt_polygon_connected_component",
    }


def write_building_sample(
    output_root: Path,
    split_name: str,
    damage_class: str,
    building_id: str,
    pre_crop: np.ndarray,
    post_crop: np.ndarray,
    mask_crop: np.ndarray,
    metadata: dict,
) -> None:
    """Write one object-level sample folder."""
    sample_dir = output_root / split_name / damage_class / building_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    diff_crop = cv2.absdiff(pre_crop, post_crop)
    save_rgb(sample_dir / "pre.png", pre_crop)
    save_rgb(sample_dir / "post.png", post_crop)
    save_rgb(sample_dir / "diff.png", diff_crop)
    cv2.imwrite(str(sample_dir / "mask.png"), mask_crop)
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def extract_buildings_for_sample(
    data_dir: Path,
    output_root: Path,
    sample_id: str,
    split_name: str,
    data_split: str,
    crop_size: int,
    padding: int,
    min_area: int,
) -> Counter:
    """Extract all valid building instances for one xBD image pair."""
    pre_path, post_path, label_path = image_pair_paths(data_dir, sample_id, data_split)
    if not pre_path.exists() or not post_path.exists() or not label_path.exists():
        return Counter({"missing_files": 1})

    pre_image = load_image_rgb(pre_path)
    post_image = load_image_rgb(post_path)
    height, width = post_image.shape[:2]
    features = read_label_features_quiet(label_path)
    valid_features = [
        feature
        for feature in features
        if feature.get("properties", {}).get("subtype") in XBD_TO_FOLDER
    ]
    polygons = parse_polygons(valid_features, warn_invalid=False)

    stats: Counter = Counter()
    for polygon_index, (xbd_label, polygon_arrays) in enumerate(polygons):
        damage_class = XBD_TO_FOLDER[xbd_label]
        for part_index, polygon_array in enumerate(polygon_arrays):
            raw_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(raw_mask, [polygon_array], color=1)
            component_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)

            for component_index in range(1, component_count):
                area = int(component_stats[component_index, cv2.CC_STAT_AREA])
                if area < min_area:
                    stats["skipped_tiny"] += 1
                    continue

                x = int(component_stats[component_index, cv2.CC_STAT_LEFT])
                y = int(component_stats[component_index, cv2.CC_STAT_TOP])
                component_width = int(component_stats[component_index, cv2.CC_STAT_WIDTH])
                component_height = int(component_stats[component_index, cv2.CC_STAT_HEIGHT])
                bbox = (x, y, x + component_width, y + component_height)
                crop_bbox = square_bbox_from_component(x, y, component_width, component_height, width, height, padding)
                xmin, ymin, xmax, ymax = crop_bbox
                if xmax <= xmin or ymax <= ymin:
                    stats["skipped_empty_crop"] += 1
                    continue

                component_mask = (labels == component_index).astype(np.uint8)
                pre_crop = resize_rgb(pre_image[ymin:ymax, xmin:xmax], crop_size)
                post_crop = resize_rgb(post_image[ymin:ymax, xmin:xmax], crop_size)
                mask_crop = resize_mask(component_mask[ymin:ymax, xmin:xmax] * 255, crop_size)
                polygon_points = polygon_array.reshape(-1, 2).astype(int).tolist()
                building_id = f"{sample_id}_b{polygon_index:04d}_{part_index:02d}_{component_index:02d}"
                metadata = component_metadata(
                    sample_id,
                    building_id,
                    damage_class,
                    component_mask,
                    bbox,
                    crop_bbox,
                    polygon_points,
                )
                write_building_sample(
                    output_root,
                    split_name,
                    damage_class,
                    building_id,
                    pre_crop,
                    post_crop,
                    mask_crop,
                    metadata,
                )
                stats[damage_class] += 1
                stats["written"] += 1

    if not polygons:
        stats["no_valid_polygons"] += 1
    return stats


def save_summary(summary: dict[str, Counter], output_root: Path) -> None:
    """Save extraction counts as JSON and CSV."""
    output_root.mkdir(parents=True, exist_ok=True)
    serializable = {split: dict(counter) for split, counter in summary.items()}
    (output_root / "extraction_summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    with (output_root / "extraction_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["split", "metric", "count"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, counter in summary.items():
            for metric, count in sorted(counter.items()):
                writer.writerow({"split": split_name, "metric": metric, "count": count})


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 11: extract building-level xBD crop dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--data-split", type=str, default="train")
    parser.add_argument("--output-root", type=Path, default=Path("data") / "week11_buildings")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=32)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    args = parser.parse_args()

    summary: dict[str, Counter] = {}
    for split_name in ["train", "val", "test"]:
        sample_ids = read_split_file(args.split_dir / f"{split_name}.txt")
        if args.max_samples_per_split is not None:
            sample_ids = sample_ids[: args.max_samples_per_split]

        split_stats: Counter = Counter()
        for index, sample_id in enumerate(sample_ids, start=1):
            split_stats.update(
                extract_buildings_for_sample(
                    args.data_dir,
                    args.output_root,
                    sample_id,
                    split_name,
                    args.data_split,
                    args.crop_size,
                    args.padding,
                    args.min_area,
                )
            )
            if index % 25 == 0:
                print(f"{split_name}: processed {index}/{len(sample_ids)} samples written={split_stats['written']}")
        summary[split_name] = split_stats
        print(f"{split_name}: {dict(split_stats)}")

    save_summary(summary, args.output_root)


if __name__ == "__main__":
    main()

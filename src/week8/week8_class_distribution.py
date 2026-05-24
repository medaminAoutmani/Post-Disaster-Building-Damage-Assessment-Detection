"""Week 8 class-distribution audit for rare damage classes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


VALID_DAMAGE_CLASSES = {"no-damage", "minor-damage", "major-damage", "destroyed"}
SUBTYPE_TO_CLASS_ID = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
CLASS_NAMES = {
    0: "background",
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}
MOROCCO_KEYWORDS = ["earthquake", "flood", "flooding", "wildfire", "fire"]


def read_split_file(split_path: Path) -> list[str]:
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_pair_paths(data_dir: Path, sample_id: str, split: str = "train") -> tuple[Path, Path, Path]:
    images_dir = data_dir / split / "images"
    labels_dir = data_dir / split / "labels"
    pre_image = images_dir / f"{sample_id}_pre_disaster.png"
    post_image = images_dir / f"{sample_id}_post_disaster.png"
    post_label = labels_dir / f"{sample_id}_post_disaster.json"
    return pre_image, post_image, post_label


def read_features(label_path: Path) -> list[dict]:
    with label_path.open("r", encoding="utf-8") as handle:
        label_data = json.load(handle)
    return label_data.get("features", {}).get("xy", [])


def disaster_group(sample_id: str, keywords: list[str]) -> str:
    lowered = sample_id.lower()
    for keyword in keywords:
        if keyword in lowered:
            if keyword == "flooding":
                return "flood"
            if keyword == "fire":
                return "wildfire"
            return keyword
    return "other"


def empty_class_counter() -> Counter:
    return Counter({class_id: 0 for class_id in CLASS_NAMES})


def split_top_level_groups(text: str) -> list[str]:
    groups = []
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start = index + 1
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(text[start:index])
                start = None
    return groups


def parse_wkt_polygon_rings(wkt_text: str) -> list[list[tuple[float, float]]]:
    """Parse exterior rings from simple xBD POLYGON/MULTIPOLYGON WKT."""
    text = wkt_text.strip()
    upper = text.upper()
    if upper.startswith("POLYGON"):
        body = text[text.find("(") :]
        rings = split_top_level_groups(body)
        return [parse_coordinate_ring(rings[0])] if rings else []
    if upper.startswith("MULTIPOLYGON"):
        body = text[text.find("(") :]
        polygons = split_top_level_groups(body)
        exterior_rings = []
        for polygon in polygons:
            rings = split_top_level_groups(polygon)
            if rings:
                exterior_rings.append(parse_coordinate_ring(rings[0]))
        return exterior_rings
    return []


def parse_coordinate_ring(ring_text: str) -> list[tuple[float, float]]:
    points = []
    for coordinate_pair in ring_text.split(","):
        values = coordinate_pair.strip().split()
        if len(values) < 2:
            continue
        try:
            points.append((float(values[0]), float(values[1])))
        except ValueError:
            continue
    return points


def create_mask(image_size: tuple[int, int], features: list[dict]) -> np.ndarray:
    width, height = image_size
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for feature in features:
        subtype = feature.get("properties", {}).get("subtype")
        class_id = SUBTYPE_TO_CLASS_ID.get(subtype)
        if class_id is None:
            continue
        for ring in parse_wkt_polygon_rings(feature.get("wkt", "")):
            if len(ring) >= 3:
                draw.polygon(ring, fill=class_id)
    return np.asarray(mask_image, dtype=np.uint8)


def analyze_sample(data_dir: Path, data_split: str, sample_id: str) -> tuple[Counter, Counter, set[int]] | None:
    _, post_path, label_path = image_pair_paths(data_dir, sample_id, data_split)
    if not post_path.exists() or not label_path.exists():
        return None

    with Image.open(post_path) as image:
        image_size = image.size

    features = [
        feature
        for feature in read_features(label_path)
        if feature.get("properties", {}).get("subtype") in VALID_DAMAGE_CLASSES
    ]
    if not features:
        return None

    mask = create_mask(image_size, features)
    if int(mask.sum()) == 0:
        return None

    pixel_counts = Counter({class_id: int((mask == class_id).sum()) for class_id in CLASS_NAMES})
    building_counts = empty_class_counter()
    for feature in features:
        subtype = feature.get("properties", {}).get("subtype")
        class_id = SUBTYPE_TO_CLASS_ID.get(subtype)
        if class_id is not None:
            building_counts[class_id] += 1
    present_classes = {class_id for class_id, count in pixel_counts.items() if count > 0}
    return pixel_counts, building_counts, present_classes


def class_rows(split_name: str, totals: dict) -> list[dict]:
    total_pixels = sum(totals["pixels"].values())
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        pixels = int(totals["pixels"][class_id])
        rows.append(
            {
                "split": split_name,
                "class_id": class_id,
                "class_name": class_name,
                "pixels": pixels,
                "pixel_percent": pixels / total_pixels if total_pixels else 0.0,
                "images_containing_class": int(totals["images"][class_id]),
                "buildings": int(totals["buildings"][class_id]),
            }
        )
    return rows


def disaster_rows(split_name: str, disaster_totals: dict[str, dict]) -> list[dict]:
    rows = []
    for disaster, totals in sorted(disaster_totals.items()):
        total_pixels = sum(totals["pixels"].values())
        for class_id, class_name in CLASS_NAMES.items():
            pixels = int(totals["pixels"][class_id])
            rows.append(
                {
                    "split": split_name,
                    "disaster": disaster,
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixels": pixels,
                    "pixel_percent": pixels / total_pixels if total_pixels else 0.0,
                    "images_containing_class": int(totals["images"][class_id]),
                    "buildings": int(totals["buildings"][class_id]),
                }
            )
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_split(data_dir: Path, split_dir: Path, split_name: str, data_split: str, keywords: list[str] | None) -> tuple[dict, dict]:
    sample_ids = read_split_file(split_dir / f"{split_name}.txt")
    if keywords:
        lowered = [keyword.lower() for keyword in keywords]
        sample_ids = [sample_id for sample_id in sample_ids if any(keyword in sample_id.lower() for keyword in lowered)]

    totals = {"pixels": empty_class_counter(), "images": empty_class_counter(), "buildings": empty_class_counter()}
    disaster_totals = defaultdict(lambda: {"pixels": empty_class_counter(), "images": empty_class_counter(), "buildings": empty_class_counter()})
    valid_samples = 0

    for sample_id in sample_ids:
        sample_result = analyze_sample(data_dir, data_split, sample_id)
        if sample_result is None:
            continue
        pixel_counts, building_counts, present_classes = sample_result
        valid_samples += 1
        disaster = disaster_group(sample_id, MOROCCO_KEYWORDS)
        for class_id in CLASS_NAMES:
            totals["pixels"][class_id] += pixel_counts[class_id]
            totals["buildings"][class_id] += building_counts[class_id]
            disaster_totals[disaster]["pixels"][class_id] += pixel_counts[class_id]
            disaster_totals[disaster]["buildings"][class_id] += building_counts[class_id]
            if class_id in present_classes:
                totals["images"][class_id] += 1
                disaster_totals[disaster]["images"][class_id] += 1

    totals["valid_samples"] = valid_samples
    return totals, disaster_totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute train/val class counts for Week 8 imbalance analysis.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week8")
    parser.add_argument("--morocco-adaptation", action="store_true")
    parser.add_argument("--disaster-keywords", nargs="*", default=None)
    args = parser.parse_args()

    keywords = MOROCCO_KEYWORDS if args.morocco_adaptation else args.disaster_keywords
    class_distribution_rows = []
    per_disaster_rows = []
    summary_rows = []
    for split_name in ["train", "val"]:
        totals, disaster_totals = analyze_split(args.data_dir, args.split_dir, split_name, args.data_split, keywords)
        class_distribution_rows.extend(class_rows(split_name, totals))
        per_disaster_rows.extend(disaster_rows(split_name, disaster_totals))
        summary_rows.append({"split": split_name, "valid_samples": totals["valid_samples"]})

    write_csv(class_distribution_rows, args.output_dir / "class_distribution_train_val.csv")
    write_csv(per_disaster_rows, args.output_dir / "per_disaster_class_distribution.csv")
    write_csv(summary_rows, args.output_dir / "sample_count_summary.csv")
    print(f"Wrote Week 8 class audit to {args.output_dir}")


if __name__ == "__main__":
    main()

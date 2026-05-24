"""Select minority-rich xBD samples for Week 8 training expansion.

The script scans a candidate xBD split, counts class pixels/buildings, and writes
a ranked CSV. It does not copy or modify data.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def sample_id_from_post_label(path: Path) -> str:
    return path.name.replace("_post_disaster.json", "")


def read_features(label_path: Path) -> list[dict]:
    with label_path.open("r", encoding="utf-8") as handle:
        label_data = json.load(handle)
    return label_data.get("features", {}).get("xy", [])


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


def parse_wkt_polygon_rings(wkt_text: str) -> list[list[tuple[float, float]]]:
    text = wkt_text.strip()
    upper = text.upper()
    if upper.startswith("POLYGON"):
        rings = split_top_level_groups(text[text.find("(") :])
        return [parse_coordinate_ring(rings[0])] if rings else []
    if upper.startswith("MULTIPOLYGON"):
        exterior_rings = []
        for polygon in split_top_level_groups(text[text.find("(") :]):
            rings = split_top_level_groups(polygon)
            if rings:
                exterior_rings.append(parse_coordinate_ring(rings[0]))
        return exterior_rings
    return []


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


def belongs_to_scope(sample_id: str, keywords: list[str] | None) -> bool:
    if not keywords:
        return True
    lowered = sample_id.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def sample_paths(data_dir: Path, data_split: str, sample_id: str) -> dict[str, Path]:
    split_dir = data_dir / data_split
    return {
        "pre_image": split_dir / "images" / f"{sample_id}_pre_disaster.png",
        "post_image": split_dir / "images" / f"{sample_id}_post_disaster.png",
        "pre_label": split_dir / "labels" / f"{sample_id}_pre_disaster.json",
        "post_label": split_dir / "labels" / f"{sample_id}_post_disaster.json",
    }


def analyze_sample(data_dir: Path, data_split: str, sample_id: str) -> dict | None:
    paths = sample_paths(data_dir, data_split, sample_id)
    if not paths["post_image"].exists() or not paths["post_label"].exists():
        return None

    features = [
        feature
        for feature in read_features(paths["post_label"])
        if feature.get("properties", {}).get("subtype") in VALID_DAMAGE_CLASSES
    ]
    if not features:
        return None

    with Image.open(paths["post_image"]) as image:
        image_size = image.size
    mask = create_mask(image_size, features)
    if int(mask.sum()) == 0:
        return None

    pixel_counts = {CLASS_NAMES[class_id]: int((mask == class_id).sum()) for class_id in CLASS_NAMES}
    building_counts = {CLASS_NAMES[class_id]: 0 for class_id in CLASS_NAMES}
    for feature in features:
        class_id = SUBTYPE_TO_CLASS_ID.get(feature.get("properties", {}).get("subtype"))
        if class_id is not None:
            building_counts[CLASS_NAMES[class_id]] += 1

    minority_pixels = pixel_counts["minor_damage"] + pixel_counts["major_damage"]
    minority_buildings = building_counts["minor_damage"] + building_counts["major_damage"]
    score = (
        5.0 * building_counts["minor_damage"]
        + 3.0 * building_counts["major_damage"]
        + 0.5 * building_counts["destroyed"]
        + 0.0005 * pixel_counts["minor_damage"]
        + 0.0003 * pixel_counts["major_damage"]
    )

    return {
        "sample_id": sample_id,
        "score": score,
        "minority_pixels": minority_pixels,
        "minority_buildings": minority_buildings,
        **{f"{class_name}_pixels": pixel_counts[class_name] for class_name in CLASS_NAMES.values()},
        **{f"{class_name}_buildings": building_counts[class_name] for class_name in CLASS_NAMES.values()},
        "pre_image": str(paths["pre_image"]),
        "post_image": str(paths["post_image"]),
        "pre_label": str(paths["pre_label"]),
        "post_label": str(paths["post_label"]),
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank xBD samples by minor/major damage content.")
    parser.add_argument("--candidate-data-dir", type=Path, required=True)
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--output-csv", type=Path, default=Path("results") / "week8" / "selected_extra_minority_samples.csv")
    parser.add_argument("--morocco-adaptation", action="store_true")
    parser.add_argument("--keywords", nargs="*", default=None)
    parser.add_argument("--min-minor-buildings", type=int, default=1)
    parser.add_argument("--min-major-buildings", type=int, default=0)
    parser.add_argument("--min-minority-pixels", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()

    keywords = MOROCCO_KEYWORDS if args.morocco_adaptation else args.keywords
    label_dir = args.candidate_data_dir / args.data_split / "labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    rows = []
    for label_path in sorted(label_dir.glob("*_post_disaster.json")):
        sample_id = sample_id_from_post_label(label_path)
        if not belongs_to_scope(sample_id, keywords):
            continue
        row = analyze_sample(args.candidate_data_dir, args.data_split, sample_id)
        if row is None:
            continue
        if row["minor_damage_buildings"] < args.min_minor_buildings:
            continue
        if row["major_damage_buildings"] < args.min_major_buildings:
            continue
        if row["minority_pixels"] < args.min_minority_pixels:
            continue
        rows.append(row)

    rows.sort(key=lambda row: (row["score"], row["minor_damage_buildings"], row["major_damage_buildings"]), reverse=True)
    if args.top_k > 0:
        rows = rows[: args.top_k]
    write_csv(rows, args.output_csv)
    print(f"Selected {len(rows)} minority-rich samples")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()

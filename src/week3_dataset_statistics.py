"""Build Week 3 dataset quality statistics for xBD segmentation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from week1_preprocessing import create_damage_mask, image_pair_paths, parse_polygons
from week2_dataset import VALID_DAMAGE_CLASSES, read_label_features_quiet, sample_ids_from_labels


def collect_dataset_statistics(data_dir: Path, split: str = "train") -> tuple[dict[str, int], Counter]:
    """Count valid/skipped samples and valid buildings by damage class."""
    sample_ids = sample_ids_from_labels(data_dir, split)
    sample_counts: Counter = Counter(total_samples=len(sample_ids))
    class_counts: Counter = Counter()

    for sample_id in sample_ids:
        _, post_path, label_path = image_pair_paths(data_dir, sample_id, split)
        if not post_path.exists() or not label_path.exists():
            sample_counts["skipped_samples"] += 1
            sample_counts["missing_files"] += 1
            continue

        post_image = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
        if post_image is None:
            sample_counts["skipped_samples"] += 1
            sample_counts["missing_files"] += 1
            continue

        try:
            features = read_label_features_quiet(label_path)
        except Exception:
            sample_counts["skipped_samples"] += 1
            sample_counts["invalid_label_json"] += 1
            continue

        valid_features = []
        sample_class_counts: Counter = Counter()
        for feature in features:
            subtype = feature.get("properties", {}).get("subtype", "un-classified")
            if subtype in VALID_DAMAGE_CLASSES:
                valid_features.append(feature)
                sample_class_counts[subtype] += 1
            else:
                sample_counts["ignored_unclassified_or_unknown_buildings"] += 1

        if not valid_features:
            sample_counts["skipped_samples"] += 1
            sample_counts["no_valid_damage_class"] += 1
            continue

        polygons = parse_polygons(valid_features, warn_invalid=False)
        if not polygons:
            sample_counts["skipped_samples"] += 1
            sample_counts["invalid_polygons"] += 1
            continue

        mask = create_damage_mask(post_image.shape, polygons, warn_empty=False)
        if int(mask.sum()) == 0:
            sample_counts["skipped_samples"] += 1
            sample_counts["empty_masks"] += 1
            continue

        sample_counts["valid_samples"] += 1
        class_counts.update(sample_class_counts)

    return dict(sample_counts), class_counts


def save_metrics_csv(sample_counts: dict[str, int], class_counts: Counter, output_path: Path) -> None:
    """Save report-friendly dataset statistics as Metric,Value rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("total samples", sample_counts.get("total_samples", 0)),
        ("valid samples", sample_counts.get("valid_samples", 0)),
        ("skipped samples", sample_counts.get("skipped_samples", 0)),
        ("empty masks", sample_counts.get("empty_masks", 0)),
        ("missing files", sample_counts.get("missing_files", 0)),
        ("invalid label json", sample_counts.get("invalid_label_json", 0)),
        ("invalid polygons", sample_counts.get("invalid_polygons", 0)),
        ("no valid damage class", sample_counts.get("no_valid_damage_class", 0)),
        (
            "ignored unclassified/unknown buildings",
            sample_counts.get("ignored_unclassified_or_unknown_buildings", 0),
        ),
    ]
    for subtype in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
        rows.append((f"buildings {subtype}", class_counts.get(subtype, 0)))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Week 3 dataset cleaning statistics.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "logs")
    parser.add_argument("--output-file", type=Path, default=None)
    args = parser.parse_args()

    sample_counts, class_counts = collect_dataset_statistics(args.data_dir, args.split)
    output_path = args.output_file or args.output_dir / f"week3_{args.split}_dataset_statistics.csv"
    save_metrics_csv(sample_counts, class_counts, output_path)

    print("Metric\tValue")
    for metric, value in [
        ("valid samples", sample_counts.get("valid_samples", 0)),
        ("skipped samples", sample_counts.get("skipped_samples", 0)),
        ("empty masks", sample_counts.get("empty_masks", 0)),
    ]:
        print(f"{metric}\t{value}")
    for subtype in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
        print(f"buildings {subtype}\t{class_counts.get(subtype, 0)}")
    print(f"saved\t{output_path}")


if __name__ == "__main__":
    main()

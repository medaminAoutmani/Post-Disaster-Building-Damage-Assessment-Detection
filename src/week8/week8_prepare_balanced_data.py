"""Prepare selected Week 8 minority samples for balanced training.

This script copies selected extra xBD samples into data/week8_extra, creates a
new train split, and writes before/after class-distribution comparison files.
It does not modify the original data/ folder or splits/train.txt.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from week8_class_distribution import CLASS_NAMES, MOROCCO_KEYWORDS, analyze_sample, read_split_file


FILE_COLUMNS = {
    "pre_image": ("images", "_pre_disaster.png"),
    "post_image": ("images", "_post_disaster.png"),
    "pre_label": ("labels", "_pre_disaster.json"),
    "post_label": ("labels", "_post_disaster.json"),
}
MINORITY_CLASSES = ["minor_damage", "major_damage"]


def read_selected_rows(selected_csv: Path, top_k: int | None) -> list[dict]:
    with selected_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if top_k is not None and top_k > 0:
        rows = rows[:top_k]
    return rows


def belongs_to_scope(sample_id: str, keywords: list[str]) -> bool:
    lowered = sample_id.lower()
    return any(keyword in lowered for keyword in keywords)


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def copy_selected_samples(rows: list[dict], output_data_dir: Path, overwrite: bool) -> list[dict]:
    copied_rows = []
    for row in rows:
        sample_id = row["sample_id"]
        copied_files = []
        for column, (subdir, suffix) in FILE_COLUMNS.items():
            source = Path(row[column])
            destination = output_data_dir / "train" / subdir / f"{sample_id}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists():
                raise FileNotFoundError(f"Missing selected source file: {source}")
            if destination.exists() and not overwrite:
                copied_files.append(str(destination))
                continue
            shutil.copy2(source, destination)
            copied_files.append(str(destination))
        copied_rows.append(
            {
                "sample_id": sample_id,
                "copied_file_count": len(copied_files),
                "copied_files": ";".join(copied_files),
                "minor_damage_buildings": row.get("minor_damage_buildings", ""),
                "major_damage_buildings": row.get("major_damage_buildings", ""),
                "minor_damage_pixels": row.get("minor_damage_pixels", ""),
                "major_damage_pixels": row.get("major_damage_pixels", ""),
            }
        )
    return copied_rows


def write_lines(lines: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def empty_class_counter() -> Counter:
    return Counter({class_id: 0 for class_id in CLASS_NAMES})


def summarize_sources(sources: list[tuple[Path, str, list[str]]]) -> dict:
    totals = {"pixels": empty_class_counter(), "images": empty_class_counter(), "buildings": empty_class_counter()}
    valid_samples = 0
    for data_dir, data_split, sample_ids in sources:
        for sample_id in sample_ids:
            result = analyze_sample(data_dir, data_split, sample_id)
            if result is None:
                continue
            pixel_counts, building_counts, present_classes = result
            valid_samples += 1
            for class_id in CLASS_NAMES:
                totals["pixels"][class_id] += pixel_counts[class_id]
                totals["buildings"][class_id] += building_counts[class_id]
                if class_id in present_classes:
                    totals["images"][class_id] += 1
    totals["valid_samples"] = valid_samples
    return totals


def distribution_rows(stage: str, totals: dict) -> list[dict]:
    total_pixels = sum(totals["pixels"].values())
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        pixels = int(totals["pixels"][class_id])
        rows.append(
            {
                "stage": stage,
                "class_id": class_id,
                "class_name": class_name,
                "pixels": pixels,
                "pixel_percent": pixels / total_pixels if total_pixels else 0.0,
                "images_containing_class": int(totals["images"][class_id]),
                "buildings": int(totals["buildings"][class_id]),
                "valid_samples": int(totals["valid_samples"]),
            }
        )
    return rows


def minority_comparison_rows(before: dict, after: dict) -> list[dict]:
    rows = []
    name_to_id = {name: class_id for class_id, name in CLASS_NAMES.items()}
    for class_name in MINORITY_CLASSES:
        class_id = name_to_id[class_name]
        before_pixels = int(before["pixels"][class_id])
        after_pixels = int(after["pixels"][class_id])
        before_buildings = int(before["buildings"][class_id])
        after_buildings = int(after["buildings"][class_id])
        rows.append(
            {
                "class_name": class_name,
                "before_pixels": before_pixels,
                "after_pixels": after_pixels,
                "added_pixels": after_pixels - before_pixels,
                "before_buildings": before_buildings,
                "after_buildings": after_buildings,
                "added_buildings": after_buildings - before_buildings,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy selected Week 8 samples and build a balanced train split.")
    parser.add_argument("--selected-csv", type=Path, default=Path("results") / "week8" / "selected_extra_minority_samples.csv")
    parser.add_argument("--original-data-dir", type=Path, default=Path("data"))
    parser.add_argument("--extra-data-dir", type=Path, default=Path("data") / "week8_extra")
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-train-split", type=Path, default=Path("splits") / "week8_train_balanced.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week8")
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-morocco-adaptation", action="store_false", dest="morocco_adaptation")
    parser.set_defaults(morocco_adaptation=True)
    args = parser.parse_args()

    selected_rows = read_selected_rows(args.selected_csv, args.top_k)
    if not selected_rows:
        raise SystemExit(f"No selected samples found in {args.selected_csv}")

    copied_rows = copy_selected_samples(selected_rows, args.extra_data_dir, args.overwrite)
    write_csv(copied_rows, args.output_dir / "copied_extra_minority_samples.csv")

    original_train_ids = read_split_file(args.split_dir / "train.txt")
    if args.morocco_adaptation:
        original_train_ids = [sample_id for sample_id in original_train_ids if belongs_to_scope(sample_id, MOROCCO_KEYWORDS)]
    selected_ids = [row["sample_id"] for row in selected_rows]
    balanced_train_ids = unique_preserve_order(original_train_ids + selected_ids)
    write_lines(balanced_train_ids, args.output_train_split)

    before = summarize_sources([(args.original_data_dir, "train", original_train_ids)])
    after = summarize_sources(
        [
            (args.original_data_dir, "train", original_train_ids),
            (args.extra_data_dir, "train", selected_ids),
        ]
    )
    write_csv(
        distribution_rows("before", before) + distribution_rows("after", after),
        args.output_dir / "class_distribution_week8_before_after.csv",
    )
    write_csv(
        minority_comparison_rows(before, after),
        args.output_dir / "minority_before_after.csv",
    )

    print(f"Copied {len(copied_rows)} selected samples into {args.extra_data_dir}")
    print(f"Wrote balanced train split: {args.output_train_split}")
    print(f"Wrote comparison: {args.output_dir / 'minority_before_after.csv'}")


if __name__ == "__main__":
    main()

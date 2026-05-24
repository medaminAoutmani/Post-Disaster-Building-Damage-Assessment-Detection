"""Prune xBD samples that are outside the Morocco-adaptation scope.

Dry-run is the default. Deletion requires both --delete and the confirmation
token so candidate files can be inspected first.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MOROCCO_KEYWORDS = ["earthquake", "flood", "flooding", "wildfire", "fire"]
VALID_DAMAGE_CLASSES = {"no-damage", "minor-damage", "major-damage", "destroyed"}
CONFIRMATION_TOKEN = "DELETE_NON_MOROCCO_DATA"


def sample_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in [
        "_pre_disaster.png",
        "_post_disaster.png",
        "_pre_disaster.json",
        "_post_disaster.json",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def collect_sample_ids(data_dir: Path, data_split: str) -> list[str]:
    split_dir = data_dir / data_split
    sample_ids = set()
    for directory, patterns in [
        (split_dir / "images", ["*_pre_disaster.png", "*_post_disaster.png"]),
        (split_dir / "labels", ["*_pre_disaster.json", "*_post_disaster.json"]),
    ]:
        if not directory.exists():
            continue
        for pattern in patterns:
            sample_ids.update(sample_id_from_path(path) for path in directory.glob(pattern))
    return sorted(sample_ids)


def sample_files(data_dir: Path, data_split: str, sample_id: str) -> list[Path]:
    split_dir = data_dir / data_split
    return [
        split_dir / "images" / f"{sample_id}_pre_disaster.png",
        split_dir / "images" / f"{sample_id}_post_disaster.png",
        split_dir / "labels" / f"{sample_id}_pre_disaster.json",
        split_dir / "labels" / f"{sample_id}_post_disaster.json",
    ]


def belongs_to_morocco_scope(sample_id: str, keywords: list[str]) -> bool:
    lowered = sample_id.lower()
    return any(keyword in lowered for keyword in keywords)


def label_has_valid_damage(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    try:
        with label_path.open("r", encoding="utf-8") as handle:
            label_data = json.load(handle)
    except Exception:
        return False
    features = label_data.get("features", {}).get("xy", [])
    return any(feature.get("properties", {}).get("subtype") in VALID_DAMAGE_CLASSES for feature in features)


def reason_for_candidate(data_dir: Path, data_split: str, sample_id: str, keywords: list[str], mode: str) -> str | None:
    is_morocco_scope = belongs_to_morocco_scope(sample_id, keywords)
    post_label = data_dir / data_split / "labels" / f"{sample_id}_post_disaster.json"
    has_valid_damage = label_has_valid_damage(post_label)

    reasons = []
    if not is_morocco_scope:
        reasons.append("outside_morocco_adaptation")
    if not has_valid_damage:
        reasons.append("no_valid_damage_label")

    if mode == "non_morocco" and not is_morocco_scope:
        return "outside_morocco_adaptation"
    if mode == "invalid" and not has_valid_damage:
        return "no_valid_damage_label"
    if mode == "both" and reasons:
        return "+".join(reasons)
    return None


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prune_candidates(rows: list[dict], data_root: Path) -> tuple[int, int]:
    deleted_files = 0
    freed_bytes = 0
    resolved_root = data_root.resolve()
    for row in rows:
        for file_path_text in row["files"].split(";"):
            file_path = Path(file_path_text)
            if not file_path.exists():
                continue
            resolved_path = file_path.resolve()
            if resolved_root not in resolved_path.parents:
                raise ValueError(f"Refusing to delete outside data root: {resolved_path}")
            file_size = file_path.stat().st_size
            file_path.unlink()
            deleted_files += 1
            freed_bytes += file_size
    return deleted_files, freed_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Find or delete non-Morocco xBD samples to recover storage.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--output-csv", type=Path, default=Path("results") / "week8" / "deletion_candidates.csv")
    parser.add_argument("--mode", choices=["non_morocco", "invalid", "both"], default="non_morocco")
    parser.add_argument("--keywords", nargs="*", default=MOROCCO_KEYWORDS)
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--confirm-delete", default="")
    args = parser.parse_args()

    rows = []
    for sample_id in collect_sample_ids(args.data_dir, args.data_split):
        reason = reason_for_candidate(args.data_dir, args.data_split, sample_id, args.keywords, args.mode)
        if reason is None:
            continue
        files = [path for path in sample_files(args.data_dir, args.data_split, sample_id) if path.exists()]
        rows.append(
            {
                "sample_id": sample_id,
                "reason": reason,
                "file_count": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "files": ";".join(str(path) for path in files),
            }
        )

    write_csv(rows, args.output_csv)
    total_bytes = sum(int(row["bytes"]) for row in rows)
    print(f"Found {len(rows)} candidate samples")
    print(f"Candidate size: {total_bytes / (1024 ** 3):.2f} GB")
    print(f"Wrote {args.output_csv}")

    if not args.delete:
        print("Dry run only. Re-run with --delete --confirm-delete DELETE_NON_MOROCCO_DATA to remove files.")
        return
    if args.confirm_delete != CONFIRMATION_TOKEN:
        raise SystemExit(f"Deletion refused. Pass --confirm-delete {CONFIRMATION_TOKEN}")

    deleted_files, freed_bytes = prune_candidates(rows, args.data_dir)
    print(f"Deleted {deleted_files} files")
    print(f"Freed {freed_bytes / (1024 ** 3):.2f} GB")


if __name__ == "__main__":
    main()

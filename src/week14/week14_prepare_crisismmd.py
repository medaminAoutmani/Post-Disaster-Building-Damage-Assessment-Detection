"""Prepare CrisisMMD v2 text tasks for Week 14.

Outputs one CSV per task with a common schema:
split,event,tweet_id,image_id,tweet_text,label,label_id,confidence
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


INFO_LABELS = ["not_informative", "informative"]
HUMAN_LABELS = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage",
    "other_relevant_information",
    "not_humanitarian",
]
DAMAGE_LABELS = ["severe_damage", "mild_damage", "little_or_no_damage"]
IGNORE_LABELS = {"", "nan", "none", "null", "dont_know_or_cant_judge", "don't_know_or_can't_judge"}

DEFAULT_VAL_EVENTS = {"mexico_earthquake"}
DEFAULT_TEST_EVENTS = {"srilanka_floods"}


def clean_label(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower().replace(" ", "_").replace(",", "")


def parse_confidence(value: str | None) -> float:
    try:
        parsed = float(value or "nan")
    except ValueError:
        return math.nan
    return parsed


def event_from_path(path: Path) -> str:
    return path.name.replace("_final_data.tsv", "")


def split_for_event(event: str, val_events: set[str], test_events: set[str]) -> str:
    if event in test_events:
        return "test"
    if event in val_events:
        return "val"
    return "train"


def read_annotations(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((root / "annotations").glob("*_final_data.tsv")):
        if path.name.startswith("._"):
            continue
        event = event_from_path(path)
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                row["event"] = event
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No CrisisMMD TSV files found under {root / 'annotations'}")
    return rows


def best_text_rows(rows: list[dict[str, str]], label_col: str, conf_col: str, allowed: list[str]) -> list[dict[str, str]]:
    allowed_set = set(allowed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = clean_label(row.get(label_col))
        if label in IGNORE_LABELS or label not in allowed_set:
            continue
        grouped[str(row["tweet_id"])].append(row)

    selected = []
    for tweet_rows in grouped.values():
        selected.append(max(tweet_rows, key=lambda item: parse_confidence(item.get(conf_col))))
    return selected


def build_task_rows(
    rows: list[dict[str, str]],
    label_col: str,
    conf_col: str,
    labels: list[str],
    val_events: set[str],
    test_events: set[str],
    dedupe_by_tweet: bool = True,
) -> list[dict[str, str | int | float]]:
    source_rows = best_text_rows(rows, label_col, conf_col, labels) if dedupe_by_tweet else rows
    label_to_id = {label: index for index, label in enumerate(labels)}
    task_rows = []
    for row in source_rows:
        label = clean_label(row.get(label_col))
        if label in IGNORE_LABELS or label not in label_to_id:
            continue
        task_rows.append(
            {
                "split": split_for_event(row["event"], val_events, test_events),
                "event": row["event"],
                "tweet_id": row["tweet_id"],
                "image_id": row.get("image_id", ""),
                "tweet_text": row.get("tweet_text", ""),
                "label": label,
                "label_id": label_to_id[label],
                "confidence": parse_confidence(row.get(conf_col)),
            }
        )
    return task_rows


def write_csv(rows: list[dict[str, str | int | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "event", "tweet_id", "image_id", "tweet_text", "label", "label_id", "confidence"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str | int | float]]) -> dict[str, dict[str, int]]:
    by_split = Counter(str(row["split"]) for row in rows)
    by_label = Counter(str(row["label"]) for row in rows)
    return {"split_counts": dict(by_split), "label_counts": dict(by_label)}


def write_emotion_prompt_jsonl(rows: list[dict[str, str | int | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            prompt = (
                "Task: Classify the crisis tweet emotion into exactly one label from "
                "[Fear, Sadness, Anger, Hope, Neutral]. Answer only one label.\n\n"
                f"Tweet: {row['tweet_text']}"
            )
            payload = {
                "tweet_id": row["tweet_id"],
                "event": row["event"],
                "tweet_text": row["tweet_text"],
                "prompt": prompt,
                "allowed_labels": ["Fear", "Sadness", "Anger", "Hope", "Neutral"],
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14: prepare CrisisMMD v2 text task CSVs.")
    parser.add_argument("--crisismmd-root", type=Path, default=Path("data") / "CrisisMMD_v2.0")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week14_crisismmd" / "processed")
    parser.add_argument("--val-events", nargs="*", default=sorted(DEFAULT_VAL_EVENTS))
    parser.add_argument("--test-events", nargs="*", default=sorted(DEFAULT_TEST_EVENTS))
    args = parser.parse_args()

    rows = read_annotations(args.crisismmd_root)
    val_events = set(args.val_events)
    test_events = set(args.test_events)

    tasks = {
        "informativeness": build_task_rows(rows, "text_info", "text_info_conf", INFO_LABELS, val_events, test_events),
        "humanitarian": build_task_rows(rows, "text_human", "text_human_conf", HUMAN_LABELS, val_events, test_events),
        "damage_severity": build_task_rows(rows, "image_damage", "image_damage_conf", DAMAGE_LABELS, val_events, test_events),
    }
    for task_name, task_rows in tasks.items():
        write_csv(task_rows, args.output_dir / f"{task_name}.csv")

    write_emotion_prompt_jsonl(tasks["humanitarian"], args.output_dir / "emotion_prompts.jsonl")
    (args.output_dir / "label_maps.json").write_text(
        json.dumps({"informativeness": INFO_LABELS, "humanitarian": HUMAN_LABELS, "damage_severity": DAMAGE_LABELS}, indent=2),
        encoding="utf-8",
    )
    summary = {task_name: summarize(task_rows) for task_name, task_rows in tasks.items()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

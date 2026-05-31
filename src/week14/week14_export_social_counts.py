"""Export CrisisMMD aggregate counts for Week 15 fusion."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


HUMANITARIAN_ALIASES = {
    "affected_individuals": "affected_people",
    "infrastructure_and_utility_damage": "infrastructure_damage",
    "rescue_volunteering_or_donation_effort": "rescue_efforts",
    "injured_or_dead_people": "injured_dead",
    "missing_or_found_people": "missing_or_found_people",
    "vehicle_damage": "vehicle_damage",
    "other_relevant_information": "other_relevant_information",
}


def read_rows(path: Path, split: str | None, event: str | None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if split:
        rows = [row for row in rows if row.get("split") == split]
    if event:
        rows = [row for row in rows if row.get("event") == event]
    return rows


def representative_posts(rows: list[dict[str, str]], limit: int) -> list[str]:
    posts = []
    seen = set()
    for row in rows:
        text = row.get("tweet_text", "").strip()
        if text and text not in seen:
            posts.append(text)
            seen.add(text)
        if len(posts) >= limit:
            break
    return posts


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14: export social-media aggregate counts for Week 15 fusion.")
    parser.add_argument("--processed-dir", type=Path, default=Path("results") / "week14_crisismmd" / "processed")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--event", help="Optional CrisisMMD event filter.")
    parser.add_argument("--top-posts", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=Path("results") / "week15_inputs" / "social.json")
    args = parser.parse_args()

    info_rows = read_rows(args.processed_dir / "informativeness.csv", args.split, args.event)
    humanitarian_rows = read_rows(args.processed_dir / "humanitarian.csv", args.split, args.event)
    emotion_rows = read_rows(args.processed_dir / "emotion.csv", args.split, args.event)
    disaster_rows = read_rows(args.processed_dir / "disaster_type.csv", args.split, args.event)

    informative_rows = [row for row in info_rows if row.get("label") == "informative"]
    humanitarian_counts = Counter()
    for row in humanitarian_rows:
        label = row.get("label", "")
        if label == "not_humanitarian":
            continue
        humanitarian_counts[HUMANITARIAN_ALIASES.get(label, label)] += 1

    emotion_counts = Counter()
    for row in emotion_rows:
        label = row.get("emotion") or row.get("label", "")
        if label:
            emotion_counts[label.lower()] += 1

    disaster_counts = Counter()
    for row in disaster_rows:
        label = row.get("disaster_type") or row.get("label", "")
        if label:
            disaster_counts[label.lower()] += 1

    candidate_posts = informative_rows or humanitarian_rows
    if not candidate_posts and (emotion_rows or disaster_rows):
        candidate_posts = emotion_rows or disaster_rows
    output = {
        "source": "week14_crisismmd",
        "processed_dir": str(args.processed_dir),
        "split": args.split,
        "event": args.event,
        "informative_posts": len(informative_rows),
        "humanitarian": dict(humanitarian_counts),
        "emotion": dict(emotion_counts),
        "disaster_type": dict(disaster_counts),
        "representative_posts": representative_posts(candidate_posts, args.top_posts),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

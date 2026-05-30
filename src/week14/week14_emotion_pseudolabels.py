"""Filter LLM pseudo-labels for Week 14 emotion detection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EMOTION_LABELS = ["Fear", "Sadness", "Anger", "Hope", "Neutral"]


def normalize_emotion(value: str | None) -> str:
    if value is None:
        return ""
    normalized = value.strip().strip('"').strip("'").lower()
    for label in EMOTION_LABELS:
        if normalized == label.lower():
            return label
    return ""


def load_base_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows[str(row["tweet_id"])] = row
    return rows


def iter_llm_outputs(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def parse_confidence(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14: majority-vote emotion pseudo-labels.")
    parser.add_argument("--base-csv", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "humanitarian.csv")
    parser.add_argument("--llm-jsonl", type=Path, required=True, help="JSONL with tweet_id plus emotion/label/output.")
    parser.add_argument("--output-csv", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "emotion.csv")
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument("--min-agreement", type=float, default=0.67)
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Drop individual LLM labels below this confidence when provided.")
    args = parser.parse_args()

    base_rows = load_base_rows(args.base_csv)
    votes: dict[str, list[str]] = defaultdict(list)
    for payload in iter_llm_outputs(args.llm_jsonl):
        tweet_id = str(payload.get("tweet_id", ""))
        raw_label = payload.get("emotion") or payload.get("label") or payload.get("output")
        label = normalize_emotion(raw_label)
        confidence = parse_confidence(payload.get("confidence"))
        if tweet_id in base_rows and label and confidence >= args.min_confidence:
            votes[tweet_id].append(label)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "event", "tweet_id", "tweet_text", "emotion", "label_id", "agreement", "num_votes"]
    kept = 0
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for tweet_id, labels in sorted(votes.items()):
            counts = Counter(labels)
            emotion, count = counts.most_common(1)[0]
            agreement = count / len(labels)
            if len(labels) < args.min_votes or agreement < args.min_agreement:
                continue
            base = base_rows[tweet_id]
            writer.writerow(
                {
                    "split": base["split"],
                    "event": base["event"],
                    "tweet_id": tweet_id,
                    "tweet_text": base["tweet_text"],
                    "emotion": emotion,
                    "label_id": EMOTION_LABELS.index(emotion),
                    "agreement": round(agreement, 4),
                    "num_votes": len(labels),
                }
            )
            kept += 1
    print(f"Kept {kept} emotion pseudo-labels at agreement >= {args.min_agreement}.")


if __name__ == "__main__":
    main()

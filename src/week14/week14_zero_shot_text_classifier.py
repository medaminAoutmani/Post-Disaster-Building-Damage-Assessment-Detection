"""Zero-shot DeBERTa crisis tweet classification for Week 14.

This script uses an NLI/zero-shot DeBERTa checkpoint directly, without
fine-tuning on the local CrisisMMD labels. It can classify crisis tweets into
emotion labels and disaster-type labels, then export CSVs compatible with the
Week 15 social fusion inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EMOTION_LABELS = ["Fear", "Sadness", "Anger", "Hope", "Neutral"]
DISASTER_LABELS = ["earthquake", "flood", "hurricane", "wildfire", "other disaster"]
DEFAULT_VAL_EVENTS = {"mexico_earthquake"}
DEFAULT_TEST_EVENTS = {"srilanka_floods"}
IGNORE_LABELS = {"", "nan", "none", "null", "dont_know_or_cant_judge", "don't_know_or_can't_judge"}
DEFAULT_MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1"


def clean_label(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower().replace(" ", "_").replace(",", "")


def safe_float(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
        return math.nan


def split_for_event(event: str, val_events: set[str], test_events: set[str]) -> str:
    if event in test_events:
        return "test"
    if event in val_events:
        return "val"
    return "train"


def event_from_path(path: Path) -> str:
    return path.name.replace("_final_data.tsv", "")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_raw_annotation_rows(root: Path, val_events: set[str], test_events: set[str]) -> list[dict[str, str]]:
    annotation_dir = root / "annotations" if (root / "annotations").exists() else root
    grouped: dict[str, dict[str, str]] = {}
    for path in sorted(annotation_dir.glob("*_final_data.tsv")):
        if path.name.startswith("._"):
            continue
        event = event_from_path(path)
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                tweet_id = str(row.get("tweet_id", "")).strip()
                tweet_text = str(row.get("tweet_text", "")).strip()
                if not tweet_id or not tweet_text:
                    continue
                candidate = {
                    "split": split_for_event(event, val_events, test_events),
                    "event": event,
                    "tweet_id": tweet_id,
                    "image_id": str(row.get("image_id", "")),
                    "tweet_text": tweet_text,
                    "source_text_info": clean_label(row.get("text_info")),
                    "source_text_human": clean_label(row.get("text_human")),
                    "source_confidence": str(row.get("text_info_conf", "")),
                }
                previous = grouped.get(tweet_id)
                if previous is None or safe_float(candidate["source_confidence"]) > safe_float(previous.get("source_confidence")):
                    grouped[tweet_id] = candidate
    if not grouped:
        raise FileNotFoundError(f"No CrisisMMD annotation rows found under {annotation_dir}")
    return list(grouped.values())


def filter_rows(rows: list[dict[str, str]], split: str | None, event: str | None, informative_only: bool) -> list[dict[str, str]]:
    filtered = rows
    if split:
        filtered = [row for row in filtered if row.get("split") == split]
    if event:
        filtered = [row for row in filtered if row.get("event") == event]
    if informative_only:
        filtered = [
            row
            for row in filtered
            if row.get("label") == "informative" or row.get("source_text_info") in {"informative", ""}
        ]
    return filtered


def normalize_output_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def batched(rows: list[dict[str, str]], batch_size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def classify_rows(
    rows: list[dict[str, str]],
    classifier,
    labels: list[str],
    hypothesis_template: str,
    batch_size: int,
) -> list[dict[str, str | float]]:
    predictions: list[dict[str, str | float]] = []
    for batch_rows in batched(rows, batch_size):
        texts = [row["tweet_text"] for row in batch_rows]
        outputs = classifier(
            texts,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
            batch_size=batch_size,
            truncation=True,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        for row, output in zip(batch_rows, outputs):
            label = str(output["labels"][0])
            score = float(output["scores"][0])
            predictions.append(
                {
                    "split": row.get("split", ""),
                    "event": row.get("event", ""),
                    "tweet_id": row.get("tweet_id", ""),
                    "tweet_text": row.get("tweet_text", ""),
                    "label": normalize_output_label(label),
                    "display_label": label,
                    "confidence": round(score, 6),
                }
            )
    return predictions


def write_csv(rows: list[dict[str, str | int | float]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_emotion_csv(predictions: list[dict[str, str | float]], path: Path, labels: list[str]) -> None:
    label_to_id = {label.lower(): index for index, label in enumerate(labels)}
    rows = []
    for item in predictions:
        emotion = str(item["display_label"])
        rows.append(
            {
                "split": item["split"],
                "event": item["event"],
                "tweet_id": item["tweet_id"],
                "tweet_text": item["tweet_text"],
                "emotion": emotion,
                "label_id": label_to_id[emotion.lower()],
                "agreement": item["confidence"],
                "num_votes": 1,
            }
        )
    write_csv(rows, path, ["split", "event", "tweet_id", "tweet_text", "emotion", "label_id", "agreement", "num_votes"])


def write_disaster_csv(predictions: list[dict[str, str | float]], path: Path, labels: list[str]) -> None:
    label_to_id = {normalize_output_label(label): index for index, label in enumerate(labels)}
    rows = []
    for item in predictions:
        label = str(item["label"])
        rows.append(
            {
                "split": item["split"],
                "event": item["event"],
                "tweet_id": item["tweet_id"],
                "tweet_text": item["tweet_text"],
                "disaster_type": label,
                "label_id": label_to_id[label],
                "confidence": item["confidence"],
            }
        )
    write_csv(rows, path, ["split", "event", "tweet_id", "tweet_text", "disaster_type", "label_id", "confidence"])


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


def write_social_json(
    rows: list[dict[str, str]],
    emotion_predictions: list[dict[str, str | float]],
    disaster_predictions: list[dict[str, str | float]],
    output_json: Path,
    source: str,
    split: str | None,
    event: str | None,
    top_posts: int,
) -> None:
    emotion_counts = Counter(str(item["label"]) for item in emotion_predictions)
    disaster_counts = Counter(str(item["label"]) for item in disaster_predictions)
    by_tweet: dict[str, dict[str, Any]] = {
        row["tweet_id"]: {"tweet_id": row["tweet_id"], "tweet_text": row["tweet_text"]}
        for row in rows
    }
    for item in emotion_predictions:
        by_tweet[str(item["tweet_id"])].update(
            {
                "emotion": item["label"],
                "emotion_display": item["display_label"],
                "emotion_confidence": item["confidence"],
            }
        )
    for item in disaster_predictions:
        by_tweet[str(item["tweet_id"])].update(
            {
                "disaster_type": item["label"],
                "disaster_type_display": item["display_label"],
                "disaster_type_confidence": item["confidence"],
            }
        )
    output = {
        "source": source,
        "split": split,
        "event": event,
        "classified_posts": len(rows),
        "informative_posts": len(rows),
        "emotion": dict(emotion_counts),
        "disaster_type": dict(disaster_counts),
        "included_tweets": list(by_tweet.values()),
        "representative_posts": representative_posts(rows, top_posts),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")


def rows_from_tweets(tweets: list[str], event: str = "", split: str = "") -> list[dict[str, str]]:
    rows = []
    seen = set()
    for index, tweet in enumerate(tweets, start=1):
        text = tweet.strip()
        if not text or text in seen:
            continue
        rows.append(
            {
                "split": split,
                "event": event,
                "tweet_id": f"included_tweet_{index}",
                "tweet_text": text,
            }
        )
        seen.add(text)
    return rows


def build_zero_shot_social_payload(
    tweets: list[str],
    classifier,
    event: str = "",
    split: str = "",
    emotion_labels: list[str] | None = None,
    disaster_labels: list[str] | None = None,
    batch_size: int = 8,
    top_posts: int = 5,
    source: str = "week14_zero_shot_deberta_streamlit",
) -> dict[str, Any]:
    labels_emotion = parse_labels(emotion_labels, EMOTION_LABELS)
    labels_disaster = parse_labels(disaster_labels, DISASTER_LABELS)
    rows = rows_from_tweets(tweets, event=event, split=split)
    emotion_predictions = classify_rows(
        rows,
        classifier,
        labels_emotion,
        hypothesis_template="The emotion expressed in this crisis tweet is {}.",
        batch_size=batch_size,
    )
    disaster_predictions = classify_rows(
        rows,
        classifier,
        labels_disaster,
        hypothesis_template="This crisis tweet is about a {}.",
        batch_size=batch_size,
    )

    by_tweet: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_tweet[row["tweet_id"]] = {"tweet_id": row["tweet_id"], "tweet_text": row["tweet_text"]}
    for item in emotion_predictions:
        by_tweet[str(item["tweet_id"])].update(
            {
                "emotion": item["label"],
                "emotion_display": item["display_label"],
                "emotion_confidence": item["confidence"],
            }
        )
    for item in disaster_predictions:
        by_tweet[str(item["tweet_id"])].update(
            {
                "disaster_type": item["label"],
                "disaster_type_display": item["display_label"],
                "disaster_type_confidence": item["confidence"],
            }
        )

    emotion_counts = Counter(str(item["label"]) for item in emotion_predictions)
    disaster_counts = Counter(str(item["label"]) for item in disaster_predictions)
    return {
        "source": source,
        "split": split or None,
        "event": event or None,
        "classified_posts": len(rows),
        "informative_posts": len(rows),
        "emotion": dict(emotion_counts),
        "disaster_type": dict(disaster_counts),
        "included_tweets": list(by_tweet.values()),
        "representative_posts": representative_posts(rows, top_posts),
    }


def parse_labels(values: list[str] | None, defaults: list[str]) -> list[str]:
    if not values:
        return defaults
    labels = [value.strip() for value in values if value.strip()]
    if len(labels) < 2:
        raise ValueError("Zero-shot classification needs at least two candidate labels.")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14: zero-shot DeBERTa emotion and disaster-type classification.")
    parser.add_argument("--input-csv", type=Path, help="Optional prepared CSV with tweet_text, tweet_id, split, and event columns.")
    parser.add_argument("--crisismmd-root", type=Path, default=Path("data") / "CrisisMMD_v2.0")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "week14_crisismmd" / "zero_shot")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--event", help="Optional CrisisMMD event filter, for example mexico_earthquake.")
    parser.add_argument("--val-events", nargs="*", default=sorted(DEFAULT_VAL_EVENTS))
    parser.add_argument("--test-events", nargs="*", default=sorted(DEFAULT_TEST_EVENTS))
    parser.add_argument("--emotion-labels", nargs="*", help="Override emotion labels.")
    parser.add_argument("--disaster-labels", nargs="*", help="Override disaster-type labels.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, help="Optional cap for smoke tests or fast demos.")
    parser.add_argument("--include-non-informative", action="store_true", help="Classify all rows instead of only informative/raw rows.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a pipeline device index such as 0.")
    parser.add_argument("--top-posts", type=int, default=5)
    parser.add_argument("--write-social-json", action="store_true", help="Also write a Week 15-style social JSON with zero-shot counts.")
    parser.add_argument("--social-json", type=Path, default=Path("results") / "week15_inputs" / "social_zero_shot.json")
    args = parser.parse_args()

    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Install transformers, sentencepiece, and torch to run zero-shot DeBERTa classification.") from exc

    val_events = set(args.val_events)
    test_events = set(args.test_events)
    rows = read_csv_rows(args.input_csv) if args.input_csv else read_raw_annotation_rows(args.crisismmd_root, val_events, test_events)
    rows = filter_rows(rows, args.split, args.event, informative_only=not args.include_non_informative)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No rows remain after filtering.")

    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif args.device == "cpu":
        device = -1
    elif args.device == "cuda":
        device = 0
    else:
        device = int(args.device)

    classifier = pipeline("zero-shot-classification", model=args.model_name, tokenizer=args.model_name, device=device)
    emotion_labels = parse_labels(args.emotion_labels, EMOTION_LABELS)
    disaster_labels = parse_labels(args.disaster_labels, DISASTER_LABELS)

    emotion_predictions = classify_rows(
        rows,
        classifier,
        emotion_labels,
        hypothesis_template="The emotion expressed in this crisis tweet is {}.",
        batch_size=args.batch_size,
    )
    disaster_predictions = classify_rows(
        rows,
        classifier,
        disaster_labels,
        hypothesis_template="This crisis tweet is about a {}.",
        batch_size=args.batch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_emotion_csv(emotion_predictions, args.output_dir / "emotion.csv", emotion_labels)
    write_disaster_csv(disaster_predictions, args.output_dir / "disaster_type.csv", disaster_labels)
    write_csv(
        emotion_predictions,
        args.output_dir / "emotion_predictions.csv",
        ["split", "event", "tweet_id", "tweet_text", "label", "display_label", "confidence"],
    )
    write_csv(
        disaster_predictions,
        args.output_dir / "disaster_type_predictions.csv",
        ["split", "event", "tweet_id", "tweet_text", "label", "display_label", "confidence"],
    )

    summary = {
        "model_name": args.model_name,
        "rows": len(rows),
        "split": args.split,
        "event": args.event,
        "emotion_counts": dict(Counter(str(item["label"]) for item in emotion_predictions)),
        "disaster_type_counts": dict(Counter(str(item["label"]) for item in disaster_predictions)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.write_social_json:
        write_social_json(
            rows,
            emotion_predictions,
            disaster_predictions,
            args.social_json,
            source=f"week14_zero_shot:{args.model_name}",
            split=args.split,
            event=args.event,
            top_posts=args.top_posts,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

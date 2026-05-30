"""Generate Week 14 emotion pseudo-labels with the Gemini API.

Input:  results/week14_crisismmd/processed/emotion_prompts.jsonl
Output: results/week14_crisismmd/processed/emotion_llm_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


EMOTION_LABELS = ["Fear", "Sadness", "Anger", "Hope", "Neutral"]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for payload in read_jsonl(path):
        tweet_id = str(payload.get("tweet_id", ""))
        if tweet_id:
            ids.add(tweet_id)
    return ids


def parse_label(payload: dict) -> dict[str, str | float]:
    emotion = str(payload.get("emotion", "")).strip()
    if emotion not in EMOTION_LABELS:
        raise ValueError(f"Invalid emotion label: {emotion}")
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"emotion": emotion, "confidence": max(0.0, min(confidence, 1.0))}


def parse_output(text: str) -> dict[str, str | float]:
    payload = json.loads(text)
    return parse_label(payload)


def parse_batch_output(text: str) -> list[dict[str, str | float]]:
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("Expected Gemini to return a JSON array.")
    labels = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Expected object in Gemini array, got {type(item)}")
        parsed = parse_label(item)
        parsed["tweet_id"] = str(item.get("tweet_id", ""))
        labels.append(parsed)
    return labels


def label_one(client, types, model: str, tweet_text: str) -> dict[str, str | float]:
    prompt = (
        "You label crisis-related tweets for emotion analysis.\n"
        "Choose the dominant emotion expressed by the tweet text.\n"
        "Use exactly one label from Fear, Sadness, Anger, Hope, Neutral.\n"
        "If the tweet is mostly factual, informational, promotional, or unclear, choose Neutral.\n"
        "Return JSON only.\n\n"
        f"Tweet: {tweet_text}"
    )
    schema = {
        "type": "object",
        "properties": {
            "emotion": {"type": "string", "enum": EMOTION_LABELS},
            "confidence": {"type": "number"},
        },
        "required": ["emotion", "confidence"],
    }
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0.0,
        ),
    )
    return parse_output(response.text)


def label_batch(client, types, model: str, rows: list[dict]) -> list[dict[str, str | float]]:
    tweet_lines = "\n".join(f"- tweet_id={row['tweet_id']}: {row['tweet_text']}" for row in rows)
    prompt = (
        "You label crisis-related tweets for emotion analysis.\n"
        "For each tweet, choose the dominant emotion expressed by the tweet text.\n"
        "Use exactly one label from Fear, Sadness, Anger, Hope, Neutral.\n"
        "If a tweet is mostly factual, informational, promotional, or unclear, choose Neutral.\n"
        "Return JSON only as an array with one object per tweet.\n\n"
        f"Tweets:\n{tweet_lines}"
    )
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "tweet_id": {"type": "string"},
                "emotion": {"type": "string", "enum": EMOTION_LABELS},
                "confidence": {"type": "number"},
            },
            "required": ["tweet_id", "emotion", "confidence"],
        },
    }
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0.0,
        ),
    )
    labels = parse_batch_output(response.text)
    by_id = {str(item["tweet_id"]): item for item in labels}
    ordered = []
    for row in rows:
        tweet_id = str(row["tweet_id"])
        if tweet_id not in by_id:
            raise ValueError(f"Gemini response missing tweet_id={tweet_id}")
        ordered.append(by_id[tweet_id])
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14: generate LLM emotion pseudo-labels with Gemini.")
    parser.add_argument("--prompts-jsonl", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "emotion_prompts.jsonl")
    parser.add_argument("--output-jsonl", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "emotion_llm_outputs.jsonl")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=500, help="Number of new tweets to label. Use -1 for all.")
    parser.add_argument("--events", nargs="*", default=None, help="Only label these CrisisMMD event names.")
    parser.add_argument("--batch-size", type=int, default=10, help="Tweets to label per Gemini request.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds to sleep between requests.")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("Install the Gemini SDK first: pip install google-genai") from exc

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script.")

    client = genai.Client(api_key=api_key)
    seen = completed_ids(args.output_jsonl)
    allowed_events = set(args.events) if args.events else None
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    pending: list[dict] = []

    def flush_batch(output) -> None:
        nonlocal written, pending
        if not pending:
            return
        rows = pending
        pending = []
        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                if len(rows) == 1:
                    labels = [label_one(client, types, args.model, str(rows[0]["tweet_text"]))]
                else:
                    labels = label_batch(client, types, args.model, rows)
                for row, label in zip(rows, labels):
                    payload = {
                        "tweet_id": str(row["tweet_id"]),
                        "event": row.get("event", ""),
                        "tweet_text": row.get("tweet_text", ""),
                        "emotion": label["emotion"],
                        "confidence": label["confidence"],
                        "model": args.model,
                    }
                    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    written += 1
                output.flush()
                if written % 25 == 0 or written >= rows_to_report:
                    print(f"Labeled {written} new tweets...")
                return
            except Exception as exc:  # API/network/JSON validation retry boundary.
                last_error = exc
                time.sleep(min(2**attempt, 30))
        print(f"Skipped batch of {len(rows)} tweets after {args.retries} failures: {last_error}")

    rows_to_report = args.limit if args.limit >= 0 else 0
    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for row in read_jsonl(args.prompts_jsonl):
            tweet_id = str(row["tweet_id"])
            if tweet_id in seen:
                continue
            if allowed_events is not None and str(row.get("event", "")) not in allowed_events:
                continue
            if args.limit >= 0 and written >= args.limit:
                break
            pending.append(row)
            if len(pending) >= max(args.batch_size, 1):
                flush_batch(output)
                if args.sleep > 0:
                    time.sleep(args.sleep)
        flush_batch(output)

    print(f"Done. Wrote {written} new labels to {args.output_jsonl}")


if __name__ == "__main__":
    main()

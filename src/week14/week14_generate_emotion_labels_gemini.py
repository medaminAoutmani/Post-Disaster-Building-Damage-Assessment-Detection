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


def parse_output(text: str) -> dict[str, str | float]:
    payload = json.loads(text)
    emotion = str(payload.get("emotion", "")).strip()
    if emotion not in EMOTION_LABELS:
        raise ValueError(f"Invalid emotion label: {emotion}")
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"emotion": emotion, "confidence": max(0.0, min(confidence, 1.0))}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14: generate LLM emotion pseudo-labels with Gemini.")
    parser.add_argument("--prompts-jsonl", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "emotion_prompts.jsonl")
    parser.add_argument("--output-jsonl", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "emotion_llm_outputs.jsonl")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=500, help="Number of new tweets to label. Use -1 for all.")
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
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for row in read_jsonl(args.prompts_jsonl):
            tweet_id = str(row["tweet_id"])
            if tweet_id in seen:
                continue
            if args.limit >= 0 and written >= args.limit:
                break

            last_error = None
            for attempt in range(1, args.retries + 1):
                try:
                    label = label_one(client, types, args.model, str(row["tweet_text"]))
                    payload = {
                        "tweet_id": tweet_id,
                        "event": row.get("event", ""),
                        "tweet_text": row.get("tweet_text", ""),
                        "emotion": label["emotion"],
                        "confidence": label["confidence"],
                        "model": args.model,
                    }
                    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    output.flush()
                    written += 1
                    if written % 25 == 0:
                        print(f"Labeled {written} new tweets...")
                    break
                except Exception as exc:  # API/network/JSON validation retry boundary.
                    last_error = exc
                    time.sleep(min(2**attempt, 30))
            else:
                print(f"Skipped tweet_id={tweet_id} after {args.retries} failures: {last_error}")

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Done. Wrote {written} new labels to {args.output_jsonl}")


if __name__ == "__main__":
    main()

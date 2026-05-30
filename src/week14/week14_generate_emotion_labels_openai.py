"""Generate Week 14 emotion pseudo-labels with the OpenAI API.

Input:  results/week14_crisismmd/processed/emotion_prompts.jsonl
Output: results/week14_crisismmd/processed/emotion_llm_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
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


def label_one(client, model: str, tweet_text: str) -> dict[str, str | float]:
    instructions = (
        "You label crisis-related tweets for emotion analysis. "
        "Choose the dominant emotion expressed by the tweet text. "
        "Use exactly one label from Fear, Sadness, Anger, Hope, Neutral. "
        "If the tweet is mostly factual, informational, promotional, or unclear, choose Neutral. "
        "Return only the requested JSON object."
    )
    schema = {
        "type": "json_schema",
        "name": "emotion_label",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "emotion": {"type": "string", "enum": EMOTION_LABELS},
                "confidence": {"type": "number"},
            },
            "required": ["emotion", "confidence"],
        },
    }
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=f"Tweet: {tweet_text}",
        text={"format": schema},
    )
    return parse_output(response.output_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14: generate LLM emotion pseudo-labels with OpenAI.")
    parser.add_argument("--prompts-jsonl", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "emotion_prompts.jsonl")
    parser.add_argument("--output-jsonl", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "emotion_llm_outputs.jsonl")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=500, help="Number of new tweets to label. Use -1 for all.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds to sleep between requests.")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK first: pip install openai") from exc

    client = OpenAI()
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
                    label = label_one(client, args.model, str(row["tweet_text"]))
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

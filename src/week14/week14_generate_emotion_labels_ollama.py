"""Generate Week 14 emotion pseudo-labels with a local Ollama model.

Start Ollama first:
    ollama serve

Then run:
    python src/week14/week14_generate_emotion_labels_ollama.py --model llama3:8b --limit 500
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
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


def post_ollama(host: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def label_batch(host: str, model: str, rows: list[dict], timeout: int) -> list[dict[str, str | float]]:
    tweet_lines = "\n".join(f"- tweet_id={row['tweet_id']}: {row['tweet_text']}" for row in rows)
    prompt = (
        "Label crisis-related tweets for emotion analysis.\n"
        "For each tweet, choose the dominant emotion from exactly one of these labels:\n"
        "Fear, Sadness, Anger, Hope, Neutral.\n"
        "Use Neutral for factual, informational, promotional, or unclear tweets.\n"
        "Return only valid JSON as an array. Do not include markdown.\n\n"
        "JSON schema:\n"
        '[{"tweet_id":"string","emotion":"Fear|Sadness|Anger|Hope|Neutral","confidence":0.0}]\n\n'
        f"Tweets:\n{tweet_lines}"
    )
    response = post_ollama(
        host,
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        },
        timeout,
    )
    text = str(response.get("response", "")).strip()
    payload = json.loads(text)
    if isinstance(payload, dict) and "labels" in payload:
        payload = payload["labels"]
    if isinstance(payload, dict) and {"emotion", "confidence"}.issubset(payload):
        payload["tweet_id"] = str(rows[0]["tweet_id"])
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Expected Ollama to return a JSON array.")

    by_id = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Expected object in Ollama array, got {type(item)}")
        parsed = parse_label(item)
        parsed["tweet_id"] = str(item.get("tweet_id", ""))
        by_id[str(parsed["tweet_id"])] = parsed

    ordered = []
    for row in rows:
        tweet_id = str(row["tweet_id"])
        if tweet_id not in by_id:
            raise ValueError(f"Ollama response missing tweet_id={tweet_id}")
        ordered.append(by_id[tweet_id])
    return ordered


def label_rows_with_fallback(host: str, model: str, rows: list[dict], timeout: int) -> list[dict[str, str | float]]:
    try:
        return label_batch(host, model, rows, timeout)
    except Exception:
        if len(rows) == 1:
            raise
    labels = []
    for row in rows:
        labels.extend(label_batch(host, model, [row], timeout))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14: generate local Ollama emotion pseudo-labels.")
    parser.add_argument("--prompts-jsonl", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "emotion_prompts.jsonl")
    parser.add_argument("--output-jsonl", type=Path, default=Path("results") / "week14_crisismmd" / "processed" / "emotion_llm_outputs.jsonl")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--limit", type=int, default=500, help="Number of new tweets to label. Use -1 for all.")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    seen = completed_ids(args.output_jsonl)
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
                labels = label_rows_with_fallback(args.host, args.model, rows, args.timeout)
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
                if written % 25 == 0:
                    print(f"Labeled {written} new tweets...")
                return
            except Exception as exc:
                last_error = exc
                time.sleep(2 * attempt)
        print(f"Skipped batch of {len(rows)} tweets after {args.retries} failures: {last_error}")

    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for row in read_jsonl(args.prompts_jsonl):
            if args.limit >= 0 and written >= args.limit:
                break
            if str(row["tweet_id"]) in seen:
                continue
            pending.append(row)
            if len(pending) >= max(args.batch_size, 1):
                flush_batch(output)
                if args.sleep > 0:
                    time.sleep(args.sleep)
        flush_batch(output)

    print(f"Done. Wrote {written} new labels to {args.output_jsonl}")


if __name__ == "__main__":
    main()

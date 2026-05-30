"""Convert fused event JSON files into retrieval-ready JSONL documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def top_items(counts: dict[str, int], limit: int = 3) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]


def sentence_from_counts(title: str, counts: dict[str, int]) -> str:
    if not counts:
        return f"{title}: no available counts."
    parts = [f"{name.replace('_', ' ')} ({value})" for name, value in top_items(counts)]
    return f"{title}: " + ", ".join(parts) + "."


def build_document(event: dict[str, Any]) -> dict[str, Any]:
    satellite = event["satellite_assessment"]
    topology = event["topology_validation"]
    social = event["social_media"]
    humanitarian = social.get("humanitarian", {})
    emotion = social.get("emotion", {})
    tweets = social.get("representative_posts", [])

    satellite_summary = (
        f"{satellite.get('total_damaged', 0)} damaged buildings detected: "
        f"{satellite.get('destroyed', 0)} destroyed, {satellite.get('major', 0)} major, "
        f"{satellite.get('minor', 0)} minor."
    )
    topology_summary = (
        f"Topology validation is {'positive' if topology.get('validated') else 'not confirmed'} "
        f"with {float(topology.get('confidence', 0.0)):.0%} confidence."
    )
    social_summary = (
        f"{social.get('informative_posts', 0)} informative posts. "
        f"{sentence_from_counts('Humanitarian topics', humanitarian)} "
        f"{sentence_from_counts('Emotions', emotion)}"
    )
    text = "\n".join(
        [
            f"Event: {event['event']}",
            satellite_summary,
            topology_summary,
            social_summary,
            "Representative posts:",
            *[f"- {tweet}" for tweet in tweets[:5]],
        ]
    )
    return {
        "id": event["event"],
        "text": text,
        "metadata": {
            "event": event["event"],
            "satellite_summary": satellite_summary,
            "topology_summary": topology_summary,
            "informative_posts": social.get("informative_posts", 0),
            "top_humanitarian": top_items(humanitarian),
            "top_emotion": top_items(emotion),
        },
        "payload": event,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 16: build retrieval documents from fused event JSON.")
    parser.add_argument("--event-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-jsonl", type=Path, default=Path("results") / "week16_rag" / "event_documents.jsonl")
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as file:
        for path in args.event_json:
            event = json.loads(path.read_text(encoding="utf-8"))
            file.write(json.dumps(build_document(event), ensure_ascii=True) + "\n")
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()

"""Build a unified disaster-event representation from vision and NLP signals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path | None, default: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_satellite(payload: dict[str, Any]) -> dict[str, Any]:
    destroyed = int(payload.get("destroyed", 0))
    major = int(payload.get("major", payload.get("major_damage", 0)))
    minor = int(payload.get("minor", payload.get("minor_damage", 0)))
    confidence = payload.get("confidence")
    result: dict[str, Any] = {
        "destroyed": destroyed,
        "major": major,
        "minor": minor,
        "total_damaged": destroyed + major + minor,
    }
    if confidence is not None:
        result["confidence"] = float(confidence)
    return result


def normalize_social(payload: dict[str, Any]) -> dict[str, Any]:
    humanitarian = dict(payload.get("humanitarian", {}))
    emotion = dict(payload.get("emotion", payload.get("emotions", {})))
    disaster_type = dict(payload.get("disaster_type", payload.get("disaster_types", {})))
    result: dict[str, Any] = {
        "informative_posts": int(payload.get("informative_posts", 0)),
        "humanitarian": {key: int(value) for key, value in humanitarian.items()},
        "emotion": {key: int(value) for key, value in emotion.items()},
        "disaster_type": {key: int(value) for key, value in disaster_type.items()},
        "representative_posts": list(payload.get("representative_posts", payload.get("top_tweets", []))),
    }
    if "classified_posts" in payload:
        result["classified_posts"] = int(payload.get("classified_posts", 0))
    if "included_tweets" in payload:
        result["included_tweets"] = list(payload.get("included_tweets", []))
    return result


def build_event(
    event: str,
    satellite: dict[str, Any],
    social: dict[str, Any],
    source_files: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "event": event,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "satellite_assessment": normalize_satellite(satellite),
        "social_media": normalize_social(social),
        "source_files": {key: value for key, value in source_files.items() if value},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 15: fuse satellite and social-media event signals.")
    parser.add_argument("--event", required=True, help="Stable event identifier, for example example_event.")
    parser.add_argument("--satellite-json", type=Path)
    parser.add_argument("--social-json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results") / "week15_fusion" / "event.json")
    args = parser.parse_args()

    satellite = read_json(args.satellite_json, {"destroyed": 12, "major": 35, "minor": 48, "confidence": 0.87})
    social = read_json(
        args.social_json,
        {
            "informative_posts": 834,
            "humanitarian": {"infrastructure_damage": 210, "affected_people": 155, "rescue_efforts": 94, "injured_dead": 42},
            "emotion": {"fear": 215, "hope": 71, "anger": 34, "sadness": 89},
            "disaster_type": {"earthquake": 834},
            "representative_posts": [],
        },
    )

    event = build_event(
        args.event,
        satellite,
        social,
        {
            "satellite_json": str(args.satellite_json) if args.satellite_json else None,
            "social_json": str(args.social_json) if args.social_json else None,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(event, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

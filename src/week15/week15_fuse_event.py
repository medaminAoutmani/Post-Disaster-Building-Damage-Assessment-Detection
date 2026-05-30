"""Build a unified disaster-event representation from vision and NLP signals.

Topology/TDA can be attached as an optional experimental analysis branch, but it
is excluded from the final fusion payload by default because Week 13 showed it
did not improve the preferred vision model.
"""

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


def normalize_topology(payload: dict[str, Any]) -> dict[str, Any]:
    confidence = payload.get("confidence", payload.get("topology_confidence", 0.0))
    return {
        "validated": bool(payload.get("validated", False)),
        "confidence": float(confidence),
        "role": payload.get("role", "calibration_validation_anomaly_detection"),
    }


def normalize_social(payload: dict[str, Any]) -> dict[str, Any]:
    humanitarian = dict(payload.get("humanitarian", {}))
    emotion = dict(payload.get("emotion", payload.get("emotions", {})))
    return {
        "informative_posts": int(payload.get("informative_posts", 0)),
        "humanitarian": {key: int(value) for key, value in humanitarian.items()},
        "emotion": {key: int(value) for key, value in emotion.items()},
        "representative_posts": list(payload.get("representative_posts", payload.get("top_tweets", []))),
    }


def build_event(
    event: str,
    satellite: dict[str, Any],
    topology: dict[str, Any] | None,
    social: dict[str, Any],
    source_files: dict[str, str | None],
) -> dict[str, Any]:
    fused_event = {
        "event": event,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "satellite_assessment": normalize_satellite(satellite),
        "social_media": normalize_social(social),
        "excluded_branches": {
            "topology_tda": {
                "excluded_from_final_fusion": topology is None,
                "reason": (
                    "Week 13 topology/TDA validation was retained as a negative-result analysis branch "
                    "because it did not improve the preferred ConvNeXt-Tiny gated vision model."
                ),
            }
        },
        "source_files": {key: value for key, value in source_files.items() if value},
    }
    if topology is not None:
        fused_event["topology_analysis"] = normalize_topology(topology)
        fused_event["excluded_branches"]["topology_tda"]["excluded_from_final_fusion"] = True
        fused_event["excluded_branches"]["topology_tda"]["attached_for_analysis_only"] = True
    return fused_event


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 15: fuse satellite and social-media event signals.")
    parser.add_argument("--event", required=True, help="Stable event identifier, for example example_event.")
    parser.add_argument("--satellite-json", type=Path)
    parser.add_argument("--topology-json", type=Path, help="Optional Week 13 TDA analysis JSON. Excluded from final fusion logic.")
    parser.add_argument("--social-json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results") / "week15_fusion" / "event.json")
    args = parser.parse_args()

    satellite = read_json(args.satellite_json, {"destroyed": 12, "major": 35, "minor": 48, "confidence": 0.87})
    topology = read_json(args.topology_json, {}) if args.topology_json else None
    social = read_json(
        args.social_json,
        {
            "informative_posts": 834,
            "humanitarian": {"infrastructure_damage": 210, "affected_people": 155, "rescue_efforts": 94, "injured_dead": 42},
            "emotion": {"fear": 215, "hope": 71, "anger": 34, "sadness": 89},
            "representative_posts": [],
        },
    )

    event = build_event(
        args.event,
        satellite,
        topology,
        social,
        {
            "satellite_json": str(args.satellite_json) if args.satellite_json else None,
            "topology_json": str(args.topology_json) if args.topology_json else None,
            "social_json": str(args.social_json) if args.social_json else None,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(event, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

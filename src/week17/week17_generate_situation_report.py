"""Generate a disaster situation report from a fused event or retrieval document."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_payload(path: Path, event_id: str | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No documents found in {path}")
        if event_id is not None:
            for row in rows:
                if row.get("id") == event_id or row.get("metadata", {}).get("event") == event_id:
                    return row.get("payload", row)
            raise ValueError(f"Event {event_id!r} was not found in {path}")
        payload = rows[0]
        return payload.get("payload", payload)
    payload = json.loads(text)
    return payload.get("payload", payload)


def sorted_counts(counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def confidence_label(satellite_confidence: float | None) -> str:
    score = 0.0 if satellite_confidence is None else satellite_confidence
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def count_summary(counts: list[tuple[str, int]], empty_message: str) -> str:
    if not counts:
        return empty_message
    return ", ".join(f"{name.replace('_', ' ')} ({value})" for name, value in counts[:3])


def build_prompt(event: dict[str, Any]) -> str:
    return (
        "Generate a concise disaster situation report with these sections: "
        "Physical damage assessment, Humanitarian impact, Public sentiment, "
        "Recommended priorities, Confidence assessment.\n\n"
        f"Input JSON:\n{json.dumps(event, indent=2)}"
    )


def call_ollama(prompt: str, model: str, url: str) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload.get("response", "")).strip()


def template_report(event: dict[str, Any]) -> str:
    satellite = event["satellite_assessment"]
    social = event["social_media"]
    topology = satellite.get("topology_validation", {})
    humanitarian = sorted_counts(social.get("humanitarian", {}))
    emotion = sorted_counts(social.get("emotion", {}))
    disaster_type = sorted_counts(social.get("disaster_type", {}))
    total = satellite.get("total_damaged", satellite.get("destroyed", 0) + satellite.get("major", 0) + satellite.get("minor", 0))
    label = confidence_label(satellite.get("confidence"))

    priority_candidates = []
    if satellite.get("destroyed", 0) or satellite.get("major", 0):
        priority_candidates.append("Infrastructure inspection")
    if dict(humanitarian).get("rescue_efforts", 0) or dict(humanitarian).get("affected_people", 0):
        priority_candidates.append("Search and rescue support")
    if emotion and emotion[0][0] in {"fear", "anger", "sadness"}:
        priority_candidates.append("Public communication")
    priority_candidates.append("Cross-check satellite and field reports")

    lines = [
        "DISASTER SITUATION REPORT",
        "",
        f"Event: {event['event'].replace('_', ' ').title()}",
        "",
        "1. Physical damage assessment",
        (
            f"Satellite imagery indicates {total} damaged buildings: "
            f"{satellite.get('destroyed', 0)} destroyed, {satellite.get('major', 0)} major damage, "
            f"and {satellite.get('minor', 0)} minor damage."
        ),
        (
            f"Topology validation checked {topology.get('validated_buildings', 0)} building classifications "
            f"with {float(topology.get('topology_cnn_agreement_rate', 0.0)):.0%} CNN agreement."
            if topology
            else "Topology validation was not available for this run."
        ),
        "",
        "2. Humanitarian impact",
        f"Analysis of {social.get('informative_posts', 0)} informative social-media posts indicates the likely disaster type is "
        + count_summary(disaster_type, "not available")
        + ". The strongest humanitarian topics are "
        + count_summary(humanitarian, "not available")
        + ".",
        "",
        "3. Public sentiment",
        "Dominant emotions are " + count_summary(emotion, "not available") + ".",
        "",
        "4. Recommended priorities",
        *[f"{index}. {item}" for index, item in enumerate(dict.fromkeys(priority_candidates), start=1)],
        "",
        "5. Confidence assessment",
        f"Vision-model confidence: {float(satellite.get('confidence', 0.0)):.0%}.",
        f"Overall confidence: {label}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 17: generate a disaster situation report.")
    parser.add_argument("--input-json", type=Path, required=True, help="Fused event JSON, retrieval document JSON, or Week 16 JSONL document store.")
    parser.add_argument("--event", help="Event ID to select when --input-json points to a multi-event JSONL file.")
    parser.add_argument("--output-md", type=Path, default=Path("results") / "week17_reports" / "situation_report.md")
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--ollama-model", default="llama3")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    args = parser.parse_args()

    event = load_payload(args.input_json, event_id=args.event)
    report = ""
    if args.use_ollama:
        try:
            report = call_ollama(build_prompt(event), args.ollama_model, args.ollama_url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"Ollama generation failed, using template fallback: {exc}")
    if not report:
        report = template_report(event)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report + "\n", encoding="utf-8")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()

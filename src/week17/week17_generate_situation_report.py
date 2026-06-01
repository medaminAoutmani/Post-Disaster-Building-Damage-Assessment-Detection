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
        "You are a disaster-response analyst writing for emergency coordinators. "
        "Generate a long, human-sounding disaster situation report in paragraph layout. "
        "Use clear section headings, but write the content as connected paragraphs rather than bullet lists. "
        "The report should be professional, specific, and cautious about uncertainty. Include these sections: "
        "Executive overview, Physical damage assessment, Humanitarian impact, Public sentiment, "
        "Operational priorities, and Confidence assessment. "
        "Mention the satellite/CV precision or confidence signals and the NLP/social-media confidence signals when present. "
        "Do not invent locations, casualty numbers, or field facts that are not supported by the input JSON.\n\n"
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

    topology_sentence = (
        f"Topology validation was also applied to {topology.get('validated_buildings', 0)} building classifications, "
        f"with an agreement rate of {float(topology.get('topology_cnn_agreement_rate', 0.0)):.0%} against the CNN predictions. "
        "This should be read as a structural consistency check rather than a replacement for the vision model."
        if topology
        else "Topology validation was not available for this run, so the visual assessment relies on the satellite model output and any manual counts provided in the payload."
    )
    priorities = ", ".join(dict.fromkeys(priority_candidates))
    event_name = event["event"].replace("_", " ").title()
    vision_confidence = float(satellite.get("confidence", 0.0))

    paragraphs = [
        "# Disaster Situation Report",
        f"## Executive overview\n\nThe fused assessment for {event_name} indicates a significant damage signal across the satellite-derived building inventory, with {total} buildings categorized as damaged. The current event picture combines visual damage evidence with social-media indicators, so the report should be treated as an operational summary for triage rather than a final field-verified assessment. The strongest available confidence signal is {label.lower()}, with the vision branch reporting approximately {vision_confidence:.0%} confidence where that value is present.",
        (
            "## Physical damage assessment\n\n"
            f"The satellite assessment identifies {satellite.get('destroyed', 0)} destroyed buildings, "
            f"{satellite.get('major', 0)} buildings with major damage, and {satellite.get('minor', 0)} buildings with minor damage. "
            "Destroyed and major-damage detections should receive the earliest review because they are the clearest indicators of severe structural impact and possible access constraints. "
            f"{topology_sentence}"
        ),
        (
            "## Humanitarian impact\n\n"
            f"The social-media branch analyzed {social.get('informative_posts', 0)} informative posts. "
            f"The most visible disaster-type signal is {count_summary(disaster_type, 'not available')}, while the leading humanitarian themes are {count_summary(humanitarian, 'not available')}. "
            "These topics suggest where responders may need to look for public requests, infrastructure disruption reports, and community-level needs, but they should be cross-checked with official channels before resource decisions are finalized."
        ),
        (
            "## Public sentiment\n\n"
            f"The dominant emotional signals are {count_summary(emotion, 'not available')}. "
            "A strong fear, anger, or sadness signal can indicate uncertainty, perceived delays, or unresolved safety concerns among affected residents. "
            "Public communication should therefore be factual, frequent, and careful to separate confirmed information from model-derived estimates."
        ),
        (
            "## Operational priorities\n\n"
            f"The recommended near-term priorities are {priorities}. "
            "Response teams should first verify the most severe visual detections, compare them with field reports, and then use social-media clusters to guide situational awareness around affected people, rescue needs, and infrastructure access. "
            "The fusion output is most useful as a prioritization layer: it helps decide where human review should begin, not where it should end."
        ),
        (
            "## Confidence assessment\n\n"
            f"Overall confidence is assessed as {label.lower()}. The computer-vision component reports {vision_confidence:.0%} confidence when available, and the NLP branch contributes supporting evidence through the volume and consistency of classified posts. "
            "Because both branches can be affected by missing imagery, class imbalance, duplicated posts, and incomplete ground truth, the safest interpretation is to use this report as an early warning and coordination product that remains open to revision as verified field information arrives."
        ),
    ]
    return "\n\n".join(paragraphs)


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

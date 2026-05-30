"""Export compact topology validation JSON for Week 15 fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD_PATHS = [
    Path("results") / "week13_topology" / "threshold" / "topology_threshold.json",
    Path("results") / "week13" / "week13_topology" / "threshold" / "topology_threshold.json",
]
DEFAULT_HYBRID_PATHS = [
    Path("results") / "week13_topology" / "hybrid" / "metrics" / "hybrid_metrics.json",
    Path("results") / "week13" / "week13_topology" / "hybrid" / "metrics" / "hybrid_metrics.json",
]


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 13: export topology validation signal for Week 15 fusion.")
    parser.add_argument("--threshold-json", type=Path, default=first_existing(DEFAULT_THRESHOLD_PATHS))
    parser.add_argument("--hybrid-metrics-json", type=Path, default=first_existing(DEFAULT_HYBRID_PATHS))
    parser.add_argument("--output-json", type=Path, default=Path("results") / "week15_inputs" / "topology.json")
    parser.add_argument("--min-confidence", type=float, default=0.50)
    args = parser.parse_args()

    threshold_payload = read_json(args.threshold_json)
    hybrid_payload = read_json(args.hybrid_metrics_json)
    threshold_metrics = threshold_payload.get("metrics", {})
    confidence = float(threshold_metrics.get("f1", threshold_metrics.get("precision", 0.0)))
    validated = bool(threshold_payload) and confidence >= args.min_confidence

    output = {
        "validated": validated,
        "topology_confidence": confidence,
        "role": "confidence_calibration_anomaly_detection_validation",
        "threshold_json": str(args.threshold_json),
        "hybrid_metrics_json": str(args.hybrid_metrics_json),
        "topology_threshold_metrics": threshold_metrics,
        "hybrid_summary": {
            "num_corrections": hybrid_payload.get("num_corrections"),
            "ambiguity_margin": hybrid_payload.get("ambiguity_margin"),
            "baseline_macro_f1": hybrid_payload.get("baseline_metrics", {}).get("macro_f1"),
            "hybrid_macro_f1": hybrid_payload.get("hybrid_metrics", {}).get("macro_f1"),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

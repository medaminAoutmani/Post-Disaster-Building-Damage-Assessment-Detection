"""Templates for temporal failure analysis."""

from __future__ import annotations

from pathlib import Path


def write_failure_analysis_template(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """# Week 7 Temporal Failure Analysis

## Minor vs Major Confusion

- Observation:
- Likely cause:

## False Destroyed Predictions

- Observation:
- Likely cause:

## Boundary Uncertainty

- Observation:
- Likely cause:

## Missed Collapsed Buildings

- Observation:
- Likely cause:

## Temporal Evidence

- Compare pre/post difference maps and attention maps.
""",
        encoding="utf-8",
    )


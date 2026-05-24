"""Publication figure helpers for Week 7."""

from __future__ import annotations

from pathlib import Path

from week7_analysis import summarize_week7_experiments


def prepare_paper_figure_manifest(results_root: Path = Path("results") / "week7") -> Path:
    output_path = results_root / "paper_figures" / "figure_manifest.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summarize_week7_experiments(results_root, results_root / "comparative_analysis" / "week7_summary.csv")
    output_path.write_text(
        """# Week 7 Paper Figure Manifest

- Figure 1: Temporal Siamese pipeline
- Figure 2: Fusion strategy comparison
- Figure 3: Attention map examples
- Figure 4: Prediction panels and failure cases
- Figure 5: Confusion matrix and rare-class recall
""",
        encoding="utf-8",
    )
    return output_path


"""Week 7 ablation table helpers."""

from __future__ import annotations

from pathlib import Path

from week6.week6_ablation import compare_to_baseline, write_ablation_summary


FUSION_ABLATIONS = ["concat", "difference", "concat_difference", "gated_fusion"]
ATTENTION_ABLATIONS = ["no_attention", "bottleneck_attention", "cbam", "non_local"]


def save_week7_ablation(rows: list[dict], output_dir: Path) -> None:
    write_ablation_summary(compare_to_baseline(rows, baseline_name="siamese_concat"), output_dir)


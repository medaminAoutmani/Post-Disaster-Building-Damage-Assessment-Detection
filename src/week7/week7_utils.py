"""Week 7 utilities."""

from __future__ import annotations

from pathlib import Path

from week6.week6_utils import (
    collect_environment_metadata,
    ensure_experiment_dirs,
    fix_seed,
    get_device,
    load_checkpoint,
    save_checkpoint,
    save_experiment_config,
    save_json,
    setup_file_logger,
)


WEEK7_EXPERIMENTS = [
    "experiment_siamese_concat",
    "experiment_siamese_difference",
    "experiment_siamese_concat_difference",
    "experiment_siamese_bottleneck_attention",
    "experiment_siamese_cbam",
    "experiment_siamese_nonlocal",
    "experiment_siamese_weighted_sampler",
    "experiment_siamese_tversky",
]


def create_week7_results_tree(results_root: Path = Path("results") / "week7") -> None:
    for experiment in WEEK7_EXPERIMENTS:
        ensure_experiment_dirs(results_root, experiment)
    for folder in [
        "ablation_studies/ablation_plots",
        "comparative_analysis",
        "final_model/inference_examples",
        "paper_figures",
        "attention_maps",
        "temporal_difference_maps",
        "feature_visualizations",
        "class_activation_maps",
        "failure_analysis",
    ]:
        directory = results_root / folder
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)


"""Train Week 10C soft building-weighted focal damage-loss experiments."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
WEEK9_DIR = SRC_DIR / "week9"
if str(WEEK9_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK9_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from week9_train_multitask import parse_args, train_experiment


def main() -> None:
    args = parse_args()
    if args.experiment_name is None:
        args.experiment_name = "experiment_week10c_soft_building_weighted_focal"

    args.results_root = Path("results") / "week10"
    args.experiment = "multitask_cbam_difference"
    args.fusion = "difference"
    args.attention = "cbam"
    args.damage_loss = "soft_building_weighted_focal"
    args.damage_class_weights = None
    args.sampler = "shuffle"

    experiment_dir = train_experiment(args)
    print(f"Finished {experiment_dir}")


if __name__ == "__main__":
    main()

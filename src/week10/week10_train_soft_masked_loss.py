"""Train Week 10A.1 soft building-weighted damage-loss experiments."""

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
        args.experiment_name = "experiment_week10a1_soft_building_weighted_damage_loss"
    args.results_root = Path("results") / "week10"
    args.damage_loss = "soft_building_weighted_ce_dice"
    experiment_dir = train_experiment(args)
    print(f"Finished {experiment_dir}")


if __name__ == "__main__":
    main()

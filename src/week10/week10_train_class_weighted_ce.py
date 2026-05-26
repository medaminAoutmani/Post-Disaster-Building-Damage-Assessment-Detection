"""Train Week 10B damage loss-shape experiments."""

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


TRIAL_WEIGHTS = {
    "a": [1.0, 1.0, 2.5, 1.5, 1.2],
    "b": [1.0, 1.0, 4.0, 1.5, 1.2],
    "c": [1.0, 1.0, 6.0, 2.0, 1.5],
}

LOSS_NAMES = {
    "ce_dice": "ce_dice_0_7_0_3",
    "weighted_ce": "weighted_cross_entropy",
    "weighted_ce_dice": "weighted_cross_entropy_dice",
    "focal": "focal",
}

UNWEIGHTED_LOSSES = {"ce_dice"}


def main() -> None:
    args = parse_args()
    trial = args.week10b_trial
    loss_key = args.week10b_loss
    args.results_root = Path("results") / "week10"
    args.damage_class_weights = None if loss_key in UNWEIGHTED_LOSSES else TRIAL_WEIGHTS[trial]
    args.damage_loss = LOSS_NAMES[loss_key]
    args.sampler = "shuffle"

    if args.experiment_name is None:
        suffix = "" if loss_key in UNWEIGHTED_LOSSES else f"_trial_{trial}"
        args.experiment_name = f"experiment_week10b_{loss_key}{suffix}"

    experiment_dir = train_experiment(args)
    print(f"Finished {experiment_dir}")


if __name__ == "__main__":
    main()

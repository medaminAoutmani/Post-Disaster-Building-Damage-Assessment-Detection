"""Train Week 7 Siamese non-local attention experiment."""

from __future__ import annotations

import sys

from week7_experiment_runner import main


if __name__ == "__main__":
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", "siamese_nonlocal"])
    main()


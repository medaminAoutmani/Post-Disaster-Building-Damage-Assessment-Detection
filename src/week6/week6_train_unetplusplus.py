"""Train the Week 6 UNet++ experiment."""

from __future__ import annotations

import sys

from week6_experiment_runner import main


if __name__ == "__main__":
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", "unetplusplus"])
    main()


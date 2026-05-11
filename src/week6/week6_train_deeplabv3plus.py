"""Train the Week 6 DeepLabV3 experiment.

The filename is kept for continuity with the original Week 6 plan, but the
actual torchvision architecture is DeepLabV3 rather than DeepLabV3+.
"""

from __future__ import annotations

import sys

from week6_experiment_runner import main


if __name__ == "__main__":
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", "deeplabv3"])
    main()

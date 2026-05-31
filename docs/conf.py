from __future__ import annotations

from pathlib import Path


project = "Damage Detection Project"
author = "Damage Detection Project Contributors"

extensions = ["myst_parser"]
templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

root_doc = "index"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

repo_root = Path(__file__).resolve().parents[1]

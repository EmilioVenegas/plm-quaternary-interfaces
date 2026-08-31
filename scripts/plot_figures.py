#!/usr/bin/env python3
"""Generate all publication-quality figures for the manuscript.

Dispatches to the individual modular figure generators in docs/figures/scripts/:
  - Figure 1: 01_plot_schematic.py
  - Figure 2: 02_plot_double_dissociation.py
  - Figure 3: 03_plot_mediation_forest.py
  - Figure 4: 04_plot_scaling_collapse.py
  - Figure 5: 05_plot_evolutionary_regimes.py
  - Figure 6: 06_plot_binder_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_SCRIPTS_DIR = REPO_ROOT / "docs" / "figures" / "scripts"
sys.path.insert(0, str(DOCS_SCRIPTS_DIR))

from plot_all_figures import main as plot_all_main


def main():
    plot_all_main()


if __name__ == "__main__":
    main()

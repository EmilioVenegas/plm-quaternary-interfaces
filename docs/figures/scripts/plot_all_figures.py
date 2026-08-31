#!/usr/bin/env python3
"""Run all individual figure generation scripts in docs/figures/scripts/.

Artifacts generated in docs/figures/:
  1. 01_expression_confound_schematic.png
  2. 02_double_dissociation_scatter.png
  3. 03_mediation_forest_plot.png
  4. 04_model_scaling_collapse.png
  5. 05_evolutionary_regimes.png (Minimalist Tri-Panel Trajectory Cards)
  6. 06_binder_filter_depletion.png (Refined 2-Panel Filter Trap Audit)
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import importlib


def main():
    print("=" * 78)
    print("Generating All 6 Publication Figures (docs/figures/scripts/)")
    print("=" * 78)

    modules = [
        ("Figure 1", "01_plot_schematic"),
        ("Figure 2", "02_plot_double_dissociation"),
        ("Figure 3", "03_plot_mediation_forest"),
        ("Figure 4", "04_plot_scaling_collapse"),
        ("Figure 5", "05_plot_evolutionary_regimes"),
        ("Figure 6", "06_plot_binder_filter"),
    ]

    for fig_label, mod_name in modules:
        print(f"\n[{fig_label}] Running {mod_name}.py ...")
        mod = importlib.import_module(mod_name)
        mod.main()

    print("\n" + "=" * 78)
    print("All 6 publication figures generated successfully in docs/figures/")
    print("=" * 78)


if __name__ == "__main__":
    main()

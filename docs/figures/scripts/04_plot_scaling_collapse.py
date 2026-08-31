#!/usr/bin/env python3
"""Figure 4: Model scaling trajectory showing widening interface gap from 600M to 6.35B.

Output: docs/figures/04_model_scaling_collapse.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
DOCS_FIGS_DIR = REPO_ROOT / "docs" / "figures"

# Publication styling defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8.8,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.7,
    "grid.color": "#f1f5f9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
    "figure.autolayout": False,
})

PALETTE = {
    "Abundance": "#3b6998",
    "Binding": "#9c3848",
    "Partial": "#c07f1a",
    "ESMC-600M": "#5a7d9a",
    "ESM2-650M": "#3b6998",
    "ESM2-3B": "#4f5fc4",
    "ESMC-6B": "#7c5cc7",
}


def load_scaling_data():
    with open(RESULTS_DIR / "mediation_summary.json") as f:
        mediation_data = json.load(f)

    test_data = {}
    for arm in ["esm2-650m", "esm2-3b", "esmc-600m", "esmc-6b"]:
        tpath = RESULTS_DIR / f"test_{arm}.json"
        if tpath.exists():
            with open(tpath) as f:
                test_data[arm] = json.load(f)

    return test_data, mediation_data


def plot_figure_4_scaling_collapse(test_data, mediation_data):
    """Figure 4: Tracking interface correlation collapse and widening gap across parameter scales."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300, facecolor="#ffffff")

    scale_models = [
        {"id": "esmc-600m", "name": "ESMC\n600M", "params": 0.60, "family": "ESMC"},
        {"id": "esm2-650m", "name": "ESM2\n650M", "params": 0.65, "family": "ESM2"},
        {"id": "esm2-3b",   "name": "ESM2\n2.84B", "params": 2.84, "family": "ESM2"},
        {"id": "esmc-6b",   "name": "ESMC\n6.35B", "params": 6.35, "family": "ESMC"},
    ]

    names = [m["name"] for m in scale_models]
    rho_abund = []
    rho_bind = []
    drop_pct = []
    rho_partial = []

    for m in scale_models:
        t = test_data[m["id"]]["subgroup_correlations"]
        ra = t["Interface_Abundance"]["spearman_rho"]
        rb = t["Interface_Binding"]["spearman_rho"]
        rho_abund.append(ra)
        rho_bind.append(rb)
        drop_pct.append((rb - ra) / ra * 100)

        rp = mediation_data[m["id"]]["compartments"]["Interface"]["rho_partial_plm_binding_given_abundance"]
        rho_partial.append(rp)

    x = np.arange(len(scale_models))

    # Panel A: Absolute Correlations
    ax1.text(-0.12, 1.05, "a", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="top")
    ax1.plot(x, rho_abund, marker="o", lw=1.8, markersize=6.5, color=PALETTE["Abundance"], label="Interface Monomer Abundance ($\\rho$)")
    ax1.plot(x, rho_bind, marker="s", lw=1.8, markersize=6.5, color=PALETTE["Binding"], label="Interface Complex Binding ($\\rho$)")
    ax1.plot(x, rho_partial, marker="^", lw=1.6, linestyle="--", markersize=6.5, color=PALETTE["Partial"], label="Partial Interface $\\rho(\\mathrm{PLM}, \\mathrm{Bind} \\mid \\mathrm{Abund})$")

    for i in range(len(x)):
        ax1.text(x[i], rho_abund[i] + 0.025, f"{rho_abund[i]:.3f}", ha="center", fontsize=8.0, fontweight="bold", color=PALETTE["Abundance"])
        ax1.text(x[i], rho_bind[i] - 0.035, f"{rho_bind[i]:.3f}", ha="center", fontsize=8.0, fontweight="bold", color=PALETTE["Binding"])
        ax1.text(x[i], rho_partial[i] - 0.035, f"{rho_partial[i]:.3f}", ha="center", fontsize=8.0, fontweight="bold", color=PALETTE["Partial"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9.0)
    ax1.set_ylabel("Spearman Rank Correlation ($\\rho$)", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(-0.12, 0.48)
    ax1.axhline(0, color="#64748b", linestyle=":", lw=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True)
    ax1.legend(loc="lower left", fontsize=8.0, framealpha=0.95, edgecolor="#cbd5e1")

    # Panel B: Drop Percentage / Widening Defect
    ax2.text(-0.12, 1.05, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")
    model_bar_colors = [PALETTE["ESMC-600M"], PALETTE["ESM2-650M"], PALETTE["ESM2-3B"], PALETTE["ESMC-6B"]]
    bars = ax2.bar(x, drop_pct, color=model_bar_colors, width=0.48, edgecolor="#334155", lw=0.8)
    for bar, dp in zip(bars, drop_pct):
        y_val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, y_val / 2, f"{dp:.1f}%", ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color="#ffffff")

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.0)
    ax2.set_ylabel("Relative Interface Collapse ($\\Delta\\rho / \\rho_{\\mathrm{abund}}$ %)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(-95, 0)
    ax2.axhline(0, color="#475569", linestyle="-", lw=0.8, zorder=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, axis="y")
    plt.tight_layout()
    out_path = DOCS_FIGS_DIR / "04_model_scaling_collapse.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    print("=" * 78)
    print("Generating Figure 4: Model Scaling Trajectory")
    print("=" * 78)
    test_data, mediation_data = load_scaling_data()
    plot_figure_4_scaling_collapse(test_data, mediation_data)
    print("=" * 78)


if __name__ == "__main__":
    main()

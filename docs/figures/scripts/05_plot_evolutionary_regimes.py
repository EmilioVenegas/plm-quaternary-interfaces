#!/usr/bin/env python3
"""Figure 5: Evolutionary Stratification Regimes (Minimalist Tri-Panel Trajectory Architecture).

Output: docs/figures/05_evolutionary_regimes.png
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
    "grid.color": "#f8fafc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
    "figure.autolayout": False,
})

PALETTE = {
    # Assay Readout Types
    "Abundance": "#3b6998",  # Muted steel/slate blue
    "Binding": "#9c3848",    # Muted rose/wine red
    "Partial": "#c07f1a",    # Warm antique ochre/amber

    # Evolutionary PPI Regimes
    "Homooligomer": "#7c5cc7",
    "Natural_Heterodimer": "#2d6a4f",
    "Synthetic_CrossSpecies": "#c07f1a",
}


def load_evo_data():
    with open(RESULTS_DIR / "evolutionary_stratification.json") as f:
        return json.load(f)


def plot_figure_5_evolutionary_regimes(evo_data):
    """Figure 5: Minimalist tri-panel trajectory cards comparing evolutionary regimes."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.0), dpi=300, facecolor="#ffffff", sharey=True)
    plt.subplots_adjust(wspace=0.18, left=0.08, right=0.96, top=0.82, bottom=0.22)

    regimes = [
        ("Homooligomer", "Class 1: Homooligomer", "p53 tetramer (1OLG, N = 513)",
         "Destructive Self-Interference", "#7c5cc7"),
        ("Natural_Heterodimer", "Class 2: Natural Heterodimer", "HLA-A*02:01, GB1 (N = 1,011)",
         "Monomer Folding Confound", "#2d6a4f"),
        ("Synthetic_CrossSpecies", "Class 3: Synthetic / De Novo", "KRAS-DARPin, RBD-ACE2 (N = 738)",
         "Evolutionary Uncoupling", "#c07f1a"),
    ]

    x_pos = [0, 1, 2]
    x_labels = ["Monomer\nAbundance", "Marginal\nBinding", "Partial\nBinding"]

    for idx, (reg_id, title, subtitle, badge_text, badge_col) in enumerate(regimes):
        ax = axes[idx]
        ax.set_facecolor("#ffffff")

        # Clean zero reference line
        ax.axhline(0, color="#94a3b8", linestyle="-", lw=0.8, alpha=0.8, zorder=1)

        # Multi-model ensemble background traces (ESM2-650M, ESM2-3B, ESMC-600M)
        for m in ["esm2-650m", "esm2-3b", "esmc-600m"]:
            m_c = evo_data["arms"][m]["classes"][reg_id]["compartments"]["Interface"]
            y_m = [m_c["rho_plm_abundance"], m_c["rho_plm_binding"], m_c["rho_partial_plm_binding_given_abundance"]]
            ax.plot(x_pos, y_m, color="#cbd5e1", lw=1.0, linestyle=":", alpha=0.7, zorder=2)
            ax.scatter(x_pos, y_m, color="#cbd5e1", s=18, alpha=0.6, zorder=2)

        # Flagship model (ESMC-6B)
        c_6b = evo_data["arms"]["esmc-6b"]["classes"][reg_id]["compartments"]["Interface"]
        y_6b = [c_6b["rho_plm_abundance"], c_6b["rho_plm_binding"], c_6b["rho_partial_plm_binding_given_abundance"]]

        # Clean trajectory segments
        ax.plot([0, 1], [y_6b[0], y_6b[1]], color="#334155", lw=1.8, linestyle="-", zorder=3)
        ax.plot([1, 2], [y_6b[1], y_6b[2]], color="#64748b", lw=1.8, linestyle="--", zorder=3)

        # Markers
        ax.scatter(x_pos[0], y_6b[0], color=PALETTE["Abundance"], edgecolors="#1e293b", s=75, marker="o", lw=1.1, zorder=5)
        ax.scatter(x_pos[1], y_6b[1], facecolors="white", edgecolors=PALETTE["Binding"], s=75, marker="o", lw=1.8, zorder=5)
        ax.scatter(x_pos[2], y_6b[2], color=PALETTE["Partial"], edgecolors="#1e293b", s=65, marker="s", lw=1.1, zorder=5)

        # Value text labels (clean floating text, no heavy bounding box)
        for xp, yp, col in zip(x_pos, y_6b, [PALETTE["Abundance"], PALETTE["Binding"], PALETTE["Partial"]]):
            v_off = 0.055 if yp >= 0 else -0.075
            va = "bottom" if yp >= 0 else "top"
            ax.text(xp, yp + v_off, f"{yp:+.2f}", ha="center", va=va,
                    fontsize=8.2, fontweight="bold", color=col)

        # Header Typography
        ax.text(0.5, 1.22, title, transform=ax.transAxes, ha="center", fontsize=9.2, fontweight="bold", color="#0f172a")
        ax.text(0.5, 1.13, subtitle, transform=ax.transAxes, ha="center", fontsize=7.6, color="#64748b")
        ax.text(0.5, 1.04, badge_text, transform=ax.transAxes, ha="center", fontsize=7.2, fontweight="bold", color=badge_col)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8.2, fontweight="bold", color="#1e293b")
        ax.tick_params(axis="x", pad=5)
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(-0.70, 0.70)

        if idx == 0:
            ax.set_ylabel("Spearman Rank Correlation ($\\rho$)", fontsize=9.0, fontweight="bold", color="#1e293b")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False if idx > 0 else True)
        ax.grid(True, axis="y", color="#f8fafc", linestyle="--", lw=0.8)

    # Clean, separated borderless legend at bottom
    leg_ax = fig.add_axes([0.05, 0.02, 0.90, 0.05], facecolor="#ffffff")
    leg_ax.axis("off")
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="white", markerfacecolor=PALETTE["Abundance"], markeredgecolor="#1e293b", markeredgewidth=1.1, markersize=6.5, label="Monomer Abundance $\\rho(\\mathrm{Abund})$"),
        plt.Line2D([0], [0], marker="o", color="white", markerfacecolor="white", markeredgecolor=PALETTE["Binding"], markeredgewidth=1.8, markersize=6.5, label="Marginal Binding $\\rho(\\mathrm{Bind})$"),
        plt.Line2D([0], [0], marker="s", color="white", markerfacecolor=PALETTE["Partial"], markeredgecolor="#1e293b", markeredgewidth=1.1, markersize=6.5, label="Partial Binding $\\rho(\\mathrm{Bind} \\mid \\mathrm{Abund})$"),
        plt.Line2D([0], [0], color="#cbd5e1", linestyle=":", lw=1.2, marker="o", markersize=3.5, label="600M\u20133B Ensemble"),
    ]
    leg_ax.legend(handles=legend_elements, loc="center", frameon=False, fontsize=8.0, ncol=4,
                  columnspacing=1.8, handletextpad=0.6)

    out_path = DOCS_FIGS_DIR / "05_evolutionary_regimes.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    print("=" * 78)
    print("Generating Figure 5: Evolutionary Stratification Regimes")
    print("=" * 78)
    evo_data = load_evo_data()
    plot_figure_5_evolutionary_regimes(evo_data)
    print("=" * 78)


if __name__ == "__main__":
    main()

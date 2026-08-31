#!/usr/bin/env python3
"""Figure 2: Multi-panel scatter plot comparing PLM vs DMS Abundance & Binding across compartments.

Output: docs/figures/02_double_dissociation_scatter.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
DOCS_FIGS_DIR = REPO_ROOT / "docs" / "figures"
DOCS_FIGS_DIR.mkdir(parents=True, exist_ok=True)

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
    "Core": "#475569",       # Deep slate grey
    "Surface": "#2d6a4f",    # Muted forest/sage green
    "Interface": "#a83232",  # Subdued brick/crimson red
}


def load_scores():
    scores_path = RESULTS_DIR / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found at {scores_path}")
    return pd.read_csv(scores_path)


def plot_figure_2_double_dissociation(df_scores):
    """Figure 2: Multi-panel scatter plot comparing PLM vs DMS Abundance & Binding across compartments."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), dpi=300, sharex=True, sharey=True, facecolor="#ffffff")

    compartments = ["Core", "Surface", "Interface"]
    comp_colors = {"Core": PALETTE["Core"], "Surface": PALETTE["Surface"], "Interface": PALETTE["Interface"]}
    comp_ns = {"Core": "N = 4,405", "Surface": "N = 3,976", "Interface": "N = 2,262"}

    df = df_scores.copy()
    for sys_id in df["system"].unique():
        idx = df["system"] == sys_id
        df.loc[idx, "dms_abund_z"] = stats.zscore(df.loc[idx, "dms_score_abundance"].dropna())
        df.loc[idx, "dms_bind_z"] = stats.zscore(df.loc[idx, "dms_score_binding"].dropna())
        df.loc[idx, "plm_z"] = stats.zscore(df.loc[idx, "zeroshot_esmc-6b"].dropna())

    panel_labels = [["a", "b", "c"], ["d", "e", "f"]]

    # Top row: PLM vs Abundance (Monomer Stability)
    for col_idx, comp in enumerate(compartments):
        ax = axes[0, col_idx]
        sub = df[df["compartment"] == comp].dropna(subset=["plm_z", "dms_abund_z"])
        rho, _ = stats.spearmanr(sub["plm_z"], sub["dms_abund_z"])

        ax.scatter(sub["plm_z"], sub["dms_abund_z"], color=comp_colors[comp], alpha=0.18, s=10, edgecolors="none")
        m, b = np.polyfit(sub["plm_z"], sub["dms_abund_z"], 1)
        x_line = np.linspace(-3.5, 3.5, 100)
        ax.plot(x_line, m * x_line + b, color="#334155", lw=1.5, linestyle="-")

        ax.text(-0.10, 1.05, panel_labels[0][col_idx], transform=ax.transAxes, fontsize=11.5, fontweight="bold", va="top")
        ax.set_title(f"{comp} ({comp_ns[comp]})\nMonomer Abundance", fontsize=9.5, fontweight="bold", color="#1e293b", pad=5)
        ax.text(0.05, 0.90, f"Spearman $\\rho = {rho:+.3f}$", transform=ax.transAxes,
                fontsize=9.0, fontweight="bold", color="#1e293b",
                bbox=dict(boxstyle="round,pad=0.22", fc="#f8fafc", ec="#cbd5e1", lw=0.7, alpha=0.95))
        ax.set_ylabel("Monomer Abundance ($z$-score)" if col_idx == 0 else "", fontsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True)

    # Bottom row: PLM vs Binding Affinity
    for col_idx, comp in enumerate(compartments):
        ax = axes[1, col_idx]
        sub = df[df["compartment"] == comp].dropna(subset=["plm_z", "dms_bind_z"])
        rho, _ = stats.spearmanr(sub["plm_z"], sub["dms_bind_z"])

        ax.scatter(sub["plm_z"], sub["dms_bind_z"], color=comp_colors[comp], alpha=0.18, s=10, edgecolors="none")
        m, b = np.polyfit(sub["plm_z"], sub["dms_bind_z"], 1)
        x_line = np.linspace(-3.5, 3.5, 100)
        ax.plot(x_line, m * x_line + b, color="#a83232" if comp == "Interface" else "#334155",
                lw=1.5, linestyle="--" if comp == "Interface" else "-")

        ax.text(-0.10, 1.05, panel_labels[1][col_idx], transform=ax.transAxes, fontsize=11.5, fontweight="bold", va="top")
        ax.set_title(f"{comp} ({comp_ns[comp]})\nComplex Binding", fontsize=9.5, fontweight="bold",
                     color="#a83232" if comp == "Interface" else "#1e293b", pad=5)
        ax.text(0.05, 0.90, f"Spearman $\\rho = {rho:+.3f}$", transform=ax.transAxes,
                fontsize=9.0, fontweight="bold", color="#a83232" if comp == "Interface" else "#1e293b",
                bbox=dict(boxstyle="round,pad=0.22", fc="#fef2f2" if comp == "Interface" else "#f8fafc",
                          ec="#fca5a5" if comp == "Interface" else "#cbd5e1", lw=0.7, alpha=0.95))
        ax.set_xlabel("ESMC-6B Zero-Shot Score ($z$-score)", fontsize=9.5)
        ax.set_ylabel("Complex Binding ($z$-score)" if col_idx == 0 else "", fontsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True)

    axes[0, 0].set_xlim(-3.8, 3.8)
    axes[0, 0].set_ylim(-3.8, 3.8)
    plt.tight_layout()
    out_path = DOCS_FIGS_DIR / "02_double_dissociation_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    print("=" * 78)
    print("Generating Figure 2: Double-Dissociation Scatter Plots")
    print("=" * 78)
    df_scores = load_scores()
    plot_figure_2_double_dissociation(df_scores)
    print("=" * 78)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figure 6: Binder Filter Trap & False-Negative Depletion (Refined 2-Panel Audit).

Output: docs/figures/06_binder_filter_depletion.png
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
    # Model Architectures
    "ESMC-600M": "#5a7d9a",  # Soft slate cyan
    "ESM2-650M": "#3b6998",  # Muted steel blue
    "ESM2-3B": "#4f5fc4",    # Subdued indigo
    "ESMC-6B": "#7c5cc7",    # Muted deep violet
}


def load_binder_data():
    with open(RESULTS_DIR / "binder_filter_audit.json") as f:
        return json.load(f)


def plot_figure_6_binder_filter(binder_data):
    """Figure 6: Refined 2-panel architecture with shaded operational band and discrete markers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=300, facecolor="#ffffff")
    plt.subplots_adjust(wspace=0.24, left=0.08, right=0.96, top=0.88, bottom=0.14)

    models = ["esm2-650m", "esm2-3b", "esmc-600m", "esmc-6b"]
    model_labels = {
        "esm2-650m": "ESM2-650M",
        "esm2-3b": "ESM2-3B",
        "esmc-600m": "ESMC-600M",
        "esmc-6b": "ESMC-6B",
    }
    model_colors = {
        "esm2-650m": PALETTE["ESM2-650M"],
        "esm2-3b": PALETTE["ESM2-3B"],
        "esmc-600m": PALETTE["ESMC-600M"],
        "esmc-6b": PALETTE["ESMC-6B"],
    }

    thresholds = [10, 20, 30, 50]

    # =========================================================================
    # Panel A: Interface False-Negative Rate (FNR)
    # =========================================================================
    ax1.text(-0.12, 1.06, "a", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="top")
    ax1.set_title("Interface False-Negative Rate (FNR)", fontsize=9.5, fontweight="bold", color="#0f172a", pad=6)

    # Shaded Operational Filter Band (Top 10% - 20%)
    ax1.axvspan(10, 20, color="#fef2f2", alpha=0.55, zorder=0)

    for model in models:
        sim = binder_data["filter_simulations"][model]["thresholds_simulation"]
        fnr = [s["interface_false_negative_rate"] * 100 for s in sim]
        ax1.plot(thresholds, fnr, marker="o", lw=1.8, markersize=5.5, label=model_labels[model], color=model_colors[model], zorder=3)
        ax1.scatter(20, fnr[1], s=48, color=model_colors[model], edgecolors="#1e293b", lw=1.1, zorder=5)

    ax1.axvline(20, color="#a83232", linestyle="--", lw=1.1, alpha=0.85, zorder=2)

    ax1.text(21.5, 52, "Standard Filter: Top 20%\n(Discards ~73% \u2013 77%\nof Interface Hits)",
             color="#8a1f2c", fontsize=7.6, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="#fbeef0", ec="#f2c9ce", lw=0.7), zorder=6)

    ax1.text(15, 91.5, "Standard Filter Range\n(Top 10%\u201320%)",
             color="#991b1b", fontsize=6.8, fontweight="bold", ha="center",
             bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec="#fca5a5", lw=0.5, alpha=0.9), zorder=6)

    ax1.set_xlabel("PLM Likelihood Filter Cutoff (Top X%)", fontsize=9.0, fontweight="bold", color="#1e293b")
    ax1.set_ylabel("Interface False-Negative Rate (%)", fontsize=9.0, fontweight="bold", color="#1e293b")
    ax1.set_ylim(42, 96)
    ax1.set_xticks(thresholds)
    ax1.set_xticklabels([f"Top {t}%" for t in thresholds], fontsize=8.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, color="#f1f5f9", linestyle="--", lw=0.6)
    ax1.legend(loc="lower left", fontsize=7.8, edgecolor="#cbd5e1", framealpha=0.95)

    # =========================================================================
    # Panel B: Interface Depletion Rate vs Non-Interface
    # =========================================================================
    ax2.text(-0.12, 1.06, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")
    ax2.set_title("Interface Depletion Rate vs. Non-Interface", fontsize=9.5, fontweight="bold", color="#0f172a", pad=6)

    # Shaded Operational Filter Band (Top 10% - 20%)
    ax2.axvspan(10, 20, color="#fef2f2", alpha=0.55, zorder=0)

    for model in models:
        sim = binder_data["filter_simulations"][model]["thresholds_simulation"]
        dep = [s["interface_depletion_rate"] * 100 for s in sim]
        ax2.plot(thresholds, dep, marker="s", lw=1.8, markersize=5.5, label=model_labels[model], color=model_colors[model], zorder=3)
        ax2.scatter(20, dep[1], s=48, color=model_colors[model], edgecolors="#1e293b", lw=1.1, zorder=5)

    ax2.axvline(20, color="#a83232", linestyle="--", lw=1.1, alpha=0.85, zorder=2)
    ax2.axhline(0, color="#64748b", linestyle=":", lw=0.8, zorder=1)

    ax2.set_xlabel("PLM Likelihood Filter Cutoff (Top X%)", fontsize=9.0, fontweight="bold", color="#1e293b")
    ax2.set_ylabel("Interface Depletion vs. Non-Interface (%)", fontsize=9.0, fontweight="bold", color="#1e293b")
    ax2.set_xticks(thresholds)
    ax2.set_xticklabels([f"Top {t}%" for t in thresholds], fontsize=8.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, color="#f1f5f9", linestyle="--", lw=0.6)
    ax2.legend(loc="upper right", fontsize=7.8, edgecolor="#cbd5e1", framealpha=0.95)

    out_path = DOCS_FIGS_DIR / "06_binder_filter_depletion.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    print("=" * 78)
    print("Generating Figure 6: Binder Filter Trap Audit")
    print("=" * 78)
    binder_data = load_binder_data()
    plot_figure_6_binder_filter(binder_data)
    print("=" * 78)


if __name__ == "__main__":
    main()

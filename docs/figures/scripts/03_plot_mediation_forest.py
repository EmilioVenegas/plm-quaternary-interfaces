#!/usr/bin/env python3
"""Figure 3: Forest plot of marginal vs partial rank correlations across compartments & architectures.

Output: docs/figures/03_mediation_forest_plot.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def load_mediation_data():
    with open(RESULTS_DIR / "mediation_summary.json") as f:
        return json.load(f)


def plot_figure_3_mediation_forest(mediation_data):
    """Figure 3: Forest plot of marginal vs partial rank correlations with aligned data table."""
    models = ["esmc-6b", "esmc-600m", "esm2-3b", "esm2-650m"]
    model_meta = {
        "esmc-6b": ("ESMC-6B", "6.35B"),
        "esmc-600m": ("ESMC-600M", "600M"),
        "esm2-3b": ("ESM2-3B", "2.84B"),
        "esm2-650m": ("ESM2-650M", "650M"),
    }
    compartments = ["Core", "Surface", "Interface"]

    fig = plt.figure(figsize=(11.5, 5.8), dpi=300, facecolor="#ffffff")
    ax_left = fig.add_axes([0.04, 0.16, 0.22, 0.75], facecolor="#ffffff")
    ax_plot = fig.add_axes([0.27, 0.16, 0.38, 0.75], facecolor="#ffffff")
    ax_table = fig.add_axes([0.67, 0.16, 0.30, 0.75], facecolor="#ffffff")

    ax_left.axis("off")
    ax_table.axis("off")

    row_data = []
    current_y = 0.0

    for mod_idx, model in enumerate(reversed(models)):
        m_data = mediation_data[model]["compartments"]

        y_bottom = current_y - 0.38
        y_top = current_y + 2.38
        band_color = "#f8fafc" if mod_idx % 2 == 0 else "#ffffff"

        ax_left.add_patch(patches.Rectangle((0, y_bottom), 1.0, y_top - y_bottom,
                                            facecolor=band_color, edgecolor="none", zorder=0))
        ax_plot.add_patch(patches.Rectangle((-0.15, y_bottom), 0.65, y_top - y_bottom,
                                            facecolor=band_color, edgecolor="none", zorder=0))
        ax_table.add_patch(patches.Rectangle((0, y_bottom), 1.0, y_top - y_bottom,
                                             facecolor=band_color, edgecolor="none", zorder=0))

        for comp in reversed(compartments):
            c_info = m_data[comp]
            m_rho = c_info["rho_plm_binding"]
            p_rho = c_info["rho_partial_plm_binding_given_abundance"]
            d_rho = p_rho - m_rho

            row_data.append((model, comp, c_info["n"], m_rho, p_rho, d_rho, current_y))
            current_y += 1.0
        current_y += 0.5

    total_h = current_y - 0.2
    y_lim = (-0.5, total_h)

    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(y_lim)

    ax_plot.set_xlim(-0.15, 0.48)
    ax_plot.set_ylim(y_lim)

    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(y_lim)

    # Headers
    header_y = total_h - 0.05
    ax_left.text(0.05, header_y, "Model Family", fontsize=9.0, fontweight="bold", color="#334155")
    ax_left.text(0.60, header_y, "Compartment", fontsize=9.0, fontweight="bold", color="#334155")
    ax_left.plot([0.02, 0.98], [header_y - 0.35, header_y - 0.35], color="#cbd5e1", lw=0.8)

    col_x = [0.12, 0.38, 0.65, 0.90]
    ax_table.text(col_x[0], header_y, "Sample N", fontsize=9.0, fontweight="bold", color="#334155", ha="center")
    ax_table.text(col_x[1], header_y, "Marginal $\\rho$", fontsize=9.0, fontweight="bold", color="#334155", ha="center")
    ax_table.text(col_x[2], header_y, "Partial $\\rho$", fontsize=9.0, fontweight="bold", color="#334155", ha="center")
    ax_table.text(col_x[3], header_y, "$\\Delta\\rho$", fontsize=9.0, fontweight="bold", color="#334155", ha="center")
    ax_table.plot([0.02, 0.98], [header_y - 0.35, header_y - 0.35], color="#cbd5e1", lw=0.8)

    ax_plot.axvline(0, color="#64748b", linestyle="--", lw=0.9, alpha=0.85, zorder=1)

    # Render Rows
    for model, comp, n, m_rho, p_rho, d_rho, y in row_data:
        color = PALETTE[comp]
        is_int = (comp == "Interface")

        ax_left.text(0.60, y, comp, fontsize=8.5, color=color if is_int else "#334155",
                     fontweight="bold" if is_int else "normal", va="center")

        ax_plot.plot([m_rho, p_rho], [y, y], color="#94a3b8", lw=1.3, zorder=2)
        ax_plot.scatter(m_rho, y, color="white", edgecolors=color, s=48, lw=1.6, zorder=3)
        ax_plot.scatter(p_rho, y, color=color, marker="s", s=42, zorder=4)

        weight = "bold" if is_int else "normal"
        t_color = "#a83232" if is_int else "#334155"
        ax_table.text(col_x[0], y, f"{n:,}", fontsize=8.2, color="#64748b", ha="center", va="center")
        ax_table.text(col_x[1], y, f"{m_rho:+.3f}", fontsize=8.2, color="#334155", ha="center", va="center")
        ax_table.text(col_x[2], y, f"{p_rho:+.3f}", fontsize=8.2, fontweight=weight, color=t_color, ha="center", va="center")
        ax_table.text(col_x[3], y, f"{d_rho:+.2f}", fontsize=8.2, fontweight=weight, color=t_color, ha="center", va="center")

    for model in models:
        mod_rows = [r for r in row_data if r[0] == model]
        mid_y = (mod_rows[0][6] + mod_rows[-1][6]) / 2.0
        name, params = model_meta[model]
        ax_left.text(0.05, mid_y, f"{name}\n({params})", fontsize=8.5, fontweight="bold",
                     color="#0f172a", va="center")

    ax_plot.set_xlabel("Spearman Rank Correlation ($\\rho$)", fontsize=9.5, fontweight="bold", color="#1e293b")
    ax_plot.set_yticks([])
    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    ax_plot.spines["left"].set_color("#cbd5e1")
    ax_plot.grid(True, axis="x", color="#f1f5f9", linestyle="--", lw=0.6)

    # Legend
    leg_ax = fig.add_axes([0.15, 0.02, 0.70, 0.08], facecolor="#ffffff")
    leg_ax.axis("off")

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="white", markeredgecolor="#475569", markeredgewidth=1.5, markersize=6, label="Marginal: $\\rho(\\mathrm{PLM}, \\mathrm{Binding})$"),
        plt.Line2D([0], [0], marker="s", color="#475569", markersize=6, label="Partial: $\\rho(\\mathrm{PLM}, \\mathrm{Binding} \\mid \\mathrm{Abundance})$"),
        patches.Patch(color=PALETTE["Core"], label="Core"),
        patches.Patch(color=PALETTE["Surface"], label="Surface"),
        patches.Patch(color=PALETTE["Interface"], label="Quaternary Interface"),
    ]
    leg_ax.legend(handles=legend_elements, loc="center", framealpha=0.95, facecolor="white",
                  edgecolor="#e2e8f0", fontsize=8.2, ncol=5)

    out_path = DOCS_FIGS_DIR / "03_mediation_forest_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    print("=" * 78)
    print("Generating Figure 3: Mediation Analysis Forest Plot")
    print("=" * 78)
    mediation_data = load_mediation_data()
    plot_figure_3_mediation_forest(mediation_data)
    print("=" * 78)


if __name__ == "__main__":
    main()

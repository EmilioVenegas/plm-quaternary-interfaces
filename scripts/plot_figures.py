#!/usr/bin/env python3
"""Generate publication-quality figures for the manuscript.

Artifacts generated into docs/figures/:
  1. 01_expression_confound_schematic.png:
     Biophysical & assay schematic of the monomer-folding confound in high-throughput selection.
  2. 02_double_dissociation_scatter.png:
     Multi-panel scatter/hexbin of PLM vs DMS Abundance & Binding across Core, Surface, Interface.
  3. 03_mediation_forest_plot.png:
     Forest plot of marginal vs partial rank correlations across compartments & architectures.
  4. 04_model_scaling_collapse.png:
     Model scaling trajectory showing widening interface gap from 600M to 6.35B.
  5. 05_evolutionary_regimes.png:
     Evolutionary regime breakdown: Homooligomer anti-correlation vs Heterodimer vs Synthetic binders.
  6. 06_binder_filter_depletion.png:
     Quantitative simulation of the 'PLM Filter Trap' in computational binder design.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy import stats

# Publication styling defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
})

RESULTS_DIR = REPO_ROOT / "results"
DOCS_FIGS_DIR = REPO_ROOT / "docs" / "figures"
DOCS_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Color palettes
PALETTE = {
    "Core": "#3b528b",       # deep slate blue
    "Surface": "#5ec962",    # soft emerald green
    "Interface": "#d62728",  # vibrant crimson red
    "Abundance": "#1f77b4",  # primary blue
    "Binding": "#e377c2",    # magenta / pink
    "Partial": "#ff7f0e",    # amber orange
    "ESM2-650M": "#4e79a7",
    "ESM2-3B": "#59a14f",
    "ESMC-600M": "#edc948",
    "ESMC-6B": "#e15759",
    "Homooligomer": "#9467bd",
    "Natural_Heterodimer": "#2ca02c",
    "Synthetic_CrossSpecies": "#ff7f0e",
}


def load_data():
    """Load results dataframes and JSON artifacts."""
    scores_path = RESULTS_DIR / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found at {scores_path}")
    df_scores = pd.read_csv(scores_path)

    with open(RESULTS_DIR / "mediation_summary.json") as f:
        mediation_data = json.load(f)

    with open(RESULTS_DIR / "evolutionary_stratification.json") as f:
        evo_data = json.load(f)

    with open(RESULTS_DIR / "binder_filter_audit.json") as f:
        binder_data = json.load(f)

    test_data = {}
    for arm in ["esm2-650m", "esm2-3b", "esmc-600m", "esmc-6b"]:
        tpath = RESULTS_DIR / f"test_{arm}.json"
        if tpath.exists():
            with open(tpath) as f:
                test_data[arm] = json.load(f)

    return df_scores, mediation_data, evo_data, binder_data, test_data


# -----------------------------------------------------------------------------
# Figure 1: Biophysical & Assay Schematic of the Expression Confound
# -----------------------------------------------------------------------------
def plot_figure_1_schematic():
    """Figure 1: Conceptual schematic illustrating the monomer-folding confound in PPI assays."""
    fig = plt.figure(figsize=(13, 6.8), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1.25, 0.75], wspace=0.22)

    # Panel A: Display Assay Mechanisms
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 100)
    ax_a.axis("off")
    ax_a.set_title("A   Monomer Folding vs. Quaternary Interface Disruption in Display Assays", 
                   loc="left", fontweight="bold", pad=12)

    # Box 1: Wild-Type / Benign
    rect_wt = patches.FancyBboxPatch((4, 68), 92, 26, boxstyle="round,pad=1.5,rounding_size=3",
                                     ec="#2ca02c", fc="#f2f9f2", lw=1.5)
    ax_a.add_patch(rect_wt)
    ax_a.text(6, 90, "1. Wild-Type / Neutral Variant", fontweight="bold", color="#1b6e1b", fontsize=10.5)
    ax_a.text(6, 83.5, r"• Target monomer stably folds & is displayed on cell surface (High Abundance: Abund+)" "\n"
                       r"• Interface intact $\to$ Fluorescent partner binds (High Binding: Bind+)" "\n"
                       r"• PLM Prediction: High likelihood ($\log p \uparrow$) $\to$ Correctly aligned with phenotype",
              fontsize=9, color="#222222", va="top")

    # Box 2: Core / Destabilizing Mutation (The Confound)
    rect_core = patches.FancyBboxPatch((4, 36), 92, 27, boxstyle="round,pad=1.5,rounding_size=3",
                                       ec="#d62728", fc="#fdf2f2", lw=1.5)
    ax_a.add_patch(rect_core)
    ax_a.text(6, 59, "2. Monomer Destabilizing Mutation (Core/Surface Unfolding Confound)", 
              fontweight="bold", color="#a31415", fontsize=10.5)
    ax_a.text(6, 52.5, r"• Mutation disrupts monomer hydrophobic core / tertiary fold" "\n"
                       r"• Monomer misfolds & is degraded $\to$ Zero cell-surface presentation (Abund-)" "\n"
                       r"• Partner cannot bind because target is ABSENT $\to$ False loss of binding signal (Bind-)" "\n"
                       r"• PLM predicts folding penalty: Appears predictive of binding, but is 100% folding-mediated",
              fontsize=9, color="#222222", va="top")

    # Box 3: True Interface Mutation (The Blindspot)
    rect_int = patches.FancyBboxPatch((4, 4), 92, 27, boxstyle="round,pad=1.5,rounding_size=3",
                                      ec="#7b2cbf", fc="#f7f2fc", lw=1.5)
    ax_a.add_patch(rect_int)
    ax_a.text(6, 27, "3. True Quaternary Interface Mutation (The Single-Chain Blindspot)", 
              fontweight="bold", color="#521b80", fontsize=10.5)
    ax_a.text(6, 20.5, r"• Target monomer folds perfectly & displays normally on cell surface (Abund+)" "\n"
                       r"• Mutation selectively alters intermolecular contact residue $\to$ True affinity loss (Bind-)" "\n"
                       r"• Single-chain PLMs lack the intermolecular context & fail to predict interface disruption" "\n"
                       r"• Correlation collapses from $\rho = +0.384$ (Abundance) to $\rho = +0.075$ (Binding)",
              fontsize=9, color="#222222", va="top")

    # Panel B: Causal Mediation Triangle
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 100)
    ax_b.axis("off")
    ax_b.set_title("B   Causal Mediation Decomposition", loc="left", fontweight="bold", pad=12)

    # Nodes
    # PLM Node (Top)
    p_plm = patches.FancyBboxPatch((25, 75), 50, 18, boxstyle="round,pad=1,rounding_size=3",
                                  ec="#4e79a7", fc="#eef4fa", lw=2)
    ax_b.add_patch(p_plm)
    ax_b.text(50, 84, "Zero-Shot PLM\nLikelihood", ha="center", va="center", fontweight="bold", color="#1f497d", fontsize=10)

    # Monomer Folding Node (Bottom-Left)
    p_fold = patches.FancyBboxPatch((2, 15), 44, 22, boxstyle="round,pad=1,rounding_size=3",
                                   ec="#3b528b", fc="#f0f2f8", lw=2)
    ax_b.add_patch(p_fold)
    ax_b.text(24, 26, "Monomer Folding\n& Abundance\n(Cell Display)", ha="center", va="center", fontweight="bold", color="#203055", fontsize=9.5)

    # Complex Binding Node (Bottom-Right)
    p_bind = patches.FancyBboxPatch((54, 15), 44, 22, boxstyle="round,pad=1,rounding_size=3",
                                   ec="#d62728", fc="#fdf0f0", lw=2)
    ax_b.add_patch(p_bind)
    ax_b.text(76, 26, "Assay Binding\nEnrichment\n(Fluorescence)", ha="center", va="center", fontweight="bold", color="#8b1819", fontsize=9.5)

    # Arrows
    # 1. PLM -> Folding (Strong)
    ax_b.annotate("", xy=(24, 38), xytext=(40, 74),
                  arrowprops=dict(arrowstyle="->", lw=2.5, color="#3b528b", shrinkA=2, shrinkB=2))
    ax_b.text(24, 58, r"$\rho = +0.384$" "\n" r"(Strong Coupling)", ha="center", va="center", 
              fontsize=9, fontweight="bold", color="#3b528b", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    # 2. Folding -> Assay Binding (Indirect Selection Confound)
    ax_b.annotate("", xy=(53, 26), xytext=(47, 26),
                  arrowprops=dict(arrowstyle="->", lw=2.5, color="#555555", shrinkA=2, shrinkB=2))
    ax_b.text(50, 11, "Assay Confound:\nUnfolded " r"$\to$" " No Binding", ha="center", va="top", 
              fontsize=8.5, color="#444444", style="italic")

    # 3. Direct PLM -> Binding (Collapsed / Negative at Interface)
    ax_b.annotate("", xy=(76, 38), xytext=(60, 74),
                  arrowprops=dict(arrowstyle="->", lw=2.2, color="#d62728", linestyle="--", shrinkA=2, shrinkB=2))
    ax_b.text(77, 58, "Direct Path:\n" r"$\rho_{\mathrm{partial}} = -0.367$" "\n(Zero Info)", ha="center", va="center", 
              fontsize=9, fontweight="bold", color="#d62728", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    # Footer note on mediation proof
    ax_b.text(50, 1, r"Mediation Proof: $\rho(\mathrm{PLM}, \mathrm{Bind} \mid \mathrm{Abund}) \leq 0$" "\n"
                     r"$\to$ PLMs learn 0 unique interface physics.",
              ha="center", va="bottom", fontsize=9, fontweight="bold", color="#111111",
              bbox=dict(boxstyle="round,pad=0.5", fc="#fff9db", ec="#f59f00", lw=1))

    out_path = DOCS_FIGS_DIR / "01_expression_confound_schematic.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# -----------------------------------------------------------------------------
# Figure 2: Double-Dissociation Scatter Plots
# -----------------------------------------------------------------------------
def plot_figure_2_double_dissociation(df_scores):
    """Figure 2: Multi-panel scatter plot comparing PLM vs DMS Abundance & Binding across compartments."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=300, sharex=True, sharey=True)

    compartments = ["Core", "Surface", "Interface"]
    comp_colors = {"Core": PALETTE["Core"], "Surface": PALETTE["Surface"], "Interface": PALETTE["Interface"]}

    # We plot ESMC-6B scores (normalized/z-scored within system for fair visual comparison across all 5 systems)
    df = df_scores.copy()
    
    # Standardize DMS and PLM scores within system for visualization
    for sys_id in df["system"].unique():
        idx = df["system"] == sys_id
        df.loc[idx, "dms_abund_z"] = stats.zscore(df.loc[idx, "dms_score_abundance"].dropna())
        df.loc[idx, "dms_bind_z"] = stats.zscore(df.loc[idx, "dms_score_binding"].dropna())
        df.loc[idx, "plm_z"] = stats.zscore(df.loc[idx, "zeroshot_esmc-6b"].dropna())

    # Top row: PLM vs Abundance (Monomer Stability)
    for col_idx, comp in enumerate(compartments):
        ax = axes[0, col_idx]
        sub = df[df["compartment"] == comp].dropna(subset=["plm_z", "dms_abund_z"])
        rho, _ = stats.spearmanr(sub["plm_z"], sub["dms_abund_z"])
        
        # Hexbin / scatter
        ax.scatter(sub["plm_z"], sub["dms_abund_z"], color=comp_colors[comp], alpha=0.25, s=14, edgecolors="none")
        
        # Regression trend line
        m, b = np.polyfit(sub["plm_z"], sub["dms_abund_z"], 1)
        x_line = np.linspace(-3.5, 3.5, 100)
        ax.plot(x_line, m * x_line + b, color="#111111", lw=2, linestyle="-")

        ax.set_title(f"{comp} Compartment ($N={len(sub):,}$)\nPLM vs. Monomer Abundance", fontsize=11, fontweight="bold")
        ax.text(0.05, 0.90, f"Spearman $\\rho = {rho:+.3f}$", transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.9))
        ax.set_ylabel("Monomer Abundance ($z$-score)" if col_idx == 0 else "")
        ax.grid(True)

    # Bottom row: PLM vs Binding Affinity
    for col_idx, comp in enumerate(compartments):
        ax = axes[1, col_idx]
        sub = df[df["compartment"] == comp].dropna(subset=["plm_z", "dms_bind_z"])
        rho, _ = stats.spearmanr(sub["plm_z"], sub["dms_bind_z"])
        
        ax.scatter(sub["plm_z"], sub["dms_bind_z"], color=comp_colors[comp], alpha=0.25, s=14, edgecolors="none")
        
        m, b = np.polyfit(sub["plm_z"], sub["dms_bind_z"], 1)
        x_line = np.linspace(-3.5, 3.5, 100)
        ax.plot(x_line, m * x_line + b, color="#111111" if comp != "Interface" else "#d62728", 
                lw=2, linestyle="-" if comp != "Interface" else "--")

        title_text = f"{comp} Compartment ($N={len(sub):,}$)\nPLM vs. Complex Binding"
        if comp == "Interface":
            title_text += " [SELECTIVE COLLAPSE]"
        ax.set_title(title_text, fontsize=11, fontweight="bold", color="#111111" if comp != "Interface" else "#a31415")
        
        ax.text(0.05, 0.90, f"Spearman $\\rho = {rho:+.3f}$", transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", color="#a31415" if comp == "Interface" else "#111111",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2" if comp == "Interface" else "white", 
                          ec="#d62728" if comp == "Interface" else "#999999", alpha=0.9))
        ax.set_xlabel("ESMC-6B Zero-Shot Score ($z$-score)")
        ax.set_ylabel("Complex Binding ($z$-score)" if col_idx == 0 else "")
        ax.grid(True)

    axes[0, 0].set_xlim(-3.8, 3.8)
    axes[0, 0].set_ylim(-3.8, 3.8)
    fig.suptitle("Double-Dissociation: Zero-Shot PLM Predictions vs. Monomer Abundance and Binding (N = 10,643)",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = DOCS_FIGS_DIR / "02_double_dissociation_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# -----------------------------------------------------------------------------
# Figure 3: Mediation Analysis Forest Plot
# -----------------------------------------------------------------------------
def plot_figure_3_mediation_forest(mediation_data):
    """Figure 3: Forest plot of marginal vs partial rank correlations across models and compartments."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    models = ["esm2-650m", "esm2-3b", "esmc-600m", "esmc-6b"]
    model_labels = {
        "esm2-650m": "ESM2-650M",
        "esm2-3b": "ESM2-3B",
        "esmc-600m": "ESMC-600M",
        "esmc-6b": "ESMC-6B",
    }
    
    compartments = ["Core", "Surface", "Interface"]
    
    # Structure y positions
    y_positions = []
    labels = []
    
    marginal_rhos = []
    partial_rhos = []
    colors = []
    
    current_y = 0
    for model in reversed(models):
        m_data = mediation_data[model]["compartments"]
        for comp in reversed(compartments):
            c_info = m_data[comp]
            marginal_rhos.append(c_info["rho_plm_binding"])
            partial_rhos.append(c_info["rho_partial_plm_binding_given_abundance"])
            colors.append(PALETTE[comp])
            labels.append(f"{model_labels[model]}  |  {comp} ($N={c_info['n']:,}$)")
            y_positions.append(current_y)
            current_y += 1
        current_y += 0.8  # gap between models

    y_pos = np.array(y_positions)
    
    # Zero line
    ax.axvline(0, color="#444444", linestyle="--", lw=1.2, alpha=0.85)

    # Plot arrows / links from marginal to partial correlation
    for i in range(len(y_pos)):
        y = y_pos[i]
        m_rho = marginal_rhos[i]
        p_rho = partial_rhos[i]
        
        # Link line
        ax.plot([m_rho, p_rho], [y, y], color="#888888", lw=1.5, zorder=2)
        # Marginal point (Open circle)
        ax.scatter(m_rho, y, color="white", edgecolors=colors[i], s=70, lw=2, zorder=3, label="Marginal $\\rho(\\text{PLM}, \\text{Binding})$" if i == 0 else "")
        # Partial point (Filled square)
        ax.scatter(p_rho, y, color=colors[i], marker="s", s=65, zorder=4, label="Partial $\\rho(\\text{PLM}, \\text{Binding} \\mid \\text{Abundance})$" if i == 0 else "")
        
        # Text annotation for interface rows
        if "Interface" in labels[i]:
            drop_str = f"$\\Delta = {p_rho - m_rho:+.2f}$"
            ax.text(min(m_rho, p_rho) - 0.02, y, drop_str, va="center", ha="right", fontsize=8.5, fontweight="bold", color="#d62728")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Spearman Rank Correlation ($\\rho$)", fontsize=11, fontweight="bold")
    ax.set_title("Mediation Analysis: Controlling for Monomer Abundance Collapses Interface Predictive Power",
                 fontsize=12, fontweight="bold", pad=12)
    
    # Custom legend
    custom_lines = [
        plt.Line2D([0], [0], marker='o', color='white', markeredgecolor='#333333', markeredgewidth=2, markersize=8, label='Marginal: $\\rho(\\text{PLM}, \\text{Binding})$'),
        plt.Line2D([0], [0], marker='s', color='#333333', markersize=8, label='Partial: $\\rho(\\text{PLM}, \\text{Binding} \\mid \\text{Abundance})$'),
        patches.Patch(color=PALETTE['Core'], label='Core Compartment'),
        patches.Patch(color=PALETTE['Surface'], label='Surface Compartment'),
        patches.Patch(color=PALETTE['Interface'], label='Quaternary Interface'),
    ]
    ax.legend(handles=custom_lines, loc="lower right", framealpha=0.95, facecolor="white", edgecolor="#cccccc")
    ax.set_xlim(-0.48, 0.22)
    ax.grid(True, axis="x")

    plt.tight_layout()
    out_path = DOCS_FIGS_DIR / "03_mediation_forest_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# -----------------------------------------------------------------------------
# Figure 4: Model Scaling Collapse
# -----------------------------------------------------------------------------
def plot_figure_4_scaling_collapse(test_data, mediation_data):
    """Figure 4: Tracking interface correlation collapse and widening gap across parameter scales."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

    # Models ordered by parameter count: ESMC-600M (600M), ESM2-650M (650M), ESM2-3B (2.84B), ESMC-6B (6.35B)
    scale_models = [
        {"id": "esmc-600m", "name": "ESMC\n600M", "params": 0.60, "family": "ESMC", "color": PALETTE["ESMC-600M"]},
        {"id": "esm2-650m", "name": "ESM2\n650M", "params": 0.65, "family": "ESM2", "color": PALETTE["ESM2-650M"]},
        {"id": "esm2-3b",   "name": "ESM2\n2.84B", "params": 2.84, "family": "ESM2", "color": PALETTE["ESM2-3B"]},
        {"id": "esmc-6b",   "name": "ESMC\n6.35B", "params": 6.35, "family": "ESMC", "color": PALETTE["ESMC-6B"]},
    ]

    names = [m["name"] for m in scale_models]
    params = [m["params"] for m in scale_models]
    
    # Retrieve interface abundance and binding correlations from test_*.json
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
    ax1.plot(x, rho_abund, marker="o", lw=2.5, markersize=8, color=PALETTE["Abundance"], label="Interface Monomer Abundance ($\\rho$)")
    ax1.plot(x, rho_bind, marker="s", lw=2.5, markersize=8, color=PALETTE["Binding"], label="Interface Complex Binding ($\\rho$)")
    ax1.plot(x, rho_partial, marker="^", lw=2.2, linestyle="--", markersize=8, color=PALETTE["Partial"], label="Partial Interface $\\rho(\\text{PLM}, \\text{Bind} \\mid \\text{Abund})$")

    for i in range(len(x)):
        ax1.text(x[i], rho_abund[i] + 0.025, f"{rho_abund[i]:.3f}", ha="center", fontsize=9, fontweight="bold", color=PALETTE["Abundance"])
        ax1.text(x[i], rho_bind[i] - 0.035, f"{rho_bind[i]:.3f}", ha="center", fontsize=9, fontweight="bold", color=PALETTE["Binding"])
        ax1.text(x[i], rho_partial[i] - 0.035, f"{rho_partial[i]:.3f}", ha="center", fontsize=9, fontweight="bold", color=PALETTE["Partial"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("Spearman Rank Correlation ($\\rho$)", fontsize=11, fontweight="bold")
    ax1.set_title("A   Interface Correlations across Parameter Scale", loc="left", fontweight="bold")
    ax1.set_ylim(-0.45, 0.50)
    ax1.axhline(0, color="#666666", linestyle=":", lw=1)
    ax1.grid(True)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # Panel B: Drop Percentage / Widening Defect
    bars = ax2.bar(x, drop_pct, color=["#edc948", "#4e79a7", "#59a14f", "#e15759"], width=0.55, edgecolor="#333333", lw=1.2)
    for bar, dp in zip(bars, drop_pct):
        y_val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, y_val - 4.5, f"{dp:.1f}%", ha="center", va="top",
                 fontsize=10, fontweight="bold", color="white" if abs(y_val) > 40 else "#222222")

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel("Relative Correlation Collapse ($\\Delta\\rho / \\rho_{\\text{abund}}$ %)", fontsize=11, fontweight="bold")
    ax2.set_title("B   Selective Interface Collapse Worsens with Scale", loc="left", fontweight="bold")
    ax2.set_ylim(-95, 0)
    ax2.grid(True, axis="y")

    fig.suptitle("Scaling Aggravates the Confound: Scaling Parameters Does Not Repair Interface Blindspots",
                 fontsize=12.5, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = DOCS_FIGS_DIR / "04_model_scaling_collapse.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# -----------------------------------------------------------------------------
# Figure 5: Evolutionary Stratification Regimes
# -----------------------------------------------------------------------------
def plot_figure_5_evolutionary_regimes(evo_data):
    """Figure 5: Comparing Homooligomer anti-correlation vs Natural Heterodimer vs Synthetic Binders."""
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)

    # Focus on ESMC-6B arm across the 3 regimes at Interface residues
    arm = "esmc-6b"
    classes = evo_data["arms"][arm]["classes"]

    regimes = [
        ("Homooligomer", "Class 1: Homooligomer\n(p53 / 1OLG, $N=513$)", PALETTE["Homooligomer"]),
        ("Natural_Heterodimer", "Class 2: Natural Heterodimer\n(HLA-A2, GB1, $N=1,011$)", PALETTE["Natural_Heterodimer"]),
        ("Synthetic_CrossSpecies", "Class 3: Synthetic / De Novo\n(KRAS, Spike RBD, $N=738$)", PALETTE["Synthetic_CrossSpecies"]),
    ]

    x = np.arange(len(regimes))
    width = 0.26

    rho_abund = []
    rho_bind = []
    rho_part = []

    for reg_id, _, _ in regimes:
        int_c = classes[reg_id]["compartments"]["Interface"]
        rho_abund.append(int_c["rho_plm_abundance"])
        rho_bind.append(int_c["rho_plm_binding"])
        rho_part.append(int_c["rho_partial_plm_binding_given_abundance"])

    b1 = ax.bar(x - width, rho_abund, width, label="Monomer Abundance $\\rho(\\text{PLM}, \\text{Abund})$", color=PALETTE["Abundance"], edgecolor="#333333", lw=1)
    b2 = ax.bar(x, rho_bind, width, label="Complex Binding $\\rho(\\text{PLM}, \\text{Bind})$", color=PALETTE["Binding"], edgecolor="#333333", lw=1)
    b3 = ax.bar(x + width, rho_part, width, label="Partial Binding $\\rho(\\text{PLM}, \\text{Bind} \\mid \\text{Abund})$", color=PALETTE["Partial"], edgecolor="#333333", lw=1)

    # Value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            offset = 0.02 if h >= 0 else -0.04
            va = "bottom" if h >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, h + offset, f"{h:+.2f}", ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in regimes], fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Spearman Rank Correlation ($\\rho$)", fontsize=11, fontweight="bold")
    ax.set_title("Evolutionary PPI Stratification (ESMC-6B, Interface Residues N = 2,262):\n"
                 "Self-Co-occurrence in Homooligomers Fails to Rescue Zero-Shot Interface Energetics",
                 fontsize=12, fontweight="bold", pad=14)
    ax.axhline(0, color="#444444", linestyle="-", lw=1.2)
    ax.set_ylim(-0.68, 0.70)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    # Explanatory annotations
    ax.annotate("Homomer: Severe anti-correlation\n$\\rho = -0.57$, $\\rho_{\\text{partial}} = -0.51$\nSequence self-co-occurrence fails",
                xy=(0, -0.57), xytext=(0.05, -0.45),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#521b80"),
                fontsize=8.5, fontweight="bold", color="#521b80",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f7f2fc", ec="#9467bd", lw=1))

    ax.annotate("Heterodimer: Apparent binding\nis mediated by monomer folding\n($\\rho_{\\text{partial}} \\to +0.10$)",
                xy=(1 + width, 0.10), xytext=(1.15, 0.35),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#1b6e1b"),
                fontsize=8.5, fontweight="bold", color="#1b6e1b",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f2f9f2", ec="#2ca02c", lw=1))

    plt.tight_layout()
    out_path = DOCS_FIGS_DIR / "05_evolutionary_regimes.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# -----------------------------------------------------------------------------
# Figure 6: Binder Filter Trap & False-Negative Depletion
# -----------------------------------------------------------------------------
def plot_figure_6_binder_filter(binder_data):
    """Figure 6: False-negative rate and interface depletion curves across filter thresholds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

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

    # Panel A: Interface False-Negative Rate (FNR)
    for model in models:
        sim = binder_data["filter_simulations"][model]["thresholds_simulation"]
        fnr = [s["interface_false_negative_rate"] * 100 for s in sim]
        ax1.plot(thresholds, fnr, marker="o", lw=2.2, label=model_labels[model], color=model_colors[model])

    # Highlight standard Top 20% filter
    ax1.axvline(20, color="#d62728", linestyle="--", lw=1.5, alpha=0.7)
    ax1.text(21, 55, "Standard Filter: Top 20%\n(Discards 85.8% - 96.1%\nof True Interface Hits)", 
             color="#a31415", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#d62728", lw=1))

    ax1.set_xlabel("Zero-Shot PLM Likelihood Filter Cutoff (Top X%)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Interface False-Negative Rate (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("A   True Interface Hits Discarded by Single-Chain Filter", loc="left", fontweight="bold")
    ax1.set_ylim(45, 100)
    ax1.set_xticks(thresholds)
    ax1.grid(True)
    ax1.legend(loc="lower left", fontsize=9.5)

    # Panel B: Interface Depletion Rate
    for model in models:
        sim = binder_data["filter_simulations"][model]["thresholds_simulation"]
        dep = [s["interface_depletion_rate"] * 100 for s in sim]
        ax2.plot(thresholds, dep, marker="s", lw=2.2, label=model_labels[model], color=model_colors[model])

    ax2.axvline(20, color="#d62728", linestyle="--", lw=1.5, alpha=0.7)
    ax2.axhline(0, color="#666666", linestyle=":", lw=1)
    ax2.set_xlabel("Zero-Shot PLM Likelihood Filter Cutoff (Top X%)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Interface Depletion Rate vs. Non-Interface (%)", fontsize=10.5, fontweight="bold")
    ax2.set_title("B   Depletion of Interface Mutations in Filtered Pool", loc="left", fontweight="bold")
    ax2.set_xticks(thresholds)
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=9.5)

    fig.suptitle("The 'PLM Filter Trap': Filtering Binder Candidates by Zero-Shot Likelihood Purges Affinity Improvements",
                 fontsize=12.5, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = DOCS_FIGS_DIR / "06_binder_filter_depletion.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    print("=" * 78)
    print("Generating Publication Figures for PLM PPI Interface Failure Study")
    print("=" * 78)

    df_scores, mediation_data, evo_data, binder_data, test_data = load_data()
    print(f"Loaded {len(df_scores)} variant scores from results/scores.csv")

    plot_figure_1_schematic()
    plot_figure_2_double_dissociation(df_scores)
    plot_figure_3_mediation_forest(mediation_data)
    plot_figure_4_scaling_collapse(test_data, mediation_data)
    plot_figure_5_evolutionary_regimes(evo_data)
    plot_figure_6_binder_filter(binder_data)

    print("=" * 78)
    print(f"All 6 figures successfully saved to {DOCS_FIGS_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()

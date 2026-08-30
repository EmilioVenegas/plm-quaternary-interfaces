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
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from diagram import Ctx, card, paragraph, bullets, pill, PAPER, WASH, HAIR, INK, BODY, MUTED, DOT, SLATE, EMERALD, ROSE, INDIGO, VIOLET, AMBER, flow, descend
from scipy import stats

# Publication styling defaults (Nature / Science guidelines)
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

RESULTS_DIR = REPO_ROOT / "results"
DOCS_FIGS_DIR = REPO_ROOT / "docs" / "figures"
DOCS_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Subdued, mature, publication-grade academic palette
PALETTE = {
    # Structural Compartments
    "Core": "#475569",       # Deep slate grey
    "Surface": "#2d6a4f",    # Muted forest/sage green
    "Interface": "#a83232",  # Subdued brick/crimson red

    # Assay / Readout Types
    "Abundance": "#3b6998",  # Muted steel/slate blue
    "Binding": "#9c3848",    # Muted rose/wine red
    "Partial": "#c07f1a",    # Warm antique ochre/amber

    # Model Architectures (Muted sequential blues/purples)
    "ESMC-600M": "#5a7d9a",  # Soft slate cyan
    "ESM2-650M": "#3b6998",  # Muted steel blue
    "ESM2-3B": "#4f5fc4",    # Subdued indigo
    "ESMC-6B": "#7c5cc7",    # Muted deep violet

    # Evolutionary PPI Regimes
    "Homooligomer": "#7c5cc7",       # Muted violet
    "Natural_Heterodimer": "#2d6a4f", # Muted forest green
    "Synthetic_CrossSpecies": "#c07f1a", # Muted warm amber
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
# Real-structure assets for the Figure 1 schematic.
#
# Panel A illustrates the confound with cartoon renders of an ACTUAL benchmark complex
# (6H46: KRAS G-domain + DARPin K55), produced by scripts/render_structures.py under a
# headless PyMOL environment. Every layer shares one camera and canvas, so cropping them
# all to a common union bbox preserves the true relative geometry of the complex -- that
# is what makes "target + partner_bound" reproduce the real interface, and lets the
# dissociated case swap in a rigid-body-displaced partner without anything shifting.
# -----------------------------------------------------------------------------
STRUCT_ASSET_DIR = DOCS_FIGS_DIR / "assets"
STRUCT_LAYERS = ("mol_target", "mol_target_mut", "mol_partner_bound",
                 "mol_partner_away", "mol_coil_ensemble")
_STRUCT_CACHE: dict | None = None


def load_structure_layers() -> dict:
    """Load, common-crop and downsample the PyMOL cartoon layers.

    Returns a dict with:
        layers  {name: RGBA uint8 array}, all identically sized
        boxes   {name: (x0, y0, x1, y1)} alpha bbox normalised to the common crop,
                image convention (y measured downward from the top)
        aspect  width / height of the common crop
    """
    global _STRUCT_CACHE
    if _STRUCT_CACHE is not None:
        return _STRUCT_CACHE

    from PIL import Image

    missing = [n for n in STRUCT_LAYERS if not (STRUCT_ASSET_DIR / f"{n}.png").exists()]
    if missing:
        raise FileNotFoundError(
            f"missing structure assets {missing} in {STRUCT_ASSET_DIR}. "
            "Regenerate them with a PyMOL environment:\n"
            "    <pymol-env>/bin/python scripts/render_structures.py")

    raw = {n: Image.open(STRUCT_ASSET_DIR / f"{n}.png").convert("RGBA") for n in STRUCT_LAYERS}

    def alpha_bbox(img):
        a = np.asarray(img)[..., 3]
        ys, xs = np.nonzero(a > 8)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    boxes_px = {n: alpha_bbox(im) for n, im in raw.items()}
    ux0 = min(b[0] for b in boxes_px.values())
    uy0 = min(b[1] for b in boxes_px.values())
    ux1 = max(b[2] for b in boxes_px.values())
    uy1 = max(b[3] for b in boxes_px.values())
    uw, uh = ux1 - ux0, uy1 - uy0

    # Downsample: final placement is <2 in at 300 dpi (~600 px), so 900 px on the long
    # edge is ample and keeps these arrays small.
    target_w = 900
    scale = min(1.0, target_w / uw)
    out_size = (max(1, int(round(uw * scale))), max(1, int(round(uh * scale))))

    layers, boxes = {}, {}
    for n, im in raw.items():
        layers[n] = np.asarray(im.crop((ux0, uy0, ux1, uy1)).resize(out_size, Image.LANCZOS))
        x0, y0, x1, y1 = boxes_px[n]
        boxes[n] = ((x0 - ux0) / uw, (y0 - uy0) / uh, (x1 - ux0) / uw, (y1 - uy0) / uh)

    # Locate the flagged interface residue by differencing the plain and mutation-marked
    # target layers, so the callout anchor is derived and never hand-placed.
    # Detection is by COLOUR, not alpha: both layers render the same molecular surface,
    # so they share an identical silhouette and an alpha diff finds nothing. The only
    # thing that changes is the recoloured patch over the mutated residue.
    plain = np.asarray(raw["mol_target"]).astype(np.int16)
    marked = np.asarray(raw["mol_target_mut"]).astype(np.int16)
    colour_delta = np.abs(marked[..., :3] - plain[..., :3]).sum(axis=-1)
    diff_ys, diff_xs = np.nonzero((colour_delta > 40) & (marked[..., 3] > 8))
    if diff_xs.size:
        mut_center = (float(diff_xs.mean() - ux0) / uw, float(diff_ys.mean() - uy0) / uh)
    else:  # defensive: fall back to the target's right edge
        bx0, by0, bx1, by1 = boxes["mol_target"]
        mut_center = (bx1, (by0 + by1) / 2)

    meta_path = STRUCT_ASSET_DIR / "structure_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    mut_label = meta.get("mutation_site", "interface residue")
    if len(mut_label) > 3 and mut_label[:3].isalpha():
        # HIS94 -> H94, matching the one-letter variant notation used elsewhere.
        three_to_one = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
                        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
        mut_label = three_to_one.get(mut_label[:3].upper(), mut_label[:1]) + mut_label[3:]

    _STRUCT_CACHE = {"layers": layers, "boxes": boxes, "aspect": uw / uh,
                     "mut_center": mut_center, "mut_label": mut_label,
                     "pdb_id": meta.get("pdb_id", ""),
                     "coil_model": meta.get("coil_model", {})}
    return _STRUCT_CACHE


# Figure 1: Biophysical & Assay Schematic of the Expression Confound
# -----------------------------------------------------------------------------
def plot_figure_1_schematic():
    """Figure 1: Biophysical and assay schematic of the monomer-folding confound in PPI selections.

    Deliberately austere: no in-figure title, subtitle or provenance footnote -- those live in
    the manuscript caption (@fig-schematic), which is where a journal expects them. Panels
    carry plain letter labels, cards are unfilled with black hairline borders, and colour is
    reserved for the status badges and the molecules themselves.
    """
    WIDTH = 11.5
    HEIGHT = 6.12
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    # Panel A/C: restrained palette -- desaturated inks, black card rules, colour reserved
    # for status badges. Panel B and the molecular renders keep their original saturation
    # (the causal diagram is the analytical core of the figure and reads better in full
    # colour, and the cartoons are the scientific content).
    C_INK = "#1a1a1a"
    C_BODY = "#3f3f46"
    C_MUTED = "#71717a"
    C_HAIR = "#d4d4d8"
    C_RULE = "#000000"        # card borders
    C_PAPER = "#ffffff"

    C_POS = "#3f6b52"         # muted green: positive readout
    C_NEG = "#9b4444"         # muted red: negative readout
    C_BLUE_TEXT = "#3a5a80"   # muted blue: model / structure text

    BADGE_POS = "#4b7f5e"
    BADGE_CONF = "#a85454"
    BADGE_COLL = "#6b5b95"

    # Panel B (unmuted)
    B_BLUE = "#2563eb"
    B_BLUE_TEXT = "#1d4ed8"
    B_ROSE = "#ef4444"
    B_ROSE_TEXT = "#b91c1c"
    B_SLATE = "#475569"
    B_AMBER_TEXT = "#b45309"
    B_AMBER_BG = "#fffbeb"
    B_AMBER_BORDER = "#fde68a"
    B_BLUE_BORDER = "#bfdbfe"
    B_ROSE_BORDER = "#fecaca"

    def panel_label(x, y, text):
        """Plain letter label -- no pill, no box."""
        ax.text(x, y, text, fontsize=7.4, fontweight="bold", color=C_MUTED,
                ha="left", va="center", family="Liberation Sans")

    # ---------------- Panel A ----------------
    panel_label(0.40, HEIGHT - 0.22, "A \u00b7 SELECTION PHENOMENOLOGY IN EXPRESSION-COUPLED SELECTION ASSAYS")

    w_panel_a = 6.70
    h_panel_a = 5.25
    y_top_a = HEIGHT - 0.40

    w_case = 2.10
    h_case = 4.95
    y_case_top = y_top_a - 0.15
    x_cases = [0.52, 0.52 + w_case + 0.13, 0.52 + 2 * (w_case + 0.13)]

    def draw_case_card(x, y, w, h, case_num, title, badge_text, badge_color, mode):
        # Unfilled card, black hairline rule.
        ax.add_patch(FancyBboxPatch((x, y - h), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=C_PAPER, edgecolor=C_RULE, lw=0.8, zorder=2))

        ax.text(x + 0.10, y - 0.15, f"CASE {case_num}", fontsize=6.8, fontweight="bold",
                color=C_MUTED, family="Liberation Sans")
        ax.text(x + 0.10, y - 0.33, title, fontsize=7.3, fontweight="bold",
                color=C_INK, family="Liberation Sans")
        # Status badge keeps a colour fill -- it is the one categorical cue per card.
        ax.text(x + w - 0.10, y - 0.23, badge_text, fontsize=5.8, fontweight="bold",
                color=C_PAPER, ha="right", va="center", family="Liberation Sans",
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.10",
                          facecolor=badge_color, edgecolor="none"), zorder=6)
        ax.plot([x + 0.10, x + w - 0.10], [y - 0.46, y - 0.46], color=C_HAIR, lw=0.7, zorder=3)

        # ---- illustration area (no background fill) ---------------------------------
        ill_y_top = y - 0.50
        ill_h = 2.45

        # Lipid bilayer glyph: two leaflets of head groups with acyl tails between.
        # Deliberately schematic -- rationale in the tether comment below.
        mem_y = ill_y_top - ill_h + 0.36
        head_r = 0.023
        leaflet_gap = 0.072
        y_out = mem_y + leaflet_gap
        y_in = mem_y
        for hx in np.arange(x + 0.11, x + w - 0.10, 2 * head_r + 0.014):
            ax.plot([hx, hx], [y_in + head_r, y_out - head_r], color="#d4d4d8", lw=0.7, zorder=3.8)
            ax.add_patch(Circle((hx, y_out), head_r, facecolor="#c9ccd1",
                                edgecolor="#a1a1aa", lw=0.4, zorder=4))
            ax.add_patch(Circle((hx, y_in), head_r, facecolor="#c9ccd1",
                                edgecolor="#a1a1aa", lw=0.4, zorder=4))
        ax.text(x + w / 2, y_in - 0.13, "Display / expression host surface", ha="center",
                va="center", fontsize=5.3, color=C_MUTED, family="Liberation Sans", zorder=4)

        # ---- real-structure composite ------------------------------------------------
        # Layers share one camera, so drawing them into the same rect reproduces the
        # true crystallographic geometry of 6H46.
        struct = load_structure_layers()
        img_x0 = x + 0.08
        img_w = w - 0.16
        img_h = img_w / struct["aspect"]
        img_y0 = mem_y + 0.30   # headroom for the anchor tether below the molecules
        img_y1 = img_y0 + img_h

        def box_data(name):
            """Layer alpha bbox -> data coords (x0, y_bottom, x1, y_top)."""
            bx0, by0, bx1, by1 = struct["boxes"][name]
            return (img_x0 + bx0 * img_w, img_y1 - by1 * img_h,
                    img_x0 + bx1 * img_w, img_y1 - by0 * img_h)

        def blit(name):
            ax.imshow(struct["layers"][name], extent=(img_x0, img_x0 + img_w, img_y0, img_y1),
                      zorder=5, aspect="auto", interpolation="antialiased")

        target_layer = {"wt": "mol_target", "misfolded": "mol_coil_ensemble",
                        "interface": "mol_target_mut"}[mode]
        partner_layer = "mol_partner_bound" if mode == "wt" else "mol_partner_away"
        blit(target_layer)
        blit(partner_layer)

        tx0, ty0, tx1, ty1 = box_data(target_layer)
        px0, py0, px1, py1 = box_data(partner_layer)

        # Generic surface-fusion tether. Deliberately schematic, NOT a molecular model:
        #   - the five benchmark systems span yeast display, mammalian display, mRNA
        #     display and a cell-line assay, so committing to one anchor (e.g. Aga2p)
        #     would re-introduce the over-specification this panel avoids;
        #   - Aga2p has no experimental structure, so drawing it would mean mixing a
        #     predicted model into a panel whose other molecules are crystallographic;
        #   - an all-atom bilayer at this scale is a dense smear that competes with the
        #     complex for attention while carrying no information.
        mem_top = y_out + head_r
        stalk_x = x + 0.55
        ax.text(x + 0.23, mem_y + 0.21, "Surface\nanchor", ha="center", va="center",
                fontsize=5.0, color=C_MUTED, family="Liberation Sans", zorder=5)

        if mode == "misfolded":
            # Nothing reaches the surface: a stub tether capped with a clash marker.
            stub_top = mem_y + 0.30
            ax.plot([stalk_x, stalk_x], [mem_top, stub_top], color="#52525b", lw=2.0, zorder=4)
            ax.plot([stalk_x - 0.045, stalk_x + 0.045], [stub_top - 0.045, stub_top + 0.045],
                    color=C_NEG, lw=1.9, zorder=6)
            ax.plot([stalk_x - 0.045, stalk_x + 0.045], [stub_top + 0.045, stub_top - 0.045],
                    color=C_NEG, lw=1.9, zorder=6)
        else:
            ax.plot([stalk_x, stalk_x], [mem_top, ty0 + 0.05], color="#52525b", lw=2.0, zorder=4)

        # ---- readout labels above each molecule ----------------------------------------
        # The FITC/PE probe discs were dropped: they duplicated the Abund/Bind labels and
        # the readout rows below, and named fluorophores specific to one assay platform.
        abund_ok = mode != "misfolded"
        bind_ok = mode == "wt"
        name_y = img_y1 + 0.08
        status_y = img_y1 + 0.22
        t_mid, p_mid = (tx0 + tx1) / 2, (px0 + px1) / 2

        ax.text(t_mid, status_y, "Abundance +" if abund_ok else "Abundance \u2212",
                fontsize=5.9, fontweight="bold", ha="center", va="center",
                color=C_POS if abund_ok else C_NEG, family="Liberation Sans", zorder=9)
        ax.text(t_mid, name_y,
                {"wt": "KRAS (folded)", "misfolded": "KRAS (unfolded, model)",
                 "interface": "KRAS (folded)"}[mode],
                fontsize=5.2, ha="center", va="center",
                color=C_BODY, family="Liberation Sans", zorder=9)

        ax.text(p_mid, status_y, "Binding +" if bind_ok else "Binding \u2212",
                fontsize=5.9, fontweight="bold", ha="center", va="center",
                color=C_POS if bind_ok else C_NEG, family="Liberation Sans", zorder=9)
        ax.text(p_mid, name_y, "DARPin (bound)" if bind_ok else "DARPin (unbound)",
                fontsize=5.2, ha="center", va="center",
                color=C_BODY, family="Liberation Sans", zorder=9)

        if mode == "interface":
            # The recoloured patch in mol_target_mut is the real interface contact residue.
            mx, my = struct["mut_center"]
            mdx = img_x0 + mx * img_w
            mdy = img_y1 - my * img_h
            ax.annotate(struct["mut_label"], xy=(mdx, mdy), xytext=(mdx - 0.24, mdy + 0.30),
                        arrowprops=dict(arrowstyle="-", color=C_NEG, lw=0.8,
                                        shrinkA=0, shrinkB=3),
                        fontsize=5.2, fontweight="bold", color=C_NEG,
                        family="Liberation Sans", ha="center", va="center", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor=C_HAIR, lw=0.5, alpha=0.92))

        # ---- readout rows ---------------------------------------------------------------
        r_y = y - 3.07

        def row_item(y_pos, label, val, val_color=C_INK, is_bold=False):
            ax.text(x + 0.10, y_pos, label, fontsize=6.0, color=C_MUTED,
                    family="Liberation Sans", va="center")
            ax.text(x + w - 0.10, y_pos, val, fontsize=6.0,
                    fontweight="bold" if is_bold else "normal",
                    color=val_color, family="Liberation Sans", ha="right", va="center")

        row_item(r_y - 0.10, "Target monomer:",
                 "Stably folded" if abund_ok else "Misfolded / degraded",
                 val_color=C_POS if abund_ok else C_NEG, is_bold=True)
        row_item(r_y - 0.32, "Cell-surface abundance:",
                 "High" if abund_ok else "Zero",
                 val_color=C_POS if abund_ok else C_NEG)
        row_item(r_y - 0.54, "Complex binding:",
                 "Bound" if bind_ok else "Abolished",
                 val_color=C_POS if bind_ok else C_NEG)
        plm_text = r"$\Delta\log p \geq 0$ (high)" if abund_ok else r"$\Delta\log p \ll 0$ (penalty)"
        row_item(r_y - 0.76, "PLM zero-shot:", plm_text,
                 val_color=C_BLUE_TEXT if abund_ok else C_NEG, is_bold=True)

        ax.plot([x + 0.10, x + w - 0.10], [r_y - 0.94, r_y - 0.94], color=C_HAIR, lw=0.7)

        # ---- outcome (no fill, no tinted box) -------------------------------------------
        concl_y = r_y - 1.10
        head, body, head_c = {
            "wt": ("TRUE POSITIVE",
                   "Folded and binding intact;\nmodel score aligns with assay.", C_POS),
            "misfolded": ("BENCHMARK CONFOUND",
                          "PLM predicts folding collapse,\nnot quaternary affinity.", C_NEG),
            "interface": ("ZERO-SHOT COLLAPSE",
                          "Monomer folds, affinity lost;\n"
                          r"model is blind ($\rho \rightarrow +0.075$).", "#5b4a7a"),
        }[mode]
        ax.text(x + 0.10, concl_y, head, fontsize=6.3, fontweight="bold",
                color=head_c, family="Liberation Sans", va="top")
        ax.text(x + 0.10, concl_y - 0.22, body, fontsize=5.6, color=C_BODY,
                family="Liberation Sans", va="top")

    draw_case_card(x_cases[0], y_case_top, w_case, h_case, "1", "WILD-TYPE / BENIGN",
                   "TRUE POSITIVE", BADGE_POS, "wt")
    draw_case_card(x_cases[1], y_case_top, w_case, h_case, "2", "CORE DESTABILIZATION",
                   "CONFOUND", BADGE_CONF, "misfolded")
    draw_case_card(x_cases[2], y_case_top, w_case, h_case, "3", "INTERFACE MUTATION",
                   "COLLAPSE", BADGE_COLL, "interface")

    # ---------------- Panel B ----------------
    panel_label(7.35, HEIGHT - 0.22, "B \u00b7 CAUSAL MEDIATION DECOMPOSITION")

    w_panel_b = 3.75
    h_panel_b = 2.50
    y_top_b = HEIGHT - 0.40

    ax.add_patch(FancyBboxPatch((7.35, y_top_b - h_panel_b), w_panel_b, h_panel_b,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=C_PAPER, edgecolor=C_RULE, lw=0.8, zorder=1))

    ax.text(7.50, y_top_b - 0.22, "Monomer folding mediates the apparent binding signal",
            fontsize=7.4, fontweight="bold", color=B_BLUE_TEXT, family="Liberation Sans")

    def draw_dag_node(nx, ny, nw, nh, label, sublabel, edge_c, text_c):
        ax.add_patch(FancyBboxPatch((nx, ny - nh), nw, nh,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=C_PAPER, edgecolor=edge_c, lw=1.0, zorder=3))
        ax.text(nx + nw / 2, ny - 0.16, label, fontsize=6.8, fontweight="bold", color=text_c,
                ha="center", va="center", family="Liberation Sans", zorder=4)
        ax.text(nx + nw / 2, ny - 0.34, sublabel, fontsize=5.6, color=C_MUTED,
                ha="center", va="center", family="Liberation Sans", zorder=4)

    n1_w, n1_h = 1.85, 0.48
    n1_x, n1_y = 7.35 + (w_panel_b - n1_w) / 2, y_top_b - 0.45
    draw_dag_node(n1_x, n1_y, n1_w, n1_h, "Zero-shot PLM score",
                  r"$\Delta\log p$ (evolutionary prior)", B_BLUE, B_BLUE_TEXT)

    n2_w, n2_h = 1.48, 0.48
    n2_x, n2_y = 7.50, y_top_b - 1.46
    draw_dag_node(n2_x, n2_y, n2_w, n2_h, "Monomer folding",
                  r"$y_{\mathrm{abundance}}$ (cell display)", B_SLATE, C_INK)

    n3_w, n3_h = 1.48, 0.48
    n3_x, n3_y = 7.35 + w_panel_b - n3_w - 0.15, y_top_b - 1.46
    draw_dag_node(n3_x, n3_y, n3_w, n3_h, "Assay readout",
                  r"$y_{\mathrm{binding}}$ (FACS readout)", B_ROSE, B_ROSE_TEXT)

    ax.add_patch(FancyArrowPatch((n1_x + 0.35, n1_y - n1_h - 0.02), (n2_x + n2_w / 2, n2_y + 0.02),
                                 arrowstyle="-|>", mutation_scale=11, color=B_BLUE,
                                 lw=1.8, zorder=2))
    ax.text(7.68, y_top_b - 1.02, r"$\rho = +0.384$" + "\n(fold prior)", fontsize=5.6,
            fontweight="bold", color=B_BLUE_TEXT, ha="center", family="Liberation Sans",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=C_PAPER, edgecolor=B_BLUE_BORDER, lw=0.6))

    ax.add_patch(FancyArrowPatch((n2_x + n2_w + 0.02, n2_y - n2_h / 2), (n3_x - 0.02, n3_y - n3_h / 2),
                                 arrowstyle="-|>", mutation_scale=11, color=B_SLATE,
                                 lw=1.8, zorder=2))
    ax.text(7.35 + w_panel_b / 2, y_top_b - 1.30, "Assay confound:\nunfolded \u2192 no signal",
            fontsize=5.2, fontweight="bold", color=B_SLATE, ha="center", family="Liberation Sans")

    ax.add_patch(FancyArrowPatch((n1_x + n1_w - 0.35, n1_y - n1_h - 0.02), (n3_x + n3_w / 2, n3_y + 0.02),
                                 arrowstyle="-|>", mutation_scale=11, color=B_ROSE,
                                 linestyle="--", lw=1.8, zorder=2))
    ax.text(10.28, y_top_b - 1.02, "Direct path:\n" + r"$\rho_{\mathrm{partial}} = +0.009$",
            fontsize=5.6, fontweight="bold", color=B_ROSE_TEXT, ha="center", family="Liberation Sans",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=C_PAPER, edgecolor=B_ROSE_BORDER, lw=0.6))

    ax.add_patch(FancyBboxPatch((7.50, y_top_b - h_panel_b + 0.12), w_panel_b - 0.30, 0.40,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=B_AMBER_BG, edgecolor=B_AMBER_BORDER, lw=0.8, zorder=2))
    ax.text(7.35 + w_panel_b / 2, y_top_b - h_panel_b + 0.32,
            r"$\rho(\mathrm{PLM},\,\mathrm{Binding} \mid \mathrm{Abundance}) \to 0 \Rightarrow \mathrm{zero\ unique\ mutual\ information}$",
            fontsize=6.5, fontweight="bold", color=B_AMBER_TEXT, ha="center", va="center")

    # ---------------- Panel C ----------------
    y_top_c = y_top_b - h_panel_b - 0.50
    panel_label(7.35, y_top_c + 0.18, "C \u00b7 WITHIN-TARGET DOUBLE-DISSOCIATION BENCHMARK")

    w_panel_c = 3.75
    h_panel_c = 2.50

    ax.add_patch(FancyBboxPatch((7.35, y_top_c - h_panel_c), w_panel_c, h_panel_c,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=C_PAPER, edgecolor=C_RULE, lw=0.8, zorder=1))

    ax.text(7.50, y_top_c - 0.22, r"Spearman $\rho$ across structural compartments ($N = 10{,}643$)",
            fontsize=7.2, fontweight="bold", color=C_INK, family="Liberation Sans")

    comp_data = [
        ("Core residues", "+0.170", "+0.227", "Fold prediction"),
        ("Surface residues", "+0.169", "+0.060", "Fold prediction"),
        ("Interface residues", "+0.384", "+0.075", "SELECTIVE COLLAPSE"),
    ]

    table_y = y_top_c - 0.50
    ax.text(7.50, table_y, "Compartment", fontsize=6.2, fontweight="bold", color=C_MUTED,
            family="Liberation Sans")
    ax.text(8.85, table_y, r"$\rho(\mathrm{abundance})$", fontsize=6.2, fontweight="bold",
            color=C_MUTED, ha="center")
    ax.text(9.62, table_y, r"$\rho(\mathrm{binding})$", fontsize=6.2, fontweight="bold",
            color=C_MUTED, ha="center")
    ax.text(10.52, table_y, "Outcome", fontsize=6.2, fontweight="bold", color=C_MUTED,
            family="Liberation Sans", ha="center")
    ax.plot([7.50, 10.95], [table_y - 0.10, table_y - 0.10], color=C_INK, lw=0.7)

    row_y = table_y - 0.28
    for comp_name, r_abund, r_bind, outcome in comp_data:
        is_int = "Interface" in comp_name
        ax.text(7.50, row_y - 0.11, comp_name, fontsize=6.5,
                fontweight="bold" if is_int else "normal",
                color=C_NEG if is_int else C_INK, family="Liberation Sans")
        ax.text(8.85, row_y - 0.11, r_abund, fontsize=6.5, color=C_BLUE_TEXT,
                fontweight="bold", ha="center", family="Liberation Sans")
        ax.text(9.62, row_y - 0.11, r_bind, fontsize=6.5,
                color=C_NEG if is_int else C_POS,
                fontweight="bold", ha="center", family="Liberation Sans")
        ax.text(10.52, row_y - 0.11, outcome, fontsize=5.8,
                fontweight="bold" if is_int else "normal",
                color=C_NEG if is_int else C_MUTED, ha="center", family="Liberation Sans")
        if is_int:
            ax.plot([7.50, 10.95], [row_y - 0.26, row_y - 0.26], color=C_HAIR, lw=0.6)
        row_y -= 0.36

    ax.text(7.50, y_top_c - h_panel_c + 0.58, "The diagnostic double-dissociation test",
            fontsize=6.5, fontweight="bold", color=C_INK, family="Liberation Sans")
    ax.text(7.50, y_top_c - h_panel_c + 0.44,
            "Holding library, host and sequence constant while measuring both folding\n"
            "and complex affinity shows that zero-shot PLMs are blind to quaternary\n"
            "interface contacts.",
            fontsize=5.7, color=C_BODY, family="Liberation Sans", va="top")

    out_path = DOCS_FIGS_DIR / "01_expression_confound_schematic.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# -----------------------------------------------------------------------------
# Figure 2: Double-Dissociation Scatter Plots
# -----------------------------------------------------------------------------
def plot_figure_2_double_dissociation(df_scores):
    """Figure 2: Multi-panel scatter plot comparing PLM vs DMS Abundance & Binding across compartments."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), dpi=300, sharex=True, sharey=True, facecolor="#ffffff")

    compartments = ["Core", "Surface", "Interface"]
    comp_colors = {"Core": PALETTE["Core"], "Surface": PALETTE["Surface"], "Interface": PALETTE["Interface"]}
    comp_ns = {"Core": "N = 4,405", "Surface": "N = 3,976", "Interface": "N = 2,262"}

    # Standardize DMS and PLM scores within system for fair visual comparison across systems
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

        # Model band background across all three zones
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
        current_y += 0.5  # gap between models

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

    # Forest plot header line
    ax_plot.axvline(0, color="#64748b", linestyle="--", lw=0.9, alpha=0.85, zorder=1)

    # Render Rows
    for model, comp, n, m_rho, p_rho, d_rho, y in row_data:
        color = PALETTE[comp]
        is_int = (comp == "Interface")

        # 1. Left axis text
        ax_left.text(0.60, y, comp, fontsize=8.5, color=color if is_int else "#334155",
                     fontweight="bold" if is_int else "normal", va="center")

        # 2. Forest plot markers
        ax_plot.plot([m_rho, p_rho], [y, y], color="#94a3b8", lw=1.3, zorder=2)
        ax_plot.scatter(m_rho, y, color="white", edgecolors=color, s=48, lw=1.6, zorder=3)
        ax_plot.scatter(p_rho, y, color=color, marker="s", s=42, zorder=4)

        # 3. Table values
        weight = "bold" if is_int else "normal"
        t_color = "#a83232" if is_int else "#334155"
        ax_table.text(col_x[0], y, f"{n:,}", fontsize=8.2, color="#64748b", ha="center", va="center")
        ax_table.text(col_x[1], y, f"{m_rho:+.3f}", fontsize=8.2, color="#334155", ha="center", va="center")
        ax_table.text(col_x[2], y, f"{p_rho:+.3f}", fontsize=8.2, fontweight=weight, color=t_color, ha="center", va="center")
        ax_table.text(col_x[3], y, f"{d_rho:+.2f}", fontsize=8.2, fontweight=weight, color=t_color, ha="center", va="center")

    # Render Model Names on the left (vertically centered per model group)
    for model in models:
        mod_rows = [r for r in row_data if r[0] == model]
        mid_y = (mod_rows[0][6] + mod_rows[-1][6]) / 2.0
        name, params = model_meta[model]
        ax_left.text(0.05, mid_y, f"{name}\n({params})", fontsize=8.5, fontweight="bold",
                     color="#0f172a", va="center")

    # Polish ax_plot
    ax_plot.set_xlabel("Spearman Rank Correlation ($\\rho$)", fontsize=9.5, fontweight="bold", color="#1e293b")
    ax_plot.set_yticks([])
    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    ax_plot.spines["left"].set_color("#cbd5e1")
    ax_plot.grid(True, axis="x", color="#f1f5f9", linestyle="--", lw=0.6)

    # Global Legend at bottom
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


# -----------------------------------------------------------------------------
# Figure 4: Model Scaling Collapse
# -----------------------------------------------------------------------------
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
        # Center text inside the bar
        ax2.text(bar.get_x() + bar.get_width() / 2, y_val / 2, f"{dp:.1f}%", ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color="#ffffff")

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.0)
    ax2.set_ylabel("Relative Interface Collapse ($\\Delta\\rho / \\rho_{\\mathrm{abund}}$ %)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(-95, 0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, axis="y")

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
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300, facecolor="#ffffff")

    arm = "esmc-6b"
    classes = evo_data["arms"][arm]["classes"]

    regimes = [
        ("Homooligomer", "Class 1: Homooligomer\n(p53, $N=513$)", PALETTE["Homooligomer"]),
        ("Natural_Heterodimer", "Class 2: Natural Heterodimer\n(HLA-A2, GB1, $N=1,011$)", PALETTE["Natural_Heterodimer"]),
        ("Synthetic_CrossSpecies", "Class 3: Synthetic / De Novo\n(KRAS, Spike RBD, $N=738$)", PALETTE["Synthetic_CrossSpecies"]),
    ]

    x = np.arange(len(regimes))
    width = 0.22

    rho_abund = []
    rho_bind = []
    rho_part = []

    for reg_id, _, _ in regimes:
        int_c = classes[reg_id]["compartments"]["Interface"]
        rho_abund.append(int_c["rho_plm_abundance"])
        rho_bind.append(int_c["rho_plm_binding"])
        rho_part.append(int_c["rho_partial_plm_binding_given_abundance"])

    b1 = ax.bar(x - width, rho_abund, width, label="Monomer Abundance $\\rho(\\mathrm{PLM}, \\mathrm{Abund})$", color=PALETTE["Abundance"], edgecolor="#334155", lw=0.7)
    b2 = ax.bar(x, rho_bind, width, label="Complex Binding $\\rho(\\mathrm{PLM}, \\mathrm{Bind})$", color=PALETTE["Binding"], edgecolor="#334155", lw=0.7)
    b3 = ax.bar(x + width, rho_part, width, label="Partial Binding $\\rho(\\mathrm{PLM}, \\mathrm{Bind} \\mid \\mathrm{Abund})$", color=PALETTE["Partial"], edgecolor="#334155", lw=0.7)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            offset = 0.02 if h >= 0 else -0.04
            va = "bottom" if h >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:+.2f}", ha="center", va=va, fontsize=8.0, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in regimes], fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Spearman Rank Correlation ($\\rho$)", fontsize=9.5, fontweight="bold")
    ax.axhline(0, color="#475569", linestyle="-", lw=0.8)
    ax.set_ylim(-0.68, 0.68)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.0, edgecolor="#cbd5e1")

    # Elegant callout cards positioned cleanly in open white space
    ax.annotate("Homomer: Anti-correlation\n$\\rho = -0.57$, $\\rho_{\\mathrm{partial}} = -0.51$",
                xy=(0, -0.35), xytext=(-0.50, -0.32),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#7c5cc7"),
                fontsize=7.5, fontweight="bold", color="#432d7a",
                bbox=dict(boxstyle="round,pad=0.25", fc="#f3eefb", ec="#ddd0f0", lw=0.7))

    ax.annotate("Heterodimer: Binding mediated\nby folding ($\\rho_{\\mathrm{partial}} = +0.05$)",
                xy=(1 + width, 0.08), xytext=(1.45, 0.42),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#2d6a4f"),
                fontsize=7.5, fontweight="bold", color="#155c37",
                bbox=dict(boxstyle="round,pad=0.25", fc="#eaf6ef", ec="#c8e6d3", lw=0.7))

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300, facecolor="#ffffff")

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
    ax1.text(-0.12, 1.05, "a", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="top")
    for model in models:
        sim = binder_data["filter_simulations"][model]["thresholds_simulation"]
        fnr = [s["interface_false_negative_rate"] * 100 for s in sim]
        ax1.plot(thresholds, fnr, marker="o", lw=1.8, markersize=5.5, label=model_labels[model], color=model_colors[model])

    ax1.axvline(20, color="#a83232", linestyle="--", lw=1.1, alpha=0.8)
    ax1.text(21.5, 52, "Standard Filter: Top 20%\n(Discards ~73% \u2013 77%\nof Interface Hits)",
             color="#8a1f2c", fontsize=7.8, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="#fbeef0", ec="#f2c9ce", lw=0.7))

    ax1.set_xlabel("PLM Likelihood Filter Cutoff (Top X%)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Interface False-Negative Rate (%)", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(45, 100)
    ax1.set_xticks(thresholds)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True)
    ax1.legend(loc="lower left", fontsize=8.0, edgecolor="#cbd5e1")

    # Panel B: Interface Depletion Rate
    ax2.text(-0.12, 1.05, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")
    for model in models:
        sim = binder_data["filter_simulations"][model]["thresholds_simulation"]
        dep = [s["interface_depletion_rate"] * 100 for s in sim]
        ax2.plot(thresholds, dep, marker="s", lw=1.8, markersize=5.5, label=model_labels[model], color=model_colors[model])

    ax2.axvline(20, color="#a83232", linestyle="--", lw=1.1, alpha=0.8)
    ax2.axhline(0, color="#64748b", linestyle=":", lw=0.8)
    ax2.set_xlabel("PLM Likelihood Filter Cutoff (Top X%)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Interface Depletion Rate vs. Non-Interface (%)", fontsize=9.5, fontweight="bold")
    ax2.set_xticks(thresholds)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=8.0, edgecolor="#cbd5e1")

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

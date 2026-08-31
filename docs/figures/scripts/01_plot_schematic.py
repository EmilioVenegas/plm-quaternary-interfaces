#!/usr/bin/env python3
"""Figure 1: Biophysical & Assay Schematic of the Monomer-Folding Confound in PPI Selections.

Output: docs/figures/01_expression_confound_schematic.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np
from diagram import (
    Ctx, card, paragraph, bullets, pill,
    PAPER, WASH, HAIR, INK, BODY, MUTED, DOT,
    SLATE, EMERALD, ROSE, INDIGO, VIOLET, AMBER,
    flow, descend
)

DOCS_FIGS_DIR = REPO_ROOT / "docs" / "figures"
STRUCT_ASSET_DIR = DOCS_FIGS_DIR / "assets"
STRUCT_LAYERS = ("mol_target", "mol_target_mut", "mol_partner_bound",
                 "mol_partner_away", "mol_coil_ensemble")
_STRUCT_CACHE: dict | None = None


def load_structure_layers() -> dict:
    """Load, common-crop and downsample the PyMOL cartoon layers."""
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

    target_w = 900
    scale = min(1.0, target_w / uw)
    out_size = (max(1, int(round(uw * scale))), max(1, int(round(uh * scale))))

    layers, boxes = {}, {}
    for n, im in raw.items():
        layers[n] = np.asarray(im.crop((ux0, uy0, ux1, uy1)).resize(out_size, Image.LANCZOS))
        x0, y0, x1, y1 = boxes_px[n]
        boxes[n] = ((x0 - ux0) / uw, (y0 - uy0) / uh, (x1 - ux0) / uw, (y1 - uy0) / uh)

    plain = np.asarray(raw["mol_target"]).astype(np.int16)
    marked = np.asarray(raw["mol_target_mut"]).astype(np.int16)
    colour_delta = np.abs(marked[..., :3] - plain[..., :3]).sum(axis=-1)
    diff_ys, diff_xs = np.nonzero((colour_delta > 40) & (marked[..., 3] > 8))
    if diff_xs.size:
        mut_center = (float(diff_xs.mean() - ux0) / uw, float(diff_ys.mean() - uy0) / uh)
    else:
        bx0, by0, bx1, by1 = boxes["mol_target"]
        mut_center = (bx1, (by0 + by1) / 2)

    meta_path = STRUCT_ASSET_DIR / "structure_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    mut_label = meta.get("mutation_site", "interface residue")
    if len(mut_label) > 3 and mut_label[:3].isalpha():
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


def plot_figure_1_schematic():
    """Figure 1: Biophysical and assay schematic of the monomer-folding confound in PPI selections."""
    WIDTH = 11.5
    HEIGHT = 6.12
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    C_INK = "#1a1a1a"
    C_BODY = "#3f3f46"
    C_MUTED = "#71717a"
    C_HAIR = "#d4d4d8"
    C_RULE = "#000000"
    C_PAPER = "#ffffff"

    C_POS = "#3f6b52"
    C_NEG = "#9b4444"
    C_BLUE_TEXT = "#3a5a80"

    BADGE_POS = "#4b7f5e"
    BADGE_CONF = "#a85454"
    BADGE_COLL = "#6b5b95"

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
        ax.add_patch(FancyBboxPatch((x, y - h), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=C_PAPER, edgecolor=C_RULE, lw=0.8, zorder=2))

        # Top header row: Case number on left, status badge on right (aligned vertically)
        ax.text(x + 0.10, y - 0.15, f"CASE {case_num}", fontsize=6.5, fontweight="bold",
                color=C_MUTED, family="Liberation Sans", va="center")
        ax.text(x + w - 0.10, y - 0.15, badge_text, fontsize=5.6, fontweight="bold",
                color=C_PAPER, ha="right", va="center", family="Liberation Sans",
                bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.10",
                          facecolor=badge_color, edgecolor="none"), zorder=6)

        # Second header row: Full-width card title
        ax.text(x + 0.10, y - 0.32, title, fontsize=7.1, fontweight="bold",
                color=C_INK, family="Liberation Sans", va="center")
        ax.plot([x + 0.10, x + w - 0.10], [y - 0.44, y - 0.44], color=C_HAIR, lw=0.7, zorder=3)
        ill_y_top = y - 0.50
        ill_h = 2.45

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

        struct = load_structure_layers()
        img_x0 = x + 0.08
        img_w = w - 0.16
        img_h = img_w / struct["aspect"]
        img_y0 = mem_y + 0.30
        img_y1 = img_y0 + img_h

        def box_data(name):
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

        mem_top = y_out + head_r
        stalk_x = x + 0.55
        ax.text(x + 0.23, mem_y + 0.21, "Surface\nanchor", ha="center", va="center",
                fontsize=5.0, color=C_MUTED, family="Liberation Sans", zorder=5)

        if mode == "misfolded":
            stub_top = mem_y + 0.30
            ax.plot([stalk_x, stalk_x], [mem_top, stub_top], color="#52525b", lw=2.0, zorder=4)
            ax.plot([stalk_x - 0.045, stalk_x + 0.045], [stub_top - 0.045, stub_top + 0.045],
                    color=C_NEG, lw=1.9, zorder=6)
            ax.plot([stalk_x - 0.045, stalk_x + 0.045], [stub_top + 0.045, stub_top - 0.045],
                    color=C_NEG, lw=1.9, zorder=6)
        else:
            ax.plot([stalk_x, stalk_x], [mem_top, ty0 + 0.05], color="#52525b", lw=2.0, zorder=4)

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

        rows_y_top = ill_y_top - ill_h - 0.05
        row_h = 0.24
        rows = [
            ("Monomer fold:", "Native fold (+)" if mode != "misfolded" else "Misfolded / cleared (\u2212)", C_POS if mode != "misfolded" else C_NEG),
            ("Complex formed:", "Bound (+)" if mode == "wt" else "Dissociated (\u2212)", C_POS if mode == "wt" else C_NEG),
            ("Abundance assay:", "High surface display (+)" if mode != "misfolded" else "No surface display (\u2212)", C_POS if mode != "misfolded" else C_NEG),
            ("Binding assay:", "Signal detected (+)" if mode == "wt" else "No signal (\u2212)", C_POS if mode == "wt" else C_NEG),
        ]
        for i, (k, v, vc) in enumerate(rows):
            ry = rows_y_top - i * row_h
            ax.text(x + 0.10, ry, k, fontsize=5.3, color=C_MUTED, family="Liberation Sans", va="top")
            ax.text(x + w - 0.10, ry, v, fontsize=5.4, fontweight="bold", color=vc,
                    family="Liberation Sans", ha="right", va="top")

        box_y = rows_y_top - len(rows) * row_h - 0.03
        box_h = 0.78
        ax.add_patch(FancyBboxPatch((x + 0.08, box_y - box_h), w - 0.16, box_h,
                                    boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor="#fafafa", edgecolor=C_HAIR, lw=0.6, zorder=2))
        ax.text(x + 0.12, box_y - 0.07, "EVALUATION ARTIFACT", fontsize=5.2, fontweight="bold",
                color=C_MUTED, family="Liberation Sans", va="top")

        verdicts = {
            "wt": (
                "True positive (concordant):",
                "Variant folds and binds target partner.\nBoth assays concordant; no ambiguity.",
            ),
            "misfolded": (
                "Confounded selection (spurious):",
                "Misfolded monomer cleared before display;\nassay scores loss-of-binding spuriously.",
            ),
            "interface": (
                "Interface failure (blindness):",
                "Monomer folds and displays normally,\nbut partner contact lost; PLM fails zero-shot.",
            ),
        }
        v_title, v_body = verdicts[mode]
        ax.text(x + 0.12, box_y - 0.20, v_title, fontsize=5.2, fontweight="bold",
                color=C_INK if mode == "wt" else C_NEG,
                family="Liberation Sans", va="top")
        ax.text(x + 0.12, box_y - 0.35, v_body, fontsize=4.8, color=C_BODY,
                family="Liberation Sans", va="top", linespacing=1.25)

    draw_case_card(x_cases[0], y_case_top, w_case, h_case,
                   1, "Wild-Type / Neutral Variant",
                   "CONCORDANT (+)", BADGE_POS, "wt")
    draw_case_card(x_cases[1], y_case_top, w_case, h_case,
                   2, "Core / Destabilising Variant",
                   "CONFOUNDED (\u2212)", BADGE_CONF, "misfolded")
    draw_case_card(x_cases[2], y_case_top, w_case, h_case,
                   3, "True Interface Disruptor",
                   "COLLAPSE (\u2212)", BADGE_COLL, "interface")

    # ---------------- Panel B: Causal DAG ----------------
    x_b = 7.50
    w_b = 3.60
    y_b_top = y_top_a
    panel_label(x_b - 0.10, HEIGHT - 0.22, "B \u00b7 CAUSAL STRUCTURE OF DISPLAY ASSAYS")

    h_dag = 2.40
    y_dag_top = y_b_top - 0.15

    def draw_node(x, y, w, h, title, subtitle, border_c, bg_c, text_c):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=bg_c, edgecolor=border_c, lw=1.0, zorder=4))
        ax.text(x, y + 0.08, title, fontsize=6.8, fontweight="bold",
                color=text_c, ha="center", va="center", family="Liberation Sans", zorder=5)
        ax.text(x, y - 0.10, subtitle, fontsize=5.2,
                color=C_MUTED, ha="center", va="center", family="Liberation Sans", zorder=5)

    def draw_arrow(x0, y0, x1, y1, text, text_pos, color, style="-", bold=False):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.28",
                                    lw=1.2 if not bold else 1.6, color=color,
                                    linestyle=style, shrinkA=3, shrinkB=3), zorder=3)
        if text:
            tx, ty = text_pos
            ax.text(tx, ty, text, fontsize=5.2, fontweight="bold" if bold else "normal",
                    color=color, ha="center", va="center", family="Liberation Sans",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=C_PAPER, edgecolor="none", alpha=0.9), zorder=6)

    nx_mut, ny_mut = x_b + 0.50, y_dag_top - 0.35
    nx_fold, ny_fold = x_b + 1.80, y_dag_top - 0.35
    nx_bind, ny_bind = x_b + 1.80, y_dag_top - 1.25
    nx_disp, ny_disp = x_b + 3.10, y_dag_top - 0.80

    draw_node(nx_mut, ny_mut, 0.90, 0.42, "Mutation", "Sequence change", C_HAIR, C_PAPER, C_INK)
    draw_node(nx_fold, ny_fold, 1.15, 0.42, "Monomer Folding", "Expression / stability", B_BLUE_BORDER, C_PAPER, B_BLUE_TEXT)
    draw_node(nx_bind, ny_bind, 1.15, 0.42, "Interface Binding", "Intermolecular affinity", B_ROSE_BORDER, C_PAPER, B_ROSE_TEXT)
    draw_node(nx_disp, ny_disp, 1.05, 0.48, "Display Readout", "FACS binding signal", B_AMBER_BORDER, B_AMBER_BG, B_AMBER_TEXT)

    draw_arrow(nx_mut + 0.45, ny_mut, nx_fold - 0.575, ny_fold, "\u03b2\u2081 (monomer)", (x_b + 1.15, ny_mut + 0.12), B_BLUE)
    draw_arrow(nx_mut + 0.35, ny_mut - 0.21, nx_bind - 0.575, ny_bind + 0.10, "\u03b2\u2082 (interface)", (x_b + 0.95, y_dag_top - 0.95), B_ROSE)
    draw_arrow(nx_fold + 0.575, ny_fold - 0.10, nx_disp - 0.525, ny_disp + 0.15, "Gating (confound)", (x_b + 2.50, y_dag_top - 0.45), B_BLUE, bold=True)
    draw_arrow(nx_bind + 0.575, ny_bind + 0.10, nx_disp - 0.525, ny_disp - 0.15, "True affinity", (x_b + 2.50, y_dag_top - 1.18), B_ROSE, bold=True)

    # ---------------- Panel C: Double Dissociation Matrix ----------------
    panel_label(x_b - 0.10, y_dag_top - 1.70, "C \u00b7 WITHIN-TARGET DOUBLE DISSOCIATION")

    y_mat_top = y_dag_top - 1.90
    w_mat = 3.60
    h_mat = 2.45

    def draw_diag_cell(x, y, w, h, title, subtitle, obs_txt, plm_txt, verdict_txt, color):
        ax.add_patch(FancyBboxPatch((x, y - h), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor=C_PAPER, edgecolor=C_HAIR, lw=0.6, zorder=2))
        ax.text(x + 0.10, y - 0.12, title, fontsize=6.2, fontweight="bold",
                color=C_INK, family="Liberation Sans", va="top")
        ax.text(x + 0.10, y - 0.25, subtitle, fontsize=5.0,
                color=C_MUTED, family="Liberation Sans", va="top")
        ax.text(x + 0.10, y - 0.42, obs_txt, fontsize=5.1,
                color=C_BODY, family="Liberation Sans", va="top")
        ax.text(x + 0.10, y - 0.56, plm_txt, fontsize=5.1,
                color=C_BODY, family="Liberation Sans", va="top")
        ax.text(x + w - 0.10, y - h + 0.12, verdict_txt, fontsize=5.4, fontweight="bold",
                color=color, ha="right", va="bottom", family="Liberation Sans")

    w_col = 1.68
    h_row = 1.05
    x_c0 = x_b + 0.05
    x_c1 = x_c0 + w_col + 0.12
    y_r0 = y_mat_top - 0.15
    y_r1 = y_r0 - h_row - 0.10

    draw_diag_cell(x_c0, y_r0, w_col, h_row,
                   "Core / Fold Disruptor", "(Hydrophobic core)",
                   "Observed: Low Abund, Low Bind",
                   "PLM Score: Severe penalty",
                   "CONCORDANT (+)", C_POS)
    draw_diag_cell(x_c1, y_r0, w_col, h_row,
                   "Surface Neutral", "(Solvent-exposed)",
                   "Observed: High Abund, High Bind",
                   "PLM Score: Neutral / benign",
                   "CONCORDANT (+)", C_POS)
    draw_diag_cell(x_c0, y_r1, w_col, h_row,
                   "Interface Contact Disruptor", "(Epitope contact)",
                   "Observed: High Abund, Low Bind",
                   "PLM Score: Neutral / benign",
                   "ZERO-SHOT COLLAPSE (\u2212)", C_NEG)
    draw_diag_cell(x_c1, y_r1, w_col, h_row,
                   "Abundance-Destabilised Binder", "(Fold-compromised)",
                   "Observed: Low Abund, Low Bind",
                   "PLM Score: Severe penalty",
                   "SPURIOUS SIGNAL (\u2212)", C_NEG)

    out_path = DOCS_FIGS_DIR / "01_expression_confound_schematic.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    print("=" * 78)
    print("Generating Figure 1: Expression Confound Schematic")
    print("=" * 78)
    plot_figure_1_schematic()
    print("=" * 78)


if __name__ == "__main__":
    main()

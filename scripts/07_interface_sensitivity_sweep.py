#!/usr/bin/env python3
"""Stage 7: Interface Definition Sensitivity and Structural Sweep.

Evaluates the robustness of the core findings across a two-dimensional grid of
structural interface cutoffs:
  - delta-SASA thresholds: [2.0, 5.0, 10.0, 15.0, 20.0] Angstrom^2
  - min-distance thresholds: [3.5, 4.0, 4.5, 5.0, 6.0] Angstrom

For each combination (dsasa, dist):
  1. Partitions variants into Interface (dsasa >= threshold or min_dist <= threshold),
     Core (non-interface & rSASA < 0.20), and Surface (non-interface & rSASA >= 0.20).
  2. For each model arm (esm2-650m, esm2-3b, esmc-600m, esmc-6b), computes:
     - Interface variant count N_int
     - Interface Spearman rho(PLM, Abundance)
     - Interface Spearman rho(PLM, Binding)
     - Interface Partial Spearman rho(PLM, Binding | Abundance)
     - Three-way interaction coefficient beta_(PLM x Binding x Interface) (beta_7)
  3. Verifies that beta_7 < 0 and rho_partial <= 0 remain strictly robust across all cutoffs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from plmppi.stats import (
    fit_clustered_ols,
    partial_spearman,
    prepare_analysis_frame,
)

DEFAULT_DSASA_THRESHOLDS: list[float] = [2.0, 5.0, 10.0, 15.0, 20.0]
DEFAULT_DIST_THRESHOLDS: list[float] = [3.5, 4.0, 4.5, 5.0, 6.0]
DEFAULT_RSASA_THRESHOLD: float = 0.20
DEFAULT_ARMS: list[str] = ["esm2-650m", "esm2-3b", "esmc-600m", "esmc-6b"]

FEATURE_NAMES: list[str] = [
    "Intercept",
    "PLM",
    "Binding",
    "Interface",
    "PLM:Binding",
    "PLM:Interface",
    "Binding:Interface",
    "PLM:Binding:Interface",
]


def detect_arms(df: pd.DataFrame) -> list[str]:
    """Detects available model score arms from dataframe columns."""
    zs = {c.removeprefix("zeroshot_") for c in df.columns if c.startswith("zeroshot_")}
    return sorted(zs) if zs else list(DEFAULT_ARMS)


def classify_compartments(
    df: pd.DataFrame,
    dsasa_threshold: float,
    dist_threshold: float,
    rsasa_threshold: float = DEFAULT_RSASA_THRESHOLD,
) -> pd.DataFrame:
    """Reclassifies variants into Interface, Core, and Surface based on geometric cutoffs.

    - Interface: delta-SASA >= dsasa_threshold OR min_distance <= dist_threshold
    - Core: non-interface AND rSASA < rsasa_threshold
    - Surface: non-interface AND rSASA >= rsasa_threshold
    """
    dsasa_col = "dsasa" if "dsasa" in df.columns else "delta_sasa"
    dist_col = "min_dist" if "min_dist" in df.columns else "min_distance"
    rsasa_col = "rsasa"

    if dsasa_col not in df.columns:
        raise KeyError(f"Neither 'dsasa' nor 'delta_sasa' found in columns: {df.columns.tolist()}")
    if dist_col not in df.columns:
        raise KeyError(f"Neither 'min_dist' nor 'min_distance' found in columns: {df.columns.tolist()}")
    if rsasa_col not in df.columns:
        raise KeyError(f"'rsasa' not found in columns: {df.columns.tolist()}")

    is_int = (df[dsasa_col] >= dsasa_threshold) | (df[dist_col] <= dist_threshold)
    is_core = (~is_int) & (df[rsasa_col] < rsasa_threshold)

    out = df.copy()
    out["compartment"] = np.where(is_int, "Interface", np.where(is_core, "Core", "Surface"))
    return out


def compute_arm_metrics(df_classified: pd.DataFrame, arm: str) -> dict[str, Any]:
    """Computes interface correlations, partial correlation, and three-way interaction OLS for an arm."""
    zs_col = f"zeroshot_{arm}"
    if zs_col not in df_classified.columns:
        raise KeyError(f"Column {zs_col} not found in dataframe")

    sub_int = df_classified[df_classified["compartment"] == "Interface"].dropna(
        subset=[zs_col, "dms_score_abundance", "dms_score_binding"]
    )
    n_int = int(len(sub_int))

    if n_int < 4:
        return {
            "n_interface": n_int,
            "rho_abundance": float("nan"),
            "p_abundance": float("nan"),
            "rho_binding": float("nan"),
            "p_binding": float("nan"),
            "rho_partial": float("nan"),
            "beta_three_way": float("nan"),
            "se_three_way": float("nan"),
            "t_stat_three_way": float("nan"),
            "p_three_way": float("nan"),
        }

    rho_ab, p_ab = stats.spearmanr(sub_int[zs_col], sub_int["dms_score_abundance"])
    rho_bi, p_bi = stats.spearmanr(sub_int[zs_col], sub_int["dms_score_binding"])
    rho_part = partial_spearman(
        sub_int[zs_col], sub_int["dms_score_binding"], sub_int["dms_score_abundance"]
    )

    # Fit clustered three-way interaction OLS
    frame = prepare_analysis_frame(df_classified, arm=arm)
    plm = frame["plm_z"].to_numpy()
    bind = frame["is_binding"].to_numpy()
    inter = frame["is_interface"].to_numpy()
    y = frame["dms_z"].to_numpy()
    clusters = frame["system"].to_numpy()
    N = len(frame)

    X = np.column_stack([
        np.ones(N),
        plm,
        bind,
        inter,
        plm * bind,
        plm * inter,
        bind * inter,
        plm * bind * inter,
    ])

    ols_res = fit_clustered_ols(X, y, clusters, FEATURE_NAMES)
    param = ols_res["params"]["PLM:Binding:Interface"]

    return {
        "n_interface": n_int,
        "rho_abundance": float(round(rho_ab, 4)),
        "p_abundance": float(p_ab),
        "rho_binding": float(round(rho_bi, 4)),
        "p_binding": float(p_bi),
        "rho_partial": float(round(rho_part, 4)),
        "beta_three_way": float(round(param["coef"], 4)),
        "se_three_way": float(round(param["se"], 4)),
        "t_stat_three_way": float(round(param["t_stat"], 4)),
        "p_three_way": float(param["p_val"]),
    }


def run_sensitivity_sweep(
    df: pd.DataFrame,
    dsasa_thresholds: list[float] | None = None,
    dist_thresholds: list[float] | None = None,
    arms: list[str] | None = None,
    rsasa_threshold: float = DEFAULT_RSASA_THRESHOLD,
) -> dict[str, Any]:
    """Runs a 2D parameter grid sweep over interface cutoff definitions."""
    if dsasa_thresholds is None:
        dsasa_thresholds = list(DEFAULT_DSASA_THRESHOLDS)
    if dist_thresholds is None:
        dist_thresholds = list(DEFAULT_DIST_THRESHOLDS)
    if arms is None:
        arms = detect_arms(df)

    grid_entries: list[dict[str, Any]] = []

    for dsasa in dsasa_thresholds:
        for dist in dist_thresholds:
            df_classified = classify_compartments(
                df,
                dsasa_threshold=dsasa,
                dist_threshold=dist,
                rsasa_threshold=rsasa_threshold,
            )

            n_int = int((df_classified["compartment"] == "Interface").sum())
            n_core = int((df_classified["compartment"] == "Core").sum())
            n_surf = int((df_classified["compartment"] == "Surface").sum())

            cell_models: dict[str, Any] = {}
            for arm in arms:
                cell_models[arm] = compute_arm_metrics(df_classified, arm=arm)

            grid_entries.append(
                {
                    "delta_sasa_threshold": float(dsasa),
                    "distance_threshold": float(dist),
                    "n_interface": n_int,
                    "n_core": n_core,
                    "n_surface": n_surf,
                    "models": cell_models,
                }
            )

    # Compute arm-level summaries
    summary_by_arm: dict[str, Any] = {}
    all_arms_beta_negative = True
    all_arms_rho_nonpositive = True

    for arm in arms:
        betas = [entry["models"][arm]["beta_three_way"] for entry in grid_entries]
        rho_parts = [entry["models"][arm]["rho_partial"] for entry in grid_entries]

        all_beta_neg = all(b < 0 for b in betas if not np.isnan(b))
        all_rho_nonpos = all(r <= 0 for r in rho_parts if not np.isnan(r))

        if not all_beta_neg:
            all_arms_beta_negative = False
        if not all_rho_nonpos:
            all_arms_rho_nonpositive = False

        summary_by_arm[arm] = {
            "beta_three_way_min": float(np.min(betas)),
            "beta_three_way_max": float(np.max(betas)),
            "beta_three_way_mean": float(round(float(np.mean(betas)), 4)),
            "all_beta_negative": bool(all_beta_neg),
            "rho_partial_min": float(np.min(rho_parts)),
            "rho_partial_max": float(np.max(rho_parts)),
            "rho_partial_mean": float(round(float(np.mean(rho_parts)), 4)),
            "all_rho_partial_nonpositive": bool(all_rho_nonpos),
        }

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_total_variants": int(len(df)),
            "delta_sasa_thresholds": dsasa_thresholds,
            "distance_thresholds": dist_thresholds,
            "rsasa_threshold": rsasa_threshold,
            "arms": arms,
        },
        "grid": grid_entries,
        "summary": summary_by_arm,
        "robustness_verdict": {
            "all_beta_negative_across_all_cutoffs": bool(all_arms_beta_negative),
            "all_rho_partial_nonpositive_across_all_cutoffs": bool(all_arms_rho_nonpositive),
            "conclusion": (
                "The finding of systematic PLM interface failure (beta_7 < 0, partial rho <= 0) "
                "is invariant to geometric cutoff choices."
                if all_arms_beta_negative and all_arms_rho_nonpositive
                else "Sensitivity detected: Cutoff choices impact interaction signs."
            ),
        },
    }


def format_summary_matrix(results: dict[str, Any]) -> str:
    """Formats ASCII summary matrices for terminal display."""
    lines = []
    meta = results["metadata"]
    dsasa_threshs = meta["delta_sasa_thresholds"]
    dist_threshs = meta["distance_thresholds"]
    arms = meta["arms"]
    grid = results["grid"]

    # Index grid by (dsasa, dist)
    grid_map = {(entry["delta_sasa_threshold"], entry["distance_threshold"]): entry for entry in grid}

    lines.append("=" * 80)
    lines.append("INTERFACE DEFINITION SENSITIVITY SWEEP SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Grid: {len(dsasa_threshs)} dsasa cutoffs x {len(dist_threshs)} distance cutoffs = {len(grid)} configurations")
    lines.append(f"Total variants evaluated: {meta['n_total_variants']}")

    for arm in arms:
        summary = results["summary"][arm]
        lines.append("\n" + "-" * 80)
        lines.append(f"Model Arm: {arm}")
        lines.append(f"  beta_7 range       : [{summary['beta_three_way_min']:+.4f}, {summary['beta_three_way_max']:+.4f}] (mean = {summary['beta_three_way_mean']:+.4f}) -> all < 0: {summary['all_beta_negative']}")
        lines.append(f"  rho_partial range  : [{summary['rho_partial_min']:+.4f}, {summary['rho_partial_max']:+.4f}] (mean = {summary['rho_partial_mean']:+.4f}) -> all <= 0: {summary['all_rho_partial_nonpositive']}")

        # 1. beta_7 Matrix
        lines.append("\n  Three-way interaction coefficient beta_7 (PLM x Binding x Interface):")
        header = "    dsasa \\ dist |" + "".join(f"  {d:4.1f} A   |" for d in dist_threshs)
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for dsasa in dsasa_threshs:
            row = f"     {dsasa:4.1f} A^2   |"
            for dist in dist_threshs:
                b = grid_map[(dsasa, dist)]["models"][arm]["beta_three_way"]
                row += f"  {b:+.4f}  |"
            lines.append(row)

        # 2. rho_partial Matrix
        lines.append("\n  Partial correlation rho(PLM, Binding | Abundance) at Interface:")
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for dsasa in dsasa_threshs:
            row = f"     {dsasa:4.1f} A^2   |"
            for dist in dist_threshs:
                rp = grid_map[(dsasa, dist)]["models"][arm]["rho_partial"]
                row += f"  {rp:+.4f}  |"
            lines.append(row)

    verdict = results["robustness_verdict"]
    lines.append("\n" + "=" * 80)
    lines.append("OVERALL ROBUSTNESS VERDICT:")
    lines.append(f"  All beta_7 < 0 across all cutoffs and arms        : {verdict['all_beta_negative_across_all_cutoffs']}")
    lines.append(f"  All rho_partial <= 0 across all cutoffs and arms   : {verdict['all_rho_partial_nonpositive_across_all_cutoffs']}")
    lines.append(f"  Conclusion: {verdict['conclusion']}")
    lines.append("=" * 80)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run 2D interface sensitivity sweep across delta-SASA and distance thresholds."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=REPO_ROOT / "results" / "scores.csv",
        help="Input CSV with model scores and structural geometry metrics",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "interface_sensitivity_sweep.json",
        help="Output JSON path for sensitivity sweep results",
    )
    parser.add_argument(
        "--dsasa",
        type=float,
        nargs="+",
        default=DEFAULT_DSASA_THRESHOLDS,
        help="Delta-SASA thresholds (Angstrom^2)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        nargs="+",
        default=DEFAULT_DIST_THRESHOLDS,
        help="Distance thresholds (Angstrom)",
    )
    parser.add_argument(
        "--rsasa",
        type=float,
        default=DEFAULT_RSASA_THRESHOLD,
        help="Relative SASA threshold for Core vs Surface partitioning (default: 0.20)",
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=None,
        help="Model arms to evaluate (default: auto-detected)",
    )
    args = parser.parse_args(argv)

    if not args.scores.exists():
        print(f"Error: scores file {args.scores} does not exist", file=sys.stderr)
        return 1

    df_scores = pd.read_csv(args.scores)
    arms = args.arms if args.arms is not None else detect_arms(df_scores)
    print(
        f"[read] {args.scores}: {len(df_scores)} variants, sweeping {len(args.dsasa)}x{len(args.distance)} grid for arms {arms}",
        flush=True,
    )

    results = run_sensitivity_sweep(
        df_scores,
        dsasa_thresholds=args.dsasa,
        dist_thresholds=args.distance,
        arms=arms,
        rsasa_threshold=args.rsasa,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[write] saved sensitivity sweep results to {args.out}", flush=True)

    summary_text = format_summary_matrix(results)
    print(summary_text, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

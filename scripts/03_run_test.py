#!/usr/bin/env python3
"""Stage 3: Statistical hypothesis testing for the PLM PPI interface failure study."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from plmppi.stats import ALPHA, prepare_analysis_frame, run_three_way_interaction_test


def detect_arms(df: pd.DataFrame) -> list[str]:
    zs = {c.removeprefix("zeroshot_") for c in df.columns if c.startswith("zeroshot_")}
    return sorted(zs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run statistical hypothesis tests on scored variants."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=REPO_ROOT / "results" / "scores.csv",
        help="Input CSV with model scores",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=10000,
        help="Number of permutation iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutations",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Output directory for test JSON artifacts",
    )
    args = parser.parse_args(argv)

    if not args.scores.exists():
        print(f"Error: scores file {args.scores} does not exist", file=sys.stderr)
        return 1

    df_scores = pd.read_csv(args.scores)
    arms = detect_arms(df_scores)
    print(f"[read] {args.scores}: {len(df_scores)} observations across arms {arms}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for arm in arms:
        print(f"\n{'='*78}\n[test] running statistical analysis for arm: {arm}\n{'='*78}", flush=True)
        analysis_frame = prepare_analysis_frame(df_scores, arm=arm)
        res = run_three_way_interaction_test(analysis_frame, n_perm=args.n_perm, seed=args.seed)
        res["arm"] = arm
        res["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

        all_results[arm] = res

        out_path = args.out_dir / f"test_{arm}.json"
        with out_path.open("w") as f:
            json.dump(res, f, indent=2)
        print(f"[write] saved test results to {out_path}", flush=True)

        # Print report
        print(f"\nArm: {arm}")
        print(f"Primary interaction term: {res['primary_term']}")
        print(f"  beta (three-way)      : {res['beta_three_way']:+.4f}")
        print(f"  p-value (clustered)   : {res['p_clustered']:.6e}")
        print(f"  p-value (permutation) : {res['p_permutation']:.6e}")
        print(f"  p-value (wild boot)   : {res['p_wild_bootstrap']:.6e}")
        print(f"  Verdict (alpha={ALPHA}): {res['verdict']}")

        print("\n--- Econometric Robustness: Statistical Significance ---")
        print(f"{'Method':<36} {'Test Stat':<12} {'p-value':<12} {'Inference'}")
        print("-" * 75)
        crv_t = res["ols_summary"]["params"]["PLM:Binding:Interface"]["t_stat"]
        print(f"{'Cluster-Robust OLS (CRV1)':<36} {crv_t:+11.3f} {res['p_clustered']:<12.4e} {'p < alpha' if res['p_clustered'] < ALPHA else 'n.s.'}")
        wcb_t = res["wild_cluster_bootstrap"]["t_stat_orig"]
        print(f"{'Webb 6-Point Wild Cluster Bootstrap':<36} {wcb_t:+11.3f} {res['p_wild_bootstrap']:<12.4e} {'p < alpha' if res['p_wild_bootstrap'] < ALPHA else 'n.s.'}")
        print(f"{'Stratified Permutation Test':<36} {'N/A':<12} {res['p_permutation']:<12.4e} {'p < alpha' if res['p_permutation'] < ALPHA else 'n.s.'}")

        print("\n--- System Fixed-Effects OLS Model ---")
        print(f"{'Parameter':<28} {'Coefficient':<12} {'SE':<10} {'t-stat':<10} {'p-value':<12}")
        print("-" * 75)
        for p_name, p_data in res["fixed_effects_summary"]["params"].items():
            print(f"{p_name:<28} {p_data['coef']:+11.4f} {p_data['se']:<10.4f} {p_data['t_stat']:+9.3f} {p_data['p_val']:<12.4e}")

        print("\n--- Leave-One-System-Out (LOSO) Jackknife Sensitivity ---")
        print(f"{'Omitted System':<20} {'N_obs':<8} {'Beta (3-way)':<14} {'SE':<10} {'p-val':<12} {'IF_Abund rho':<14} {'IF_Bind rho':<14}")
        print("-" * 95)
        for sys_id, loso_data in res["leave_one_system_out"].items():
            print(
                f"{sys_id:<20} {loso_data['n_obs']:<8d} {loso_data['beta_three_way']:+13.4f} "
                f"{loso_data['se']:<10.4f} {loso_data['p_val']:<12.4e} "
                f"{loso_data['rho_interface_abundance']:+13.3f} {loso_data['rho_interface_binding']:+13.3f}"
            )

        print("\n--- Subgroup Correlations (PLM vs DMS) ---")
        print(f"{'Subgroup':<22} {'N':<6} {'Spearman rho':<14} {'p-val':<12} {'Pearson r':<12} {'p-val':<12}")
        print("-" * 75)
        for k, v in res["subgroup_correlations"].items():
            print(f"{k:<22} {v['n']:<6d} {v['spearman_rho']:+13.3f} {v['spearman_p']:<12.4e} {v['pearson_r']:+11.3f} {v['pearson_p']:<12.4e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

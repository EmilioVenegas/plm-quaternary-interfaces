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
        print(f"  Verdict (alpha={ALPHA}): {res['verdict']}")

        print("\nSubgroup correlations (PLM vs DMS):")
        for k, v in res["subgroup_correlations"].items():
            print(f"  {k:20s}: n={v['n']:5d}, Spearman rho={v['spearman_rho']:+.3f}, Pearson r={v['pearson_r']:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Stage 11: Dual-Scoring Mitigation Strategy and Pareto Frontier Analysis.

Evaluates the composite design objective:
  Score_dual(alpha) = z(Delta_log_P_ProteinMPNN_complex) + alpha * z(Delta_log_P_ESM2_monomer)
across alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 100.0].

Quantifies the Pareto frontier for computational binder design:
  - Interface False Negative Rate (FNR) on affinity-enhancing mutations
  - Monomer Expressibility Recovery (% of selected variants with positive abundance DMS)
  - Interface Binding correlation rho(Dual, Binding)
  - Interface Abundance correlation rho(Dual, Abundance)

Outputs:
  results/dual_scoring_mitigation.json: Full sweep and Pareto frontier metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from plmppi.stats import evaluate_dual_scoring_frontier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run dual-scoring mitigation strategy and Pareto frontier analysis."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=REPO_ROOT / "results" / "scores.csv",
        help="Path to PLM scores CSV",
    )
    parser.add_argument(
        "--mpnn-scores",
        type=Path,
        default=REPO_ROOT / "results" / "proteinmpnn_scores.csv",
        help="Path to ProteinMPNN scores CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "dual_scoring_mitigation.json",
        help="Output path for dual scoring mitigation JSON",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=0.20,
        help="Filter selection threshold (default: 0.20 for top 20%)",
    )
    args = parser.parse_args(argv)

    if not args.scores.exists():
        print(f"Error: {args.scores} not found", file=sys.stderr)
        return 1
    if not args.mpnn_scores.exists():
        print(f"Error: {args.mpnn_scores} not found", file=sys.stderr)
        return 1

    df_scores = pd.read_csv(args.scores)
    df_mpnn = pd.read_csv(args.mpnn_scores)

    df_combined = df_scores.copy()
    df_combined["zeroshot_proteinmpnn"] = df_mpnn["zeroshot_proteinmpnn"]

    print(f"[read] loaded {len(df_combined)} variants for dual-scoring analysis", flush=True)

    frontier = evaluate_dual_scoring_frontier(
        df=df_combined,
        top_pct=args.top_pct,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(frontier, f, indent=2)
    print(f"[write] wrote Pareto frontier results to {args.out}", flush=True)

    print("\n" + "=" * 95)
    print(f"{'Strategy / Alpha':<32} {'Top 20% Int FNR':<18} {'Hits Retained':<16} {'Expressibility %':<18} {'Utility Score'}")
    print("=" * 95)
    for entry in frontier["alpha_sweep"]:
        print(
            f"{entry['label']:<32} {entry['interface_fnr_pct']:>6.1f}%             "
            f"{entry['interface_hits_retained']:>4d}/{entry['interface_total_hits']}        "
            f"{entry['monomer_expressibility_pct']:>6.1f}%             "
            f"{entry['utility_score']:>6.2f}"
        )

    print(f"\n[conclusion] {frontier['conclusion']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

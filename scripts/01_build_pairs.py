#!/usr/bin/env python3
"""Stage 1: Build paired variant observation table with structural compartments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from plmppi.data import (
    PRIMARY_SYSTEMS,
    audit_provenance_summary,
    build_all_pairs,
    get_matched_cohort,
    load_reference,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build paired DMS observations with structural compartment assignments."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "pairs.csv",
        help="Output CSV path for paired variants",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "results" / "pairs_summary.json",
        help="Output JSON path for summary statistics",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=REPO_ROOT / "data" / "proteingym" / "DMS_substitutions_ref.csv",
        help="ProteinGym reference CSV path",
    )
    parser.add_argument(
        "--bulk-dir",
        type=Path,
        default=REPO_ROOT / "data" / "proteingym" / "bulk" / "DMS_ProteinGym_substitutions",
        help="Directory with ProteinGym assay CSVs",
    )
    parser.add_argument(
        "--struct-dir",
        type=Path,
        default=REPO_ROOT / "data" / "structures",
        help="Directory with complex PDB structures",
    )
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    print("[stage 1] building paired dataset across 5 primary PPI systems...", flush=True)
    df_pairs = build_all_pairs(
        systems=PRIMARY_SYSTEMS,
        ref_path=args.ref,
        bulk_dir=args.bulk_dir,
        struct_dir=args.struct_dir,
    )

    df_pairs.to_csv(args.out, index=False)
    print(f"[stage 1] wrote {len(df_pairs)} paired variants to {args.out}", flush=True)

    # Compute summary breakdown
    per_system = {}
    for sys_id, group in df_pairs.groupby("system"):
        counts = group["compartment"].value_counts().to_dict()
        per_system[sys_id] = {
            "n_variants": int(len(group)),
            "n_positions": int(group["position"].nunique()),
            "compartments": {k: int(v) for k, v in counts.items()},
        }

    total_counts = df_pairs["compartment"].value_counts().to_dict()
    summary = {
        "n_total_variants": int(len(df_pairs)),
        "n_systems": int(df_pairs["system"].nunique()),
        "total_compartments": {k: int(v) for k, v in total_counts.items()},
        "per_system": per_system,
    }

    # Provenance summary audit
    prov_df = audit_provenance_summary()
    summary["provenance_audit"] = prov_df.to_dict(orient="records")
    summary["cohorts"] = {
        "strict_matched": [rec.system.system_id for rec in get_matched_cohort(include_conditional=False)],
        "conditional_matched": [rec.system.system_id for rec in get_matched_cohort(include_conditional=True)],
        "legacy_all": [s.system_id for s in PRIMARY_SYSTEMS],
    }

    with args.summary.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stage 1] wrote summary to {args.summary}", flush=True)

    print("\n[summary breakdown]")
    print(f"Total paired variants: {len(df_pairs)}")
    for comp, count in total_counts.items():
        print(f"  {comp:12s}: {count:6d} ({count / len(df_pairs) * 100:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

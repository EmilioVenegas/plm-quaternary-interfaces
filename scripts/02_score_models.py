#!/usr/bin/env python3
"""Stage 2: Zero-shot masked marginal scoring across ESM2 and ESMC model arms."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from plmppi.data import PRIMARY_SYSTEMS, load_reference
from plmppi.models import MODELS, load_model
from plmppi.scoring import score_variant_dataframe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score paired variants with protein language models."
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=REPO_ROOT / "results" / "pairs.csv",
        help="Input CSV with paired variants",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "scores.csv",
        help="Output CSV with scores",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="esm2-650m",
        help="Comma-separated list of model arms to score (e.g. esm2-650m,esm2-3b,esmc-600m,esmc-6b)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for masked forward passes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Carry over existing completed arm columns from output file",
    )
    args = parser.parse_args(argv)

    if not args.variants.exists():
        print(f"Error: variants file {args.variants} does not exist", file=sys.stderr)
        return 1

    df_variants = pd.read_csv(args.variants)
    print(f"[read] {args.variants}: {len(df_variants)} variants across {df_variants['system'].nunique()} systems", flush=True)

    # Initialize or load output table
    if args.resume and args.out.exists():
        df_out = pd.read_csv(args.out)
        print(f"[resume] loaded existing {args.out} with {len(df_out)} rows", flush=True)
    else:
        df_out = df_variants.copy()

    ref_df = load_reference()
    # Map system_id -> target_seq
    system_seqs: dict[str, str] = {}
    for sys_obj in PRIMARY_SYSTEMS:
        match = ref_df.query("DMS_id == @sys_obj.dms_abundance")
        if not match.empty:
            system_seqs[sys_obj.system_id] = match.iloc[0]["target_seq"]

    target_arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    for arm in target_arms:
        zs_col = f"zeroshot_{arm}"
        lw_col = f"logp_wt_{arm}"

        if args.resume and zs_col in df_out.columns and lw_col in df_out.columns:
            if df_out[zs_col].notna().all():
                print(f"[{arm}] already complete in {args.out}, skipping (resume active)", flush=True)
                continue

        print(f"\n[{arm}] loading model...", flush=True)
        t0 = time.time()
        model, tok = load_model(arm)
        load_time = time.time() - t0
        print(f"[{arm}] loaded in {load_time:.1f}s", flush=True)

        arm_zs = []
        arm_lw = []

        # Score per system
        for sys_id, group in df_variants.groupby("system", sort=False):
            if sys_id not in system_seqs:
                raise KeyError(f"Target sequence not found for system {sys_id}")
            t_seq = system_seqs[sys_id]

            t_start = time.time()
            zs, lw = score_variant_dataframe(
                model=model,
                tok=tok,
                df=group,
                target_seq=t_seq,
                batch_size=args.batch_size,
            )
            elapsed = time.time() - t_start
            n_pos = group["position"].nunique()
            print(f"[{arm}] {sys_id}: {n_pos} positions, {len(group)} variants in {elapsed:.1f}s ({elapsed/max(1, n_pos):.3f}s/pos)", flush=True)

            arm_zs.extend(zs)
            arm_lw.extend(lw)

        df_out[zs_col] = arm_zs
        df_out[lw_col] = arm_lw

        # Save checkpoint
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.out, index=False)
        print(f"[write] updated {args.out} with columns {zs_col}, {lw_col}", flush=True)

    print(f"\n[done] all target arms scored successfully: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

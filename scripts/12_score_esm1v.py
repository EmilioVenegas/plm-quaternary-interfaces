#!/usr/bin/env python3
"""Stage 12: Score paired variants with ESM-1v historical reference model.

Evaluates Meta's ESM-1v (facebook/esm1v_t33_650M_UR90S_1, Meier et al., 2021)
in zero-shot sequence masked-marginal mode across all 5 quaternary interface systems (N = 10,643 variants):
  - SARS-CoV-2 RBD / ACE2 (6M0J)
  - KRAS / DARPin K55 (6H46)
  - HLA-A2 / TAPBPR (5OPI)
  - GB1 / IgG-Fc (1FCC)
  - p53 homotetramer (1OLG)

Outputs:
  results/esm1v_scores.csv: pairs.csv + zeroshot_esm1v column
  results/esm1v/test_esm1v.json: 3-way interaction test via scripts/03_run_test.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from plmppi.data import PRIMARY_SYSTEMS, load_reference
from plmppi.scoring import score_variant_dataframe


def score_pairs_with_esm1v(
    df_pairs: pd.DataFrame,
    model: EsmForMaskedLM,
    tok: AutoTokenizer,
    ref_df: pd.DataFrame | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Maps ESM-1v masked marginal log-odds scores to all variants in df_pairs."""
    if ref_df is None:
        ref_df = load_reference()

    zs_scores: dict[tuple[str, int, str, str], float] = {}
    logp_wt_scores: dict[tuple[str, int, str, str], float] = {}

    for sys_obj in PRIMARY_SYSTEMS:
        match = ref_df.query("DMS_id == @sys_obj.dms_abundance")
        target_seq = match.iloc[0]["target_seq"]

        sys_pairs = df_pairs[df_pairs["system"] == sys_obj.system_id].copy()
        positions = sys_pairs["position"].astype(int)
        min_p, max_p = int(positions.min()), int(positions.max())

        t0 = time.time()
        # For long sequences exceeding 1000 aa (e.g. Spike 1273 aa with RBD 333-526), slice domain to fit position embeddings
        if len(target_seq) > 1000 and (max_p - min_p + 1) < 400:
            offset = min_p - 1
            sub_seq = target_seq[offset:max_p]
            sub_df = sys_pairs.copy()
            sub_df["position"] = sub_df["position"] - offset
            zs, lp = score_variant_dataframe(model, tok, sub_df, sub_seq, batch_size=batch_size)
        else:
            zs, lp = score_variant_dataframe(model, tok, sys_pairs, target_seq, batch_size=batch_size)
        elapsed = time.time() - t0

        print(
            f"[{sys_obj.system_id}] {len(sys_pairs)} variants scored in {elapsed:.2f}s "
            f"({len(sys_pairs)/elapsed:.1f} var/s)",
            flush=True,
        )

        for (idx, r), z, l in zip(sys_pairs.iterrows(), zs, lp):
            zs_scores[(sys_obj.system_id, int(r["position"]), r["wt"], r["mut"])] = float(z)
            logp_wt_scores[(sys_obj.system_id, int(r["position"]), r["wt"], r["mut"])] = float(l)

    df_out = df_pairs.copy()
    df_out["zeroshot_esm1v"] = df_out.apply(
        lambda r: zs_scores.get((r["system"], int(r["position"]), r["wt"], r["mut"]), np.nan),
        axis=1,
    )
    df_out["logp_wt_esm1v"] = df_out.apply(
        lambda r: logp_wt_scores.get((r["system"], int(r["position"]), r["wt"], r["mut"]), np.nan),
        axis=1,
    )

    return df_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score paired variants with ESM-1v historical reference model."
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=REPO_ROOT / "results" / "pairs.csv",
        help="Input variants pairs CSV",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="facebook/esm1v_t33_650M_UR90S_1",
        help="HuggingFace model repository for ESM-1v",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "esm1v_scores.csv",
        help="Output CSV for ESM-1v scores",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "esm1v",
        help="Output directory for test_esm1v.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size of masked positions",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=10000,
        help="Number of permutations for three-way test",
    )
    args = parser.parse_args(argv)

    if not args.variants.exists():
        print(f"Error: {args.variants} not found", file=sys.stderr)
        return 1

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[device] loading ESM-1v ({args.repo}) on {device}...", flush=True)

    tok = AutoTokenizer.from_pretrained(args.repo)
    model = EsmForMaskedLM.from_pretrained(args.repo, torch_dtype=torch.float16).to(device).eval()

    df_pairs = pd.read_csv(args.variants)
    df_scored = score_pairs_with_esm1v(
        df_pairs=df_pairs,
        model=model,
        tok=tok,
        batch_size=args.batch_size,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(args.out, index=False)
    print(f"[write] wrote {len(df_scored)} scores to {args.out}", flush=True)

    # Run Stage 3 test pipeline on ESM-1v scores
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "03_run_test.py"),
        "--scores",
        str(args.out),
        "--out-dir",
        str(args.out_dir),
        "--n-perm",
        str(args.n_perm),
    ]
    print(f"\n[run] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=True)

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

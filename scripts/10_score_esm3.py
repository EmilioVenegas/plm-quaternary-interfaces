#!/usr/bin/env python3
"""Stage 10: Score paired variants with ESM3-1.4B sequence mode.

Evaluates EvolutionaryScale's ESM3-sm-open-v1 (1.4B parameters, Hayes et al., Science 2024)
in zero-shot sequence masked-marginal mode across all 5 quaternary interface systems (N = 10,643 variants):
  - SARS-CoV-2 RBD / ACE2 (6M0J)
  - KRAS / DARPin K55 (6H46)
  - HLA-A2 / TAPBPR (5OPI)
  - GB1 / IgG-Fc (1FCC)
  - p53 homotetramer (1OLG)

Outputs:
  results/esm3_scores.csv: pairs.csv + zeroshot_esm3-1.4b column
  results/esm3/test_esm3-1.4b.json: 3-way interaction test via scripts/03_run_test.py
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
import torch.nn.functional as F

from plmppi.data import PRIMARY_SYSTEMS, load_reference

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def score_target_sequence_esm3(
    model: torch.nn.Module,
    seq: str,
    positions_to_score: list[int] | set[int],
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Computes masked-marginal log probabilities for requested 1-indexed positions using ESM3."""
    tok = model.tokenizers.sequence
    aa_to_id = {a: tok.convert_tokens_to_ids(a) for a in AA_LIST}

    positions = sorted(set(positions_to_score))
    if not positions:
        return {}

    min_pos = min(positions)
    max_pos = max(positions)

    # For long full-protein sequences (e.g. Spike 1273 aa with RBD 333-526), slice domain to fit VRAM
    if len(seq) > 500 and (max_pos - min_pos + 1) < 300:
        sub_seq = seq[min_pos - 1 : max_pos]
        offset = min_pos - 1
    else:
        sub_seq = seq
        offset = 0

    raw_ids = [tok.convert_tokens_to_ids(a) for a in sub_seq]
    full_ids = [tok.bos_token_id] + raw_ids + [tok.eos_token_id]

    out_log_probs: dict[int, dict[str, float]] = {}

    with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
        for pos in positions:
            sub_p = pos - offset  # 1-indexed position in sub_seq
            if not (1 <= sub_p <= len(sub_seq)):
                continue

            ids_copy = list(full_ids)
            ids_copy[sub_p] = tok.mask_token_id  # 1-indexed matches +1 BOS offset

            batch_input = torch.tensor([ids_copy], device=device)
            out = model(sequence_tokens=batch_input)
            logits = out.sequence_logits[0, sub_p, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)

            out_log_probs[pos] = {a: float(log_probs[aa_to_id[a]].item()) for a in AA_LIST}

    return out_log_probs


def score_pairs_with_esm3(
    df_pairs: pd.DataFrame,
    model: torch.nn.Module,
    structures_dir: Path,
    device: torch.device,
    ref_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Maps ESM3 masked marginal log-odds scores to all variants in df_pairs."""
    if ref_df is None:
        ref_df = load_reference()

    system_scores: dict[tuple[str, int, str, str], float] = {}
    system_logp_wt: dict[tuple[str, int, str, str], float] = {}
    diagnostics: dict[str, float] = {}

    for sys_obj in PRIMARY_SYSTEMS:
        match = ref_df.query("DMS_id == @sys_obj.dms_abundance")
        target_seq = match.iloc[0]["target_seq"]

        sys_pairs = df_pairs[df_pairs["system"] == sys_obj.system_id]
        positions_to_score = set(sys_pairs["position"].astype(int))

        t0 = time.time()
        pos_log_probs = score_target_sequence_esm3(
            model=model,
            seq=target_seq,
            positions_to_score=positions_to_score,
            device=device,
        )
        elapsed = time.time() - t0

        # Self-check native sequence recovery on scored positions
        wt_logps = []
        recoveries = []
        for p in sorted(pos_log_probs.keys()):
            wt_aa = target_seq[p - 1]
            lp_dict = pos_log_probs[p]
            if wt_aa in lp_dict:
                wt_logps.append(lp_dict[wt_aa])
                pred_aa = max(lp_dict, key=lp_dict.get)
                recoveries.append(pred_aa == wt_aa)

        mean_wt_logp = float(np.mean(wt_logps)) if wt_logps else float("nan")
        recovery = float(np.mean(recoveries)) if recoveries else float("nan")

        diagnostics[f"{sys_obj.system_id}_mean_wt_logp"] = mean_wt_logp
        diagnostics[f"{sys_obj.system_id}_recovery"] = recovery

        print(
            f"[{sys_obj.system_id}] {len(positions_to_score)} positions scored in {elapsed:.2f}s "
            f"({len(positions_to_score)/elapsed:.1f} pos/s), mean wt logP={mean_wt_logp:.3f}, recovery={recovery*100:.1f}%",
            flush=True,
        )

        for _, row in sys_pairs.iterrows():
            pos = int(row["position"])
            wt = str(row["wt"])
            mut = str(row["mut"])

            if pos in pos_log_probs and wt in pos_log_probs[pos] and mut in pos_log_probs[pos]:
                lp_mut = pos_log_probs[pos][mut]
                lp_wt = pos_log_probs[pos][wt]
                zs = float(lp_mut - lp_wt)
                system_scores[(sys_obj.system_id, pos, wt, mut)] = zs
                system_logp_wt[(sys_obj.system_id, pos, wt, mut)] = lp_wt

    df_out = df_pairs.copy()
    df_out["zeroshot_esm3-1.4b"] = df_out.apply(
        lambda r: system_scores.get((r["system"], int(r["position"]), r["wt"], r["mut"]), np.nan),
        axis=1,
    )
    df_out["logp_wt_esm3-1.4b"] = df_out.apply(
        lambda r: system_logp_wt.get((r["system"], int(r["position"]), r["wt"], r["mut"]), np.nan),
        axis=1,
    )

    coverage = float(df_out["zeroshot_esm3-1.4b"].notna().mean())
    diagnostics["total_coverage"] = coverage
    print(f"\n[coverage] {df_out['zeroshot_esm3-1.4b'].notna().sum()}/{len(df_out)} variants scored ({coverage*100:.2f}%)", flush=True)

    return df_out, diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score paired variants with ESM3-1.4B sequence mode."
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=REPO_ROOT / "results" / "pairs.csv",
        help="Input variants pairs CSV",
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=REPO_ROOT / "data" / "structures",
        help="Directory with crystal complex PDB structures",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "esm3_scores.csv",
        help="Output CSV for ESM3 scores",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "esm3",
        help="Output directory for test_esm3-1.4b.json",
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

    try:
        from esm.models.esm3 import ESM3
    except ImportError as err:
        print(f"Error: ESM3 requires the 'esm' package in .venv-esmc: {err}", file=sys.stderr)
        return 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] loading ESM3-sm-open-v1 on {device}...", flush=True)

    model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
    model.eval()

    df_pairs = pd.read_csv(args.variants)
    df_scored, diag = score_pairs_with_esm3(
        df_pairs=df_pairs,
        model=model,
        structures_dir=args.structures_dir,
        device=device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(args.out, index=False)
    print(f"[write] wrote {len(df_scored)} scores to {args.out}", flush=True)

    # Run Stage 3 test pipeline on ESM3 scores
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

#!/usr/bin/env python3
"""Stage 9: Score variants with ProteinMPNN structure-conditioned inverse folding model.

Evaluates ProteinMPNN (Dauparas et al., Science 2022) as a 3D structure-conditioned comparator
on the exact same 10,643 paired DMS variants across all 5 quaternary complex systems:
  - 6M0J (SARS-CoV-2 RBD + ACE2)
  - 6H46 (KRAS + DARPin K55)
  - 5OPI (HLA-A2 + B2M + TAPBPR)
  - 1FCC (GB1 + IgG-Fc)
  - 1OLG (p53 homotetramer)

Computes masked conditional log-odds given the complete 3D backbone coordinates of the complex:
  zeroshot_proteinmpnn = log P(mut | complex_backbone, native_context) - log P(wt | complex_backbone, native_context)

Outputs:
  results/proteinmpnn_scores.csv: pairs.csv + zeroshot_proteinmpnn column
  results/proteinmpnn/test_proteinmpnn.json: 3-way interaction test via scripts/03_run_test.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Ensure src/ and vendor/ProteinMPNN are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "vendor" / "ProteinMPNN"))

import numpy as np
import pandas as pd
import torch

from plmppi.data import PRIMARY_SYSTEMS, load_reference
from plmppi.interfaces import align_pdb_to_target_seq
from protein_mpnn_utils import ProteinMPNN, StructureDatasetPDB, parse_PDB, tied_featurize

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def compute_proteinmpnn_scores_for_system(
    model: ProteinMPNN,
    pdb_path: Path,
    target_chain: str,
    device: torch.device,
    n_samples: int = 10,
) -> tuple[np.ndarray, str]:
    """Runs ProteinMPNN conditional scoring on target chain within full complex backbone.

    Returns:
        log_probs_target: (target_len, 20) array of conditional log-probabilities
        target_seq: target chain amino acid sequence
    """
    pdb_dict_list = parse_PDB(str(pdb_path))
    b = pdb_dict_list[0]
    all_chains = [k.replace("seq_chain_", "") for k in b.keys() if k.startswith("seq_chain_")]
    partner_chains = [c for c in all_chains if c != target_chain]

    chain_dict = {b["name"]: ([target_chain], partner_chains)}
    dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=2000)
    batch = dataset[0]

    (
        X,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        letter_list_list,
        visible_list_list,
        masked_list_list,
        masked_chain_length_list_list,
        chain_M_pos,
        omit_AA_mask,
        residue_idx,
        dihedral_mask,
        tied_pos_list_of_lists_list,
        pssm_coef_all,
        pssm_bias_all,
        pssm_log_odds_all,
        bias_by_res_all,
        tied_beta,
    ) = tied_featurize([batch], device, chain_dict=chain_dict)

    log_cond_samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_c = model.conditional_probs(
                X,
                S,
                mask,
                chain_M * chain_M_pos,
                residue_idx,
                chain_encoding_all,
                randn_1,
                False,
            )
            log_cond_samples.append(log_c.cpu().numpy())

    log_cond_avg = np.mean(log_cond_samples, axis=0)
    target_seq = b[f"seq_chain_{target_chain}"]
    target_len = len(target_seq)
    log_probs_target = log_cond_avg[0, :target_len, :20]

    return log_probs_target, target_seq


def score_pairs_with_proteinmpnn(
    df_pairs: pd.DataFrame,
    model: ProteinMPNN,
    structures_dir: Path,
    device: torch.device,
    ref_df: pd.DataFrame | None = None,
    n_samples: int = 10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Maps ProteinMPNN conditional log-probs to all variants in df_pairs."""
    if ref_df is None:
        ref_df = load_reference()

    system_scores: dict[tuple[str, int, str, str], float] = {}
    diagnostics: dict[str, float] = {}

    for sys_obj in PRIMARY_SYSTEMS:
        pdb_path = structures_dir / f"{sys_obj.pdb_id}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        log_probs, mpnn_seq = compute_proteinmpnn_scores_for_system(
            model=model,
            pdb_path=pdb_path,
            target_chain=sys_obj.target_chain,
            device=device,
            n_samples=n_samples,
        )

        # Self-check native sequence recovery
        wt_indices = [ALPHABET.index(a) if a in ALPHABET else 0 for a in mpnn_seq]
        wt_log_probs = [log_probs[i, wt_indices[i]] for i in range(len(mpnn_seq))]
        predicted_indices = np.argmax(log_probs, axis=-1)
        recovery = float(np.mean([predicted_indices[i] == wt_indices[i] for i in range(len(mpnn_seq))]))
        mean_wt_logp = float(np.mean(wt_log_probs))

        diagnostics[f"{sys_obj.system_id}_mean_wt_logp"] = mean_wt_logp
        diagnostics[f"{sys_obj.system_id}_recovery"] = recovery

        print(
            f"[{sys_obj.system_id}] target={sys_obj.target_chain}, len={len(mpnn_seq)}, "
            f"mean wt logP={mean_wt_logp:.3f}, native recovery={recovery*100:.1f}%",
            flush=True,
        )

        match = ref_df.query("DMS_id == @sys_obj.dms_abundance")
        target_seq = match.iloc[0]["target_seq"]

        pdb_to_target = align_pdb_to_target_seq(mpnn_seq, target_seq)
        target_to_pdb = {v: k for k, v in pdb_to_target.items()}

        sys_pairs = df_pairs[df_pairs["system"] == sys_obj.system_id]
        for _, row in sys_pairs.iterrows():
            pos = int(row["position"])
            wt = str(row["wt"])
            mut = str(row["mut"])
            if pos in target_to_pdb:
                p_idx = target_to_pdb[pos]
                wt_idx = ALPHABET.index(wt)
                mut_idx = ALPHABET.index(mut)
                score = float(log_probs[p_idx, mut_idx] - log_probs[p_idx, wt_idx])
                system_scores[(sys_obj.system_id, pos, wt, mut)] = score

    df_out = df_pairs.copy()
    df_out["zeroshot_proteinmpnn"] = df_out.apply(
        lambda r: system_scores.get((r["system"], int(r["position"]), r["wt"], r["mut"]), np.nan),
        axis=1,
    )

    coverage = float(df_out["zeroshot_proteinmpnn"].notna().mean())
    diagnostics["total_coverage"] = coverage
    print(f"\n[coverage] {df_out['zeroshot_proteinmpnn'].notna().sum()}/{len(df_out)} variants scored ({coverage*100:.2f}%)", flush=True)

    return df_out, diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score paired variants with structure-conditioned ProteinMPNN."
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
        "--weights",
        type=Path,
        default=REPO_ROOT / "vendor" / "ProteinMPNN" / "vanilla_model_weights" / "v_48_020.pt",
        help="Path to ProteinMPNN model weights",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "proteinmpnn_scores.csv",
        help="Output CSV for ProteinMPNN scores",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "proteinmpnn",
        help="Output directory for test_proteinmpnn.json",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of decoding order samples to average",
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device}", flush=True)

    checkpoint = torch.load(str(args.weights), map_location=device)
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df_pairs = pd.read_csv(args.variants)
    df_scored, diag = score_pairs_with_proteinmpnn(
        df_pairs=df_pairs,
        model=model,
        structures_dir=args.structures_dir,
        device=device,
        n_samples=args.n_samples,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(args.out, index=False)
    print(f"[write] wrote {len(df_scored)} scores to {args.out}", flush=True)

    # Run Stage 3 test pipeline on ProteinMPNN scores
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

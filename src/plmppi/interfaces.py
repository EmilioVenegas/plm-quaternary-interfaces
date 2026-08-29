"""Structural interface and SASA computation for protein complexes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, ShrakeRupley, is_aa
from Bio.SeqUtils import seq1

# Empirical theoretical max SASA (Tien et al. 2013, PLOS ONE 8(11): e80635)
MAX_SASA = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}

DEFAULT_MAX_SASA = 200.0
INTERFACE_DSASA_THRESHOLD = 5.0  # Angstrom^2
INTERFACE_DISTANCE_THRESHOLD = 4.5  # Angstrom
CORE_RSASA_THRESHOLD = 0.20  # relative SASA < 20%


def align_pdb_to_target_seq(pdb_seq: str, target_seq: str) -> dict[int, int]:
    """Maps 0-indexed PDB chain residue index to 1-indexed target_seq position.

    Returns:
        Dict mapping pdb_idx (0 to len(pdb_seq)-1) -> target_pos (1 to len(target_seq)).
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(pdb_seq, target_seq)
    if not alignments:
        raise ValueError("Could not align PDB sequence to target sequence")

    best_aln = alignments[0]
    pdb_blocks, tgt_blocks = best_aln.aligned

    mapping: dict[int, int] = {}
    for (p_start, p_end), (t_start, t_end) in zip(pdb_blocks, tgt_blocks):
        for offset in range(p_end - p_start):
            p_idx = p_start + offset
            t_pos = (t_start + offset) + 1  # 1-indexed
            mapping[p_idx] = t_pos

    return mapping


def analyze_complex_structure(
    pdb_path: str | Path,
    target_chain_id: str,
    partner_chain_ids: list[str],
) -> pd.DataFrame:
    """Computes per-residue monomer and complex SASA, delta-SASA, and interface distance.

    Returns DataFrame with columns:
        [res_idx, pdb_res_num, aa, sasa_mono, sasa_comp, dsasa, rsasa, min_dist, compartment]
    """
    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()
    path_str = str(pdb_path)

    # 1. Complex SASA
    struct_comp = parser.get_structure("comp", path_str)
    sr.compute(struct_comp, level="R")

    # 2. Monomer SASA (detach all chains except target_chain_id)
    struct_mono = parser.get_structure("mono", path_str)
    model_mono = struct_mono[0]
    for c in list(model_mono.get_chains()):
        if c.id != target_chain_id:
            model_mono.detach_child(c.id)
    sr.compute(struct_mono, level="R")

    # 3. Collect partner heavy atom coordinates
    partner_atoms = []
    for c in struct_comp[0].get_chains():
        if c.id in partner_chain_ids:
            for r in c:
                if is_aa(r):
                    for a in r:
                        if not a.name.startswith("H"):
                            partner_atoms.append(a.coord)
    partner_coords = np.array(partner_atoms) if partner_atoms else np.empty((0, 3))

    # 4. Iterate over target chain amino acid residues
    res_comp = [r for r in struct_comp[0][target_chain_id] if is_aa(r)]
    res_mono = [r for r in struct_mono[0][target_chain_id] if is_aa(r)]

    rows = []
    for idx, (rc, rm) in enumerate(zip(res_comp, res_mono)):
        aa = seq1(rc.get_resname())
        res_num = rc.get_id()[1]
        sasa_c = float(getattr(rc, "sasa", 0.0))
        sasa_m = float(getattr(rm, "sasa", 0.0))
        dsasa = max(0.0, sasa_m - sasa_c)
        max_s = MAX_SASA.get(aa, DEFAULT_MAX_SASA)
        rsasa = sasa_m / max_s

        target_atoms = [a.coord for a in rc if not a.name.startswith("H")]
        if len(target_atoms) > 0 and len(partner_coords) > 0:
            dists = np.min(
                np.linalg.norm(
                    np.array(target_atoms)[:, None, :] - partner_coords[None, :, :],
                    axis=2,
                )
            )
            min_dist = float(dists)
        else:
            min_dist = 999.0

        if dsasa >= INTERFACE_DSASA_THRESHOLD or min_dist <= INTERFACE_DISTANCE_THRESHOLD:
            compartment = "Interface"
        elif rsasa < CORE_RSASA_THRESHOLD:
            compartment = "Core"
        else:
            compartment = "Surface"

        rows.append(
            {
                "res_idx": idx,
                "pdb_res_num": res_num,
                "aa": aa,
                "sasa_mono": sasa_m,
                "sasa_comp": sasa_c,
                "dsasa": dsasa,
                "rsasa": rsasa,
                "min_dist": min_dist,
                "compartment": compartment,
            }
        )

    return pd.DataFrame(rows)


def get_system_compartments(
    pdb_path: str | Path,
    target_chain_id: str,
    partner_chain_ids: list[str],
    target_seq: str,
) -> dict[int, dict[str, Any]]:
    """Analyzes complex and maps compartments to 1-indexed target_seq positions.

    Returns:
        Dict mapping target_pos -> {
            "aa": str,
            "compartment": "Interface" | "Core" | "Surface",
            "dsasa": float,
            "rsasa": float,
            "min_dist": float,
            "pdb_res_num": int
        }
    """
    df_res = analyze_complex_structure(pdb_path, target_chain_id, partner_chain_ids)
    pdb_seq = "".join(df_res["aa"].tolist())

    pdb_to_target = align_pdb_to_target_seq(pdb_seq, target_seq)

    out: dict[int, dict[str, Any]] = {}
    for _, row in df_res.iterrows():
        p_idx = int(row["res_idx"])
        if p_idx in pdb_to_target:
            t_pos = pdb_to_target[p_idx]
            out[t_pos] = {
                "aa": row["aa"],
                "compartment": row["compartment"],
                "dsasa": float(row["dsasa"]),
                "rsasa": float(row["rsasa"]),
                "min_dist": float(row["min_dist"]),
                "pdb_res_num": int(row["pdb_res_num"]),
            }

    return out

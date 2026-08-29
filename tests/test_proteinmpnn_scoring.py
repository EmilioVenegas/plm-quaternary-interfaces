"""Unit tests for Stage 9: ProteinMPNN position mapping and scoring logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmppi.interfaces import align_pdb_to_target_seq

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def test_align_pdb_to_target_seq_mapping():
    pdb_seq = "SHMTEYKLVVVG"
    target_seq = "MTEYKLVVVGAVGVGK"

    mapping = align_pdb_to_target_seq(pdb_seq, target_seq)
    # pdb index 2 ('M') should map to target position 1
    assert 2 in mapping
    assert mapping[2] == 1
    assert mapping[11] == 10


def test_synthetic_proteinmpnn_log_odds_computation():
    # Synthetic log-probabilities for a 3-residue target chain: M-A-Y
    target_seq = "MAY"
    mpnn_seq = "MAY"

    # Shape (3, 20)
    log_probs = np.zeros((3, 20))
    # Assign higher log-prob to native wt
    for i, a in enumerate(mpnn_seq):
        wt_idx = ALPHABET.index(a)
        log_probs[i, wt_idx] = -0.5
        # Assign lower log-prob to mutant
        for m_idx in range(20):
            if m_idx != wt_idx:
                log_probs[i, m_idx] = -2.5

    pdb_to_target = align_pdb_to_target_seq(mpnn_seq, target_seq)
    target_to_pdb = {v: k for k, v in pdb_to_target.items()}

    # Construct test dataframe
    df_pairs = pd.DataFrame([
        {"system": "TEST", "position": 1, "wt": "M", "mut": "A"},
        {"system": "TEST", "position": 2, "wt": "A", "mut": "G"},
        {"system": "TEST", "position": 3, "wt": "Y", "mut": "W"},
    ])

    scores = []
    for _, r in df_pairs.iterrows():
        pos = int(r["position"])
        p_idx = target_to_pdb[pos]
        wt_idx = ALPHABET.index(r["wt"])
        mut_idx = ALPHABET.index(r["mut"])
        score = float(log_probs[p_idx, mut_idx] - log_probs[p_idx, wt_idx])
        scores.append(score)

    df_pairs["zeroshot_proteinmpnn"] = scores

    assert len(df_pairs) == 3
    # Every mutant has log-prob -2.5 vs WT -0.5 -> delta score = -2.0
    for s in df_pairs["zeroshot_proteinmpnn"]:
        assert np.isclose(s, -2.0)

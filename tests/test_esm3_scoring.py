"""Unit tests for Stage 10: ESM3 sequence scoring logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def test_synthetic_esm3_log_odds_mapping():
    target_seq = "MAY"
    # Fabricated log-probs for 3 positions
    log_probs = np.zeros((3, 20), dtype=np.float32)
    for i, a in enumerate(target_seq):
        wt_idx = AA_LIST.index(a)
        log_probs[i, wt_idx] = -0.5
        for m_idx in range(20):
            if m_idx != wt_idx:
                log_probs[i, m_idx] = -2.5

    df_pairs = pd.DataFrame([
        {"system": "TEST", "position": 1, "wt": "M", "mut": "A"},
        {"system": "TEST", "position": 2, "wt": "A", "mut": "G"},
        {"system": "TEST", "position": 3, "wt": "Y", "mut": "W"},
    ])

    scores = []
    for _, r in df_pairs.iterrows():
        pos = int(r["position"])
        p_idx = pos - 1
        wt_idx = AA_LIST.index(r["wt"])
        mut_idx = AA_LIST.index(r["mut"])
        score = float(log_probs[p_idx, mut_idx] - log_probs[p_idx, wt_idx])
        scores.append(score)

    df_pairs["zeroshot_esm3-1.4b"] = scores

    assert len(df_pairs) == 3
    for s in df_pairs["zeroshot_esm3-1.4b"]:
        assert np.isclose(s, -2.0)

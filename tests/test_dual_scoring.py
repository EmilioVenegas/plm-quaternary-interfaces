"""Unit tests for Stage 11: Dual-Scoring Mitigation and Pareto Frontier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plmppi.stats import evaluate_dual_scoring_frontier


def test_dual_scoring_frontier_synthetic():
    np.random.seed(42)
    rows = []
    systems = ["SARS-CoV-2_RBD", "KRAS", "HLA-A2", "GB1", "p53"]
    for s in systems:
        for i in range(100):
            rows.append({
                "system": s,
                "position": i + 1,
                "wt": "A",
                "mut": "G",
                "compartment": "Interface" if i < 30 else ("Core" if i < 65 else "Surface"),
                "dms_score_abundance": float(np.random.randn()),
                "dms_score_binding": float(np.random.randn()),
                "zeroshot_proteinmpnn": float(np.random.randn()),
                "zeroshot_esm2-650m": float(np.random.randn()),
            })
    df_syn = pd.DataFrame(rows)
    frontier = evaluate_dual_scoring_frontier(df_syn, alpha_list=(0.0, 0.5, 1.0, 100.0), top_pct=0.20)

    assert "alpha_sweep" in frontier
    assert len(frontier["alpha_sweep"]) == 4
    assert "optimal_mitigation" in frontier
    assert "conclusion" in frontier
    for e in frontier["alpha_sweep"]:
        assert 0.0 <= e["interface_fnr_pct"] <= 100.0
        assert 0.0 <= e["monomer_expressibility_pct"] <= 100.0

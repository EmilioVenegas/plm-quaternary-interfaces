"""Tests for binder design filter audit and multi-chain rescue (Piece B)."""

import numpy as np
import pandas as pd
import pytest
from plmppi.stats import simulate_plm_filter_trap


def test_simulate_plm_filter_trap_metrics():
    np.random.seed(42)
    # Create dataset with clear depletion at interface
    # Interface beneficial mutations have lower PLM scores than core/surface beneficial mutations
    n_int = 100
    n_non_int = 200

    int_rows = [
        {
            "system": "KRAS",
            "position": i,
            "wt": "A",
            "mut": "G",
            "compartment": "Interface",
            "dms_score_abundance": 0.5,
            "dms_score_binding": 0.8,  # Beneficial
            "zeroshot_test": float(np.random.normal(-2.0, 0.5)),  # Low PLM score
        }
        for i in range(1, n_int + 1)
    ]

    non_int_rows = [
        {
            "system": "KRAS",
            "position": i + 100,
            "wt": "A",
            "mut": "G",
            "compartment": "Surface",
            "dms_score_abundance": 0.5,
            "dms_score_binding": 0.8,  # Beneficial
            "zeroshot_test": float(np.random.normal(0.0, 0.5)),  # High PLM score
        }
        for i in range(1, n_non_int + 1)
    ]

    df_test = pd.DataFrame(int_rows + non_int_rows)
    res = simulate_plm_filter_trap(df_test, arm="test", thresholds=(0.10, 0.20, 0.50))

    assert "thresholds_simulation" in res
    assert len(res["thresholds_simulation"]) == 3

    # Check top 20% filter
    t20 = res["thresholds_simulation"][1]
    assert t20["filter_top_pct"] == 20
    assert t20["n_beneficial_interface"] == 100
    assert t20["n_beneficial_non_interface"] == 200
    # Interface depletion rate should be positive (> 0.5) because interface scores are much lower
    assert t20["interface_depletion_rate"] > 0.5
    assert t20["interface_false_negative_rate"] > 0.8


def test_simulate_plm_filter_trap_missing_arm():
    df_empty = pd.DataFrame({"system": ["p53"], "compartment": ["Interface"]})
    with pytest.raises(KeyError):
        simulate_plm_filter_trap(df_empty, arm="nonexistent_arm")

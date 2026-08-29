"""Tests for evolutionary class stratification (Piece A)."""

import numpy as np
import pandas as pd
import pytest
from plmppi.stats import (
    EVOLUTIONARY_REGIMES,
    SYSTEM_TO_CLASS,
    partial_spearman,
    stratify_by_evolutionary_class,
)


def test_partial_spearman_perfect_mediation():
    # If y is entirely mediated by z, partial correlation rho(x, y | z) should be near 0
    np.random.seed(42)
    N = 500
    z = np.random.randn(N)
    x = z + np.random.normal(0, 0.5, size=N)
    y = 2.0 * z + np.random.normal(0, 0.5, size=N)

    r_raw, _ = stats_spearman = pd.Series(x).corr(pd.Series(y), method="spearman"), 0
    r_part = partial_spearman(x, y, z)

    assert abs(r_part) < abs(r_raw)
    assert abs(r_part) < 0.15


def test_partial_spearman_independent_signal():
    # If y has independent signal from x orthogonal to z
    np.random.seed(42)
    N = 500
    z = np.random.randn(N)
    x = np.random.randn(N)
    y = 3.0 * x + 0.5 * z

    r_part = partial_spearman(x, y, z)
    assert r_part > 0.8


def test_evolutionary_regimes_mapping():
    assert "Homooligomer" in EVOLUTIONARY_REGIMES
    assert "Natural_Heterodimer" in EVOLUTIONARY_REGIMES
    assert "Synthetic_CrossSpecies" in EVOLUTIONARY_REGIMES

    for sys_id, class_id in SYSTEM_TO_CLASS.items():
        assert class_id in EVOLUTIONARY_REGIMES
        assert sys_id in EVOLUTIONARY_REGIMES[class_id]["systems"]


def test_stratify_by_evolutionary_class_synthetic():
    np.random.seed(123)
    systems = ["p53", "HLA-A2", "KRAS"]
    rows = []
    for s in systems:
        for p in range(1, 31):
            comp = "Interface" if p <= 15 else "Surface"
            rows.append(
                {
                    "system": s,
                    "position": p,
                    "wt": "A",
                    "mut": "G",
                    "compartment": comp,
                    "dms_score_abundance": float(np.random.randn()),
                    "dms_score_binding": float(np.random.randn()),
                    "zeroshot_mock": float(np.random.randn()),
                }
            )

    df_synth = pd.DataFrame(rows)
    res = stratify_by_evolutionary_class(df_synth, arm="mock", n_perm=50, seed=42)

    assert "classes" in res
    assert "Homooligomer" in res["classes"]
    assert "Natural_Heterodimer" in res["classes"]
    assert "Synthetic_CrossSpecies" in res["classes"]
    assert "formal_answer" in res
    assert "question" in res["formal_answer"]
    assert "hierarchical_interaction_model" in res

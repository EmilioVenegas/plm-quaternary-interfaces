"""Unit tests for Stage 8: Interface Hotspot vs Rim Stratification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmppi.stats import assign_hotspot_rim, stratify_by_hotspot


def test_assign_hotspot_rim_tertile_split():
    # Synthetic DataFrame with 9 interface residues
    dsasa_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    df = pd.DataFrame({
        "position": list(range(1, 10)),
        "dsasa": dsasa_values,
        "compartment": ["Interface"] * 9,
    })
    labels = assign_hotspot_rim(df, dsasa_col="dsasa")
    assert len(labels) == 9
    assert list(labels.unique()) in [
        ["Rim", "Mid", "Hotspot"],
        ["Hotspot", "Mid", "Rim"],
    ] or set(labels.unique()) == {"Hotspot", "Mid", "Rim"}
    # Lowest 3 should be Rim, highest 3 Hotspot
    assert labels.iloc[0] == "Rim"
    assert labels.iloc[-1] == "Hotspot"


def test_stratify_by_hotspot_synthetic():
    np.random.seed(42)
    n_per_sys = 60
    systems = ["SARS-CoV-2_RBD", "KRAS", "HLA-A2", "GB1", "p53"]
    rows = []
    for s in systems:
        for i in range(n_per_sys):
            dsasa = np.random.uniform(5.0, 100.0)
            # Create a synthetic signal where abundance tracks plm, but binding is decoupled
            plm = np.random.randn()
            ab = 0.5 * plm + np.random.randn() * 0.5
            bi = -0.3 * plm + np.random.randn() * 0.5 if dsasa > 50.0 else 0.2 * plm + np.random.randn() * 0.5
            rows.append({
                "system": s,
                "position": i + 1,
                "wt": "A",
                "mut": "G",
                "compartment": "Interface",
                "dsasa": dsasa,
                "rsasa": 0.3,
                "min_dist": 3.5,
                "dms_score_abundance": ab,
                "dms_score_binding": bi,
                "zeroshot_test_arm": plm,
            })
    df_syn = pd.DataFrame(rows)
    res = stratify_by_hotspot(df_syn, arm="test_arm", n_perm=100, seed=42)

    assert "arm" in res
    assert res["arm"] == "test_arm"
    assert "groups" in res
    assert "Hotspot" in res["groups"]
    assert "Rim" in res["groups"]
    assert "Mid" in res["groups"]
    assert "All" in res["groups"]
    assert "interaction_test" in res
    assert "verdict" in res["interaction_test"]
    assert not np.isnan(res["groups"]["Hotspot"]["rho_partial_plm_binding_given_abundance"])
    assert not np.isnan(res["groups"]["Rim"]["rho_partial_plm_binding_given_abundance"])

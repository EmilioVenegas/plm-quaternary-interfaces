"""Tests for statistical modeling and hypothesis testing."""

import numpy as np
import pandas as pd
import pytest
from plmppi.stats import (
    fit_clustered_ols,
    fit_system_fixed_effects_ols,
    prepare_analysis_frame,
    run_leave_one_system_out_test,
    run_three_way_interaction_test,
    standardize_series,
    wild_cluster_bootstrap,
)


def test_standardize_series():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    s_z = standardize_series(s)
    assert np.isclose(s_z.mean(), 0.0)
    assert np.isclose(s_z.std(), 1.0)


def test_fit_clustered_ols_recovers_ground_truth():
    np.random.seed(42)
    N = 1000
    clusters = np.random.choice(["Sys1", "Sys2", "Sys3", "Sys4", "Sys5"], size=N)
    x1 = np.random.randn(N)
    x2 = np.random.choice([0.0, 1.0], size=N)

    # True beta: intercept=1.0, x1=2.5, x2=-1.5, x1*x2=0.8
    y = 1.0 + 2.5 * x1 - 1.5 * x2 + 0.8 * (x1 * x2) + np.random.normal(0, 0.2, size=N)

    X = np.column_stack([np.ones(N), x1, x2, x1 * x2])
    names = ["Intercept", "x1", "x2", "x1:x2"]

    res = fit_clustered_ols(X, y, clusters, names)
    assert res["n_obs"] == N
    assert res["n_clusters"] == 5
    assert np.isclose(res["params"]["Intercept"]["coef"], 1.0, atol=0.1)
    assert np.isclose(res["params"]["x1"]["coef"], 2.5, atol=0.1)
    assert np.isclose(res["params"]["x2"]["coef"], -1.5, atol=0.1)
    assert np.isclose(res["params"]["x1:x2"]["coef"], 0.8, atol=0.1)
    assert res["params"]["x1:x2"]["p_val"] < 0.01


def test_run_three_way_interaction_test_synthetic():
    np.random.seed(123)
    df_raw = pd.DataFrame(
        {
            "system": ["SysA"] * 100 + ["SysB"] * 100,
            "position": list(range(1, 101)) * 2,
            "wt": ["A"] * 200,
            "mut": ["C"] * 200,
            "compartment": ["Interface"] * 50
            + ["Core"] * 50
            + ["Interface"] * 50
            + ["Surface"] * 50,
            "dms_score_abundance": np.random.randn(200),
            "dms_score_binding": np.random.randn(200),
            "zeroshot_mock": np.random.randn(200),
        }
    )

    frame = prepare_analysis_frame(df_raw, arm="mock")
    assert len(frame) == 400  # 200 * 2 (Abundance & Binding)
    assert "dms_z" in frame.columns
    assert "plm_z" in frame.columns

    res = run_three_way_interaction_test(frame, n_perm=100, seed=42)
    assert "beta_three_way" in res
    assert "p_clustered" in res
    assert "verdict" in res
    assert "subgroup_correlations" in res
    assert "p_wild_bootstrap" in res
    assert "wild_cluster_bootstrap" in res
    assert "fixed_effects_summary" in res
    assert "leave_one_system_out" in res


def test_wild_cluster_bootstrap():
    np.random.seed(42)
    N = 600
    clusters = np.random.choice(["Sys1", "Sys2", "Sys3", "Sys4", "Sys5", "Sys6"], size=N)
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    y = 1.0 + 3.0 * x1 + 0.0 * x2 + np.random.normal(0, 0.5, size=N)
    X = np.column_stack([np.ones(N), x1, x2])

    wcb_sig = wild_cluster_bootstrap(X, y, clusters, feature_idx=1, n_boot=500, seed=42)
    assert wcb_sig["p_wild_bootstrap"] < 0.05
    assert "t_stat_orig" in wcb_sig

    wcb_null = wild_cluster_bootstrap(X, y, clusters, feature_idx=2, n_boot=500, seed=42)
    assert wcb_null["p_wild_bootstrap"] > 0.05


def test_fit_system_fixed_effects_ols():
    np.random.seed(42)
    N = 500
    clusters = np.random.choice(["SysA", "SysB", "SysC", "SysD"], size=N)
    x = np.random.randn(N)
    # System effects: SysA=1, SysB=2, SysC=3, SysD=4
    sys_effects = {"SysA": 1.0, "SysB": 2.0, "SysC": 3.0, "SysD": 4.0}
    fe = np.array([sys_effects[c] for c in clusters])
    y = fe + 2.5 * x + np.random.normal(0, 0.2, size=N)
    X = np.column_stack([np.ones(N), x])

    res = fit_system_fixed_effects_ols(X, y, clusters, ["Intercept", "x"])
    assert len(res["system_dummies"]) == 4
    assert np.isclose(res["params"]["x"]["coef"], 2.5, atol=0.1)
    for s in ["SysA", "SysB", "SysC", "SysD"]:
        assert np.isclose(res["params"][f"FE_{s}"]["coef"], sys_effects[s], atol=0.2)


def test_run_leave_one_system_out_test():
    np.random.seed(42)
    df_raw = pd.DataFrame(
        {
            "system": ["SysA"] * 100 + ["SysB"] * 100 + ["SysC"] * 100,
            "position": list(range(1, 101)) * 3,
            "wt": ["A"] * 300,
            "mut": ["C"] * 300,
            "compartment": ["Interface"] * 50 + ["Core"] * 50 + ["Interface"] * 50 + ["Surface"] * 50 + ["Interface"] * 50 + ["Core"] * 50,
            "dms_score_abundance": np.random.randn(300),
            "dms_score_binding": np.random.randn(300),
            "zeroshot_mock": np.random.randn(300),
        }
    )
    frame = prepare_analysis_frame(df_raw, arm="mock")
    loso = run_leave_one_system_out_test(frame)
    assert set(loso.keys()) == {"SysA", "SysB", "SysC"}
    for sys_id, r in loso.items():
        assert r["omitted_system"] == sys_id
        assert r["n_obs"] == 400
        assert "beta_three_way" in r
        assert "rho_interface_abundance" in r
        assert "rho_interface_binding" in r

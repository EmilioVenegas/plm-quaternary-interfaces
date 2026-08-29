"""Hermetic unit tests for interface definition sensitivity and structural sweep."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[1]
_script_path = REPO_ROOT / "scripts" / "07_interface_sensitivity_sweep.py"
_spec = importlib.util.spec_from_file_location("interface_sensitivity_sweep", _script_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

classify_compartments = _module.classify_compartments
compute_arm_metrics = _module.compute_arm_metrics
detect_arms = _module.detect_arms
format_summary_matrix = _module.format_summary_matrix
run_sensitivity_sweep = _module.run_sensitivity_sweep

@pytest.fixture
def synthetic_scores_df() -> pd.DataFrame:
    """Generates a reproducible synthetic dataframe with known structural and score properties."""
    np.random.seed(42)
    n_variants = 120

    # Systems: 3 systems with 40 variants each
    systems = ["SysA"] * 40 + ["SysB"] * 40 + ["SysC"] * 40
    positions = list(range(1, 41)) * 3

    # Define geometric properties
    # Positions 1-10: High dsasa (15.0), moderate dist (4.0), high rsasa (0.50) -> Interface at most cutoffs
    # Positions 11-20: Low dsasa (1.0), close dist (3.0), high rsasa (0.40) -> Interface at dist >= 3.0
    # Positions 21-30: Zero dsasa (0.0), far dist (12.0), low rsasa (0.05) -> Core
    # Positions 31-40: Zero dsasa (0.0), far dist (15.0), high rsasa (0.60) -> Surface

    dsasa_list = []
    dist_list = []
    rsasa_list = []

    for _ in range(3):
        for pos in range(1, 41):
            if pos <= 10:
                dsasa_list.append(15.0 + np.random.uniform(0, 2))
                dist_list.append(4.0)
                rsasa_list.append(0.50)
            elif pos <= 20:
                dsasa_list.append(1.0)
                dist_list.append(3.0 + np.random.uniform(0, 0.5))
                rsasa_list.append(0.40)
            elif pos <= 30:
                dsasa_list.append(0.0)
                dist_list.append(12.0)
                rsasa_list.append(0.05)
            else:
                dsasa_list.append(0.0)
                dist_list.append(15.0)
                rsasa_list.append(0.60)

    # Correlated scores
    dms_ab = np.random.randn(n_variants)
    dms_bi = dms_ab * 0.7 + np.random.randn(n_variants) * 0.3
    zs_mock1 = dms_ab * 0.6 + np.random.randn(n_variants) * 0.4
    zs_mock2 = dms_ab * 0.5 + np.random.randn(n_variants) * 0.5

    df = pd.DataFrame(
        {
            "system": systems,
            "position": positions,
            "wt": ["A"] * n_variants,
            "mut": ["G"] * n_variants,
            "compartment": ["Unknown"] * n_variants,
            "dsasa": dsasa_list,
            "min_dist": dist_list,
            "rsasa": rsasa_list,
            "dms_score_abundance": dms_ab,
            "dms_score_binding": dms_bi,
            "zeroshot_mock1": zs_mock1,
            "zeroshot_mock2": zs_mock2,
        }
    )
    return df


def test_classify_compartments_partition_invariants(synthetic_scores_df: pd.DataFrame):
    """Verifies that partition is mutually exclusive and exhaustive across the dataset."""
    df_classified = classify_compartments(
        synthetic_scores_df, dsasa_threshold=5.0, dist_threshold=4.5, rsasa_threshold=0.20
    )

    counts = df_classified["compartment"].value_counts()
    n_int = counts.get("Interface", 0)
    n_core = counts.get("Core", 0)
    n_surf = counts.get("Surface", 0)

    assert n_int + n_core + n_surf == len(synthetic_scores_df)
    assert set(df_classified["compartment"].unique()).issubset({"Interface", "Core", "Surface"})


def test_classify_compartments_threshold_monotonicity(synthetic_scores_df: pd.DataFrame):
    """Higher dsasa threshold or lower distance threshold should strictly reduce or preserve interface variants."""
    # Permissive cutoff: dsasa >= 2.0 OR dist <= 5.0
    df_permissive = classify_compartments(
        synthetic_scores_df, dsasa_threshold=2.0, dist_threshold=5.0
    )
    n_int_permissive = (df_permissive["compartment"] == "Interface").sum()

    # Strict cutoff: dsasa >= 20.0 OR dist <= 3.5
    df_strict = classify_compartments(
        synthetic_scores_df, dsasa_threshold=20.0, dist_threshold=3.5
    )
    n_int_strict = (df_strict["compartment"] == "Interface").sum()

    assert n_int_permissive >= n_int_strict


def test_classify_compartments_column_aliases():
    """Verifies fallback support for 'delta_sasa' and 'min_distance' column names."""
    df_alias = pd.DataFrame(
        {
            "delta_sasa": [10.0, 1.0, 0.0],
            "min_distance": [5.0, 3.0, 10.0],
            "rsasa": [0.30, 0.10, 0.50],
        }
    )
    df_classified = classify_compartments(df_alias, dsasa_threshold=5.0, dist_threshold=4.0)
    assert df_classified.iloc[0]["compartment"] == "Interface"  # delta_sasa >= 5.0
    assert df_classified.iloc[1]["compartment"] == "Interface"  # min_distance <= 4.0
    assert df_classified.iloc[2]["compartment"] == "Surface"    # rsasa >= 0.20


def test_detect_arms(synthetic_scores_df: pd.DataFrame):
    arms = detect_arms(synthetic_scores_df)
    assert arms == ["mock1", "mock2"]


def test_compute_arm_metrics(synthetic_scores_df: pd.DataFrame):
    df_classified = classify_compartments(
        synthetic_scores_df, dsasa_threshold=5.0, dist_threshold=4.5
    )
    metrics = compute_arm_metrics(df_classified, arm="mock1")

    expected_keys = {
        "n_interface",
        "rho_abundance",
        "p_abundance",
        "rho_binding",
        "p_binding",
        "rho_partial",
        "beta_three_way",
        "se_three_way",
        "t_stat_three_way",
        "p_three_way",
    }
    assert set(metrics.keys()) == expected_keys
    assert metrics["n_interface"] > 0
    assert not np.isnan(metrics["rho_abundance"])
    assert not np.isnan(metrics["beta_three_way"])


def test_run_sensitivity_sweep_grid_structure(synthetic_scores_df: pd.DataFrame):
    dsasa_cutoffs = [2.0, 10.0]
    dist_cutoffs = [3.5, 5.0]

    sweep_results = run_sensitivity_sweep(
        synthetic_scores_df,
        dsasa_thresholds=dsasa_cutoffs,
        dist_thresholds=dist_cutoffs,
        arms=["mock1", "mock2"],
    )

    assert "metadata" in sweep_results
    assert "grid" in sweep_results
    assert "summary" in sweep_results
    assert "robustness_verdict" in sweep_results

    # 2 x 2 grid = 4 cells
    assert len(sweep_results["grid"]) == 4

    for entry in sweep_results["grid"]:
        assert "delta_sasa_threshold" in entry
        assert "distance_threshold" in entry
        assert "n_interface" in entry
        assert "n_core" in entry
        assert "n_surface" in entry
        assert "mock1" in entry["models"]
        assert "mock2" in entry["models"]

    # Summary entries
    assert "mock1" in sweep_results["summary"]
    assert "mock2" in sweep_results["summary"]
    assert "beta_three_way_min" in sweep_results["summary"]["mock1"]
    assert "rho_partial_mean" in sweep_results["summary"]["mock1"]


def test_format_summary_matrix(synthetic_scores_df: pd.DataFrame):
    sweep_results = run_sensitivity_sweep(
        synthetic_scores_df,
        dsasa_thresholds=[5.0, 10.0],
        dist_thresholds=[4.0, 5.0],
        arms=["mock1"],
    )
    summary_text = format_summary_matrix(sweep_results)
    assert "INTERFACE DEFINITION SENSITIVITY SWEEP SUMMARY" in summary_text
    assert "Model Arm: mock1" in summary_text
    assert "OVERALL ROBUSTNESS VERDICT:" in summary_text


def test_compute_arm_metrics_small_n():
    """When fewer than 4 interface observations exist, returns NaNs gracefully without crashing."""
    df_tiny = pd.DataFrame(
        {
            "system": ["SysA", "SysA"],
            "position": [1, 2],
            "wt": ["A", "A"],
            "mut": ["G", "C"],
            "compartment": ["Interface", "Interface"],
            "dms_score_abundance": [0.5, 0.2],
            "dms_score_binding": [0.3, 0.1],
            "zeroshot_mock": [-1.0, -2.0],
        }
    )
    metrics = compute_arm_metrics(df_tiny, arm="mock")
    assert metrics["n_interface"] == 2
    assert np.isnan(metrics["rho_abundance"])
    assert np.isnan(metrics["beta_three_way"])


def test_classify_compartments_missing_columns():
    """Raises KeyError when required geometry columns are missing."""
    df_bad = pd.DataFrame({"dummy": [1, 2, 3]})
    with pytest.raises(KeyError):
        classify_compartments(df_bad, dsasa_threshold=5.0, dist_threshold=4.5)

"""Tests for dataset loading and curation."""

import pandas as pd
import pytest
from plmppi.data import (
    PRIMARY_SYSTEMS,
    build_system_dataset,
    load_reference,
    parse_single_mutants,
)


def test_primary_systems_configuration():
    assert len(PRIMARY_SYSTEMS) == 5
    sys_ids = {s.system_id for s in PRIMARY_SYSTEMS}
    assert sys_ids == {"SARS-CoV-2_RBD", "KRAS", "HLA-A2", "GB1", "p53"}


def test_parse_single_mutants():
    df_raw = pd.DataFrame(
        {
            "mutant": ["A12C", "K15R:D20E", "G100W", "invalid", "", "T5A"],
            "DMS_score": [1.5, -0.5, -2.1, 0.0, 0.0, 0.8],
        }
    )
    df_clean = parse_single_mutants(df_raw)
    assert len(df_clean) == 3
    assert df_clean["position"].tolist() == [12, 100, 5]
    assert df_clean["wt"].tolist() == ["A", "G", "T"]
    assert df_clean["mut"].tolist() == ["C", "W", "A"]
    assert df_clean["dms_score"].tolist() == [1.5, -2.1, 0.8]


def test_load_reference():
    ref = load_reference()
    assert len(ref) >= 200
    assert "DMS_id" in ref.columns
    assert "target_seq" in ref.columns


def test_build_system_dataset_sars2():
    ref = load_reference()
    sys = PRIMARY_SYSTEMS[0]  # SARS-CoV-2 RBD
    df_sys = build_system_dataset(sys, ref)
    assert len(df_sys) > 3000
    assert "compartment" in df_sys.columns
    assert "dms_score_abundance" in df_sys.columns
    assert "dms_score_binding" in df_sys.columns
    assert set(df_sys["compartment"].unique()) == {"Interface", "Core", "Surface"}

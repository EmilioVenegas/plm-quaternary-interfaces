"""Tests for dataset loading and curation."""

import pandas as pd
import pytest
from plmppi.data import (
    InclusionStatus,
    PRIMARY_SYSTEMS,
    audit_provenance_summary,
    build_system_dataset,
    get_legacy_cohort,
    get_matched_cohort,
    get_system_registry,
    load_reference,
    parse_single_mutants,
    validate_registry_invariants,
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


def test_provenance_registry_covers_primary_systems_and_is_valid():
    registry = get_system_registry()
    primary_ids = [system.system_id for system in PRIMARY_SYSTEMS]

    assert list(registry) == primary_ids
    assert [record.system.system_id for record in registry.values()] == primary_ids
    assert validate_registry_invariants() == []


def test_provenance_cohorts_preserve_their_auditable_membership_and_order():
    strict_ids = [
        record.system.system_id for record in get_matched_cohort()
    ]
    conditional_ids = [
        record.system.system_id
        for record in get_matched_cohort(include_conditional=True)
    ]
    legacy_ids = [record.system.system_id for record in get_legacy_cohort()]

    assert strict_ids == ["SARS-CoV-2_RBD", "HLA-A2"]
    assert conditional_ids == ["SARS-CoV-2_RBD", "KRAS", "HLA-A2"]
    assert legacy_ids == [system.system_id for system in PRIMARY_SYSTEMS]


def test_provenance_audit_summary_has_one_complete_row_per_primary_system():
    summary = audit_provenance_summary()
    required_columns = {
        "system_id",
        "inclusion_status",
        "abundance_dms_id",
        "binding_dms_id",
        "abundance_assay_type",
        "binding_assay_type",
        "matched_library",
        "direct_match",
        "rationale",
    }

    assert len(summary) == len(PRIMARY_SYSTEMS)
    assert summary["system_id"].tolist() == [
        system.system_id for system in PRIMARY_SYSTEMS
    ]
    assert required_columns.issubset(summary.columns)
    assert summary["inclusion_status"].tolist() == [
        InclusionStatus.INCLUDED.value,
        InclusionStatus.CONDITIONAL.value,
        InclusionStatus.INCLUDED.value,
        InclusionStatus.EXCLUDED.value,
        InclusionStatus.EXCLUDED.value,
    ]

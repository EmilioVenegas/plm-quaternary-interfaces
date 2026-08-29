"""Tests for structural interface and SASA computations."""

import numpy as np
import pytest
from plmppi.interfaces import (
    DEFAULT_MAX_SASA,
    MAX_SASA,
    align_pdb_to_target_seq,
    analyze_complex_structure,
    get_system_compartments,
)


def test_max_sasa_values():
    assert len(MAX_SASA) == 20
    assert MAX_SASA["A"] == 129.0
    assert MAX_SASA["W"] == 285.0
    assert MAX_SASA["G"] == 104.0


def test_align_pdb_to_target_seq_exact():
    pdb_seq = "ACDEFGHIKL"
    target_seq = "MMMZZACDEFGHIKLPPP"
    mapping = align_pdb_to_target_seq(pdb_seq, target_seq)
    # ACDEFGHIKL starts at target_seq index 5 (0-indexed) -> position 6 (1-indexed)
    assert mapping[0] == 6
    assert mapping[9] == 15
    assert len(mapping) == 10


def test_align_pdb_to_target_seq_with_mismatch():
    pdb_seq = "ACDEFGHIKL"
    target_seq = "ACDEFGXIKL"  # one mismatch
    mapping = align_pdb_to_target_seq(pdb_seq, target_seq)
    assert len(mapping) == 10
    assert mapping[0] == 1
    assert mapping[9] == 10


def test_analyze_complex_structure_6m0j():
    pdb_path = "data/structures/6M0J.pdb"
    df = analyze_complex_structure(pdb_path, target_chain_id="E", partner_chain_ids=["A"])
    assert len(df) > 100
    assert set(df["compartment"].unique()) == {"Interface", "Core", "Surface"}
    # Verify interface residues exist with dsasa >= 5 or dist <= 4.5
    int_df = df[df["compartment"] == "Interface"]
    assert len(int_df) >= 15
    for _, r in int_df.iterrows():
        assert r["dsasa"] >= 5.0 or r["min_dist"] <= 4.5

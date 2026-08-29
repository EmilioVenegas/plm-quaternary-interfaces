"""Coordinate mapping: alignment-derived, never an assumed offset.

Offline by construction. `build_map` only touches an in-memory Biopython aligner, so
these tests exercise the real code path with no files and no sockets. The single
network-dependent function, `pdb_ssbond`, is marked and skipped by default.
"""

import os

import pytest

from plmconfound.mapping import build_map, pdb_ssbond

# A non-repetitive stretch of sequence: distinct enough that a global alignment has a
# single sane solution, long enough that a spurious shifted alignment cannot compete
# with the leading-gap solution under BLAST's affine gap costs (-11 open, -1 extend).
CORE = "MKQWLVDFGHSRYPTNACEIKLMQVWFHDGRST"

# `pdb_ssbond` reaches out to files.rcsb.org. It is marked `network` *and* gated on an
# environment variable so the default selection is hermetic even when the marker is not
# deselected with `-m 'not network'`.
NETWORK_ENABLED = os.environ.get("PLMCONFOUND_NETWORK_TESTS") == "1"


def test_identical_sequences_give_the_identity_map():
    """Same sequence in, identity map out, identity 1.0, every residue aligned."""
    mp, ident, n_aligned = build_map(CORE, CORE)
    assert n_aligned == len(CORE), f"expected {len(CORE)} aligned pairs, got {n_aligned}"
    assert ident == pytest.approx(1.0), f"identity should be 1.0, got {ident}"
    assert mp == {i: i for i in range(1, len(CORE) + 1)}, "not the identity map"


def test_map_is_one_indexed():
    """Keys and values are 1-indexed, matching UniProt features and ProteinGym mutants."""
    mp, _ident, _n = build_map(CORE, CORE)
    assert 0 not in mp, "0 must not appear: the map is 1-indexed, not 0-indexed"
    assert min(mp) == 1 and max(mp) == len(CORE), f"key range is {min(mp)}..{max(mp)}"
    for u, t in mp.items():
        assert CORE[u - 1] == CORE[t - 1], f"1-indexing broken at {u}->{t}"


def test_map_recovers_a_known_offset():
    """A 3-residue N-terminal extension must shift every mapped position by exactly 3.

    This is the trap the module exists to prevent, and it is not hypothetical:

    * TEM-1 beta-lactamase carries `SSBOND CYS 77 - CYS 123` in PDB 1BTL (Ambler
      numbering), while the same two cysteines are at `target_seq` indices 75 and 121.
    * `MET_HUMAN_Estevam_2023` assays a kinase-domain-only construct, so most of MET's
      annotated phosphosites are not in the assayed region at all.

    In both cases reading the annotation's number as a sequence index would have scored
    the wrong residue -- silently, with plausible-looking output.
    """
    uniprot = "XXX" + CORE
    mp, ident, n_aligned = build_map(uniprot, CORE)
    assert n_aligned == len(CORE), f"the whole construct should align, got {n_aligned}"
    assert ident == pytest.approx(1.0), f"identity over aligned pairs should be 1.0, got {ident}"
    for i in range(1, len(CORE) + 1):
        assert mp.get(i + 3) == i, (
            f"uniprot position {i + 3} should map to target {i}, got {mp.get(i + 3)}"
        )
    # The extension itself has no counterpart in the construct and must not be invented.
    for i in (1, 2, 3):
        assert i not in mp, f"uniprot position {i} is outside the construct but was mapped"
    assert len(mp) == len(CORE), f"expected {len(CORE)} mapped positions, got {len(mp)}"


def test_map_survives_a_truncated_construct():
    """A construct that is an interior slice maps onto the correct UniProt window.

    The MET case: only part of the canonical sequence was assayed, and the unmeasurable
    positions must be *absent* from the map rather than mis-assigned.
    """
    start, stop = 10, 28  # 0-indexed slice bounds into CORE
    target = CORE[start:stop]
    mp, ident, n_aligned = build_map(CORE, target)
    assert n_aligned == len(target), f"expected {len(target)} aligned pairs, got {n_aligned}"
    assert ident == pytest.approx(1.0), f"identity should be 1.0, got {ident}"
    for i in range(1, len(target) + 1):
        assert mp.get(start + i) == i, (
            f"uniprot {start + i} should map to target {i}, got {mp.get(start + i)}"
        )
    assert 1 not in mp, "positions before the construct must not be mapped"
    assert len(CORE) not in mp, "positions after the construct must not be mapped"


def test_point_substitution_lowers_identity_but_keeps_the_mapping():
    """One mismatched residue is a construct variant, not a different protein.

    Identity is computed over aligned pairs only, so it drops to (L-1)/L while the
    coordinate correspondence stays intact. That is what makes identity usable as a
    sanity threshold rather than a coverage measure.
    """
    pos = 15  # 1-indexed position to mutate
    assert CORE[pos - 1] != "W", "pick a substitution that is actually a change"
    target = CORE[: pos - 1] + "W" + CORE[pos:]
    mp, ident, n_aligned = build_map(CORE, target)
    assert n_aligned == len(CORE), f"a point change must not open gaps, got {n_aligned}"
    assert mp == {i: i for i in range(1, len(CORE) + 1)}, "point change broke the mapping"
    expected = (len(CORE) - 1) / len(CORE)
    assert ident == pytest.approx(expected), f"expected identity {expected}, got {ident}"
    assert ident < 1.0, "a mismatch must be visible in the identity"


@pytest.mark.network
@pytest.mark.skipif(
    not NETWORK_ENABLED,
    reason="hits files.rcsb.org; run with PLMCONFOUND_NETWORK_TESTS=1 (or -m network)",
)
def test_pdb_ssbond_returns_ssbond_records():
    """TEM-1 (1BTL) has exactly one SSBOND record, in PDB author (Ambler) numbering."""
    records = pdb_ssbond("1BTL")
    assert records, "1BTL should report at least one SSBOND record"
    assert all(ln.startswith("SSBOND") for ln in records), f"non-SSBOND lines in {records}"
    assert any("CYS" in ln for ln in records), f"SSBOND record without CYS: {records}"

"""Data-layer pure functions and path wiring.

Only the parsing functions and the module-level paths are exercised by default: no
UniProt REST calls, and no reads of the ~1 GB ProteinGym bulk directory. The one test
that genuinely touches bulk data is marked `slow` and gated on an environment variable
so the default selection stays fast and hermetic.
"""

import os
from pathlib import Path

import pytest

from plmconfound.data import (
    BULK_DIR,
    PROTEINGYM_REF,
    load_assay,
    parse_features,
    parse_mutants,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SLOW_ENABLED = os.environ.get("PLMCONFOUND_SLOW_TESTS") == "1"

# One of the assays measured to carry usable PTM coverage; used only by the slow test.
SAMPLE_DMS_ID = "MK01_HUMAN_Brenan_2016"


# --------------------------------------------------------------------------------------
# parse_mutants
# --------------------------------------------------------------------------------------


def test_parse_mutants_single():
    """'A1B' -> exactly one (wt, pos, mut) triple, with an int position."""
    got = parse_mutants("A1B")
    assert got == [("A", 1, "B")], f"expected [('A', 1, 'B')], got {got}"
    wt, pos, mut = got[0]
    assert isinstance(wt, str) and len(wt) == 1, f"wt should be a single character, got {wt!r}"
    assert isinstance(mut, str) and len(mut) == 1, f"mut should be a single character, got {mut!r}"
    assert isinstance(pos, int) and not isinstance(pos, bool), (
        f"position must be an int, got {type(pos).__name__}"
    )


def test_parse_mutants_multi_is_colon_joined():
    """ProteinGym joins multi-mutants with ':'; each part becomes its own triple."""
    got = parse_mutants("A1B:C2D")
    assert got == [("A", 1, "B"), ("C", 2, "D")], f"got {got}"
    assert len(got) == 2, "a double mutant must yield two triples, not one merged record"
    assert all(isinstance(p, int) for _wt, p, _mut in got), "all positions must be ints"


def test_parse_mutants_multi_digit_positions():
    """Positions are parsed from the middle of the token, so they are not width-limited."""
    got = parse_mutants("C1138Y:S2041I")
    assert got == [("C", 1138, "Y"), ("S", 2041, "I")], f"got {got}"


def test_parse_mutants_triple_keeps_order():
    """Order is preserved: downstream code pairs positions with scores positionally."""
    got = parse_mutants("M1A:K10R:T100S")
    assert [p for _wt, p, _mut in got] == [1, 10, 100], f"order not preserved: {got}"


# --------------------------------------------------------------------------------------
# parse_features
# --------------------------------------------------------------------------------------


def _feature(ftype, start, end=None, description=""):
    """Synthetic UniProtKB feature in the REST JSON shape `parse_features` consumes."""
    end = start if end is None else end
    return {
        "type": ftype,
        "description": description,
        "location": {"start": {"value": start}, "end": {"value": end}},
    }


SYNTHETIC_ENTRY = {
    "primaryAccession": "P00000",
    "sequence": {"value": "MKQWNATSGGCCKK"},
    "features": [
        _feature("Modified residue", 5, description="Phosphoserine"),
        _feature("Modified residue", 7, description="Phosphothreonine"),
        _feature("Modified residue", 9, description="N6-acetyllysine"),
        _feature("Glycosylation", 11, description="N-linked (GlcNAc...) asparagine"),
        _feature("Disulfide bond", 11, 12, description=""),
        _feature("Disulfide bond", 13, 13, description="Interchain"),
        _feature("Active site", 4, description="Proton acceptor"),
    ],
}


def test_parse_features_buckets_phospho_by_description():
    """'Modified residue' is split on the description prefix; only phospho is in scope.

    UniProt files acetylation, hydroxylation, methylation and phosphorylation under one
    feature type. Only the phospho subset shares the Ser/Thr acceptor chemistry the
    family contrast is built on.
    """
    out = parse_features(SYNTHETIC_ENTRY)
    assert out["phospho"] == [(5, "Phosphoserine"), (7, "Phosphothreonine")], (
        f"phospho bucket wrong: {out['phospho']}"
    )
    for pos, desc in out["phospho"]:
        assert isinstance(pos, int), f"phospho position must be int, got {type(pos).__name__}"
        assert desc.lower().startswith("phospho"), f"non-phospho description in bucket: {desc!r}"


def test_parse_features_non_phospho_modres_is_segregated():
    """An acetyl-lysine must not leak into the phospho bucket."""
    out = parse_features(SYNTHETIC_ENTRY)
    assert out["other_modres"] == [(9, "N6-acetyllysine")], (
        f"other_modres bucket wrong: {out['other_modres']}"
    )
    assert (9, "N6-acetyllysine") not in out["phospho"], "acetylation leaked into phospho"


def test_parse_features_glycosylation_goes_to_carbohyd():
    """Glycosylation features land in `carbohyd` as (position, description)."""
    out = parse_features(SYNTHETIC_ENTRY)
    assert out["carbohyd"] == [(11, "N-linked (GlcNAc...) asparagine")], (
        f"carbohyd bucket wrong: {out['carbohyd']}"
    )


def test_parse_features_disulfide_is_a_three_tuple():
    """An intrachain bond is kept as (start, end, description) -- both partners needed."""
    out = parse_features(SYNTHETIC_ENTRY)
    assert len(out["disulfid"]) == 1, f"expected one usable bond, got {out['disulfid']}"
    bond = out["disulfid"][0]
    assert len(bond) == 3, f"disulfide record must be a 3-tuple, got {bond}"
    p1, p2, _desc = bond
    assert (p1, p2) == (11, 12), f"expected the (11, 12) bond, got {(p1, p2)}"
    assert isinstance(p1, int) and isinstance(p2, int), "bond positions must be ints"


def test_parse_features_skips_interchain_disulfide():
    """`start == end` marks an interchain bond: its partner is on another chain.

    There is no intra-construct Cys pair to mutate, so keeping it would inflate the
    bond inventory with pairs that cannot possibly have a double mutant.
    """
    out = parse_features(SYNTHETIC_ENTRY)
    assert all(p1 != p2 for p1, p2, _d in out["disulfid"]), (
        f"interchain (start == end) bond was not dropped: {out['disulfid']}"
    )
    assert 13 not in {p1 for p1, _p2, _d in out["disulfid"]}, "position 13 should be skipped"


def test_parse_features_ignores_unrelated_feature_types():
    """Active sites and other feature types are not collected into any bucket."""
    out = parse_features(SYNTHETIC_ENTRY)
    assert set(out) == {"phospho", "carbohyd", "disulfid", "other_modres"}, (
        f"unexpected bucket set: {sorted(out)}"
    )
    assert 4 not in {p for p, _d in out["phospho"] + out["other_modres"] + out["carbohyd"]}, (
        "the Active site feature was collected as a modification"
    )


def test_parse_features_entry_without_features_is_empty_not_error():
    """A missing or null `features` key is a real UniProt shape and must not raise.

    An empty result here has to be distinguishable from a failed lookup, which is what
    `repair_sequenceless_entries` exists to catch -- so this path returns empty buckets
    rather than throwing.
    """
    for entry in ({}, {"features": None}, {"features": []}):
        out = parse_features(entry)
        assert out == {"phospho": [], "carbohyd": [], "disulfid": [], "other_modres": []}, (
            f"expected empty buckets for {entry!r}, got {out}"
        )


# --------------------------------------------------------------------------------------
# path wiring
# --------------------------------------------------------------------------------------


def test_paths_are_absolute_and_inside_the_repo():
    """Both data paths are absolute and repo-relative, so scripts are cwd-independent."""
    for name, path in (("PROTEINGYM_REF", PROTEINGYM_REF), ("BULK_DIR", BULK_DIR)):
        assert isinstance(path, Path), f"{name} should be a Path, got {type(path).__name__}"
        assert path.is_absolute(), f"{name} must be absolute, got {path}"
        assert REPO_ROOT in path.parents, f"{name} ({path}) is not under the repo root {REPO_ROOT}"


def test_bulk_dir_layout():
    """The reference CSV and the bulk assay directory sit under data/proteingym/."""
    assert PROTEINGYM_REF.name == "DMS_substitutions_ref.csv", f"got {PROTEINGYM_REF.name}"
    assert BULK_DIR.name == "DMS_ProteinGym_substitutions", f"got {BULK_DIR.name}"
    assert PROTEINGYM_REF.parent == BULK_DIR.parent.parent, (
        f"unexpected layout: {PROTEINGYM_REF.parent} vs {BULK_DIR}"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not SLOW_ENABLED,
    reason="reads the ~1 GB ProteinGym bulk data; run with PLMCONFOUND_SLOW_TESTS=1",
)
def test_load_assay_reads_a_contributing_assay():
    """Smoke test against real bulk data: two columns, parseable mutant strings."""
    path = BULK_DIR / f"{SAMPLE_DMS_ID}.csv"
    assert path.exists(), f"bulk assay missing: {path}"
    df = load_assay(SAMPLE_DMS_ID)
    assert list(df.columns) == ["mutant", "DMS_score"], f"unexpected columns: {list(df.columns)}"
    assert len(df) > 0, f"{SAMPLE_DMS_ID} came back empty"
    triples = parse_mutants(df["mutant"].iloc[0])
    assert triples and all(isinstance(p, int) for _wt, p, _mut in triples), (
        f"unparseable mutant string: {df['mutant'].iloc[0]!r}"
    )

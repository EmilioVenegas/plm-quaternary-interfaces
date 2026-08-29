"""Regression guard on the committed reconnaissance numbers.

These are NOT arbitrary golden numbers. Every figure asserted below is a *published
claim* in `docs/recon/recon_ptm_disulfide_results.md`: the design's total observation
count, the limiting cell that sets its power, the two usable-position counts that
justify the arm selection, the structural zero at the Asn acceptor, the floor-censoring
audit that answers the assay-truncation threat, and the Track B result that the
disulfide-epistasis study is not feasible in existing public data.

Consequently a failure here is never "just update the expected value". It means one of:

  * a real regression -- a change to the coverage logic in `scripts/recon_ptm_disulfide.py`
    or in `plmconfound` silently moved the accounting, and the committed write-up no
    longer describes the code; or
  * a genuine new finding -- the inputs (ProteinGym release, UniProt annotations, MaveDB
    contents) changed and the study's coverage really is different, in which case the new
    numbers must be re-derived, re-checked against the pre-registration's inclusion rules,
    and written up. The write-up is the deliverable, not this file.

Either way the correct response is to investigate the diff, not to edit the constants.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "results" / "recon" / "recon_ptm_disulfide_summary.json"

pytestmark = pytest.mark.skipif(
    not SUMMARY_PATH.exists(),
    reason=(
        f"committed reconnaissance summary not found at {SUMMARY_PATH}; "
        "run scripts/recon_ptm_disulfide.py to regenerate it"
    ),
)


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY_PATH.read_text())


@pytest.fixture(scope="module")
def track_a(summary):
    return summary["trackA"]


@pytest.fixture(scope="module")
def track_b(summary):
    return summary["trackB"]


def _check(section, key, expected, actual):
    assert actual == expected, (
        f"{section}.{key}: committed {expected!r}, got {actual!r}. This figure is a "
        "published claim in docs/recon/recon_ptm_disulfide_results.md -- investigate the "
        "change, do not update the constant."
    )


# --------------------------------------------------------------------------------------
# Track A -- the PTM design that the confirmatory study is built on
# --------------------------------------------------------------------------------------


def test_track_a_design_size(track_a):
    """Headline design size and the cell that actually limits the test's power."""
    _check("trackA", "total_design_n", 14130, track_a["total_design_n"])
    _check("trackA", "ptm_site_n", 2847, track_a["ptm_site_n"])
    _check("trackA", "control_n", 11283, track_a["control_n"])
    _check("trackA", "ptm_limiting_cell", 260, track_a["ptm_limiting_cell"])
    # Internal consistency: the two site classes must exhaust the design.
    assert track_a["ptm_site_n"] + track_a["control_n"] == track_a["total_design_n"], (
        f"{track_a['ptm_site_n']} + {track_a['control_n']} != {track_a['total_design_n']}"
    )


def test_track_a_usable_positions(track_a):
    """The per-locus usable-position counts that determined which arms exist at all."""
    _check("trackA", "phospho_usable_positions", 40, track_a["phospho_usable_positions"])
    _check("trackA", "glyc_plus2_usable_positions", 110, track_a["glyc_plus2_usable_positions"])


def test_asn_acceptor_is_a_structural_zero(track_a):
    """Zero preserving substitutions at the sequon Asn -- the designed negative control.

    This zero is a chemistry prediction, not missing data: {N} is a one-member acceptor
    family, so no non-wild-type substitution at the acceptor can preserve the glycan.
    A non-zero here would mean the family definitions had drifted.
    """
    _check(
        "trackA",
        "asn_acceptor_preserving_substitutions_available",
        0,
        track_a["asn_acceptor_preserving_substitutions_available"],
    )
    assert track_a["asn_acceptor_positions_with_dms"] > 0, (
        "the zero is only meaningful if Asn acceptor positions were actually screened; "
        f"got {track_a['asn_acceptor_positions_with_dms']} positions with DMS coverage"
    )


def test_floor_censoring_audit(track_a):
    """The assay-floor threat check: 18 of 19 strata carry zero floor fraction."""
    cens = track_a["censoring"]
    _check("censoring", "strata_checked", 19, cens["strata_checked"])
    _check(
        "censoring",
        "strata_with_zero_floor_fraction",
        18,
        cens["strata_with_zero_floor_fraction"],
    )
    assert cens["strata_with_zero_floor_fraction"] <= cens["strata_checked"], (
        f"{cens['strata_with_zero_floor_fraction']} clean strata out of "
        f"{cens['strata_checked']} checked is impossible"
    )


def test_worst_case_censoring_drop_still_powers_the_design(track_a):
    """Dropping the single censored assay leaves 12,173 observations, as committed."""
    worst = track_a["censoring"]["worst_case_drop_censored_assay"]
    _check("worst_case_drop_censored_assay", "total_design_n", 12173, worst["total_design_n"])
    assert worst["total_design_n"] < track_a["total_design_n"], (
        "the worst-case drop must be smaller than the full design; got "
        f"{worst['total_design_n']} vs {track_a['total_design_n']}"
    )


# --------------------------------------------------------------------------------------
# Track B -- the disulfide-epistasis study, ruled out on feasibility
# --------------------------------------------------------------------------------------


def test_track_b_bond_inventory(track_b):
    """263 annotated bonds examined, 191 mapped with both cysteines verified in sequence."""
    _check("trackB", "annotated_bonds_examined", 263, track_b["annotated_bonds_examined"])
    _check(
        "trackB",
        "bonds_mapped_and_cys_verified",
        191,
        track_b["bonds_mapped_and_cys_verified"],
    )
    assert track_b["bonds_mapped_and_cys_verified"] <= track_b["annotated_bonds_examined"], (
        "more bonds verified than examined"
    )


def test_track_b_has_no_double_mutant_coverage(track_b):
    """Zero double mutants hitting both cysteines of any bond: the feasibility verdict.

    This is the number that killed the disulfide-epistasis design. It must stay a
    measured zero from an exhaustive scan, not a zero from an aborted one -- hence the
    accompanying scan-size assertions.
    """
    _check("trackB", "bonds_with_any_double_mutant", 0, track_b["bonds_with_any_double_mutant"])
    _check("megascale", "doubles_1plus_wt_cys", 0, track_b["megascale"]["doubles_1plus_wt_cys"])
    assert track_b["megascale"]["doubles"] > 0, (
        "the megascale zero is only informative if doubles were actually scanned; got "
        f"{track_b['megascale']['doubles']}"
    )
    assert track_b["total_multimutant_rows_scanned"] > 0, (
        "ProteinGym multi-mutant rows must have been scanned for the zero to mean anything"
    )


def test_track_b_mavedb_scan_is_complete(track_b):
    """The MaveDB sweep resolved every score set and found 26 two-Cys rows, all in GFP."""
    mave = track_b["mavedb"]
    _check("mavedb", "rows_hitting_two_wt_cys", 26, mave["rows_hitting_two_wt_cys"])
    assert mave["all_resolved"] is True, (
        "an unresolved MaveDB score set means the feasibility scan was incomplete, so a "
        f"zero cannot be claimed; got all_resolved={mave['all_resolved']!r}"
    )
    assert mave["multi_variant_rows_scanned"] > 0, (
        "no multi-variant rows scanned: the MaveDB arm of the scan did not run"
    )

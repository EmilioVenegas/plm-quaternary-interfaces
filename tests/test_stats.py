"""Statistics: within-assay standardisation, the interaction statistic, and its null.

Every frame here is hand-built so the expected value is *arithmetic* rather than a
golden number copied out of a previous run: each 2x2 cell mean is chosen to be exact in
binary floating point, so the difference-of-differences can be written out by hand in
the test and checked against the implementation.

Offline and fast: no data files, no models, no network.
"""

import numpy as np
import pandas as pd
import pytest

from plmconfound.stats import (
    add_residual,
    interaction_stat,
    stratified_permutation_test,
    within_assay_z,
)

# A permutation test is O(n_perm) evaluations of the statistic. The pipeline runs 10,000
# shuffles for a publishable p-value; the tests use 200 because they check *properties*
# (determinism under a fixed seed, null length, p in range, planted signal recovered)
# rather than a precise p-value, and 200 keeps the whole module well under a second.
N_PERM = 200


def _cell_frame(ptm_pres, ptm_abol, ctl_pres, ctl_abol, dms_id="A1"):
    """Build a 2x2 frame from four explicit lists of residuals, one per cell."""
    rows = []
    for residuals, is_ptm, is_pres in (
        (ptm_pres, True, True),
        (ptm_abol, True, False),
        (ctl_pres, False, True),
        (ctl_abol, False, False),
    ):
        for r in residuals:
            rows.append({"dms_id": dms_id, "residual": r, "is_ptm": is_ptm,
                         "is_preserving": is_pres})
    return pd.DataFrame(rows).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# within_assay_z
# --------------------------------------------------------------------------------------


def test_within_assay_z_centres_and_scales_within_group():
    """[1,2,3] has mean 2 and sample sd 1, so its z-scores are exactly [-1, 0, 1]."""
    df = pd.DataFrame({"DMS_id": ["A", "A", "A"], "DMS_score": [1.0, 2.0, 3.0]})
    z = within_assay_z(df, "DMS_id", "DMS_score")
    assert np.allclose(z.to_numpy(), [-1.0, 0.0, 1.0]), f"got {z.tolist()}"


def test_within_assay_z_groups_do_not_leak():
    """A ten-fold wider second assay must not shift or rescale the first one.

    This is the whole point of standardising per assay: under a *global* z-score group A
    would come out as roughly [-0.8, -0.7, -0.7] instead of [-1, 0, 1], because assay B's
    dynamic range would dominate the pooled mean and sd.
    """
    df = pd.DataFrame(
        {
            "DMS_id": ["A", "A", "A", "B", "B", "B"],
            "DMS_score": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    z = within_assay_z(df, "DMS_id", "DMS_score").to_numpy()
    assert np.allclose(z[:3], [-1.0, 0.0, 1.0]), f"group A leaked: {z[:3].tolist()}"
    assert np.allclose(z[3:], [-1.0, 0.0, 1.0]), f"group B leaked: {z[3:].tolist()}"
    # Same shape, different units -> identical standardised values. That equality is
    # exactly the commensurability the residual analysis needs.
    assert np.allclose(z[:3], z[3:]), "identically shaped assays must standardise alike"


def test_within_assay_z_is_mean_zero_unit_sd_per_group():
    """Per-group summary statistics, on an asymmetric group to catch a pooled-sd bug."""
    df = pd.DataFrame(
        {
            "DMS_id": ["A"] * 4 + ["B"] * 5,
            "DMS_score": [0.5, -3.0, 7.25, 1.0, 100.0, 101.5, 99.0, 98.25, 103.0],
        }
    )
    df["z"] = within_assay_z(df, "DMS_id", "DMS_score")
    for name, grp in df.groupby("DMS_id"):
        assert abs(grp["z"].mean()) < 1e-12, f"group {name} not centred: {grp['z'].mean()}"
        assert abs(grp["z"].std() - 1.0) < 1e-12, f"group {name} not unit sd: {grp['z'].std()}"


def test_within_assay_z_singleton_group_is_nan():
    """A one-row assay carries no information about its own spread, so z must be NaN.

    Returning 0.0 instead would inject a spurious "exactly average" row into every cell
    mean downstream.
    """
    df = pd.DataFrame({"DMS_id": ["A", "B", "B"], "DMS_score": [5.0, 1.0, 3.0]})
    z = within_assay_z(df, "DMS_id", "DMS_score").to_numpy()
    assert np.isnan(z[0]), f"singleton group should be NaN, got {z[0]}"
    assert not np.isnan(z[1:]).any(), "two-row group must still standardise"


# --------------------------------------------------------------------------------------
# add_residual
# --------------------------------------------------------------------------------------


def test_add_residual_is_dms_z_minus_zs_z():
    """`residual` is defined once, here, as DMS_z - zs_z; nothing may re-derive it."""
    df = pd.DataFrame(
        {
            "DMS_id": ["A", "A", "A", "B", "B", "B"],
            "DMS_score": [1.0, 2.0, 3.0, -4.0, 0.0, 4.0],
            "zeroshot_score": [-1.0, 0.0, 1.0, 10.0, 12.0, 14.0],
        }
    )
    out = add_residual(df)
    assert {"DMS_z", "zs_z", "residual"} <= set(out.columns), f"columns: {list(out.columns)}"
    expected = out["DMS_z"].to_numpy() - out["zs_z"].to_numpy()
    assert np.allclose(out["residual"].to_numpy(), expected), "residual != DMS_z - zs_z"


def test_add_residual_cancels_when_model_tracks_assay_exactly():
    """If the model's ranking is an affine image of the measurements, residuals vanish.

    Standardisation removes scale and offset, so a model that is perfectly calibrated up
    to a linear transform has zero error in these units. That is the null the interaction
    statistic is measured against.
    """
    df = pd.DataFrame(
        {
            "DMS_id": ["A"] * 4,
            "DMS_score": [1.0, 2.0, 3.0, 4.0],
            "zeroshot_score": [-10.0, -8.0, -6.0, -4.0],  # = 2 * DMS_score - 12
        }
    )
    out = add_residual(df)
    assert np.allclose(out["residual"].to_numpy(), 0.0, atol=1e-12), (
        f"affinely calibrated model should leave no residual, got {out['residual'].tolist()}"
    )


def test_add_residual_returns_same_object():
    """Documented to annotate in place and return the frame; call sites rely on both."""
    df = pd.DataFrame(
        {"DMS_id": ["A", "A"], "DMS_score": [1.0, 2.0], "zeroshot_score": [0.0, 1.0]}
    )
    assert add_residual(df) is df, "add_residual must return the frame it annotated"


# --------------------------------------------------------------------------------------
# interaction_stat
# --------------------------------------------------------------------------------------


def test_interaction_stat_matches_hand_computed_cell_means():
    """Four exact cell means -> the expected statistic is arithmetic, not a golden number.

        ptm/preserving   [3.0, 5.0] -> 4.0
        ptm/abolishing   [0.0, 1.0] -> 0.5
        ctl/preserving   [2.0, 4.0] -> 3.0
        ctl/abolishing   [1.0, 3.0] -> 2.0

        (4.0 - 0.5) - (3.0 - 2.0) = 2.5
    """
    df = _cell_frame([3.0, 5.0], [0.0, 1.0], [2.0, 4.0], [1.0, 3.0])
    got = interaction_stat(df)
    assert got == pytest.approx(2.5), f"expected 2.5, got {got}"


def test_interaction_stat_is_zero_when_gaps_match():
    """Equal preserving-minus-abolishing gaps at both site classes must cancel exactly.

    This is the confound the design exists to break: a model that is simply worse at
    conserved columns produces the same inner gap at PTM sites and at matched controls,
    and the outer subtraction sends it to zero.
    """
    #   ptm gap = 1.5 - 0.5 = 1.0 ; control gap = 3.0 - 2.0 = 1.0
    df = _cell_frame([1.0, 2.0], [0.0, 1.0], [2.0, 4.0], [1.0, 3.0])
    got = interaction_stat(df)
    assert got == pytest.approx(0.0, abs=1e-12), f"expected ~0, got {got}"


def test_interaction_stat_sign_follows_the_ptm_specific_excess():
    """A PTM-specific *excess* gap is positive; a PTM-specific deficit is negative."""
    excess = _cell_frame([4.0], [0.0], [1.0], [0.0])   # 4.0 - 1.0
    deficit = _cell_frame([1.0], [0.0], [4.0], [0.0])  # 1.0 - 4.0
    assert interaction_stat(excess) == pytest.approx(3.0)
    assert interaction_stat(deficit) == pytest.approx(-3.0)


def test_interaction_stat_ignores_nan_residuals():
    """NaN residuals (e.g. singleton-assay z-scores) are skipped, not propagated."""
    df = _cell_frame([3.0, 5.0, float("nan")], [0.0, 1.0], [2.0, 4.0], [1.0, 3.0])
    got = interaction_stat(df)
    assert got == pytest.approx(2.5), f"NaN row must not change the statistic, got {got}"


def test_interaction_stat_empty_cell_is_nan_not_zero():
    """An empty preserving cell is the one-member-acceptor-family case: NaN, never 0.

    At a Tyr phosphosite or a sequon Asn no preserving substitution exists at all. Such a
    configuration must not be reported as "zero effect measured".
    """
    df = _cell_frame([], [0.0, 1.0], [2.0, 4.0], [1.0, 3.0])
    got = interaction_stat(df)
    assert np.isnan(got), f"empty cell must give NaN, got {got}"


# --------------------------------------------------------------------------------------
# stratified_permutation_test
# --------------------------------------------------------------------------------------


def _strata(df):
    """Row-index blocks as the pipeline builds them: one block per (assay, site-class)."""
    return list(df.groupby(["dms_id", "is_ptm"]).groups.values())


def _planted_frame(n=20):
    """PTM sites: preserving = +2, abolishing = -2. Controls: no gap at all.

    Observed statistic = (2 - (-2)) - (0 - 0) = 4.0, which is the extreme of its own
    stratified null (only the identity and the fully flipped labelling reach it), so the
    p-value must come out at the resolution floor.
    """
    return _cell_frame([2.0] * n, [-2.0] * n, [0.0] * n, [0.0] * n)


def _null_frame(n=20):
    """Residuals vary but are *balanced* across families inside each stratum.

    Each family gets the same multiset of residuals, so the observed gap is exactly zero
    at both site classes while the shuffled null still has real spread. That makes the
    expected p-value 1.0 by construction rather than by luck.
    """
    half = n // 2
    return _cell_frame(
        [2.0] * half + [-2.0] * half,
        [2.0] * half + [-2.0] * half,
        [1.0] * half + [-1.0] * half,
        [1.0] * half + [-1.0] * half,
    )


def test_permutation_test_is_deterministic_for_a_fixed_seed():
    """Same seed, same frame -> byte-identical null and p. Reruns must be reproducible."""
    df = _planted_frame()
    strata = _strata(df)
    obs1, p1, null1 = stratified_permutation_test(df, strata, n_perm=N_PERM, seed=7)
    obs2, p2, null2 = stratified_permutation_test(df, strata, n_perm=N_PERM, seed=7)
    assert obs1 == obs2, f"observed statistic is not a pure function: {obs1} vs {obs2}"
    assert p1 == p2, f"p-value not reproducible under a fixed seed: {p1} vs {p2}"
    assert np.array_equal(null1, null2, equal_nan=True), "null distribution not reproducible"


def test_permutation_test_shapes_and_ranges():
    """Contract: p in [0,1], null of length n_perm, observed == interaction_stat."""
    df = _planted_frame()
    observed, p, null = stratified_permutation_test(df, _strata(df), n_perm=N_PERM, seed=0)
    assert null.shape == (N_PERM,), f"null must have length {N_PERM}, got {null.shape}"
    assert 0.0 <= p <= 1.0, f"p out of range: {p}"
    assert observed == pytest.approx(interaction_stat(df)), (
        "the permutation test must report the same observed statistic as interaction_stat"
    )


def test_permutation_test_recovers_a_planted_interaction():
    """A large planted PTM-specific gap must be significant against its own null."""
    df = _planted_frame()
    observed, p, null = stratified_permutation_test(df, _strata(df), n_perm=N_PERM, seed=0)
    assert observed == pytest.approx(4.0), f"planted statistic should be 4.0, got {observed}"
    assert p < 0.05, f"planted interaction should be significant, got p={p}"
    # The planted labelling is the extreme of its own stratified null: with 10 (+2) and
    # 10 (-2) residuals per PTM stratum, the gap is 0.8k - 4 in the number k of (+2) rows
    # labelled preserving, so |gap| can never exceed the observed 4.0.
    assert np.abs(null).max() <= abs(observed) + 1e-12, (
        f"null exceeded the attainable maximum: {np.abs(null).max()} vs {abs(observed)}"
    )
    assert null.std() > 0.0, "stratified shuffle produced a degenerate null distribution"


def test_permutation_test_does_not_flag_a_null_frame():
    """Chemistry-blind residuals must not produce significance, however they are shuffled."""
    df = _null_frame()
    observed, p, null = stratified_permutation_test(df, _strata(df), n_perm=N_PERM, seed=0)
    assert observed == pytest.approx(0.0, abs=1e-12), f"null frame should give ~0, got {observed}"
    assert p > 0.05, f"null frame must not be significant, got p={p}"
    # The shuffle has to actually do something, otherwise the previous line is vacuous.
    assert null.std() > 0.0, "stratified shuffle produced a degenerate null distribution"


def test_permutation_shuffle_stays_inside_strata():
    """Labels are permuted within strata only, so per-stratum label counts are invariant.

    An unstratified shuffle would let the null absorb between-assay and site-vs-control
    composition differences, and a statistic driven purely by which positions were
    selected would still look significant.
    """
    df = _planted_frame()
    before = df.groupby(["dms_id", "is_ptm"])["is_preserving"].sum().to_dict()
    stratified_permutation_test(df, _strata(df), n_perm=N_PERM, seed=1)
    after = df.groupby(["dms_id", "is_ptm"])["is_preserving"].sum().to_dict()
    assert before == after, (
        f"the input frame's labels must be left untouched: {before} -> {after}"
    )


def test_permutation_test_handles_multiple_assays():
    """Two assays with opposite baselines: strata keep each assay's own level out of the null."""
    a = _cell_frame([2.0] * 10, [-2.0] * 10, [0.0] * 10, [0.0] * 10, dms_id="A1")
    b = _cell_frame([12.0] * 10, [8.0] * 10, [10.0] * 10, [10.0] * 10, dms_id="B2")
    df = pd.concat([a, b], ignore_index=True).reset_index(drop=True)
    strata = _strata(df)
    assert len(strata) == 4, f"expected 4 (assay, site-class) strata, got {len(strata)}"
    observed, p, null = stratified_permutation_test(df, strata, n_perm=N_PERM, seed=0)
    # Both assays carry the same +4 PTM-specific gap after their own baselines cancel.
    assert observed == pytest.approx(4.0), f"expected 4.0, got {observed}"
    assert p < 0.05, f"consistent planted effect across assays should be significant, p={p}"
    assert null.shape == (N_PERM,)

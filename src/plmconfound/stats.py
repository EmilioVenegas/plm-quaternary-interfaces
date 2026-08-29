"""
The confound test's statistics: within-assay residuals, the interaction statistic and
its stratified permutation null.

The quantity of interest is not "does the model score PTM sites badly" -- deep mutational
scanning assays disagree wildly about scale, sign convention and dynamic range, so any
raw comparison across assays measures assay design rather than model behaviour. Instead
each assay is standardised twice, once on the measured fitness and once on the model's
zero-shot score, and the residual ``DMS_z - zs_z`` is the *per-mutation model error* in
units of that assay's own spread: positive means the model was more optimistic than
reality, negative means more pessimistic. Standardising within assay is what makes
residuals from a beta-lactamase and a calmodulin assay commensurable at all.

On top of that the test is a difference of differences, not a difference. A PLM is
expected to be worse at conserved positions in general, so the preserving-minus-
abolishing residual gap measured at PTM sites is compared against the same gap measured
at matched non-PTM control positions. Only the *excess* gap -- the interaction --
distinguishes "the model encodes modification chemistry" from "the model is bad at
conserved columns", which is the confound the study exists to break.
"""

import numpy as np
import pandas as pd


def within_assay_z(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Z-score ``value_col`` within each level of ``group_col``.

    Per-assay standardisation, not global: DMS assays are reported in incomparable
    units (log-enrichment, growth rate, ddG, fluorescence), and a global z-score would
    let the assay with the widest dynamic range dominate every downstream mean.

    Uses the pandas default ddof=1 sample standard deviation, matching the metal study
    whose published numbers this refactor must reproduce. A single-row group therefore
    yields NaN rather than 0, which is correct: one mutation carries no information
    about its assay's spread and must not enter the means as a spurious zero.
    """
    return df.groupby(group_col)[value_col].transform(lambda x: (x - x.mean()) / x.std())


def add_residual(
    df: pd.DataFrame,
    dms_col: str = "DMS_score",
    zs_col: str = "zeroshot_score",
    group_col: str = "DMS_id",
) -> pd.DataFrame:
    """Attach ``DMS_z``, ``zs_z`` and ``residual = DMS_z - zs_z`` in place, return ``df``.

    The residual is the model's signed error against measured fitness after both have
    been put on the same within-assay scale. It is the analysis unit for every test
    below, so it is computed exactly once here rather than re-derived per call site --
    a second, subtly different residual definition elsewhere would silently change the
    committed interaction statistic.
    """
    df["DMS_z"] = within_assay_z(df, group_col, dms_col)
    df["zs_z"] = within_assay_z(df, group_col, zs_col)
    df["residual"] = df["DMS_z"] - df["zs_z"]
    return df


def _cell_mean(resid: np.ndarray, mask: np.ndarray) -> float:
    """Mean residual over one 2x2 cell, NaN for an empty cell.

    Mirrors ``Series.mean()``: NaN residuals are skipped and a cell with no usable rows
    returns NaN instead of raising or silently contributing 0. An empty cell is the
    one-member-acceptor-family case from ``chemistry.preserving_family``, and it must
    propagate as NaN so that a site with no possible preserving substitution cannot be
    mistaken for a site with a zero-sized preserving effect.
    """
    sel = resid[mask]
    return float(sel.mean()) if sel.size else float("nan")


def _interaction_from_arrays(resid: np.ndarray, site: np.ndarray, fam: np.ndarray) -> float:
    """Difference-of-differences kernel shared by the statistic and its permutation null.

    Factored out so the 10,000-shuffle loop does not copy a DataFrame per iteration; the
    arithmetic is identical to the pandas path in ``interaction_stat``.
    """
    ok = ~np.isnan(resid)
    return (
        (_cell_mean(resid, ok & site & fam) - _cell_mean(resid, ok & site & ~fam))
        - (_cell_mean(resid, ok & ~site & fam) - _cell_mean(resid, ok & ~site & ~fam))
    )


def interaction_stat(
    df: pd.DataFrame,
    site_col: str = "is_ptm",
    fam_col: str = "is_preserving",
    value_col: str = "residual",
) -> float:
    """Excess preserving-minus-abolishing residual gap at PTM sites over control sites.

    ``(site & preserving) - (site & abolishing)`` minus the same contrast at non-site
    positions. The outer subtraction is the point of the whole design: a PLM that is
    merely bad at conserved columns produces the same inner gap at both site classes and
    cancels to zero here, so a non-zero value can only come from chemistry that is
    specific to the modified positions.
    """
    resid = df[value_col].to_numpy(dtype=float)
    site = df[site_col].to_numpy(dtype=bool)
    fam = df[fam_col].to_numpy(dtype=bool)
    return _interaction_from_arrays(resid, site, fam)


def stratified_permutation_test(
    df: pd.DataFrame,
    strata,
    n_perm: int = 10000,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Two-sided permutation p-value for ``interaction_stat``, shuffling within strata.

    The preserving/abolishing label is permuted *inside* each stratum only, where a
    stratum is normally one (assay, site-class) group. That holds fixed everything the
    null must not be allowed to exploit: the assay's dynamic range, its baseline model
    accuracy, and the site/control composition. An unstratified shuffle would let the
    null distribution absorb between-assay and site-vs-control differences, so a large
    observed statistic driven purely by which positions were selected would still look
    significant. The metal study's shuffled-label control is the empirical form of the
    same argument -- randomising site identity put the true statistic at the 45th
    percentile of its own null, showing the design generates no significance on its own.

    ``strata`` is an iterable of row-index arrays as produced by
    ``df.groupby([...]).groups.values()``; those are index *labels* used here as
    positions, so ``df`` must carry a default RangeIndex (call ``reset_index(drop=True)``
    first if it does not).

    Returns ``(observed, p_value, null_distribution)``, with the p-value computed on
    absolute values because either sign is a departure from "chemistry-blind" -- the
    direction is a separate, pre-registered check, not part of the test.
    """
    rng = np.random.default_rng(seed)
    resid = df["residual"].to_numpy(dtype=float)
    site = df["is_ptm"].to_numpy(dtype=bool)
    fam = df["is_preserving"].to_numpy(dtype=bool)
    observed = _interaction_from_arrays(resid, site, fam)

    blocks = [np.asarray(idx) for idx in strata]
    null = np.zeros(n_perm)
    shuffled = fam.copy()
    for i in range(n_perm):
        shuffled[:] = fam
        for idx in blocks:
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[i] = _interaction_from_arrays(resid, site, shuffled)
    p = float((np.abs(null) >= np.abs(observed)).mean())
    return observed, p, null

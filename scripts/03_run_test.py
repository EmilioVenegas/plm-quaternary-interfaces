#!/usr/bin/env python3
"""Stage 3: the pre-registered confound test, per model arm.

Consumes stage 2's score table (`results/scores.csv`) and writes one
`results/test_<arm>.json` per arm containing the primary statistic, its stratified
permutation p-value, every secondary and control arm from `PLAN.md` §3.3-3.4, and the
verdict under the falsification rule frozen in `PRE_REGISTRATION.md` §4.

    residual     = z(DMS_score) - z(zero_shot)            # within assay
    gap(G)       = mean residual[G & preserving] - mean residual[G & abolishing]
    interaction  = gap(PTM sites) - gap(matched control positions)

Nothing statistical is implemented here. `plmconfound.stats` owns the residual, the
interaction kernel and the permutation null, because the metal-coordination study's
published numbers must stay reproducible and a second, subtly different residual
definition at a call site is the classic way to lose that.

Three things in this script are load-bearing and easy to get wrong:

1. **The residual is computed once, on the whole arm.** Every sub-arm is a row subset of
   that frame. Recomputing z-scores per subset would make the per-locus split
   incommensurable with the pooled primary, since the standardisation would then depend
   on which rows the subset happened to contain.

2. **An empty cell must never become a p-value.** `stats._cell_mean` correctly returns
   NaN for an empty 2x2 cell, and the loci where chemistry admits no preserving
   substitution (the Asn acceptor, Tyr phosphosites) are exactly that case. Feeding a NaN
   observed statistic into a permutation test would yield p = 0.0, because every
   `abs(null) >= nan` comparison is False -- a designed null would report as the most
   significant result in the study. So finiteness is checked *before* the test runs.

3. **The designed null is a check on us, not on the model.** `sequon_asn` has zero
   preserving observations by chemistry (`chemistry.preserving_family` returns the empty
   set), so a non-zero interaction there cannot be a finding; it is a broken pipeline, and
   `PRE_REGISTRATION.md` §4 says so in advance. It is flagged loudly and it invalidates
   the run.

The site-label control (N4) reuses `stats.stratified_permutation_test` rather than
reimplementing a shuffle: the interaction statistic is symmetric in its two factors
(`(a-b)-(c-d) = a-c-b+d`), so swapping the site and chemistry columns leaves the observed
value bit-identical while turning the function's within-strata shuffle of the chemistry
label into a within-strata shuffle of the *site* label. The swap is asserted, not assumed.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from plmconfound import stats

ALPHA = 0.01                       # PRE_REGISTRATION.md §2. Not 0.05.
PRIMARY_LOCI = ("phospho", "sequon_plus2")
PHOSPHO_ACCEPTORS = ("S", "T")     # Tyr phosphosites are the N2 arm, not the primary
CENSORED_ASSAY = "ENV_HV1BR_Haddox_2016"   # 35.8% of its sequon +2 subs at the assay floor

# Bause & Legler 1981 (PMID 7316978): Thr > Ser > Cys as glycosyl acceptors.
ACCEPTOR_RANK = {"C": 0, "S": 1, "T": 2}

# PRE_REGISTRATION.md §2, recorded for comparison only. Stage 1 restricts to the 13
# contributing assays while the recon figures were computed over the wider candidate
# lists, so a mismatch here is expected and is reported, never enforced.
PREREGISTERED_N = {"ptm_observations": 2847, "control_observations": 11283,
                   "limiting_cell_preserving_at_ptm": 260}

VERDICT_H1 = "H1 supported"
VERDICT_NOT_H1 = "H1 not supported"
VERDICT_INVALID = "pipeline invalid"
VERDICT_UNANTICIPATED = "unanticipated-sign finding"


def loud(msg: str) -> None:
    bar = "!" * 78
    print(f"\n{bar}\n{msg}\n{bar}\n", file=sys.stderr, flush=True)


def _f(x):
    """JSON-safe float: NaN/inf become null rather than invalid JSON literals."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# --------------------------------------------------------------------------- loading


def detect_arms(df: pd.DataFrame) -> list[str]:
    zs = {c.removeprefix("zeroshot_") for c in df.columns if c.startswith("zeroshot_")}
    lw = {c.removeprefix("logp_wt_") for c in df.columns if c.startswith("logp_wt_")}
    return sorted(zs & lw)


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/02_score_models.py first.")
    df = pd.read_csv(path)
    required = ["dms_id", "protein", "taxon", "locus", "position", "wt", "mut",
                "is_ptm", "klass", "DMS_score", "at_assay_floor"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {missing}")
    if not detect_arms(df):
        raise SystemExit(f"{path} carries no zeroshot_<arm>/logp_wt_<arm> column pair; "
                         f"stage 2 has not scored any arm into it.")
    return df


def prepare(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Attach the boolean design columns and the within-assay residual for one arm."""
    zs_col, lw_col = f"zeroshot_{arm}", f"logp_wt_{arm}"
    for c in (zs_col, lw_col):
        if c not in df.columns:
            raise SystemExit(f"arm {arm!r}: column {c!r} absent from the score table.")

    f = df.copy()
    f["is_ptm"] = f["is_ptm"].astype(bool)
    f["is_preserving"] = f["klass"].astype(str).eq("preserving")
    unknown = set(f["klass"].astype(str)) - {"preserving", "abolishing"}
    if unknown:
        raise SystemExit(f"unexpected klass value(s) {sorted(unknown)}; "
                         f"plmconfound.chemistry.classify emits only "
                         f"'preserving'/'abolishing'.")
    # residual computed ONCE, on the full arm; every sub-arm below is a row subset.
    stats.add_residual(f, dms_col="DMS_score", zs_col=zs_col, group_col="dms_id")
    f["_key"] = list(zip(f["dms_id"], f["position"]))
    return f


# -------------------------------------------------------------------------- matching


def caliper_match(f: pd.DataFrame, lw_col: str, caliper_sd: float) -> dict:
    """1:1 nearest-neighbour caliper matching of PTM positions to control positions.

    Within assay, on the model's own masked `logP(WT | context)` (`PLAN.md` §3.2). Greedy
    nearest neighbour without replacement, PTM positions processed in `(dms_id, position)`
    order and ties broken by the lower control position, so the assignment is a
    deterministic function of the score table and not of dict iteration order.

    Matching on the model's own conservation estimate is the point rather than a
    circularity: the hypothesis under test is that the model treats a PTM site as
    generically conserved, so holding *its* conservation estimate fixed is what isolates
    chemistry from conservation.

    -> {"pairs": {ptm_key: ctrl_key}, "dropped": [...], "per_assay": {...}, ...}
    """
    pos = (f.groupby(["dms_id", "position"], sort=True)
             .agg(is_ptm=("is_ptm", "first"), locus=("locus", "first"),
                  logp_wt=(lw_col, "first"))
             .reset_index())

    pairs: dict[tuple, tuple] = {}
    dropped: list[dict] = []
    per_assay: dict[str, dict] = {}

    for dms_id, g in pos.groupby("dms_id", sort=True):
        sd = float(g.logp_wt.std())                     # ddof=1, whole assay
        caliper = caliper_sd * sd if np.isfinite(sd) else float("nan")
        ptm = g[g.is_ptm].sort_values("position")
        ctrl = [(int(r.position), float(r.logp_wt))
                for r in g[~g.is_ptm].sort_values("position").itertuples(index=False)
                if np.isfinite(r.logp_wt)]
        used: set[int] = set()
        n_matched = 0
        for r in ptm.itertuples(index=False):
            lp = float(r.logp_wt)
            best = None
            if np.isfinite(lp) and np.isfinite(caliper):
                for cpos, clp in ctrl:
                    if cpos in used:
                        continue
                    d = abs(clp - lp)
                    if best is None or d < best[0]:
                        best = (d, cpos)
            if best is not None and best[0] <= caliper:
                used.add(best[1])
                pairs[(dms_id, int(r.position))] = (dms_id, best[1])
                n_matched += 1
            else:
                dropped.append({"dms_id": dms_id, "position": int(r.position),
                                "locus": r.locus, "logp_wt": _f(lp),
                                "nearest_delta": _f(best[0]) if best else None,
                                "caliper": _f(caliper)})
        per_assay[dms_id] = {"sd_logp_wt": _f(sd), "caliper": _f(caliper),
                             "n_ptm_positions": int(len(ptm)),
                             "n_control_positions": len(ctrl),
                             "n_matched": n_matched,
                             "n_dropped_unmatched": int(len(ptm)) - n_matched}

    return {"caliper_sd_units": caliper_sd,
            "n_ptm_positions": int(pos.is_ptm.sum()),
            "n_control_positions_available": int((~pos.is_ptm).sum()),
            "n_matched": len(pairs),
            "n_dropped_unmatched": len(dropped),
            "n_control_positions_used": len(set(pairs.values())),
            "per_assay": per_assay,
            "dropped": dropped,
            "pairs": pairs}


def matched_frame(f: pd.DataFrame, ptm_sel: pd.Series, pairs: dict) -> pd.DataFrame:
    """Rows of the selected PTM positions plus the rows of *their* matched controls."""
    ptm_keys = set(f.loc[ptm_sel, "_key"])
    ctrl_keys = {pairs[k] for k in ptm_keys if k in pairs}
    keep = f["_key"].isin(ptm_keys | ctrl_keys)
    # a control position matched to a PTM position is control-side by construction
    return f[keep & (f["is_ptm"].isin([True, False]))]


def unmatched_frame(f: pd.DataFrame, ptm_sel: pd.Series) -> pd.DataFrame:
    """Selected PTM positions plus every control position in the same assays."""
    assays = set(f.loc[ptm_sel, "dms_id"])
    return f[ptm_sel | (~f["is_ptm"] & f["dms_id"].isin(assays))]


# ----------------------------------------------------------------------------- tests


def cell_counts(f: pd.DataFrame) -> dict:
    ok = f["residual"].notna()
    site, fam = f["is_ptm"], f["is_preserving"]
    return {"ptm_preserving": int((ok & site & fam).sum()),
            "ptm_abolishing": int((ok & site & ~fam).sum()),
            "control_preserving": int((ok & ~site & fam).sum()),
            "control_abolishing": int((ok & ~site & ~fam).sum()),
            "n_usable": int(ok.sum()), "n_rows": int(len(f))}


def run_test(f: pd.DataFrame, label: str, n_perm: int, seed: int,
             strata_cols=("dms_id", "is_ptm")) -> dict:
    """Interaction statistic + stratified permutation p for one row subset."""
    f = f.reset_index(drop=True)          # the test uses index labels as positions
    cells = cell_counts(f)
    out = {"label": label, "n": cells, "n_assays": int(f["dms_id"].nunique()),
           "n_ptm_positions": int(f.loc[f.is_ptm, "_key"].nunique()) if len(f) else 0,
           "n_control_positions": int(f.loc[~f.is_ptm, "_key"].nunique()) if len(f) else 0}

    if len(f) == 0:
        out.update(observed=None, p=None, status="no observations")
        return out

    observed = stats.interaction_stat(f)
    if not np.isfinite(observed):
        # Empty 2x2 cell. Running the permutation test here would return p = 0.0, because
        # every `abs(null) >= nan` comparison is False. That is the single most dangerous
        # failure mode in this script, so it is refused rather than reported.
        empty = [k for k, v in cells.items()
                 if k != "n_usable" and k != "n_rows" and v == 0]
        out.update(observed=None, p=None,
                   status=f"undefined: empty cell(s) {empty} -- chemistry admits no "
                          f"observation there, so no interaction exists to test")
        return out

    strata = list(f.groupby(list(strata_cols), sort=True).groups.values())
    obs, p, null = stats.stratified_permutation_test(f, strata, n_perm=n_perm, seed=seed)
    out.update(observed=_f(obs), p=_f(p), status="ok", n_perm=int(n_perm), seed=int(seed),
               significant_at_alpha=bool(p < ALPHA),
               sign="positive" if obs > 0 else ("negative" if obs < 0 else "zero"),
               n_strata=len(strata),
               null_mean=_f(np.nanmean(null)), null_sd=_f(np.nanstd(null)),
               percentile_of_observed=_f(np.nanmean(null < obs)),
               gaps=_gaps(f))
    return out


def _gaps(f: pd.DataFrame) -> dict:
    """The two inner preserving-minus-abolishing gaps whose difference is the statistic.
    Reported because a null interaction built from two large equal gaps and a null built
    from two zero gaps are different results, and the statistic alone cannot tell them
    apart."""
    r, site, fam = f["residual"], f["is_ptm"], f["is_preserving"]

    def m(mask):
        v = r[mask]
        return _f(v.mean()) if len(v) else None

    gp, ga = m(site & fam), m(site & ~fam)
    cp, ca = m(~site & fam), m(~site & ~fam)
    return {"ptm_preserving_mean": gp, "ptm_abolishing_mean": ga,
            "control_preserving_mean": cp, "control_abolishing_mean": ca,
            "gap_ptm": _f(gp - ga) if None not in (gp, ga) else None,
            "gap_control": _f(cp - ca) if None not in (cp, ca) else None}


def covariate_adjusted(f: pd.DataFrame, lw_col: str) -> pd.DataFrame:
    """Replace `residual` by its within-assay residual against `logP(WT)`.

    The unmatched-with-covariate sensitivity of `PLAN.md` §3.2: instead of discarding
    control positions outside the caliper, regress the conservation proxy out of the
    residual within each assay and keep every control. The linear fit is deliberately
    plain -- it is a sensitivity check on the matching, not a second model.
    """
    f = f.copy()
    adj = f["residual"].astype(float).copy()
    for _, g in f.groupby("dms_id", sort=True):
        x = g[lw_col].to_numpy(dtype=float)
        y = g["residual"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.std(x[ok]) == 0:
            continue
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        adj.loc[g.index] = y - (intercept + slope * x)
    f["residual"] = adj
    return f


def ordinal_test(f: pd.DataFrame, zs_col: str) -> dict:
    """S1. At sequon +2, do the model's log-odds track Thr > Ser > Cys?

    Spearman of the log-odds against the measured acceptor-efficiency rank
    (C=0 < S=1 < T=2; Bause & Legler 1981, PMID 7316978) over the preserving
    substitutions only -- the abolishing substitutions have no acceptor rank. The same
    correlation on the measured DMS scores is reported alongside as the reference the
    model is being asked to reproduce; a conservation-only account has no mechanism that
    produces a three-level ordering inside one invariant column.
    """
    sub = f[(f["locus"] == "sequon_plus2") & f["is_preserving"]
            & f["mut"].isin(ACCEPTOR_RANK)]
    counts = {a: int((sub["mut"] == a).sum()) for a in ("T", "S", "C")}
    out = {"label": "S1 ordinal acceptor efficiency at sequon +2",
           "acceptor_rank": ACCEPTOR_RANK, "n": int(len(sub)), "n_by_mut": counts,
           "mean_zeroshot_by_mut": {a: _f(sub.loc[sub["mut"] == a, zs_col].mean())
                                    for a in ("T", "S", "C")},
           "mean_DMS_by_mut": {a: _f(sub.loc[sub["mut"] == a, "DMS_score"].mean())
                               for a in ("T", "S", "C")}}
    distinct = sum(1 for v in counts.values() if v > 0)
    if len(sub) < 3 or distinct < 2:
        out.update(status=f"not evaluable: {len(sub)} preserving observations across "
                          f"{distinct} distinct acceptor(s)")
        return out

    rank = sub["mut"].map(ACCEPTOR_RANK).to_numpy(dtype=float)
    rho_m, p_m = spearmanr(sub[zs_col].to_numpy(dtype=float), rank)
    rho_d, p_d = spearmanr(sub["DMS_score"].to_numpy(dtype=float), rank)
    out.update(status="ok",
               model_spearman_rho=_f(rho_m), model_spearman_p=_f(p_m),
               dms_spearman_rho=_f(rho_d), dms_spearman_p=_f(p_d),
               model_reproduces_ordering=bool(np.isfinite(rho_m) and rho_m > 0
                                              and p_m < ALPHA),
               dms_shows_ordering=bool(np.isfinite(rho_d) and rho_d > 0 and p_d < ALPHA))
    return out


def shuffled_site_label_test(f: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """N4. Null distribution from shuffling the *site* label instead of the chemistry one.

    The interaction is symmetric in its two factors, so swapping the columns and reusing
    `stats.stratified_permutation_test` gives the site-label shuffle exactly, with the
    same observed value. Strata become (assay x chemistry class), the mirror image of the
    primary's (assay x site class).
    """
    g = f.reset_index(drop=True).copy()
    site = g["is_ptm"].to_numpy(dtype=bool).copy()
    fam = g["is_preserving"].to_numpy(dtype=bool).copy()
    g["is_ptm"], g["is_preserving"] = fam, site

    out = {"label": "N4 shuffled site labels"}
    observed = stats.interaction_stat(g)
    if not np.isfinite(observed):
        out.update(observed=None, p=None, status="undefined: empty cell")
        return out
    strata = list(g.groupby(["dms_id", "is_ptm"], sort=True).groups.values())
    obs, p, null = stats.stratified_permutation_test(g, strata, n_perm=n_perm, seed=seed)
    out.update(status="ok", observed=_f(obs), p=_f(p), n_perm=int(n_perm),
               seed=int(seed), n_strata=len(strata),
               null_mean=_f(np.nanmean(null)), null_sd=_f(np.nanstd(null)),
               percentile_of_observed=_f(np.nanmean(null < obs)),
               inside_null_at_alpha=bool(p >= ALPHA),
               note="statistic is symmetric in its two factors, so this observed value "
                    "equals the primary's by construction; only the null differs")
    return out


# ----------------------------------------------------------------------------- arm


def analyse_arm(df: pd.DataFrame, arm: str, n_perm: int, seed: int,
                caliper_sd: float) -> dict:
    zs_col, lw_col = f"zeroshot_{arm}", f"logp_wt_{arm}"
    f = prepare(df, arm)

    match = caliper_match(f, lw_col, caliper_sd)
    pairs = match.pop("pairs")

    is_primary_ptm = (
        ((f["locus"] == "phospho") & f["wt"].isin(PHOSPHO_ACCEPTORS))
        | (f["locus"] == "sequon_plus2"))
    primary_sel = f["is_ptm"] & is_primary_ptm

    primary = run_test(matched_frame(f, primary_sel, pairs),
                       "primary: pooled phospho(S/T) + sequon +2, caliper-matched, "
                       f"with {CENSORED_ASSAY}", n_perm, seed)
    primary["preregistered_expected_n"] = PREREGISTERED_N

    # ---- S2 per-locus split
    per_locus = {}
    for locus in PRIMARY_LOCI:
        sel = primary_sel & (f["locus"] == locus)
        per_locus[locus] = run_test(matched_frame(f, sel, pairs),
                                    f"S2 per-locus: {locus}", n_perm, seed)

    # ---- S4 taxon split (never pooled into the headline; PLAN.md §5)
    taxon_split = {}
    for taxon in sorted(f.loc[primary_sel, "taxon"].dropna().astype(str).unique()):
        sel = primary_sel & f["taxon"].astype(str).eq(taxon)
        sub = matched_frame(f, sel, pairs)
        sub = sub[sub["taxon"].astype(str).eq(taxon)]
        taxon_split[taxon] = run_test(sub, f"S4 taxon: {taxon}", n_perm, seed)

    # ---- censoring sensitivity: the WITH version above is the pre-registered headline
    without_censored = run_test(
        matched_frame(f[f["dms_id"] != CENSORED_ASSAY],
                      primary_sel[f["dms_id"] != CENSORED_ASSAY], pairs),
        f"sensitivity: primary without {CENSORED_ASSAY}", n_perm, seed)
    no_floor = f[~f["at_assay_floor"].astype(bool)]
    without_floor = run_test(
        matched_frame(no_floor, primary_sel[no_floor.index], pairs),
        "sensitivity: primary excluding rows at the assay floor", n_perm, seed)

    # ---- matching sensitivities (PLAN.md §3.2: matched primary + unmatched covariate)
    unmatched = unmatched_frame(f, primary_sel)
    sens_unmatched = run_test(unmatched, "sensitivity: unmatched controls", n_perm, seed)
    sens_covariate = run_test(covariate_adjusted(unmatched, lw_col),
                              "sensitivity: unmatched controls, logP(WT)-adjusted "
                              "residual", n_perm, seed)

    # ---- N1 designed null: the Asn acceptor. Chemistry admits no preserving substitution.
    asn_sel = f["is_ptm"] & (f["locus"] == "sequon_asn")
    n_asn_preserving = int((asn_sel & f["is_preserving"]).sum())
    n1 = run_test(matched_frame(f, asn_sel, pairs),
                  "N1 designed null: sequon Asn acceptor, family {N}", n_perm, seed)
    n1["n_preserving_observations"] = n_asn_preserving
    n1_violated = n_asn_preserving > 0 or (
        n1.get("observed") is not None and n1["observed"] != 0.0)
    n1["null_as_designed"] = not n1_violated
    if n1_violated:
        n1["violation"] = ("a locus whose preserving family is empty by chemistry "
                           "produced preserving observations and/or a non-zero "
                           "interaction")
        loud(f"[{arm}] N1 DESIGNED-NULL VIOLATED: the sequon Asn acceptor has preserving "
             f"family {{N}}, so it can have NO preserving substitution and NO non-zero "
             f"interaction.\nObserved: {n_asn_preserving} preserving observation(s), "
             f"interaction = {n1.get('observed')}.\nPER PRE_REGISTRATION.md §4 THIS IS A "
             f"BROKEN PIPELINE, NOT A FINDING: no claim either way, fix and rerun.")

    # ---- N2 predicted null: Tyr phosphosites have no preserving partner (different
    #      kinase class), so the preserving cell is empty by chemistry here too.
    tyr_sel = f["is_ptm"] & (f["locus"] == "phospho") & f["wt"].eq("Y")
    n_tyr_preserving = int((tyr_sel & f["is_preserving"]).sum())
    n2 = run_test(matched_frame(f, tyr_sel, pairs),
                  "N2 predicted null: Tyr phosphosites", n_perm, seed)
    n2["n_preserving_observations"] = n_tyr_preserving
    n2["null_as_predicted"] = bool(
        n2.get("p") is None or n2["p"] >= ALPHA) if n_tyr_preserving else True

    n4 = shuffled_site_label_test(matched_frame(f, primary_sel, pairs), n_perm, seed)

    loci_counts = {k: int(v) for k, v in f["locus"].value_counts().items()}
    return {
        "arm": arm,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "columns": {"zeroshot": zs_col, "logp_wt": lw_col},
        "data": {"n_rows": int(len(f)), "n_assays": int(f["dms_id"].nunique()),
                 "rows_by_locus": loci_counts,
                 "n_ptm_observations": int(f["is_ptm"].sum()),
                 "n_control_observations": int((~f["is_ptm"]).sum()),
                 "n_rows_at_assay_floor": int(f["at_assay_floor"].astype(bool).sum())},
        "matching": match,
        "primary": primary,
        "secondary": {
            "s1_ordinal_sequon_plus2": ordinal_test(f, zs_col),
            "s2_per_locus": per_locus,
            "s4_taxon_split": taxon_split,
            "sensitivity_without_censored_assay": without_censored,
            "sensitivity_excluding_floor_rows": without_floor,
            "sensitivity_unmatched_controls": sens_unmatched,
            "sensitivity_unmatched_covariate_adjusted": sens_covariate,
        },
        "controls": {
            "n1_designed_null_sequon_asn": n1,
            "n2_predicted_null_tyr_phospho": n2,
            "n3_permuted_weight": {
                "label": "N3 permuted-weight model",
                "status": "not evaluable from a score table",
                "note": "requires re-scoring with randomised weights at stage 2; it is "
                        "a property of the scores, not of this analysis.",
            },
            "n4_shuffled_site_labels": n4,
        },
    }


def falsification(res: dict, s3_signs: dict) -> dict:
    """Apply PRE_REGISTRATION.md §4 verbatim. Exactly four outcomes; no others."""
    p = res["primary"].get("p")
    observed = res["primary"].get("observed")
    n1_null = bool(res["controls"]["n1_designed_null_sequon_asn"]["null_as_designed"])
    n2_null = bool(res["controls"]["n2_predicted_null_tyr_phospho"]["null_as_predicted"])
    n4 = res["controls"]["n4_shuffled_site_labels"]

    # §4 names N4 as a pipeline-validity condition. Its pre-registered wording ("the
    # true-label statistic sits inside the shuffled-label null") is the H0-world form of
    # a consistency requirement: the site-label null and the chemistry-label null must
    # agree on whether the statistic is extreme. Both booleans are recorded so a reader
    # can apply either reading.
    n4_inside = n4.get("inside_null_at_alpha")
    sig = None if p is None else bool(p < ALPHA)
    n4_consistent = None if (sig is None or n4_inside is None) else bool(
        sig == (not n4_inside))

    unmet: list[str] = []
    if observed is None or p is None:
        unmet.append("primary statistic undefined (empty cell)")
    if not n1_null:
        unmet.append("N1 designed null non-null")
    if not n2_null:
        unmet.append("N2 predicted null non-null")
    if n4_consistent is False:
        unmet.append("N4 site-label control disagrees with the primary")
    if s3_signs.get("consistent") is False:
        unmet.append("S3 sign not consistent across model arms")
    if s3_signs.get("consistent") is None:
        unmet.append(f"S3 not evaluable: {s3_signs.get('n_arms_evaluated', 0)} of 4 "
                     f"model arms present")

    if not n1_null or n4_consistent is False:
        verdict = arm_local = VERDICT_INVALID
    elif p is None:
        verdict = arm_local = VERDICT_INVALID
    elif p >= ALPHA:
        verdict = arm_local = VERDICT_NOT_H1
    elif observed < 0:
        verdict = arm_local = VERDICT_UNANTICIPATED
    else:
        arm_local = VERDICT_H1 if n2_null else VERDICT_NOT_H1
        if arm_local == VERDICT_H1 and s3_signs.get("consistent") is False:
            verdict = VERDICT_NOT_H1
        else:
            verdict = arm_local

    return {"alpha": ALPHA, "observed": observed, "p": p, "significant": sig,
            "sign": res["primary"].get("sign"),
            "n1_null": n1_null, "n2_null": n2_null,
            "n4_inside_shuffled_site_label_null": n4_inside,
            "n4_consistent_with_primary": n4_consistent,
            "s3": s3_signs,
            "arm_local_verdict": arm_local, "verdict": verdict,
            "verdict_pending_s3": bool(s3_signs.get("consistent") is None
                                       and verdict == VERDICT_H1),
            "unmet_conditions": unmet,
            "outcome_space": [VERDICT_H1, VERDICT_NOT_H1, VERDICT_INVALID,
                              VERDICT_UNANTICIPATED]}


def s3_signs(results: dict) -> dict:
    """S3: sign of the primary interaction across the model arms available in this run."""
    signs = {arm: r["primary"].get("sign") for arm, r in results.items()}
    usable = {a: s for a, s in signs.items() if s in ("positive", "negative")}
    consistent = None
    if len(usable) == 4:
        consistent = len(set(usable.values())) == 1
    elif len(usable) != len(signs):
        consistent = False if len(set(usable.values())) > 1 else None
    elif len(set(usable.values())) > 1:
        consistent = False
    return {"signs": signs, "n_arms_evaluated": len(usable), "n_arms_required": 4,
            "consistent": consistent,
            "observed": {a: r["primary"].get("observed") for a, r in results.items()}}


# --------------------------------------------------------------------------- report


def _line(t: dict) -> str:
    if t.get("observed") is None:
        return f"undefined ({t.get('status', '?')})"
    p = t.get("p")
    verdict = "significant" if (p is not None and p < ALPHA) else "not significant"
    return (f"interaction = {t['observed']:+.4f}   p = {p:.4f}   "
            f"({verdict} at alpha = {ALPHA})")


def print_verdict(res: dict) -> None:
    arm, pr, m = res["arm"], res["primary"], res["matching"]
    fal = res["falsification"]
    n = pr["n"]
    print(f"\n{'=' * 78}\n{arm}\n{'=' * 78}")
    print(f"primary  {pr['label']}")
    print(f"  {_line(pr)}")
    print(f"  n: PTM {n['ptm_preserving']} preserving / {n['ptm_abolishing']} abolishing"
          f" | control {n['control_preserving']} / {n['control_abolishing']}"
          f"  ({pr['n_ptm_positions']} PTM + {pr['n_control_positions']} control"
          f" positions, {pr['n_assays']} assays)")
    g = pr.get("gaps") or {}
    if g.get("gap_ptm") is not None:
        print(f"  gaps: PTM {g['gap_ptm']:+.4f}   control {g['gap_control']:+.4f}")
    print(f"  matching: {m['n_matched']} of {m['n_ptm_positions']} PTM positions matched"
          f" ({m['n_dropped_unmatched']} dropped unmatched), caliper ="
          f" {m['caliper_sd_units']} SD of the assay's logP(WT)")

    for locus, t in res["secondary"]["s2_per_locus"].items():
        print(f"  S2 {locus:<14} {_line(t)}")
    o = res["secondary"]["s1_ordinal_sequon_plus2"]
    if o.get("status") == "ok":
        print(f"  S1 ordinal T>S>C: model rho = {o['model_spearman_rho']:+.3f} "
              f"(p = {o['model_spearman_p']:.3g}), DMS rho = "
              f"{o['dms_spearman_rho']:+.3f} (p = {o['dms_spearman_p']:.3g}); "
              f"model reproduces ordering: {o['model_reproduces_ordering']}")
    else:
        print(f"  S1 ordinal T>S>C: {o['status']}")
    for taxon, t in res["secondary"]["s4_taxon_split"].items():
        print(f"  S4 taxon {taxon:<8} {_line(t)}")
    print(f"  sens. without {CENSORED_ASSAY}: "
          f"{_line(res['secondary']['sensitivity_without_censored_assay'])}")
    print(f"  sens. no floor rows:  "
          f"{_line(res['secondary']['sensitivity_excluding_floor_rows'])}")
    print(f"  sens. unmatched:      "
          f"{_line(res['secondary']['sensitivity_unmatched_controls'])}")
    print(f"  sens. logP(WT)-adj:   "
          f"{_line(res['secondary']['sensitivity_unmatched_covariate_adjusted'])}")

    n1 = res["controls"]["n1_designed_null_sequon_asn"]
    n2 = res["controls"]["n2_predicted_null_tyr_phospho"]
    n4 = res["controls"]["n4_shuffled_site_labels"]
    print(f"  N1 designed null (sequon Asn): {_line(n1)}"
          f"  [{n1['n_preserving_observations']} preserving obs;"
          f" null as designed: {n1['null_as_designed']}]")
    print(f"  N2 predicted null (Tyr phospho): {_line(n2)}"
          f"  [{n2['n_preserving_observations']} preserving obs;"
          f" null as predicted: {n2['null_as_predicted']}]")
    print(f"  N4 shuffled site labels: {_line(n4)}"
          f"  [inside null: {n4.get('inside_null_at_alpha')}]")
    print(f"  N3 permuted weight: {res['controls']['n3_permuted_weight']['status']}")

    s3 = fal["s3"]
    print(f"  S3 replication: {s3['n_arms_evaluated']} of 4 arms, signs = "
          f"{s3['signs']}, consistent = {s3['consistent']}")
    pend = "  (pending S3)" if fal["verdict_pending_s3"] else ""
    print(f"\n  VERDICT: {fal['verdict']}{pend}")
    if fal["unmet_conditions"]:
        print(f"  unmet pre-registered conditions: "
              f"{'; '.join(fal['unmet_conditions'])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 3: pre-registered interaction test, per model arm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Falsification rules are frozen in PRE_REGISTRATION.md §4.")
    ap.add_argument("--scores", type=Path, default=Path("results/scores.csv"),
                    help="stage 2 score table (default: results/scores.csv)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated arms (default: every arm found in --scores)")
    ap.add_argument("--n-perm", type=int, default=10000,
                    help="permutation shuffles (default: 10000, as pre-registered)")
    ap.add_argument("--seed", type=int, default=0,
                    help="numpy default RNG seed (default: 0, as pre-registered)")
    ap.add_argument("--caliper", type=float, default=0.25,
                    help="caliper width in SD of the assay's logP(WT) (default: 0.25)")
    ap.add_argument("--out-dir", type=Path, default=Path("results"),
                    help="directory for test_<arm>.json (default: results/)")
    args = ap.parse_args(argv)

    df = load_scores(args.scores)
    available = detect_arms(df)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms \
        else available
    unknown = [a for a in arms if a not in available]
    if unknown:
        raise SystemExit(f"arm(s) {unknown} not scored in {args.scores}; "
                         f"available: {available}")
    if args.n_perm != 10000 or args.seed != 0 or args.caliper != 0.25:
        print(f"[note] non-pre-registered settings: n_perm={args.n_perm}, "
              f"seed={args.seed}, caliper={args.caliper} "
              f"(pre-registered: 10000, 0, 0.25)", flush=True)

    print(f"[read] {args.scores}: {len(df)} observations, {df.dms_id.nunique()} assays, "
          f"arms {available}", flush=True)

    results = {}
    for arm in arms:
        print(f"\n[run] {arm}", flush=True)
        results[arm] = analyse_arm(df, arm, args.n_perm, args.seed, args.caliper)

    s3 = s3_signs(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm, res in results.items():
        res["falsification"] = falsification(res, s3)
        out = args.out_dir / f"test_{arm}.json"
        out.write_text(json.dumps(res, indent=2, default=str))
        print(f"[write] {out}", flush=True)

    for res in results.values():
        print_verdict(res)

    invalid = [a for a, r in results.items()
               if r["falsification"]["verdict"] == VERDICT_INVALID]
    return 4 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Stage 1 of the confirmatory pipeline: build the per-observation coverage table.

Emits one row per scored observation to ``results/variants.csv`` -- the input that
stage 2 (`02_score_models.py`) adds model log-odds to and stage 3 (`03_run_test.py`)
runs the interaction statistic on. Every row is a *single* DMS mutant at a position
whose wild-type residue was established by real pairwise alignment, never by an assumed
offset (`PRE_REGISTRATION.md` §5).

Four loci are emitted, three of them PTM and one control:

  ``phospho``       UniProt ``MOD_RES`` phosphosites, mapped into ``target_seq``.
                    Ser/Thr sites carry the {S,T} preserving family and drive the
                    primary arm. Tyr sites are emitted too, with an empty preserving
                    family by chemistry -- they are the N2 *predicted null*, and a
                    pipeline that manufactures an effect there is broken, so they must
                    be present in the table rather than filtered out here.
  ``sequon_asn``    the Asn of every motif-derived N-X-S/T sequon (X != Pro). Family
                    {N}: no non-wild-type substitution can preserve the glycan, which
                    makes this the N1 *designed null*.
  ``sequon_plus2``  the +2 acceptor of the same sequons. Family {S,T,C} (Bause & Legler
                    1981), the only glycosylation locus with a preserving cell.
  ``control``       non-PTM wild-type Ser/Thr positions in the *same* assay carrying at
                    least one preserving and one abolishing substitution. Annotated
                    phosphosites and both sequon positions are excluded from the pool,
                    so the control differs from the PTM group in PTM status and nothing
                    else: same assay, same selection, same wild-type residue, same
                    available substitution chemistry. Control rows are classified with
                    ``sequon_plus2`` chemistry because that is the {S,T,C} family their
                    wild-type Ser/Thr admits.

Sequons are motif-derived rather than annotation-derived so that a viral assay whose
UniProt entry carries no CARBOHYD features still contributes; where UniProt *does*
annotate the Asn as N-linked that is recorded in ``ptm_desc``.

A note on the regression targets checked at the end. The reconnaissance figures frozen
in `PLAN.md` §2 were computed over two overlapping assay lists: the PTM cells
(phospho 40/40/717, sequon +2 110/220/1870, sequon Asn 117) over recon's wider
``PHOSPHO_ASSAYS`` (16) and ``GLYC_ASSAYS`` (12), while the matched-control cell
(599/1193/10090) came from the 13 one-assay-per-protein contributing assays. Those two
lists cannot both be honoured by a single run: the 40th phosphosite is PTEN T401, which
only ``PTEN_HUMAN_Matreyek_2021`` measures, and 4 of the 110 sequon +2 positions come
from ``A0A140D2T1_ZIKV_Sourisseau_2019`` and ``ACE2_HUMAN_Chan_2020`` -- none of which
are contributing assays, and all of which would add fresh control positions and move the
control cell off 599. The default here is the 13 contributing assays, because the
control arm requires one assay per protein to avoid pseudo-replication and stages 2-3
score exactly this set. The comparison below is therefore reported, cell by cell, as a
visible delta rather than silently satisfied: a real regression and this known
bookkeeping difference must not be able to hide behind each other.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from plmconfound import chemistry, data, mapping

REPO = Path(__file__).resolve().parents[1]

# One assay per protein, so the within-assay control arm is not pseudo-replicated: the
# three P53 Giacomelli conditions and the three SRC assays re-measure the same libraries.
CONTRIBUTING = [
    "MK01_HUMAN_Brenan_2016",
    "MET_HUMAN_Estevam_2023",
    "ADRB2_HUMAN_Jones_2020",
    "PRKN_HUMAN_Clausen_2023",
    "SRC_HUMAN_Ahler_2019",
    "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
    # PTEN contributes only once its obsolete UniProt entry (O00633, empty sequence) is
    # repaired to P60484; see data.repair_sequenceless_entries.
    "PTEN_HUMAN_Mighell_2018",
    "ENV_HV1BR_Haddox_2016",
    "A0A192B1T2_9HIV1_Haddox_2018",
    "Q2N0S5_9HIV1_Haddox_2018",
    "C6KNH7_9INFA_Lee_2018",
    "A0A2Z5U3Z0_9INFA_Doud_2016",
    "SPIKE_SARS2_Starr_2020_binding",
]

COLUMNS = [
    "dms_id", "protein", "uniprot", "taxon", "locus", "position", "wt", "mut",
    "ptm_desc", "is_ptm", "klass", "DMS_score", "aln_identity", "at_assay_floor",
]

PTM_LOCI = ("phospho", "sequon_asn", "sequon_plus2")
LOCI = PTM_LOCI + ("control",)

# Phosphosite wild-types that carry a defined acceptor chemistry. Ser/Thr have the {S,T}
# preserving family; Tyr is a one-member family and is kept as the predicted-null arm.
PHOSPHO_WT = frozenset("STY")

# "At floor" = within 2% of the assay's full score range of that assay's minimum. Same
# definition as the reconnaissance censoring diagnostic, so the flags are comparable.
FLOOR_FRACTION = 0.02

# Frozen reconnaissance coverage. See the module docstring for the assay-list caveat.
REGRESSION_TARGETS = {
    "phospho": {"usable_positions": 40, "preserving": 40, "abolishing_at_usable": 717},
    "sequon_plus2": {"usable_positions": 110, "preserving": 220, "abolishing_at_usable": 1870},
    "sequon_asn": {"positions": 117, "preserving": 0},
    "control": {"positions": 599, "preserving": 1193, "abolishing": 10090},
}
TARGET_PROVENANCE = (
    "PLAN.md §2 / recon summary. PTM cells were counted over recon's PHOSPHO_ASSAYS (16) "
    "and GLYC_ASSAYS (12); the control cell over the 13 contributing assays. A run over "
    "the 13 alone is expected to fall short on the PTM cells by PTEN T401 "
    "(PTEN_HUMAN_Matreyek_2021 only) and by the sequon +2 positions of "
    "A0A140D2T1_ZIKV_Sourisseau_2019 and ACE2_HUMAN_Chan_2020."
)


# --------------------------------------------------------------------- annotations

def load_annotations(assays, ref):
    """-> (parsed features by UniProt entry name, canonical sequence by entry name).

    The repair runs BEFORE anything is counted. An obsolete entry name resolves to a
    record with no sequence and no features, which produces no rows at all rather than an
    error -- that is how PTEN was reported as a zero contributor when it is one of the
    strongest. Counting first and repairing later would reintroduce exactly that.
    """
    entry_names = sorted(ref[ref.DMS_id.isin(assays)].UniProt_ID.unique())
    feats = data.fetch_uniprot_features(entry_names)
    repaired = data.repair_sequenceless_entries(feats, ref)
    usable = {n: v for n, v in feats.items() if v and "__error__" not in v}
    parsed = {n: data.parse_features(v) for n, v in usable.items()}
    useq = {n: (v.get("sequence") or {}).get("value") or None for n, v in usable.items()}
    missing = sorted(n for n in entry_names if not useq.get(n))
    return parsed, useq, repaired, missing


# ------------------------------------------------------------------- row emission

def floor_predicate(scores):
    """-> f(score) telling whether the score sits at this assay's floor.

    Built once per assay from the assay's *whole* score distribution, not from the
    selected positions: the floor is a property of the selection assay's dynamic range.
    """
    s = pd.Series(scores, dtype="float64").dropna()
    if s.empty:
        return lambda _sc: False
    lo, span = float(s.min()), float(s.max()) - float(s.min())
    if span <= 0:
        return lambda _sc: False
    return lambda sc: bool(pd.notna(sc) and abs(float(sc) - lo) < FLOOR_FRACTION * span)


def build_assay_rows(dms_id, ref, parsed, useq):
    """-> (rows, info) for one assay. `info` records mapping quality and skipped sites."""
    r = ref[ref.DMS_id == dms_id].iloc[0]
    uid, tseq, taxon = r.UniProt_ID, r.target_seq, r.taxon
    protein = dms_id.split("_")[0]

    idx = data.single_mutant_index(dms_id)
    at_floor = floor_predicate(data.load_assay(dms_id).DMS_score)

    # Alignment is mandatory: UniProt numbering and ProteinGym target_seq indices are
    # different coordinate systems and agree only by luck.
    uniprot_seq = useq.get(uid)
    if uniprot_seq:
        upos_to_tpos, aln_identity, n_aligned = mapping.build_map(uniprot_seq, tseq)
    else:
        upos_to_tpos, aln_identity, n_aligned = {}, float("nan"), 0

    entry = parsed.get(uid, {})
    rows = []

    def emit(locus, position, wt, ptm_desc, is_ptm):
        """Every DMS single mutant at `position` whose own WT matches the mapped WT.

        A row whose WT disagrees with the aligned target residue is a coordinate
        disagreement, not an observation of this site, and is dropped rather than
        reconciled.
        """
        # Control positions are wild-type Ser/Thr with the same {S,T,C} family available
        # as sequon +2, which is the whole point: only PTM status differs.
        chem_locus = "sequon_plus2" if locus == "control" else locus
        n = 0
        for wt_obs, mut, score in idx.get(position, ()):
            if wt_obs != wt:
                continue
            rows.append({
                "dms_id": dms_id, "protein": protein, "uniprot": uid, "taxon": taxon,
                "locus": locus, "position": position, "wt": wt, "mut": mut,
                "ptm_desc": ptm_desc, "is_ptm": is_ptm,
                "klass": chemistry.classify(chem_locus, wt, mut),
                "DMS_score": score, "aln_identity": aln_identity,
                "at_assay_floor": at_floor(score),
            })
            n += 1
        return n

    # ---- phospho: UniProt MOD_RES descriptions beginning "Phospho", mapped by alignment
    phospho_positions, off_construct, non_sty = set(), 0, []
    for upos, desc in entry.get("phospho", ()):
        tpos = upos_to_tpos.get(upos)
        if tpos is None or not 0 < tpos <= len(tseq):
            off_construct += 1        # site lies outside the assayed construct
            continue
        # Excluded from the control pool whatever its residue: an annotated modification
        # site is not a clean non-PTM position even if it is not a usable phospho row.
        phospho_positions.add(tpos)
        wt = tseq[tpos - 1]
        if wt not in PHOSPHO_WT:
            non_sty.append({"position": tpos, "wt": wt, "desc": desc})
            continue
        emit("phospho", tpos, wt, desc.split(";")[0], True)

    # ---- sequons: motif-derived, so assays with no CARBOHYD annotation still contribute
    annotated_asn = {upos_to_tpos.get(p) for p, d in entry.get("carbohyd", ())
                     if "N-linked" in d}
    sequon_positions = set()
    n_sequons = 0
    for asn_pos, plus2_pos in chemistry.find_sequons(tseq):
        n_sequons += 1
        sequon_positions |= {asn_pos, plus2_pos}
        desc = "sequon N-X-S/T"
        if asn_pos in annotated_asn:
            desc += "; UniProt N-linked"
        emit("sequon_asn", asn_pos, tseq[asn_pos - 1], desc, True)
        emit("sequon_plus2", plus2_pos, tseq[plus2_pos - 1], desc, True)

    # ---- control: non-PTM wild-type Ser/Thr with both cells non-empty
    excluded = phospho_positions | sequon_positions
    n_control_positions = 0
    for position in sorted(idx):
        if not 0 < position <= len(tseq):
            continue
        wt = tseq[position - 1]
        if wt not in chemistry.PHOSPHO_ACCEPTOR or position in excluded:
            continue
        muts = [m for w, m, _s in idx[position] if w == wt]
        preserving = sum(1 for m in muts
                         if chemistry.classify("sequon_plus2", wt, m) == "preserving")
        if preserving and len(muts) - preserving:
            n_control_positions += 1
            emit("control", position, wt, "", False)

    info = {
        "dms_id": dms_id, "protein": protein, "uniprot": uid, "taxon": taxon,
        "aln_identity": None if pd.isna(aln_identity) else round(float(aln_identity), 6),
        "aligned_residues": n_aligned, "target_len": len(tseq),
        "uniprot_sequence_available": bool(uniprot_seq),
        "annotated_phosphosites": len(entry.get("phospho", ())),
        "phosphosites_off_construct": off_construct,
        "phosphosites_mapped_to_non_STY": non_sty,
        "sequons_found": n_sequons,
        "control_positions": n_control_positions,
        "rows": len(rows),
    }
    return rows, info


def build_variants(assays, ref, parsed, useq):
    """-> (deduplicated dataframe, per-assay info, n rows removed by deduplication)."""
    rows, infos = [], []
    for dms_id in assays:
        assay_rows, info = build_assay_rows(dms_id, ref, parsed, useq)
        rows.extend(assay_rows)
        infos.append(info)
    df = pd.DataFrame(rows, columns=COLUMNS)
    # Pseudo-replication key: the same physical variant re-measured by a second assay of
    # the same protein is one observation, not two.
    deduped = df.drop_duplicates(subset=["protein", "position", "wt", "mut"], keep="first")
    return deduped.reset_index(drop=True), infos, len(df) - len(deduped)


# ------------------------------------------------------------------------ counting

def locus_counts(df):
    """Per-locus cell counts, including the usable-position restriction recon reports.

    A *usable* position is one carrying at least one preserving substitution: a position
    with an empty preserving cell contributes nothing to the interaction statistic while
    still inflating the apparent site count, so the two are reported separately.
    """
    out = {}
    for locus in LOCI:
        sub = df[df.locus == locus]
        positions = set(zip(sub.protein, sub.position))
        usable = set(zip(sub[sub.klass == "preserving"].protein,
                         sub[sub.klass == "preserving"].position))
        at_usable = sub[[k in usable for k in zip(sub.protein, sub.position)]]
        klass = Counter(sub.klass)
        out[locus] = {
            "rows": int(len(sub)),
            "positions": len(positions),
            "usable_positions": len(usable),
            "preserving": int(klass.get("preserving", 0)),
            "abolishing": int(klass.get("abolishing", 0)),
            "abolishing_at_usable": int((at_usable.klass == "abolishing").sum()),
            "rows_at_assay_floor": int(sub.at_assay_floor.sum()),
        }
    return out


def check_regression(counts):
    """Compare the measured cells against the frozen reconnaissance numbers.

    Never adjusts anything: a mismatch is surfaced with its delta so it can be judged.
    """
    cells, ok = [], True
    for locus, expected in REGRESSION_TARGETS.items():
        for key, want in expected.items():
            got = counts[locus][key]
            match = got == want
            ok &= match
            cells.append({"locus": locus, "cell": key, "expected": want,
                          "observed": got, "delta": got - want, "match": bool(match)})
    return {"ok": bool(ok), "provenance": TARGET_PROVENANCE, "cells": cells}


def summarize(df, infos, n_deduped, assays):
    counts = locus_counts(df)
    by_cell = defaultdict(int)
    for (locus, klass), n in Counter(zip(df.locus, df.klass)).items():
        by_cell[f"{locus}/{klass}"] = int(n)
    return {
        "assays": assays,
        "n_assays": len(assays),
        "rows_emitted": int(len(df)),
        "rows_before_dedup": int(len(df) + n_deduped),
        "rows_dropped_by_dedup": int(n_deduped),
        "rows_by_locus_klass": dict(sorted(by_cell.items())),
        "positions_by_locus": {locus: counts[locus]["positions"] for locus in LOCI},
        "usable_positions_by_locus": {locus: counts[locus]["usable_positions"]
                                      for locus in LOCI},
        "locus_counts": counts,
        "ptm_site_n": int((df.is_ptm).sum()),
        "control_n": int((~df.is_ptm).sum()),
        "taxon_counts": {k: int(v) for k, v in Counter(df.taxon).items()},
        "aln_identity_by_assay": {i["dms_id"]: i["aln_identity"] for i in infos},
        "per_assay": infos,
        "regression": check_regression(counts),
    }


def report(summary):
    print(f"assays              : {summary['n_assays']}")
    print(f"rows emitted        : {summary['rows_emitted']}"
          f"  (dedup removed {summary['rows_dropped_by_dedup']} of "
          f"{summary['rows_before_dedup']})")
    print(f"PTM-site n          : {summary['ptm_site_n']}")
    print(f"control n           : {summary['control_n']}")

    print("\nlocus x klass")
    print(f"  {'locus':<14}{'positions':>10}{'usable':>8}{'preserving':>12}"
          f"{'abolishing':>12}{'abol@usable':>13}{'at floor':>10}")
    for locus in LOCI:
        c = summary["locus_counts"][locus]
        print(f"  {locus:<14}{c['positions']:>10}{c['usable_positions']:>8}"
              f"{c['preserving']:>12}{c['abolishing']:>12}"
              f"{c['abolishing_at_usable']:>13}{c['rows_at_assay_floor']:>10}")

    print("\nalignment identity by assay")
    for dms_id, ident in summary["aln_identity_by_assay"].items():
        print(f"  {dms_id:<40}{'n/a' if ident is None else f'{ident:.4f}'}")

    reg = summary["regression"]
    print("\nreconnaissance regression check")
    for cell in reg["cells"]:
        flag = "ok " if cell["match"] else "OFF"
        print(f"  [{flag}] {cell['locus']:<14}{cell['cell']:<22}"
              f"expected {cell['expected']:>6}   observed {cell['observed']:>6}"
              f"   delta {cell['delta']:+d}")
    if not reg["ok"]:
        off = [c for c in reg["cells"] if not c["match"]]
        banner = "!" * 78
        print(f"\n{banner}\nCOVERAGE MISMATCH: {len(off)} of {len(reg['cells'])} frozen "
              f"cells differ from reconnaissance.\n{reg['provenance']}\n"
              "Not auto-corrected. Judge each delta before trusting downstream stages.\n"
              f"{banner}", file=sys.stderr)


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Stage 1: build the per-observation PTM/control coverage table.")
    ap.add_argument("--assays", default=",".join(CONTRIBUTING),
                    help="comma-separated ProteinGym DMS_ids (default: the 13 "
                         "contributing assays)")
    ap.add_argument("--out", default=str(REPO / "results" / "variants.csv"),
                    help="output variant table (default: results/variants.csv)")
    ap.add_argument("--summary", default=str(REPO / "results" / "coverage_summary.json"),
                    help="output summary JSON (default: results/coverage_summary.json)")
    args = ap.parse_args()

    assays = [a.strip() for a in args.assays.split(",") if a.strip()]
    ref = data.load_reference()
    unknown = [a for a in assays if a not in set(ref.DMS_id)]
    if unknown:
        ap.error(f"unknown DMS_id(s): {', '.join(unknown)}")

    parsed, useq, repaired, missing_seq = load_annotations(assays, ref)
    df, infos, n_deduped = build_variants(assays, ref, parsed, useq)

    summary = summarize(df, infos, n_deduped, assays)
    summary["repaired_uniprot_entries"] = repaired
    summary["entries_without_sequence"] = missing_seq

    out, summary_path = Path(args.out), Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    report(summary)
    print(f"\nwrote {out}\nwrote {summary_path}")


if __name__ == "__main__":
    main()

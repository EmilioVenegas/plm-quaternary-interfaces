#!/usr/bin/env python3
"""Stage 2: masked-marginal log-odds for every scored observation, one model arm at a time.

Reads the per-observation table written by stage 1 (`results/variants.csv`) and adds, for
each requested model arm, the two columns the confirmatory test consumes:

    zeroshot_<arm>   log P(mut | masked context) - log P(wt | masked context)
    logp_wt_<arm>    log P(wt  | masked context)

The second column is not a diagnostic; it is the matching variable. `PLAN.md` §3.2 holds
positional conservation fixed by caliper-matching PTM sites to non-PTM control positions
on the model's *own* masked wild-type log-probability, so `logp_wt` has to come out of the
same forward pass as the score it will later be used to explain.

One forward per (assay, position), not per mutant
------------------------------------------------
A masked position yields a distribution over the whole amino-acid alphabet in a single
pass, so all 19 substitutions at a position are read off one softmax. Re-masking per
mutant would multiply the compute by ~19 and -- because `log_softmax` is computed in fp32
over identical logits -- return bit-identical numbers. `scoring.masked_logprobs_batched`
additionally packs `--batch-size` independent single-mask rows into one forward, which is
what makes the CPU-offloaded ESMC-6B arm affordable (measured 1.77 s/position at batch 1
versus 0.23 s/position at batch 128: the cost is streaming weights across PCIe, and
independent masked rows amortise one stream).

Restartability
--------------
The four arms are not all reachable from one interpreter: the `esm` package that ESMC
needs is deliberately absent from the core environment (it pulls its own torch and would
disturb the pinned environment the reconnaissance numbers were produced in). So the run is
split across two interpreters and stitched together with `--resume`:

    .venv/bin/python       scripts/02_score_models.py --arms esm2-650m,esm2-3b
    .venv-esmc/bin/python  scripts/02_score_models.py --arms esmc-600m,esmc-6b --resume

`--resume` keeps arm columns already complete in the output file and re-scores nothing.
The output is rewritten after every arm, so an interrupted multi-arm run never loses a
finished arm.

Coordinate sanity check
-----------------------
Every arm is checked against TEM-1's disulfide before it scores anything: masking
`BLAT_ECOLX_Stiffler_2015` `target_seq` positions 75 and 121 must put cysteine at the
argmax. Those two residues are the real C75-C121 bond (PDB 1BTL annotates it as Ambler
77/123 -- a different coordinate system, which is exactly why the check is worth having),
and a model that does not name them is being fed a wrong offset or a tokenizer whose
+1 BOS convention differs. That failure is silent in the score table and fatal downstream,
so it is checked loudly here rather than inferred from a null result later.
"""

import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

# TEM-1 beta-lactamase, ProteinGym `target_seq` indices of the C75-C121 disulfide.
BLAT_ASSAY = "BLAT_ECOLX_Stiffler_2015"
BLAT_CYS = (75, 121)

# Identifies one observation. `PRE_REGISTRATION.md` §5 deduplicates on
# (protein, position, wt, mut); `dms_id` is carried along so a resume merge cannot
# silently pair rows across assays if a future stage 1 relaxes the dedup.
KEY = ["dms_id", "protein", "position", "wt", "mut"]

ESMC_PYTHON = ".venv-esmc/bin/python"


def loud(msg: str) -> None:
    """Warnings that must not scroll past unnoticed: a wrong coordinate system produces a
    perfectly well-formed score table, so the only defence is an unmissable banner."""
    bar = "!" * 78
    print(f"\n{bar}\n{msg}\n{bar}\n", file=sys.stderr, flush=True)


def arm_columns(arm: str) -> tuple[str, str]:
    return f"zeroshot_{arm}", f"logp_wt_{arm}"


def esmc_available() -> bool:
    """EvolutionaryScale's `esm` package, imported lazily by `models._load_esmc`."""
    return importlib.util.find_spec("esm") is not None


def esmc_hint(arms) -> str:
    listed = ",".join(arms)
    return (
        f"arm(s) {listed} need EvolutionaryScale's `esm` package, which is deliberately "
        f"absent from the core environment (it pulls its own torch and would disturb the "
        f"pinned environment every committed number was produced against).\n"
        f"Run them from the separate interpreter instead:\n"
        f"    {ESMC_PYTHON} scripts/02_score_models.py --arms {listed} --resume\n"
        f"(`--resume` keeps the arms already scored in the output file.) If that "
        f"interpreter does not exist yet: python -m venv .venv-esmc && "
        f"{ESMC_PYTHON} -m pip install -r requirements-esmc.txt"
    )


def target_sequences() -> dict[str, str]:
    """{dms_id: target_seq}. The ProteinGym reference is the only record of the construct
    sequence that every assay position indexes into; positions are never re-derived."""
    from plmconfound.data import load_reference

    ref = load_reference()
    return dict(zip(ref.DMS_id, ref.target_seq))


def sanity_check_disulfide(arm, model, tok, seqs, batch_size) -> dict:
    """Masked argmax at TEM-1 C75/C121 must be `C`. Warns loudly, never raises: the check
    validates the tokenizer/offset convention, and a report that says which arm failed is
    more useful than an aborted run that says nothing about the other arms."""
    from plmconfound import scoring

    seq = seqs.get(BLAT_ASSAY)
    if not isinstance(seq, str):
        loud(f"[{arm}] {BLAT_ASSAY} is absent from the ProteinGym reference, so the "
             f"C75/C121 coordinate sanity check could NOT be run for this arm.")
        return {"status": "skipped", "reason": f"{BLAT_ASSAY} not in reference"}

    lp = scoring.masked_logprobs_batched(model, tok, seq, list(BLAT_CYS),
                                         batch_size=batch_size)
    report = {"status": "ok", "assay": BLAT_ASSAY, "positions": {}}
    for p in BLAT_CYS:
        d = lp[int(p)]
        top = max(d, key=d.get)
        report["positions"][str(p)] = {
            "target_seq_residue": seq[p - 1],
            "argmax": top,
            "logp_argmax": round(d[top], 4),
            "logp_C": round(d["C"], 4),
            "argmax_is_C": top == "C",
        }
    bad = [p for p, v in report["positions"].items() if not v["argmax_is_C"]]
    shown = "  ".join(
        f"{p}: wt={v['target_seq_residue']} argmax={v['argmax']} "
        f"logP(C)={v['logp_C']:+.3f}" for p, v in report["positions"].items())
    print(f"[{arm}] disulfide sanity check ({BLAT_ASSAY}): {shown}", flush=True)
    if bad:
        report["status"] = "FAILED"
        loud(f"[{arm}] SANITY CHECK FAILED: masked argmax at {BLAT_ASSAY} target_seq "
             f"position(s) {', '.join(bad)} is not C, but those positions are TEM-1's "
             f"C75-C121 disulfide cysteines.\nThis means the coordinate mapping or the "
             f"tokenizer's BOS offset is wrong for this arm. Scores produced now are "
             f"NOT trustworthy and must not be used for the confirmatory test.")
    return report


def score_arm(arm, df, seqs, batch_size) -> tuple[pd.Series, pd.Series, dict]:
    """Score every row of `df` under one arm. -> (zeroshot, logp_wt, report)."""
    from plmconfound import models, scoring

    print(f"\n[{arm}] loading model", flush=True)
    t_load = time.time()
    model, tok = models.load_model(arm, mode="auto")
    print(f"[{arm}] loaded in {time.time() - t_load:.1f}s", flush=True)

    zs = pd.Series(float("nan"), index=df.index, dtype=float)
    lw = pd.Series(float("nan"), index=df.index, dtype=float)
    report = {"arm": arm, "n_rows": int(len(df)), "assays": {}}
    mismatches: list[str] = []
    n_positions = 0
    t0 = time.time()

    try:
        report["sanity_check"] = sanity_check_disulfide(arm, model, tok, seqs, batch_size)

        for dms_id, g in df.groupby("dms_id", sort=True):
            seq = seqs.get(dms_id)
            if not isinstance(seq, str):
                raise SystemExit(f"{dms_id} has no target_seq in the ProteinGym "
                                 f"reference; cannot score it.")
            positions = sorted(int(p) for p in g.position.unique())
            out_of_range = [p for p in positions if not 1 <= p <= len(seq)]
            if out_of_range:
                raise SystemExit(
                    f"{dms_id}: position(s) {out_of_range} fall outside target_seq "
                    f"(length {len(seq)}). Stage 1 must emit 1-indexed target_seq "
                    f"positions from plmconfound.mapping.build_map.")

            t_assay = time.time()
            lp = scoring.masked_logprobs_batched(model, tok, seq, positions,
                                                batch_size=batch_size)
            zs_vals, lw_vals = [], []
            for row in g.itertuples(index=False):
                pos, wt, mut = int(row.position), row.wt, row.mut
                d = lp[pos]
                if seq[pos - 1] != wt:
                    mismatches.append(f"{dms_id}:{pos} target_seq={seq[pos - 1]} wt={wt}")
                lw_vals.append(d[wt])
                zs_vals.append(d[mut] - d[wt])
            zs.loc[g.index] = zs_vals
            lw.loc[g.index] = lw_vals

            dt = time.time() - t_assay
            n_positions += len(positions)
            report["assays"][dms_id] = {"n_positions": len(positions),
                                        "n_rows": int(len(g)),
                                        "seconds": round(dt, 2)}
            print(f"[{arm}] {dms_id}: {len(positions)} positions, {len(g)} rows, "
                  f"{dt:.1f}s ({dt / max(len(positions), 1):.3f}s/position)", flush=True)
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:      # pragma: no cover - torch is a hard dep of this stage
            pass

    report["n_positions"] = n_positions
    report["seconds_total"] = round(time.time() - t0, 2)
    report["n_wt_mismatches"] = len(mismatches)
    if mismatches:
        report["wt_mismatch_examples"] = mismatches[:10]
        loud(f"[{arm}] {len(mismatches)} row(s) whose stated wild-type residue disagrees "
             f"with target_seq at the mapped position. Every position must come from "
             f"plmconfound.mapping.build_map; an assumed offset is prohibited.\n"
             f"First examples: {', '.join(mismatches[:10])}")
    print(f"[{arm}] done: {n_positions} positions, {len(df)} rows, "
          f"{report['seconds_total']:.1f}s", flush=True)
    return zs, lw, report


def load_variants(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/01_build_coverage.py first.")
    df = pd.read_csv(path)
    missing = [c for c in KEY + ["locus", "klass", "DMS_score"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {missing}")
    dup = df.duplicated(subset=KEY)
    if dup.any():
        raise SystemExit(
            f"{path} carries {int(dup.sum())} duplicate rows on {KEY}. "
            f"PRE_REGISTRATION.md §5 deduplicates on (protein, position, wt, mut); "
            f"pooling re-measurements of the same library would multiply-count variants.")
    return df


def merge_resume(variants: pd.DataFrame, out_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Carry complete arm columns over from an existing output file.
    -> (frame with those columns attached, list of arms already complete)."""
    prev = pd.read_csv(out_path)
    prev_arms = sorted({c.removeprefix("zeroshot_") for c in prev.columns
                        if c.startswith("zeroshot_")}
                       & {c.removeprefix("logp_wt_") for c in prev.columns
                          if c.startswith("logp_wt_")})
    if not prev_arms:
        return variants, []
    if any(c not in prev.columns for c in KEY):
        raise SystemExit(f"{out_path} lacks the key columns {KEY}; delete it or rerun "
                         f"without --resume.")
    if prev.duplicated(subset=KEY).any():
        raise SystemExit(f"{out_path} is not unique on {KEY}; delete it or rerun without "
                         f"--resume.")

    cols = [c for a in prev_arms for c in arm_columns(a)]
    merged = variants.merge(prev[KEY + cols], on=KEY, how="left")
    complete = []
    for arm in prev_arms:
        zc, lc = arm_columns(arm)
        if merged[zc].notna().all() and merged[lc].notna().all():
            complete.append(arm)
        else:
            n = int(merged[zc].isna().sum())
            print(f"[resume] {arm}: {n} of {len(merged)} rows unscored -> re-scoring",
                  flush=True)
            merged = merged.drop(columns=[zc, lc])
    return merged, complete


def write_out(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[write] {out_path}  ({len(df)} rows, {len(df.columns)} columns)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2: masked-marginal log-odds per model arm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="ESMC arms must be run from .venv-esmc/bin/python with --resume.")
    ap.add_argument("--arms", default="esm2-650m",
                    help="comma-separated model arms (default: esm2-650m)")
    ap.add_argument("--variants", type=Path, default=Path("results/variants.csv"),
                    help="stage 1 observation table (default: results/variants.csv)")
    ap.add_argument("--out", type=Path, default=Path("results/scores.csv"),
                    help="output score table (default: results/scores.csv)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="independent masked rows per forward pass (default: 32)")
    ap.add_argument("--resume", action="store_true",
                    help="keep arms already complete in --out and score only the rest")
    args = ap.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        raise SystemExit("--arms is empty")
    if len(set(arms)) != len(arms):
        raise SystemExit(f"--arms repeats an arm: {arms}")

    from plmconfound.models import MODELS

    unknown = [a for a in arms if a not in MODELS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(MODELS)}")

    df = load_variants(args.variants)
    print(f"[read] {args.variants}: {len(df)} observations, "
          f"{df.dms_id.nunique()} assays, "
          f"{df.groupby(['dms_id', 'position']).ngroups} positions", flush=True)

    done: list[str] = []
    if args.resume and args.out.exists():
        df, done = merge_resume(df, args.out)
        if done:
            print(f"[resume] already complete in {args.out}: {', '.join(done)}",
                  flush=True)
    todo = [a for a in arms if a not in done]

    blocked = [a for a in todo if MODELS[a]["family"] == "esmc"] if not esmc_available() \
        else []
    todo = [a for a in todo if a not in blocked]

    if not todo:
        if blocked:
            print(esmc_hint(blocked), file=sys.stderr)
            return 2
        print("nothing to do: every requested arm is already complete in "
              f"{args.out}", flush=True)
        return 0

    seqs = target_sequences()
    reports = {}
    for arm in todo:
        zc, lc = arm_columns(arm)
        zs, lw, report = score_arm(arm, df, seqs, args.batch_size)
        df[zc], df[lc] = zs, lw
        reports[arm] = report
        write_out(df, args.out)       # after every arm: an interrupted run keeps this one

    print("\n[summary]", flush=True)
    for arm, rep in reports.items():
        sanity = rep.get("sanity_check", {}).get("status", "?")
        print(f"  {arm}: {rep['n_positions']} positions, {rep['n_rows']} rows, "
              f"{rep['seconds_total']:.1f}s, sanity check {sanity}, "
              f"{rep['n_wt_mismatches']} wt mismatches", flush=True)
    if done:
        print(f"  carried over unchanged: {', '.join(done)}", flush=True)

    failed = [a for a, r in reports.items()
              if r.get("sanity_check", {}).get("status") == "FAILED"]
    if blocked:
        print(esmc_hint(blocked), file=sys.stderr)
    if failed:
        loud(f"arm(s) {', '.join(failed)} failed the C75/C121 coordinate sanity check; "
             f"do not run stage 3 on them until the offset is fixed.")
        return 3
    return 2 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())

"""
Reconnaissance pass: coverage feasibility of two candidate PLM-blind-spot studies.

  Track A -- PTM-site chemistry-vs-conservation confound (phosphosites, N-glycosylation sequons)
  Track B -- disulfide-bond paired epistasis (additive masked-marginal vs joint pseudo-likelihood)

This script is the executable record of the reconnaissance reported in
`research/recon_ptm_disulfide_results.md`. It is NOT the confirmatory analysis; it
produces coverage counts, mapping validations and scoring-mechanics validations only.

Everything below was executed against real data on 2026-08-28 (ProteinGym v1.3,
UniProtKB REST, RCSB PDB, Tsuboyama 2023 Zenodo 7992926, ESM2-650M/3B on an RTX 4060 8GB).

Pipeline
  1. ProteinGym v1.3 reference metadata + all 217 substitution assay CSVs.
  2. UniProtKB REST feature fetch (MOD_RES phospho / CARBOHYD / DISULFID) for every
     assay's UniProt entry name; 183/186 entry names resolve directly, 1 more via
     accession fallback, 2 are obsolete/deleted entries (A0A192B1T2_9HIV1 has no
     sequence record; ANCSZ is an ancestral synthetic construct).
  3. UniProt -> ProteinGym `target_seq` position mapping by real pairwise alignment
     (Biopython PairwiseAligner, blastp matrix, global). Never by assumed offset:
     TEM-1's PDB numbering is Ambler (SSBOND 77-123) while its target_seq index is
     75/121, and MET_HUMAN_Estevam_2023 is a kinase-domain-only construct offset by
     1058 residues.
  4. Per-position substitution counts from the real DMS single-mutant rows, split into
     modification-preserving vs modification-abolishing families.
  5. Track B: annotated-disulfide coverage, plus an annotation-independent superset
     bound (every double mutant in ProteinGym that hits two wild-type cysteines).
  6. Track B: MegaScale (776,298 rows) scan for wild-type-cysteine multi-mutants.
  7. ESM2 scoring mechanics: additive masked-marginal vs joint (double-mask) vs
     symmetrized conditional pseudo-likelihood, validated on a real disulfide with
     two negative controls.

Compatible ("modification-preserving") families are primary-source grounded, NOT
assumed by analogy:
  - Phospho-acceptor: {S,T} are interchangeable Ser/Thr-kinase acceptors; Tyr sites
    have NO preserving partner (Tyr kinases are a separate class), so the acceptor
    family at a Tyr phosphosite has exactly one member -- the same 1-member collapse
    that zeroed out CYP2C9/GAL4 in the metal project.
  - N-glycosylation sequon Asn (position +0): family = {N} only. No other residue is a
    glycosyl acceptor, so a modification-preserving non-WT substitution CANNOT exist.
  - N-glycosylation sequon position +2: family = {S,T,C}. Bause & Legler 1981,
    Biochem J 195:639-644 (PMID 7316978) measured Thr, Ser AND Cys hexapeptides as
    competent glycosyl acceptors ("the threonine-, serine- and cysteine-containing
    derivatives could be glycosylated, although at very different rates, whereas the
    valine and O-methylthreonine analogues did not work as glycosyl acceptors").
    Cys is therefore preserving-but-attenuated (2-3x lower acceptor activity), and the
    +2 locus is the ONLY locus in the whole glycosylation track where a preserving
    substitution exists at all.
"""

import csv
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from Bio.Align import PairwiseAligner
from transformers import AutoModelForMaskedLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
PG = REPO / "data" / "proteingym"
BULK = PG / "bulk" / "DMS_ProteinGym_substitutions"
REF_CSV = PG / "DMS_substitutions_ref.csv"
FEAT_CACHE = REPO / "data" / "ptm" / "uniprot_features.json"
MEGASCALE = REPO / "data" / "megascale" / "Processed_K50_dG_datasets" / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
OUT = REPO / "results" / "recon"

MODEL_650M = "facebook/esm2_t33_650M_UR50D"
AA = list("ACDEFGHIKLMNPQRSTVWY")
UNIPROT_FIELDS = "accession,id,protein_name,sequence,ft_disulfid,ft_mod_res,ft_carbohyd,ft_act_site"
SEQUON = re.compile(r"N[^P][ST]")          # Asn-X-Ser/Thr, X != Pro
PHOSPHO_ACCEPTOR = set("ST")               # Tyr handled separately: 1-member family
GLYC_PLUS2 = set("STC")                    # Bause & Legler 1981 (PMID 7316978)

# ---------------------------------------------------------------- data plumbing

def load_reference():
    return pd.read_csv(REF_CSV)


def load_assay(dms_id):
    return pd.read_csv(BULK / f"{dms_id}.csv", usecols=["mutant", "DMS_score"])


def split_muts(s):
    """'A1B:C2D' -> [('A',1,'B'), ('C',2,'D')]. ProteinGym multi-mutants are ':'-joined."""
    return [(p[0], int(p[1:-1]), p[-1]) for p in str(s).split(":")]


def fetch_uniprot_features(entry_names, cache=FEAT_CACHE):
    """UniProtKB REST lookup by entry name, with accession fallback. Cached on disk."""
    sess = requests.Session()
    feats = json.loads(cache.read_text()) if cache.exists() else {}

    def one(query):
        for attempt in range(3):
            try:
                r = sess.get("https://rest.uniprot.org/uniprotkb/search",
                             params={"query": query, "fields": UNIPROT_FIELDS,
                                     "format": "json", "size": 1}, timeout=60)
                r.raise_for_status()
                res = r.json().get("results", [])
                return res[0] if res else None
            except Exception:
                time.sleep(1)
        return None

    missing = [n for n in entry_names if not feats.get(n)]
    with ThreadPoolExecutor(max_workers=12) as ex:
        for n, v in zip(missing, ex.map(lambda n: one(f"id:{n}"), missing)):
            feats[n] = v or one(f"accession:{n.split('_')[0]}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(feats))
    return feats


def repair_sequenceless_entries(feats, ref, cache=FEAT_CACHE):
    """Some ProteinGym `UniProt_ID` entry names resolve to obsolete/demerged UniProt
    records that carry NO sequence and NO features. Left alone this fails SILENTLY: the
    coverage functions simply emit no rows, and a real candidate looks like a zero.

    That is exactly what happened to PTEN. ProteinGym lists `PTEN_HUMAN`, which
    `id:PTEN_HUMAN` resolves to accession O00633 -- a record with an empty sequence and
    zero features -- rather than to human PTEN P60484 and its 11 annotated phosphosites
    (including the characterised CK2/ROCK1 tail cluster S380/T382/T383/S385). PTEN's DMS
    assays cover residues 1-403, so the sites ARE measured; the annotation lookup was the
    only thing missing.

    Repair rule, deliberately strict: re-query by the entry name's gene prefix and accept
    the replacement ONLY if its sequence is an exact match to that assay's ProteinGym
    `target_seq`. An annotation source is never adopted on name resemblance alone -- a
    wrong-organism ortholog would silently shift every position.
    """
    sess = requests.Session()

    def query(q):
        try:
            r = sess.get("https://rest.uniprot.org/uniprotkb/search",
                         params={"query": q, "fields": UNIPROT_FIELDS, "format": "json", "size": 10},
                         timeout=60)
            return r.json().get("results", [])
        except Exception:
            return []

    # The repair is written back into the feature cache, so a rerun sees nothing left to
    # repair. Persist the record itself, otherwise provenance silently disappears on the
    # second run and the report's claim becomes unverifiable from the artifact.
    ledger = cache.parent / "uniprot_repairs.json"
    repaired = json.loads(ledger.read_text()) if ledger.exists() else {}
    sequenceless = [n for n, v in feats.items()
                    if v and "__error__" not in v and not (v.get("sequence") or {}).get("value")]
    found = {}
    for name in sequenceless:
        targets = ref[ref.UniProt_ID == name].target_seq.unique()
        gene = name.split("_")[0]
        for cand in query(f"gene:{gene} AND reviewed:true"):
            seq = (cand.get("sequence") or {}).get("value", "")
            if seq and any(seq == t for t in targets):     # exact identity or nothing
                feats[name] = cand
                found[name] = cand["primaryAccession"]
                break
    if found:
        cache.write_text(json.dumps(feats))
        repaired.update(found)
        ledger.write_text(json.dumps(repaired, indent=1))
    return repaired


def parse_features(entry):
    """UniProt feature list -> {phospho:[(pos,desc)], carbohyd:[...], disulfid:[(p1,p2,desc)]}."""
    out = {"phospho": [], "carbohyd": [], "disulfid": [], "other_modres": []}
    for f in entry.get("features") or []:
        loc = f["location"]
        s, e = loc["start"].get("value"), loc["end"].get("value")
        desc = f.get("description") or ""
        if f["type"] == "Modified residue":
            bucket = "phospho" if desc.lower().startswith("phospho") else "other_modres"
            out[bucket].append((s, desc))
        elif f["type"] == "Glycosylation":
            out["carbohyd"].append((s, desc))
        elif f["type"] == "Disulfide bond" and s and e and s != e:
            out["disulfid"].append((s, e, desc))
    return out


# ------------------------------------------------------- numbering: real alignment

def make_aligner():
    a = PairwiseAligner(scoring="blastp")
    a.mode = "global"
    a.open_gap_score = -11
    a.extend_gap_score = -1
    return a


ALIGNER = make_aligner()


def build_map(uniprot_seq, target_seq):
    """-> (dict uniprot_1idx -> target_1idx, identity over aligned pairs, n_aligned).

    Mandatory: PDB numbers, UniProt numbers and ProteinGym target_seq indices are three
    different coordinate systems. This is the only sanctioned way to relate them.
    """
    aln = ALIGNER.align(uniprot_seq, target_seq)[0]
    mp, same, tot = {}, 0, 0
    for (us, ue), (ts, _te) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(ue - us):
            u, t = us + k, ts + k
            mp[u + 1] = t + 1
            tot += 1
            same += uniprot_seq[u] == target_seq[t]
    return mp, (same / tot if tot else 0.0), tot


def assay_single_mutant_index(dms_id):
    """-> {position: [(wt, mut, DMS_score), ...]} over single mutants only."""
    idx = defaultdict(list)
    df = load_assay(dms_id)
    for s, sc in zip(df.mutant, df.DMS_score):
        parts = split_muts(s)
        if len(parts) == 1:
            wt, pos, mut = parts[0]
            idx[pos].append((wt, mut, sc))
    return idx


# --------------------------------------------------------------- Track B coverage

def disulfide_coverage(ref, feats, parsed, useq):
    """Per annotated disulfide: is it mapped, are both mapped residues really Cys,
    how many single mutants at each, and how many double mutants hit BOTH."""
    rows = []
    inv = pd.DataFrame([{"UniProt_ID": n, "n_ssbond": len(p["disulfid"])} for n, p in parsed.items()])
    merged = ref.merge(inv, on="UniProt_ID", how="left")
    for dms_id in merged[merged.n_ssbond > 0].DMS_id:
        r = ref[ref.DMS_id == dms_id].iloc[0]
        uid, tseq = r.UniProt_ID, r.target_seq
        u = useq.get(uid)
        bonds = parsed[uid]["disulfid"]
        if not u:
            rows.append({"DMS_id": dms_id, "status": "NO_UNIPROT_SEQ", "n_bonds_annot": len(bonds)})
            continue
        mp, ident, _ = build_map(u, tseq)
        singles, doubles = defaultdict(list), defaultdict(int)
        for s in load_assay(dms_id).mutant:
            parts = split_muts(s)
            if len(parts) == 1:
                singles[parts[0][1]].append(parts[0])
            else:
                doubles[frozenset(p[1] for p in parts)] += 1
        for (a, b, _desc) in bonds:
            ta, tb = mp.get(a), mp.get(b)
            ok = ta is not None and tb is not None and tseq[ta - 1] == "C" and tseq[tb - 1] == "C"
            rows.append({
                "DMS_id": dms_id, "uniprot": uid, "aln_identity": round(ident, 3),
                "u_pos": (a, b), "t_pos": (ta, tb), "cys_verified": ok,
                "n_single_a": len(singles.get(ta, [])) if ok else 0,
                "n_single_b": len(singles.get(tb, [])) if ok else 0,
                "n_double_both_cys": sum(1 for k in doubles if ok and {ta, tb} <= set(k)),
                "status": "OK" if ok else "MAP_FAIL",
            })
    return pd.DataFrame(rows)


def cys_pair_superset_bound(ref):
    """Annotation-independent upper bound for Track B: EVERY ProteinGym double mutant
    that simultaneously substitutes two wild-type cysteines, whether or not those
    cysteines form an annotated disulfide. If this is ~0, no disulfide annotation
    source can rescue the track."""
    stats, pairs_out = [], []
    for dms_id in ref[ref.DMS_number_multiple_mutants > 0].DMS_id:
        tseq = ref[ref.DMS_id == dms_id].iloc[0].target_seq
        n_multi = n_cyspair = 0
        pairs = defaultdict(int)
        for s in load_assay(dms_id).mutant:
            parts = split_muts(s)
            if len(parts) < 2:
                continue
            n_multi += 1
            cys_hit = [p for p in parts if p[0] == "C"]
            if len(cys_hit) >= 2:
                n_cyspair += 1
                pairs[tuple(sorted(p[1] for p in cys_hit))] += 1
        stats.append({"DMS_id": dms_id, "n_cys_in_target": tseq.count("C"),
                      "n_multi_rows": n_multi, "n_rows_hitting_2plus_cys": n_cyspair,
                      "n_distinct_cys_pairs": len(pairs)})
        pairs_out += [{"DMS_id": dms_id, "cys_pair": p, "n_rows": c} for p, c in pairs.items()]
    return pd.DataFrame(stats), pd.DataFrame(pairs_out)


def megascale_cys_scan(csv_path=MEGASCALE, chunksize=200_000):
    """Does the 776k-variant MegaScale set mutate native cysteines, singly or in pairs?"""
    pat_single = re.compile(r"^[A-Z]\d+[A-Z]$")
    pat_double = re.compile(r"^[A-Z]\d+[A-Z]:[A-Z]\d+[A-Z]$")
    out = dict(rows=0, singles=0, singles_wt_cys=0, subs_to_cys=0,
               doubles=0, doubles_1plus_wt_cys=0, doubles_both_wt_cys=0)
    for chunk in pd.read_csv(csv_path, usecols=["mut_type"], chunksize=chunksize, low_memory=False):
        out["rows"] += len(chunk)
        for mt in chunk.mut_type:
            s = str(mt)
            if pat_single.match(s):
                out["singles"] += 1
                out["singles_wt_cys"] += s[0] == "C"
                out["subs_to_cys"] += s[-1] == "C"
            elif pat_double.match(s):
                out["doubles"] += 1
                wts = [p[0] for p in s.split(":")]
                out["doubles_1plus_wt_cys"] += wts.count("C") >= 1
                out["doubles_both_wt_cys"] += wts.count("C") == 2
    return out


MAVEDB_TERMS = [
    "disulfide", "disulphide", "cysteine", "cystine", "lysozyme", "trypsin inhibitor", "BPTI",
    "ribonuclease", "RNase", "insulin", "scFv", "antibody", "nanobody", "VHH", "immunoglobulin",
    "EGF", "toxin", "defensin", "protease inhibitor", "thioredoxin", "secreted", "glycoprotein",
    "spike", "envelope", "hemagglutinin", "protein G", "GFP", "fluorescent", "kinase", "folding",
    "stability", "thermodynamic", "epistasis", "double mutant", "combinatorial",
]
MAVEDB_MULTI = re.compile(r"p\.\[([^\]]*;[^\]]*)\]")
MAVEDB_VAR = re.compile(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")


def mavedb_cyspair_scan(terms=MAVEDB_TERMS, workers=10):
    """Last escape hatch for Track B: does ANY dataset in MaveDB (the largest DMS
    repository, 2,805 score sets) carry a multi-mutant at two wild-type cysteines?

    MaveDB exposes no unfiltered list endpoint and its text search caps at 100 hits, so
    the candidate pool is built from a broad term sweep biased hard toward
    disulfide-bearing protein classes and toward the assay designs that produce multi-
    mutants at all. Detection is on the HGVS protein string itself (`p.[Cys47Arg;
    Cys69Arg]`), so the wild-type residue comes from the record, not from a translation
    or an alignment we could get wrong.
    """
    base = "https://api.mavedb.org/api/v1"
    pool = {}
    for t in terms:
        try:
            rr = requests.post(f"{base}/score-sets/search", json={"text": t}, timeout=120).json()
            for ss in rr.get("scoreSets", []):
                if any((tg.get("targetSequence") or {}).get("sequence") for tg in ss.get("targetGenes") or []):
                    pool[ss["urn"]] = ss
        except Exception:
            continue

    def scan(urn):
        # 502s are transient. A silently dropped score set would manufacture a fake null,
        # so failures are retried with backoff and then reported explicitly as FAILED.
        for attempt in range(6):
            try:
                r = requests.get(f"{base}/score-sets/{urn}/scores",
                                 params={"drop_na_columns": "true"}, timeout=300)
                if not r.ok:
                    time.sleep(2 * (attempt + 1))
                    continue
                n_multi = n_cys2 = 0
                pairs = defaultdict(int)
                for m in MAVEDB_MULTI.finditer(r.text):
                    n_multi += 1
                    cys = [int(p) for (ref_aa, p, _alt) in MAVEDB_VAR.findall(m.group(1)) if ref_aa == "Cys"]
                    if len(cys) >= 2:
                        n_cys2 += 1
                        pairs[tuple(sorted(cys))] += 1
                return dict(urn=urn, status="ok", n_multi=n_multi, n_cys2=n_cys2, pairs=dict(pairs))
            except Exception:
                time.sleep(2 * (attempt + 1))
        return dict(urn=urn, status="FAILED", n_multi=0, n_cys2=0, pairs={})

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(scan, sorted(pool)))
    df = pd.DataFrame([{
        "urn": x["urn"], "title": pool[x["urn"]].get("title", "")[:120], "status": x["status"],
        "n_multi_variant_rows": x["n_multi"], "n_rows_2plus_wt_cys": x["n_cys2"],
        "cys_pairs": json.dumps({str(k): v for k, v in x["pairs"].items()}),
    } for x in res])
    return df


# --------------------------------------------------------------- Track A coverage

def phospho_coverage(ref, parsed, useq, dms_ids):
    rows = []
    for dms_id in dms_ids:
        r = ref[ref.DMS_id == dms_id].iloc[0]
        uid, tseq = r.UniProt_ID, r.target_seq
        idx = assay_single_mutant_index(dms_id)
        mp, ident, _ = (build_map(useq[uid], tseq) if useq.get(uid) else ({}, 0.0, 0))
        for (pos, desc) in parsed[uid]["phospho"]:
            t = mp.get(pos)
            wt = tseq[t - 1] if t else None
            muts = [m for (w, m, _sc) in idx.get(t, []) if w == wt] if t else []
            # preserving partner exists only for Ser/Thr sites (Tyr kinases are a separate class)
            fam = PHOSPHO_ACCEPTOR if wt in PHOSPHO_ACCEPTOR else set()
            rows.append({
                "track": "A-phospho", "DMS_id": dms_id, "protein": dms_id.split("_")[0],
                "uniprot": uid, "aln_ident": round(ident, 3), "u_pos": pos, "t_pos": t,
                "wt": wt, "ptm": desc.split(";")[0], "mapped": t is not None,
                "wt_is_STY": (wt in "STY") if wt else False, "n_dms_subs": len(muts),
                "n_compatible": sum(1 for m in muts if m in fam and m != wt),
                "n_incompatible": sum(1 for m in muts if m not in fam or m == wt),
                "muts": "".join(sorted(muts)),
            })
    return pd.DataFrame(rows)


def glycosylation_coverage(ref, parsed, useq, dms_ids):
    """Motif-derived sequons (N-X-[ST], X!=P) so that assays whose UniProt entry was
    deleted are still covered; UniProt CARBOHYD annotation recorded where available."""
    rows = []
    for dms_id in dms_ids:
        r = ref[ref.DMS_id == dms_id].iloc[0]
        uid, tseq = r.UniProt_ID, r.target_seq
        idx = assay_single_mutant_index(dms_id)
        ann, ident = set(), float("nan")
        if useq.get(uid):
            mp, ident, _ = build_map(useq[uid], tseq)
            ann = {mp.get(p) for (p, d) in parsed[uid]["carbohyd"] if "N-linked" in d}
        for mo in SEQUON.finditer(tseq):
            n_pos, plus2 = mo.start() + 1, mo.start() + 3
            for role, pos, fam in (("sequon_N", n_pos, {"N"}), ("sequon_ST+2", plus2, GLYC_PLUS2)):
                wt = tseq[pos - 1]
                muts = [m for (w, m, _sc) in idx.get(pos, []) if w == wt]
                rows.append({
                    "track": "A-glyc", "DMS_id": dms_id, "protein": dms_id.split("_")[0],
                    "uniprot": uid, "aln_ident": ident, "role": role, "t_pos": pos, "wt": wt,
                    "uniprot_annotated": n_pos in ann, "n_dms_subs": len(muts),
                    # at sequon_N the family is {N}: a preserving non-WT substitution cannot exist
                    "n_compatible": sum(1 for m in muts if m in fam and m != wt),
                    "n_incompatible": sum(1 for m in muts if m not in fam or m == wt),
                    "muts": "".join(sorted(muts)),
                })
    return pd.DataFrame(rows)


def matched_control_coverage(ref, parsed, useq, dms_ids):
    """Within-assay matched control: non-PTM Ser/Thr positions that carry at least one
    preserving ({S,T,C}) and one abolishing substitution. Same residue identity, same
    assay, same available substitution chemistry -- only PTM status differs. This
    removes both the cross-protein confound and the family-size confound."""
    rows = []
    for dms_id in dms_ids:
        r = ref[ref.DMS_id == dms_id].iloc[0]
        uid, tseq = r.UniProt_ID, r.target_seq
        idx = assay_single_mutant_index(dms_id)
        excl = set()
        if useq.get(uid):
            mp, _, _ = build_map(useq[uid], tseq)
            excl |= {mp.get(p) for (p, _d) in parsed[uid]["phospho"]}
        for mo in SEQUON.finditer(tseq):
            excl |= {mo.start() + 1, mo.start() + 3}
        n_pos = n_c = n_i = 0
        for pos, subs in idx.items():
            wt = tseq[pos - 1] if 0 < pos <= len(tseq) else None
            if wt not in "ST" or pos in excl:
                continue
            muts = [m for (w, m, _sc) in subs if w == wt]
            c = sum(1 for m in muts if m in GLYC_PLUS2 and m != wt)
            i = len(muts) - c
            if c and i:
                n_pos += 1
                n_c += c
                n_i += i
        rows.append({"DMS_id": dms_id, "ctrl_positions": n_pos, "ctrl_comp": n_c, "ctrl_incomp": n_i})
    return pd.DataFrame(rows)


def censoring_diagnostic(ref, parsed, useq, dms_ids):
    """Is Track A's usable coverage floor-censored the way the GFP Cys pairs were?

    The GFP pair data existed but was worthless: 11 of 13 variants pinned at one repeated
    value. The same artifact would silently hollow out Track A's n, so every contributing
    assay x locus is measured against its own assay's floor and modal pile-up before the
    coverage counts are trusted.
    """
    rows = []
    for dms_id in dms_ids:
        r = ref[ref.DMS_id == dms_id].iloc[0]
        uid, tseq = r.UniProt_ID, r.target_seq
        idx = assay_single_mutant_index(dms_id)
        ph, gl = [], []
        if useq.get(uid):
            mp, _, _ = build_map(useq[uid], tseq)
            for (p, _d) in parsed[uid]["phospho"]:
                t = mp.get(p)
                if t and tseq[t - 1] in PHOSPHO_ACCEPTOR and any(
                        m in PHOSPHO_ACCEPTOR for (_w, m, _s) in idx.get(t, [])):
                    ph.append(t)
        for mo in SEQUON.finditer(tseq):
            t2 = mo.start() + 3
            if any(m in GLYC_PLUS2 for (_w, m, _s) in idx.get(t2, [])):
                gl.append(t2)

        scores = load_assay(dms_id).DMS_score.dropna()
        lo, hi = scores.min(), scores.max()
        vc = scores.round(3).value_counts()
        modal_val, modal_n = vc.index[0], vc.iloc[0]
        for tag, positions in (("phospho", ph), ("glyc+2", gl)):
            tgt = np.array([sc for p in positions for (_w, _m, sc) in idx.get(p, [])
                            if pd.notna(sc)])
            if not len(tgt):
                continue
            rows.append({
                "assay": f"{dms_id.split('_')[0]} [{tag}]", "DMS_id": dms_id, "locus": tag,
                "n": int(len(tgt)), "assay_min": round(float(lo), 3), "assay_max": round(float(hi), 3),
                "modal_value": round(float(modal_val), 3),
                "modal_share_assay": round(float(modal_n / len(scores)), 3),
                # "at floor" = within 2% of the assay's full score range of its minimum
                "frac_target_at_floor": round(float(np.mean(np.abs(tgt - lo) < 0.02 * (hi - lo))), 3),
                "frac_target_at_modal": round(float(np.mean(np.abs(tgt - modal_val) < 1e-3)), 3),
                "target_mean": round(float(tgt.mean()), 3),
                "assay_mean": round(float(scores.mean()), 3),
            })
    return pd.DataFrame(rows)


def dedup_cells(df, fam_of_wt):
    """Collapse pseudo-replication: the three P53_Giacomelli conditions and the three SRC
    assays re-measure the SAME library, so pooling them would triple-count variants.
    Key on (protein, position, wt, mut)."""
    comp, incomp, pos = set(), set(), set()
    for _, r in df.iterrows():
        if r["wt"] is None:
            continue
        fam = fam_of_wt(r["wt"])
        key = (r["protein"], r["t_pos"], r["wt"])
        for mt in r["muts"]:
            (comp if (mt in fam and mt != r["wt"]) else incomp).add(key + (mt,))
        if r["n_dms_subs"]:
            pos.add(key)
    usable_pos = {k for k in pos if any(c[:3] == k for c in comp)}
    return usable_pos, comp, {c for c in incomp if c[:3] in usable_pos}


# ------------------------------------------------- ESM2 scoring mechanics (Track B)

def load_esm(name=MODEL_650M, dtype=None, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"dtype": dtype} if dtype else {}
    model = AutoModelForMaskedLM.from_pretrained(name, **kw).to(device).eval()
    return tok, model


@torch.no_grad()
def logprobs_masked(tok, model, seq, mask_positions, subs=None, device="cuda"):
    """Mask `mask_positions` (1-indexed) JOINTLY in one forward pass, after optionally
    applying `subs` {pos: aa}. -> {pos: {aa: logprob}}.

    Masking k positions in a single pass yields each position's conditional given the
    rest of the sequence with all k positions hidden -- that is the joint/double-mask
    quantity, not k independent marginals.
    """
    s = list(seq)
    for p, a in (subs or {}).items():
        s[p - 1] = a
    enc = tok("".join(s), return_tensors="pt").to(device)
    for p in mask_positions:
        enc["input_ids"][0, p] = tok.mask_token_id      # +1 shift for <cls>
    lp = F.log_softmax(model(**enc).logits[0], dim=-1)
    return {p: {a: lp[p, tok.convert_tokens_to_ids(a)].item() for a in AA} for p in mask_positions}


def score_pair(tok, model, seq, i, j, mi, mj):
    """Three scoring schemes for the double mutant (i->mi, j->mj).

    additive     = sum_k [ log P(mut_k | x_\\k)  - log P(wt_k | x_\\k) ]           (ESM-1v protocol)
    joint        = sum_k [ log P(mut_k | x_\\{i,j}) - log P(wt_k | x_\\{i,j}) ]     (both masked)
    conditional  = 0.5 * [ path(i->j) + path(j->i) ], where
                   path(i->j) = [log P(mi|x_\\i) - log P(wi|x_\\i)]
                              + [log P(mj|x with i:=mi, \\j) - log P(wj|x with i:=mi, \\j)]

    epsilon = conditional - additive is the model's implied pairwise epistasis. The
    additive protocol is epsilon-blind BY CONSTRUCTION; the question is whether the
    model's own representation carries a non-zero epsilon at disulfides, and with the
    correct sign.
    """
    wi, wj = seq[i - 1], seq[j - 1]
    li = logprobs_masked(tok, model, seq, [i])[i]
    lj = logprobs_masked(tok, model, seq, [j])[j]
    additive = (li[mi] - li[wi]) + (lj[mj] - lj[wj])

    both = logprobs_masked(tok, model, seq, [i, j])
    joint = (both[i][mi] - both[i][wi]) + (both[j][mj] - both[j][wj])

    lj_gi = logprobs_masked(tok, model, seq, [j], subs={i: mi})[j]
    li_gj = logprobs_masked(tok, model, seq, [i], subs={j: mj})[i]
    path_ij = (li[mi] - li[wi]) + (lj_gi[mj] - lj_gi[wj])
    path_ji = (lj[mj] - lj[wj]) + (li_gj[mi] - li_gj[wi])
    cond = 0.5 * (path_ij + path_ji)

    return dict(additive=additive, joint_doublemask=joint, conditional_sym=cond,
                eps_cond_minus_add=cond - additive, eps_joint_minus_add=joint - additive)


def pdb_ssbond(pdb_id):
    """Real SSBOND records straight from RCSB -- the structural ground truth. Cheaper and
    more auditable than a full-PDB bulk mirror for a few hundred specific entries."""
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=60)
    r.raise_for_status()
    return [ln for ln in r.text.splitlines() if ln.startswith("SSBOND")]


# ------------------------------------------------------------------------- driver

PHOSPHO_ASSAYS = [
    "SRC_HUMAN_Ahler_2019", "SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM", "SRC_HUMAN_Nguyen_2022",
    "MK01_HUMAN_Brenan_2016", "MET_HUMAN_Estevam_2023", "RAF1_HUMAN_Zinkus-Boltz_2019",
    "P53_HUMAN_Giacomelli_2018_Null_Etoposide", "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
    "P53_HUMAN_Giacomelli_2018_WT_Nutlin", "P53_HUMAN_Kotler_2018",
    "PTEN_HUMAN_Matreyek_2021", "PTEN_HUMAN_Mighell_2018", "ADRB2_HUMAN_Jones_2020",
    "YAP1_HUMAN_Araya_2012", "PRKN_HUMAN_Clausen_2023", "BRCA1_HUMAN_Findlay_2018",
]
GLYC_ASSAYS = [
    "A0A140D2T1_ZIKV_Sourisseau_2019", "A0A192B1T2_9HIV1_Haddox_2018", "Q2N0S5_9HIV1_Haddox_2018",
    "ENV_HV1B9_DuenasDecamp_2016", "ENV_HV1BR_Haddox_2016", "A0A2Z5U3Z0_9INFA_Doud_2016",
    "A0A2Z5U3Z0_9INFA_Wu_2014", "A4D664_9INFA_Soh_2019", "C6KNH7_9INFA_Lee_2018",
    "SPIKE_SARS2_Starr_2020_binding", "SPIKE_SARS2_Starr_2020_expression", "ACE2_HUMAN_Chan_2020",
]
# Assays that actually contribute usable coverage (see report); the matched control group
# is drawn from these, one assay per protein so the control arm is not pseudo-replicated.
CONTRIBUTING = [
    "MK01_HUMAN_Brenan_2016", "MET_HUMAN_Estevam_2023", "ADRB2_HUMAN_Jones_2020",
    "PRKN_HUMAN_Clausen_2023", "SRC_HUMAN_Ahler_2019", "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
    # PTEN only contributes once its obsolete UniProt entry is repaired (see
    # repair_sequenceless_entries); Mighell activity assay has the deeper coverage of the two.
    "PTEN_HUMAN_Mighell_2018",
    "ENV_HV1BR_Haddox_2016", "A0A192B1T2_9HIV1_Haddox_2018", "Q2N0S5_9HIV1_Haddox_2018",
    "C6KNH7_9INFA_Lee_2018", "A0A2Z5U3Z0_9INFA_Doud_2016", "SPIKE_SARS2_Starr_2020_binding",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load_reference()
    feats = fetch_uniprot_features(sorted(ref.UniProt_ID.unique()))
    # obsolete entry names resolve to sequence-less records and fail silently -- repair
    # them (sequence-exact-match only) BEFORE any coverage is counted.
    repaired = repair_sequenceless_entries(feats, ref)
    parsed = {n: parse_features(v) for n, v in feats.items() if v and "__error__" not in v}
    useq = {n: (v.get("sequence") or {}).get("value") for n, v in feats.items()
            if v and "__error__" not in v}

    summary = {"n_assays": len(ref), "n_uniprot_entries": len(feats),
               "n_entries_with_sequence": sum(1 for s in useq.values() if s),
               "repaired_sequenceless_entries": repaired,
               "still_sequenceless": sorted(n for n, s in useq.items() if not s)}

    # ---- Track B
    ssb = disulfide_coverage(ref, feats, parsed, useq)
    ssb.to_csv(OUT / "trackB_disulfide_coverage.csv", index=False)
    scan, pairs = cys_pair_superset_bound(ref)
    scan.to_csv(OUT / "trackB_cyspair_multimutant_scan.csv", index=False)
    summary["trackB"] = {
        "annotated_bonds_examined": int(len(ssb)),
        "bonds_mapped_and_cys_verified": int((ssb.get("cys_verified") == True).sum()),
        "bonds_with_both_cys_single_coverage": int(((ssb.n_single_a > 0) & (ssb.n_single_b > 0)).sum()),
        "bonds_with_any_double_mutant": int((ssb.n_double_both_cys > 0).sum()),
        "total_multimutant_rows_scanned": int(scan.n_multi_rows.sum()),
        "rows_hitting_two_wt_cys": int(scan.n_rows_hitting_2plus_cys.sum()),
        "distinct_cys_pairs": int(len(pairs)),
    }
    if MEGASCALE.exists():
        summary["trackB"]["megascale"] = megascale_cys_scan()

    mv = mavedb_cyspair_scan()
    mv.to_csv(OUT / "trackB_mavedb_cyspair_scan.csv", index=False)
    summary["trackB"]["mavedb"] = {
        "score_sets_scanned": int(len(mv)),
        "all_resolved": bool((mv.status == "ok").all()),
        "multi_variant_rows_scanned": int(mv.n_multi_variant_rows.sum()),
        "rows_hitting_two_wt_cys": int(mv.n_rows_2plus_wt_cys.sum()),
        "score_sets_with_such_rows": int((mv.n_rows_2plus_wt_cys > 0).sum()),
        "hits": mv[mv.n_rows_2plus_wt_cys > 0][["urn", "title", "cys_pairs"]].to_dict("records"),
    }

    # ---- Track A
    pa = phospho_coverage(ref, parsed, useq, PHOSPHO_ASSAYS)
    ga = glycosylation_coverage(ref, parsed, useq, GLYC_ASSAYS)
    ct = matched_control_coverage(ref, parsed, useq, CONTRIBUTING)
    pa.to_csv(OUT / "trackA_phospho_coverage.csv", index=False)
    ga.to_csv(OUT / "trackA_glycosylation_coverage.csv", index=False)
    ct.to_csv(OUT / "trackA_matched_control_coverage.csv", index=False)

    ph_pos, ph_comp, ph_inc = dedup_cells(
        pa[pa.wt_is_STY], lambda wt: PHOSPHO_ACCEPTOR if wt in PHOSPHO_ACCEPTOR else set())
    gl_pos, gl_comp, gl_inc = dedup_cells(
        ga[ga.role == "sequon_ST+2"], lambda wt: GLYC_PLUS2)
    asn = ga[ga.role == "sequon_N"]
    summary["trackA"] = {
        "phospho_usable_positions": len(ph_pos),
        "phospho_compatible": len(ph_comp), "phospho_incompatible": len(ph_inc),
        "glyc_plus2_usable_positions": len(gl_pos),
        "glyc_plus2_compatible": len(gl_comp), "glyc_plus2_incompatible": len(gl_inc),
        "asn_acceptor_positions_with_dms": int((asn.n_dms_subs > 0).sum()),
        "asn_acceptor_preserving_substitutions_available": 0,   # family {N} has one member
        "ptm_site_n": len(ph_comp) + len(ph_inc) + len(gl_comp) + len(gl_inc),
        "ptm_limiting_cell": len(ph_comp) + len(gl_comp),
        "control_positions": int(ct.ctrl_positions.sum()),
        "control_n": int(ct.ctrl_comp.sum() + ct.ctrl_incomp.sum()),
    }
    summary["trackA"]["total_design_n"] = summary["trackA"]["ptm_site_n"] + summary["trackA"]["control_n"]

    # ---- censoring: the artifact that made GFP's Cys pairs worthless. Checked, not assumed.
    cen = censoring_diagnostic(ref, parsed, useq, CONTRIBUTING)
    cen.to_csv(OUT / "trackA_censoring_diagnostic.csv", index=False)
    CENSORED = "ENV_HV1BR_Haddox_2016"
    g_no, c_no, i_no = dedup_cells(
        ga[(ga.role == "sequon_ST+2") & (ga.DMS_id != CENSORED)], lambda wt: GLYC_PLUS2)
    ctrl_no = int(ct[ct.DMS_id != CENSORED][["ctrl_comp", "ctrl_incomp"]].to_numpy().sum())
    summary["trackA"]["censoring"] = {
        "strata_checked": int(len(cen)),
        "strata_with_zero_floor_fraction": int((cen.frac_target_at_floor == 0).sum()),
        "worst_stratum": cen.loc[cen.frac_target_at_floor.idxmax(), "assay"],
        "worst_floor_fraction": float(cen.frac_target_at_floor.max()),
        "max_modal_share_assay": float(cen.modal_share_assay.max()),
        # verdict must survive discarding the worst offender outright
        "worst_case_drop_censored_assay": {
            "dropped": CENSORED,
            "ptm_site_n": len(ph_comp) + len(ph_inc) + len(c_no) + len(i_no),
            "ptm_limiting_cell": len(ph_comp) + len(c_no),
            "control_n": ctrl_no,
            "total_design_n": len(ph_comp) + len(ph_inc) + len(c_no) + len(i_no) + ctrl_no,
        },
    }

    # ---- scoring mechanics, validated against real structural ground truth
    tok, model = load_esm(dtype=torch.float16)
    blat = ref[ref.DMS_id == "BLAT_ECOLX_Stiffler_2015"].iloc[0].target_seq
    gfp = ref[ref.DMS_id == "GFP_AEQVI_Sarkisyan_2016"].iloc[0].target_seq
    summary["mechanics"] = {
        "ssbond_1BTL": pdb_ssbond("1BTL"),                 # CYS 77 - CYS 123 (Ambler)
        "ssbond_1EMA": pdb_ssbond("1EMA"),                 # [] -- avGFP Cys are free thiols
        "TEM1_C75_C121_disulfide_CtoA": score_pair(tok, model, blat, 75, 121, "A", "A"),
        "TEM1_C75_C121_disulfide_CtoS": score_pair(tok, model, blat, 75, 121, "S", "S"),
        "GFP_C48_C70_free_thiols_CtoA": score_pair(tok, model, gfp, 48, 70, "A", "A"),
        "TEM1_76_122_distance_matched_control": score_pair(tok, model, blat, 76, 122, "A", "A"),
    }

    with open(OUT / "recon_ptm_disulfide_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

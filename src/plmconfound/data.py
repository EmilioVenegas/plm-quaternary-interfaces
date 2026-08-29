"""ProteinGym assay loading and UniProt annotation retrieval.

Every number in this study is a join between two independently maintained resources:
ProteinGym's DMS tables (which define the assayed construct and its own 1-indexed
`target_seq` numbering) and UniProtKB's feature annotations (which define where the PTMs
are, in UniProt numbering). This module owns the retrieval half of that join and nothing
else; the coordinate reconciliation lives in `plmconfound.mapping` and is not optional.

Path constants are resolved from the installed package location so that the same code
works from a checkout, an editable install, or a script in `bin/`. Resolution happens at
import; no file is opened and no request is issued until a function is called.
"""

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
PROTEINGYM_DIR = REPO / "data" / "proteingym"
PROTEINGYM_REF = PROTEINGYM_DIR / "DMS_substitutions_ref.csv"
BULK_DIR = PROTEINGYM_DIR / "bulk" / "DMS_ProteinGym_substitutions"
FEATURE_CACHE = REPO / "data" / "ptm" / "uniprot_features.json"
REPAIR_LEDGER = FEATURE_CACHE.parent / "uniprot_repairs.json"

UNIPROT_FIELDS = "accession,id,protein_name,sequence,ft_disulfid,ft_mod_res,ft_carbohyd,ft_act_site"

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def load_reference() -> pd.DataFrame:
    """The ProteinGym substitution reference: one row per assay, and the only place the
    construct sequence (`target_seq`) that all assay positions index into is recorded."""
    return pd.read_csv(PROTEINGYM_REF)


def load_assay(dms_id: str, columns=("mutant", "DMS_score")) -> pd.DataFrame:
    """Read one assay table. Restricted to the columns actually used, because several
    ProteinGym files carry tens of extra per-variant columns that dwarf the two needed."""
    return pd.read_csv(BULK_DIR / f"{dms_id}.csv", usecols=list(columns))


def parse_mutants(s: str) -> list[tuple[str, int, str]]:
    """'A1B:C2D' -> [('A',1,'B'), ('C',2,'D')]. ProteinGym multi-mutants are ':'-joined."""
    return [(p[0], int(p[1:-1]), p[-1]) for p in str(s).split(":")]


def single_mutant_index(dms_id: str) -> dict[int, list[tuple[str, str, float]]]:
    """-> {position: [(wt, mut, DMS_score), ...]} over single mutants only.

    Multi-mutants are dropped rather than attributed to a position: the PTM-family
    contrast is a claim about substituting one residue, and a double mutant's score
    cannot be assigned to either site without an epistasis model.
    """
    idx = defaultdict(list)
    df = load_assay(dms_id)
    for s, sc in zip(df.mutant, df.DMS_score):
        parts = parse_mutants(s)
        if len(parts) == 1:
            wt, pos, mut = parts[0]
            idx[pos].append((wt, mut, sc))
    return idx


def fetch_uniprot_features(entry_names, cache=None) -> dict:
    """UniProtKB REST lookup by entry name, with accession fallback. Cached on disk.

    Threaded because the join needs a few hundred entries and the REST endpoint's latency,
    not its throughput, is the bottleneck. Retried three times because a transient 5xx
    would otherwise be indistinguishable from a genuinely unannotated protein -- and an
    absent annotation silently removes a candidate assay from the study.
    """
    cache = FEATURE_CACHE if cache is None else cache
    sess = requests.Session()
    feats = json.loads(cache.read_text()) if cache.exists() else {}

    def one(query):
        for _ in range(3):
            try:
                r = sess.get(UNIPROT_SEARCH,
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


def repair_sequenceless_entries(feats: dict, ref, cache=None) -> dict[str, str]:
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
    cache = FEATURE_CACHE if cache is None else cache
    sess = requests.Session()

    def query(q):
        try:
            r = sess.get(UNIPROT_SEARCH,
                         params={"query": q, "fields": UNIPROT_FIELDS, "format": "json", "size": 10},
                         timeout=60)
            return r.json().get("results", [])
        except Exception:
            return []

    # The repair is written back into the feature cache, so a rerun sees nothing left to
    # repair. Persist the record itself, otherwise provenance silently disappears on the
    # second run and the report's claim becomes unverifiable from the artifact.
    ledger = cache.parent / REPAIR_LEDGER.name
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


def parse_features(entry: dict) -> dict:
    """UniProt feature list -> {phospho:[(pos,desc)], carbohyd:[...], disulfid:[(p1,p2,desc)]}.

    'Modified residue' covers everything from phosphorylation to acetylation to
    hydroxylation, so it is split on the description prefix: only the phospho subset shares
    the Ser/Thr acceptor chemistry the family contrast is built on. Interchain disulfides
    are annotated with start == end and are dropped -- their partner is on another chain,
    so no intra-construct Cys pair exists to mutate.
    """
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

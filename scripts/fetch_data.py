#!/usr/bin/env python3
"""
Re-acquire every bulk dataset this project depends on.

The heavy inputs are deliberately NOT committed (see `.gitignore`: the ProteinGym bulk
assay CSVs, the ProteinGym/MegaScale zips, and the RCSB PDB entries), because together
they are ~2 GB and all of them are byte-reproducible from public, versioned endpoints.
Reproducibility therefore lives in this script rather than in git history: a clean clone
plus `python scripts/fetch_data.py` reconstructs exactly the tree the analysis was run
against on 2026-08-28.

Every URL and every byte count below was verified in-session during reconnaissance
(`docs/recon/recon_ptm_disulfide_results.md`, section 1). The sizes are asserted, not
logged: a silently truncated 1 GB download that still parses as CSV is the single most
expensive failure mode here, because it produces plausible-looking but wrong counts far
downstream. On mismatch this script deletes the partial file and fails loudly.

Idempotent by construction. Each target is checked *before* any network call: a file
that already exists at its verified size (or, for PDB entries, passes a structural
sanity check) is skipped. Re-running costs a few stat() calls, not 2 GB of traffic.

Bulk-download decision, recorded here because it is a scientific choice and not a
convenience one: the **full PDB mirror (~1.6 TB) is deliberately not used.** Only a
handful of specific entries are ever needed as structural ground truth, and per-entry
fetch from `files.rcsb.org` is exact, auditable and instant for that many. Mirroring the
PDB would add a terabyte of storage and a synchronisation problem in exchange for
nothing this study can use.

Dependencies: standard library + `requests`. No pandas — this script must be runnable in
a bare environment before the analysis env exists.

Usage:
    python scripts/fetch_data.py                      # fetch everything missing
    python scripts/fetch_data.py --only structures    # one dataset group
    python scripts/fetch_data.py --dry-run            # plan only, no network, no writes
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
PG_DIR = REPO / "data" / "proteingym"
PG_BULK_DIR = PG_DIR / "bulk"
MEGA_DIR = REPO / "data" / "megascale"
STRUCT_DIR = REPO / "data" / "structures"

TIMEOUT = 120  # seconds; generous because Zenodo throttles the 1 GB record
CHUNK = 1 << 20  # 1 MiB streaming chunk: bounded memory on the 1 GB archive

# --- ProteinGym v1.3 -------------------------------------------------------------
# The reference table is the entry point for everything: it carries target_seq (the
# third coordinate system, alongside UniProt and PDB numbering), pdb_file and pdb_range.
PG_REF_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
    "reference_files/DMS_substitutions.csv"
)
PG_REF_PATH = PG_DIR / "DMS_substitutions_ref.csv"
PG_REF_BYTES = 208_734  # measured 2026-08-28; ~204 KiB
# The reference file tracks `main`, so upstream metadata edits can move its byte count
# without invalidating it. Row count is the load-bearing invariant, not size.
PG_REF_TOLERANCE = 0.20
PG_N_ASSAYS = 217

PG_BULK_URL = (
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
    "DMS_ProteinGym_substitutions.zip"
)
PG_BULK_ZIP = PG_DIR / "DMS_ProteinGym_substitutions.zip"
PG_BULK_BYTES = 43_021_128  # verified HTTP 200, exact
PG_BULK_MEMBER_DIR = "DMS_ProteinGym_substitutions"  # top-level dir inside the zip
PG_BULK_UNPACKED = PG_BULK_DIR / PG_BULK_MEMBER_DIR  # 217 assay CSVs, ~1.0 GB

# --- Tsuboyama 2023 MegaScale ----------------------------------------------------
# Zenodo record 7992926. The archive also holds Dataset1 (1.28 GB) and two heat-map
# zips (~357 MB) that this project never opens; extracting them would triple the
# on-disk footprint for no analysis. Only the variant table and the three DMS
# manifests are unpacked.
MEGA_URL = (
    "https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip?download=1"
)
MEGA_ZIP = MEGA_DIR / "Processed_K50_dG_datasets.zip"
MEGA_BYTES = 1_013_678_473  # exact
MEGA_MEMBERS = {
    # 776,298-row variant table: the multi-mutant source scanned for Cys-Cys pairs.
    "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv": 697_658_024,
    "Processed_K50_dG_datasets/Single_DMS_list.csv": 425_270,
    "Processed_K50_dG_datasets/Double_DMS_list.csv": 85_739,
    "Processed_K50_dG_datasets/Triple_DMS_list.csv": 1_563,
}

# --- RCSB PDB --------------------------------------------------------------------
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
# Structural ground truth. PDB residue numbers, UniProt positions and ProteinGym
# target_seq indices are three different coordinate systems that agree by luck about
# half the time, so every one of these is consumed through pairwise alignment, never
# through an assumed offset.
PDB_IDS = {
    "1BTL": "TEM-1 beta-lactamase; real SSBOND CYS A 77 - CYS A 123 (Ambler numbering, "
            "= target_seq 75/121): the offset trap, caught from primary data",
    "1EMA": "avGFP; zero SSBOND records - the negative control for disulfide scoring",
    "3CLN": "calmodulin, Ca2+ sites (prior metal study)",
    "1OG5": "CYP2C9, heme-Fe (prior metal study)",
    "5K48": "VIM-2 metallo-beta-lactamase, Zn2+ (prior metal study)",
    "8T2J": "PPM1D, Mg2+ (prior metal study)",
    "1D66": "GAL4 Zn2Cys6; deposited with Cd2+ as crystallographic surrogate",
    "5BON": "NUDT15, Mg2+ (prior metal study)",
}
# A PDB entry for a real structure is never this small; an HTML error page or an RCSB
# "not found" stub is. Cheap guard against saving a 404 body as a .pdb file.
PDB_MIN_BYTES = 20_000

GROUPS = ("proteingym", "megascale", "structures")


class FetchError(RuntimeError):
    """A download completed but did not match its verified size or content."""


@dataclass
class Check:
    """One row of the final PASS/FAIL table: a single file that must exist on disk."""

    dataset: str
    target: Path
    expected: str
    actual: str
    ok: bool
    note: str = ""

    @property
    def rel(self) -> str:
        try:
            return str(self.target.relative_to(REPO))
        except ValueError:
            return str(self.target)


def human(n: int) -> str:
    """Byte counts in this project span 1.5 KB manifests to 1 GB archives."""
    if n < 1024:
        return f"{n:,} B"
    scaled = float(n)
    for unit in ("KB", "MB", "GB"):
        scaled /= 1024
        if scaled < 1024 or unit == "GB":
            return f"{scaled:,.1f} {unit}"
    return f"{n:,} B"


def size_ok(actual: int, expected: int, tolerance: float = 0.0) -> bool:
    return abs(actual - expected) <= expected * tolerance


def present(path: Path, expected: int, tolerance: float = 0.0) -> bool:
    return path.exists() and size_ok(path.stat().st_size, expected, tolerance)


def download(
    url: str,
    dest: Path,
    expected: int | None = None,
    tolerance: float = 0.0,
    minimum: int | None = None,
) -> int:
    """Stream `url` to `dest`, refusing to leave a wrong-sized file behind.

    Writes to a sibling `.part` and only renames after the size check passes. A
    truncated archive that still unzips partially, or a proxy error page saved under a
    data filename, would otherwise poison every downstream count silently.

    `expected` is used where the byte count is verified upstream (the two archives, the
    reference table); `minimum` is the weaker check for RCSB entries, whose per-entry
    sizes are not published and are validated structurally by `valid_pdb` instead.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with requests.get(url, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        with part.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                if chunk:
                    written += fh.write(chunk)

    problem = ""
    if expected is not None and not size_ok(written, expected, tolerance):
        band = f" (+/-{tolerance:.0%})" if tolerance else ""
        problem = f"expected {expected:,} B{band}, got {written:,} B"
    elif minimum is not None and written < minimum:
        problem = f"expected at least {minimum:,} B, got {written:,} B"
    if problem:
        part.unlink(missing_ok=True)
        raise FetchError(
            f"{url}\n  {problem}"
            "  -- partial file removed, nothing written to the data tree"
        )
    part.replace(dest)
    return written


def count_assay_rows(path: Path) -> int:
    """Row count is the reference table's real invariant: 217 substitution assays."""
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def valid_pdb(path: Path, pdb_id: str) -> tuple[bool, str]:
    """Reject anything that is not the requested coordinate file.

    RCSB answers unknown ids with HTML, and an HTML body saved as `1EMA.pdb` parses as
    "a file that exists" to every naive existence check downstream.
    """
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size < PDB_MIN_BYTES:
        return False, f"{size:,} B below plausibility floor"
    text = path.read_text(encoding="utf-8", errors="replace")
    header = text.split("\n", 1)[0]
    if not header.startswith("HEADER") or pdb_id not in header:
        return False, "no HEADER record for this entry"
    if "\nATOM  " not in text:
        return False, "no ATOM records"
    return True, ""


# --------------------------------------------------------------------------------
# dataset handlers: each returns its PASS/FAIL rows and performs no work in dry-run
# --------------------------------------------------------------------------------


def do_proteingym(dry_run: bool) -> list[Check]:
    checks: list[Check] = []

    # (a) reference metadata
    if present(PG_REF_PATH, PG_REF_BYTES, PG_REF_TOLERANCE):
        actual = PG_REF_PATH.stat().st_size
        checks.append(
            Check("proteingym/ref", PG_REF_PATH, human(PG_REF_BYTES), human(actual),
                  True, "already present")
        )
    elif dry_run:
        checks.append(
            Check("proteingym/ref", PG_REF_PATH, human(PG_REF_BYTES), "-", True,
                  f"WOULD FETCH {PG_REF_URL}")
        )
    else:
        got = download(PG_REF_URL, PG_REF_PATH, PG_REF_BYTES, PG_REF_TOLERANCE)
        n = count_assay_rows(PG_REF_PATH)
        ok = n == PG_N_ASSAYS
        checks.append(
            Check("proteingym/ref", PG_REF_PATH, f"{PG_N_ASSAYS} assays", f"{n} assays",
                  ok, f"downloaded {human(got)}" if ok else "unexpected assay count")
        )

    # (b) bulk substitution assays
    n_csv = len(list(PG_BULK_UNPACKED.glob("*.csv"))) if PG_BULK_UNPACKED.is_dir() else 0
    if n_csv == PG_N_ASSAYS:
        checks.append(
            Check("proteingym/bulk", PG_BULK_UNPACKED, f"{PG_N_ASSAYS} CSVs",
                  f"{n_csv} CSVs", True, "already extracted")
        )
        return checks
    if dry_run:
        checks.append(
            Check("proteingym/bulk", PG_BULK_UNPACKED, f"{PG_N_ASSAYS} CSVs",
                  f"{n_csv} CSVs", True,
                  f"WOULD FETCH {PG_BULK_URL} ({human(PG_BULK_BYTES)} zip -> ~1.0 GB)")
        )
        return checks

    if not present(PG_BULK_ZIP, PG_BULK_BYTES):
        download(PG_BULK_URL, PG_BULK_ZIP, PG_BULK_BYTES)
    PG_BULK_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PG_BULK_ZIP) as zf:
        zf.extractall(PG_BULK_DIR)
    n_csv = len(list(PG_BULK_UNPACKED.glob("*.csv")))
    checks.append(
        Check("proteingym/bulk", PG_BULK_UNPACKED, f"{PG_N_ASSAYS} CSVs",
              f"{n_csv} CSVs", n_csv == PG_N_ASSAYS,
              f"extracted from {human(PG_BULK_BYTES)} zip")
    )
    return checks


def do_megascale(dry_run: bool) -> list[Check]:
    wanted = {name: MEGA_DIR / name for name in MEGA_MEMBERS}
    missing = [n for n, p in wanted.items() if not present(p, MEGA_MEMBERS[n])]

    if not missing:
        return [
            Check("megascale", p, human(MEGA_MEMBERS[n]), human(p.stat().st_size),
                  True, "already extracted")
            for n, p in wanted.items()
        ]
    if dry_run:
        rows = [
            Check("megascale", MEGA_ZIP, human(MEGA_BYTES),
                  human(MEGA_ZIP.stat().st_size) if MEGA_ZIP.exists() else "-",
                  True, f"WOULD FETCH {MEGA_URL}" if not present(MEGA_ZIP, MEGA_BYTES)
                  else "archive present, would extract")
        ]
        rows += [
            Check("megascale", wanted[n], human(MEGA_MEMBERS[n]), "-", True,
                  "WOULD EXTRACT")
            for n in missing
        ]
        return rows

    if not present(MEGA_ZIP, MEGA_BYTES):
        download(MEGA_URL, MEGA_ZIP, MEGA_BYTES)
    MEGA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(MEGA_ZIP) as zf:
        for name in missing:
            dest = wanted[name]
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Member-by-member so Dataset1 and the heat-map zips stay in the archive.
            with zf.open(name) as src, dest.open("wb") as fh:
                shutil.copyfileobj(src, fh, CHUNK)

    return [
        Check("megascale", p, human(MEGA_MEMBERS[n]),
              human(p.stat().st_size) if p.exists() else "missing",
              present(p, MEGA_MEMBERS[n]),
              "extracted" if n in missing else "already extracted")
        for n, p in wanted.items()
    ]


def do_structures(dry_run: bool) -> list[Check]:
    checks: list[Check] = []
    for pdb_id, why in PDB_IDS.items():
        dest = STRUCT_DIR / f"{pdb_id}.pdb"
        ok, reason = valid_pdb(dest, pdb_id)
        if ok:
            checks.append(
                Check("structures", dest, "valid PDB",
                      human(dest.stat().st_size), True, "already present")
            )
            continue
        if dry_run:
            checks.append(
                Check("structures", dest, "valid PDB", reason, True,
                      f"WOULD FETCH {PDB_URL.format(pdb_id=pdb_id)}  [{why}]")
            )
            continue
        # RCSB publishes no per-entry size, so only a plausibility floor is asserted at
        # download time; valid_pdb() below does the real verification.
        download(PDB_URL.format(pdb_id=pdb_id), dest, minimum=PDB_MIN_BYTES)
        ok, reason = valid_pdb(dest, pdb_id)
        if not ok:
            dest.unlink(missing_ok=True)
        checks.append(
            Check("structures", dest, "valid PDB",
                  human(dest.stat().st_size) if ok else reason, ok,
                  "downloaded" if ok else "removed: not a coordinate file")
        )
    return checks


HANDLERS = {
    "proteingym": do_proteingym,
    "megascale": do_megascale,
    "structures": do_structures,
}


def report(checks: list[Check], dry_run: bool) -> int:
    width_t = max((len(c.rel) for c in checks), default=6)
    width_d = max((len(c.dataset) for c in checks), default=7)
    print()
    print(f"{'STATUS':<6}  {'DATASET':<{width_d}}  {'TARGET':<{width_t}}  "
          f"{'EXPECTED':>14}  {'ACTUAL':>14}  NOTE")
    for c in checks:
        status = "PLAN" if dry_run else ("PASS" if c.ok else "FAIL")
        print(f"{status:<6}  {c.dataset:<{width_d}}  {c.rel:<{width_t}}  "
              f"{c.expected:>14}  {c.actual:>14}  {c.note}")
    failed = [c for c in checks if not c.ok]
    print()
    if dry_run:
        print(f"dry run: {len(checks)} target(s) inspected, nothing downloaded.")
        return 0
    if failed:
        print(f"FAILED: {len(failed)} of {len(checks)} target(s) did not verify.")
        return 1
    print(f"OK: {len(checks)} of {len(checks)} target(s) verified.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-acquire the gitignored bulk datasets (idempotent).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--only", choices=(*GROUPS, "all"), default="all",
                    help="restrict to one dataset group (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be fetched, with sizes, and exit")
    args = ap.parse_args(argv)

    groups = GROUPS if args.only == "all" else (args.only,)
    checks: list[Check] = []
    for group in groups:
        try:
            checks.extend(HANDLERS[group](args.dry_run))
        except (FetchError, requests.RequestException, zipfile.BadZipFile) as exc:
            print(f"\n{group}: {exc}", file=sys.stderr)
            checks.append(Check(group, REPO, "-", "-", False, f"aborted: {exc}"))
    return report(checks, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

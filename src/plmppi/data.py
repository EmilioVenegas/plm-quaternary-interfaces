"""Dataset loading and paired assay curation for the PLM PPI study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from plmppi.interfaces import get_system_compartments

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
PROTEINGYM_REF_PATH = DEFAULT_DATA_DIR / "proteingym" / "DMS_substitutions_ref.csv"
BULK_DIR_PATH = DEFAULT_DATA_DIR / "proteingym" / "bulk" / "DMS_ProteinGym_substitutions"
STRUCTURES_DIR_PATH = DEFAULT_DATA_DIR / "structures"


@dataclass(frozen=True)
class PPISystem:
    system_id: str
    target_name: str
    pdb_id: str
    target_chain: str
    partner_chains: tuple[str, ...]
    dms_abundance: str
    dms_binding: str
    description: str


PRIMARY_SYSTEMS: tuple[PPISystem, ...] = (
    PPISystem(
        system_id="SARS-CoV-2_RBD",
        target_name="Spike RBD",
        pdb_id="6M0J",
        target_chain="E",
        partner_chains=("A",),
        dms_abundance="SPIKE_SARS2_Starr_2020_expression",
        dms_binding="SPIKE_SARS2_Starr_2020_binding",
        description="SARS-CoV-2 Spike receptor-binding domain interacting with ACE2 receptor",
    ),
    PPISystem(
        system_id="KRAS",
        target_name="KRAS G-domain",
        pdb_id="6H46",
        target_chain="A",
        partner_chains=("B",),
        dms_abundance="RASK_HUMAN_Weng_2022_abundance",
        dms_binding="RASK_HUMAN_Weng_2022_binding-DARPin_K55",
        description="KRAS small GTPase interacting with engineered DARPin K55",
    ),
    PPISystem(
        system_id="HLA-A2",
        target_name="HLA-A*02:01",
        pdb_id="5OPI",
        target_chain="A",
        partner_chains=("B", "C"),
        dms_abundance="Q53Z42_HUMAN_McShan_2019_expression",
        dms_binding="Q53Z42_HUMAN_McShan_2019_binding-TAPBPR",
        description="MHC class I heavy chain interacting with beta-2M and TAPBPR chaperone",
    ),
    PPISystem(
        system_id="GB1",
        target_name="Protein G B1 domain",
        pdb_id="1FCC",
        target_chain="C",
        partner_chains=("A", "B"),
        dms_abundance="SPG1_STRSG_Wu_2016",
        dms_binding="SPG1_STRSG_Olson_2014",
        description="Streptococcal protein G domain B1 interacting with IgG Fc",
    ),
    PPISystem(
        system_id="p53",
        target_name="p53 tetramerization domain",
        pdb_id="1OLG",
        target_chain="A",
        partner_chains=("B", "C", "D"),
        dms_abundance="P53_HUMAN_Giacomelli_2018_Null_Nutlin",
        dms_binding="P53_HUMAN_Giacomelli_2018_WT_Nutlin",
        description="p53 tumor suppressor homotetramer oligomerization domain",
    ),
)


def load_reference(ref_path: Path = PROTEINGYM_REF_PATH) -> pd.DataFrame:
    """Loads ProteinGym substitution reference metadata."""
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference table not found: {ref_path}")
    return pd.read_csv(ref_path)


def parse_single_mutants(df: pd.DataFrame) -> pd.DataFrame:
    """Filters multi-mutants and extracts (position, wt, mut, dms_score)."""
    rows = []
    for _, r in df.iterrows():
        m = str(r["mutant"]).strip()
        if ":" in m:
            continue
        if len(m) < 3:
            continue
        try:
            wt = m[0]
            pos = int(m[1:-1])
            mut = m[-1]
            score = float(r["DMS_score"])
            rows.append({"position": pos, "wt": wt, "mut": mut, "dms_score": score})
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(rows)


def build_system_dataset(
    sys: PPISystem,
    ref_df: pd.DataFrame,
    bulk_dir: Path = BULK_DIR_PATH,
    struct_dir: Path = STRUCTURES_DIR_PATH,
) -> pd.DataFrame:
    """Builds paired observations with structural compartments for one system."""
    matching_ref = ref_df.query("DMS_id == @sys.dms_abundance")
    if matching_ref.empty:
        raise ValueError(f"Assay {sys.dms_abundance} not found in reference table")
    target_seq = matching_ref.iloc[0]["target_seq"]

    pdb_path = struct_dir / f"{sys.pdb_id}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    comp_map = get_system_compartments(
        pdb_path=pdb_path,
        target_chain_id=sys.target_chain,
        partner_chain_ids=list(sys.partner_chains),
        target_seq=target_seq,
    )

    ab_csv = bulk_dir / f"{sys.dms_abundance}.csv"
    bi_csv = bulk_dir / f"{sys.dms_binding}.csv"

    if not ab_csv.exists() or not bi_csv.exists():
        raise FileNotFoundError(f"Assay CSVs missing for system {sys.system_id}")

    df_ab = parse_single_mutants(pd.read_csv(ab_csv, usecols=["mutant", "DMS_score"]))
    df_bi = parse_single_mutants(pd.read_csv(bi_csv, usecols=["mutant", "DMS_score"]))

    # Inner merge on exact (position, wt, mut)
    merged = pd.merge(
        df_ab,
        df_bi,
        on=["position", "wt", "mut"],
        suffixes=("_abundance", "_binding"),
    )

    # Attach structural compartment metadata
    merged["system"] = sys.system_id
    merged["pdb_id"] = sys.pdb_id
    merged["dms_abundance"] = sys.dms_abundance
    merged["dms_binding"] = sys.dms_binding
    merged["compartment"] = merged["position"].map(
        lambda p: comp_map[p]["compartment"] if p in comp_map else None
    )
    merged["dsasa"] = merged["position"].map(
        lambda p: comp_map[p]["dsasa"] if p in comp_map else None
    )
    merged["rsasa"] = merged["position"].map(
        lambda p: comp_map[p]["rsasa"] if p in comp_map else None
    )
    merged["min_dist"] = merged["position"].map(
        lambda p: comp_map[p]["min_dist"] if p in comp_map else None
    )

    # Keep only residues with structural mapping
    valid = merged.dropna(subset=["compartment"]).copy().reset_index(drop=True)
    return valid


def build_all_pairs(
    systems: tuple[PPISystem, ...] = PRIMARY_SYSTEMS,
    ref_path: Path = PROTEINGYM_REF_PATH,
    bulk_dir: Path = BULK_DIR_PATH,
    struct_dir: Path = STRUCTURES_DIR_PATH,
) -> pd.DataFrame:
    """Constructs the combined paired dataset across all PPI systems."""
    ref_df = load_reference(ref_path)
    frames = []
    for sys in systems:
        df_sys = build_system_dataset(sys, ref_df, bulk_dir, struct_dir)
        frames.append(df_sys)

    combined = pd.concat(frames, ignore_index=True)
    return combined

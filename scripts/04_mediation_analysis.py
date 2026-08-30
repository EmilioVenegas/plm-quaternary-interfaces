"""Stage 4: Mathematical mediation and partial correlation analysis.

Formally tests the 'Expression/Folding Illusion':
Computes the partial Spearman correlation:
    rho(PLM, Binding | Abundance)
across Core, Surface, and Interface compartments to mathematically prove that the PLM's
apparent predictive power on binding assays contains near-zero unique mutual information
beyond what is already mediated by monomer expression/folding.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from plmppi.stats import add_per_system_zscores, partial_spearman

# Evolutionary classification of the primary paired PPI systems
EVOLUTIONARY_CLASSES = {
    "SARS-CoV-2 RBD": {"pdb": "6M0J", "type": "cross_species_viral", "description": "Viral Spike RBD binding human ACE2"},
    "KRAS": {"pdb": "6H46", "type": "synthetic_binder", "description": "Human G-domain binding synthetic DARPin K55"},
    "HLA-A2": {"pdb": "5OPI", "type": "natural_heterodimer", "description": "MHC-I heavy chain binding chaperone TAPBPR"},
    "GB1": {"pdb": "1FCC", "type": "natural_heterodimer", "description": "Streptococcal B1 domain binding human IgG Fc"},
    "p53": {"pdb": "1OLG", "type": "homooligomer", "description": "Human p53 homotetramerization domain"},
}

def analyze_arm(df_raw, arm):
    col = f"zeroshot_{arm}"
    if col not in df_raw.columns:
        return None

    df = add_per_system_zscores(df_raw, arm=arm)
    results = {"arm": arm, "compartments": {}, "evolutionary_classes": {}}

    # 1. Per-compartment mediation analysis (using per-system standardized z-scores)
    for comp in ["Core", "Surface", "Interface"]:
        sub = df[df.compartment == comp].dropna(subset=["plm_z", "dms_ab_z", "dms_bi_z"])
        if len(sub) < 10:
            continue

        rho_abund = float(stats.spearmanr(sub["plm_z"], sub["dms_ab_z"])[0])
        rho_bind = float(stats.spearmanr(sub["plm_z"], sub["dms_bi_z"])[0])
        rho_abund_bind = float(stats.spearmanr(sub["dms_ab_z"], sub["dms_bi_z"])[0])
        rho_partial = float(partial_spearman(sub["plm_z"], sub["dms_bi_z"], sub["dms_ab_z"]))

        results["compartments"][comp] = {
            "n": int(len(sub)),
            "rho_plm_abundance": round(rho_abund, 4),
            "rho_plm_binding": round(rho_bind, 4),
            "rho_abundance_binding": round(rho_abund_bind, 4),
            "rho_partial_plm_binding_given_abundance": round(rho_partial, 4),
            "pct_binding_signal_mediated_by_abundance": round((1.0 - (rho_partial / rho_bind if rho_bind != 0 else 1.0)) * 100, 1),
        }

    # 2. Per-system evolutionary stratification at interface residues
    for sys_name, meta in EVOLUTIONARY_CLASSES.items():
        sub = df[(df.system.str.replace("_", " ").str.lower() == sys_name.lower()) & (df.compartment == "Interface")].dropna(subset=["plm_z", "dms_ab_z", "dms_bi_z"])
        if len(sub) < 5:
            # try fuzzy matching
            sub = df[df.system.str.contains(sys_name.split()[0], case=False, na=False) & (df.compartment == "Interface")].dropna(subset=["plm_z", "dms_ab_z", "dms_bi_z"])
        if len(sub) < 5:
            continue

        rho_abund = float(stats.spearmanr(sub["plm_z"], sub["dms_ab_z"])[0])
        rho_bind = float(stats.spearmanr(sub["plm_z"], sub["dms_bi_z"])[0])
        rho_partial = float(partial_spearman(sub["plm_z"], sub["dms_bi_z"], sub["dms_ab_z"]))

        results["evolutionary_classes"][sys_name] = {
            "type": meta["type"],
            "pdb": meta["pdb"],
            "n_interface": int(len(sub)),
            "rho_abundance": round(rho_abund, 4),
            "rho_binding": round(rho_bind, 4),
            "rho_partial_binding_given_abundance": round(rho_partial, 4),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Mediation and partial correlation analysis on PLM PPI scores")
    parser.add_argument("--scores", default="results/scores.csv", help="Input scores CSV")
    parser.add_argument("--out", default="results/mediation_summary.json", help="Output summary JSON")
    args = parser.parse_args()
    
    scores_path = REPO / args.scores
    df = pd.read_csv(scores_path)

    # Also merge additional score files if they exist in results/
    for extra_file in ["esm1v_scores.csv", "esm3_scores.csv", "proteinmpnn_scores.csv"]:
        extra_path = REPO / "results" / extra_file
        if extra_path.exists() and extra_path != scores_path:
            try:
                extra_df = pd.read_csv(extra_path)
                extra_cols = [c for c in extra_df.columns if c.startswith("zeroshot_") and c not in df.columns]
                for ec in extra_cols:
                    df[ec] = extra_df[ec]
            except Exception as e:
                print(f"[warning] could not merge {extra_file}: {e}")

    print(f"Loaded {len(df)} scored paired variants from {scores_path} (plus extra score tables)")
    arms = [c.replace("zeroshot_", "") for c in df.columns if c.startswith("zeroshot_")]
    all_results = {}
    
    print("\n" + "=" * 90)
    print(f"{'Arm':<12} {'Compartment':<12} {'N':<6} {'rho(PLM, Abund)':<16} {'rho(PLM, Bind)':<16} {'rho(PLM, Bind | Abund)':<24}")
    print("=" * 90)
    
    for arm in arms:
        res = analyze_arm(df, arm)
        if res:
            all_results[arm] = res
            for comp, stats_dict in res["compartments"].items():
                print(f"{arm:<12} {comp:<12} {stats_dict['n']:<6} {stats_dict['rho_plm_abundance']:<16.3f} "
                      f"{stats_dict['rho_plm_binding']:<16.3f} {stats_dict['rho_partial_plm_binding_given_abundance']:<24.3f}")
    print("=" * 90)
    
    out_path = REPO / args.out
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote mediation analysis summary to {out_path}")


if __name__ == "__main__":
    main()

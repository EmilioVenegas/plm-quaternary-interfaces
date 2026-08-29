#!/usr/bin/env python3
"""Stage 8: Interface Hotspot vs Rim Stratification Analysis.

Piece: stratifies the N = 2,262 Interface-compartment paired variants by structural MAGNITUDE
rather than the binary Interface/Core/Surface label, to test whether the PLM interface blindspot
already reported for the aggregate "Interface" compartment (rho_partial ~ -0.19 to -0.37 across
arms, see MASTERPLAN.md) is uniform across the interface footprint, or concentrated at energetic
hotspots.

Split rule (documented, reproducible): within Interface-compartment rows only, take a TERTILE
split on `dsasa` (delta-SASA buried upon complex formation -- a continuous per-residue burial
magnitude already computed by `plmppi.interfaces.analyze_complex_structure`). The top tertile
(most buried, most energetically central to the binding interface) is labeled "Hotspot"; the
bottom tertile (marginally buried, edge-of-interface) is labeled "Rim"; the middle tertile is
labeled "Mid" and excluded from the Hotspot-vs-Rim contrast to maximize separation between the
two extreme groups. `dsasa` was chosen over `min_dist` as the primary split axis because it is a
direct measure of buried contact area (the physical quantity underlying interface energetics),
whereas `min_dist` only captures proximity to the nearest partner atom; `min_dist` (and `rsasa`)
medians are still reported per group as corroborating structural context.

For each of the 4 PLM arms (esm2-650m, esm2-3b, esmc-600m, esmc-6b), this script:
  1. Splits Interface variants into Hotspot / Mid / Rim via `plmppi.stats.assign_hotspot_rim`.
  2. Computes per-subgroup N, rho(PLM, Abundance), rho(PLM, Binding), and the partial rank
     correlation rho(PLM, Binding | Abundance) via `plmppi.stats.partial_spearman`.
  3. Fits a clustered-OLS interaction model (dms_z ~ PLM * Binding * Hotspot) on the Hotspot U Rim
     subset (Mid excluded), by reusing `plmppi.stats.run_three_way_interaction_test` verbatim --
     see `plmppi.stats.stratify_by_hotspot` docstring for how the "Interface" indicator it expects
     is repurposed to mean Hotspot(1)/Rim(0) here.
  4. Writes a JSON summary (mirroring `results/evolutionary_stratification.json`'s shape) with a
     `formal_answer` stating whether interface degradation is uniform or hotspot-concentrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from plmppi.stats import stratify_by_hotspot


def detect_arms(df: pd.DataFrame) -> list[str]:
    zs = {c.removeprefix("zeroshot_") for c in df.columns if c.startswith("zeroshot_")}
    return sorted(zs)


def build_formal_answer(all_arm_results: dict[str, dict]) -> dict[str, object]:
    """Compares Hotspot vs Rim partial rho across arms and states the plain-language verdict."""
    hotspot_rhos = []
    rim_rhos = []
    per_arm_summary = {}
    for arm, res in all_arm_results.items():
        hs = res["groups"].get("Hotspot", {})
        rim = res["groups"].get("Rim", {})
        hs_rho = hs.get("rho_partial_plm_binding_given_abundance", float("nan"))
        rim_rho = rim.get("rho_partial_plm_binding_given_abundance", float("nan"))
        hotspot_rhos.append(hs_rho)
        rim_rhos.append(rim_rho)
        per_arm_summary[arm] = {
            "hotspot_rho_partial": hs_rho,
            "rim_rho_partial": rim_rho,
            "hotspot_more_negative": bool(hs_rho < rim_rho),
            "beta_hotspot_interaction": res["interaction_test"]["beta_hotspot_interaction"],
        }

    valid_pairs = [(h, r) for h, r in zip(hotspot_rhos, rim_rhos) if h == h and r == r]
    n_hotspot_worse = sum(1 for h, r in valid_pairs if h < r)
    concentrated_at_hotspots = bool(valid_pairs) and n_hotspot_worse == len(valid_pairs)
    uniform_degradation = bool(valid_pairs) and n_hotspot_worse == 0

    if concentrated_at_hotspots:
        conclusion = (
            "PLM interface blindness is NOT uniform: it is concentrated at energetic hotspots. "
            "In all model arms, the partial rank correlation rho(PLM, Binding | Abundance) is "
            "MORE negative at Hotspot (top-tertile dsasa) residues than at Rim (bottom-tertile "
            "dsasa) residues, meaning single-chain PLMs are least informative about binding "
            "precisely where quaternary contacts are most extensive and energetically load-bearing."
        )
    elif uniform_degradation:
        conclusion = (
            "PLM interface blindness is essentially uniform across the interface footprint: the "
            "partial rank correlation rho(PLM, Binding | Abundance) is not more negative at "
            "Hotspot residues than at Rim residues in any arm, indicating the failure is a "
            "property of being in contact with the partner chain at all, not of contact magnitude."
        )
    else:
        conclusion = (
            "PLM interface blindness is mixed across arms: some models show more negative partial "
            "correlations at Hotspot residues than at Rim residues and others do not, so the "
            "hotspot-concentration effect is not uniformly directional across the 4 arms tested."
        )

    return {
        "question": "Is PLM interface blindness uniform across the Interface compartment, or concentrated at energetic hotspots (high-dsasa, close-contact residues)?",
        "concentrated_at_hotspots_all_arms": concentrated_at_hotspots,
        "uniform_degradation_all_arms": uniform_degradation,
        "n_arms_with_hotspot_more_negative": n_hotspot_worse,
        "n_arms_compared": len(valid_pairs),
        "per_arm": per_arm_summary,
        "conclusion": conclusion,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Interface Hotspot-vs-Rim structural-magnitude stratification analysis."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=REPO_ROOT / "results" / "scores.csv",
        help="Input CSV with model scores",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "hotspot_stratification.json",
        help="Output JSON path for hotspot stratification results",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=10000,
        help="Number of permutation/bootstrap iterations for the interaction test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutations",
    )
    args = parser.parse_args(argv)

    if not args.scores.exists():
        print(f"Error: scores file {args.scores} does not exist", file=sys.stderr)
        return 1

    df_scores = pd.read_csv(args.scores)
    arms = detect_arms(df_scores)
    print(f"[read] Loaded {len(df_scores)} paired variants from {args.scores}", flush=True)
    print(f"[setup] Hotspot stratification detected across {len(arms)} arms: {arms}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_arm_results = {}

    for arm in arms:
        print("\n" + "=" * 98)
        print(f"  HOTSPOT STRATIFICATION: ARM = {arm.upper()}")
        print("=" * 98)

        res = stratify_by_hotspot(df_scores, arm=arm, n_perm=args.n_perm, seed=args.seed)
        all_arm_results[arm] = res

        print(f"\nSplit rule: {res['split_rule']}")
        print(
            f"N Interface = {res['n_interface_variants']}  "
            f"(Hotspot = {res['n_hotspot']}, Mid = {res['n_mid']}, Rim = {res['n_rim']})"
        )

        print(f"\n{'Group':<10} {'N':<6} {'rho(Abundance)':<16} {'rho(Binding)':<16} {'rho(Bind | Abund)':<20}")
        print("-" * 72)
        for group, s in res["groups"].items():
            print(
                f"{group:<10} {s['n_variants']:<6} {s['rho_plm_abundance']:<16.3f} "
                f"{s['rho_plm_binding']:<16.3f} {s['rho_partial_plm_binding_given_abundance']:<20.3f}"
            )

        it = res["interaction_test"]
        print("\n[Hotspot-vs-Rim Interaction Model: dms_z ~ PLM * Binding * Hotspot]")
        print(f"beta(PLM:Binding:Hotspot) = {it['beta_hotspot_interaction']:+.4f}")
        print(f"p_clustered = {it['p_clustered']}, p_permutation = {it['p_permutation']}")

    formal_answer = build_formal_answer(all_arm_results)
    print("\n[Formal Statistical Verdict]")
    print(f"Central Question: {formal_answer['question']}")
    print(f"\nSynthesis:\n  {formal_answer['conclusion']}\n")

    summary_doc = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "n_total_variants": int(len(df_scores)),
            "n_interface_variants": int((df_scores["compartment"] == "Interface").sum()),
            "systems": df_scores["system"].value_counts().to_dict(),
            "compartments": df_scores["compartment"].value_counts().to_dict(),
        },
        "split_methodology": (
            "Tertile split on dsasa (delta-SASA buried upon complex formation) within "
            "Interface-compartment rows only: top tertile = Hotspot, bottom tertile = Rim, "
            "middle tertile excluded from the Hotspot-vs-Rim contrast. See module docstring in "
            "scripts/08_hotspot_stratification.py and plmppi.stats.assign_hotspot_rim for detail."
        ),
        "arms": all_arm_results,
        "formal_answer": formal_answer,
    }

    with args.out.open("w") as f:
        json.dump(summary_doc, f, indent=2)
    print(f"\n[write] Wrote hotspot stratification results to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Partner-Conditioned PPI Benchmark Plan

## Objective

Replace the unsupported claim that sequence-only PLMs are universally blind to quaternary interfaces with a testable benchmark question:

> Does a model predict mutation effects that are specific to an interaction partner after target-intrinsic expression effects are controlled?

A target-only score is constant for a mutated target regardless of the partner assay. It can therefore be evaluated for target viability and raw binding association, but it cannot identify partner-selective effects without partner-conditioned input.

## Evidence audit and cohort policy

The current five-system table is not a uniform set of matched expression-plus-binding experiments. `src/plmppi/data.py` inner-merges mutation labels but does not establish that the two measurements have the same biological meaning.

| System | Current local evidence | Provenance status | Use in revised work |
|---|---|---|---|
| SARS-CoV-2 RBD / ACE2 | Starr 2020 expression and ACE2-binding assays; both metadata directionalities are positive | `included` | Strict matched-expression/binding cohort |
| HLA-A2 / TAPBPR | McShan 2019 surface-expression and TAPBPR-binding assays; both metadata directionalities are positive | `included` | Strict matched-expression/binding cohort; report the chaperone context |
| KRAS / DARPin K55 | Same-study abundance and binding phenotypes, but local metadata labels both as yeast-growth fitness readouts | `conditional` | Analyze only after primary-source confirmation of construct, library, and phenotype interpretation |
| GB1 / IgG Fc | Wu 2016 and Olson 2014 are separate IgG-binding studies; neither is a monomer-abundance readout | `excluded` | Legacy replication only; never label as abundance versus binding |
| p53 / Nutlin | p53-null and p53-WT Nutlin-3 growth phenotypes; both are organismal-fitness proxies and are documented with negative raw directionality | `excluded` | Exclude from direct PPI and expression analyses |

The registry must preserve the legacy five-system cohort for reproducibility, but all revised summaries must explicitly name their provenance tier.

## Benchmark tasks

For a target mutation $m$, measured expression $A_m$, and partner-specific binding readout $B_{m,p}$:

1. **Expression:** predict $A_m$ from the mutant target sequence.
2. **Raw binding:** predict $B_{m,p}$ from the mutant target sequence.
3. **Expression-adjusted binding:** quantify association with $B_{m,p}$ after adjustment for $A_m$. This is descriptive, not causal mediation.
4. **Partner differential binding:** predict

   $$D_{m,p,q} = B_{m,p} - B_{m,q}.$$

   This is the primary benchmark endpoint. It suppresses target-intrinsic effects shared across partners and requires partner-conditioned information.

## Candidate acquisition gates

No target currently satisfies the multi-partner requirement using only local files. Candidate additions must meet all conditions below before ingestion:

1. Same target construct and mutant library across expression and at least two partner-binding measurements.
2. Named partners and compatible measurement direction/scale.
3. Source data available with mutation-level values and position mapping.
4. An available complex structure, or an explicit statement that structural-model comparison is out of scope.

Priority candidates:

- **KRAS G-domain:** Weng et al. reports the local DARPin K55 arm and additional partner screens. Ingest only after supplement-level confirmation of matching library and score semantics.
- **SARS-CoV-2 RBD:** Starr/Bloom-lab yeast-display data provide local ACE2 values and published antibody or ACE2-ortholog screens. Ingest only after verifying construct/library reuse and compatible scores.

A candidate failing any gate remains outside the primary benchmark.

## Analysis policy

- Report target-level effect sizes and confidence intervals; do not use mutation count as the effective number of independent biological systems.
- Use expression-adjusted association as a label, never as proof of causal mediation or mutual information.
- Compare target-only PLMs, a partner-sequence baseline, and complex-aware models on the same endpoint.
- Evaluate engineering utility at fixed experimental budget with enrichment, precision, recall, and uncertainty relative to random selection. Do not use false-negative rate alone for top-$k$ filtering.
- Do not claim architecture-wide failure, scaling causality, or a universal filter trap from the current data.

## Decision gates

| Gate | Required evidence | Outcome |
|---|---|---|
| Provenance | Registry validation confirms direct assay semantics | Build strict and conditional cohorts |
| Acquisition | At least one multi-partner target passes all four gates | Implement partner-differential ingestion and evaluation |
| Benchmark | Partner-conditioned models improve over target-only scores on partner differential effects with target-level uncertainty | Advance the partner-specificity claim |
| Publication | At least two independent multi-partner targets or one target plus orthogonal affinity validation | Make a general benchmark claim |

Until the acquisition gate passes, the repository should describe a phenotype-aware audit and a benchmark protocol, not report partner-specificity results.

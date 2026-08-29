# Pre-registration: PTM chemistry vs positional conservation in PLM zero-shot scoring

**Frozen on first scored variant.** Nothing below may be edited afterwards except by adding a dated
amendment section at the bottom, stating what changed and why. Design rationale and measured
coverage: [`PLAN.md`](PLAN.md). Evidence base:
[`docs/recon/recon_ptm_disulfide_results.md`](docs/recon/recon_ptm_disulfide_results.md).

The point of writing this before running anything is that the outcome space is small and the
temptation to reinterpret is large. The prior study in this line returned a clean negative
([`docs/prior_metal_result.md`](docs/prior_metal_result.md)) precisely because its falsification rule
was fixed in advance.

---

## 1. Hypotheses

**H1 (confound present).** At experimentally annotated PTM sites, ESM2 zero-shot masked-marginal
scoring fails to penalise modification-abolishing substitutions relative to
modification-preserving ones *to the degree the measured fitness data requires*, beyond how it treats
chemically identical substitutions at conservation-matched non-PTM positions.

Operationally, H1 predicts a **positive interaction statistic**:

```
residual     = z(DMS_score) - z(zero_shot_score)                 # within assay
gap(G)       = mean residual[G & preserving] - mean residual[G & abolishing]
interaction  = gap(PTM sites) - gap(matched control positions)
H1: interaction > 0
```

The residual is signed so that a positive value means the model was **too generous** relative to
measured fitness. A positive interaction therefore means excess generosity toward preserving
substitutions that is *specific to PTM sites*, not a general property of conserved positions.

**H0 (conservation only).** interaction ≈ 0. The model treats PTM sites like any other conserved
position; the preserving/abolishing asymmetry, if present, is general and not PTM-specific.

**H1 is a directional hypothesis.** A significant interaction of the *opposite* sign does not support
H1 and must be reported as a distinct, unanticipated finding — not folded into "we found an effect."

## 2. Primary outcome

- **Statistic:** interaction as defined above.
- **Locus set:** phospho Ser/Thr sites and glycosylation sequon +2 positions, pooled.
- **Model:** ESM2-650M (`facebook/esm2_t33_650M_UR50D`), for continuity with the prior study.
- **Test:** stratified permutation, 10,000 shuffles, `preserving` label permuted **within
  assay × site-class strata**, `numpy` default RNG seeded 0.
- **Threshold:** two-sided **α = 0.01**, matching the prior study. Not 0.05.
- **Expected n:** 2,847 PTM-site and 11,283 control observations; limiting cell 260 preserving
  substitutions at PTM sites.

## 3. Secondary and control arms

All are reported regardless of outcome. None can rescue a null primary.

| # | arm | pre-registered expectation |
|---|---|---|
| S1 | Ordinal test at sequon +2: do model log-odds track the measured acceptor efficiency Thr > Ser > Cys (Bause & Legler 1981, PMID 7316978)? | under H1, model fails to reproduce the ordering |
| S2 | Per-locus split (phospho alone, sequon +2 alone) | same sign in both, or the pooled result is not interpretable |
| S3 | Replication across ESM2-3B, ESMC-600M, ESMC-6B | consistent sign in all four arms |
| S4 | Taxon split (human phospho vs viral glyc) | effect not attributable to taxon alone |
| N1 | **Designed null** — Asn acceptor locus, family `{N}`, 117 positions, 0 preserving substitutions by chemistry | **exactly zero**; a non-zero result means the pipeline is broken |
| N2 | **Predicted null** — Tyr phosphosites, no preserving partner exists | ≈ 0 |
| N3 | Permuted-weight model | baseline DMS correlation collapses; interaction not significant |
| N4 | Shuffled site labels | true-label statistic sits inside the null distribution |

## 4. Falsification rule

Fixed in advance:

- **H1 supported** iff the primary interaction is **positive** and permutation **p < 0.01**, *and*
  N1 and N2 return null as specified, *and* the sign is consistent across all four model arms (S3).
- **H1 not supported** if p ≥ 0.01. This is a real, reportable outcome and will be written up as
  such. The prior metal study reached exactly this state (interaction −0.24, p = 0.42) and was
  reported as a clean negative.
- **Pipeline invalid** — no claim either way, fix and rerun — if N1 (a locus where chemistry admits
  no preserving substitution) yields a non-zero effect, or if N3/N4 fail.
- **Unanticipated finding** — reported separately, not as H1 support — if the interaction is
  significant with negative sign.

## 5. Inclusion, exclusion, and matching (fixed before scoring)

**Sites included.** UniProt-annotated phosphosites (`MOD_RES` description beginning "Phospho") whose
mapped wild-type residue is Ser or Thr; sequence-motif sequons `N-X-[ST]`, `X != P`, at the +2
position. Position mapping **must** go through pairwise alignment of the UniProt canonical sequence
to the ProteinGym `target_seq`; assumed offsets are prohibited.

**Substitution classes.** Preserving = `{S,T}` at a Ser/Thr phosphosite, `{S,T,C}` at sequon +2,
`{}` at the Asn acceptor. Abolishing = everything else, wild-type excluded.

**Deduplication.** Keyed on `(protein, position, wt, mut)`. The three `P53_HUMAN_Giacomelli_2018`
conditions and the three `SRC_HUMAN` assays re-measure the same libraries; pooling them would
triple-count the same variants.

**Control matching.** Non-PTM wild-type Ser/Thr positions in the same assays, carrying ≥1 preserving
and ≥1 abolishing substitution, caliper-matched within assay on the model's masked
`logP(WT | context)`. Caliper width and the unmatched covariate-adjusted sensitivity are both
specified in code before scoring begins.

**Censoring.** Positions whose substitutions pile up at the assay floor are flagged, not silently
pooled. `ENV_HV1BR_Haddox_2016` is known censored (35.8% of its sequon +2 substitutions at the
floor); the primary is reported **with and without** it, and the pre-registered headline is the
**with**-version so the exclusion cannot be chosen after seeing the result.

**Directionality.** Every contributing assay's sign convention is verified against ProteinGym
`raw_DMS_directionality` before any scoring. A wrong sign inverts the residual and would fabricate
or destroy the effect.

## 6. What is NOT being tested

- Whether PLMs are useful variant-effect predictors in general. They are; the prior study measured
  Spearman ρ = 0.42 at functional positions.
- Whether supervised PTM-site predictors work. Different question, different literature
  (Phosformer, PTM-Mamba are supervised classifiers, not zero-shot likelihood scorers).
- Disulfide paired epistasis. Killed for lack of ground truth: zero experimental double mutants at
  any annotated disulfide across 2,793,651 multi-mutant rows in three repositories. Retained only as
  a methods note (ε = +4.41 nats at a real disulfide vs +0.003 at a matched control).
- Anything requiring training, fine-tuning, or a wet lab. Inference only.

## 7. Honesty commitments

Carried over from the prior study, and both were load-bearing during reconnaissance:

1. Every literature claim is tagged **primary-source-verified** (the paper was opened and can be
   quoted) or **unverified**. Reconnaissance caught a delegated agent fabricating a quote,
   attributing it to the wrong paper with a wrong PMID, and asserting the chemical opposite of what
   the real source says.
2. Every count comes from really parsed data, never from a name suggesting coverage should exist.
   Reconnaissance caught PTEN reported as a zero contributor when the truth was a silent UniProt
   lookup failure; PTEN is in fact one of the strongest contributors.
3. Negative and null results are reported with the same prominence as positive ones.

---

## Amendments

*(none)*

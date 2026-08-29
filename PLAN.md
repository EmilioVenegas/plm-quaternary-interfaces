# Main plan: does a protein language model encode PTM chemistry, or only positional conservation?

Status: **reconnaissance complete, confirmatory study specified, not yet run.**
Falsification rules are fixed in [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) and must not be
edited after the first scored variant. Reconnaissance evidence:
[`docs/recon/recon_ptm_disulfide_results.md`](docs/recon/recon_ptm_disulfide_results.md).

---

## 1. The question

A phosphosite and an N-glycosylation sequon are both *chemically specific* and *evolutionarily
conserved*. Those two properties make different predictions for exactly one class of substitution:

| substitution | conservation view | chemistry view |
|---|---|---|
| `S -> T` at a Ser phosphosite | conservative, tolerated | **preserving** — Ser/Thr kinases accept both |
| `S -> A` at a Ser phosphosite | conservative-ish, tolerated | **abolishing** — no hydroxyl, no phosphate |
| `T -> S` at sequon +2 | conservative, tolerated | **preserving** — both are glycosyl acceptors |
| `T -> A` at sequon +2 | conservative-ish, tolerated | **abolishing** — occupancy goes to zero |

A model that has learned "this column is conserved, penalise everything" scores those pairs alike.
A model that has learned the chemistry separates them. The study asks which ESM2 does, with
positional conservation held fixed.

This matters because these models are used to triage human variants. A confound here puts the error
in the functionally interesting class — the substitution that silently removes a regulatory site —
and puts it exactly where the model looks most confident. Confident and wrong is worse than
uncertain.

## 2. What reconnaissance already settled

Numbers below are from really parsed data, regenerable by `scripts/recon_ptm_disulfide.py`.

**Coverage is adequate and the design survives its own worst case.**

| cell | n |
|---|---|
| Phospho Ser/Thr sites: positions / preserving / abolishing | 40 / 40 / 717 |
| Glyc sequon +2: positions / preserving / abolishing | 110 / 220 / 1,870 |
| PTM-site subtotal | **2,847** (limiting cell **260**) |
| Matched within-assay control (599 positions) | **11,283** |
| **Total** | **14,130** |

Versus the completed metal-coordination study: **17.4×** total n (14,130 vs 812) and **5.3×** the
limiting cell (260 vs 49). Dropping the single floor-censored assay still leaves 12,173 / cell 200.

**The chemistry was corrected against primary sources, and it changed the design.** The dispatch
brief assumed three-member acceptor families at both loci. Both were wrong:

- N-glycosylation's Asn acceptor is a **one-member family** `{N}`. Measured: 117 covered positions,
  2,145 available substitutions, **0** preserving. This is the same coverage collapse that zeroed
  CYP2C9 and GAL4 in the metal study — it just lands on a locus nobody flagged. It becomes a
  *designed negative control*, not a loss.
- Sequon **+2** admits `{S, T, C}` — Bause & Legler 1981 (PMID 7316978) measured Thr, Ser *and* Cys
  as competent acceptors with a graded efficiency Thr > Ser > Cys. That yields an **ordinal**
  prediction a conservation-only account cannot make.
- Tyr phosphosites have **no** preserving partner (Tyr kinases are a different enzyme class), so
  they form a second predicted-null arm.

**A worked example already shows the effect with the sign inverted.** At ERK2/MAPK1 activation-loop
Thr185 (TEY motif verified: `target_seq[182:190] == "FLTEYVAT"`), ESM2 ranks `T185S` the *most
tolerated* of 19 substitutions by a 6-nat margin (−3.34), while measured DMS ranks it the **worst of
19** (−10.43). At SARS-CoV-2 Spike T345 the model is correct. Two sites, opposite outcomes — which
is the argument for running the test rather than predicting it.

**Hardware is not the constraint.** Masked-marginal scoring needs ~1 forward pass per masked
position, so the entire study is ~1,500 forwards. All four model arms are measured feasible on an
8 GB GPU; ESMC-6B runs via CPU offload at 2.93 GB peak and 0.23 s/position.

## 3. Design

### 3.1 Primary test

Reuses the metal study's machinery unchanged: within-assay z-scored residual, interaction
statistic, stratified permutation test.

```
residual        = z(DMS_score) - z(zero_shot_score)          # within assay
gap(group)      = mean residual[group & preserving] - mean residual[group & abolishing]
interaction     = gap(PTM sites) - gap(matched control positions)
```

Positive interaction = the model over-credits preserving substitutions *specifically* at PTM sites,
beyond how it treats chemically identical substitutions at conservation-matched non-PTM positions.
Significance by stratified permutation (10,000 shuffles, `preserving` label shuffled within
assay × site-class strata), pre-registered α = 0.01.

### 3.2 The control group, and why it is within-assay

**Matched control = non-PTM wild-type Ser/Thr positions inside the same assays** carrying at least
one `{S,T,C}` preserving and one abolishing substitution. Measured: 599 positions, n = 11,283.

This is a deliberate correction of the metal study's weakness. There, BLAT_ECOLX served as an
external control, which introduced a second protein-family confound into a test already about
position class — and the result was directionally consistent with the control, which is precisely
the outcome you cannot interpret when protein identity and site class covary.

The within-assay control is **residue-identity matched**: same assay, same selection pressure, same
wild-type residue, same available substitution chemistry. Only PTM status differs.

Conservation is held fixed by **caliper matching on the model's own masked wild-type
log-probability** (`logP(WT | context)`), matched within assay. Since the hypothesis is that the
model treats PTM sites as generically conserved, matching on the model's own conservation estimate
is the correct isolation of chemistry from conservation — not circularity. Reported two ways:
caliper-matched primary, and unmatched-with-covariate as sensitivity.

### 3.3 Arms

| arm | locus | prediction under H1 | prediction under H0 |
|---|---|---|---|
| Primary | phospho Ser/Thr + sequon +2, pooled | interaction > 0 | interaction ≈ 0 |
| Ordinal | sequon +2 only | model log-odds fail to track Thr > Ser > Cys | no ordering either way |
| Designed null | Asn acceptor (family `{N}`) | **exactly 0** — chemistry admits no preserving substitution | 0 |
| Predicted null | Tyr phosphosites | ≈ 0 — no preserving partner exists | 0 |
| Replication | 4 models (ESM2-650M/3B, ESMC-600M/6B) | consistent sign across all four | inconsistent / null |

The designed-null and predicted-null arms are the falsifiers. A pipeline that reports an effect
where chemistry guarantees none is broken, and that is a check on us, not on the model.

### 3.4 Negative controls carried over from the metal study

- **Permuted-weight model** — confirms the baseline DMS correlation is learned, not architectural.
- **Shuffled site labels** — confirms the statistic cannot be manufactured by position selection.

### 3.5 Model arms

Primary is ESM2-650M, for continuity: the metal null was obtained with that exact model and
protocol, so changing checkpoints would confound "different result" with "different model". But the
hypothesis is about the masked-marginal *protocol*, and a single checkpoint invites "ESM2 artifact",
so all four arms run and all four are reported. ~15 min total added compute.

ESM3 is excluded on **data** grounds, not hardware: ESM3-open removed a ~4M-sequence Viral Denylist
from training (Hayes et al., *Science* 2025), and the glycosylation arm is 110 of 150 positions and
almost entirely viral — the nuisance variable would equal the independent variable.

## 4. Pipeline

| stage | script | output |
|---|---|---|
| 0 | `scripts/fetch_data.py` | bulk datasets (gitignored, regenerable) |
| 1 | `scripts/01_build_coverage.py` | per-position site table: assay, locus, position, WT, PTM type, preserving/abolishing counts, censoring flags |
| 2 | `scripts/02_score_models.py` | masked-marginal log-odds for every (assay, position, mutant) × 4 models |
| 3 | `scripts/03_run_test.py` | matched control assignment, residuals, interaction statistic, permutation p, all arms and controls |

Stage 1 is already validated by the reconnaissance pipeline; stages 2–3 extend it. Every stage
writes a machine-readable artifact to `results/` so any claim in the writeup is regenerable.

## 5. Threats to validity, and what is done about each

| threat | status | mitigation |
|---|---|---|
| Floor censoring hollowing out n | **measured**: 18 of 19 assay × locus strata at 0.000 floor fraction | one exception (`ENV_HV1BR`, 35.8%); report with and without, censored-aware sensitivity |
| Pseudo-replication (P53 ×3, SRC ×3 re-measure the same libraries) | **handled** | dedupe on `(protein, position, wt, mut)` |
| Assay directionality sign conventions differ per DMS | **open** | verify each contributing assay against ProteinGym `raw_DMS_directionality` before scoring |
| Conservation matching looks circular | **addressed by design** | matching variable is the model's own `logP(WT)`; report matched and covariate-adjusted |
| Glyc arm is almost entirely viral; phospho arm human | **open** | stratify by taxon, report arms separately, never pool a taxon effect into the headline |
| UniProt MOD_RES includes low-evidence sites | **open** | sensitivity restricted to sites with experimental (MS / mutagenesis) evidence |
| Silent annotation-lookup failure faking a zero | **fixed** | `repair_sequenceless_entries` (PTEN was a false zero: `PTEN_HUMAN` → obsolete O00633, empty sequence, instead of P60484) |
| Coordinate-system mismatch | **fixed** | every position mapped by real pairwise alignment; assumed offsets prohibited |

## 6. The second track, killed and kept

Disulfide-bond paired epistasis asked whether the additive masked-marginal protocol can represent a
*joint* two-cysteine constraint. Answer: **the model can, the protocol discards it.** Measured
ε = joint − additive: **+4.41 nats** at the real TEM-1 C75/C121 disulfide, **+0.34** at GFP's
C48/C70 free-thiol pair (PDB 1EMA: zero SSBOND records), **+0.003** at a distance-matched non-Cys
pair.

Killed as a study for lack of ground truth, not lack of signal: **zero** experimental double mutants
exist at any annotated disulfide across **2,793,651** multi-mutant rows in ProteinGym, MegaScale and
MaveDB. The cause is physical — a lone cysteine scrambles and aggregates, so biochemists mutate them
in pairs on purpose — so reviving it needs a new wet-lab dataset, not a better search.

Kept as a **methods note**: the ε contrast is a real, cheap, reportable result about what the
standard protocol throws away. It cannot carry a paper because its sign cannot be validated against
any experiment.

## 7. Definition of done

1. All four model arms scored; every arm and control in §3.3–3.4 reported, including the ones
   predicted to be null.
2. Primary interaction statistic and permutation p reported against the pre-registered α, whichever
   way it falls. **A null is a publishable result here** — the metal study was, and the honest
   negative is what makes the positive credible.
3. Every number in the writeup regenerable from a committed script.
4. Every literature claim tagged primary-source-verified or explicitly unverified.
5. Threat table §5 fully closed: no row left "open".

# Do protein language models encode PTM chemistry, or only positional conservation?

Post-translational modification sites are strongly conserved, and protein language models
are very good at detecting conservation. That makes a specific question hard to answer by
inspection: at an experimentally validated phosphosite or N-glycosylation sequon, does
ESM2's zero-shot masked-marginal score distinguish a modification-**preserving**
substitution (S↔T, which leaves a phospho-acceptor or a sequon +2 acceptor in place) from
a modification-**abolishing** one (S→A), *once positional conservation is controlled for*?
If the model has learned the chemistry, the preserving substitution should be scored
distinctly better than the abolishing ones at PTM sites specifically, over and above
whatever it does at conservation-matched non-PTM positions in the same assay. If the model
has only learned "this column does not vary", the PTM-site contrast and the control
contrast collapse onto each other. This repository tests that interaction.

## Status

**Reconnaissance complete. The confirmatory study is specified and has not been run.**

Everything reported below is coverage, feasibility and mechanism reconnaissance —
measured on real data, regenerable from the scripts in this repository. No hypothesis test
has been executed. The design, the analysis plan and the frozen decision rules live in:

- [`PLAN.md`](PLAN.md) — implementation and execution plan
- [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) — hypotheses, statistic, and decision
  thresholds fixed before the run
- [`docs/recon/recon_ptm_disulfide_results.md`](docs/recon/recon_ptm_disulfide_results.md)
  — the full reconnaissance report; every number on this page is sourced from it

## Headline reconnaissance findings

| Quantity | Value |
|---|---|
| Total usable design n | **14,130** |
| PTM-site substitutions | 2,847 |
| Matched within-assay control substitutions (599 control positions) | 11,283 |
| Usable PTM positions | **150** (40 phospho Ser/Thr, 110 glycosylation sequon +2) |
| Limiting cell (preserving substitutions) | **260** |
| vs. prior metal-coordination study (n = 812) | **17.4×** total |
| vs. its limiting cell (49 compatible substitutions) | **5.3×** |
| vs. its control arm (n = 284) | 39.7× |
| Censoring-robust n (dropping the one floor-censored assay) | **12,173** (15.0×), limiting cell 200 |

Two chemistry corrections found during reconnaissance changed the design and are load-bearing:

| Locus | Preserving family | Preserving substitutions per position |
|---|---|---|
| Phospho Ser/Thr site | `{S,T}` | 1 |
| Phospho Tyr site | `{Y}` | 0 — Tyr kinases are a separate enzyme class; S/T at a Tyr site is not phospho-preserving |
| Sequon Asn acceptor (+0) | `{N}` | 0 — N→D and N→Q are steric/charge mimics, not glycosyl acceptors |
| Sequon **+2** (the S/T) | `{S,T,C}` | 2, on a graded Thr > Ser > Cys scale |

The `{S,T,C}` family at +2 is primary-source verified (Bause & Legler 1981, *Biochem J*
195:639–644, PMID 7316978): threonine-, serine- and cysteine-containing sequon peptides are
all glycosylated, at Vmax ratios that give the track an ordinal prediction the
conservation-only hypothesis does not make. The Asn-acceptor locus, with 2,145 available
substitutions across 117 positions and **0** of them preserving, is retained as a designed
negative control whose predicted effect is exactly zero.

Coordinate handling is not negotiable in this codebase: PDB residue numbers, UniProt
positions and ProteinGym `target_seq` indices are three different systems that agree by
luck roughly half the time. Every mapped position goes through real pairwise alignment
(`plmconfound.mapping.build_map`); assumed offsets are prohibited. Reconnaissance caught
two cases where an offset guess would have been wrong — TEM-1's `SSBOND 1 CYS A 77 CYS A
123` sitting at `target_seq` indices 75/121, and `MET_HUMAN_Estevam_2023` being a
kinase-domain-only construct in which only 4 of 13 annotated phosphosites actually fall.

## The worked example that motivates the study

**ERK2/MAPK1 activation-loop Thr185**, assay `MK01_HUMAN_Brenan_2016`. The canonical TEY
motif is confirmed structurally rather than assumed: `target_seq[182:190] == "FLTEYVAT"`,
placing T185/Y187 exactly where human UniProt numbering says (the literature commonly uses
rat T183/Y185).

| | ESM2 log-odds | measured DMS |
|---|---|---|
| T185**S** (preserving) | **−3.34** — rank 1 of 19, most tolerated, by a 6-nat margin | **−10.43** — rank 19 of 19, *worst* |
| mean of the 18 abolishing substitutions | −11.36 | −9.15 |
| gap (preserving − abolishing) | **+8.02** | **−1.28** |

ESM2 gives the chemically conservative substitution an enormous pass; the experiment says
it is the single worst substitution at that position. **The sign is inverted.** A pooled,
all-positions Spearman correlation cannot see this.

The contrast case matters as much: at **SARS-CoV-2 Spike T345** (sequon N343-A344-T345,
confirmed at `target_seq[340:347] == "VFNATRF"`), the model is right — T345S is ranked
first (−0.41 vs −2.05 mean abolishing, gap +1.64) and the DMS agrees (−0.02 vs −0.29, gap
+0.27). At the Asn acceptor N343 of the same sequon, all 19 substitutions are abolishing
and there is no preserving substitution to get right or wrong. Two sites, opposite
outcomes: which is the argument for running the test rather than assuming its result.

## A second track, killed: disulfide-bond paired epistasis

The original scoping covered a second hypothesis — that the standard additive
masked-marginal protocol is blind to the coupling between the two cysteines of a
disulfide. Reconnaissance settled it in both directions, and the honest report is mixed.

**The model does represent the coupling.** Using a joint double-mask score and a
symmetrized conditional path (three forward passes; mechanics derived and validated in
§4.1 of the report), ε = joint − additive measures what the additive protocol throws away:

| pair | structural ground truth | ε |
|---|---|---|
| TEM-1 C75/C121 → A,A | `1BTL`: `SSBOND 1 CYS A 77 CYS A 123` | **+4.41** nats |
| GFP C48/C70 → A,A | `1EMA`: **0** `SSBOND` records — free thiols | +0.34 |
| TEM-1 G76/S122 → A,A | non-Cys, identical sequence separation | +0.003 |

ε is ~13× larger at a real disulfide than at a chemically similar free-thiol pair and
~1,300× larger than at a distance-matched non-Cys pair, with the biophysically correct
positive sign (once one partner is gone, removing the second no longer strands a reactive
thiol). So this is a *protocol* artifact, not a representational deficiency.

**The track was killed for lack of ground truth.** A 4-state double-mutant cycle needs both
single mutants and the double. Across **2,793,651** multi-mutant rows in three repositories
— ProteinGym (1,769,456), MaveDB (814,077 across 494 score sets), MegaScale (210,118
doubles) — there are **zero** experimental double mutants at any annotated disulfide. Of
263 annotated bonds across all 217 ProteinGym assays, 191 are structurally verified and 58
have single-mutant coverage at both cysteines; 0 have a double mutant hitting both. The
only Cys–Cys observations anywhere are 13 floor-censored, multiply-substituted variants at
GFP's free-thiol pair. The cause is physical and therefore permanent: an unpaired thiol
scrambles and aggregates, so the disulfide-engineering literature mutates cysteines in
pairs on purpose and omits the single-Cys variants a cycle requires.

This is reported as a negative result worth publishing as a short methods note — a
searched-and-closed absence with a measured mechanism attached — not as a failed track.
Reviving it would require a new wet-lab dataset, not a better search.

## Repository layout

| Path | Contents |
|---|---|
| [`src/plmconfound/`](src/plmconfound/) | the package: `chemistry`, `stats`, `data`, `mapping`, `scoring`, `models` |
| [`scripts/`](scripts/) | pipeline entry points (reconnaissance, model feasibility, the prior metal-coordination study) |
| [`docs/recon/`](docs/recon/) | [reconnaissance report](docs/recon/recon_ptm_disulfide_results.md) and the original [dispatch brief](docs/recon/exploration_prompt_ptm_disulfide.md) |
| [`results/recon/`](results/recon/) | coverage and diagnostic artifacts (per-track coverage CSVs, censoring diagnostic, feasibility JSON) — all regenerable |
| [`data/`](data/) | acquired datasets: ProteinGym reference + assays, UniProt feature cache, PDB records, MegaScale. Bulk downloads are gitignored |
| [`tests/`](tests/) | package tests |
| [`bin/setup.sh`](bin/setup.sh) | environment bootstrap |

## Setup

Two virtual environments are required, and this is not incidental tidiness. The core
pipeline is pinned ([`requirements-core.lock.txt`](requirements-core.lock.txt)) against a
specific torch/transformers pair; the `esm` package used for the ESMC models pulls its own
torch and would silently replace it, breaking the pinned core environment and with it the
reproducibility of every number above. The ESMC stack therefore gets its own env
([`requirements-esmc.lock.txt`](requirements-esmc.lock.txt)).

```
bin/setup.sh
```

creates both (`.venv` for the core pipeline, `.venv-esmc` for ESMC) and is the supported
path. Python 3.12; import name `plmconfound`.

## Hardware

Developed on an RTX 4060 Laptop (8,188 MiB VRAM) with 62 GB system RAM. Inference only —
nothing here trains a model.

The 8 GB ceiling is **not** the binding constraint for this workload. Masked-marginal
scoring needs roughly one forward pass per masked position — about 1,500 for the whole of
the confirmatory study, not millions — so seconds-per-forward is acceptable and system RAM
becomes substitutable for VRAM.

| model | params | mode | peak VRAM | throughput | full study (~1,500 positions) |
|---|---|---|---|---|---|
| `esm2_t33_650M_UR50D` | 0.65 B | GPU fp16 | 1.45 GB | 0.031 s @286 res | ~1 min |
| `esm2_t36_3B_UR50D` | 2.84 B | GPU fp16 | 5.93 GB | 0.093 s @286 res | ~3 min |
| `biohub/ESMC-600M` | 0.58 B | GPU fp16 | 2.39 GB | 0.022 s @286 res | <1 min |
| `biohub/ESMC-6B` | 6.35 B | GPU fp16 | — | OOM | infeasible |
| `biohub/ESMC-6B` | 6.35 B | **CPU offload**, batch 128 | **2.93 GB** | **0.23 s/position** | ~6 min |

ESMC-6B's 25.4 GB of fp32 weights cannot fit, but accelerate's `cpu_offload` keeps them in
system RAM and streams each submodule to the GPU on demand — 0.16 GB peak at batch 1,
2.93 GB at batch 128, where batching amortises one weight stream across many independent
masked variants for a 7.7× speedup (1.77 → 0.23 s/position). ESM2-650M remains the primary
model for continuity with the prior study; all four are feasible, so the confirmatory run
replicates across them rather than resting on one checkpoint. ESM3 is excluded on data
grounds, not hardware: its open release was trained with a ~4M-sequence viral denylist, and
110 of the 150 usable positions are viral glycoproteins, which would make the nuisance
variable identical to the independent variable.

## Provenance and honesty

This repository continues a completed prior study — a test of the same interaction at
metal-coordinating residues, which returned a clean negative: n = 812 (528 metal-position,
284 control), observed interaction statistic −0.24, stratified permutation p = 0.42,
against a pre-registered threshold of p < 0.01. That study also established the baseline
validity of the protocol (zero-shot score correlates with measured fitness at both metal
and control positions). The present study is the better-powered follow-up at a locus class
where the chemistry admits a preserving substitution.

It inherits that project's discipline:

- Every literature claim is tagged `[VERIFIED-IN-SESSION]` (opened and read directly) or
  `[SCOUT-REPORTED, not independently verified]`. Anything still carrying the latter tag
  must be re-read before it enters a manuscript.
- Every count in the report comes from data that was actually parsed, including the zeros.
  Three of eleven named phospho proteins (BRCA1, RAF1, YAP1) contribute exactly nothing
  and are reported as such; one named candidate (`A4D664_9INFA_Soh_2019`) turned out to be
  influenza PB2 polymerase rather than hemagglutinin, with 0 sequons in 759 residues.

Two real errors were caught this way during reconnaissance, and both are recorded rather
than quietly fixed:

1. **A fabricated literature quote from a delegated agent.** A subagent supplied a verbatim
   quote, attribution and PMID asserting that cysteine at the sequon +2 position abolishes
   glycosylation. The paper cited does not contain the sentence, the PMID was wrong, and
   the real source (Bause & Legler 1981) reports the *opposite*: cysteine peptides are
   glycosylated. The error had pointed the design in the wrong direction.
2. **A silent UniProt lookup failure that made PTEN a false zero.** ProteinGym lists PTEN's
   `UniProt_ID` as `PTEN_HUMAN`, which resolves to obsolete accession O00633 — an empty
   sequence with zero features — rather than human PTEN P60484. The coverage function found
   no phosphosites, emitted no rows, and PTEN appeared as a clean zero indistinguishable
   from BRCA1's genuine one. Nothing raised an error. Repaired (accepting a replacement
   only when its sequence exactly matches that assay's `target_seq`), PTEN is one of the
   track's strongest contributors: 11 annotated phosphosites, all mapped, all 11 with DMS
   coverage, including the CK2/ROCK1 C-terminal cluster S380/T382/T383/S385. The repair
   moved the headline figures from n = 13,359 / limiting cell 250 to n = 14,130 / cell 260,
   and the phospho arm from 30 to 40 usable positions.

A zero produced by a lookup failure and a zero produced by biology must never render
identically. Both errors were caught by mechanically auditing claims against source data,
not by reading the prose.

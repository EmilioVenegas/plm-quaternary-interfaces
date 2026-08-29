# Deep-dive exploration prompt: PTM-site confound & disulfide-bond paired epistasis

Status: **EXECUTED 2026-08-28. Results: `research/recon_ptm_disulfide_results.md`.**
Verdict reached: Track A = GO (narrowed to phospho Ser/Thr sites + glycosylation sequon
position +2, real n=14,130, censoring-robust at 12,173); Track B = KILL as designed —
zero experimental double mutants exist at any annotated disulfide across 2,793,651
multi-mutant rows in ProteinGym, MegaScale and MaveDB. The discount this brief
anticipated did apply: 5 of the 28 named candidate assays contribute nothing, and one
named candidate (`A4D664_9INFA`) is not even the protein the brief assumed.

Original dispatch brief follows, unchanged, for provenance.



---

## Context (shared by both tracks)

**What's already built and validated** (`analysis/metal_coordination_confound.py`):
ProteinGym DMS metadata parsing -> real holo PDB structure fetching from RCSB ->
distance-based coordinating/catalytic-residue extraction (Biopython, cross-verified
against literature) -> PDB-to-ProteinGym sequence-numbering alignment (catches Ambler-
numbering and non-human-ortholog offset traps -- this bit real twice on the metal
project) -> ESM2-650M masked-marginal zero-shot scoring -> within-assay z-scored
residual -> stratified permutation test on a (compatible - incompatible) x
(annotated-site - matched-control) interaction -> permuted-weight ESM2 control ->
shuffled-label control. All of this is reusable machinery, not a from-scratch build.

**What it found on metal coordination** (closed, do not re-litigate): null result.
Interaction p=0.42 on n=812 (528 metal-position + 284 control-position mutations).
Directionally consistent with the control, both negative controls held. Two real
metal proteins (CYP2C9, GAL4) contributed zero usable "compatible" substitutions
because their donor chemistry is Cys-only (a 1-member family can never produce a
family-preserving non-WT substitution) -- this is the exact kind of coverage collapse
this deeper pass must check for up front, not discover after building the pipeline.

**Resource constraints -- corrected, read carefully:**
- GPU: RTX 4060 Laptop, **8GB VRAM, inference-only**. ESM2-650M confirmed to load and
  score ~55 positions across 7 proteins in 6 seconds. This is the one hard, real
  constraint: no training, no fine-tuning, no multi-GPU. Larger ESM2 checkpoints (3B,
  15B) should be *checked* for single-sequence batch-1 inference feasibility, not
  assumed impossible -- do not rule them out without trying.
- **Disk: 100GB+ available, not a constraint.** The earlier scouting pass flagged
  UniRef50's ~100GB full fasta as a "disk/bandwidth bottleneck" for a *different*
  (pretraining-leakage) direction and steered toward API-only lookups to avoid it --
  that reasoning does not apply here and should not be imported by default. For these
  two tracks: full local bulk downloads (complete PDB `SSBOND` export, full
  PhosphoSitePlus/dbPTM dumps, the complete 776k-variant Tsuboyama MegaScale set,
  full BioLiP) are all cheap and preferable to API-only workarounds where a full local
  copy gives more reliable, offline-reproducible cross-referencing. Only avoid a bulk
  download if there's a *real* reason (e.g. licensing gate, or the API is genuinely
  faster/simpler for a small lookup) -- state that reason explicitly if you make that
  call, don't default to it.
- No wet lab. No paid compute.

**Standing discipline (non-negotiable, carried over from the metal project):**
- Every "prior art" claim must be tagged primary-source-verified (you read the actual
  paper/dataset page) vs. search-engine-synthesized (a search tool composited it across
  sources and might be wrong). Do not present synthesis as fact.
- Every "N candidate assays/proteins" claim must be a real count from real data
  (an actually-parsed CSV or a real API response), not "the protein name suggests this
  should have coverage." The metal project's CYP2C9/GAL4 surprise is the cautionary
  tale: named-candidate counts and *usable* counts are different numbers.
- Numbering: PDB residue numbers, UniProt/PhosphoSitePlus site numbers, and ProteinGym
  `target_seq` 1-indexed positions are three different coordinate systems that agree
  by luck about half the time. Verify every mapped position by real sequence alignment
  (Biopython `PairwiseAligner`, as done for calmodulin and BLAT_ECOLX), never by
  assuming they match.

---

## Track A: PTM-site chemistry-vs-conservation confound

**Claim to sharpen and test:** At experimentally validated phosphorylation sites
(Ser/Thr/Tyr) and N-linked glycosylation sequons (N-X-[S/T], X!=P), does ESM2 zero-shot
masked-marginal scoring fail to disproportionately penalize modification-abolishing
substitutions (S->A, N->Q) relative to modification-compatible ones (S<->T, N->D), once
positional conservation is controlled for -- i.e. does it treat these positions as
generically conserved rather than specifically PTM-functional?

**Tasks:**
1. Verify the literature-gap claim directly. Read (not synthesize) the actual
   Phosformer and PTM-Mamba papers/repos and confirm they are supervised classifiers on
   frozen/fine-tuned embeddings (a different question from a zero-shot chemistry-vs-
   conservation confound test) rather than something that already runs this exact
   comparison.
2. For every one of the ~20 named candidate ProteinGym assays (SRC, MAPK1/ERK2, MET,
   RAF1, p53 x2, PTEN x2, ADRB2, YAP1, PRKN, BRCA1, plus the glycosylation-sequon set:
   HIV Env x2-3, Zika Env, Influenza HA x2, SARS-CoV-2 Spike, ACE2), check for real,
   not assumed:
   a. Does the DMS assay's mutated region actually include the annotated PTM residue
      position, or does the library only cover a different domain of the protein?
   b. What is the PhosphoSitePlus/UniProt-sourced coordinate for the site, and does it
      require realignment to ProteinGym's `target_seq` indexing (assume yes until
      checked)?
   c. How many *distinct* compatible-family and incompatible-family substitutions are
      actually present in the DMS single-mutant rows at that position? (Need >=1 of
      each per position, ideally several, for the position to contribute usable signal
      -- this is exactly the check that zeroed out CYP2C9 and GAL4 in the metal
      project.)
3. Build a real per-assay coverage table (assay, position, WT residue, PTM type,
   n_compatible, n_incompatible) -- the Track A equivalent of the metal project's
   `site_df` verification table -- before writing any scoring code.
4. Assess family-granularity risk specifically for this track: the phospho-acceptor
   family {S,T,Y} has 3 members (better than metal thiolate's 1-member {C} problem) and
   the glycosylation-sequon Asn family {N,D,Q} has 3 members -- confirm this actually
   yields non-zero compatible-substitution coverage per protein, don't just assert it
   by analogy.
5. Decide and justify the matched-control group: an external non-PTM protein (mirroring
   BLAT_ECOLX's role for the metal test) vs. non-PTM conserved positions *within the
   same* kinase/glycoprotein assays (higher power, avoids introducing a second
   protein-family confound, but needs its own "is this really non-functional" check).
   Recommend one, with reasoning.
6. Produce a realistic final-n estimate (usable compatible + incompatible mutation
   count) before recommending this track be built. If it's not clearly larger than the
   metal project's n=812, say so.

## Track B: Disulfide-bond paired epistasis

**Claim to sharpen and test:** ESM2's masked-marginal scoring is additive/independent
by construction (each position scored via its own conditional distribution given the
unmasked rest of the sequence). Real disulfide bonds impose a *joint* constraint on
two positions simultaneously -- a single free-thiol mutation is typically far more
deleterious than the additive sum of "how tolerant is position i" + "how tolerant is
position j" would predict, and some double mutants (e.g. compensatory Cys-pair shifts)
rescue function in ways addition can't capture. Does comparing ESM2's additive score
against a joint/conditional pseudo-likelihood score reveal this blind spot -- i.e. can
the model represent the epistasis at all, and if so does it get the *sign* right?

**Tasks:**
1. Read the three prior-art anchors directly, primary source, not summary: Kalinina Lab
   (Feb 2026), Nambiar & Maslov (Sep 2025), Beltran & Lehner (*Nature* 2025, already
   read once this project -- re-check specifically for anything on paired-Cys epistasis,
   which wasn't the focus of the earlier read). State plainly whether the exact
   additive-vs-joint pseudo-likelihood test at disulfides has already been run by any
   of them, with a direct quote/citation, not a paraphrase.
2. Since disk is not constrained: pull the **full local PDB `SSBOND` records** (RCSB
   bulk metadata export, well under 1GB, not the ~100GB UniRef case from a different
   direction) and cross-reference against *every* ProteinGym assay with a curated PDB
   structure (not just the ones already named by the scout) to find every protein with
   both (a) an experimentally resolved disulfide bond and (b) DMS coverage at both
   cysteine positions.
3. Quantify real double-mutant coverage per candidate protein: how many actual
   multi-mutant DMS rows (`DMS_number_multiple_mutants` > 0 assays) target **both**
   cysteines of the same real disulfide simultaneously? This is the likely binding
   constraint -- most DMS libraries are single-mutant saturation designs, and where
   double mutants exist they're usually a random combinatorial subset, not systematic
   Cys-Cys pair coverage. Report the real achievable n; do not assume it's adequate.
4. Check the full local Tsuboyama MegaScale set (Zenodo 7992926, 776k variants, 350
   natural + 100 de novo domains) for the same thing -- does it include any
   combinatorial double mutants at native Cys pairs, or is it single-mutant-only (as
   the earlier scout's summary suggested)? Verify directly from the downloaded data,
   not from the summary.
5. Work out the exact ESM2 scoring mechanics before committing: can a single forward
   pass mask both cysteine positions jointly and read off a proper joint distribution,
   or does this require two passes (mask i, condition on masked j vs. real j) to build
   a conditional pseudo-likelihood? Validate the chosen formula on one worked example
   (a real, well-characterized disulfide, e.g. in one of the assay proteins already on
   disk) before generalizing -- the same discipline as the VIM-2 3H/DCH validation that
   caught the metal pipeline's correctness early.
6. If ProteinGym-wide pooled coverage turns out thin (likely, per point 3), assess
   whether a single deep-dive protein with genuine combinatorial Cys mutagenesis data
   (classic disulfide-engineering study systems -- T4 lysozyme, BPTI, RNase A, or
   similar -- check for public combinatorial or double-mutant-cycle data, not just DMS)
   is a more viable substitute design than a pooled cross-protein test.

---

## Cross-track deliverable (required, not optional)

A single comparison covering both tracks:

| | Track A (PTM confound) | Track B (disulfide epistasis) |
|---|---|---|
| Real achievable n (post-coverage-check) | | |
| Statistical power vs. the null metal result (n=812) | | |
| Data/feasibility risk found | | |
| Novelty confidence (primary-source-verified gap vs. still-uncertain) | | |
| Reuses existing pipeline as-is / needs new scoring mechanics | | |

Plus an explicit recommendation: run Track A alone, Track B alone, both (sequentially
or as one combined paper with two independent confound tests), or neither if both
collapse under real coverage checks the way two of six metal-protein candidates did.

## Acceptance criteria

- Every named candidate assay/protein carries a real, checked count -- not "should have
  coverage," an actual number pulled from parsing the real DMS CSV or querying a real
  API/local database.
- At least one fully worked numeric example per track, structurally validated against
  known biochemistry (mirrors the VIM-2 3H/DCH check).
- Explicit go / kill / narrow verdict per track, each with the specific reason.
- Every literature claim tagged primary-source-verified or synthesis-derived.
- No execution of the actual confound test itself in this pass -- this is reconnaissance
  and pipeline-design-readiness, not the confirmatory run.

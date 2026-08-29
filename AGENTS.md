# Agent instructions

Canonical instructions for any coding agent working in this repository (Claude Code,
Antigravity, Oh My Pi, or otherwise). `CLAUDE.md` points here.

## What this project is

Protein language models (ESM2, ESMC) are widely used for zero-shot variant effect prediction
(VEP), yet their representations may conflate **chemical function with generic positional conservation**.
At post-translational modification (PTM) sites (Ser/Thr phosphosites and N-linked glycosylation sequons),
we test whether masked-marginal scoring distinguishes chemically **modification-preserving**
substitutions (`S<->T`, `T->C` at sequon +2) from **modification-abolishing** ones (`S/T->A`), once
positional conservation is strictly controlled for.

Prior study in this repository family: metal-coordination confound (`docs/prior_metal_result.md`),
which found a clean, controlled negative (interaction $p = 0.42$, $n = 812$). This project scales
the design to $n = 14,130$ across 150 usable positions in 13 contributing assays.

Read `PLAN.md` and `PRE_REGISTRATION.md` before executing any stage.

## Hard constraints

- **Hardware**: RTX 4060 Laptop (8,188 MiB VRAM), 62 GB system RAM.
  - Masked-marginal scoring is inference-only (~1,500 forward passes total).
  - ESM2-650M and ESMC-600M run directly in VRAM (< 2.7 GB).
  - ESM2-3B runs in fp16 (5.93 GB peak).
  - ESMC-6B (6.35B params) runs via `accelerate.cpu_offload` with batch size 128 (2.93 GB peak VRAM, 0.23 s/position).
  - **No training, no fine-tuning, no paid compute.**
- **No wet lab.**

## Dual environment structure — mandatory

This repository requires **two virtual environments** by architectural design:
- `.venv`: Python 3.12, PyTorch 2.5.1+cu121, Transformers 5.16.1, Biopython 1.88, `plmconfound` (editable).
  Used for data curation, coordinate mapping, ESM2 scoring, and all statistical tests.
- `.venv-esmc`: Python 3.12, `esm 3.4.0` (which pulls PyTorch 2.11.0+cu130 and Transformers 4.57.6).
  Used strictly for `esmc-600m` and `esmc-6b` scoring passes.

Do **NOT** attempt to merge these environments: `pip install esm` into `.venv` will silently replace
the pinned PyTorch/CUDA wheels and invalidate the reproducibility of the pipeline.

## Methodological standards — strictly enforced

1. **Pre-registration is frozen**: Falsification rules in `PRE_REGISTRATION.md` are locked at $\alpha = 0.01$
   with directional hypothesis $H_1: \text{interaction} > 0$. Do not alter rules after scoring.
2. **Acceptance criteria from real data only**: Every count must be drawn from parsed CSVs/JSONs.
   Never assert coverage based on protein name resemblance alone (e.g., PTEN was nearly a false zero
   due to an obsolete UniProt entry O00633; fixed via `repair_sequenceless_entries` to P60484).
3. **Coordinate alignment integrity**: PDB numbers, UniProt numbers, and ProteinGym `target_seq` indices
   are distinct systems. All mappings must go through `plmconfound.mapping.build_map` pairwise alignment.
4. **Report negative results cleanly**: A null result ($p \ge 0.01$) is a valid and publishable finding.
   Negative controls ($N_1$ Asn-acceptor designed null, $N_2$ Tyr predicted null, $N_4$ shuffled labels)
   must be reported alongside primary interaction tests.

## Working style

- Report outcomes faithfully with exact numbers.
- Ensure all tests pass (`pytest`) before committing any code.
- Keep `src/plmconfound/` clean of import-time side effects (no network I/O or model loading at import).

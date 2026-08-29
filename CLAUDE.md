# CLAUDE.md

**Read `AGENTS.md` first** — it is the canonical instruction file for this repository and
applies to you in full.

Then read:
- `PLAN.md` for the confirmatory study design and threat mitigations.
- `PRE_REGISTRATION.md` for the frozen falsification rules ($\alpha = 0.01$, directional $H_1$).
- `docs/RUNBOOK.md` for operational commands (environment setup, data fetch, stage execution, testing).
- `docs/HANDOFF.md` for current project state and immediate next actions.

## Repository Overview

- `src/plmconfound/`: Core modular package (`chemistry`, `data`, `mapping`, `models`, `scoring`, `stats`).
- `scripts/`: Stage-based execution pipeline (`01_build_coverage.py`, `02_score_models.py`, `03_run_test.py`, `fetch_data.py`).
- `tests/`: Hermetic pytest suite (108 tests passing offline).
- `docs/`: Provenance documents (`prior_metal_result.md`), operational guides (`RUNBOOK.md`, `HANDOFF.md`), and reconnaissance records (`docs/recon/`).
- `results/`: Output artifacts from pipeline runs (`variants.csv`, `scores.csv`, `test_<arm>.json`, and `results/recon/`).

## Environment Execution Rules

- Stage 1, Stage 2 (ESM2 arms), and Stage 3 MUST use `.venv/bin/python`.
- Stage 2 (ESMC arms: `esmc-600m`, `esmc-6b`) MUST use `.venv-esmc/bin/python`.
- Run pytest with `.venv/bin/pytest`.

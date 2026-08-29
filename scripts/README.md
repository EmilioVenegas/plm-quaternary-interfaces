# Pipeline & Utility Scripts

This directory contains stage-based pipeline runners and utility scripts for the PTM confound study.

---

## Script Index

### 1. Confirmatory Pipeline Stages

| Script | Purpose | Environment | Primary Inputs | Primary Outputs |
| :--- | :--- | :--- | :--- | :--- |
| `01_build_coverage.py` | Stage 1: Maps UniProt PTM annotations & sequons onto target sequences and emits full observation table. | `.venv` | `data/proteingym/` | `results/variants.csv`, `results/coverage_summary.json` |
| `02_score_models.py` | Stage 2: Runs masked-marginal zero-shot scoring across specified model arms. | `.venv` / `.venv-esmc` | `results/variants.csv` | `results/scores.csv` |
| `03_run_test.py` | Stage 3: Runs caliper matching, within-assay residual computation, and stratified permutation testing. | `.venv` | `results/scores.csv` | `results/test_<arm>.json` |

### 2. Data & Infrastructure Utilities

| Script | Purpose | Environment | Primary Inputs | Primary Outputs |
| :--- | :--- | :--- | :--- | :--- |
| `fetch_data.py` | Idempotent dataset & PDB downloader with size and checksum validation. | `.venv` | Remote URLs | `data/` |
| `model_feasibility.py` | VRAM and throughput profiling across ESM2 and ESMC architectures (batch scaling, offload). | `.venv-esmc` | `data/` | `results/model_feasibility.json` |
| `recon_ptm_disulfide.py`| Complete exploratory reconnaissance pipeline (reproduces initial report metrics). | `.venv` | `data/` | `results/recon/` |
| `metal_coordination_confound.py` | Archival script from prior study evaluating metal coordination in PLMs ($n = 812$). | `.venv` | `data/` | Reference |

---

## Example Pipeline Workflow

```bash
# 1. Generate observation table
.venv/bin/python scripts/01_build_coverage.py

# 2. Score primary model
.venv/bin/python scripts/02_score_models.py --arms esm2-650m

# 3. Evaluate statistical hypothesis
.venv/bin/python scripts/03_run_test.py --arms esm2-650m
```

# Runbook

Operational instructions for running, testing, and replicating the PTM-confound study.

---

## 1. Environment Setup

The repository uses two virtual environments to isolate conflicting dependencies (`esm` vs core PyTorch/Transformers):

```bash
# Set up both environments (requires uv or python3.12-venv)
bin/setup.sh

# Or set up core only (for data curation and ESM2 scoring):
bin/setup.sh --core-only

# Or set up ESMC only (~5.8 GB):
bin/setup.sh --esmc-only
```

Verify environments:
```bash
.venv/bin/python -c "import plmconfound, torch; print('Core OK:', plmconfound.__version__, torch.__version__, 'CUDA:', torch.cuda.is_available())"
.venv-esmc/bin/python -c "import esm, torch; print('ESMC OK:', esm.__version__, torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

---

## 2. Dataset Acquisition

Bulk datasets are excluded from git. Re-fetch or verify local datasets:

```bash
# Dry run (inspect expected files and sizes)
.venv/bin/python scripts/fetch_data.py --dry-run

# Fetch missing datasets (ProteinGym bulk, MegaScale, PDB structures)
.venv/bin/python scripts/fetch_data.py --only all
```

---

## 3. Running the Test Suite

Run unit and regression tests (hermetic, offline, no GPU required):

```bash
.venv/bin/pytest -v
```

Optional markers (require network or slow bulk data):
```bash
# Include network test for RCSB PDB ssbond lookup:
PLMCONFOUND_NETWORK_TESTS=1 .venv/bin/pytest -v -m network

# Include slow tests reading bulk CSVs:
PLMCONFOUND_SLOW_TESTS=1 .venv/bin/pytest -v -m slow
```

---

## 4. Confirmatory Pipeline Execution

### Stage 1: Build Observation Table
```bash
.venv/bin/python scripts/01_build_coverage.py \
    --out results/variants.csv \
    --summary results/coverage_summary.json
```
Output: `results/variants.csv` containing ~16,755 observations (PTM-sites + matched within-assay controls across 13 contributing assays).

### Stage 2: Masked-Marginal Scoring

**Primary model (ESM2-650M):**
```bash
.venv/bin/python scripts/02_score_models.py \
    --variants results/variants.csv \
    --out results/scores.csv \
    --arms esm2-650m
```

**Replication model (ESM2-3B):**
```bash
.venv/bin/python scripts/02_score_models.py \
    --variants results/variants.csv \
    --out results/scores.csv \
    --arms esm2-3b \
    --resume
```

**Replication models (ESMC-600M & ESMC-6B):**
```bash
.venv-esmc/bin/python scripts/02_score_models.py \
    --variants results/variants.csv \
    --out results/scores.csv \
    --arms esmc-600m \
    --resume

.venv-esmc/bin/python scripts/02_score_models.py \
    --variants results/variants.csv \
    --out results/scores.csv \
    --arms esmc-6b \
    --batch-size 128 \
    --resume
```

### Stage 3: Statistical Hypothesis Testing
```bash
.venv/bin/python scripts/03_run_test.py \
    --scores results/scores.csv \
    --arms esm2-650m,esm2-3b,esmc-600m,esmc-6b \
    --n-perm 10000 \
    --seed 0 \
    --caliper 0.25 \
    --out-dir results/
```
Output: `results/test_<arm>.json` and formatted console output reporting the pre-registered verdict.

---

## 5. Benchmarking & Hardware Profiling

To re-verify model feasibility and VRAM allocation across model arms:
```bash
.venv-esmc/bin/python scripts/model_feasibility.py
```
Output: `results/model_feasibility.json`.

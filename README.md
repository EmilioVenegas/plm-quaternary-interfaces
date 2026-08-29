# Quaternary Interface Failure Modes in Protein Language Models (`plmppi`)

Testing whether monomeric protein language models (ESM2, ESMC) exhibit a systematic failure mode
at quaternary oligomeric interfaces when predicting complex binding versus monomer stability on identical target proteins.

---

## Architecture & Quick Start

This project uses a dual virtual environment architecture:
- `.venv`: Python 3.12, PyTorch 2.5.1+cu121, Transformers, Biopython, SciPy (`plmppi`). Used for structural analysis, data curation, ESM2 scoring, and statistical testing.
- `.venv-esmc`: Python 3.12, `esm 3.4.0` (PyTorch 2.11.0+cu130, Transformers 4.57.6). Used strictly for ESMC scoring passes.

### Pipeline Execution

```bash
# 1. Build paired variant observation table with structural compartments
.venv/bin/python scripts/01_build_pairs.py --out results/pairs.csv --summary results/pairs_summary.json

# 2. Score model arms across ESM2 and ESMC
.venv/bin/python scripts/02_score_models.py --variants results/pairs.csv --out results/scores.csv --arms esm2-650m
.venv/bin/python scripts/02_score_models.py --variants results/pairs.csv --out results/scores.csv --arms esm2-3b --batch-size 4 --resume
.venv-esmc/bin/python scripts/02_score_models.py --variants results/pairs.csv --out results/scores.csv --arms esmc-600m --resume
.venv-esmc/bin/python scripts/02_score_models.py --variants results/pairs.csv --out results/scores.csv --arms esmc-6b --batch-size 64 --resume

# 3. Statistical hypothesis testing (pre-registered three-way interaction test)
.venv/bin/python scripts/03_run_test.py --scores results/scores.csv --n-perm 10000 --seed 42 --out-dir results/

# 4. Run test suite
.venv/bin/pytest -v
```

---

## 5 Primary Paired PPI Systems

1. **SARS-CoV-2 RBD** (`6M0J` chain E): `SPIKE_SARS2_Starr_2020_expression` vs `SPIKE_SARS2_Starr_2020_binding` (ACE2 complex).
2. **KRAS G-domain** (`6H46` chain A): `RASK_HUMAN_Weng_2022_abundance` vs `RASK_HUMAN_Weng_2022_binding-DARPin_K55` (DARPin K55 complex).
3. **HLA-A*02:01** (`5OPI` chain A): `Q53Z42_HUMAN_McShan_2019_expression` vs `Q53Z42_HUMAN_McShan_2019_binding-TAPBPR` (TAPBPR complex).
4. **GB1** (`1FCC` chain C): `SPG1_STRSG_Wu_2016` vs `SPG1_STRSG_Olson_2014` (IgG Fc complex).
5. **p53 tetramer domain** (`1OLG` chain A): `P53_HUMAN_Giacomelli_2018_Null_Nutlin` vs `P53_HUMAN_Giacomelli_2018_WT_Nutlin` (homotetramer).

---

## Pre-Registration

All hypothesis tests, sample sizes, and falsification criteria are pre-registered in [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) at $\alpha = 10^{-5}$.
Design rationale and structural rules are detailed in [`PLAN.md`](PLAN.md).

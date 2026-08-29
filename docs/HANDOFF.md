# Project Handoff

Status: **Reconnaissance complete; standalone package and pipeline initialized; test suite passing (108/108); ready for confirmatory execution.**

---

## 1. What Has Been Done

1. **Reconnaissance Completed (`docs/recon/`)**:
   - Evaluated Track A (PTM-site chemistry-vs-conservation confound) and Track B (disulfide paired epistasis).
   - Provenance documented in `docs/recon/recon_ptm_disulfide_results.md` and `docs/prior_metal_result.md`.
   - **Track A is the winner**: Usable $n = 14,130$ across 150 positions, limiting cell $n = 260$ (5.3× prior metal baseline of 49).
   - **Track B killed**: 0 double mutants exist at native disulfides across 2,793,651 multi-mutant rows in ProteinGym, MegaScale, and MaveDB.
2. **Package Structured (`src/plmconfound/`)**:
   - `chemistry.py`: Sequon regex, acceptor families `{S,T}`, `{S,T,C}` (Bause & Legler 1981), `{N}`.
   - `data.py`: ProteinGym and UniProt loaders with sequence-exact repair (`repair_sequenceless_entries` fixes PTEN P60484).
   - `mapping.py`: Global pairwise alignment (`build_map`) preventing coordinate offsets (TEM-1 Ambler offset, MET domain truncation).
   - `models.py`: ESM2 (650M/3B) and ESMC (600M/6B CPU offload).
   - `scoring.py`: Batched masked log-probability and multi-position epistasis scoring.
   - `stats.py`: Within-assay standardization, residual computation, interaction statistics, and stratified permutation testing.
3. **Pipeline Scripts Written (`scripts/`)**:
   - `01_build_coverage.py`: Emits `results/variants.csv`.
   - `02_score_models.py`: Batched scoring with resume capability across models.
   - `03_run_test.py`: Stratified permutation test and pre-registered falsification verdicts.
   - `fetch_data.py`: Safe, idempotent dataset downloader with integrity verification.
   - `model_feasibility.py` & `recon_ptm_disulfide.py`: Fully reproducible benchmark & recon pipelines.
4. **Testing & Environments**:
   - 108 unit and regression tests in `tests/` passing cleanly in 0.5s.
   - Dual virtualenvs provisioned: `.venv` (core) and `.venv-esmc` (ESMC).

---

## 2. Immediate Next Actions

To execute the full confirmatory study:

1. **Run Stage 1**:
   ```bash
   .venv/bin/python scripts/01_build_coverage.py --out results/variants.csv --summary results/coverage_summary.json
   ```
2. **Run Stage 2 (Primary & Replication Arms)**:
   ```bash
   .venv/bin/python scripts/02_score_models.py --variants results/variants.csv --out results/scores.csv --arms esm2-650m
   .venv/bin/python scripts/02_score_models.py --variants results/variants.csv --out results/scores.csv --arms esm2-3b --resume
   .venv-esmc/bin/python scripts/02_score_models.py --variants results/variants.csv --out results/scores.csv --arms esmc-600m --resume
   .venv-esmc/bin/python scripts/02_score_models.py --variants results/variants.csv --out results/scores.csv --arms esmc-6b --batch-size 128 --resume
   ```
3. **Run Stage 3 (Statistical Evaluation)**:
   ```bash
   .venv/bin/python scripts/03_run_test.py --scores results/scores.csv --n-perm 10000 --seed 0 --caliper 0.25 --out-dir results/
   ```

---

## 3. Key Invariants & Rules

- **Do NOT edit `PRE_REGISTRATION.md`**: Falsification rules ($\alpha = 0.01$, directional interaction) are locked.
- **Do NOT merge virtualenvs**: Keep `.venv` and `.venv-esmc` separate.
- **Always run pytest before committing**: `.venv/bin/pytest -v`.

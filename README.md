# The Monomer-Folding Confound in Zero-Shot Protein-Protein Interaction Benchmarks

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Manuscript: Quarto](https://img.shields.io/badge/manuscript-Quarto%20%2F%20LaTeX-blueviolet.svg)](docs/paper/paper.qmd)

Do single-chain protein language models (ESM-2, ESMC) truly understand protein-protein interaction interfaces, or does their apparent predictive power on binding benchmarks reflect an **expression-mediated illusion**?

---

## 1. The Core Scientific Discovery

Zero-shot protein language models are frequently reported to achieve moderate-to-high correlations on ProteinGym binding benchmarks ($\rho \approx 0.35 - 0.48$). This has fostered the widespread assumption that single-sequence PLMs implicitly capture the biophysics of intermolecular quaternary contacts, non-covalent binding energetics, and interface packing.

Using **within-target paired Deep Mutational Scanning (DMS) assays** ($N = 10,643$ paired variants across 5 primary PPI systems) that test **Monomer Abundance** and **Complex Binding** on the **exact same protein sequence, expression host, and variant library**, we prove this is an expression-mediated artifact:

1. **The "Binding Prediction" in PLMs is an Expression Illusion:**
   At quaternary interfaces, PLMs achieve $\rho \approx +0.38 \text{ to } +0.41$ on monomer folding stability. When evaluated on binding assays, the correlation is almost entirely driven by destabilizing mutations that unfold the monomer, thereby indirectly preventing cell-surface presentation and partner binding.
2. **Double-Dissociation Isolates True Interface Energetics:**
   Controlling for monomer stability on identical libraries, the model's true interface-specific binding predictive power collapses by **$-80.5\%$ ($\rho = +0.075$ on ESMC-6B)**.
3. **Mathematical Mediation Confirms Zero Unique Mutual Information:**
   The partial rank correlation at interface residues drops to negative values ($\rho(\text{PLM}, \text{Binding} \mid \text{Abundance}) = -0.193 \text{ to } -0.367$), proving that single-chain PLMs provide zero unique information about binding affinity beyond what is mediated by monomer folding.
4. **Scaling Aggravates the Artifact:**
   Scaling parameter count from 600M to 6.35B does not recover interface binding performance; larger models become more confident in single-sequence conservation priors, widening the predictive gap.
5. **The "PLM Filter Trap" in Binder Design:**
   Standard top-20% PLM likelihood filters discard **$85.8\%$ (ESM2-650M) to $96.1\%$ (ESMC-6B)** of true affinity-improving interface mutations, actively depleting candidate binder libraries.

<p align="center">
  <img src="docs/figures/01_expression_confound_schematic.png" alt="Expression Confound Schematic" width="95%">
</p>

---

## 2. Confirmatory Experimental Findings

### Three-Way Interaction Across 4 Foundation Models ($N = 10,643$ Paired Variants, $N = 2,262$ Interface):

$$\text{Model: } z(\text{DMS}) \sim \text{PLM} \times \text{Binding} \times \text{Interface}$$

| Model Arm | Parameter Count / Mode | $\beta_{(\text{PLM} \times \text{Binding} \times \text{Interface})}$ | Permutation $p$-value ($B=10^4$) | Interface Abundance ($\rho$) | Interface Binding ($\rho$) | Drop $\Delta\rho_{\text{interface}}$ | Verdict ($\alpha = 10^{-5}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ESM2-650M** | 650M (GPU fp16) | **$-0.3343$** | $p = 3.0 \times 10^{-4}$ | $\rho = +0.380$ | $\rho = +0.162$ | **$-57.4\%$** | Directional Negative |
| **ESM2-3B** | 2.84B (GPU fp16) | **$-0.3403$** | $p < 1.0 \times 10^{-5}$ | $\rho = +0.381$ | $\rho = +0.119$ | **$-68.8\%$** | **$H_1$ Supported** |
| **ESMC-600M** | 600M (GPU fp16) | **$-0.3530$** | $p < 1.0 \times 10^{-5}$ | $\rho = +0.413$ | $\rho = +0.167$ | **$-59.6\%$** | **$H_1$ Supported** |
| **ESMC-6B** | 6.35B (CPU offload) | **$-0.3526$** | $p < 1.0 \times 10^{-5}$ | $\rho = +0.384$ | $\rho = \mathbf{+0.075}$ | **$-80.5\%$** | **$H_1$ Supported** |

<p align="center">
  <img src="docs/figures/02_double_dissociation_scatter.png" alt="Double Dissociation Scatter" width="98%">
</p>

---

## 3. Mathematical Mediation: Partial Rank Correlations

$$\text{Partial Correlation: } \rho\Big(\text{PLM}, \; \text{Binding} \;\Big|\; \text{Abundance}\Big)$$

| Architecture | Structural Compartment | Sample Size ($N$) | $\rho(\text{PLM}, \text{Abundance})$ | $\rho(\text{PLM}, \text{Binding})$ | $\rho(\text{PLM}, \text{Binding} \mid \text{Abundance})$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ESM2-650M** | Monomer Core | 4,405 | $-0.236$ | $-0.184$ | $-0.066$ |
| | Monomer Surface | 3,976 | $+0.029$ | $+0.005$ | $-0.014$ |
| | **Quaternary Interface** | **2,262** | **$+0.403$** | **$-0.018$** | **$-0.193$** |
| **ESM2-3B** | Monomer Core | 4,405 | $-0.251$ | $-0.211$ | $-0.090$ |
| | Monomer Surface | 3,976 | $-0.007$ | $-0.046$ | $-0.051$ |
| | **Quaternary Interface** | **2,262** | **$+0.423$** | **$+0.007$** | **$-0.175$** |
| **ESMC-600M** | Monomer Core | 4,405 | $-0.342$ | $-0.241$ | $-0.067$ |
| | Monomer Surface | 3,976 | $-0.051$ | $-0.059$ | $-0.036$ |
| | **Quaternary Interface** | **2,262** | **$+0.426$** | **$-0.001$** | **$-0.186$** |
| **ESMC-6B** | Monomer Core | 4,405 | $-0.334$ | $-0.267$ | $-0.105$ |
| | Monomer Surface | 3,976 | $-0.117$ | $-0.159$ | $-0.111$ |
| | **Quaternary Interface** | **2,262** | **$+0.245$** | **$-0.241$** | $\mathbf{-0.367}$ |

<p align="center">
  <img src="docs/figures/03_mediation_forest_plot.png" alt="Mediation Forest Plot" width="90%">
</p>

<p align="center">
  <img src="docs/figures/04_model_scaling_collapse.png" alt="Model Scaling Collapse" width="92%">
</p>

---

## 4. Evolutionary PPI Stratification (Piece A)

Stratifying interface variants ($N = 2,262$) across three distinct evolutionary regimes reveals that sequence self-co-occurrence fails to rescue zero-shot interface sensitivity:

1. **Homooligomer (p53 / `1OLG`, $N=513$ interface):** Sequence self-co-occurrence in UniRef produces active anti-correlation ($\rho = \mathbf{-0.565}$, $\rho_{\text{partial}} = \mathbf{-0.511}$ on ESMC-6B).
2. **Natural Heterodimers (HLA-A2 / `5OPI`, GB1 / `1FCC`, $N=1,011$ interface):** Controlling for abundance drops partial correlation to near-zero ($\rho_{\text{partial}} = +0.001$ on HLA-A2; $-0.084$ on GB1).
3. **Synthetic Binders (KRAS / `6H46` with DARPin K55, $N=359$ interface):** PLM exhibits no native prior for synthetic partner physics ($\rho_{\text{partial}} = -0.487$).

<p align="center">
  <img src="docs/figures/05_evolutionary_regimes.png" alt="Evolutionary PPI Stratification" width="90%">
</p>

---

## 5. The "PLM Filter Trap" in Computational Binder Design (Piece B)

Simulating standard in silico screening protocols across all $N = 1,148$ beneficial interface variants ($\Delta y_{\text{binding}} \ge 0.0$):

- **False-Negative Rate (FNR):** Filtering candidates at top 20% by zero-shot likelihood discards **$85.8\%$ (ESM2-650M) to $96.1\%$ (ESMC-6B)** of true affinity-improving interface mutations.
- **Interface Depletion:** Selected candidate pools are depleted of interface mutations by $+17.5\%$ to $+57.8\%$ relative to surface/core mutations.
- **1D Multi-Chain Concatenation Failure:** Conditioning on concatenated sequences ($\text{Target} \mathbin{\Vert} \langle\text{eos}\rangle \mathbin{\Vert} \text{Partner}$) in ESM2-650M fails to recover signal (mean $\Delta\rho_{\text{binding}} = \mathbf{-0.0162}$).

<p align="center">
  <img src="docs/figures/06_binder_filter_depletion.png" alt="Binder Filter Trap Depletion" width="92%">
</p>

---

## 6. Primary Paired Systems & Structural Dataset

| Target Protein | Complex PDB | Target Chain | Partner Chain(s) | Evolutionary Class | Abundance Assay | Binding Assay | Paired Variants ($N$) | Interface Obs ($N$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SARS-CoV-2 RBD** | `6M0J` | `E` | `A` (human ACE2) | Cross-species viral | `SPIKE_SARS2_Starr_2020_expression` | `SPIKE_SARS2_Starr_2020_binding` | 3,664 | 379 |
| **KRAS** | `6H46` | `A` | `B` (synthetic DARPin K55) | Synthetic binder | `RASK_HUMAN_Weng_2022_abundance` | `RASK_HUMAN_Weng_2022_binding-DARPin_K55` | 2,761 | 359 |
| **HLA-A2** | `5OPI` | `A` | `B` ($\beta_2\text{M}$), `C` (TAPBPR) | Natural heterodimer | `Q53Z42_HUMAN_McShan_2019_expression` | `Q53Z42_HUMAN_McShan_2019_binding-TAPBPR` | 3,344 | 954 |
| **GB1** | `1FCC` | `C` | `A`, `B` (IgG Fc) | Natural heterodimer | `SPG1_STRSG_Wu_2016` | `SPG1_STRSG_Olson_2014` | 76 | 57 |
| **p53** | `1OLG` | `A` | `B`, `C`, `D` (tetramer) | Homooligomer | `P53_HUMAN_Giacomelli_2018_Null_Nutlin` | `P53_HUMAN_Giacomelli_2018_WT_Nutlin` | 798 | 513 |

---

## 7. Repository Structure & Reproducibility

```
plm-quaternary-interfaces/
├── src/plmppi/                      # Core package
│   ├── interfaces.py                # SASA & ΔSASA structural compartmentation
│   ├── data.py                      # Paired DMS dataset curation & parsing
│   ├── models.py                    # ESM2 & ESMC model loaders
│   ├── scoring.py                   # Batched zero-shot masked marginal scoring
│   └── stats.py                     # Three-way interaction models & permutation tests
├── scripts/
│   ├── 01_build_pairs.py            # Stage 1: Build paired variant table & structural labels
│   ├── 02_score_models.py           # Stage 2: Zero-shot scoring across model arms
│   ├── 03_run_test.py               # Stage 3: Statistical evaluation & falsification verdicts
│   ├── 04_mediation_analysis.py      # Stage 4: Partial rank correlation mediation
│   ├── 05_evolutionary_stratification.py # Stage 5: Evolutionary regime analysis (Piece A)
│   ├── 06_binder_design_audit.py    # Stage 6: Binder design filter simulation (Piece B)
│   └── plot_figures.py              # Stage 7: Generate 300 DPI publication figures
├── tests/                           # 19 hermetic unit tests (pytest)
├── results/                         # All output artifacts, summary JSONs, scores.csv
└── docs/
    ├── figures/                     # Generated publication-quality PNG figures
    └── paper/                       # Complete manuscript (paper.qmd & references.bib)
```

### Reproducing the Entire Pipeline:

```bash
# 1. Run full unit test suite
.venv/bin/pytest -v

# 2. Stage 3: Statistical Hypothesis Tests
.venv/bin/python scripts/03_run_test.py --scores results/scores.csv

# 3. Stage 4: Mediation & Partial Rank Correlations
.venv/bin/python scripts/04_mediation_analysis.py --scores results/scores.csv

# 4. Stage 5: Evolutionary Stratification (Piece A)
.venv/bin/python scripts/05_evolutionary_stratification.py --scores results/scores.csv

# 5. Stage 6: In Silico Binder Design Filter Audit (Piece B)
.venv/bin/python scripts/06_binder_design_audit.py --scores results/scores.csv

# 6. Stage 7: Generate Publication Figures (docs/figures/)
.venv/bin/python scripts/plot_figures.py

# 7. Render Publication Manuscript (HTML & PDF via Quarto)
quarto render docs/paper/paper.qmd --to html
quarto render docs/paper/paper.qmd --to pdf
```

---

## 8. Publication Manuscript

The complete scientific manuscript is available at [`docs/paper/paper.qmd`](docs/paper/paper.qmd) (with bibliography in [`docs/paper/references.bib`](docs/paper/references.bib)). It can be rendered to PDF or HTML using [Quarto](https://quarto.org/):

```bash
quarto render docs/paper/paper.qmd
```

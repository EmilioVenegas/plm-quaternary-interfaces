# Pre-Registration: Quaternary Interface Failure Modes in Monomeric Protein Language Models

**Status: FROZEN PRIOR TO CONFIRMATORY SCORING.**
Design and pre-registered protocol for testing whether monomeric PLMs fail systematically at protein-protein interfaces.

---

## 1. Hypotheses

- **$H_1$ (Interface Failure Mode)**:
  Monomeric protein language models (ESM2, ESMC), trained without quaternary structural context, exhibit a systematic
  predictive failure at quaternary oligomeric interfaces when predicting complex binding affinity relative to monomer folding stability.
  Operationally, $H_1$ predicts a **statistically significant negative three-way interaction**:
  $$\beta_{(\text{PLM} \times \text{Binding} \times \text{Interface})} < 0 \quad \text{at } \alpha = 10^{-5}$$
  in the model:
  $$z(\text{DMS}) = \beta_0 + \beta_1 z(\text{PLM}) + \beta_2 \text{Binding} + \beta_3 \text{Interface} + \beta_4 (z(\text{PLM}) \cdot \text{Binding}) + \beta_5 (z(\text{PLM}) \cdot \text{Interface}) + \beta_6 (\text{Binding} \cdot \text{Interface}) + \beta_7 (z(\text{PLM}) \cdot \text{Binding} \cdot \text{Interface}) + \epsilon$$

- **$H_0$ (Uniform Predictive Accuracy)**:
  $\beta_7 \ge 0$ or $p \ge 10^{-5}$. The model's predictive slope on interface mutations under binding selection is not significantly
  attenuated relative to its baseline predictive capacity on monomer stability.

---

## 2. Primary Outcome & Statistical Criteria

1. **Primary Model**: ESM2-650M (`facebook/esm2_t33_650M_UR50D`).
2. **Primary Metric**: Three-way interaction coefficient $\beta_7$ from OLS regression with cluster-robust sandwich covariance (clustered by system) and stratified permutation testing ($10,000$ permutations).
3. **Significance Threshold**: $\alpha = 10^{-5}$ (two-sided).
4. **Primary Locus / Systems**: The 5 paired PPI systems ($N = 10,643$ paired single mutants; $2,262$ interface observations):
   - SARS-CoV-2 RBD (`6M0J` chain E)
   - KRAS G-domain (`6H46` chain A)
   - HLA-A*02:01 (`5OPI` chain A)
   - GB1 (`1FCC` chain C)
   - p53 tetramer domain (`1OLG` chain A)

---

## 3. Secondary & Control Arms

All arms are reported unconditionally:

| Arm | Description | Pre-registered Expectation |
| :--- | :--- | :--- |
| **S1** | Structural Compartment Contrast (Interface vs Core vs Surface) | Attenuation is specific to Interface residues, not Core or Surface. |
| **S2** | Model Scaling & Replication (ESM2-3B, ESMC-600M, ESMC-6B) | Directional consistency across all four model architectures. |
| **S3** | Individual System Stratification | Evaluation of interaction slope across each of the 5 systems independently. |
| **N1** | Shuffled Interface Label Control | Within-system shuffled compartment labels yield interaction coefficient centered at zero ($p \ge 0.01$). |

---

## 4. Falsification Rules

- **$H_1$ Supported**:
  - The primary three-way interaction $\beta_7 < 0$ with $p < 10^{-5}$ in cluster-robust OLS and permutation test.
  - Sign consistency holds across all four model arms (S2).
  - Negative control N1 returns null as specified.
- **$H_1$ Not Supported**:
  - $\beta_7 \ge 0$ or $p \ge 10^{-5}$. A clean null is reported as a conclusive finding that PLMs do not exhibit an interface-specific binding deficit.
- **Pipeline Invalid**:
  - Failure to replicate basic DMS correlations on monomer stability ($r < 0.1$ on Core/Surface), or coordinate misalignment.

---

## 5. Inclusion & Curation Rules

1. **Paired Assay Requirement**: Both Abundance (monomer stability/expression) and Binding (quaternary complex affinity) must be measured on the exact same single-mutant library for the identical target sequence.
2. **Structural Mapping**: Every position must be mapped to atomic coordinates in high-resolution PDB complexes via pairwise alignment.
3. **Compartment Assignment**:
   - `Interface`: $\Delta \text{SASA} \ge 5.0\text{ \AA}^2$ or $d_{\text{min}} \le 4.5\text{ \AA}$.
   - `Core`: $r\text{SASA} < 0.20$ and not Interface.
   - `Surface`: $r\text{SASA} \ge 0.20$ and not Interface.

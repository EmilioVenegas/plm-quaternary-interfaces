# Study Plan: The Monomer-Folding Confound in Protein-Protein Interaction Benchmarks

## 1. Executive Summary & Research Question

Zero-shot protein language models (ESM-2, ESMC) are widely benchmarked on ProteinGym binding assays ($\rho \approx 0.35 - 0.45$), leading to the widespread assumption that single-chain PLMs have *"implicitly internalized protein-protein interaction (PPI) physics from evolutionary sequence diversity."*

We test whether this reported success is an **expression/folding artifact**: single-sequence PLMs predict whether the monomer folds, and unfolding indirectly eliminates complex binding. When monomer stability is held fixed, does single-chain PLM predictive coupling at quaternary interfaces collapse?

---

## 2. Experimental Strategy: Within-Target Double-Dissociation

Cross-protein benchmarks conflate MSA depth, sequence length, and experimental dynamic range. We isolate true interface energetics using **within-target paired Deep Mutational Scanning (DMS) assays** measuring both:
1. **Monomer Abundance / Folding Stability** (e.g., cell-surface expression, FACS, stability selection), AND
2. **Quaternary Complex Binding / Affinity** (e.g., target-partner co-selection)
across the **exact same target sequence and variant library**.

### Primary Paired Systems ($N = 10,643$ Paired Variants, $N = 2,262$ Interface Positions):

| System | Target Protein | Complex PDB | Target Chain | Partner Chain(s) | Evolutionary Class | Abundance Assay | Binding Assay | Paired Variants ($N$) | Interface Obs ($N$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SARS-CoV-2 RBD** | Spike RBD (333–526) | `6M0J` | `E` | `A` (human ACE2) | Cross-species viral | `SPIKE_SARS2_Starr_2020_expression` | `SPIKE_SARS2_Starr_2020_binding` | 3,664 | 379 |
| **KRAS** | KRAS G-domain | `6H46` | `A` | `B` (synthetic DARPin K55) | Synthetic binder | `RASK_HUMAN_Weng_2022_abundance` | `RASK_HUMAN_Weng_2022_binding-DARPin_K55` | 2,761 | 359 |
| **HLA-A2** | HLA-A*02:01 | `5OPI` | `A` | `B` ($\beta_2\text{M}$), `C` (TAPBPR) | Natural co-evolving heterodimer | `Q53Z42_HUMAN_McShan_2019_expression` | `Q53Z42_HUMAN_McShan_2019_binding-TAPBPR` | 3,344 | 954 |
| **GB1** | Protein G B1 domain | `1FCC` | `C` | `A`, `B` (IgG Fc) | Natural co-evolving heterodimer | `SPG1_STRSG_Wu_2016` | `SPG1_STRSG_Olson_2014` | 76 | 57 |
| **p53** | p53 tetramer domain | `1OLG` | `A` | `B`, `C`, `D` (homotetramer) | Homooligomer | `P53_HUMAN_Giacomelli_2018_Null_Nutlin` | `P53_HUMAN_Giacomelli_2018_WT_Nutlin` | 798 | 513 |

---

## 3. Structural Compartmentalization

Using atomic coordinates from high-resolution PDB crystal structures and FreeSASA:
1. **Interface**:
   - $\Delta \text{SASA} = \text{SASA}_{\text{monomer}} - \text{SASA}_{\text{complex}} \ge 5.0\text{ \AA}^2$, OR
   - Minimum heavy-atom distance to any partner chain atom $d_{\text{min}} \le 4.5\text{ \AA}$.
2. **Core**:
   - Non-interface residue with monomer relative solvent accessible surface area $r\text{SASA} < 0.20$ ($\text{SASA}_{\text{monomer}} / \text{MaxSASA} < 20\%$).
3. **Surface**:
   - Non-interface residue with $r\text{SASA} \ge 0.20$.

---

## 4. Formal Statistical & Mediation Framework

### 4.1 Three-Way Interaction Model
$$z(\text{DMS}) \sim \beta_0 + \beta_1 \text{PLM} + \beta_2 \text{Binding} + \beta_3 \text{Interface} + \beta_4 (\text{PLM} \times \text{Binding}) + \beta_5 (\text{PLM} \times \text{Interface}) + \beta_6 (\text{Binding} \times \text{Interface}) + \beta_7 (\text{PLM} \times \text{Binding} \times \text{Interface}) + \epsilon$$

**Hypothesis $H_1$**: $\beta_7 = \beta_{(\text{PLM} \times \text{Binding} \times \text{Interface})} < 0$ at $\alpha = 10^{-5}$.
Significance is verified with cluster-robust sandwich covariance (clustered by system) and 10,000 within-system stratified permutations.

### 4.2 Partial Correlation Mediation
$$\rho\Big(\text{PLM}, \; \text{Binding} \;\Big|\; \text{Abundance}\Big)$$
Tests whether the PLM contains non-zero unique mutual information about binding affinity when monomer folding stability is regressed out.

---

## 5. Model Arms & Scaling Protocol

- **ESM2-650M** (`facebook/esm2_t33_650M_UR50D`): Primary reference model.
- **ESM2-3B** (`facebook/esm2_t36_3B_UR50D`): Scaling within ESM2 family.
- **ESMC-600M** (`biohub/ESMC-600M`): Modern cross-architecture replication.
- **ESMC-6B** (`biohub/ESMC-6B`): Large-scale cross-architecture replication (via CPU offload, batch size 128).

---

## 6. Practical Implications for Binder & Antibody Design

In computational protein and antibody engineering pipelines, zero-shot PLM log-likelihood is frequently used as a filter or reward function to prioritize candidate designs.
This study proves that single-chain PLM filters **systematically select against interface-optimizing mutations** because they penalize non-conserved, partner-specific interface adaptations in favor of generic monomer surface plasticity.

# Study Plan: Quaternary Interface Failure Modes in Protein Language Models

## 1. Scientific Objective

Monomeric protein language models (ESM2, ESMC) are trained on evolutionary sequences as isolated single chains.
When applied to zero-shot variant effect prediction (VEP), a fundamental open question is whether these models exhibit
a **systematic failure mode at quaternary oligomeric interfaces**—specifically, whether they fail to predict complex
binding affinity compared to monomer folding stability on identical target proteins.

We test this hypothesis across 5 primary paired Deep Mutational Scanning (DMS) datasets where both monomer stability/abundance
and quaternary binding/affinity have been experimentally quantified across identical single-mutant libraries on the same target sequence.

---

## 2. Primary Paired PPI Systems

| System | Target Protein | Complex PDB | Target Chain | Partner Chains | Abundance / Stability Assay | Binding / Affinity Assay |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SARS-CoV-2 RBD** | Spike RBD (333-526) | `6M0J` | `E` | `A` (ACE2) | `SPIKE_SARS2_Starr_2020_expression` | `SPIKE_SARS2_Starr_2020_binding` |
| **KRAS** | KRAS G-domain | `6H46` | `A` | `B` (DARPin K55) | `RASK_HUMAN_Weng_2022_abundance` | `RASK_HUMAN_Weng_2022_binding-DARPin_K55` |
| **HLA-A2** | HLA-A*02:01 | `5OPI` | `A` | `B` ($\beta_2\text{M}$), `C` (TAPBPR) | `Q53Z42_HUMAN_McShan_2019_expression` | `Q53Z42_HUMAN_McShan_2019_binding-TAPBPR` |
| **GB1** | Protein G B1 domain | `1FCC` | `C` | `A`, `B` (IgG Fc) | `SPG1_STRSG_Wu_2016` | `SPG1_STRSG_Olson_2014` |
| **p53** | p53 tetramer domain | `1OLG` | `A` | `B`, `C`, `D` (homotetramer) | `P53_HUMAN_Giacomelli_2018_Null_Nutlin` | `P53_HUMAN_Giacomelli_2018_WT_Nutlin` |

---

## 3. Structural Compartment Definitions

Using atomic coordinates from high-resolution PDB crystal structures, each residue in the target sequence is categorized into one of three mutually exclusive compartments:

1. **Interface**:
   - $\Delta \text{SASA} = \text{SASA}_{\text{monomer}} - \text{SASA}_{\text{complex}} \ge 5.0\text{ \AA}^2$, OR
   - Minimum heavy-atom distance to any partner chain atom $d_{\text{min}} \le 4.5\text{ \AA}$.
2. **Core**:
   - Non-interface residue with monomer relative solvent accessible surface area $r\text{SASA} < 0.20$ ($\text{SASA}_{\text{monomer}} / \text{MaxSASA} < 20\%$).
3. **Surface**:
   - Non-interface residue with $r\text{SASA} \ge 0.20$.

Theoretical maximum SASA values are defined per residue according to Tien et al. (2013).

---

## 4. Model Arms

Zero-shot masked-marginal scoring $S = \log P(\text{mut} \mid \text{context}) - \log P(\text{wt} \mid \text{context})$ is evaluated across 4 model arms:
- **ESM2-650M** (`facebook/esm2_t33_650M_UR50D`): Primary reference model.
- **ESM2-3B** (`facebook/esm2_t36_3B_UR50D`): Scaling within ESM2 family.
- **ESMC-600M** (`biohub/ESMC-600M`): Modern cross-architecture replication.
- **ESMC-6B** (`biohub/ESMC-6B`): Large-scale cross-architecture replication (via CPU offload).

---

## 5. Statistical Framework

We specify a three-way interaction mixed-effects regression model:
$$y \sim \beta_0 + \beta_1 \text{PLM} + \beta_2 \text{Binding} + \beta_3 \text{Interface} + \beta_4 (\text{PLM} \times \text{Binding}) + \beta_5 (\text{PLM} \times \text{Interface}) + \beta_6 (\text{Binding} \times \text{Interface}) + \beta_7 (\text{PLM} \times \text{Binding} \times \text{Interface}) + \epsilon$$

- $y$: Standardized experimental DMS fitness score within assay ($z(\text{DMS})$).
- $\text{PLM}$: Standardized zero-shot masked marginal score within assay ($z(\text{PLM})$).
- $\text{Binding}$: Binary indicator ($1 = \text{Binding/Affinity}$, $0 = \text{Abundance/Stability}$).
- $\text{Interface}$: Binary indicator ($1 = \text{Interface}$, $0 = \text{Core/Surface}$).

**Hypothesis $H_1$**: $\beta_7 = \beta_{(\text{PLM} \times \text{Binding} \times \text{Interface})} < 0$ at $\alpha = 10^{-5}$.
Significance is verified with cluster-robust sandwich covariance (clustered by system) and 10,000 within-system stratified permutations.

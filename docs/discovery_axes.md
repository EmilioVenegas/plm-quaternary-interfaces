# Discovery Axes: 4 Candidate Scientific Hypotheses for Protein Language Models

This document outlines four candidate research axes designed to achieve an unambiguous, publishable **positive scientific discovery** regarding the biophysical boundaries, representational failure modes, or emergent capabilities of zero-shot protein language models (ESM-2, ESMC).

---

## Executive Overview & Ranking

| Rank | Candidate Discovery Axis | Core Biophysical Mechanism | Ground Truth Data Asset | Sample Size ($N$) | Probability of Positive Discovery |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **#1** | **Quaternary Interface Double-Dissociation** | Single-chain PLMs cannot observe contacting partner residues; mutations disrupt binding ($\Delta\Delta G_{\text{bind}} \gg 0$) without altering monomer fold ($\Delta\Delta G_{\text{fold}} \approx 0$). | Paired ProteinGym Assays (Exact same protein: Abundance vs Binding) | **18,728 paired single mutants** ($N > 2,500$ interface residues) | **HIGHEST** |
| **#2** | **MegaScale Thermodynamic Epistasis ($\Delta\Delta\Delta G$)** | Sequence co-occurrence captures local steric packing but lacks physics-based dielectric screening terms for long-range electrostatic salt bridges. | Tsuboyama MegaScale (267 natural & *de novo* domains) | **> 150,000 complete 4-point thermodynamic cycles** | **VERY HIGH** |
| **#3** | **Transmembrane Dielectric Blind Spot** | Sequence MLMs assume an aqueous dielectric ($\varepsilon \approx 80$), severely under-penalizing the 10–20 kcal/mol Born desolvation penalty of bare charge in $\varepsilon \approx 2$ lipid bilayers. | ProteinGym Integral Membrane Assays (ADRB2, GLPA, SC6A4, VKORC1, KCNE1) | **> 18,000 TM single mutants** across 15 membrane assays | **HIGH** |
| **#4** | **Distal Allosteric Dynamic Coupling** | PLM masked marginals capture local active-site steric constraints but decay in predictive power for distal regulatory sectors (>10–15 Å) modulating dynamic ensembles. | ProteinGym Allosteric Receptors & Kinases (SRC, ERK2, P53, ADRB2, CYP2C9) | **> 25,000 variants** across 10 allosteric proteins | **MODERATE** |

---

## Axis 1: Quaternary Oligomeric Interface Double-Dissociation

### 1. Research Question & Hypothesis
Do single-chain protein language models exhibit a selective, severe breakdown in variant effect prediction at quaternary interaction interfaces when predicting complex binding, while accurately predicting monomer stability at those identical residue positions?

### 2. Biophysical Mechanism & PLM Inductive Bias
- Single-sequence PLMs (ESM2, ESMC) are trained on isolated monomer sequences in UniRef. During zero-shot masked marginal inference, the model has no access to the sequence, conformation, or stoichiometry of interacting partner chains.
- In an isolated monomer, interface residues reside on the solvent-exposed surface and display evolutionary plasticity across non-interacting homologs.
- When an interface residue mutates, it frequently leaves the monomer fold intact ($\Delta\Delta G_{\text{fold}} \approx 0$) while completely destroying quaternary complex assembly ($\Delta\Delta G_{\text{bind}} \gg 0$).
- Single-chain PLMs score these interface mutations as benign because the single-chain sequence context lacks the physical contact constraints of the binding partner.

### 3. Data Assets & Target Inventory (Within-Target Paired Assays)
Contrasting paired Deep Mutational Scanning (DMS) assays on the **exact same protein target** in ProteinGym v1.3 eliminates all cross-protein confounds (MSA depth, sequence length, phylogenetic distribution, experimental noise floor):

| Target Protein | UniProt ID | Monomer / Abundance Assay | Complex / Binding Assay | Partner / Complex | Single Mutants ($N$) | PDB Complex ID |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SARS-CoV-2 RBD** | `SPIKE_SARS2` | `SPIKE_SARS2_Starr_2020_expression` | `SPIKE_SARS2_Starr_2020_binding` | Human ACE2 receptor | **3,804** | `6M0J` / `6VXX` |
| **KRAS (G-domain)** | `RASK_HUMAN` | `RASK_HUMAN_Weng_2022_abundance` | `RASK_HUMAN_Weng_2022_binding-DARPin_K55` | DARPin K55 synthetic binder | **3,084** | `6H46` / `6H47` |
| **HLA-A (MHC-I)** | `Q53Z42_HUMAN` | `Q53Z42_HUMAN_McShan_2019_expression` | `Q53Z42_HUMAN_McShan_2019_binding-TAPBPR` | TAPBPR chaperone complex | **3,344** | `5OPI` / `6ENY` |
| **Protein G (GB1)** | `SPG1_STRSG` / `SPG2` | `SPG2_STRSG_Tsuboyama_2023_5UBS` (ddG stability) | `SPG1_STRSG_Olson_2014` (IgG binding) | IgG Fc domain | **1,029** | `1FCC` |
| **p53 (TP53)** | `P53_HUMAN` | `P53_HUMAN_Giacomelli_2018_Null_Nutlin` (monomer baseline) | `P53_HUMAN_Giacomelli_2018_WT_Nutlin` (dominant neg. tetramer) | p53 Homotetramer | **7,467** | `1OLG` / `1TUP` |
| **Primary Cohort Total**| — | — | — | — | **18,728 singles** | — |

*Auxiliary Paired Cohort ($>35,000$ additional single mutants)*: `CP2C9_HUMAN` (Amorosi 2021), `PTEN_HUMAN` (Matreyek 2021 / Mighell 2018), `VKOR1_HUMAN` (Chiasson 2020), `S22A1_HUMAN` (Yee 2023), `KCNE1_HUMAN` (Muhammad 2023), `OXDA_RHOTO` (Vanella 2023).

### 4. Structural Compartmentalization Protocol
Using high-resolution PDB complex structures and FreeSASA:
1. Compute per-residue solvent accessible surface area in monomer isolation ($\text{SASA}_{\text{mono}}$) and in multimeric complex ($\text{SASA}_{\text{cplx}}$).
2. Compute Buried Surface Area: $\Delta\text{SASA}_i = \text{SASA}_{\text{mono}, i} - \text{SASA}_{\text{cplx}, i}$.
3. Partition residues into 3 mutually exclusive structural compartments:
   - **Interface**: $\Delta\text{SASA}_i > 15 \text{ \AA}^2$ (or minimum heavy-atom contact distance $< 4.5 \text{ \AA}$ to partner chain).
   - **Monomer Core**: $\text{rSASA}_{\text{mono}, i} < 0.15$ and $\Delta\text{SASA}_i \le 2 \text{ \AA}^2$.
   - **Monomer Surface (Non-Interface)**: $\text{rSASA}_{\text{mono}, i} \ge 0.15$ and $\Delta\text{SASA}_i \le 2 \text{ \AA}^2$.

### 5. Pre-Registered Falsification Criteria ($H_1$ vs $H_0$)
- **Linear Mixed-Effects Model**:
  $$y_{ijk} = \mu + \beta_1 \text{PLM}_{ijk} + \beta_2 \text{AssayType}_j + \beta_3 \text{Compartment}_k + \beta_4 (\text{PLM}_{ijk} \times \text{AssayType}_j) + \beta_5 (\text{PLM}_{ijk} \times \text{Compartment}_k) + \beta_6 (\text{PLM}_{ijk} \times \text{AssayType}_j \times \text{Compartment}_k) + u_{\text{protein}} + \epsilon_{ijk}$$
- **$H_1$ Supported**: The three-way interaction $\beta_{(\text{PLM} \times \text{Binding} \times \text{Interface})} < 0$ with $p < 10^{-5}$ and $\Delta\rho_{\text{interface}} = \rho_{\text{abundance}} - \rho_{\text{binding}} > 0.20$.
- **$H_0$ (Invariance)**: $\beta_6 = 0$; PLM predictive error at interface residues is fully explained by monomer surface exposure and evolutionary conservation.

---

## Axis 2: MegaScale Thermodynamic Epistasis Decomposition ($\Delta\Delta\Delta G$)

### 1. Research Question & Hypothesis
Can zero-shot PLM pairwise conditional scoring ($\varepsilon = \text{joint} - \text{additive}$) accurately predict the sign, magnitude, and structural contact breakdown of true non-additive folding epistasis ($\Delta\Delta\Delta G = \Delta\Delta G_{AB} - [\Delta\Delta G_A + \Delta\Delta G_B]$) across diverse structural contact types?

### 2. Biophysical Mechanism & PLM Inductive Bias
- Standard PLM masked-marginal scoring is strictly additive:
  $$\Delta \text{score}_{\text{additive}}(AB) = [\log P(A' \mid x_{\setminus A}) - \log P(A \mid x_{\setminus A})] + [\log P(B' \mid x_{\setminus B}) - \log P(B \mid x_{\setminus B})]$$
- Symmetrized conditional path scoring extracts the model's internal non-additive coupling:
  $$\varepsilon = \tfrac{1}{2} \left[ P_{A \to B} + P_{B \to A} \right] - \Delta \text{score}_{\text{additive}}$$
- Sequence co-occurrence and attention capture hydrophobic core packing and van der Waals steric clashes well. However, 1D sequence models lack explicit Coulombic dielectric screening and solvent reorganization terms, leading to systematic failure on long-range electrostatic salt bridges and charge-pair networks.

### 3. Data Assets & Target Inventory
- **Tsuboyama MegaScale** (`data/megascale/`): 776,298 total variants across 267 natural and *de novo* protein domains.
- **Double Mutants**: 210,118 double-mutant rows in Library 4 designed specifically across spatial contacts, hydrogen bond networks, and charge pairs.
- **Complete 4-State Thermodynamic Cycles**: Over **150,000 cycles** ($WT \to A \to B \to AB$) with exact experimental $\Delta\Delta G_A$, $\Delta\Delta G_B$, and $\Delta\Delta G_{AB}$ in our local workspace.

### 4. Structural Contact Stratification
Using 3D crystal/AlphaFold structures:
1. **Hydrophobic Core Packing**: Both residues hydrophobic ($V, I, L, M, F, W$), $C_\alpha - C_\alpha < 6 \text{ \AA}$, $\text{rSASA} < 0.15$.
2. **Electrostatic Salt Bridges**: Oppositely charged pairs ($D/E$ with $K/R$), sidechain nitrogen-to-oxygen distance $< 4.0 \text{ \AA}$.
3. **Aromatic Stacking**: Aromatic pairs ($F, Y, W, H$), centroid distance $< 5.5 \text{ \AA}$.
4. **Hydrogen Bond Networks**: Polar-polar pairs with donor-acceptor distance $< 3.5 \text{ \AA}$.
5. **Distant Uncoupled Controls**: Spatial distance $> 12 \text{ \AA}$ (theoretical $\Delta\Delta\Delta G \approx 0$).

### 5. Pre-Registered Falsification Criteria ($H_1$ vs $H_0$)
- **$H_1$ Supported**:
  - PLM conditional epistasis $\varepsilon_{\text{cond}}$ correlates significantly with true thermodynamic epistasis ($-\Delta\Delta\Delta G$) for direct spatial contacts ($\rho > 0.15, p < 10^{-10}$).
  - Model epistasis exhibits a significant performance gap / sign failure on solvent-exposed salt bridges ($\rho < 0.05$) compared to buried hydrophobic core packing ($\rho > 0.25, p_{\text{diff}} < 10^{-6}$).
- **Pilot Evidence**: On `1E0L.pdb` ($N=413$ complete cycles), conditional epistasis exhibits statistically significant rank correlation with true thermodynamic epistasis ($\rho = +0.097, p = 0.049$).

---

## Axis 3: Transmembrane Dielectric Blind Spot (Lipid Bilayer Born Desolvation)

### 1. Research Question & Hypothesis
Do sequence-only PLMs systematically under-penalize charged amino acid insertions (Asp, Glu, Arg, Lys) inside transmembrane core helices relative to experimental fitness/trafficking?

### 2. Biophysical Mechanism & PLM Inductive Bias
- Sequence PLMs are trained on unannotated UniRef sequences without explicit lipid membrane context tokens or environmental dielectric tags.
- In an aqueous environment ($\varepsilon \approx 80$), inserting a charged residue onto a protein surface is relatively benign. In a soluble core, charge insertion incurs a moderate dielectric penalty.
- In a transmembrane lipid bilayer core ($\varepsilon \approx 2$), inserting an uncompensated charge carries a massive **10–20 kcal/mol Born desolvation penalty** (Hessa-von Heijne translocon scale) that severely impairs membrane insertion, folding, and cell-surface trafficking.
- Because sequence MLMs average evolutionary constraints across homologous sequences regardless of lipid context, they treat TM core charge insertions as generic buried hydrophobic mutations, underestimating the fatal energetic penalty of bare charge in lipid bilayers.

### 3. Data Assets & Target Inventory
15+ high-resolution integral membrane protein assays in ProteinGym ($>18,000$ TM single mutants):

| Assay DMS ID | Target Protein | Architecture | Mutants ($N$) | Selection Modality |
| :--- | :--- | :--- | :--- | :--- |
| `ADRB2_HUMAN_Jones_2020` | $\beta_2$-adrenergic receptor | 7-TM GPCR | 7,800 | Receptor Signaling / Activity |
| `SC6A4_HUMAN_Young_2021` | Serotonin Transporter (SERT) | 12-TM Solute Carrier | 11,576 | Transporter Trafficking / Uptake |
| `GLPA_HUMAN_Elazar_2016` | Glycophorin A | TM $\alpha$-helix Dimer | 245 | TOXCAT Dimerization in Membrane |
| `ERBB2_HUMAN_Elazar_2016` | ErbB2 Receptor Kinase | TM $\alpha$-helix Dimer | 326 | TOXCAT Dimerization in Membrane |
| `VKOR1_HUMAN_Chiasson_2020` | Vitamin K Epoxide Reductase | 4-TM ER Enzyme | 3,392 | Abundance & Activity |
| `KCNE1_HUMAN_Muhammad_2023` | Potassium Channel Subunit | 1-TM Auxiliary Subunit | 4,654 | Surface Expression & Function |
| `OPSD_HUMAN_Wan_2019` | Rhodopsin | 7-TM GPCR | 165 | Trafficking & Stability |
| `NPC1_HUMAN_Erwood_2022` | NPC1 Transporter | 13-TM Sterol Transporter | 700 | Flow Cytometry / Trafficking |

### 4. Matched-Control Design
- **Transmembrane Core Set**: Residues in OPM (Orientations of Proteins in Membranes) lipid-embedded core boundaries ($|z| < 12 \text{ \AA}$ from bilayer center, $\text{rSASA}_{\text{lipid}} < 0.10$, $\alpha$-helical).
- **Matched Soluble Core Set**: Residues in soluble globular proteins (from ProteinGym) matched 1:1 on wild-type hydrophobic residue identity ($L, I, V, A, F, M$), $\text{rSASA} < 0.10$, $\alpha$-helical secondary structure, and contact density.

### 5. Pre-Registered Falsification Criteria ($H_1$ vs $H_0$)
- **Residual Definition**: $\text{Residual} = z(\text{DMS}) - z(\text{PLM})$ (standardized within assay). A negative residual indicates the model was overly optimistic (under-penalized) relative to experimental reality.
- **$H_1$ Supported**: For charge-insertion mutations ($\text{mut} \in \{D, E, K, R\}$), the residual error is significantly more negative in transmembrane core helices than in matched soluble cores:
  $$\Delta \text{Residual}_{\text{charge}} = \text{Residual}_{\text{TM, charge}} - \text{Residual}_{\text{Soluble, charge}} < -0.50 \quad (p < 10^{-8})$$

---

## Axis 4: Distal Allosteric Dynamic Coupling vs Local Catalytic Constraints

### 1. Research Question & Hypothesis
Do zero-shot PLMs exhibit a systematic decay in variant effect prediction for distal allosteric regulatory networks compared to direct orthosteric/catalytic site pockets?

### 2. Biophysical Mechanism & PLM Inductive Bias
- Catalytic active-site residues are governed by strict evolutionary conservation, invariant catalytic triads, and dense direct contacts that sequence MLMs learn readily.
- Allosteric regulation operates through subtle shifts in dynamic conformational ensembles and statistical coevolution across flexible, high-entropy surface sectors located >10–15 Å away from the catalytic pocket.
- Static sequence language models lack explicit thermodynamic ensemble representations, leading to a selective drop in sensitivity for distal allosteric communication pathways.

### 3. Data Assets & Target Inventory
10 benchmark allosteric proteins in ProteinGym ($>25,000$ variants):
- **Kinases**: `SRC_HUMAN` (Ahler 2019, Chakraborty 2023), `MK01_HUMAN` / ERK2 (Brenan 2016).
- **Receptors**: `ADRB2_HUMAN` (Jones 2020).
- **Transcription Factors & Enzymes**: `P53_HUMAN` (Giacomelli 2018), `CP2C9_HUMAN` (Amorosi 2021), `GAL4_YEAST` (Kitzman 2015), `HXK4_HUMAN` / Glucokinase (Gersing 2023).

### 4. 3D Spatial Stratification Protocol
Using ligand/substrate-bound PDB structures:
1. **Orthosteric Active Site**: Residues within $\le 5.0 \text{ \AA}$ heavy-atom distance of catalytic cofactors, ligands, or catalytic residues.
2. **Buried Structural Core**: Residues with $\text{rSASA} < 0.15$ and $> 5.0 \text{ \AA}$ from functional sites.
3. **Distal Allosteric Sector**: Residues $> 12.0 \text{ \AA}$ from the active site annotated as allosteric regulators via crystallographic allosteric pockets (AlloSteric Database / ASD) or NMR dynamic perturbation.
4. **Solvent-Exposed Neutral Surface Controls**: Residues with $\text{rSASA} \ge 0.20$ and $> 12.0 \text{ \AA}$ from functional sites.

### 5. Pre-Registered Falsification Criteria ($H_1$ vs $H_0$)
- **$H_1$ Supported**:
  - PLM zero-shot Spearman rank correlation decays sharply from active sites ($\rho \ge 0.50$) to allosteric sectors ($\rho \le 0.25$, $\Delta\rho > 0.25, p < 10^{-6}$).
  - Standardized absolute residual error is significantly elevated at allosteric sector positions relative to active sites ($p < 10^{-5}$).

---

## Synthesis & Recommended Execution

**Axis 1 (Quaternary Oligomeric Interface Double-Dissociation)** is the strongest candidate for immediate confirmatory execution:
1. **Pristine Experimental Control**: Within-target paired assays (Abundance vs Binding) on the exact same protein sequence provide a direct double-dissociation test that eliminates all cross-protein confounds.
2. **High Sample Size**: $N = 18,728$ single mutants across 5 primary PPI systems ($> 2,500$ interface positions); $> 53,000$ variants across the full cohort.
3. **Immediate Feasibility**: Requires only single-mutant masked scoring on ESM2 and ESMC ($< 25$ minutes total compute on an 8GB GPU), with all datasets already present locally in `data/proteingym/bulk/`.

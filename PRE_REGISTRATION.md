# Pre-Registration: Quaternary Interface Blindspots in Single-Chain PLMs

**Frozen for Confirmatory Evaluation.** Study design and rationale: [`PLAN.md`](PLAN.md).

---

## 1. Hypotheses

### Primary Hypothesis ($H_1$)
In a standardized three-way interaction model ($z(\text{DMS}) \sim \text{PLM} \times \text{Binding} \times \text{Interface}$), the interaction coefficient $\beta_7$ is negative:
$$\beta_{(\text{PLM} \times \text{Binding} \times \text{Interface})} < 0 \quad \text{at } \alpha = 10^{-5}$$
indicating a selective degradation in predictive coupling specifically when single-chain PLMs predict complex binding at quaternary interfaces.

### Mediation Criterion ($H_{\text{mediation}}$)
Controlling for monomer abundance, the partial rank correlation at interface positions approaches zero or becomes negative:
$$\rho\Big(\text{PLM}, \; \text{Binding} \;\Big|\; \text{Abundance}\Big) \le 0$$
proving that the PLM's baseline correlation with binding assays is entirely mediated by monomer folding/expression rather than intermolecular interface physics.

### Scaling Aggravation Hypothesis ($H_{\text{scale}}$)
Scaling model parameter count from 600M to 6.35B does not recover interface binding performance ($\Delta\rho_{\text{interface}} = \rho_{\text{abundance}} - \rho_{\text{binding}}$ increases or remains severely degraded).

---

## 2. Inclusion & Exclusion Criteria

1. **Paired Assay Invariance**: Targets must possess experimentally quantified single-mutant libraries for both monomer abundance/stability and quaternary complex binding on the identical sequence construct.
2. **Structural Complex Ground Truth**: Every target must have an experimental crystal or cryo-EM complex structure in the PDB with resolution $\le 3.0 \text{ \AA}$.
3. **Compartment Partitioning**: Residues are strictly partitioned into Interface ($\Delta\text{SASA} \ge 5.0 \text{ \AA}^2$ or $d_{\text{min}} \le 4.5 \text{ \AA}$), Core ($r\text{SASA} < 0.20$), or Surface ($r\text{SASA} \ge 0.20$).

---

## 3. Falsification Rules

- **$H_1$ Supported**: $\beta_{(\text{PLM} \times \text{Binding} \times \text{Interface})} < 0$ with cluster-robust $p < 10^{-5}$, and stratified permutation $p < 10^{-5}$ across primary model arms.
- **$H_1$ Falsified**: $\beta_{(\text{PLM} \times \text{Binding} \times \text{Interface})} \ge 0$ or $p \ge 10^{-5}$.
- **Mediation Supported**: Partial correlation $\rho(\text{PLM}, \text{Binding} \mid \text{Abundance}) \le 0$ at interface residues.

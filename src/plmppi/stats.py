"""Statistical analysis and three-way interaction hypothesis testing for the PLM PPI study."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 1e-5  # Pre-registered significance threshold


def standardize_series(s: pd.Series) -> pd.Series:
    """Standardizes a series to mean 0, unit variance within non-null values."""
    valid = s.dropna()
    if len(valid) <= 1 or valid.std() == 0:
        return s - valid.mean()
    return (s - valid.mean()) / valid.std()


def prepare_analysis_frame(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Prepares the stacked analysis frame with standardized scores and interaction features.

    Each row in df represents one variant (position, wt, mut) with paired
    dms_score_abundance and dms_score_binding and PLM scores.
    We stack this into two observations per variant: one for Abundance, one for Binding.
    """
    zs_col = f"zeroshot_{arm}"
    if zs_col not in df.columns:
        raise KeyError(f"Score column {zs_col} not found in dataframe")

    rows = []
    for _, r in df.iterrows():
        sys_id = str(r["system"])
        pos = int(r["position"])
        wt = str(r["wt"])
        mut = str(r["mut"])
        comp = str(r["compartment"])
        is_int = 1.0 if comp == "Interface" else 0.0
        zs = float(r[zs_col])

        # Abundance observation
        dms_ab = float(r["dms_score_abundance"])
        rows.append(
            {
                "system": sys_id,
                "position": pos,
                "wt": wt,
                "mut": mut,
                "compartment": comp,
                "is_interface": is_int,
                "assay_type": "Abundance",
                "is_binding": 0.0,
                "dms_raw": dms_ab,
                "plm_raw": zs,
            }
        )

        # Binding observation
        dms_bi = float(r["dms_score_binding"])
        rows.append(
            {
                "system": sys_id,
                "position": pos,
                "wt": wt,
                "mut": mut,
                "compartment": comp,
                "is_interface": is_int,
                "assay_type": "Binding",
                "is_binding": 1.0,
                "dms_raw": dms_bi,
                "plm_raw": zs,
            }
        )

    f = pd.DataFrame(rows)

    # Standardize DMS and PLM scores within each (system, assay_type) assay
    f["dms_z"] = f.groupby(["system", "assay_type"])["dms_raw"].transform(standardize_series)
    f["plm_z"] = f.groupby(["system", "assay_type"])["plm_raw"].transform(standardize_series)

    return f


def fit_clustered_ols(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Fits OLS with cluster-robust sandwich covariance estimator."""
    N, K = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta

    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    meat = np.zeros((K, K))
    for g in unique_clusters:
        idx = clusters == g
        X_g = X[idx]
        u_g = residuals[idx]
        score_g = X_g.T @ u_g
        meat += np.outer(score_g, score_g)

    df_c = (G / (G - 1)) * ((N - 1) / (N - K)) if G > 1 and (N - K) > 0 else 1.0
    vcov = df_c * (XtX_inv @ meat @ XtX_inv)

    se = np.sqrt(np.maximum(0.0, np.diag(vcov)))
    t_stat = np.where(se > 0, beta / se, np.nan)
    df_t = max(1, G - 1)
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df_t))

    names = feature_names if feature_names is not None else [f"x{i}" for i in range(K)]
    params: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        params[name] = {
            "coef": float(beta[i]),
            "se": float(se[i]),
            "t_stat": float(t_stat[i]),
            "p_val": float(p_val[i]),
        }

    return {
        "n_obs": int(N),
        "n_clusters": int(G),
        "params": params,
        "beta": beta.tolist(),
        "residuals_mean": float(np.mean(residuals)),
        "residuals_std": float(np.std(residuals)),
        "r_squared": float(1.0 - np.var(residuals) / np.var(y)),
    }


def run_three_way_interaction_test(
    analysis_df: pd.DataFrame,
    n_perm: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs the pre-registered three-way interaction test:

    y ~ PLM * Binding * Interface
    H1: beta_(PLM:Binding:Interface) < 0 at alpha = 1e-5.
    """
    df = analysis_df.dropna(subset=["dms_z", "plm_z", "is_binding", "is_interface"]).copy()
    N = len(df)

    plm = df["plm_z"].to_numpy()
    bind = df["is_binding"].to_numpy()
    inter = df["is_interface"].to_numpy()
    y = df["dms_z"].to_numpy()
    clusters = df["system"].to_numpy()

    feature_names = [
        "Intercept",
        "PLM",
        "Binding",
        "Interface",
        "PLM:Binding",
        "PLM:Interface",
        "Binding:Interface",
        "PLM:Binding:Interface",
    ]

    X = np.column_stack(
        [
            np.ones(N),
            plm,
            bind,
            inter,
            plm * bind,
            plm * inter,
            bind * inter,
            plm * bind * inter,
        ]
    )

    ols_result = fit_clustered_ols(X, y, clusters, feature_names)
    primary_coef = ols_result["params"]["PLM:Binding:Interface"]["coef"]
    primary_p = ols_result["params"]["PLM:Binding:Interface"]["p_val"]

    # Stratified permutation test (permute interface status within system)
    rng = np.random.default_rng(seed)
    perm_betas = np.zeros(n_perm)

    XtX_inv = np.linalg.pinv(X.T @ X)  # baseline matrix for projection
    for b in range(n_perm):
        # Permute interface label within each system
        perm_inter = inter.copy()
        for sys_id in np.unique(clusters):
            idx = np.where(clusters == sys_id)[0]
            perm_inter[idx] = rng.permutation(perm_inter[idx])

        X_perm = np.column_stack(
            [
                np.ones(N),
                plm,
                bind,
                perm_inter,
                plm * bind,
                plm * perm_inter,
                bind * perm_inter,
                plm * bind * perm_inter,
            ]
        )
        beta_perm = np.linalg.pinv(X_perm.T @ X_perm) @ (X_perm.T @ y)
        perm_betas[b] = beta_perm[7]

    # Two-sided permutation p-value
    perm_p = float(np.mean(np.abs(perm_betas) >= np.abs(primary_coef)))

    # Subgroup correlations (Spearman rho and Pearson r)
    subgroups = {}
    for comp in ["Interface", "Core", "Surface"]:
        for atype in ["Abundance", "Binding"]:
            sub = df[(df["compartment"] == comp) & (df["assay_type"] == atype)]
            if len(sub) > 2:
                sp_r, sp_p = stats.spearmanr(sub["plm_z"], sub["dms_z"])
                pe_r, pe_p = stats.pearsonr(sub["plm_z"], sub["dms_z"])
                subgroups[f"{comp}_{atype}"] = {
                    "n": int(len(sub)),
                    "spearman_rho": float(sp_r),
                    "spearman_p": float(sp_p),
                    "pearson_r": float(pe_r),
                    "pearson_p": float(pe_p),
                }

    # Pre-registered verdict determination
    # H1 supported iff beta < 0 and (primary_p < ALPHA or perm_p < ALPHA)
    is_negative = primary_coef < 0
    is_significant = (primary_p < ALPHA) or (perm_p < ALPHA)

    if is_negative and is_significant:
        verdict = "H1 supported: PLMs fail systematically at quaternary interfaces"
    elif not is_significant:
        verdict = "H1 not supported: No significant three-way interaction"
    else:
        verdict = "Unanticipated finding: Positive interaction"

    return {
        "primary_term": "PLM:Binding:Interface",
        "beta_three_way": float(primary_coef),
        "p_clustered": float(primary_p),
        "p_permutation": float(perm_p),
        "alpha": ALPHA,
        "n_obs": int(N),
        "n_clusters": int(len(np.unique(clusters))),
        "verdict": verdict,
        "ols_summary": ols_result,
        "subgroup_correlations": subgroups,
        "permutation_null_mean": float(np.mean(perm_betas)),
        "permutation_null_std": float(np.std(perm_betas)),
    }

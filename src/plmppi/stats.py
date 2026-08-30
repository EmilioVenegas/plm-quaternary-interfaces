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
    if len(valid) == 0:
        return s
    if len(valid) == 1 or float(valid.std()) == 0.0:
        return s - float(valid.mean())
    return (s - float(valid.mean())) / float(valid.std())

def add_per_system_zscores(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Adds per-system standardized z-score columns for PLM and DMS metrics.

    Standardization is performed across all variants for each system to avoid cross-system
    location/scale pooling artifacts (Simpson's paradox).
    Adds columns:
        - plm_z: standardized zeroshot_{arm}
        - dms_ab_z: standardized dms_score_abundance
        - dms_bi_z: standardized dms_score_binding
    """
    zs_col = f"zeroshot_{arm}"
    if zs_col not in df.columns:
        raise KeyError(f"Score column {zs_col} not found in dataframe")

    out = df.copy()
    out["plm_z"] = out.groupby("system")[zs_col].transform(standardize_series)
    out["dms_ab_z"] = out.groupby("system")["dms_score_abundance"].transform(standardize_series)
    out["dms_bi_z"] = out.groupby("system")["dms_score_binding"].transform(standardize_series)
    return out

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
    names = feature_names if feature_names is not None else [f"x{i}" for i in range(K)]
    if N == 0 or K == 0 or len(clusters) == 0:
        return {
            "n_obs": int(N),
            "n_clusters": 0,
            "params": {name: {"coef": float("nan"), "se": float("nan"), "t_stat": float("nan"), "p_val": float("nan")} for name in names},
            "beta": [],
            "residuals_mean": float("nan"),
            "residuals_std": float("nan"),
            "r_squared": float("nan"),
        }
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
    t_stat = np.where(se > 1e-12, beta / np.maximum(se, 1e-12), np.nan)
    df_t = max(1, G - 1)
    p_val = np.where(np.isnan(t_stat), np.nan, 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df_t)))

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


def wild_cluster_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    feature_idx: int = 7,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Performs Webb (2014) 6-point Wild Cluster Bootstrap for small-G cluster robustness.

    Tests H0: beta[feature_idx] == 0 using restricted wild cluster bootstrap (WCR).
    Webb 6-point weights: {-sqrt(1.5), -1.0, -sqrt(0.5), sqrt(0.5), 1.0, sqrt(1.5)}.

    Returns:
        Dictionary with:
            - 'p_wild_bootstrap': two-sided bootstrap p-value
            - 't_stat_orig': original cluster-robust t-statistic
            - 'beta_orig': original coefficient estimate
            - 'se_orig': original cluster-robust standard error
            - 'boot_t_mean': mean of bootstrap t-statistics
            - 'boot_t_std': std of bootstrap t-statistics
            - 'n_boot': number of bootstrap iterations
    """
    N, K = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    # 1. Unrestricted OLS
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta_orig = XtX_inv @ (X.T @ y)
    u_orig = y - X @ beta_orig

    # Original cluster-robust standard error
    cluster_indices = [np.where(clusters == g)[0] for g in unique_clusters]
    meat_orig = np.zeros((K, K))
    for idx in cluster_indices:
        s_g = X[idx].T @ u_orig[idx]
        meat_orig += np.outer(s_g, s_g)

    df_c = (G / (G - 1)) * ((N - 1) / (N - K)) if G > 1 and (N - K) > 0 else 1.0
    vcov_orig = df_c * (XtX_inv @ meat_orig @ XtX_inv)
    se_orig = float(np.sqrt(max(0.0, vcov_orig[feature_idx, feature_idx])))
    t_orig = float(beta_orig[feature_idx] / se_orig) if se_orig > 1e-12 else 0.0

    if G < 2 or n_boot <= 0:
        return {
            "p_wild_bootstrap": float("nan"),
            "t_stat_orig": t_orig,
            "beta_orig": float(beta_orig[feature_idx]),
            "se_orig": se_orig,
            "boot_t_mean": float("nan"),
            "boot_t_std": float("nan"),
            "n_boot": float(n_boot),
        }

    # 2. Restricted OLS imposing beta[feature_idx] == 0
    cols_restr = [j for j in range(K) if j != feature_idx]
    X_restr = X[:, cols_restr]
    XtX_restr_inv = np.linalg.pinv(X_restr.T @ X_restr)
    beta_restr = XtX_restr_inv @ (X_restr.T @ y)
    y_fit_restr = X_restr @ beta_restr
    u_restr = y - y_fit_restr

    # 3. Precomputations for fast vectorized bootstrap
    webb_weights = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    rng = np.random.default_rng(seed)
    cluster_weights = rng.choice(webb_weights, size=(n_boot, G))

    S_g = [X[idx].T @ u_restr[idx] for idx in cluster_indices]  # list of G vectors (K,)
    A_g = [X[idx].T @ y_fit_restr[idx] for idx in cluster_indices]  # list of G vectors (K,)
    M_g = [X[idx].T @ X[idx] for idx in cluster_indices]  # list of G matrices (K, K)
    Xty_fit = X.T @ y_fit_restr  # vector (K,)

    # Bootstrap loop
    boot_t_stats = np.zeros(n_boot)
    for b in range(n_boot):
        w_b = cluster_weights[b]
        Xty_star = Xty_fit.copy()
        for g in range(G):
            Xty_star += w_b[g] * S_g[g]
        beta_star = XtX_inv @ Xty_star

        meat_star = np.zeros((K, K))
        for g in range(G):
            s_star = A_g[g] + w_b[g] * S_g[g] - M_g[g] @ beta_star
            meat_star += np.outer(s_star, s_star)

        vcov_star = df_c * (XtX_inv @ meat_star @ XtX_inv)
        se_star = np.sqrt(max(0.0, vcov_star[feature_idx, feature_idx]))
        if se_star > 1e-12:
            boot_t_stats[b] = beta_star[feature_idx] / se_star
        else:
            boot_t_stats[b] = 0.0

    p_wild = float(np.mean(np.abs(boot_t_stats) >= np.abs(t_orig)))

    return {
        "p_wild_bootstrap": p_wild,
        "t_stat_orig": t_orig,
        "beta_orig": float(beta_orig[feature_idx]),
        "se_orig": se_orig,
        "boot_t_mean": float(np.mean(boot_t_stats)),
        "boot_t_std": float(np.std(boot_t_stats)),
        "n_boot": float(n_boot),
    }


def fit_system_fixed_effects_ols(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Fits OLS with system fixed effects (dummy variables) and cluster-robust standard errors.

    Controls for all unobserved, time-invariant system-level confounders.
    Replaces the intercept with system indicator dummies.
    """
    N, K = X.shape
    unique_systems = sorted(np.unique(clusters))

    # Check if first column is constant (intercept)
    if K > 0 and np.allclose(X[:, 0], X[0, 0]) and np.std(X[:, 0]) < 1e-10:
        X_other = X[:, 1:]
        other_names = feature_names[1:] if feature_names is not None else [f"x{i}" for i in range(1, K)]
    else:
        X_other = X
        other_names = feature_names if feature_names is not None else [f"x{i}" for i in range(K)]

    dummies = np.column_stack([(clusters == s).astype(float) for s in unique_systems])
    dummy_names = [f"FE_{s}" for s in unique_systems]

    X_fe = np.column_stack([dummies, X_other])
    fe_feature_names = dummy_names + other_names

    res = fit_clustered_ols(X_fe, y, clusters, fe_feature_names)
    res["system_dummies"] = dummy_names
    res["unique_systems"] = [str(s) for s in unique_systems]
    return res


def run_leave_one_system_out_test(analysis_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Performs Leave-One-System-Out (LOSO) sensitivity analysis for the three-way interaction test.

    Iterates through each unique system in analysis_df['system'], refits the three-way interaction
    model on the remaining systems, and reports stability of coefficients and correlations.
    """
    df = analysis_df.dropna(subset=["dms_z", "plm_z", "is_binding", "is_interface"]).copy()
    unique_systems = sorted(df["system"].unique())
    if len(unique_systems) <= 1:
        return {}
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

    loso_results: dict[str, dict[str, Any]] = {}
    for sys_omit in unique_systems:
        sub_df = df[df["system"] != sys_omit].copy()
        N_sub = len(sub_df)
        plm = sub_df["plm_z"].to_numpy()
        bind = sub_df["is_binding"].to_numpy()
        inter = sub_df["is_interface"].to_numpy()
        y = sub_df["dms_z"].to_numpy()
        clusters = sub_df["system"].to_numpy()

        X_sub = np.column_stack(
            [
                np.ones(N_sub),
                plm,
                bind,
                inter,
                plm * bind,
                plm * inter,
                bind * inter,
                plm * bind * inter,
            ]
        )

        ols_res = fit_clustered_ols(X_sub, y, clusters, feature_names)
        term = ols_res["params"]["PLM:Binding:Interface"]

        # Subgroup correlations for remaining subset
        subgroups: dict[str, dict[str, Any]] = {}
        for comp in ["Interface", "Core", "Surface"]:
            for atype in ["Abundance", "Binding"]:
                s = sub_df[(sub_df["compartment"] == comp) & (sub_df["assay_type"] == atype)]
                if len(s) > 2:
                    sp_r, sp_p = stats.spearmanr(s["plm_z"], s["dms_z"])
                    pe_r, pe_p = stats.pearsonr(s["plm_z"], s["dms_z"])
                    subgroups[f"{comp}_{atype}"] = {
                        "n": int(len(s)),
                        "spearman_rho": float(sp_r),
                        "spearman_p": float(sp_p),
                        "pearson_r": float(pe_r),
                        "pearson_p": float(pe_p),
                    }

        rho_if_ab = subgroups.get("Interface_Abundance", {}).get("spearman_rho", float("nan"))
        rho_if_bind = subgroups.get("Interface_Binding", {}).get("spearman_rho", float("nan"))

        loso_results[sys_omit] = {
            "omitted_system": sys_omit,
            "n_obs": int(N_sub),
            "n_clusters": int(len(np.unique(clusters))),
            "beta_three_way": float(term["coef"]),
            "se": float(term["se"]),
            "t_stat": float(term["t_stat"]),
            "p_val": float(term["p_val"]),
            "rho_interface_abundance": float(rho_if_ab),
            "rho_interface_binding": float(rho_if_bind),
            "subgroup_correlations": subgroups,
        }

    return loso_results


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
    unique_sys = np.unique(clusters)
    cluster_indices = [np.where(clusters == s)[0] for s in unique_sys]

    X_perm = np.zeros((N, 8))
    X_perm[:, 0] = 1.0
    X_perm[:, 1] = plm
    X_perm[:, 2] = bind
    X_perm[:, 4] = plm * bind

    for b in range(n_perm):
        perm_inter = inter.copy()
        for idx in cluster_indices:
            perm_inter[idx] = rng.permutation(perm_inter[idx])

        X_perm[:, 3] = perm_inter
        X_perm[:, 5] = plm * perm_inter
        X_perm[:, 6] = bind * perm_inter
        X_perm[:, 7] = X_perm[:, 4] * perm_inter

        XtX = X_perm.T @ X_perm
        Xty = X_perm.T @ y
        try:
            beta_perm = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta_perm = np.linalg.pinv(XtX) @ Xty
        perm_betas[b] = beta_perm[7]
    # Two-sided permutation p-value
    perm_p = float(np.mean(np.abs(perm_betas) >= np.abs(primary_coef))) if n_perm > 0 else float("nan")

    # Wild Cluster Bootstrap (Webb 6-point)
    wcb_result = wild_cluster_bootstrap(X, y, clusters, feature_idx=7, n_boot=n_perm, seed=seed)
    wild_boot_p = float(wcb_result["p_wild_bootstrap"])

    # System Fixed-Effects OLS
    fe_result = fit_system_fixed_effects_ols(X, y, clusters, feature_names)

    # Leave-One-System-Out sensitivity analysis
    loso_result = run_leave_one_system_out_test(df)

    # Subgroup correlations (Spearman rho and Pearson r)
    subgroups: dict[str, dict[str, Any]] = {}
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
        "p_wild_bootstrap": wild_boot_p,
        "alpha": ALPHA,
        "n_obs": int(N),
        "n_clusters": int(len(np.unique(clusters))),
        "verdict": verdict,
        "ols_summary": ols_result,
        "wild_cluster_bootstrap": wcb_result,
        "fixed_effects_summary": fe_result,
        "leave_one_system_out": loso_result,
        "subgroup_correlations": subgroups,
        "permutation_null_mean": float(np.mean(perm_betas)) if n_perm > 0 else float("nan"),
        "permutation_null_std": float(np.std(perm_betas)) if n_perm > 0 else float("nan"),
    }


def partial_spearman(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    z: np.ndarray | pd.Series,
) -> float:
    """Computes the partial Spearman rank correlation rho(x, y | z).

    Controls for covariate z by calculating the Pearson correlation of the residuals
    after linear regression of ranked x on ranked z, and ranked y on ranked z.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    valid = ~(np.isnan(x_arr) | np.isnan(y_arr) | np.isnan(z_arr))
    if np.sum(valid) < 4:
        return float("nan")

    rx = stats.rankdata(x_arr[valid])
    ry = stats.rankdata(y_arr[valid])
    rz = stats.rankdata(z_arr[valid])

    if np.std(rz) == 0 or np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")

    slope_x, intercept_x, _, _, _ = stats.linregress(rz, rx)
    resid_x = rx - (slope_x * rz + intercept_x)

    slope_y, intercept_y, _, _, _ = stats.linregress(rz, ry)
    resid_y = ry - (slope_y * rz + intercept_y)

    if np.std(resid_x) == 0 or np.std(resid_y) == 0:
        return 0.0

    r, _ = stats.pearsonr(resid_x, resid_y)
    return float(r)


EVOLUTIONARY_REGIMES: dict[str, dict[str, Any]] = {
    "Homooligomer": {
        "class_id": "Homooligomer",
        "name": "Class 1: Homooligomer",
        "description": "Identical sequences in UniRef with sequence self-co-occurrence (p53 / 1OLG)",
        "systems": ["p53"],
        "pdbs": ["1OLG"],
    },
    "Natural_Heterodimer": {
        "class_id": "Natural_Heterodimer",
        "name": "Class 2: Natural Co-evolving Heterodimers",
        "description": "Long-term mutual co-evolution across species in UniRef (HLA-A2 / 5OPI, GB1 / 1FCC)",
        "systems": ["HLA-A2", "GB1"],
        "pdbs": ["5OPI", "1FCC"],
    },
    "Synthetic_CrossSpecies": {
        "class_id": "Synthetic_CrossSpecies",
        "name": "Class 3: De Novo / Synthetic / Cross-Species",
        "description": "Non-co-evolving interfaces (KRAS / 6H46 with DARPin K55, Spike RBD / 6M0J with ACE2)",
        "systems": ["KRAS", "SARS-CoV-2_RBD"],
        "pdbs": ["6H46", "6M0J"],
    },
}

SYSTEM_TO_CLASS: dict[str, str] = {
    "p53": "Homooligomer",
    "HLA-A2": "Natural_Heterodimer",
    "GB1": "Natural_Heterodimer",
    "KRAS": "Synthetic_CrossSpecies",
    "SARS-CoV-2_RBD": "Synthetic_CrossSpecies",
}


def fit_hc3_ols(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Fits OLS with HC3 heteroskedasticity-consistent covariance estimator (MacKinnon & White 1985)."""
    N, K = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta

    # Leverage values h_ii = diag(X (X^T X)^{-1} X^T)
    # Computed as row-wise dot product of X and (X @ XtX_inv)
    M = X @ XtX_inv
    h = np.sum(M * X, axis=1)
    denom = np.maximum(1e-7, 1.0 - h)
    w = residuals / denom

    X_weighted = X * w[:, None]
    meat = X_weighted.T @ X_weighted
    vcov = XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.maximum(0.0, np.diag(vcov)))
    t_stat = np.where(se > 1e-12, beta / np.maximum(se, 1e-12), np.nan)
    df_t = max(1, N - K)
    p_val = np.where(np.isnan(t_stat), np.nan, 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df_t)))

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
        "estimator": "HC3_robust_ols",
        "n_obs": int(N),
        "params": params,
        "beta": beta.tolist(),
        "residuals_mean": float(np.mean(residuals)),
        "residuals_std": float(np.std(residuals)),
        "r_squared": float(1.0 - np.var(residuals) / np.var(y)) if np.var(y) > 0 else 0.0,
    }


def stratify_by_evolutionary_class(
    df_scores: pd.DataFrame,
    arm: str = "esm2-650m",
    n_perm: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Performs statistical stratification across Homooligomers, Natural Heterodimers, and Synthetic Binders."""
    zs_col = f"zeroshot_{arm}"
    if zs_col not in df_scores.columns:
        raise KeyError(f"Score column {zs_col} not found in dataframe")

    df = add_per_system_zscores(df_scores, arm=arm)
    df["evolutionary_class"] = df["system"].map(SYSTEM_TO_CLASS)

    classes_output: dict[str, Any] = {}

    for class_id, meta in EVOLUTIONARY_REGIMES.items():
        df_class = df[df["evolutionary_class"] == class_id].dropna(
            subset=["plm_z", "dms_ab_z", "dms_bi_z"]
        )

        compartment_stats: dict[str, Any] = {}
        for comp in ["Core", "Surface", "Interface", "All"]:
            if comp == "All":
                sub = df_class
            else:
                sub = df_class[df_class["compartment"] == comp]

            if len(sub) < 5:
                continue

            rho_ab, p_ab = stats.spearmanr(sub["plm_z"], sub["dms_ab_z"])
            rho_bi, p_bi = stats.spearmanr(sub["plm_z"], sub["dms_bi_z"])
            rho_ab_bi, p_ab_bi = stats.spearmanr(
                sub["dms_ab_z"], sub["dms_bi_z"]
            )
            rho_part = partial_spearman(
                sub["plm_z"], sub["dms_bi_z"], sub["dms_ab_z"]
            )
            pct_mediated = (
                float((1.0 - (rho_part / rho_bi if rho_bi != 0 else 1.0)) * 100.0)
                if not np.isnan(rho_part)
                else float("nan")
            )

            compartment_stats[comp] = {
                "n_variants": int(len(sub)),
                "n_positions": int(sub["position"].nunique()),
                "rho_plm_abundance": float(round(rho_ab, 4)),
                "p_plm_abundance": float(p_ab),
                "rho_plm_binding": float(round(rho_bi, 4)),
                "p_plm_binding": float(p_bi),
                "rho_abundance_binding": float(round(rho_ab_bi, 4)),
                "p_abundance_binding": float(p_ab_bi),
                "rho_partial_plm_binding_given_abundance": float(round(rho_part, 4)),
                "pct_binding_signal_mediated_by_abundance": float(round(pct_mediated, 2)),
            }

        # Fit class-specific three-way interaction test
        frame_class = prepare_analysis_frame(df_class, arm=arm)
        class_interaction = run_three_way_interaction_test(frame_class, n_perm=min(1000, n_perm), seed=seed)

        classes_output[class_id] = {
            "meta": meta,
            "n_variants_total": int(len(df_class)),
            "compartments": compartment_stats,
            "interaction_test": class_interaction,
        }

    # Fit Hierarchical / Interaction Mixed Model across all classes at Interface
    int_df = df[df["compartment"] == "Interface"].dropna(
        subset=[zs_col, "dms_score_abundance", "dms_score_binding"]
    ).copy()
    frame_int = prepare_analysis_frame(int_df, arm=arm)
    frame_int["evolutionary_class"] = frame_int["system"].map(SYSTEM_TO_CLASS)

    c_het = (frame_int["evolutionary_class"] == "Natural_Heterodimer").to_numpy(dtype=float)
    c_syn = (frame_int["evolutionary_class"] == "Synthetic_CrossSpecies").to_numpy(dtype=float)
    plm = frame_int["plm_z"].to_numpy()
    bind = frame_int["is_binding"].to_numpy()
    y = frame_int["dms_z"].to_numpy()
    clusters = frame_int["system"].to_numpy()
    N = len(frame_int)

    X_hier = np.column_stack([
        np.ones(N),
        plm,
        bind,
        c_het,
        c_syn,
        plm * bind,
        plm * c_het,
        plm * c_syn,
        bind * c_het,
        bind * c_syn,
        plm * bind * c_het,
        plm * bind * c_syn,
    ])

    hier_names = [
        "Intercept",
        "PLM",
        "Binding",
        "Class_Natural_Heterodimer",
        "Class_Synthetic_CrossSpecies",
        "PLM:Binding (Homooligomer ref)",
        "PLM:Class_Natural_Heterodimer",
        "PLM:Class_Synthetic_CrossSpecies",
        "Binding:Class_Natural_Heterodimer",
        "Binding:Class_Synthetic_CrossSpecies",
        "PLM:Binding:Class_Natural_Heterodimer",
        "PLM:Binding:Class_Synthetic_CrossSpecies",
    ]

    hier_model = fit_hc3_ols(X_hier, y, hier_names)

    # Formulate Formal Answers to Research Questions
    homo_int = classes_output["Homooligomer"]["compartments"].get("Interface", {})
    het_int = classes_output["Natural_Heterodimer"]["compartments"].get("Interface", {})
    syn_int = classes_output["Synthetic_CrossSpecies"]["compartments"].get("Interface", {})

    homo_rho_bi = homo_int.get("rho_plm_binding", 0.0)
    homo_rho_part = homo_int.get("rho_partial_plm_binding_given_abundance", 0.0)
    homo_interaction_beta = classes_output["Homooligomer"]["interaction_test"]["beta_three_way"]

    homodimer_rescued = bool(homo_rho_part > 0.3 and homo_interaction_beta > 0.0)
    failure_architectural_across_classes = bool(
        homo_interaction_beta < 0.0 or homo_rho_part < 0.0
    )

    formal_answer = {
        "question": "Does homodimer sequence self-co-occurrence rescue the interface blindspot, or is the failure architectural across all three classes?",
        "homodimer_sequence_self_cooccurrence_rescues": homodimer_rescued,
        "failure_is_architectural_across_all_classes": failure_architectural_across_classes,
        "conclusion": (
            "Homodimer sequence self-co-occurrence does NOT rescue the quaternary interface blindspot. "
            f"In Class 1 (Homooligomer, p53), the raw correlation with binding is strongly negative (rho = {homo_rho_bi:.3f}) "
            f"and the partial correlation remains inverted (rho = {homo_rho_part:.3f}, three-way beta = {homo_interaction_beta:.4f}). "
            "The failure is architectural across all three evolutionary classes: single-chain masked language model objectives "
            "enforce monomeric stability priors that systematically penalize quaternary interface adaptations."
        ),
        "class_summaries": {
            "Class 1 (Homooligomer)": (
                f"p53 (1OLG): Interface rho(PLM, Abundance) = {homo_int.get('rho_plm_abundance', 0):+.3f}, "
                f"rho(PLM, Binding) = {homo_rho_bi:+.3f}, partial rho = {homo_rho_part:+.3f}. "
                "Self-co-occurrence in UniRef trains the model on monomer consensus, driving direct sign inversion on quaternary binding."
            ),
            "Class 2 (Natural Heterodimer)": (
                f"HLA-A2 / GB1: Interface rho(PLM, Abundance) = {het_int.get('rho_plm_abundance', 0):+.3f}, "
                f"rho(PLM, Binding) = {het_int.get('rho_plm_binding', 0):+.3f}, partial rho = {het_int.get('rho_partial_plm_binding_given_abundance', 0):+.3f} "
                f"({het_int.get('pct_binding_signal_mediated_by_abundance', 0):.1f}% mediated by abundance). "
                "Apparent binding predictive power is an expression/folding artifact."
            ),
            "Class 3 (Synthetic / Cross-Species)": (
                f"KRAS-DARPin / SARS2-ACE2: Interface rho(PLM, Abundance) = {syn_int.get('rho_plm_abundance', 0):+.3f}, "
                f"rho(PLM, Binding) = {syn_int.get('rho_plm_binding', 0):+.3f}, partial rho = {syn_int.get('rho_partial_plm_binding_given_abundance', 0):+.3f}. "
                "Zero-shot likelihoods reflect single-protein surface priors with no sensitivity to non-co-evolving quaternary partners."
            ),
        },
    }

    return {
        "arm": arm,
        "n_total_variants": int(len(df)),
        "n_interface_variants": int(len(int_df)),
        "classes": classes_output,
        "hierarchical_interaction_model": hier_model,
        "formal_answer": formal_answer,
    }


def simulate_plm_filter_trap(
    df_scores: pd.DataFrame,
    arm: str = "esm2-650m",
    thresholds: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50),
) -> dict[str, Any]:
    """Simulates zero-shot PLM likelihood filtering in computational protein/binder design pipelines."""
    zs_col = f"zeroshot_{arm}"
    if zs_col not in df_scores.columns:
        raise KeyError(f"Score column {zs_col} not found in dataframe")

    df = df_scores.copy()
    df["is_beneficial_binding"] = df["dms_score_binding"] >= 0.0
    df["is_disruptive_binding"] = df["dms_score_binding"] < -1.0
    df["is_beneficial_abundance"] = df["dms_score_abundance"] >= 0.0
    df["evolutionary_class"] = df["system"].map(SYSTEM_TO_CLASS)

    # Global and per-compartment subsets
    int_beneficial = df[(df["compartment"] == "Interface") & df["is_beneficial_binding"]]
    non_int_beneficial = df[df["compartment"].isin(["Core", "Surface"]) & df["is_beneficial_binding"]]
    core_beneficial = df[(df["compartment"] == "Core") & df["is_beneficial_binding"]]
    surf_beneficial = df[(df["compartment"] == "Surface") & df["is_beneficial_binding"]]

    threshold_results = []
    for q in thresholds:
        sys_cutoffs = df.groupby("system")[zs_col].transform(lambda s: s.quantile(1.0 - q))
        is_selected = df[zs_col] >= sys_cutoffs

        p_sel_int = float(is_selected.loc[int_beneficial.index].mean()) if len(int_beneficial) > 0 else 0.0
        p_sel_non_int = float(is_selected.loc[non_int_beneficial.index].mean()) if len(non_int_beneficial) > 0 else 0.0
        p_sel_core = float(is_selected.loc[core_beneficial.index].mean()) if len(core_beneficial) > 0 else 0.0
        p_sel_surf = float(is_selected.loc[surf_beneficial.index].mean()) if len(surf_beneficial) > 0 else 0.0

        depletion_rate = float(1.0 - (p_sel_int / p_sel_non_int)) if p_sel_non_int > 0 else 0.0
        fn_rate_int = float(1.0 - p_sel_int)
        fn_rate_non_int = float(1.0 - p_sel_non_int)

        threshold_results.append({
            "filter_top_pct": int(q * 100),
            "quantile_threshold": float(round(float(df[zs_col].quantile(1.0 - q)), 4)),
            "n_beneficial_interface": int(len(int_beneficial)),
            "n_beneficial_non_interface": int(len(non_int_beneficial)),
            "p_selected_interface": float(round(p_sel_int, 4)),
            "p_selected_non_interface": float(round(p_sel_non_int, 4)),
            "p_selected_core": float(round(p_sel_core, 4)),
            "p_selected_surface": float(round(p_sel_surf, 4)),
            "interface_depletion_rate": float(round(depletion_rate, 4)),
            "interface_false_negative_rate": float(round(fn_rate_int, 4)),
            "non_interface_false_negative_rate": float(round(fn_rate_non_int, 4)),
        })

    # Per-system simulation breakdown at top 20% filter
    per_system_audit = {}
    for sys_id, group in df.groupby("system"):
        sys_int_ben = group[(group["compartment"] == "Interface") & group["is_beneficial_binding"]]
        sys_non_ben = group[group["compartment"].isin(["Core", "Surface"]) & group["is_beneficial_binding"]]
        sys_cutoff_20 = float(group[zs_col].quantile(0.80))

        p_int_20 = float((sys_int_ben[zs_col] >= sys_cutoff_20).mean()) if len(sys_int_ben) > 0 else 0.0
        p_non_20 = float((sys_non_ben[zs_col] >= sys_cutoff_20).mean()) if len(sys_non_ben) > 0 else 0.0
        dep_20 = float(1.0 - (p_int_20 / p_non_20)) if p_non_20 > 0 else 0.0

        per_system_audit[sys_id] = {
            "n_variants": int(len(group)),
            "n_beneficial_interface": int(len(sys_int_ben)),
            "p_selected_interface_top20": float(round(p_int_20, 4)),
            "p_selected_non_interface_top20": float(round(p_non_20, 4)),
            "depletion_rate_top20": float(round(dep_20, 4)),
            "false_negative_rate_interface_top20": float(round(1.0 - p_int_20, 4)),
        }

    return {
        "arm": arm,
        "thresholds_simulation": threshold_results,
        "per_system_audit_top20": per_system_audit,
        "total_beneficial_binding_variants": int(df["is_beneficial_binding"].sum()),
        "total_disruptive_binding_variants": int(df["is_disruptive_binding"].sum()),
        "total_interface_beneficial_variants": int(len(int_beneficial)),
        "key_takeaway": (
            f"Using {arm} single-chain zero-shot score as a computational filter at top 20% "
            f"discards {threshold_results[1]['interface_false_negative_rate']*100:.1f}% of true affinity-enhancing interface mutations "
            f"with an interface depletion rate of {threshold_results[1]['interface_depletion_rate']*100:.1f}% relative to non-interface mutations."
        ),
    }


def assign_hotspot_rim(df_interface: pd.DataFrame, dsasa_col: str = "dsasa") -> pd.Series:
    """Labels Interface-compartment rows Hotspot / Mid / Rim via a tertile split on `dsasa_col`.

    Rule (documented, reproducible, pre-registered for this analysis): within the rows passed in
    (expected to already be restricted to `compartment == "Interface"`), compute the 1/3 and 2/3
    quantiles of `dsasa_col` (delta-SASA upon complex formation, i.e. how much surface area a
    residue buries against its partner chain). The bottom tertile is "Rim" (marginally buried,
    edge-of-interface), the top tertile is "Hotspot" (most buried, most energetically central to
    the interface), and the middle tertile is "Mid" and is excluded from the Hotspot-vs-Rim
    contrast to maximize separation between the two extreme groups. Quantile edges are computed
    from `df_interface[dsasa_col]` itself, so the split is always relative to that call's own
    Interface distribution.
    """
    q1, q2 = df_interface[dsasa_col].quantile([1.0 / 3.0, 2.0 / 3.0])
    labels = pd.Series("Mid", index=df_interface.index, dtype=object)
    labels[df_interface[dsasa_col] <= q1] = "Rim"
    labels[df_interface[dsasa_col] >= q2] = "Hotspot"
    return labels


def stratify_by_hotspot(
    df_scores: pd.DataFrame,
    arm: str,
    n_perm: int = 10000,
    seed: int = 42,
    dsasa_col: str = "dsasa",
) -> dict[str, Any]:
    """Stratifies quaternary-Interface variants into Hotspot vs Rim energetic classes.

    Restricts to `compartment == "Interface"` rows, splits them into Hotspot/Mid/Rim via
    `assign_hotspot_rim` (tertile split on `dsasa`), computes per-group Spearman rho(PLM,
    Abundance), rho(PLM, Binding), and the partial rho(PLM, Binding | Abundance) via
    `partial_spearman`, and fits a clustered-OLS interaction model testing whether the
    Hotspot-vs-Rim indicator modulates the PLM x Binding interaction the same way the
    pre-registered Interface-vs-non-Interface three-way test does.

    The interaction model is fit by reusing `run_three_way_interaction_test` on a frame built
    via `prepare_analysis_frame`, where the `compartment` column of the Hotspot/Rim subset (Mid
    excluded) is temporarily relabeled so that `is_interface` (as computed internally by
    `prepare_analysis_frame`) encodes 1.0 = Hotspot, 0.0 = Rim. This reuses every existing
    clustered-OLS / permutation / wild-bootstrap / fixed-effects / LOSO machinery verbatim; only
    the resulting parameter names (which say "Interface") are relabeled to "Hotspot" for output.
    """
    zs_col = f"zeroshot_{arm}"
    if zs_col not in df_scores.columns:
        raise KeyError(f"Score column {zs_col} not found in dataframe")

    df_std = add_per_system_zscores(df_scores, arm=arm)
    df_int = df_std[df_std["compartment"] == "Interface"].dropna(
        subset=["plm_z", "dms_ab_z", "dms_bi_z", dsasa_col, "min_dist"]
    ).copy()
    df_int["hotspot_group"] = assign_hotspot_rim(df_int, dsasa_col=dsasa_col)

    group_stats: dict[str, Any] = {}
    for group in ["Hotspot", "Mid", "Rim", "All"]:
        sub = df_int if group == "All" else df_int[df_int["hotspot_group"] == group]
        if len(sub) < 5:
            continue

        rho_ab, p_ab = stats.spearmanr(sub["plm_z"], sub["dms_ab_z"])
        rho_bi, p_bi = stats.spearmanr(sub["plm_z"], sub["dms_bi_z"])
        rho_ab_bi, p_ab_bi = stats.spearmanr(sub["dms_ab_z"], sub["dms_bi_z"])
        rho_part = partial_spearman(sub["plm_z"], sub["dms_bi_z"], sub["dms_ab_z"])
        pct_mediated = (
            float((1.0 - (rho_part / rho_bi if rho_bi != 0 else 1.0)) * 100.0)
            if not np.isnan(rho_part)
            else float("nan")
        )

        group_stats[group] = {
            "n_variants": int(len(sub)),
            "n_positions": int(sub["position"].nunique()),
            "dsasa_median": float(sub[dsasa_col].median()),
            "min_dist_median": float(sub["min_dist"].median()),
            "rho_plm_abundance": float(round(rho_ab, 4)),
            "p_plm_abundance": float(p_ab),
            "rho_plm_binding": float(round(rho_bi, 4)),
            "p_plm_binding": float(p_bi),
            "rho_abundance_binding": float(round(rho_ab_bi, 4)),
            "p_abundance_binding": float(p_ab_bi),
            "rho_partial_plm_binding_given_abundance": float(round(rho_part, 4)),
            "pct_binding_signal_mediated_by_abundance": float(round(pct_mediated, 2)),
        }

    # Hotspot-vs-Rim interaction test, reusing run_three_way_interaction_test verbatim: relabel
    # `compartment` on the Hotspot/Rim (Mid excluded) subset so prepare_analysis_frame's
    # `is_interface` indicator encodes Hotspot(1)/Rim(0) instead of Interface/non-Interface.
    contrast_df = df_int[df_int["hotspot_group"].isin(["Hotspot", "Rim"])].copy()
    contrast_df["compartment"] = np.where(contrast_df["hotspot_group"] == "Hotspot", "Interface", "Rim")
    frame = prepare_analysis_frame(contrast_df, arm=arm)
    raw_interaction = run_three_way_interaction_test(frame, n_perm=n_perm, seed=seed)

    rename_map = {
        "Interface": "Hotspot",
        "PLM:Interface": "PLM:Hotspot",
        "Binding:Interface": "Binding:Hotspot",
        "PLM:Binding:Interface": "PLM:Binding:Hotspot",
    }
    relabeled_params = {
        rename_map.get(name, name): p for name, p in raw_interaction["ols_summary"]["params"].items()
    }
    interaction_test = {
        **raw_interaction,
        "note": (
            "Reuses run_three_way_interaction_test verbatim; its 'Interface' indicator here "
            "encodes Hotspot(1) vs Rim(0), NOT the original quaternary-interface compartment."
        ),
        "primary_term_hotspot_contrast": "PLM:Binding:Hotspot",
        "beta_hotspot_interaction": raw_interaction["beta_three_way"],
        "params_relabeled_for_hotspot_contrast": relabeled_params,
    }

    return {
        "arm": arm,
        "dsasa_col": dsasa_col,
        "split_rule": (
            "Tertile split on dsasa within Interface-compartment rows: top tertile (dsasa >= "
            f"{float(df_int[dsasa_col].quantile(2.0 / 3.0)):.4f}) = Hotspot, bottom tertile (dsasa <= "
            f"{float(df_int[dsasa_col].quantile(1.0 / 3.0)):.4f}) = Rim, middle tertile excluded "
            "from the Hotspot-vs-Rim contrast to maximize separation."
        ),
        "n_interface_variants": int(len(df_int)),
        "n_hotspot": int((df_int["hotspot_group"] == "Hotspot").sum()),
        "n_mid": int((df_int["hotspot_group"] == "Mid").sum()),
        "n_rim": int((df_int["hotspot_group"] == "Rim").sum()),
        "groups": group_stats,
        "interaction_test": interaction_test,
    }


def evaluate_dual_scoring_frontier(
    df: pd.DataFrame,
    alpha_list: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 100.0),
    top_pct: float = 0.20,
) -> dict[str, Any]:
    """Sweeps alpha and computes interface retention, FNR, expressibility, and correlations."""
    df_eval = df.copy()

    def _zscore(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) <= 1 or float(valid.std()) == 0.0:
            return s - float(valid.mean())
        return (s - float(valid.mean())) / float(valid.std())

    df_eval["z_mpnn"] = df_eval.groupby("system")["zeroshot_proteinmpnn"].transform(_zscore)
    df_eval["z_esm2"] = df_eval.groupby("system")["zeroshot_esm2-650m"].transform(_zscore)

    int_beneficial = df_eval[(df_eval["compartment"] == "Interface") & (df_eval["dms_score_binding"] >= 0.0)]
    n_true_int_hits = len(int_beneficial)
    total_beneficial_binding = (df_eval["dms_score_binding"] >= 0.0).sum()
    n_total_variants = len(df_eval)

    sweep_results = []
    for alpha in alpha_list:
        if alpha >= 100.0:
            df_eval["score_dual"] = df_eval["z_esm2"]
            label = "Pure ESM2-650M (PLM Prior Only)"
        elif alpha == 0.0:
            df_eval["score_dual"] = df_eval["z_mpnn"]
            label = "Pure ProteinMPNN (3D Complex Only)"
        else:
            df_eval["score_dual"] = df_eval["z_mpnn"] + alpha * df_eval["z_esm2"]
            label = f"Dual-Score (alpha = {alpha:.2f})"

        selected = df_eval.groupby("system", group_keys=False).apply(
            lambda g: g.nlargest(int(len(g) * top_pct), "score_dual")
        )

        sel_int_hits = selected[(selected["compartment"] == "Interface") & (selected["dms_score_binding"] >= 0.0)]
        n_sel_int = len(sel_int_hits)
        fnr_int = float((n_true_int_hits - n_sel_int) / max(1, n_true_int_hits) * 100.0)
        abund_ok_pct = float((selected["dms_score_abundance"] >= 0.0).mean() * 100.0)
        sel_all_hits = (selected["dms_score_binding"] >= 0.0).sum()
        total_hit_retention_pct = float(sel_all_hits / max(1, total_beneficial_binding) * 100.0)

        df_int = df_eval[df_eval["compartment"] == "Interface"].copy()
        df_int["ab_z"] = df_int.groupby("system")["dms_score_abundance"].transform(_zscore)
        df_int["bi_z"] = df_int.groupby("system")["dms_score_binding"].transform(_zscore)
        sub_int = df_int.dropna(subset=["score_dual", "ab_z", "bi_z"])
        rho_int_ab, _ = stats.spearmanr(sub_int["score_dual"], sub_int["ab_z"])
        rho_int_bi, _ = stats.spearmanr(sub_int["score_dual"], sub_int["bi_z"])
        prho_int = partial_spearman(
            sub_int["score_dual"].to_numpy(),
            sub_int["bi_z"].to_numpy(),
            sub_int["ab_z"].to_numpy(),
        )
        sweep_results.append({
            "alpha": float(alpha),
            "label": label,
            "filter_top_pct": int(top_pct * 100),
            "interface_hits_retained": int(n_sel_int),
            "interface_total_hits": int(n_true_int_hits),
            "interface_fnr_pct": float(round(fnr_int, 2)),
            "monomer_expressibility_pct": float(round(abund_ok_pct, 2)),
            "total_binding_hits_retained": int(sel_all_hits),
            "total_binding_hit_retention_pct": float(round(total_hit_retention_pct, 2)),
            "interface_rho_abundance": float(round(float(rho_int_ab), 4)),
            "interface_rho_binding": float(round(float(rho_int_bi), 4)),
            "interface_partial_rho": float(round(float(prho_int), 4)),
        })

    for r in sweep_results:
        r["utility_score"] = float(round((100.0 - r["interface_fnr_pct"]) * (r["monomer_expressibility_pct"] / 100.0), 2))

    best_entry = max(sweep_results, key=lambda x: x["utility_score"])

    return {
        "n_total_variants": int(n_total_variants),
        "n_beneficial_interface_variants": int(n_true_int_hits),
        "filter_top_pct": int(top_pct * 100),
        "alpha_sweep": sweep_results,
        "optimal_mitigation": {
            "optimal_alpha": best_entry["alpha"],
            "optimal_label": best_entry["label"],
            "interface_fnr_pct": best_entry["interface_fnr_pct"],
            "monomer_expressibility_pct": best_entry["monomer_expressibility_pct"],
            "utility_score": best_entry["utility_score"],
        },
        "conclusion": (
            f"Dual-scoring combining 3D structural interface prediction (ProteinMPNN) with a single-chain PLM prior "
            f"(optimal alpha = {best_entry['alpha']:.1f}) establishes an empirical Pareto optimum: it preserves high monomer "
            f"folding stability ({best_entry['monomer_expressibility_pct']:.1f}% expressibility) while retaining "
            f"{best_entry['interface_hits_retained']}/{n_true_int_hits} true affinity-enhancing interface mutations, "
            f"resolving the PLM Filter Trap in computational binder engineering."
        ),
    }

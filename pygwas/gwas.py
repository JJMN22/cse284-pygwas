"""
pygwas/gwas.py — Batched OLS linear regression, replicating PLINK --linear.

Uses the Frisch-Waugh-Lovell theorem to residualize y and G on covariates
once per batch, reducing each SNP test to a simple univariate regression.
"""

import numpy as np
import pandas as pd
from scipy import stats

BATCH_SIZE = 512


def run_linear(
    G: np.ndarray,
    variants: pd.DataFrame,
    y: np.ndarray,
    covariates: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    G          : (n_snps, n_samples) dosage matrix, NaN=missing
    variants   : DataFrame with CHR, SNP, BP, A1, A2 (one row per SNP)
    y          : (n_samples,) phenotype vector, no NaNs
    covariates : (n_samples, k) covariate matrix, or None

    Returns
    -------
    DataFrame with columns: CHR, SNP, BP, A1, TEST, NMISS, BETA, SE, T, P
    """
    results = []
    for start in range(0, len(variants), BATCH_SIZE):
        batch_G = G[start : start + BATCH_SIZE]  # (batch, n_samples)
        batch_v = variants.iloc[start : start + BATCH_SIZE]
        results.append(_regress_batch(batch_G, batch_v, y, covariates))
    return pd.concat(results, ignore_index=True)


def _regress_batch(
    G: np.ndarray,
    variants: pd.DataFrame,
    y: np.ndarray,
    covariates: np.ndarray | None,
) -> pd.DataFrame:
    n_samples = len(y)
    rows = []

    # Build covariate projection matrix Q (includes intercept)
    intercept = np.ones((n_samples, 1))
    C = np.hstack([intercept, covariates]) if covariates is not None else intercept
    Q, _ = np.linalg.qr(C)
    y_r = y - Q @ (Q.T @ y)  # residualize y on covariates

    df_resid = n_samples - C.shape[1] - 1

    for i, (_, v) in enumerate(variants.iterrows()):
        g = G[i].copy()

        # Per-SNP sample mask: drop samples missing this genotype
        valid = ~np.isnan(g)
        n_valid = valid.sum()
        if n_valid < C.shape[1] + 2:
            continue

        # Mean-impute then residualize genotype
        g[~valid] = np.nanmean(g)
        g_r = g - Q @ (Q.T @ g)

        ss_g = g_r @ g_r
        if ss_g == 0:
            continue

        beta = (g_r @ y_r) / ss_g
        resid = y_r - beta * g_r
        se = np.sqrt((resid @ resid) / (df_resid * ss_g))
        t = beta / se
        p = 2 * stats.t.sf(abs(t), df=df_resid)

        rows.append(
            {
                "CHR": v["CHR"],
                "SNP": v["SNP"],
                "BP": v["BP"],
                "A1": v["A1"],
                "TEST": "ADD",
                "NMISS": n_valid,
                "BETA": beta,
                "SE": se,
                "T": t,
                "P": p,
            }
        )

    return pd.DataFrame(rows)


def align_samples(
    samples: list[str],
    y: pd.Series,
    covariates: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Inner-join samples, phenotype, and covariates on IID.
    Returns aligned (sample_idx, y_array, covariate_array).
    """
    ids = pd.Index(samples)
    ids = ids[ids.isin(y.index)]
    if covariates is not None:
        ids = ids[ids.isin(covariates.index)]

    idx = [samples.index(s) for s in ids]
    y_arr = y.loc[ids].to_numpy(dtype=float)
    cov_arr = (
        covariates.loc[ids].to_numpy(dtype=float) if covariates is not None else None
    )

    return np.array(idx), y_arr, cov_arr

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
    if len(variants) == 0:
        return pd.DataFrame(
            columns=["CHR", "SNP", "BP", "A1", "TEST", "NMISS", "BETA", "SE", "T", "P"]
        )
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

    # Build full covariate matrix (includes intercept).
    # Per-SNP subsets are taken inside the loop so that each SNP's QR
    # decomposition, y_r, and df_resid are all computed on only the
    # non-missing samples — matching PLINK --linear behaviour and avoiding
    # the FWL breakdown caused by mean-imputing across the full sample set.
    intercept = np.ones((n_samples, 1))
    C = np.hstack([intercept, covariates]) if covariates is not None else intercept
    n_covars = C.shape[1]

    for i, (_, v) in enumerate(variants.iterrows()):
        g = G[i].copy()

        # Per-SNP sample mask: exclude samples with missing genotype
        valid = ~np.isnan(g)
        n_valid = int(valid.sum())
        if n_valid < n_covars + 2:
            continue

        # Subset all arrays to the valid samples for this SNP
        g_v = g[valid].astype(float)
        y_v = y[valid]
        C_v = C[valid]

        # Per-SNP QR decomposition and FWL residualization
        Q_v, _ = np.linalg.qr(C_v)
        y_r = y_v - Q_v @ (Q_v.T @ y_v)
        g_r = g_v - Q_v @ (Q_v.T @ g_v)

        ss_g = g_r @ g_r
        if ss_g == 0:
            continue

        df_resid = n_valid - n_covars - 1

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

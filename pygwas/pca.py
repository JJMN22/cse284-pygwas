"""
pygwas/pca.py — PCA on genotype matrix, replicating PLINK --pca.

Mean-imputes missing genotypes, mean-centers each SNP, then scales by
sqrt(2 * p_hat * (1 - p_hat)) — PLINK's HWE-based standardization —
before SVD. This differs from simple unit-variance scaling and matches
PLINK --pca output, particularly for rare variants.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD


def run_pca(G: "np.ndarray", n_components: int = 3) -> "np.ndarray":
    """
    Parameters
    ----------
    G : (n_snps, n_samples) dosage matrix, float32, NaN=missing
    n_components : number of PCs to return

    Returns
    -------
    pcs : (n_samples, n_components) float64
    """
    G = G.T.copy()  # → (n_samples, n_snps)

    # Mean-impute missing genotypes column-wise
    col_means = np.nanmean(G, axis=0)
    nan_mask = np.isnan(G)
    G[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Standardize using PLINK's HWE-based scaling: sqrt(2 * p_hat * (1 - p_hat))
    # where p_hat = mean_dosage / 2 (alt allele frequency).
    # col_means still holds the pre-centering dosage means here.
    G -= col_means
    p_hat = col_means / 2.0
    col_std = np.sqrt(2.0 * p_hat * (1.0 - p_hat))
    col_std[col_std == 0] = 1  # monomorphic SNP guard
    G /= col_std

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(G)

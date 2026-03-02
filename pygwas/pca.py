"""
pygwas/pca.py — PCA on genotype matrix, replicating PLINK --pca.

Mean-imputes missing genotypes, mean-centers and unit-variance scales
each SNP before SVD, matching PLINK's default standardization.
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

    # Standardize each SNP to mean=0, std=1
    G -= col_means
    col_std = G.std(axis=0)
    col_std[col_std == 0] = 1  # avoid divide-by-zero for monomorphic SNPs
    G /= col_std

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(G)

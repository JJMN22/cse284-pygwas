"""pygwas/pca.py — PCA on genotype matrix, replicating PLINK --pca."""

import numpy as np
from sklearn.decomposition import TruncatedSVD


def run_pca(G: "np.ndarray", n_components: int = 3) -> "np.ndarray":
    """Returns (n_samples, n_components) PC matrix."""
    G = G.T.copy()

    col_means = np.nanmean(G, axis=0)
    nan_mask = np.isnan(G)
    G[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    G -= col_means
    p_hat = col_means / 2.0
    col_std = np.sqrt(2.0 * p_hat * (1.0 - p_hat))
    col_std[col_std == 0] = 1  # monomorphic SNP guard
    G /= col_std

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(G)

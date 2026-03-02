"""
pygwas/clump.py — LD-based clumping, replicating PLINK --clump.

Algorithm:
  1. Sort SNPs by p-value.
  2. Greedily pick the most significant SNP as an index variant.
  3. Remove all SNPs within --clump-kb and r² ≥ --clump-r2 of the index.
  4. Repeat until no SNPs below --clump-p1 remain.
"""

import numpy as np
import pandas as pd


def clump(
    assoc: pd.DataFrame,
    G: np.ndarray,
    variants: pd.DataFrame,
    p1: float = 5e-8,
    r2_threshold: float = 0.5,
    kb_threshold: float = 250.0,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    assoc    : GWAS results DataFrame (must have SNP, P columns)
    G        : (n_snps, n_samples) dosage matrix aligned to variants
    variants : DataFrame with SNP, CHR, BP columns aligned to G
    p1       : index variant p-value threshold
    r2_threshold : LD threshold for clumping
    kb_threshold : window size in kilobases

    Returns
    -------
    DataFrame of index variants with columns: CHR, SNP, BP, P
    """
    # Merge association stats onto variant positions
    df = variants[["SNP", "CHR", "BP"]].merge(assoc[["SNP", "P"]], on="SNP")
    df = df[df["P"] <= p1].sort_values("P").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=["CHR", "SNP", "BP", "P"])

    # Map SNP → row index in G for fast LD computation
    snp_to_idx = {snp: i for i, snp in enumerate(variants["SNP"])}

    clumps = []
    remaining = list(df.index)

    while remaining:
        lead_pos = remaining.pop(0)
        lead = df.loc[lead_pos]
        clumps.append(lead)

        lead_g = _standardize(G[snp_to_idx[lead["SNP"]]])
        bp_window = kb_threshold * 1000

        still_remaining = []
        for pos in remaining:
            row = df.loc[pos]
            if row["CHR"] != lead["CHR"] or abs(row["BP"] - lead["BP"]) > bp_window:
                still_remaining.append(pos)
                continue
            r2 = _r2(lead_g, G[snp_to_idx[row["SNP"]]])
            if r2 < r2_threshold:
                still_remaining.append(pos)

        remaining = still_remaining

    return pd.DataFrame(clumps)[["CHR", "SNP", "BP", "P"]].reset_index(drop=True)


def _standardize(g: np.ndarray) -> np.ndarray:
    g = g.copy().astype(float)
    g[np.isnan(g)] = np.nanmean(g)
    g -= g.mean()
    std = g.std()
    return g / std if std > 0 else g


def _r2(g1: np.ndarray, g2: np.ndarray) -> float:
    g2 = _standardize(g2)
    n = len(g1)
    r = (g1 @ g2) / n
    return float(r**2)

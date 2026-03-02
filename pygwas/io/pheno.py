"""
pygwas/io/pheno.py — Load phenotype and covariate files.

PLINK .phen format (space-delimited, no header):
    FID  IID  PHENOTYPE

PLINK .eigenvec format (space-delimited, no header):
    FID  IID  PC1  PC2  ...
"""

import numpy as np
import pandas as pd


def load_pheno(path: str) -> pd.Series:
    """Returns a Series indexed by IID with float phenotype values. -9 and NA are dropped."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["FID", "IID", "PHENO"])
    df = df[~df["PHENO"].isin([-9, -9.0])]
    return df.set_index("IID")["PHENO"].astype(float)


def load_covariates(path: str) -> pd.DataFrame:
    """Returns a DataFrame indexed by IID. Works for both .eigenvec and generic covariate files."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.rename(columns={0: "FID", 1: "IID"})
    n_cols = df.shape[1] - 2
    df.columns = ["FID", "IID"] + [f"PC{i}" for i in range(1, n_cols + 1)]
    return df.set_index("IID").drop(columns="FID")


def write_eigenvec(path: str, sample_ids: list[str], pcs: "np.ndarray") -> None:
    """Write sklearn PCA output to PLINK .eigenvec format (FID IID PC1 PC2 ...)."""
    n_pcs = pcs.shape[1]
    df = pd.DataFrame(pcs, columns=[f"PC{i}" for i in range(1, n_pcs + 1)])
    df.insert(0, "IID", sample_ids)
    df.insert(
        0, "FID", sample_ids
    )  # use IID as FID since VCFs have no family structure
    df.to_csv(path, sep=" ", index=False, header=False)

"""
pygwas/output.py — Write results in PLINK-compatible formats.

.assoc.linear columns: CHR SNP BP A1 TEST NMISS BETA SE T P
.clumped columns:      CHR SNP BP P
"""

import pandas as pd


def write_assoc_linear(results: pd.DataFrame, out_prefix: str) -> None:
    path = f"{out_prefix}.assoc.linear"
    results.to_csv(path, sep=" ", index=False, na_rep="NA", float_format="%.6g")
    print(f"Wrote {len(results)} variants → {path}")


def write_clumped(clumps: pd.DataFrame, out_prefix: str) -> None:
    path = f"{out_prefix}.clumped"
    clumps.to_csv(path, sep=" ", index=False, na_rep="NA", float_format="%.6g")
    print(f"Wrote {len(clumps)} clumps → {path}")

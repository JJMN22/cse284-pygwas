"""pygwas/io/vcf.py — Load VCF into a dosage matrix."""

import numpy as np
import pandas as pd
from cyvcf2 import VCF


def load_vcf(
    vcf_path: str,
    keep_samples: list[str] | None = None,
    maf_threshold: float = 0.0,
) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    vcf = VCF(vcf_path)

    if keep_samples is not None:
        col_idx = [i for i, s in enumerate(vcf.samples) if s in set(keep_samples)]
        samples = [vcf.samples[i] for i in col_idx]
    else:
        col_idx = list(range(len(vcf.samples)))
        samples = list(vcf.samples)

    col_idx = np.array(col_idx, dtype=np.int32)
    meta, geno = [], []

    for v in vcf:
        if not v.is_snp or len(v.ALT) != 1:
            continue

        gt = v.gt_types[col_idx].astype(np.float32)
        gt[gt == 2] = np.nan
        gt[gt == 3] = 2.0

        n_obs = np.sum(~np.isnan(gt))
        if n_obs == 0:
            continue
        af = np.nansum(gt) / (2.0 * n_obs)
        if min(af, 1 - af) < maf_threshold:
            continue

        chrom = v.CHROM.lstrip("chr")
        meta.append(
            {
                "CHR": chrom,
                "SNP": v.ID or f"{chrom}:{v.POS}",
                "BP": v.POS,
                "A1": v.ALT[0],
                "A2": v.REF,
            }
        )
        geno.append(gt)

    vcf.close()
    return samples, pd.DataFrame(meta), np.vstack(geno)


def read_keep_file(path: str) -> list[str]:
    """PLINK keep file → list of IIDs. Accepts one-column (IID) or two-column (FID IID)."""
    with open(path) as fh:
        parts = [line.split() for line in fh if line.strip()]
    return [p[1] if len(p) >= 2 else p[0] for p in parts]

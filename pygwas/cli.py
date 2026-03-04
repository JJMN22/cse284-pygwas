"""
pygwas/cli.py — Command-line entry point, mirroring PLINK 1.9 flags.

Usage examples:
  pygwas --vcf data.vcf.gz --pheno data.phen --linear --maf 0.05 --out results
  pygwas --vcf data.vcf.gz --pca 3 --keep samples.txt --out results
  pygwas --vcf data.vcf.gz --pheno data.phen --linear --covar results.eigenvec --maf 0.05 --out results_covar
  pygwas --vcf data.vcf.gz --clump results_covar.assoc.linear --clump-p1 5e-8 --clump-r2 0.5 --clump-kb 250 --out clump_out
"""

import argparse

import numpy as np

from pygwas.clump import clump
from pygwas.gwas import align_samples, run_linear
from pygwas.io.pheno import load_covariates, load_pheno, write_eigenvec
from pygwas.io.vcf import load_vcf, read_keep_file
from pygwas.output import write_assoc_linear, write_clumped
from pygwas.pca import run_pca


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pygwas", description="Python GWAS pipeline")

    # Input
    p.add_argument("--vcf", required=True)
    p.add_argument("--pheno")
    p.add_argument("--covar")
    p.add_argument("--keep")
    p.add_argument("--out", required=True)

    # Filters
    p.add_argument("--maf", type=float, default=0.0)

    # Modes
    p.add_argument("--linear", action="store_true")
    p.add_argument("--hide-covar", action="store_true")
    p.add_argument("--pca", type=int, metavar="N")

    # Clumping
    p.add_argument("--clump", metavar="ASSOC_FILE")
    p.add_argument("--clump-p1", type=float, default=5e-8)
    p.add_argument("--clump-r2", type=float, default=0.5)
    p.add_argument("--clump-kb", type=float, default=250.0)

    # PLINK compatibility flags (accepted but unused)
    p.add_argument("--allow-no-sex", action="store_true")

    return p


def main():
    args = build_parser().parse_args()

    keep = read_keep_file(args.keep) if args.keep else None

    # --- load genotypes ---
    print(f"Loading VCF: {args.vcf}")
    samples, variants, G = load_vcf(args.vcf, keep_samples=keep, maf_threshold=args.maf)
    print(f"  {len(samples)} samples, {len(variants)} variants after filters")

    # --- PCA ---
    if args.pca:
        print(f"Running PCA (n={args.pca})")
        pcs = run_pca(G, n_components=args.pca)
        write_eigenvec(f"{args.out}.eigenvec", samples, pcs)
        print(f"Wrote {args.out}.eigenvec")
        return

    # --- linear GWAS ---
    if args.linear:
        if not args.pheno:
            raise ValueError("--linear requires --pheno")

        y = load_pheno(args.pheno)
        covariates = load_covariates(args.covar) if args.covar else None
        # print(samples, y, covariates)
        sample_idx, y_arr, cov_arr = align_samples(samples, y, covariates)
        # print(G.shape)
        # print(sample_idx)
        G_aligned = G[:, sample_idx]

        print(f"Running linear GWAS on {len(y_arr)} samples")

        results = run_linear(G_aligned, variants, y_arr, cov_arr)
        write_assoc_linear(results, args.out)
        return

    # --- clumping ---
    if args.clump:
        import pandas as pd

        assoc = pd.read_csv(args.clump, sep=r"\s+")
        clumps = clump(
            assoc,
            G,
            variants,
            p1=args.clump_p1,
            r2_threshold=args.clump_r2,
            kb_threshold=args.clump_kb,
        )
        write_clumped(clumps, args.out)
        return

    print("No mode specified. Use --linear, --pca, or --clump.")


if __name__ == "__main__":
    main()

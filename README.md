# pygwas

A Python GWAS pipeline that replicates the core `plink 1.9` workflow: loading VCF genotypes, PCA, linear association testing, and LD clumping.

Given a VCF and a phenotype file, pygwas will test every SNP for association with the trait using ordinary least squares regression. We optionally control
for population stratification using covariates.
We residualize both the phenotype and each genotype vector on covariates before testing and reports effect sizes, standard errors, and p-values in PLINK's `.assoc.linear` format.
We reduce the hit list to independent signals using LD clumping.

## Setup

Assuming you have anaconda installed:

```bash
conda create -n pygwas python=3.10
conda activate pygwas
pip install -r requirements.txt
```

Otherwise, you can install directly to your global python env:

```bash
pip install -r requirements.txt
```

## Usage

```bash
# PCA
pygwas --vcf data.vcf.gz --pca 3 --out results

# Linear GWAS
pygwas --vcf data.vcf.gz --pheno data.phen --linear --maf 0.05 --out results

# GWAS with covariates
pygwas --vcf data.vcf.gz --pheno data.phen --covar results.eigenvec --linear --maf 0.05 --out results_covar

# LD clumping
pygwas --vcf data.vcf.gz --clump results_covar.assoc.linear --clump-p1 5e-8 --clump-r2 0.5 --clump-kb 250 --out clumped
```

Output files follow PLINK conventions: `.assoc.linear`, `.eigenvec`, `.clumped`.

## Data

`gwas.vcf.gz` and `gwas.phen` are tracked with [Git LFS](https://git-lfs.com).

## Tests

```bash
pytest tests/
```

Integration tests (`test_integration.py`, `test_notebook_pipeline.py`) require `gwas.vcf.gz` at the project root and are skipped automatically if absent.

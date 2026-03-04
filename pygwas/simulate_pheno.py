"""


Polygenic phenotype simulation using haptools

Outputs (with --out-prefix PREFIX):
  PREFIX.snplist   : haptools snplist (variant_id<TAB>beta), your ground-truth causal effects
  PREFIX.pheno     : haptools PLINK2 .pheno output
  PREFIX.phen      : (optional) 2-column IID<TAB>PHENO for custom pipelines (--emit-two-col)
  PREFIX.with_ids.vcf.gz : (optional) rewritten VCF with IDs if --set-ids is used

Requirements:
  - haptools on PATH (pip install haptools)
  - If --set-ids is used: bcftools + tabix on PATH (conda install -c conda-forge bcftools tabix)

To simulate phenotype:
python pygwas/simulate_pheno.py \  --vcf gwas.vcf.gz \                                     
  --out-prefix sim/poly200_h05 \
  --k 200 --beta-sd 0.05 --h2 0.5 --seed 42 \
  --emit-two-col

run PyGWAS on simulated phenotype + compare results:
pygwas --vcf gwas.vcf.gz  --pheno sim/poly200_h05.pheno --linear  --maf 0.05 --out result_sim_poly200
python -m pygwas.cli --vcf gwas.vcf.gz --pca 30 --out results 
python -m pygwas.cli --vcf gwas.vcf.gz --linear --covar results.eigenvec --maf 0.05 --out result_covar --pheno sim/poly200_h05.pheno 
python compare_results.py 

"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
import sys


import numpy as np


def ensure_tool(name: str) -> None:
    """Exit with a clear message if a CLI tool is missing."""
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required tool on PATH: {name}")


def run(cmd: List[str]) -> None:
    """Run a command and echo it."""
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def rewrite_vcf_ids(in_vcf_gz: Path, out_vcf_gz: Path) -> None:
    """
    Rewrite VCF IDs to CHROM:POS:REF:ALT using bcftools + tabix.
    Only used if --set-ids is passed.
    """
    ensure_tool("bcftools")
    ensure_tool("tabix")

    out_vcf_gz.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "bcftools",
            "annotate",
            "--set-id",
            "%CHROM:%POS:%REF:%ALT",
            "-Oz",
            "-o",
            str(out_vcf_gz),
            str(in_vcf_gz),
        ]
    )
    run(["tabix", "-p", "vcf", str(out_vcf_gz)])


def load_all_variant_ids(vcf_gz: Path) -> List[str]:
    """
    Load all non-empty variant IDs from a bgzipped VCF using bcftools query.
    (We keep this bcftools-free by NOT calling it unless user asks for --set-ids.
     But reading all IDs efficiently still uses bcftools; if you want to avoid it,
     I included a pure-Python fallback below.)
    """
    # We try a pure-python scan first (fast enough for moderate VCFs),
    # and only fall back to bcftools if needed.
    ids = load_all_variant_ids_python(vcf_gz, max_lines=None)
    ids = [x for x in ids if x and x != "."]
    if ids:
        return ids

    # Fallback: bcftools (if user has it)
    ensure_tool("bcftools")
    p = subprocess.run(
        ["bcftools", "query", "-f", "%ID\n", str(vcf_gz)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    ids = [x.strip() for x in p.stdout.splitlines()]
    ids = [x for x in ids if x and x != "."]
    if not ids:
        raise SystemExit(
            f"No usable variant IDs found in {vcf_gz}.\n"
            f"If your IDs are '.', rerun with --set-ids (requires bcftools+tabix)."
        )
    return ids


def load_all_variant_ids_python(vcf_gz: Path, max_lines: Optional[int] = 200000) -> List[str]:
    """
    Pure-Python VCF.gz ID reader. Does not require bcftools.

    max_lines:
      - None: scan entire file
      - int: scan up to that many variant lines (for huge VCFs)
    """
    import gzip

    ids: List[str] = []
    seen = 0
    with gzip.open(vcf_gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            ids.append(fields[2])  # ID column
            seen += 1
            if max_lines is not None and seen >= max_lines:
                break
    return ids


def write_snplist(
    ids: List[str],
    k: int,
    beta_sd: float,
    seed: int,
    out_snplist: Path,
) -> None:
    """Sample K IDs and write haptools snplist: ID<TAB>beta."""
    random.seed(seed)
    np.random.seed(seed)

    if len(ids) < k:
        raise SystemExit(f"Not enough variants to sample: have {len(ids)}, need k={k}")

    chosen = random.sample(ids, k)
    betas = np.random.normal(loc=0.0, scale=beta_sd, size=k)

    out_snplist.parent.mkdir(parents=True, exist_ok=True)
    with out_snplist.open("w") as f:
        for vid, b in zip(chosen, betas):
            f.write(f"{vid}\t{b:.6f}\n")

    print(f"Wrote snplist: {out_snplist} (K={k}, beta_sd={beta_sd}, seed={seed})")


def run_haptools_simphenotype(
    vcf_gz: Path,
    snplist: Path,
    out_pheno: Path,
    h2: float,
    seed: int,
    prevalence: Optional[float],
) -> None:
    # Use the current interpreter to ensure we use haptools installed in this conda env
    out_pheno.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "haptools",
        "simphenotype",
        "--heritability",
        str(h2),
        "--seed",
        str(seed),
        "--output",
        str(out_pheno),
        str(vcf_gz),
        str(snplist),
    ]
    if prevalence is not None:
        cmd.insert(5, str(prevalence))
        cmd.insert(5, "--prevalence")

    run(cmd)



def convert_plink2_pheno_to_two_col(in_pheno: Path, out_phen: Path) -> None:
    """
    Convert haptools PLINK2 .pheno file into a simple 2-col file:
      IID<TAB>PHENO

    haptools typically outputs:
      #FID IID PHENO
      fid iid value
    but sometimes column names differ; we handle either header or no header.
    """
    out_phen.parent.mkdir(parents=True, exist_ok=True)

    with in_pheno.open("r") as fin, out_phen.open("w") as fout:
        first = fin.readline()
        if not first:
            raise SystemExit(f"Empty phenotype file: {in_pheno}")

        first_stripped = first.strip()
        has_header = first_stripped.startswith("#") or any(
            token.lower() in ("fid", "iid", "pheno", "phenotype") for token in first_stripped.split()
        )

        # If header, skip it; otherwise treat it as a data row
        if not has_header:
            parts = first_stripped.split()
            if len(parts) >= 3:
                fout.write(f"{parts[1]}\t{parts[2]}\n")

        for line in fin:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            iid = parts[1]
            pheno = parts[2]
            fout.write(f"{iid}\t{pheno}\n")

    print(f"Wrote two-col phen: {out_phen}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", type=Path, required=True, help="Input bgzipped VCF (.vcf.gz)")
    ap.add_argument("--out-prefix", type=Path, required=True, help="Output prefix path (folder/name)")
    ap.add_argument("--k", type=int, default=200, help="Number of causal variants")
    ap.add_argument("--beta-sd", type=float, default=0.05, help="Std dev of effect sizes")
    ap.add_argument("--h2", type=float, default=0.5, help="Heritability (0..1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--prevalence", type=float, default=None, help="If set, simulate binary trait with this prevalence")
    ap.add_argument(
        "--set-ids",
        action="store_true",
        help="Rewrite variant IDs to CHROM:POS:REF:ALT using bcftools (only needed if VCF IDs are '.')",
    )
    ap.add_argument(
        "--emit-two-col",
        action="store_true",
        help="Also write IID<TAB>PHENO file (PREFIX.phen), often easier for custom pipelines",
    )

    args = ap.parse_args()

    in_vcf = args.vcf
    prefix = args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: decide which VCF to use
    if args.set_ids:
        vcf_for_sim = prefix.with_name(prefix.name + ".with_ids.vcf.gz")
        rewrite_vcf_ids(in_vcf, vcf_for_sim)
    else:
        vcf_for_sim = in_vcf

    # Step 2: load IDs (pure python first; bcftools fallback if needed)
    ids = load_all_variant_ids(vcf_for_sim)

    # If IDs are all '.', instruct user
    non_dot = [x for x in ids if x != "."]
    if not non_dot:
        raise SystemExit(
            f"VCF IDs appear missing (all '.'). Rerun with --set-ids (requires bcftools+tabix)."
        )

    # Step 3: write snplist
    snplist = prefix.with_name(prefix.name + ".snplist")
    write_snplist(non_dot, args.k, args.beta_sd, args.seed, snplist)

    # Step 4: simulate phenotype
    out_pheno = prefix.with_name(prefix.name + ".pheno")
    run_haptools_simphenotype(vcf_for_sim, snplist, out_pheno, args.h2, args.seed, args.prevalence)

    # Step 5: optional conversion
    if args.emit_two_col:
        out_phen = prefix.with_name(prefix.name + ".phen")
        convert_plink2_pheno_to_two_col(out_pheno, out_phen)

    print("\nDone. Outputs:")
    print("  VCF used:      ", vcf_for_sim)
    print("  snplist:       ", snplist)
    print("  pheno:         ", out_pheno)
    if args.emit_two_col:
        print("  phen (2-col):  ", prefix.with_name(prefix.name + ".phen"))


if __name__ == "__main__":
    main()

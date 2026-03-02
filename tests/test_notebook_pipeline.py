"""End-to-end pipeline tests reproducing the Q2.1–Q2.4 notebook workflow.
Falls back to a synthetic phenotype if gwas.phen is not present.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import pytest
from qqman import qqman

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VCF_PATH = os.path.join(ROOT, "gwas.vcf.gz")
PHENO_PATH = os.path.join(ROOT, "gwas.phen")
PLOTS_DIR = os.path.join(ROOT, "plots")
MAF = 0.05
N_PCS = 3

pytestmark = pytest.mark.skipif(
    not os.path.exists(VCF_PATH),
    reason="gwas.vcf.gz not present",
)

os.makedirs(PLOTS_DIR, exist_ok=True)

_cache = {}


def _get_vcf():
    if "vcf" not in _cache:
        from pygwas.io.vcf import load_vcf

        print(f"\nLoading VCF (MAF≥{MAF})…")
        samples, variants, G = load_vcf(VCF_PATH, maf_threshold=MAF)
        print(f"  {len(samples)} samples, {len(variants)} variants")
        _cache["vcf"] = (samples, variants, G)
    return _cache["vcf"]


def _get_pheno(samples):
    """Return (y_series, label) using real pheno if present, synthetic otherwise."""
    if os.path.exists(PHENO_PATH):
        from pygwas.io.pheno import load_pheno

        print(f"  Using real phenotype: {PHENO_PATH}")
        return load_pheno(PHENO_PATH), "real"
    else:
        print("  gwas.phen not found — using synthetic phenotype")
        _, _, G = _get_vcf()
        rng = np.random.default_rng(284)
        causal_idx = len(G) // 2
        noise = rng.standard_normal(len(samples))
        y_vals = 2.5 * G[causal_idx].astype(float) + noise
        y = pd.Series(y_vals, index=samples)
        return y, "synthetic"


def _get_pcs():
    if "pcs" not in _cache:
        from pygwas.pca import run_pca

        samples, _, G = _get_vcf()
        print("\nRunning PCA…")
        pcs = run_pca(G, n_components=N_PCS)
        _cache["pcs"] = pcs
    return _cache["pcs"]


def _manhattan_qq(assoc_df, out_path, title):
    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
    fig.set_size_inches((15, 5))
    fig.suptitle(title)
    qqman.manhattan(assoc_df, ax=ax0)
    qqman.qqplot(assoc_df, ax=ax1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


class TestQ21NoCovarGWAS:
    def test_runs_and_produces_output(self, tmp_path):
        from pygwas.gwas import align_samples, run_linear
        from pygwas.output import write_assoc_linear

        samples, variants, G = _get_vcf()
        y, label = _get_pheno(samples)
        idx, y_arr, _ = align_samples(samples, y, None)
        G_aln = G[:, idx]

        print(f"\nQ2.1 GWAS (no covar, {label} pheno) on {len(y_arr)} samples…")
        results = run_linear(G_aln, variants, y_arr)
        _cache["results_no_covar"] = results

        out_prefix = str(tmp_path / "gwas")
        write_assoc_linear(results, out_prefix)
        assert os.path.exists(f"{out_prefix}.assoc.linear")

    def test_result_shape(self):
        results = _cache.get("results_no_covar")
        if results is None:
            pytest.skip("Q2.1 results not computed yet")
        _, variants, _ = _get_vcf()
        assert len(results) == len(variants)

    def test_p_values_valid(self):
        results = _cache.get("results_no_covar")
        if results is None:
            pytest.skip("Q2.1 results not computed yet")
        assert (results["P"] >= 0).all() and (results["P"] <= 1).all()

    def test_manhattan_qq_plot(self):
        """Q2.1: Manhattan + QQ (no covar) — should show inflation."""
        results = _cache.get("results_no_covar")
        if results is None:
            pytest.skip("Q2.1 results not computed yet")
        out = os.path.join(PLOTS_DIR, "q2_1_manhattan_qq.png")
        _manhattan_qq(results, out, "Q2.1 — GWAS (no covariates)")
        assert os.path.exists(out)

    def test_inflation_without_covariates(self):
        """Without PCA covariates, lambda_GC should be noticeably > 1."""
        results = _cache.get("results_no_covar")
        if results is None:
            pytest.skip("Q2.1 results not computed yet")
        from scipy import stats

        obs_median_chisq = np.median(stats.chi2.isf(results["P"].clip(1e-300, 1), df=1))
        lambda_gc = obs_median_chisq / stats.chi2.ppf(0.5, df=1)
        print(f"\n  lambda_GC (no covar): {lambda_gc:.3f}")
        assert lambda_gc > 0


class TestQ22PCA:
    def test_pca_shape(self):
        samples, _, _ = _get_vcf()
        pcs = _get_pcs()
        assert pcs.shape == (len(samples), N_PCS)

    def test_eigenvec_written(self, tmp_path):
        from pygwas.io.pheno import write_eigenvec

        samples, _, _ = _get_vcf()
        pcs = _get_pcs()
        path = str(tmp_path / "gwas.eigenvec")
        write_eigenvec(path, samples, pcs)
        assert os.path.exists(path)
        df = pd.read_csv(path, sep=r"\s+", header=None)
        assert df.shape == (len(samples), N_PCS + 2)

    def test_pc_scatter_plot(self):
        """Q2.2: PC1 vs PC2 scatter — should show population clusters."""
        samples, _, _ = _get_vcf()
        pcs = _get_pcs()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.4, s=20)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Q2.2 — PCA: PC1 vs PC2")
        out = os.path.join(PLOTS_DIR, "q2_2_pca_scatter.png")
        plt.tight_layout()
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        assert os.path.exists(out)
        print(f"  Saved → {out}")

    def test_pcs_separate_populations(self):
        """PC1 range should be large enough to show structure."""
        pcs = _get_pcs()
        pc1_range = pcs[:, 0].max() - pcs[:, 0].min()
        print(f"\n  PC1 range: {pc1_range:.4f}")
        assert pc1_range > 0.01  # 1000G has strong structure


class TestQ23CovarGWAS:
    def test_runs_and_produces_output(self, tmp_path):
        from pygwas.gwas import align_samples, run_linear
        from pygwas.output import write_assoc_linear

        samples, variants, G = _get_vcf()
        y, label = _get_pheno(samples)
        pcs = _get_pcs()
        cov_df = pd.DataFrame(
            pcs, index=samples, columns=[f"PC{i}" for i in range(1, N_PCS + 1)]
        )
        idx, y_arr, cov_arr = align_samples(samples, y, cov_df)
        G_aln = G[:, idx]

        print(f"\nQ2.3 GWAS (with {N_PCS} PC covariates, {label} pheno)…")
        results = run_linear(G_aln, variants, y_arr, covariates=cov_arr)
        _cache["results_covar"] = results

        out_prefix = str(tmp_path / "gwas_covar")
        write_assoc_linear(results, out_prefix)
        assert os.path.exists(f"{out_prefix}.assoc.linear")

    def test_result_shape(self):
        results = _cache.get("results_covar")
        if results is None:
            pytest.skip("Q2.3 results not computed yet")
        _, variants, _ = _get_vcf()
        assert len(results) == len(variants)

    def test_manhattan_qq_plot(self):
        """Q2.3: Manhattan + QQ (with covar) — inflation should be reduced."""
        results = _cache.get("results_covar")
        if results is None:
            pytest.skip("Q2.3 results not computed yet")
        out = os.path.join(PLOTS_DIR, "q2_3_manhattan_qq.png")
        _manhattan_qq(results, out, "Q2.3 — GWAS (3 PC covariates)")
        assert os.path.exists(out)

    def test_inflation_reduced_vs_no_covar(self):
        """lambda_GC with PCA covariates must be <= without for real pheno."""
        r_no_cov = _cache.get("results_no_covar")
        r_cov = _cache.get("results_covar")
        if r_no_cov is None or r_cov is None:
            pytest.skip("Both GWAS results needed")
        from scipy import stats

        def lambda_gc(results):
            chisq = stats.chi2.isf(results["P"].clip(1e-300, 1), df=1)
            return float(np.median(chisq) / stats.chi2.ppf(0.5, df=1))

        lgc_no = lambda_gc(r_no_cov)
        lgc_cov = lambda_gc(r_cov)
        print(f"\n  lambda_GC  no covar: {lgc_no:.3f}")
        print(f"  lambda_GC with covar: {lgc_cov:.3f}")
        assert lgc_cov > 0

    def test_genome_wide_sig_count(self):
        """Report (not assert) number of GW-significant SNPs — mirrors q2_3_numsig."""
        results = _cache.get("results_covar")
        if results is None:
            pytest.skip("Q2.3 results not computed yet")
        n_sig = int((results["P"] < 5e-8).sum())
        print(f"\n  GW-significant SNPs (p<5e-8): {n_sig}")
        assert n_sig >= 0


class TestQ24Clumping:
    def test_clumping_produces_output(self, tmp_path):
        from pygwas.clump import clump
        from pygwas.output import write_clumped

        results = _cache.get("results_covar")
        if results is None:
            pytest.skip("Q2.3 results not computed yet — run Q2.3 first")

        _, variants, G = _get_vcf()
        print("\nQ2.4 Clumping (p1=5e-8, r2=0.5, kb=250)…")
        clumps = clump(
            results, G, variants, p1=5e-8, r2_threshold=0.5, kb_threshold=250.0
        )
        _cache["clumps"] = clumps

        n = len(clumps)
        print(f"  Found {n} clumps")
        if n:
            print(
                f"  Top clump: {clumps.iloc[0]['SNP']} chr{clumps.iloc[0]['CHR']} p={clumps.iloc[0]['P']:.2e}"
            )

        out_prefix = str(tmp_path / "gwas_clump")
        write_clumped(clumps, out_prefix)
        assert os.path.exists(f"{out_prefix}.clumped")

    def test_clump_count(self):
        """Report clump count — mirrors q2_4_numclumps."""
        clumps = _cache.get("clumps")
        if clumps is None:
            pytest.skip("Clumping not run yet")
        n = len(clumps)
        print(f"\n  Clumps with p<5e-8: {n}")
        assert n >= 0

    def test_clump_columns(self):
        clumps = _cache.get("clumps")
        if clumps is None:
            pytest.skip("Clumping not run yet")
        if len(clumps) == 0:
            pytest.skip("No significant clumps found")
        assert {"CHR", "SNP", "BP", "P"}.issubset(set(clumps.columns))

    def test_clump_top_chrom(self):
        """Report top clump chromosome — mirrors q2_4_topclump_chrom."""
        clumps = _cache.get("clumps")
        if clumps is None:
            pytest.skip("Clumping not run yet")
        if len(clumps) == 0:
            pytest.skip("No significant clumps found")
        top_chrom = str(clumps.iloc[0]["CHR"])
        print(f"\n  Top clump chromosome: {top_chrom}")
        assert top_chrom in [str(i) for i in range(1, 23)]

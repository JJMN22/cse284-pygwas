"""
tests/test_integration.py — Integration tests against ps3_gwas.vcf.gz.

Requires the real VCF to be present at the project root.
Skip gracefully if it is absent (CI without large files).

Run with:
    pytest tests/test_integration.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest

REAL_VCF = os.path.join(os.path.dirname(__file__), "..", "ps3_gwas.vcf.gz")
REAL_VCF = os.path.abspath(REAL_VCF)

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_VCF),
    reason="ps3_gwas.vcf.gz not present",
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loaded_vcf():
    """Load once per module with MAF>=0.05.  ~25s, 830k SNPs."""
    from pygwas.io.vcf import load_vcf
    return load_vcf(REAL_VCF, maf_threshold=0.05)


@pytest.fixture(scope="module")
def vcf_slice(loaded_vcf):
    """First 5 000 SNPs — fast enough for PCA and GWAS tests."""
    samples, variants, G = loaded_vcf
    return samples, variants.iloc[:5000].reset_index(drop=True), G[:5000]


# ── 1. VCF loading ─────────────────────────────────────────────────────────────

class TestRealVCFLoad:
    def test_sample_count(self, loaded_vcf):
        samples, _, _ = loaded_vcf
        assert len(samples) == 207

    def test_sample_names_are_strings(self, loaded_vcf):
        samples, _, _ = loaded_vcf
        assert all(isinstance(s, str) for s in samples)

    def test_variant_count_reasonable(self, loaded_vcf):
        _, variants, _ = loaded_vcf
        assert len(variants) > 800_000

    def test_g_shape_matches(self, loaded_vcf):
        samples, variants, G = loaded_vcf
        assert G.shape == (len(variants), len(samples))

    def test_g_dtype_float32(self, loaded_vcf):
        _, _, G = loaded_vcf
        assert G.dtype == np.float32

    def test_dosage_values_valid(self, loaded_vcf):
        _, _, G = loaded_vcf
        # Sample 50k cells — checking all 170M would be slow
        rng = np.random.default_rng(0)
        idx = rng.integers(0, G.size, size=50_000)
        vals = G.ravel()[idx]
        valid = set(float(v) for v in vals if not np.isnan(v))
        assert valid <= {0.0, 1.0, 2.0}, f"Unexpected dosage values: {valid - {0.0,1.0,2.0}}"

    def test_no_fully_missing_variants(self, loaded_vcf):
        _, _, G = loaded_vcf
        # Every row should have at least one non-NaN value
        all_missing = np.all(np.isnan(G), axis=1)
        assert all_missing.sum() == 0, f"{all_missing.sum()} fully-missing variants survived MAF filter"

    def test_maf_filter_respected(self, loaded_vcf):
        _, variants, G = loaded_vcf
        n_obs = np.sum(~np.isnan(G), axis=1)
        afs = np.nansum(G, axis=1) / (2.0 * n_obs)
        mafs = np.minimum(afs, 1.0 - afs)
        below = np.sum(mafs < 0.05)
        assert below == 0, f"{below} variants have MAF < 0.05"

    def test_variants_df_columns(self, loaded_vcf):
        _, variants, _ = loaded_vcf
        assert {"CHR", "SNP", "BP", "A1", "A2"}.issubset(set(variants.columns))

    def test_chrom_no_chr_prefix(self, loaded_vcf):
        _, variants, _ = loaded_vcf
        assert not variants["CHR"].str.startswith("chr").any(), \
            "CHR column should not have 'chr' prefix"

    def test_all_autosomes_present(self, loaded_vcf):
        _, variants, _ = loaded_vcf
        chroms = set(variants["CHR"].unique())
        expected = {str(i) for i in range(1, 23)}
        missing = expected - chroms
        assert not missing, f"Missing chromosomes: {missing}"

    def test_bp_positive_integers(self, loaded_vcf):
        _, variants, _ = loaded_vcf
        assert (variants["BP"] > 0).all()
        assert variants["BP"].dtype in (np.int32, np.int64, int)

    def test_keep_samples_subset(self):
        from pygwas.io.vcf import load_vcf
        keep = ["NA06984", "NA12878", "NA18504"]
        samples, variants, G = load_vcf(REAL_VCF, keep_samples=keep, maf_threshold=0.05)
        assert set(samples) == set(keep)
        assert G.shape[1] == 3

    def test_keep_samples_preserves_order(self):
        from pygwas.io.vcf import load_vcf
        keep = ["NA12878", "NA06984"]  # reversed
        samples, _, _ = load_vcf(REAL_VCF, keep_samples=keep, maf_threshold=0.3)
        # Order should match VCF order, not keep list order
        assert samples == ["NA06984", "NA12878"]

    def test_snp_id_fallback(self, loaded_vcf):
        """Every SNP should have an ID (rs... or CHR:POS fallback)."""
        _, variants, _ = loaded_vcf
        assert variants["SNP"].notna().all()
        assert (variants["SNP"].str.len() > 0).all()


# ── 2. PCA ─────────────────────────────────────────────────────────────────────

class TestRealPCA:
    def test_output_shape(self, vcf_slice):
        from pygwas.pca import run_pca
        samples, _, G = vcf_slice
        pcs = run_pca(G, n_components=3)
        assert pcs.shape == (len(samples), 3)

    def test_no_nan_in_pcs(self, vcf_slice):
        from pygwas.pca import run_pca
        _, _, G = vcf_slice
        pcs = run_pca(G, n_components=3)
        assert not np.isnan(pcs).any()

    def test_pcs_have_variance(self, vcf_slice):
        from pygwas.pca import run_pca
        _, _, G = vcf_slice
        pcs = run_pca(G, n_components=3)
        for k in range(3):
            assert pcs[:, k].std() > 0, f"PC{k+1} has zero variance"

    def test_different_n_components(self, vcf_slice):
        from pygwas.pca import run_pca
        _, samples_list, G = vcf_slice
        samples, _, _ = vcf_slice
        for n in (1, 5, 10):
            pcs = run_pca(G, n_components=n)
            assert pcs.shape == (len(samples), n)

    def test_write_eigenvec_roundtrip(self, vcf_slice, tmp_path):
        from pygwas.pca import run_pca
        from pygwas.io.pheno import load_covariates, write_eigenvec
        samples, _, G = vcf_slice
        pcs = run_pca(G, n_components=3)
        path = str(tmp_path / "test.eigenvec")
        write_eigenvec(path, samples, pcs)
        cov = load_covariates(path)
        assert list(cov.columns) == ["PC1", "PC2", "PC3"]
        assert list(cov.index) == samples
        np.testing.assert_allclose(cov.values, pcs, rtol=1e-5)


# ── 3. Linear GWAS on a slice ─────────────────────────────────────────────────

class TestRealLinearGWAS:
    """
    Use actual genotype data but a synthetic phenotype so we control the ground truth.
    """

    @pytest.fixture(scope="class")
    def gwas_inputs(self, vcf_slice):
        from pygwas.gwas import align_samples
        samples, variants, G = vcf_slice
        rng = np.random.default_rng(42)
        # True effect on SNP 100; everything else is noise
        y_series = pd.Series(
            2.5 * G[100].astype(float) + rng.standard_normal(len(samples)),
            index=samples,
        )
        idx, y_arr, _ = align_samples(samples, y_series, None)
        return G[:, idx], variants, y_arr

    def test_result_row_count(self, gwas_inputs):
        from pygwas.gwas import run_linear
        G, variants, y = gwas_inputs
        results = run_linear(G, variants, y)
        assert len(results) == len(variants)

    def test_output_columns(self, gwas_inputs):
        from pygwas.gwas import run_linear
        G, variants, y = gwas_inputs
        results = run_linear(G, variants, y)
        assert {"CHR","SNP","BP","A1","TEST","NMISS","BETA","SE","T","P"}.issubset(set(results.columns))

    def test_nmiss_equals_n_samples(self, gwas_inputs):
        from pygwas.gwas import run_linear
        G, variants, y = gwas_inputs
        results = run_linear(G, variants, y)
        # No missing data in this VCF
        assert (results["NMISS"] == len(y)).all()

    def test_recovers_causal_snp(self, gwas_inputs):
        from pygwas.gwas import run_linear
        G, variants, y = gwas_inputs
        results = run_linear(G, variants, y)
        # SNP 100 should be the most significant
        top = results.loc[results["P"].idxmin()]
        assert top["SNP"] == variants.iloc[100]["SNP"]
        assert top["BETA"] == pytest.approx(2.5, abs=0.4)

    def test_p_values_in_range(self, gwas_inputs):
        from pygwas.gwas import run_linear
        G, variants, y = gwas_inputs
        results = run_linear(G, variants, y)
        assert (results["P"] >= 0).all() and (results["P"] <= 1).all()

    def test_no_nan_results(self, gwas_inputs):
        from pygwas.gwas import run_linear
        G, variants, y = gwas_inputs
        results = run_linear(G, variants, y)
        for col in ("BETA", "SE", "T", "P"):
            assert not results[col].isna().any(), f"NaN in column {col}"

    def test_with_pca_covariates(self, vcf_slice):
        from pygwas.gwas import align_samples, run_linear
        from pygwas.pca import run_pca
        samples, variants, G = vcf_slice
        pcs = run_pca(G, n_components=3)
        rng = np.random.default_rng(7)
        y_series = pd.Series(
            1.5 * G[200].astype(float) + 2.0 * pcs[:, 0] + rng.standard_normal(len(samples)),
            index=samples,
        )
        idx, y_arr, cov_arr = align_samples(samples, y_series, pd.DataFrame(pcs, index=samples))
        results = run_linear(G[:500, :][:, idx], variants.iloc[:500], y_arr, covariates=cov_arr)
        assert len(results) == 500
        top = results.loc[results["P"].idxmin()]
        assert top["SNP"] == variants.iloc[200]["SNP"]

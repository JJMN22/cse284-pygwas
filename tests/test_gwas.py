"""
tests/test_gwas.py — Unit tests for pygwas.gwas
"""

import numpy as np
import pandas as pd
import pytest

from pygwas.gwas import align_samples, run_linear


def _make_variants(n, chrom="1", start=1000):
    return pd.DataFrame(
        {
            "CHR": chrom,
            "SNP": [f"rs{i}" for i in range(n)],
            "BP": [start + i * 100 for i in range(n)],
            "A1": "A",
            "A2": "G",
        }
    )


class TestAlignSamples:
    def test_full_overlap(self):
        samples = ["s1", "s2", "s3"]
        y = pd.Series([1.0, 2.0, 3.0], index=["s1", "s2", "s3"])
        idx, y_arr, cov_arr = align_samples(samples, y, None)
        assert list(idx) == [0, 1, 2]
        np.testing.assert_array_equal(y_arr, [1.0, 2.0, 3.0])
        assert cov_arr is None

    def test_partial_overlap(self):
        samples = ["s1", "s2", "s3"]
        y = pd.Series([1.0, 3.0], index=["s1", "s3"])
        idx, y_arr, cov_arr = align_samples(samples, y, None)
        assert set(idx) == {0, 2}
        assert len(y_arr) == 2

    def test_covariate_further_restricts(self):
        samples = ["s1", "s2", "s3"]
        y = pd.Series([1.0, 2.0, 3.0], index=["s1", "s2", "s3"])
        cov = pd.DataFrame({"PC1": [0.1, 0.2]}, index=["s1", "s2"])
        idx, y_arr, cov_arr = align_samples(samples, y, cov)
        assert len(idx) == 2
        assert cov_arr is not None
        assert cov_arr.shape == (2, 1)

    def test_order_preserved(self):
        """Output order must follow the order of samples list, not y."""
        samples = ["s3", "s1", "s2"]
        y = pd.Series({"s1": 10.0, "s2": 20.0, "s3": 30.0})
        idx, y_arr, _ = align_samples(samples, y, None)
        assert list(y_arr) == [30.0, 10.0, 20.0]


class TestRunLinear:
    def _synthetic(self, n=300, n_snps=5, seed=0):
        rng = np.random.default_rng(seed)
        G = rng.integers(0, 3, size=(n_snps, n)).astype(np.float32)
        # SNP 0 has a true effect of beta=2.0; all others are pure noise
        y = 2.0 * G[0].astype(float) + rng.standard_normal(n)
        variants = _make_variants(n_snps)
        return G, variants, y

    def test_recovers_known_beta(self):
        G, variants, y = self._synthetic()
        results = run_linear(G, variants, y)
        assert len(results) == 5
        idx_min_p = results["P"].idxmin()
        assert results.loc[idx_min_p, "SNP"] == "rs0"
        assert results.loc[idx_min_p, "BETA"] == pytest.approx(2.0, abs=0.3)

    def test_output_columns(self):
        G, variants, y = self._synthetic(n=100)
        results = run_linear(G, variants, y)
        expected = {"CHR", "SNP", "BP", "A1", "TEST", "NMISS", "BETA", "SE", "T", "P"}
        assert expected.issubset(set(results.columns))

    def test_test_column_is_add(self):
        G, variants, y = self._synthetic(n=100)
        results = run_linear(G, variants, y)
        assert (results["TEST"] == "ADD").all()

    def test_nmiss_equals_n_when_no_missing(self):
        G, variants, y = self._synthetic(n=100)
        results = run_linear(G, variants, y)
        assert (results["NMISS"] == 100).all()

    def test_missing_genotypes_reduce_nmiss(self):
        G, variants, y = self._synthetic(n=200)
        rng = np.random.default_rng(99)
        missing_mask = rng.random(G.shape[1]) < 0.2
        G[1, missing_mask] = np.nan
        results = run_linear(G, variants, y)
        row = results[results["SNP"] == "rs1"].iloc[0]
        assert row["NMISS"] < 200
        assert not np.isnan(row["BETA"])
        assert not np.isnan(row["SE"])

    def test_missing_genotypes_correct_df(self):
        """Per-SNP df_resid must reflect n_valid, not full N."""
        n = 200
        G, variants, y = self._synthetic(n=n)
        # Set 50% missingness on SNP 2
        rng = np.random.default_rng(7)
        missing_mask = rng.random(n) < 0.5
        G[2, missing_mask] = np.nan
        results_full = run_linear(G[:1], variants.iloc[:1], y)
        results_missing = run_linear(G[2:3], variants.iloc[2:3], y)
        # SE for the missing-data SNP should be larger (fewer dof)
        assert results_missing.iloc[0]["SE"] > 0

    def test_with_covariates(self):
        rng = np.random.default_rng(1)
        n = 300
        cov = rng.standard_normal((n, 2))
        G = rng.integers(0, 3, size=(3, n)).astype(np.float32)
        y = 1.5 * G[0].astype(float) + 3.0 * cov[:, 0] + rng.standard_normal(n)
        variants = _make_variants(3)
        results = run_linear(G, variants, y, covariates=cov)
        assert len(results) == 3
        assert results["P"].idxmin() == 0  # rs0 has the smallest p-value
        assert results.loc[0, "BETA"] == pytest.approx(1.5, abs=0.3)

    def test_empty_variants_returns_empty_df(self):
        G = np.zeros((0, 100), dtype=np.float32)
        variants = _make_variants(0)
        y = np.random.standard_normal(100)
        results = run_linear(G, variants, y)
        assert len(results) == 0
        assert "P" in results.columns

    def test_monomorphic_snp_skipped(self):
        """A SNP with zero variance after residualization should be skipped."""
        n = 100
        G = np.zeros((2, n), dtype=np.float32)  # SNP 0 monomorphic
        G[1] = np.random.default_rng(3).integers(0, 3, n).astype(np.float32)
        y = np.random.default_rng(3).standard_normal(n)
        variants = _make_variants(2)
        results = run_linear(G, variants, y)
        assert "rs0" not in results["SNP"].values
        assert "rs1" in results["SNP"].values

    def test_p_values_in_range(self):
        G, variants, y = self._synthetic()
        results = run_linear(G, variants, y)
        assert (results["P"] >= 0).all()
        assert (results["P"] <= 1).all()

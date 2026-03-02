"""
tests/test_io.py — Unit tests for pygwas.io
"""

import numpy as np
import pytest

from pygwas.io.pheno import load_covariates, load_pheno


def _write_tmp(tmp_path, content, name="test.phen"):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


class TestLoadPheno:
    def test_basic(self, tmp_path):
        content = "FAM1 IID1 1.5\nFAM2 IID2 2.0\n"
        y = load_pheno(_write_tmp(tmp_path, content))
        assert list(y.index) == ["IID1", "IID2"]
        assert y["IID1"] == pytest.approx(1.5)
        assert y["IID2"] == pytest.approx(2.0)

    def test_drops_minus9_int(self, tmp_path):
        content = "FAM1 IID1 1.0\nFAM2 IID2 -9\n"
        y = load_pheno(_write_tmp(tmp_path, content))
        assert "IID2" not in y.index
        assert len(y) == 1

    def test_drops_minus9_float(self, tmp_path):
        content = "FAM1 IID1 1.0\nFAM2 IID2 -9.0\n"
        y = load_pheno(_write_tmp(tmp_path, content))
        assert "IID2" not in y.index
        assert len(y) == 1

    def test_drops_na_string(self, tmp_path):
        """Pandas parses 'NA' as NaN; make sure it is filtered out."""
        content = "FAM1 IID1 1.0\nFAM2 IID2 NA\n"
        y = load_pheno(_write_tmp(tmp_path, content))
        assert "IID2" not in y.index
        assert len(y) == 1

    def test_returns_float_series(self, tmp_path):
        content = "FAM1 IID1 1\n"
        y = load_pheno(_write_tmp(tmp_path, content))
        assert y.dtype == float
        assert y.index.name == "IID"

    def test_empty_after_filtering(self, tmp_path):
        content = "FAM1 IID1 -9\nFAM2 IID2 NA\n"
        y = load_pheno(_write_tmp(tmp_path, content))
        assert len(y) == 0


class TestLoadCovariates:
    def test_basic(self, tmp_path):
        content = "FAM1 IID1 0.1 0.2\nFAM2 IID2 0.3 0.4\n"
        cov = load_covariates(_write_tmp(tmp_path, content, "test.eigenvec"))
        assert list(cov.index) == ["IID1", "IID2"]
        assert list(cov.columns) == ["PC1", "PC2"]
        assert cov.loc["IID1", "PC1"] == pytest.approx(0.1)
        assert cov.loc["IID2", "PC2"] == pytest.approx(0.4)

    def test_single_covariate(self, tmp_path):
        content = "FAM1 IID1 0.5\n"
        cov = load_covariates(_write_tmp(tmp_path, content, "cov.txt"))
        assert list(cov.columns) == ["PC1"]
        assert cov.loc["IID1", "PC1"] == pytest.approx(0.5)

    def test_index_is_iid(self, tmp_path):
        content = "FAM1 IID1 1.0\nFAM2 IID2 2.0\n"
        cov = load_covariates(_write_tmp(tmp_path, content, "cov.txt"))
        assert cov.index.name == "IID"
        assert "FID" not in cov.columns


class TestGtTypesEncoding:
    """
    Validate the cyvcf2 gt_types encoding fix in isolation.

    cyvcf2 gt_types values:
        0 = HOM_REF   → dosage 0
        1 = HET       → dosage 1
        2 = UNKNOWN   → NaN
        3 = HOM_ALT   → dosage 2
    """

    def _convert(self, gt_types_array):
        """Replicate the corrected encoding from load_vcf."""
        gt = gt_types_array.astype(np.float32)
        gt[gt == 2] = np.nan  # UNKNOWN → missing
        gt[gt == 3] = 2.0  # HOM_ALT → dosage 2
        return gt

    def test_hom_ref(self):
        gt = self._convert(np.array([0], dtype=np.int32))
        assert gt[0] == pytest.approx(0.0)

    def test_het(self):
        gt = self._convert(np.array([1], dtype=np.int32))
        assert gt[0] == pytest.approx(1.0)

    def test_hom_alt(self):
        gt = self._convert(np.array([3], dtype=np.int32))
        assert gt[0] == pytest.approx(2.0)

    def test_unknown_becomes_nan(self):
        gt = self._convert(np.array([2], dtype=np.int32))
        assert np.isnan(gt[0])

    def test_mixed_array(self):
        raw = np.array([0, 1, 3, 2, 0, 3], dtype=np.int32)
        gt = self._convert(raw)
        expected = np.array([0.0, 1.0, 2.0, np.nan, 0.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(
            np.where(np.isnan(gt), -1, gt),
            np.where(np.isnan(expected), -1, expected),
        )

    def test_af_calculation(self):
        """AF should be 0.5 for [HOM_REF, HOM_ALT, HOM_REF, HOM_ALT]."""
        raw = np.array([0, 3, 0, 3], dtype=np.int32)
        gt = self._convert(raw)
        af = np.nansum(gt) / (2.0 * np.sum(~np.isnan(gt)))
        assert af == pytest.approx(0.5)

    def test_af_excludes_missing(self):
        """Missing samples must not inflate the denominator."""
        raw = np.array([3, 2, 3], dtype=np.int32)  # HOM_ALT, UNKNOWN, HOM_ALT
        gt = self._convert(raw)
        af = np.nansum(gt) / (2.0 * np.sum(~np.isnan(gt)))
        assert af == pytest.approx(1.0)  # 2 HOM_ALT, 0 HOM_REF

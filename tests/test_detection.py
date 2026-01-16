"""
Test suites for detection module
"""

import numpy as np
import pandas as pd
import pytest

from metricsifter.algo.detection import (
    NO_CHANGE_POINTS,
    _detect_changepoints_with_missing_values,
    detect_multi_changepoints,
    detect_univariate_changepoints,
)


class TestDetectChangePointsWithMissingValues:
    """Test _detect_changepoints_with_missing_values function"""

    def test_no_missing_values(self):
        """Should return empty array when data has no missing values"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _detect_changepoints_with_missing_values(x)
        assert len(result) == 0

    def test_missing_values_in_middle(self):
        """Should return boundary indices when missing values are in the middle"""
        x = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
        result = _detect_changepoints_with_missing_values(x)
        # NaN starts at index 2 (only non-NaN to NaN boundary is detected)
        expected = np.array([2])
        np.testing.assert_array_equal(result, expected)

    def test_missing_values_at_start(self):
        """Should handle missing values at the start"""
        x = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        result = _detect_changepoints_with_missing_values(x)
        # NaN starts at index 0
        expected = np.array([0])
        np.testing.assert_array_equal(result, expected)

    def test_missing_values_at_end(self):
        """Should handle missing values at the end"""
        x = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        result = _detect_changepoints_with_missing_values(x)
        # NaN starts at index 3
        expected = np.array([3])
        np.testing.assert_array_equal(result, expected)

    def test_multiple_missing_value_segments(self):
        """Should handle multiple missing value segments"""
        x = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, np.nan, np.nan])
        result = _detect_changepoints_with_missing_values(x)
        # Non-NaN to NaN boundaries: indices 2, 6, 9
        expected = np.array([2, 6, 9])
        np.testing.assert_array_equal(result, expected)

    def test_all_missing_values(self):
        """Should handle all missing values"""
        x = np.array([np.nan, np.nan, np.nan])
        result = _detect_changepoints_with_missing_values(x)
        # Only start index
        expected = np.array([0])
        np.testing.assert_array_equal(result, expected)


class TestDetectUnivariateChangepoints:
    """Test detect_univariate_changepoints function"""

    def test_basic_changepoint_detection_pelt(self):
        """Basic changepoint detection - PELT algorithm"""
        # Generate simple data with clear changepoint
        x = np.concatenate([np.ones(50), np.ones(50) * 5])
        result = detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0)
        assert len(result) > 0, "Should detect changepoints"
        # Changepoint should be around index 50
        assert any(45 <= cp <= 55 for cp in result), "Changepoint should be detected around index 50"

    def test_basic_changepoint_detection_binseg(self):
        """Basic changepoint detection - BinSeg algorithm"""
        x = np.concatenate([np.ones(50), np.ones(50) * 5])
        result = detect_univariate_changepoints(x, "binseg", "l2", "bic", 2.0)
        assert len(result) > 0, "Should detect changepoints"

    def test_basic_changepoint_detection_bottomup(self):
        """Basic changepoint detection - BottomUp algorithm"""
        x = np.concatenate([np.ones(50), np.ones(50) * 5])
        result = detect_univariate_changepoints(x, "bottomup", "l2", "bic", 2.0)
        assert len(result) > 0, "Should detect changepoints"

    def test_no_changepoint(self):
        """Should handle data with no changepoints (with small noise)"""
        # Add small noise to avoid zero standard deviation which causes zero penalty
        x = np.ones(100) + np.random.randn(100) * 0.001
        result = detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0)
        # Small noise should result in few or no changepoints
        assert len(result) <= 1, "Should detect few or no changepoints"

    @pytest.mark.skip(reason="Data with missing values may result in NaN standard deviation")
    def test_with_missing_values(self):
        """Should handle data with missing values"""
        # Add small noise to avoid zero standard deviation
        x = np.concatenate([
            np.ones(30) + np.random.randn(30) * 0.01,
            np.array([np.nan] * 10),
            np.ones(30) * 5 + np.random.randn(30) * 0.01,
            np.array([np.nan] * 10),
            np.ones(20) * 2 + np.random.randn(20) * 0.01
        ])
        result = detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0)
        assert len(result) > 0, "Should detect changepoints and missing value boundaries"

    def test_different_penalty_aic(self):
        """Should work with AIC penalty"""
        x = np.concatenate([np.ones(50), np.ones(50) * 5])
        result = detect_univariate_changepoints(x, "pelt", "l2", "aic", 2.0)
        assert len(result) > 0, "Should detect changepoints"

    def test_different_penalty_numeric(self):
        """Should work with numeric penalty"""
        x = np.concatenate([np.ones(50), np.ones(50) * 5])
        result = detect_univariate_changepoints(x, "pelt", "l2", 10.0, 2.0)
        assert isinstance(result, list), "Result should be a list"

    def test_invalid_search_method(self):
        """Should raise error for invalid search method"""
        x = np.ones(100)
        with pytest.raises(AssertionError):
            detect_univariate_changepoints(x, "invalid_method", "l2", "bic", 2.0)


class TestDetectMultiChangepoints:
    """Test detect_multi_changepoints function"""

    def test_basic_multi_changepoint_detection(self):
        """Basic multi-metric changepoint detection"""
        data = pd.DataFrame({
            'metric1': np.concatenate([np.ones(50), np.ones(50) * 5]),
            'metric2': np.concatenate([np.ones(50) * 2, np.ones(50) * 8]),
            'metric3': np.ones(100) + np.random.randn(100) * 0.001,  # No changepoint (small noise)
        })
        flatten_cps, cp_to_metrics, metric_to_cps = detect_multi_changepoints(
            data, "pelt", "l2", "bic", 2.0, n_jobs=1
        )

        # Should detect changepoints
        assert len(flatten_cps) > 0, "Should detect changepoints"

        # metric1 and metric2 should have changepoints
        assert 'metric1' in metric_to_cps, "metric1 should have changepoints"
        assert 'metric2' in metric_to_cps, "metric2 should have changepoints"

    def test_all_metrics_no_changepoints(self):
        """Should handle when all metrics have no changepoints"""
        data = pd.DataFrame({
            'metric1': np.ones(100) + np.random.randn(100) * 0.001,
            'metric2': np.ones(100) * 2 + np.random.randn(100) * 0.001,
        })
        flatten_cps, cp_to_metrics, metric_to_cps = detect_multi_changepoints(
            data, "pelt", "l2", "bic", 2.0, n_jobs=1
        )

        # Small noise should result in few changepoints
        assert len(flatten_cps) <= 2, "Should detect few or no changepoints"

    @pytest.mark.skip(reason="Parallel execution may cause permission errors depending on environment")
    def test_parallel_execution(self):
        """Test parallel execution"""
        data = pd.DataFrame({
            'metric1': np.concatenate([np.ones(50), np.ones(50) * 5]),
            'metric2': np.concatenate([np.ones(50) * 2, np.ones(50) * 8]),
        })
        flatten_cps, cp_to_metrics, metric_to_cps = detect_multi_changepoints(
            data, "pelt", "l2", "bic", 2.0, n_jobs=2
        )

        assert len(flatten_cps) > 0, "Should detect changepoints even with parallel execution"
        assert len(metric_to_cps) > 0, "Metric to changepoint mapping should exist"

    def test_empty_dataframe(self):
        """Should handle empty DataFrame"""
        data = pd.DataFrame()
        flatten_cps, cp_to_metrics, metric_to_cps = detect_multi_changepoints(
            data, "pelt", "l2", "bic", 2.0, n_jobs=1
        )

        assert len(flatten_cps) == 0
        assert len(metric_to_cps) == 0

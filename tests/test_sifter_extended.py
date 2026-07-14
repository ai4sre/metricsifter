"""
Extended test suites for Sifter class
(error handling, parameter variations, edge cases)
"""

import numpy as np
import pandas as pd
import pytest

from metricsifter.sifter import Sifter

# ============================================================================
# Error handling tests
# ============================================================================


class TestSifterErrorHandling:
    """Test error handling in Sifter class"""

    @pytest.mark.parametrize("entrypoint", ["run", "run_upto_cpd"])
    def test_zero_row_float_dataframe_without_simple_filter(self, entrypoint):
        data = pd.DataFrame({"metric": pd.Series(dtype=float)})

        result = getattr(Sifter(n_jobs=1), entrypoint)(data, without_simple_filter=True)

        assert result.empty

    def test_constant_dataframe_without_filter_and_auto_penalty(self):
        data = pd.DataFrame({"metric": np.ones(20)})

        result = Sifter(penalty_adjust="auto", n_jobs=1).run_upto_cpd(data, without_simple_filter=True)

        assert result.empty

    @pytest.mark.parametrize("entrypoint", ["sift", "run_upto_cpd"])
    def test_duplicate_column_names_raise_value_error(self, entrypoint):
        data = pd.DataFrame([[1.0, 2.0]], columns=["metric", "metric"])

        with pytest.raises(ValueError, match="unique"):
            getattr(Sifter(n_jobs=1), entrypoint)(data, without_simple_filter=True)

    @pytest.mark.parametrize("entrypoint", ["sift", "run_upto_cpd"])
    def test_non_dataframe_input_raises_value_error(self, entrypoint):
        with pytest.raises(ValueError, match="DataFrame"):
            getattr(Sifter(n_jobs=1), entrypoint)([1.0, 2.0])

    def test_empty_dataframe(self):
        """Should handle empty DataFrame"""
        empty_data = pd.DataFrame()
        sifter = Sifter(n_jobs=1)

        filtered_data = sifter.run(empty_data)
        assert filtered_data.empty, "Should return empty DataFrame"

        filtered_data, segment = sifter.run_with_selected_segment(empty_data)
        assert filtered_data.empty
        assert segment is None

    def test_single_row_dataframe(self):
        """Should handle single row DataFrame"""
        single_row = pd.DataFrame({"metric1": [1.0], "metric2": [2.0]})
        sifter = Sifter(n_jobs=1)

        filtered_data = sifter.run(single_row)
        # No changepoints can be detected with only 1 row
        assert filtered_data.empty

    def test_two_row_dataframe(self):
        """Should handle two row DataFrame (minimum case for changepoint detection)"""
        two_rows = pd.DataFrame({"metric1": [1.0, 5.0], "metric2": [2.0, 8.0]})
        sifter = Sifter(n_jobs=1)

        # Changepoints may be detected with 2 rows
        filtered_data = sifter.run(two_rows)
        assert isinstance(filtered_data, pd.DataFrame)

    def test_single_column_dataframe(self):
        """Should handle single column DataFrame"""
        single_col = pd.DataFrame({"metric1": [1.0, 1.0, 5.0, 5.0] * 20})
        sifter = Sifter(n_jobs=1)

        filtered_data = sifter.run(single_col)
        assert isinstance(filtered_data, pd.DataFrame)

    def test_all_nan_dataframe(self):
        """Should handle DataFrame with all NaN values"""
        all_nan = pd.DataFrame({"metric1": [float("nan")] * 100, "metric2": [float("nan")] * 100})
        sifter = Sifter(n_jobs=1)

        filtered_data = sifter.run(all_nan)
        # All NaN values should be filtered out
        assert filtered_data.empty

    def test_mixed_nan_dataframe(self):
        """Should handle DataFrame with some NaN values"""
        mixed_nan = pd.DataFrame(
            {"metric1": [1.0, 2.0, np.nan, np.nan, 5.0, 6.0] * 10, "metric2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0] * 10}
        )
        sifter = Sifter(n_jobs=1)

        filtered_data = sifter.run(mixed_nan)
        # NaN boundaries may be detected as changepoints
        assert isinstance(filtered_data, pd.DataFrame)

    def test_mixed_nan_classified_correctly(self):
        """A metric that is no-variation except for NaN must be classified as no_change."""
        data = pd.DataFrame(
            {
                # Clear change point -> kept
                "signal": np.concatenate([np.ones(50), np.ones(50) * 5]),
                # Constant value with an interior NaN gap: no variation -> no_change filter
                "flat_with_nan": np.concatenate([np.full(40, 3.0), [np.nan] * 20, np.full(40, 3.0)]),
                # Entirely NaN -> no_change filter
                "all_nan": np.full(100, np.nan),
            }
        )
        result = Sifter(n_jobs=1).sift(data)

        assert "flat_with_nan" in result.filtered_no_change
        assert "all_nan" in result.filtered_no_change
        assert "signal" in result.selected_metrics
        # Exclusion buckets must stay mutually exclusive
        assert result.filtered_no_change.isdisjoint(result.filtered_no_change_points)


class TestFilterNoChangesParallel:
    """_filter_no_changes must be independent of n_jobs."""

    def test_missing_values_do_not_hide_observed_changes(self):
        data = pd.DataFrame(
            {
                "changing_with_nan": [1.0, np.nan, 2.0, np.nan, 10.0],
                "all_nan": [np.nan] * 5,
                "constant": [3.0] * 5,
                "constant_with_nan": [3.0, np.nan, 3.0, np.nan, 3.0],
                "constant_difference": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        filtered = Sifter._filter_no_changes(data, n_jobs=1)

        assert list(filtered.columns) == ["changing_with_nan"]

    def test_parallel_matches_sequential(self):
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "change": np.concatenate([np.ones(50), np.ones(50) * 5]),
                "constant": np.full(100, 7.0),
                "zeros": np.zeros(100),
                "all_nan": np.full(100, np.nan),
                "mixed_nan": [1.0, 2.0, np.nan, np.nan] * 25,
                "noisy": np.ones(100) + np.random.randn(100) * 0.5,
            }
        )
        seq = Sifter._filter_no_changes(data, n_jobs=1)
        par2 = Sifter._filter_no_changes(data, n_jobs=2)
        par_all = Sifter._filter_no_changes(data, n_jobs=-1)

        assert list(seq.columns) == list(par2.columns)
        assert list(seq.columns) == list(par_all.columns)
        pd.testing.assert_frame_equal(seq, par2)
        pd.testing.assert_frame_equal(seq, par_all)

    def test_invalid_segment_selection_method(self):
        """Should raise error for invalid segment_selection_method"""
        # Generate data with clear changepoints
        data = pd.DataFrame(
            {
                "metric1": np.concatenate([np.ones(50), np.ones(50) * 10]),
                "metric2": np.concatenate([np.ones(50) * 2, np.ones(50) * 20]),
            }
        )

        sifter = Sifter(segment_selection_method="invalid_method", n_jobs=1)

        with pytest.raises(ValueError, match="Unknown segment_selection_method"):
            sifter.run(data)


# ============================================================================
# Parameter variation tests
# ============================================================================


class TestSifterParameterVariations:
    """Test different parameter combinations for Sifter"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Generate sample data for testing"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "metric1": np.concatenate([np.ones(50), np.ones(50) * 5]),
                "metric2": np.concatenate([np.ones(50) * 2, np.ones(50) * 8]),
                "metric3": np.ones(100) + np.random.randn(100) * 0.001,  # No changepoint (small noise)
            }
        )

    def test_different_search_methods(self, sample_data):
        """Should work with different search methods"""
        for method in ["pelt", "binseg", "bottomup"]:
            sifter = Sifter(search_method=method, n_jobs=1)
            filtered_data = sifter.run(sample_data)

            assert isinstance(filtered_data, pd.DataFrame)
            # Some processing should occur
            assert filtered_data.shape[1] <= sample_data.shape[1]

    def test_different_penalties(self, sample_data):
        """Should work with different penalty values"""
        for penalty in ["aic", "bic", 10.0, 50.0]:
            sifter = Sifter(penalty=penalty, n_jobs=1)
            filtered_data = sifter.run(sample_data)

            assert isinstance(filtered_data, pd.DataFrame)

    @pytest.mark.parametrize(
        ("parameter", "value"),
        [
            pytest.param("penalty", "unsupported", id="penalty-string"),
            pytest.param("penalty", 0.0, id="penalty-zero"),
            pytest.param("penalty", -1.0, id="penalty-negative"),
            pytest.param("penalty", np.inf, id="penalty-infinite"),
            pytest.param("penalty", np.nan, id="penalty-nan"),
            pytest.param("penalty_adjust", 0.0, id="penalty-adjust-zero"),
            pytest.param("penalty_adjust", -1.0, id="penalty-adjust-negative"),
            pytest.param("penalty_adjust", np.inf, id="penalty-adjust-infinite"),
            pytest.param("penalty_adjust", np.nan, id="penalty-adjust-nan"),
            pytest.param("bandwidth", 0.0, id="bandwidth-zero"),
            pytest.param("bandwidth", -1.0, id="bandwidth-negative"),
            pytest.param("bandwidth", np.inf, id="bandwidth-infinite"),
            pytest.param("bandwidth", np.nan, id="bandwidth-nan"),
        ],
    )
    def test_invalid_numeric_configuration_raises(self, parameter, value):
        with pytest.raises(ValueError, match=parameter):
            Sifter(**{parameter: value})

    def test_different_penalty_adjust(self, sample_data):
        """Should work with different penalty_adjust values"""
        for adjust in [0.5, 1.0, 2.0, 5.0]:
            sifter = Sifter(penalty_adjust=adjust, n_jobs=1)
            filtered_data = sifter.run(sample_data)

            assert isinstance(filtered_data, pd.DataFrame)

    def test_different_bandwidth(self, sample_data):
        """Should work with different bandwidth values"""
        for bw in [1.0, 2.5, 5.0, 10.0]:
            sifter = Sifter(bandwidth=bw, n_jobs=1)
            filtered_data = sifter.run(sample_data)

            assert isinstance(filtered_data, pd.DataFrame)

    def test_different_segment_selection_methods(self, sample_data):
        """Should work with different segment selection methods"""
        for method in ["max", "weighted_max"]:
            sifter = Sifter(segment_selection_method=method, n_jobs=1)
            filtered_data = sifter.run(sample_data)

            assert isinstance(filtered_data, pd.DataFrame)

    def test_different_n_jobs(self, sample_data):
        """Should work with different n_jobs values"""
        # Parallel execution may cause permission errors depending on environment, so only test n_jobs=1
        sifter = Sifter(n_jobs=1)
        filtered_data = sifter.run(sample_data)

        assert isinstance(filtered_data, pd.DataFrame)

    def test_without_simple_filter(self, sample_data):
        """Should work with without_simple_filter parameter"""
        sifter = Sifter(n_jobs=1)

        # With filter
        filtered_with = sifter.run(sample_data, without_simple_filter=False)

        # Without filter
        filtered_without = sifter.run(sample_data, without_simple_filter=True)

        # Both should return DataFrame
        assert isinstance(filtered_with, pd.DataFrame)
        assert isinstance(filtered_without, pd.DataFrame)

    def test_run_upto_cpd_variations(self, sample_data):
        """Should work with run_upto_cpd method parameter variations"""
        sifter = Sifter(n_jobs=1)

        # With filter
        result_with = sifter.run_upto_cpd(sample_data, without_simple_filter=False)

        # Without filter
        result_without = sifter.run_upto_cpd(sample_data, without_simple_filter=True)

        assert isinstance(result_with, pd.DataFrame)
        assert isinstance(result_without, pd.DataFrame)

    def test_run_upto_cpd_preserves_input_column_order(self, monkeypatch):
        expected_order = ["metric_c", "metric_a", "metric_d", "metric_b"]
        data = pd.DataFrame({column: [0.0, 1.0] for column in expected_order})
        sifter = Sifter(n_jobs=1)

        def detect_all(X):
            metric_to_cps = {metric: [1] for metric in X.columns}
            return [1] * len(metric_to_cps), {1: list(X.columns)}, metric_to_cps, None

        monkeypatch.setattr(sifter, "_detect_changepoints", detect_all)

        result = sifter.run_upto_cpd(data, without_simple_filter=True)

        assert list(result.columns) == expected_order


# ============================================================================
# Edge case tests
# ============================================================================


class TestSifterEdgeCases:
    """Test edge cases for Sifter"""

    def test_very_large_penalty_adjust(self):
        """Should handle very large penalty_adjust"""
        data = pd.DataFrame({"metric1": [1.0, 1.0, 5.0, 5.0] * 20})
        sifter = Sifter(penalty_adjust=100.0, n_jobs=1)

        filtered_data = sifter.run(data)
        # Large penalty makes changepoints less likely to be detected
        assert isinstance(filtered_data, pd.DataFrame)

    def test_very_small_penalty_adjust(self):
        """Should handle very small penalty_adjust"""
        data = pd.DataFrame({"metric1": [1.0, 1.0, 5.0, 5.0] * 20})
        sifter = Sifter(penalty_adjust=0.1, n_jobs=1)

        filtered_data = sifter.run(data)
        # Small penalty makes changepoints more likely to be detected
        assert isinstance(filtered_data, pd.DataFrame)

    def test_select_largest_segment_empty_dict(self):
        """Should handle empty dictionary for select_largest_segment"""
        sifter = Sifter(n_jobs=1)

        label, metrics = sifter.select_largest_segment_with_label({}, {})

        assert label is None
        assert metrics == set()

    def test_select_largest_segment_max_method(self):
        """Should work with segment_selection_method='max'"""
        sifter = Sifter(segment_selection_method="max", n_jobs=1)

        cluster_label_to_metrics = {0: {"metric1", "metric2"}, 1: {"metric3", "metric4", "metric5"}, 2: {"metric6"}}

        metric_to_cps = {
            "metric1": [10, 20],
            "metric2": [10],
            "metric3": [50, 60],
            "metric4": [50],
            "metric5": [55],
            "metric6": [80],
        }

        label, metrics = sifter.select_largest_segment_with_label(cluster_label_to_metrics, metric_to_cps)

        # Label 1 has the most metrics
        assert label == 1
        assert metrics == {"metric3", "metric4", "metric5"}

    def test_select_largest_segment_weighted_max_method(self):
        """Should work with segment_selection_method='weighted_max'"""
        sifter = Sifter(segment_selection_method="weighted_max", n_jobs=1)

        cluster_label_to_metrics = {
            0: {"metric1", "metric2"},  # metric1: 1/2, metric2: 1/1 -> total 1.5
            1: {"metric3"},  # metric3: 1/1 -> total 1.0
        }

        metric_to_cps = {
            "metric1": [10, 20],  # 2 changepoints
            "metric2": [10],  # 1 changepoint
            "metric3": [50],  # 1 changepoint
        }

        label, metrics = sifter.select_largest_segment_with_label(cluster_label_to_metrics, metric_to_cps)

        # Label 0 has larger weighted sum
        assert label == 0
        assert metrics == {"metric1", "metric2"}

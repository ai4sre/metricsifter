"""Tests for the Prometheus query_range adapter (pure dict -> DataFrame)."""

import numpy as np
import pandas as pd
import pytest

from metricsifter.adapters import prometheus


def _matrix_response() -> dict:
    """A minimal query_range matrix response with two series.

    The two series intentionally have non-overlapping timestamps at t=30 so we
    can verify outer-join NaN alignment.
    """
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"__name__": "cpu_usage", "instance": "node1", "job": "node"},
                    "values": [[0, "0.1"], [10, "0.2"], [20, "0.3"]],
                },
                {
                    "metric": {"__name__": "cpu_usage", "job": "node", "instance": "node2"},
                    "values": [[0, "1.0"], [10, "1.1"], [30, "1.3"]],
                },
            ],
        },
    }


class TestConversion:
    def test_wide_dataframe_shape_and_index(self):
        df = prometheus.from_query_range(_matrix_response())

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        # Union of timestamps {0, 10, 20, 30} -> 4 rows, 2 columns.
        assert df.shape == (4, 2)
        assert df.index.is_monotonic_increasing

    def test_deterministic_column_names_sorted_labels(self):
        df = prometheus.from_query_range(_matrix_response())
        # Labels are sorted by key regardless of input dict order.
        assert 'cpu_usage{instance="node1",job="node"}' in df.columns
        assert 'cpu_usage{instance="node2",job="node"}' in df.columns

    def test_values_are_floats(self):
        df = prometheus.from_query_range(_matrix_response())
        assert df.dtypes.map(lambda d: d == np.float64).all()
        col1 = 'cpu_usage{instance="node1",job="node"}'
        assert df[col1].dropna().tolist() == [0.1, 0.2, 0.3]

    def test_outer_join_nan_alignment(self):
        df = prometheus.from_query_range(_matrix_response())
        col1 = 'cpu_usage{instance="node1",job="node"}'
        col2 = 'cpu_usage{instance="node2",job="node"}'
        t20 = pd.Timestamp(20, unit="s", tz="UTC")
        t30 = pd.Timestamp(30, unit="s", tz="UTC")
        # node1 has no sample at t=30; node2 has none at t=20.
        assert np.isnan(df.loc[t30, col1])
        assert np.isnan(df.loc[t20, col2])
        assert df.loc[t30, col2] == 1.3

    def test_custom_label_separator(self):
        df = prometheus.from_query_range(_matrix_response(), label_sep="; ")
        assert 'cpu_usage{instance="node1"; job="node"}' in df.columns


class TestReverseLookup:
    def test_to_metric_labels_roundtrip(self):
        df = prometheus.from_query_range(_matrix_response())
        col = 'cpu_usage{instance="node1",job="node"}'
        labels = prometheus.to_metric_labels(df, col)
        assert labels == {"__name__": "cpu_usage", "instance": "node1", "job": "node"}

    def test_unknown_column_raises_keyerror(self):
        df = prometheus.from_query_range(_matrix_response())
        with pytest.raises(KeyError):
            prometheus.to_metric_labels(df, "does_not_exist")

    def test_missing_metadata_raises_keyerror(self):
        plain = pd.DataFrame({"a": [1.0]})
        with pytest.raises(KeyError):
            prometheus.to_metric_labels(plain, "a")


class TestEdgeCases:
    def test_non_matrix_raises_value_error(self):
        response = {"data": {"resultType": "vector", "result": []}}
        with pytest.raises(ValueError, match="matrix"):
            prometheus.from_query_range(response)

    def test_empty_result(self):
        df = prometheus.from_query_range({"data": {"resultType": "matrix", "result": []}})
        assert df.empty
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_metric_without_name(self):
        response = {
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {"foo": "bar"}, "values": [[0, "1.0"]]}],
            }
        }
        df = prometheus.from_query_range(response)
        assert list(df.columns) == ['{foo="bar"}']

    def test_feeds_into_sifter(self):
        # The adapter output (with NaN) must be consumable by Sifter.sift.
        from metricsifter import Sifter

        df = prometheus.from_query_range(_matrix_response())
        result = Sifter(n_jobs=1).sift(df)
        assert result is not None

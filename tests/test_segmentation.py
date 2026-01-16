"""
Test suites for segmentation module
"""

import numpy as np
import pytest

from metricsifter.algo.detection import NO_CHANGE_POINTS
from metricsifter.algo.segmentation import (
    segment_changepoints_with_kde,
    segment_nested_changepoints,
)


class TestSegmentChangepointsWithKDE:
    """Test segment_changepoints_with_kde function"""

    def test_basic_segmentation(self):
        """Basic segmentation"""
        # Changepoints with two clear clusters
        change_points = [10, 12, 14, 50, 52, 54]
        time_series_length = 100

        labels, label_to_cps = segment_changepoints_with_kde(
            change_points, time_series_length, kde_bandwidth=2.5
        )

        # Should detect at least 2 clusters
        assert len(label_to_cps) >= 2, "Should detect at least 2 clusters"

        # All changepoints should belong to some label
        assert len(labels) == len(change_points)

        # Total changepoints in labels should match unique changepoints (unique_values=True)
        all_cps = np.concatenate(list(label_to_cps.values()))
        assert len(all_cps) == len(np.unique(change_points))

    def test_single_cluster(self):
        """Should handle single cluster only"""
        # Only nearby changepoints
        change_points = [50, 51, 52, 53, 54]
        time_series_length = 100

        labels, label_to_cps = segment_changepoints_with_kde(
            change_points, time_series_length, kde_bandwidth=2.5
        )

        # Should detect single cluster
        assert len(label_to_cps) == 1, "Should detect single cluster"
        assert 0 in label_to_cps

    def test_zero_bandwidth_case(self):
        """Should handle when all changepoints are the same (bandwidth 0 case)"""
        # All same changepoint
        change_points = [50, 50, 50, 50]
        time_series_length = 100

        labels, label_to_cps = segment_changepoints_with_kde(
            change_points, time_series_length, kde_bandwidth=2.5
        )

        # Should return single cluster (0)
        assert len(label_to_cps) == 1
        assert 0 in label_to_cps
        # With unique_values=True, should contain only one instance of 50
        assert len(label_to_cps[0]) == 1
        assert label_to_cps[0][0] == 50

    def test_unique_values_false(self):
        """Should preserve duplicate changepoints when unique_values=False"""
        change_points = [50, 50, 51, 51, 52]
        time_series_length = 100

        labels, label_to_cps = segment_changepoints_with_kde(
            change_points, time_series_length, kde_bandwidth=2.5, unique_values=False
        )

        # All changepoints including duplicates should be preserved
        all_cps = np.concatenate(list(label_to_cps.values()))
        assert len(all_cps) == len(change_points)

    def test_different_bandwidth(self):
        """Should work with different bandwidth values"""
        change_points = [10, 12, 14, 50, 52, 54]
        time_series_length = 100

        # Small bandwidth
        _, label_to_cps_small = segment_changepoints_with_kde(
            change_points, time_series_length, kde_bandwidth=1.0
        )

        # Large bandwidth
        _, label_to_cps_large = segment_changepoints_with_kde(
            change_points, time_series_length, kde_bandwidth=10.0
        )

        # Larger bandwidth tends to produce fewer clusters
        assert len(label_to_cps_large) <= len(label_to_cps_small) or len(label_to_cps_large) == len(label_to_cps_small)

    def test_empty_changepoints_raises_assertion(self):
        """Should raise AssertionError for empty changepoints list"""
        with pytest.raises(AssertionError):
            segment_changepoints_with_kde([], time_series_length=100, kde_bandwidth=2.5)


class TestSegmentNestedChangepoints:
    """Test segment_nested_changepoints function"""

    def test_basic_nested_segmentation(self):
        """Basic nested segmentation"""
        # List of changepoints
        flatten_change_points = [10, 12, 14, 50, 52, 54]

        # Mapping of changepoints to metrics
        cp_to_metrics = {
            10: ['metric1', 'metric2'],
            12: ['metric1'],
            14: ['metric2'],
            50: ['metric3', 'metric4'],
            52: ['metric3'],
            54: ['metric4'],
        }

        time_series_length = 100

        label_to_metrics, label_to_cps = segment_nested_changepoints(
            flatten_change_points, cp_to_metrics, time_series_length, kde_bandwidth=2.5
        )

        # Should have at least one label
        assert len(label_to_metrics) > 0

        # NO_CHANGE_POINTS label should be excluded
        assert NO_CHANGE_POINTS not in label_to_metrics

        # All metrics should belong to some label
        all_metrics = set()
        for metrics in label_to_metrics.values():
            all_metrics.update(metrics)

        assert 'metric1' in all_metrics
        assert 'metric2' in all_metrics
        assert 'metric3' in all_metrics
        assert 'metric4' in all_metrics

    def test_with_no_change_points_metrics(self):
        """Should handle metrics with NO_CHANGE_POINTS"""
        flatten_change_points = [10, 12]

        cp_to_metrics = {
            10: ['metric1'],
            12: ['metric1'],
            NO_CHANGE_POINTS: ['metric_constant'],  # Metric with no changepoints
        }

        time_series_length = 100

        label_to_metrics, label_to_cps = segment_nested_changepoints(
            flatten_change_points, cp_to_metrics, time_series_length, kde_bandwidth=2.5
        )

        # Metrics with NO_CHANGE_POINTS should not be included in results
        all_metrics = set()
        for metrics in label_to_metrics.values():
            all_metrics.update(metrics)

        assert 'metric_constant' not in all_metrics
        assert 'metric1' in all_metrics

    def test_multiple_clusters(self):
        """Should handle multiple clusters formation"""
        # Two clearly separated clusters
        flatten_change_points = [10, 11, 12, 80, 81, 82]

        cp_to_metrics = {
            10: ['metric1'],
            11: ['metric1', 'metric2'],
            12: ['metric2'],
            80: ['metric3'],
            81: ['metric3', 'metric4'],
            82: ['metric4'],
        }

        time_series_length = 100

        label_to_metrics, label_to_cps = segment_nested_changepoints(
            flatten_change_points, cp_to_metrics, time_series_length, kde_bandwidth=2.5
        )

        # Should detect multiple clusters
        assert len(label_to_metrics) >= 2

        # Each cluster should have corresponding metrics
        assert len(label_to_cps) >= 2

    def test_single_changepoint(self):
        """Should handle single changepoint only"""
        flatten_change_points = [50]

        cp_to_metrics = {
            50: ['metric1', 'metric2'],
        }

        time_series_length = 100

        label_to_metrics, label_to_cps = segment_nested_changepoints(
            flatten_change_points, cp_to_metrics, time_series_length, kde_bandwidth=2.5
        )

        # Should form single cluster
        assert len(label_to_metrics) == 1
        assert len(label_to_cps) == 1

        # Metrics should be correctly included
        all_metrics = set()
        for metrics in label_to_metrics.values():
            all_metrics.update(metrics)

        assert 'metric1' in all_metrics
        assert 'metric2' in all_metrics

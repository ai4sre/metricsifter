"""
Test suites for the diagnostic SiftResult API (Sifter.sift) and timestamp support.

All synthetic data uses a fixed seed so that the change-point classification is
deterministic and does not depend on the pyrca-based generator.
"""

import json

import numpy as np
import pandas as pd
import pytest

from metricsifter import SegmentInfo, Sifter, SiftResult


def _make_synthetic(as_datetime: bool = False) -> pd.DataFrame:
    """Build data that exercises every exclusion reason.

    - failure_{0,1,2}: level shift at t=60 -> selected (densest segment)
    - unrelated:       level shift at t=20 -> excluded (outside densest segment)
    - noise:           stationary noise    -> excluded (no change points)
    - flat_{0..5}:     constant            -> excluded (no-change filter)
    """
    rng = np.random.default_rng(0)
    length = 80
    data: dict[str, np.ndarray] = {}
    for i in range(3):
        data[f"failure_{i}"] = np.concatenate([rng.normal(0, 0.1, 60), rng.normal(5, 0.1, 20)])
    data["unrelated"] = np.concatenate([rng.normal(0, 0.1, 20), rng.normal(3, 0.1, 60)])
    data["noise"] = rng.normal(0, 0.05, length)
    for i in range(6):
        data[f"flat_{i}"] = np.full(length, float(i))

    df = pd.DataFrame(data)
    if as_datetime:
        df.index = pd.date_range("2024-01-01", periods=length, freq="1min")
    return df


@pytest.fixture
def sifter() -> Sifter:
    return Sifter(penalty_adjust=2.0, n_jobs=1)


class TestExclusionClassification:
    """The three exclusion reasons must be correct and partition the input."""

    def test_reason_sets(self, sifter):
        result = sifter.sift(_make_synthetic())

        assert result.selected_metrics == frozenset({"failure_0", "failure_1", "failure_2"})
        assert result.filtered_no_change == frozenset(f"flat_{i}" for i in range(6))
        assert result.filtered_no_change_points == frozenset({"noise"})
        assert result.filtered_out_of_segment == frozenset({"unrelated"})

    def test_reasons_partition_input(self, sifter):
        data = _make_synthetic()
        result = sifter.sift(data)

        buckets = [
            result.selected_metrics,
            result.filtered_no_change,
            result.filtered_no_change_points,
            result.filtered_out_of_segment,
        ]
        # pairwise disjoint
        for i in range(len(buckets)):
            for j in range(i + 1, len(buckets)):
                assert buckets[i].isdisjoint(buckets[j])
        # exhaustive
        union: frozenset[str] = frozenset().union(*buckets)
        assert union == frozenset(data.columns)

    def test_data_matches_selected_metrics(self, sifter):
        result = sifter.sift(_make_synthetic())
        assert set(result.data.columns) == result.selected_metrics

    def test_segments_and_scores(self, sifter):
        result = sifter.sift(_make_synthetic())

        # Two clusters: t=20 (unrelated) and t=60 (three failure metrics).
        assert len(result.segments) == 2
        assert sum(seg.selected for seg in result.segments) == 1

        selected = result.selected_segment
        assert selected is not None
        assert selected.metrics == frozenset({"failure_0", "failure_1", "failure_2"})
        assert selected.start_index == 60
        assert selected.end_index == 60
        # weighted_max score: three metrics each with a single change point -> 3.0
        assert selected.score == pytest.approx(3.0)
        # The selected segment holds the maximum score.
        assert selected.score == max(seg.score for seg in result.segments)

    def test_no_change_points_result(self, sifter):
        """Constant-only data yields an all-excluded, well-formed result."""
        data = pd.DataFrame({"a": [1.0] * 50, "b": [2.0] * 50})
        result = sifter.sift(data)

        assert result.data.empty
        assert result.selected_segment is None
        assert result.selected_metrics == frozenset()
        assert result.filtered_no_change == frozenset({"a", "b"})
        assert result.segments == []


class TestTimestampSupport:
    """DatetimeIndex input is additionally expressed as wall-clock timestamps."""

    def test_change_times_present_for_datetime_index(self, sifter):
        data = _make_synthetic(as_datetime=True)
        result = sifter.sift(data)

        assert result.metric_to_change_times is not None
        # failure metrics changed at position 60 -> index[60]
        expected = data.index[60]
        for metric in ("failure_0", "failure_1", "failure_2"):
            assert result.metric_to_change_times[metric] == [expected]
        # stationary noise has no change points
        assert result.metric_to_change_times["noise"] == []

    def test_selected_segment_times(self, sifter):
        data = _make_synthetic(as_datetime=True)
        result = sifter.sift(data)

        selected = result.selected_segment
        assert selected is not None
        assert selected.start_time == data.index[selected.start_index]
        assert selected.end_time == data.index[selected.end_index]

    def test_no_times_for_range_index(self, sifter):
        result = sifter.sift(_make_synthetic(as_datetime=False))
        assert result.metric_to_change_times is None
        assert result.selected_segment is not None
        assert result.selected_segment.start_time is None

    def test_irregular_sampling(self, sifter):
        """Non-uniform timestamps map by index reference, without breaking."""
        data = _make_synthetic(as_datetime=True)
        # make the index irregular
        offsets = np.cumsum(np.arange(1, len(data) + 1))
        data.index = pd.Timestamp("2024-01-01") + pd.to_timedelta(offsets, unit="s")

        result = sifter.sift(data)
        selected = result.selected_segment
        assert selected is not None
        assert selected.start_time == data.index[selected.start_index]


class TestSerialization:
    def test_to_json_is_valid_and_consistent(self, sifter):
        result = sifter.sift(_make_synthetic())
        parsed = json.loads(result.to_json())
        assert parsed == result.to_dict()

    def test_dict_roundtrip_range_index(self, sifter):
        result = sifter.sift(_make_synthetic())
        restored = SiftResult.from_dict(result.to_dict())
        assert restored.to_dict() == result.to_dict()
        assert isinstance(restored.selected_segment, SegmentInfo)

    def test_json_roundtrip_datetime_index(self, sifter):
        result = sifter.sift(_make_synthetic(as_datetime=True))
        restored = SiftResult.from_json(result.to_json())
        assert restored.to_dict() == result.to_dict()
        # timestamps survive the round-trip
        assert restored.metric_to_change_times is not None
        assert restored.metric_to_change_times["failure_0"] == result.metric_to_change_times["failure_0"]

    def test_dict_excludes_dataframe(self, sifter):
        d = sifter.sift(_make_synthetic()).to_dict()
        assert "data" not in d
        assert set(d.keys()) == {
            "selected_metrics",
            "excluded",
            "metric_to_change_points",
            "metric_to_change_times",
            "segments",
            "selected_segment",
        }


class TestBackwardCompatibility:
    def test_run_matches_sift_data(self, sifter):
        data = _make_synthetic()
        assert list(sifter.run(data).columns) == list(sifter.sift(data).data.columns)

    def test_run_with_selected_segment_uses_positional_segment(self, sifter):
        from metricsifter.types import Segment

        data = _make_synthetic()
        filtered, segment = sifter.run_with_selected_segment(data)

        assert isinstance(segment, Segment)
        assert isinstance(segment.start_time, int)
        assert segment.start_time == 60
        assert segment.end_time == 60
        assert 0 <= segment.start_time <= segment.end_time < len(data)
        assert filtered.shape[1] == 3

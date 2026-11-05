"""
Test suites for the diagnostic SiftResult API (Sifter.sift) and timestamp support.

All synthetic data uses a fixed seed so that the change-point classification is
deterministic and does not depend on the pyrca-based generator.
"""

import json

import numpy as np
import pandas as pd
import pytest

from metricsifter import BandwidthTuning, PenaltyTuning, SegmentInfo, Sifter, SiftResult
from metricsifter.algo import segmentation


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

    def test_no_change_points_result_keeps_detection_report(self):
        """The early return still reports per-metric change points and times."""
        data = pd.DataFrame(
            {"a": [1.0] * 50, "b": [2.0] * 50}, index=pd.date_range("2024-01-01", periods=50, freq="1min")
        )
        result = Sifter(penalty_adjust=2.0, n_jobs=1).sift(data, without_simple_filter=True)

        assert result.data.empty
        assert list(result.data.columns) == []
        assert len(result.data.index) == 0
        assert result.filtered_no_change == frozenset()
        assert result.filtered_no_change_points == frozenset({"a", "b"})
        assert result.filtered_out_of_segment == frozenset()
        assert result.metric_to_change_points == {"a": [], "b": []}
        assert result.metric_to_change_times == {"a": [], "b": []}
        assert result.selected_segment is None
        assert result.penalty_tuning is None
        assert result.bandwidth_tuning is None

    def test_empty_clusters_are_not_reported_as_segments(self, sifter, monkeypatch):
        """A KDE cluster without change points yields no SegmentInfo."""
        real_segment = segmentation.segment_nested_changepoints

        def with_empty_cluster(**kwargs):
            label_to_metrics, label_to_change_points = real_segment(**kwargs)
            label_to_change_points[-5] = np.array([], dtype=int)
            label_to_change_points[99] = np.array([], dtype=int)
            return label_to_metrics, label_to_change_points

        monkeypatch.setattr(segmentation, "segment_nested_changepoints", with_empty_cluster)
        result = sifter.sift(_make_synthetic())

        assert [seg.label for seg in result.segments] == [0, 1]
        assert result.selected_segment is not None
        assert result.selected_segment.metrics == frozenset({"failure_0", "failure_1", "failure_2"})
        assert result.selected_metrics == frozenset({"failure_0", "failure_1", "failure_2"})


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

    def test_to_dict_exact_layout_when_fully_populated(self):
        t0 = pd.Timestamp("2024-01-01 00:00")
        minute = pd.Timedelta(minutes=1)
        first = SegmentInfo(label=0, metrics=frozenset({"b", "a"}), start_index=3, end_index=7, score=1.5)
        second = SegmentInfo(
            label=1,
            metrics=frozenset({"c"}),
            start_index=10,
            end_index=12,
            start_time=t0 + 10 * minute,
            end_time=t0 + 12 * minute,
            score=2.0,
            selected=True,
        )
        result = SiftResult(
            data=pd.DataFrame({"c": [1.0]}),
            selected_metrics=frozenset({"c"}),
            filtered_no_change=frozenset({"z", "y"}),
            filtered_no_change_points=frozenset({"n"}),
            filtered_out_of_segment=frozenset({"b", "a"}),
            metric_to_change_points={"b": [np.int64(3)], "a": [7], "c": [10, 12]},
            metric_to_change_times={"b": [t0 + 3 * minute], "a": [t0 + 7 * minute], "c": [t0 + 10 * minute]},
            segments=[first, second],
            selected_segment=second,
            penalty_tuning=PenaltyTuning(
                requested="auto",
                resolved=2.0,
                grid=[1.0, 2.0],
                n_change_points=[3, 2],
                adjacent_jaccard=[0.5],
                plateau=(1.0, 2.0),
                reason="plateau",
            ),
            bandwidth_tuning=BandwidthTuning(
                requested="auto",
                resolved=1.0,
                grid=[1.0, 2.0],
                stability=[0.75, None],
                n_segments=[2, 1],
                reason="stability",
            ),
        )

        d = result.to_dict()

        second_dict = {
            "label": 1,
            "metrics": ["c"],
            "start_index": 10,
            "end_index": 12,
            "start_time": "2024-01-01T00:10:00",
            "end_time": "2024-01-01T00:12:00",
            "score": 2.0,
            "selected": True,
        }
        assert d == {
            "selected_metrics": ["c"],
            "excluded": {"no_change_filter": ["y", "z"], "no_change_points": ["n"], "out_of_segment": ["a", "b"]},
            "metric_to_change_points": {"b": [3], "a": [7], "c": [10, 12]},
            "metric_to_change_times": {
                "b": ["2024-01-01T00:03:00"],
                "a": ["2024-01-01T00:07:00"],
                "c": ["2024-01-01T00:10:00"],
            },
            "segments": [
                {
                    "label": 0,
                    "metrics": ["a", "b"],
                    "start_index": 3,
                    "end_index": 7,
                    "start_time": None,
                    "end_time": None,
                    "score": 1.5,
                    "selected": False,
                },
                second_dict,
            ],
            "selected_segment": second_dict,
            "penalty_tuning": {
                "requested": "auto",
                "resolved": 2.0,
                "grid": [1.0, 2.0],
                "n_change_points": [3, 2],
                "adjacent_jaccard": [0.5],
                "plateau": [1.0, 2.0],
                "reason": "plateau",
            },
            "bandwidth_tuning": {
                "requested": "auto",
                "resolved": 1.0,
                "grid": [1.0, 2.0],
                "stability": [0.75, None],
                "n_segments": [2, 1],
                "reason": "stability",
            },
        }
        # Key order is part of the JSON output, and numpy scalars become plain ints.
        assert list(d) == [
            "selected_metrics",
            "excluded",
            "metric_to_change_points",
            "metric_to_change_times",
            "segments",
            "selected_segment",
            "penalty_tuning",
            "bandwidth_tuning",
        ]
        assert list(d["metric_to_change_points"]) == ["b", "a", "c"]
        assert list(d["metric_to_change_times"]) == ["b", "a", "c"]
        assert type(d["metric_to_change_points"]["b"][0]) is int

    def test_to_dict_optional_fields_none(self):
        result = SiftResult(
            data=None,
            selected_metrics=frozenset(),
            filtered_no_change=frozenset(),
            filtered_no_change_points=frozenset(),
            filtered_out_of_segment=frozenset(),
        )

        assert result.to_dict() == {
            "selected_metrics": [],
            "excluded": {"no_change_filter": [], "no_change_points": [], "out_of_segment": []},
            "metric_to_change_points": {},
            "metric_to_change_times": None,
            "segments": [],
            "selected_segment": None,
            "penalty_tuning": None,
            "bandwidth_tuning": None,
        }

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
            "penalty_tuning",
            "bandwidth_tuning",
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

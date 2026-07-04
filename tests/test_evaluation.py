"""Tests for the lightweight evaluation utility."""

import pytest

from metricsifter import SelectionMetrics, evaluate_selection


class TestEvaluateSelection:
    def test_known_values(self):
        selected = {"a", "b", "c"}
        ground_truth = {"a", "b", "d"}
        # tp=2 (a,b), fp=1 (c), fn=1 (d)
        m = evaluate_selection(selected, ground_truth)
        assert m.precision == pytest.approx(2 / 3)
        assert m.recall == pytest.approx(2 / 3)
        assert m.f1 == pytest.approx(2 / 3)
        assert m.reduction_ratio is None

    def test_perfect_selection(self):
        gt = {"a", "b"}
        m = evaluate_selection({"a", "b"}, gt)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_reduction_ratio(self):
        all_metrics = {"a", "b", "c", "d", "e"}
        m = evaluate_selection({"a", "b"}, {"a", "b"}, all_metrics=all_metrics)
        # kept 2 of 5 -> dropped 3/5
        assert m.reduction_ratio == pytest.approx(3 / 5)

    def test_reduction_ratio_ignores_unknown_selected(self):
        # A selected name absent from all_metrics must not push the ratio negative.
        m = evaluate_selection({"a", "b", "ghost"}, {"a"}, all_metrics={"a", "b", "c"})
        # kept ∩ all = {a, b} -> dropped 1/3
        assert m.reduction_ratio == pytest.approx(1 / 3)

    def test_empty_selected(self):
        m = evaluate_selection(set(), {"a", "b"}, all_metrics={"a", "b", "c"})
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.reduction_ratio == pytest.approx(1.0)  # kept nothing

    def test_empty_ground_truth(self):
        m = evaluate_selection({"a"}, set())
        assert m.precision == 0.0  # tp=0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_all_empty(self):
        m = evaluate_selection(set(), set(), all_metrics=set())
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.reduction_ratio == 0.0  # zero division defined as 0.0

    def test_accepts_arbitrary_iterables(self):
        # Non-set iterables are coerced internally.
        m = evaluate_selection(["a", "a", "b"], ["a"], all_metrics=["a", "b"])
        assert m.precision == pytest.approx(0.5)
        assert m.recall == 1.0

    def test_to_dict(self):
        m = SelectionMetrics(precision=1.0, recall=0.5, f1=0.6666666666666666, reduction_ratio=0.25)
        assert m.to_dict() == {
            "precision": 1.0,
            "recall": 0.5,
            "f1": 0.6666666666666666,
            "reduction_ratio": 0.25,
        }

"""Tests for the optional matplotlib-based visualization API.

Rendering uses the non-interactive ``Agg`` backend so no display is required.
"""

import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from metricsifter import SegmentInfo, Sifter, SiftResult, plot  # noqa: E402
from tests.conftest import make_synthetic  # noqa: E402


@pytest.fixture
def result_and_data():
    data = make_synthetic()
    result = Sifter(penalty_adjust=2.0, n_jobs=1).sift(data)
    return result, data


class TestPlotSiftedMetrics:
    def test_returns_two_panel_figure(self, result_and_data):
        result, data = result_and_data
        fig = plot.plot_sifted_metrics(result, data)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        ax_before, ax_after = fig.axes
        n_markers = len(plot._change_point_positions(result))
        # Before panel draws every input metric plus one marker line per change point.
        assert len(ax_before.lines) == data.shape[1] + n_markers
        # After panel draws only the selected metrics (labeled), plus markers.
        labeled = [ln for ln in ax_after.lines if ln.get_label() in result.selected_metrics]
        assert len(labeled) == len(result.selected_metrics) == 3

    def test_change_point_markers_and_segment_band(self, result_and_data):
        result, data = result_and_data
        fig = plot.plot_sifted_metrics(result, data)
        ax_after = fig.axes[1]
        # At least one vertical change-point marker line drawn (axvline adds lines).
        # Selected metric lines are 3; the extra lines are the change-point markers.
        assert len(ax_after.lines) > len(result.selected_metrics)
        # The selected segment is shaded as a band (a Polygon patch from axvspan).
        assert len(ax_after.patches) >= 1

    def test_markers_band_and_legend_on_both_panels(self, result_and_data):
        result, data = result_and_data
        fig = plot.plot_sifted_metrics(result, data)
        ax_before, ax_after = fig.axes
        positions = plot._change_point_positions(result)

        for ax, n_series in ((ax_before, data.shape[1]), (ax_after, len(result.selected_metrics))):
            markers = ax.lines[n_series:]
            assert [ln.get_xdata()[0] for ln in markers] == [data.index[p] for p in positions]
            assert len(ax.patches) == 1
        assert ax_before.get_legend() is None
        assert sorted(t.get_text() for t in ax_after.get_legend().get_texts()) == sorted(result.selected_metrics)
        assert ax_before.get_title() == f"Before sift ({data.shape[1]} metrics)"
        assert ax_after.get_title() == "After sift (3 metrics)"
        assert ax_after.get_xlabel() == "time"

    def test_no_selection_draws_no_legend_band_or_markers(self):
        data = pd.DataFrame({"a": [1.0] * 30, "b": [2.0] * 30})
        result = Sifter(n_jobs=1).sift(data)
        fig = plot.plot_sifted_metrics(result, data)
        ax_before, ax_after = fig.axes

        assert len(ax_before.lines) == 2
        assert len(ax_after.lines) == 0
        assert ax_after.get_title() == "After sift (0 metrics)"
        assert ax_after.get_legend() is None
        assert len(ax_before.patches) == len(ax_after.patches) == 0

    def test_out_of_range_positions_and_segment_are_skipped(self):
        data = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)[::-1]})
        segment = SegmentInfo(label=0, metrics=frozenset({"a"}), start_index=3, end_index=12, selected=True)
        result = SiftResult(
            data=None,
            selected_metrics=frozenset({"a"}),
            filtered_no_change=frozenset(),
            filtered_no_change_points=frozenset(),
            filtered_out_of_segment=frozenset(),
            metric_to_change_points={"a": [-1, 3, 10], "b": [9, 25]},
            segments=[segment],
            selected_segment=segment,
        )
        fig = plot.plot_sifted_metrics(result, data)
        ax_before, ax_after = fig.axes

        # Only positions 3 and 9 lie on the 10-row axis.
        assert [ln.get_xdata()[0] for ln in ax_before.lines[2:]] == [3, 9]
        assert [ln.get_xdata()[0] for ln in ax_after.lines[1:]] == [3, 9]
        # The segment ends past the axis, so no band is drawn.
        assert len(ax_before.patches) == len(ax_after.patches) == 0

    def test_segment_band_spans_the_selected_segment(self):
        data = pd.DataFrame({"a": np.arange(10.0)})
        segment = SegmentInfo(label=0, metrics=frozenset({"a"}), start_index=2, end_index=5, selected=True)
        result = SiftResult(
            data=None,
            selected_metrics=frozenset({"a"}),
            filtered_no_change=frozenset(),
            filtered_no_change_points=frozenset(),
            filtered_out_of_segment=frozenset(),
            metric_to_change_points={"a": [2, 5]},
            segments=[segment],
            selected_segment=segment,
        )
        fig = plot.plot_sifted_metrics(result, data)

        for ax in fig.axes:
            (band,) = ax.patches
            # Data-x extent of the band (a Rectangle or a Polygon, by matplotlib version).
            xs = band.get_patch_transform().transform(band.get_path().vertices)[:, 0]
            assert (xs.min(), xs.max()) == (2.0, 5.0)


class TestPlotChangePointDensity:
    def test_returns_axes_with_segments_and_kde(self, result_and_data):
        result, data = result_and_data
        ax = plot.plot_change_point_density(result, time_series_length=len(data))

        assert isinstance(ax, Axes)
        # One shaded band per candidate segment.
        assert len(ax.patches) == len(result.segments)
        # A twin axis carrying the KDE curve was created.
        twins = [a for a in ax.figure.axes if a is not ax]
        assert len(twins) == 1
        assert len(twins[0].lines) == 1


class TestMatplotlibOptional:
    def test_import_error_has_install_hint(self, result_and_data, monkeypatch):
        result, data = result_and_data
        # Simulate matplotlib not being importable.
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

        with pytest.raises(ImportError) as excinfo:
            plot.plot_sifted_metrics(result, data)
        assert "metricsifter[viz]" in str(excinfo.value)

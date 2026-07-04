"""Tests for the optional matplotlib-based visualization API.

Rendering uses the non-interactive ``Agg`` backend so no display is required.
"""

import sys

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from metricsifter import Sifter, plot  # noqa: E402
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

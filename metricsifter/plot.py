"""Optional visualization helpers for :class:`~metricsifter.types.SiftResult`.

matplotlib is an **optional** dependency. Import this module (or call any of its
functions) without matplotlib installed and you get a clear, actionable
``ImportError`` telling you to ``pip install 'metricsifter[viz]'``. matplotlib is
never pulled into the core install.

All functions take a :class:`SiftResult` and return matplotlib artists
(``Axes`` or ``Figure``) so callers stay in control of styling and saving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from metricsifter.algo import segmentation
from metricsifter.types import SiftResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_INSTALL_HINT = (
    "matplotlib is required for metricsifter.plot but is not installed. "
    "Install the optional visualization extra with: pip install 'metricsifter[viz]'"
)


def _require_matplotlib():
    """Import matplotlib lazily, raising a clear install hint on failure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(_INSTALL_HINT) from exc
    return plt


def _change_point_positions(result: SiftResult) -> list[int]:
    """Flatten every per-metric change point into a sorted position list."""
    positions: set[int] = set()
    for cps in result.metric_to_change_points.values():
        positions.update(int(cp) for cp in cps)
    return sorted(positions)


def plot_sifted_metrics(
    result: SiftResult,
    original_data: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10, 6),
) -> "Figure":
    """Plot the before/after time series of a sift on stacked panels.

    Top panel shows every input metric; the bottom panel shows only the
    selected metrics. Detected change points are drawn as vertical marker lines
    and the selected (densest) segment range is shaded as a band on both panels.

    Args:
        result: The :class:`SiftResult` returned by :meth:`Sifter.sift`.
        original_data: The DataFrame that was passed to ``sift`` (used to draw
            the "before" panel and to look up the x-axis).
        figsize: Figure size forwarded to matplotlib.

    Returns:
        matplotlib ``Figure`` with two stacked ``Axes``.
    """
    plt = _require_matplotlib()

    fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    x = original_data.index
    for column in original_data.columns:
        ax_before.plot(x, original_data[column].to_numpy(), linewidth=0.8, alpha=0.7)
    ax_before.set_title(f"Before sift ({original_data.shape[1]} metrics)")

    selected = [c for c in original_data.columns if c in result.selected_metrics]
    for column in selected:
        ax_after.plot(x, original_data[column].to_numpy(), linewidth=1.0, label=str(column))
    ax_after.set_title(f"After sift ({len(selected)} metrics)")
    if selected:
        ax_after.legend(loc="upper left", fontsize="small")

    # Change-point markers and the selected-segment band, on both panels.
    positions = _change_point_positions(result)
    for ax in (ax_before, ax_after):
        for pos in positions:
            if 0 <= pos < len(x):
                ax.axvline(x[pos], color="tab:red", linestyle="--", linewidth=0.6, alpha=0.5)
        if result.selected_segment is not None:
            start = result.selected_segment.start_index
            end = result.selected_segment.end_index
            if 0 <= start < len(x) and 0 <= end < len(x):
                ax.axvspan(x[start], x[end], color="tab:orange", alpha=0.2)

    ax_after.set_xlabel("time")
    fig.tight_layout()
    return fig


def plot_change_point_density(
    result: SiftResult,
    time_series_length: int,
    *,
    kde_bandwidth: float | str = 2.5,
    ax: "Axes | None" = None,
    figsize: tuple[float, float] = (10, 4),
) -> "Axes":
    """Plot a change-point lag plot with segment boundaries and the KDE curve.

    Each detected change point is drawn as a rug/lag marker along the time axis.
    Segment boundaries (the span of each candidate segment) are drawn as shaded
    bands, with the selected segment highlighted. When the change points have
    non-zero variance, the same KDE density curve used internally by
    :mod:`metricsifter.algo.segmentation` is overlaid on a twin y-axis.

    Args:
        result: The :class:`SiftResult` returned by :meth:`Sifter.sift`.
        time_series_length: Length of the original series (KDE evaluation grid).
        kde_bandwidth: Bandwidth for the overlaid KDE curve. Use the same value
            passed to :class:`Sifter` (``bandwidth``) to match the segmentation.
        ax: Existing ``Axes`` to draw on. A new figure is created if ``None``.
        figsize: Figure size used when ``ax`` is ``None``.

    Returns:
        The matplotlib ``Axes`` containing the lag plot.
    """
    plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    positions = _change_point_positions(result)

    # Segment bands first so markers/curve draw on top.
    for segment in result.segments:
        color = "tab:orange" if segment.selected else "tab:blue"
        alpha = 0.25 if segment.selected else 0.1
        ax.axvspan(segment.start_index, segment.end_index, color=color, alpha=alpha)

    # Lag/rug markers for every change point.
    if positions:
        ax.plot(positions, [0.0] * len(positions), "o", color="tab:red", label="change points")

    ax.set_xlim(0, max(time_series_length - 1, 1))
    ax.set_xlabel("time (row position)")
    ax.set_ylabel("change points")
    ax.set_yticks([])

    # Overlay the internal KDE density curve when it is well-defined.
    density = segmentation.compute_kde_density(positions, time_series_length, kde_bandwidth)
    if density is not None:
        s, e = density
        ax_density = ax.twinx()
        ax_density.plot(s, e, color="tab:green", linewidth=1.5, label="KDE density")
        ax_density.set_ylabel("KDE density")

    ax.set_title("Change-point density and segments")
    return ax

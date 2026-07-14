from collections import defaultdict
from typing import Callable, Final

import numpy as np
import numpy.typing as npt
import scipy.signal
from statsmodels.nonparametric.kde import KDEUnivariate

from metricsifter.algo.detection import NO_CHANGE_POINTS

#: Bandwidth used when the ``"auto"`` stability selection cannot run or finds
#: no admissible candidate (matches the historical default).
BANDWIDTH_FALLBACK: Final[float] = 2.5

#: Number of bootstrap resamples per candidate bandwidth.
N_BOOTSTRAP: Final[int] = 20

#: Minimum distinct change points required to attempt bandwidth tuning.
MIN_UNIQUE_CHANGE_POINTS: Final[int] = 3


def segment_nested_changepoints(
    flatten_change_points: list[int],
    cp_to_metrics: dict[int, list[str]],
    time_series_length: int,
    kde_bandwidth: float | str = 2.5,
) -> tuple[dict[int, set[str]], dict[int, npt.NDArray]]:
    _, label_to_change_points = segment_changepoints_with_kde(
        flatten_change_points,
        time_series_length=time_series_length,
        kde_bandwidth=kde_bandwidth,
        unique_values=True,
    )

    label_to_metrics: dict[int, set[str]] = defaultdict(set)
    for label, cps in label_to_change_points.items():
        if label == NO_CHANGE_POINTS:  # skip no anomaly metrics
            continue
        for cp in cps:
            for metric in cp_to_metrics[cp]:
                label_to_metrics[label].add(metric)
    return label_to_metrics, label_to_change_points


def compute_kde_density(
    change_points: list[int],
    time_series_length: int,
    kde_bandwidth: str | float = 2.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Evaluate the change-point KDE on the time axis (for visualization).

    This mirrors the exact density computation performed inside
    :func:`segment_changepoints_with_kde` so that plots can overlay the same
    curve the segmentation logic reasons about. It is a pure, additive helper:
    the internal segmentation flow is unchanged.

    Returns ``(s, e)`` where ``s`` is the evaluation grid (row positions) and
    ``e`` is the estimated density, or ``None`` when a density cannot be formed
    (empty input or zero-variance change points, e.g. a single unique value).
    """
    if len(change_points) == 0:
        return None
    x = np.array(change_points, dtype=int)
    if x.std() == 0.0:
        return None
    dens = KDEUnivariate(x)
    dens.fit(kernel="gau", bw=kde_bandwidth, fft=True)
    s = np.linspace(start=0, stop=time_series_length - 1, num=time_series_length)
    e = dens.evaluate(s)
    return s, e


def segment_changepoints_with_kde(
    change_points: list[int],
    time_series_length: int,
    kde_bandwidth: str | float,
    unique_values: bool = True,
) -> tuple[np.ndarray, dict[int, npt.NDArray]]:
    if len(change_points) == 0:
        raise ValueError("change_points should not be empty")

    x = np.array(change_points, dtype=int)
    if x.std() == 0.0:  # Handling the case where there is bandwidth 0.
        return np.zeros(len(x), dtype=int), {
            0: np.unique(x) if unique_values else x
        }  # the all change points belongs to cluster 0.

    dens = KDEUnivariate(x)
    dens.fit(kernel="gau", bw=kde_bandwidth, fft=True)
    s = np.linspace(start=0, stop=time_series_length - 1, num=time_series_length)
    e = dens.evaluate(s)

    mi = scipy.signal.argrelextrema(e, np.less)[0]
    clusters = []
    if len(mi) <= 0:
        clusters.append(np.arange(len(x)))
    else:
        clusters.append(np.where(x < s[mi][0])[0])  # most left cluster
        for i_cluster in range(len(mi) - 1):  # all middle cluster
            clusters.append(np.where((x >= s[mi][i_cluster]) & (x < s[mi][i_cluster + 1]))[0])
        clusters.append(np.where(x >= s[mi][-1])[0])  # most right cluster

    labels = np.zeros(len(x), dtype=int)
    for label, x_args in enumerate(clusters):
        clusters[label] = np.unique(x[x_args]) if unique_values else x[x_args]
        labels[x_args] = label
    label_to_values: dict[int, np.ndarray] = {label: vals for label, vals in enumerate(clusters)}
    return labels, label_to_values


def _bandwidth_grid(time_series_length: int) -> list[float]:
    """Geometric candidate grid from fine (1.0) to coarse (~10% of the series)."""
    upper = max(8.0, 0.1 * time_series_length)
    return [float(h) for h in np.geomspace(1.0, upper, num=12)]


def _rebuild_aggregates(
    sampled_metrics: list[str], metric_to_cps: dict[str, list[int]]
) -> tuple[list[int], dict[int, list[str]]]:
    """Rebuild ``(flatten_change_points, cp_to_metrics)`` from a bootstrap sample.

    A metric drawn multiple times contributes its change points that many times
    to the flattened list (raising its KDE density weight, as in the full run),
    while ``cp_to_metrics`` stays name-based so the segment membership keeps set
    semantics.
    """
    flatten: list[int] = []
    cp_to_metrics: dict[int, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for metric in sampled_metrics:
        cps = metric_to_cps[metric]
        flatten.extend(cps)
        if metric not in seen:
            seen.add(metric)
            for cp in cps:
                cp_to_metrics[cp].append(metric)
    return flatten, cp_to_metrics


def _mean_pairwise_jaccard(sets: list[frozenset[str]]) -> float:
    total = 0.0
    n_pairs = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            total += len(sets[i] & sets[j]) / len(union) if union else 1.0
            n_pairs += 1
    return total / n_pairs if n_pairs else 1.0


def select_bandwidth(
    flatten_change_points: list[int],
    cp_to_metrics: dict[int, list[str]],
    metric_to_cps: dict[str, list[int]],
    time_series_length: int,
    selector: Callable[[dict[int, set[str]], dict[str, list[int]], dict[int, npt.NDArray]], tuple],
    random_state: int | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    grid: list[float] | None = None,
) -> tuple[float, dict]:
    """Pick the KDE bandwidth by bootstrap stability of the final selection.

    The segmentation aggregates the change points of *all* metrics, so the
    natural perturbation unit is the metric ensemble: for every candidate
    bandwidth, the metrics (those with at least one change point) are resampled
    with replacement ``n_bootstrap`` times, each resample is segmented and run
    through the caller's segment ``selector``, and the candidate is scored by
    the mean pairwise Jaccard similarity of the resulting selected-metric sets.
    The same resample indices are shared across all candidates (a paired
    design: differences in score reflect the bandwidth, not resampling luck).

    A candidate is only admissible when the segmentation of the **full** data
    yields at least two segments -- otherwise a large-enough bandwidth always
    wins with a trivially perfect score by merging everything into one segment.
    Ties prefer the smaller bandwidth (finer granularity is more selective,
    matching the feature-reduction goal).

    Args:
        flatten_change_points: Change points of all metrics, with multiplicity.
        cp_to_metrics: Change point -> metric names (from detection).
        metric_to_cps: Metric name -> its change points (from detection).
        time_series_length: Length of the time axis.
        selector: The segment-selection routine, called as
            ``selector(label_to_metrics, metric_to_cps, label_to_change_points)``
            and returning ``(label, selected_metrics)`` --
            :meth:`metricsifter.sifter.Sifter.select_largest_segment_with_label`
            has exactly this shape.
        random_state: Seed for the bootstrap resampling (``None`` = OS entropy).
        n_bootstrap: Number of resamples per candidate.
        grid: Candidate bandwidths (default: :func:`_bandwidth_grid`).

    Returns:
        ``(resolved, diagnostics)`` where diagnostics carries ``grid``,
        ``stability`` (``None`` for inadmissible candidates), ``n_segments``
        and ``reason``.
    """
    unique_cps = set(flatten_change_points)
    if len(unique_cps) < MIN_UNIQUE_CHANGE_POINTS:
        return BANDWIDTH_FALLBACK, {
            "grid": [],
            "stability": [],
            "n_segments": [],
            "reason": "too_few_change_points",
        }

    grid = grid if grid is not None else _bandwidth_grid(time_series_length)
    metrics = sorted(metric for metric, cps in metric_to_cps.items() if len(cps) > 0)
    n_metrics = len(metrics)
    rng = np.random.default_rng(random_state)
    # One shared set of resample indices for every candidate bandwidth.
    bootstrap_indices = [rng.integers(0, n_metrics, size=n_metrics) for _ in range(n_bootstrap)]

    stability: list[float | None] = []
    n_segments: list[int] = []
    best: tuple[tuple[float, float], float] | None = None  # ((stab, -h), h)
    for h in grid:
        _, label_to_cps_full = segment_nested_changepoints(
            flatten_change_points, cp_to_metrics, time_series_length, kde_bandwidth=h
        )
        n_seg = sum(1 for cps in label_to_cps_full.values() if len(cps) > 0)
        n_segments.append(n_seg)
        if n_seg < 2:
            stability.append(None)
            continue

        selected_sets: list[frozenset[str]] = []
        for indices in bootstrap_indices:
            sampled = [metrics[i] for i in indices]
            flatten_b, cp_to_metrics_b = _rebuild_aggregates(sampled, metric_to_cps)
            label_to_metrics_b, label_to_cps_b = segment_nested_changepoints(
                flatten_b, cp_to_metrics_b, time_series_length, kde_bandwidth=h
            )
            _, selected = selector(label_to_metrics_b, metric_to_cps, label_to_cps_b)
            selected_sets.append(frozenset(selected))
        score = _mean_pairwise_jaccard(selected_sets)
        stability.append(score)
        candidate = ((score, -h), h)
        if best is None or candidate[0] > best[0]:
            best = candidate

    diagnostics: dict = {"grid": list(grid), "stability": stability, "n_segments": n_segments}
    if best is None:
        diagnostics["reason"] = "unimodal"
        return BANDWIDTH_FALLBACK, diagnostics
    diagnostics["reason"] = "stability"
    return best[1], diagnostics

from collections import defaultdict

import numpy as np
import numpy.typing as npt
import scipy.signal
from statsmodels.nonparametric.kde import KDEUnivariate

from metricsifter.algo.detection import NO_CHANGE_POINTS


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


def segment_changepoints_with_kde(
    change_points: list[int],
    time_series_length: int,
    kde_bandwidth: str | float,
    unique_values: bool = True,
) -> tuple[np.ndarray, dict[int, npt.NDArray]]:
    assert len(change_points) > 0, "change_points should not be empty"

    x = np.array(change_points, dtype=int)
    if x.std() == .0:  # Handling the case where there is bandwidth 0.
        return np.zeros(len(x), dtype=int), {0: np.unique(x) if unique_values else x}  # the all change points belongs to cluster 0.

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
            clusters.append(np.where((x >= s[mi][i_cluster]) * (x <= s[mi][i_cluster + 1]))[0])
        clusters.append(np.where(x >= s[mi][-1])[0])  # most right cluster

    labels = np.zeros(len(x), dtype=int)
    for label, x_args in enumerate(clusters):
        clusters[label] = np.unique(x[x_args]) if unique_values else x[x_args]
        labels[x_args] = label
    label_to_values: dict[int, np.ndarray] = {label: vals for label, vals in enumerate(clusters)}
    return labels, label_to_values

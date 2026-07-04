import warnings
from collections import defaultdict
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import ruptures as rpt
from joblib import Parallel, delayed
from ruptures.exceptions import BadSegmentationParameters

NO_CHANGE_POINTS: Final[int] = -1


def _detect_changepoints_with_missing_values(x: np.ndarray) -> npt.ArrayLike:
    """
    Detect changepoints with missing values
    Sample:
        input: [1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9, np.nan, np.nan])
        output: [2 6 9]
    """
    is_nan = np.isnan(x)
    # Get the index where NaN value changes
    nans = np.where(is_nan, 1, 0)
    change_indexes = np.where(np.diff(nans) == 1)[0] + 1
    # If array starts with NaN, add 0 to indices
    if is_nan[0]:
        change_indexes = np.concatenate((np.array([0]), change_indexes))
    return change_indexes


def detect_univariate_changepoints(
    x: np.ndarray, search_method: str, cost_model: str, penalty: str | float, penalty_adjust: float
) -> list[int]:
    """Detect change points in a single metric, robust to missing values (NaN).

    NaN handling policy (all returned indices refer to positions in the original ``x``):

    * **Leading / trailing NaN** runs carry no in-window signal, so they are
      trimmed before change point detection. The left trim length is kept as an
      offset to map detected indices back to their original positions.
    * **Interior NaN** gaps are filled by **linear interpolation** before the
      detector runs. Interpolation is chosen over masking because it preserves
      the length of the series, so the change point indices produced by
      ``ruptures`` map back to the original axis with a single additive offset
      (masking would require a per-element index table and is more error-prone).
      Linear interpolation is also a defensible imputation for monitoring
      metrics, where a value between two observations is approximately linear.
    * The **penalty** is derived from ``np.nanstd`` of the interpolated core so
      that residual NaN can never turn ``sigma`` (and therefore the penalty)
      into ``NaN`` -- the previous ``np.std(x)`` produced ``NaN`` on any input
      containing missing values and broke the ``predict(pen=...)`` call.
    * Independently, the boundaries where a metric **goes missing** are treated
      as change points in their own right (see
      :func:`_detect_changepoints_with_missing_values`) and unioned with the
      detector output, because a metric that stops reporting is itself a signal.

    For inputs without any NaN this function is behaviorally identical to the
    original implementation (no trimming, ``core == x``, ``nanstd == std``).
    """
    missing_value_cps = {int(i) for i in _detect_changepoints_with_missing_values(x)}

    is_nan = np.isnan(x)
    if is_nan.all():
        # Nothing to segment; only the missing-value boundaries remain.
        return sorted(missing_value_cps)

    # Trim leading/trailing NaN and remember the left offset for index remapping.
    valid_positions = np.where(~is_nan)[0]
    left, right = int(valid_positions[0]), int(valid_positions[-1])
    core = np.asarray(x[left : right + 1], dtype=float).copy()

    # Linearly interpolate interior NaN so the axis length (hence indices) is preserved.
    core_is_nan = np.isnan(core)
    if core_is_nan.any():
        idx = np.arange(core.size)
        core[core_is_nan] = np.interp(idx[core_is_nan], idx[~core_is_nan], core[~core_is_nan])

    if core.size < 2:
        # KernelCPD requires at least min_size (=2) samples; only NaN boundaries remain.
        return sorted(missing_value_cps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        match search_method:
            case "pelt":
                searcher = rpt.KernelCPD(kernel="linear", min_size=2, jump=1)  # written in C lang
            case "binseg":
                searcher = rpt.Binseg(model=cost_model, jump=1)
            case "bottomup":
                searcher = rpt.BottomUp(model=cost_model, jump=1)
            case _:
                raise ValueError(f"search_method={search_method} is not supported.")
    sigma = np.nanstd(core)
    match penalty:
        case "aic":
            pen = sigma * sigma
        case "bic":
            pen = np.log(core.size) * sigma * sigma
        case _:
            pen = float(penalty)
    try:
        cps = searcher.fit(core).predict(pen=pen * penalty_adjust)
    except BadSegmentationParameters:
        # The core is too short for the detector to place any break (e.g. a
        # 2-3 sample series after NaN trimming); only NaN boundaries remain.
        return sorted(missing_value_cps)
    if cps is None:
        raise ValueError("Change point detection failed: predict() returned None.")
    cps = cps[:-1]  # remove the last index (== series length)
    # Map core-relative indices back onto the original series before unioning.
    remapped_cps = {int(cp) + left for cp in cps}
    return sorted(remapped_cps | missing_value_cps)


def detect_multi_changepoints(
    X: pd.DataFrame,
    search_method: str,
    cost_model: str,
    penalty: str | float,
    penalty_adjust: float,
    n_jobs: int = -1,
) -> tuple[list[int], dict[int, list[str]], dict[str, list[int]]]:
    metrics: list[str] = X.columns.tolist()
    multi_change_points = Parallel(n_jobs=n_jobs)(
        delayed(detect_univariate_changepoints)(
            X[metric].to_numpy(), search_method, cost_model, penalty, penalty_adjust
        )
        for metric in metrics
    )
    cp_to_metrics: dict[int, list[str]] = defaultdict(list)
    for metric, change_points in zip(metrics, multi_change_points):
        if change_points is None or len(change_points) < 1:
            cp_to_metrics[NO_CHANGE_POINTS].append(metric)  # cp == -1 means no change point
            continue
        for cp in change_points:
            cp_to_metrics[cp].append(metric)

    flatten_change_points: list[int] = sum(multi_change_points, [])
    metric_to_cps = {metric: cps for metric, cps in zip(metrics, multi_change_points) if cps is not None}

    return flatten_change_points, cp_to_metrics, metric_to_cps

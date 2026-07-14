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

#: Noise-scale (``sigma``) estimators supported by :func:`detect_univariate_changepoints`.
SIGMA_ESTIMATORS: Final[frozenset[str]] = frozenset({"std", "mad", "diff_std"})

#: Change-point search algorithms supported by :func:`detect_univariate_changepoints`.
SEARCH_METHODS: Final[frozenset[str]] = frozenset({"pelt", "binseg", "bottomup"})

#: Candidate ``penalty_adjust`` multipliers swept by the ``"auto"`` plateau search.
#: A geometric grid (step ``2**(1/3)``) so that relative penalty changes are uniform;
#: the power-of-two anchors make 0.5, 1.0, 2.0 (the default), 4.0 and 8.0 exact.
PENALTY_ADJUST_GRID: Final[tuple[float, ...]] = tuple(0.5 * 2 ** (k / 3) for k in range(13))

#: Minimum tolerant-Jaccard similarity between adjacent grid points for them to
#: belong to the same plateau.
PLATEAU_JACCARD_THRESHOLD: Final[float] = 0.90

#: ``penalty_adjust`` used when the plateau search finds no stable region.
PENALTY_ADJUST_FALLBACK: Final[float] = 2.0

#: Consistency constant that rescales the Median Absolute Deviation to the
#: standard deviation of a Gaussian: ``sigma = MAD / Phi^{-1}(0.75) = 1.4826 * MAD``.
_MAD_TO_SIGMA: Final[float] = 1.4826


def _estimate_sigma(core: np.ndarray, sigma_estimator: str) -> float:
    """Estimate the noise scale ``sigma`` used to derive the AIC/BIC penalty.

    The penalty in :func:`detect_univariate_changepoints` scales with ``sigma**2``.
    A ``sigma`` that is inflated by the very signal we want to detect makes the
    penalty too large and yields **false negatives** (missed change points). The
    three estimators trade off robustness against different sources of inflation:

    * ``"std"`` -- ``np.nanstd`` of the series. The historical default and the
      most efficient estimator for clean, stationary Gaussian noise. Its weakness
      is that a trend, a level shift, or outliers all enlarge the global standard
      deviation, so on contaminated series it over-penalizes and misses changes.
    * ``"mad"`` -- ``1.4826 * median(|x - median(x)|)``. The Median Absolute
      Deviation is robust to a *minority* of outliers or transient spikes (up to a
      ~50% breakdown point): the median and its absolute deviations are unaffected
      by a few extreme samples, so the penalty reflects the noise floor rather than
      the contamination. Prefer it when the data contains sparse spikes/outliers.
      It does **not** protect against a strong trend (which spreads the whole
      distribution) -- use ``"diff_std"`` for that.
    * ``"diff_std"`` -- ``nanstd(diff(x)) / sqrt(2)``. First differencing removes
      any constant level and any linear trend (their contribution to the diff is a
      constant, which carries no variance), so this estimates the noise scale
      independently of trend and level shifts. For i.i.d. noise with variance
      ``s**2`` we have ``Var(x_t - x_{t-1}) = 2*s**2``, hence the ``1/sqrt(2)``
      rescaling. Prefer it when the series trends or contains large level shifts
      that would otherwise inflate ``"std"``.

    A robust estimate that degenerates to ``0`` (e.g. MAD when more than half of
    the samples share one value, or ``diff_std`` on a staircase-flat series) would
    zero the penalty and let the detector over-segment, so non-finite or zero
    robust estimates fall back to ``np.nanstd``.

    Returns ``float(sigma)``. Raises ``ValueError`` for an unknown estimator name.
    """
    match sigma_estimator:
        case "std":
            return float(np.nanstd(core))
        case "mad":
            median = np.nanmedian(core)
            mad = np.nanmedian(np.abs(core - median))
            sigma = float(_MAD_TO_SIGMA * mad)
        case "diff_std":
            if core.size < 2:
                return float(np.nanstd(core))
            sigma = float(np.nanstd(np.diff(core)) / np.sqrt(2.0))
        case _:
            raise ValueError(
                f"sigma_estimator={sigma_estimator!r} is not supported. " f"Choose one of {sorted(SIGMA_ESTIMATORS)}."
            )
    if not np.isfinite(sigma) or sigma == 0.0:
        return float(np.nanstd(core))
    return sigma


def _detect_changepoints_with_missing_values(x: np.ndarray) -> npt.ArrayLike:
    """
    Detect changepoints with missing values
    Sample:
        input: [1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9, np.nan, np.nan])
        output: [2 6 9]
    """
    is_nan = np.isnan(x)
    if is_nan.size == 0:
        return np.array([], dtype=int)
    # Get the index where NaN value changes
    nans = np.where(is_nan, 1, 0)
    change_indexes = np.where(np.diff(nans) == 1)[0] + 1
    # If array starts with NaN, add 0 to indices
    if is_nan[0]:
        change_indexes = np.concatenate((np.array([0]), change_indexes))
    return change_indexes


def _prepare_core(x: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Trim leading/trailing NaN and linearly interpolate interior NaN.

    Returns ``(core, left)`` where ``left`` is the offset that maps
    core-relative indices back to positions in the original ``x``, or
    ``(None, 0)`` when the series is entirely NaN.
    """
    is_nan = np.isnan(x)
    if is_nan.all():
        return None, 0

    valid_positions = np.where(~is_nan)[0]
    left, right = int(valid_positions[0]), int(valid_positions[-1])
    core = np.asarray(x[left : right + 1], dtype=float).copy()

    core_is_nan = np.isnan(core)
    if core_is_nan.any():
        idx = np.arange(core.size)
        core[core_is_nan] = np.interp(idx[core_is_nan], idx[~core_is_nan], core[~core_is_nan])
    return core, left


def _validate_search_method(search_method: str) -> None:
    if search_method not in SEARCH_METHODS:
        raise ValueError(f"search_method={search_method} is not supported.")


def _build_searcher(search_method: str, cost_model: str):
    _validate_search_method(search_method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        match search_method:
            case "pelt":
                return rpt.KernelCPD(kernel="linear", min_size=2, jump=1)  # written in C lang
            case "binseg":
                return rpt.Binseg(model=cost_model, jump=1)
            case "bottomup":
                return rpt.BottomUp(model=cost_model, jump=1)
            case _:
                raise ValueError(f"search_method={search_method} is not supported.")


def _base_penalty(core: np.ndarray, penalty: str | float, sigma_estimator: str) -> float:
    """Derive the un-adjusted penalty (before the ``penalty_adjust`` multiplier)."""
    sigma = _estimate_sigma(core, sigma_estimator)
    match penalty:
        case "aic":
            return sigma * sigma
        case "bic":
            return float(np.log(core.size)) * sigma * sigma
        case _:
            return float(penalty)


def detect_univariate_changepoints(
    x: np.ndarray,
    search_method: str,
    cost_model: str,
    penalty: str | float,
    penalty_adjust: float,
    sigma_estimator: str = "std",
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
    * The **penalty** is derived from the noise scale ``sigma`` of the
      interpolated core (see :func:`_estimate_sigma`) so that residual NaN can
      never turn ``sigma`` (and therefore the penalty) into ``NaN`` -- the
      previous ``np.std(x)`` produced ``NaN`` on any input containing missing
      values and broke the ``predict(pen=...)`` call. ``sigma_estimator``
      selects how that scale is measured (``"std"`` / ``"mad"`` / ``"diff_std"``,
      default ``"std"`` for backward compatibility); see :func:`_estimate_sigma`
      for when to prefer each one.
    * Independently, the boundaries where a metric **goes missing** are treated
      as change points in their own right (see
      :func:`_detect_changepoints_with_missing_values`) and unioned with the
      detector output, because a metric that stops reporting is itself a signal.
    * Empty and constant observed series return no detector change points. This
      also avoids passing a zero penalty to search methods that require a
      strictly positive value.

    Nonconstant inputs without NaN retain the original detection path (no
    trimming, ``core == x``, ``nanstd == std``).
    """
    _validate_search_method(search_method)
    missing_value_cps = {int(i) for i in _detect_changepoints_with_missing_values(x)}

    core, left = _prepare_core(x)
    if core is None or core.size < 2 or (core == core[0]).all():
        # All-NaN input, or too short after trimming (KernelCPD needs min_size=2
        # samples), or a constant observed signal; only missing-value boundaries remain.
        return sorted(missing_value_cps)

    searcher = _build_searcher(search_method, cost_model)
    pen = _base_penalty(core, penalty, sigma_estimator)
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


def _aggregate_multi_changepoints(
    metrics: list[str], multi_change_points: list[list[int]]
) -> tuple[list[int], dict[int, list[str]], dict[str, list[int]]]:
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


def detect_multi_changepoints(
    X: pd.DataFrame,
    search_method: str,
    cost_model: str,
    penalty: str | float,
    penalty_adjust: float,
    n_jobs: int = -1,
    sigma_estimator: str = "std",
) -> tuple[list[int], dict[int, list[str]], dict[str, list[int]]]:
    metrics: list[str] = X.columns.tolist()
    multi_change_points = Parallel(n_jobs=n_jobs)(
        delayed(detect_univariate_changepoints)(
            X[metric].to_numpy(), search_method, cost_model, penalty, penalty_adjust, sigma_estimator
        )
        for metric in metrics
    )
    return _aggregate_multi_changepoints(metrics, multi_change_points)


def _univariate_penalty_path(
    x: np.ndarray,
    search_method: str,
    cost_model: str,
    penalty: str | float,
    penalty_adjust_grid: tuple[float, ...],
    sigma_estimator: str,
) -> tuple[list[list[int]], list[int]]:
    """Detect change points for every ``penalty_adjust`` candidate at once.

    The searcher is fitted once and ``predict(pen=...)`` is re-run per grid
    point, so sweeping the grid costs one ``fit`` plus ``len(grid)`` dynamic
    programs instead of ``len(grid)`` full detections.

    Returns ``(path, missing_value_cps)`` where ``path[g]`` holds the detected
    change points (remapped to original positions, **excluding** missing-value
    boundaries) for grid point ``g``. Missing-value boundaries are returned
    separately because they are penalty-invariant: including them in the
    plateau comparison would inflate every adjacent similarity toward 1.
    """
    _validate_search_method(search_method)
    missing_value_cps = sorted({int(i) for i in _detect_changepoints_with_missing_values(x)})

    core, left = _prepare_core(x)
    if core is None or core.size < 2 or (core == core[0]).all():
        return [[] for _ in penalty_adjust_grid], missing_value_cps

    searcher = _build_searcher(search_method, cost_model)
    base_pen = _base_penalty(core, penalty, sigma_estimator)
    fitted = searcher.fit(core)

    path: list[list[int]] = []
    for adjust in penalty_adjust_grid:
        try:
            cps = fitted.predict(pen=base_pen * adjust)
        except BadSegmentationParameters:
            path.append([])
            continue
        if cps is None:
            raise ValueError("Change point detection failed: predict() returned None.")
        path.append(sorted(int(cp) + left for cp in cps[:-1]))
    return path, missing_value_cps


def _tolerant_matched_count(a: list[int], b: list[int], tolerance: int) -> int:
    """Count greedily matched pairs between two sorted lists within ``tolerance``."""
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return matched


def select_penalty_adjust(
    paths: list[list[list[int]]],
    series_length: int,
    penalty_adjust_grid: tuple[float, ...] = PENALTY_ADJUST_GRID,
    plateau_threshold: float = PLATEAU_JACCARD_THRESHOLD,
) -> tuple[float, dict]:
    """Pick ``penalty_adjust`` by penalty-plateau detection (stability selection).

    The penalty multiplier acts on every metric independently, so the natural
    perturbation axis for a stability argument is the multiplier itself: a value
    whose change-point set barely moves across a wide range of neighboring
    multipliers sits far from both the over-segmentation regime (small values,
    results churn) and the missed-detection regime (large values, results decay
    to nothing).

    For each pair of adjacent grid points the change-point sets of all metrics
    are compared with a position-tolerant Jaccard similarity (tolerance
    ``max(1, round(0.01 * series_length))``, absorbing 1-2 sample jitter in
    PELT's optimum as the penalty moves). The widest run of pairs whose
    similarity is at least ``plateau_threshold`` -- with a non-empty result on
    both sides, so that the trivially stable "nothing detected" tail never
    counts -- is the plateau; its midpoint grid value is returned. Ties prefer
    the plateau whose midpoint is closest to the historical default 2.0. When
    no plateau exists, the default ``PENALTY_ADJUST_FALLBACK`` is returned.

    Args:
        paths: ``paths[m][g]`` = sorted change points of metric ``m`` at grid
            point ``g`` (missing-value boundaries excluded).
        series_length: Length of the time axis (defines the match tolerance).
        penalty_adjust_grid: Ascending candidate multipliers.
        plateau_threshold: Minimum adjacent similarity within a plateau.

    Returns:
        ``(resolved, diagnostics)`` where diagnostics carries ``grid``,
        ``n_change_points``, ``adjacent_jaccard``, ``plateau`` and ``reason``.
    """
    grid = [float(a) for a in penalty_adjust_grid]
    n_grid = len(grid)
    tolerance = max(1, round(0.01 * series_length))

    counts = [sum(len(path[g]) for path in paths) for g in range(n_grid)]
    jaccards: list[float] = []
    for g in range(n_grid - 1):
        intersection = sum(_tolerant_matched_count(path[g], path[g + 1], tolerance) for path in paths)
        union = counts[g] + counts[g + 1] - intersection
        jaccards.append(intersection / union if union > 0 else 1.0)

    diagnostics: dict = {"grid": grid, "n_change_points": counts, "adjacent_jaccard": jaccards}

    best: tuple[int, float, int, int] | None = None  # (width, -|mid - 2.0|, start, end)
    g = 0
    while g < n_grid - 1:
        if jaccards[g] >= plateau_threshold and counts[g] > 0 and counts[g + 1] > 0:
            start = g
            while g < n_grid - 1 and jaccards[g] >= plateau_threshold and counts[g] > 0 and counts[g + 1] > 0:
                g += 1
            end = g  # plateau spans grid[start..end] inclusive
            midpoint = grid[(start + end) // 2]
            candidate = (end - start, -abs(midpoint - PENALTY_ADJUST_FALLBACK), start, end)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        else:
            g += 1

    if best is None:
        diagnostics["plateau"] = None
        diagnostics["reason"] = "no_plateau"
        return PENALTY_ADJUST_FALLBACK, diagnostics

    _, _, start, end = best
    diagnostics["plateau"] = (grid[start], grid[end])
    diagnostics["reason"] = "plateau"
    return grid[(start + end) // 2], diagnostics


def detect_multi_changepoints_with_penalty_tuning(
    X: pd.DataFrame,
    search_method: str,
    cost_model: str,
    penalty: str | float,
    n_jobs: int = -1,
    sigma_estimator: str = "std",
    penalty_adjust_grid: tuple[float, ...] = PENALTY_ADJUST_GRID,
) -> tuple[list[int], dict[int, list[str]], dict[str, list[int]], float, dict]:
    """Like :func:`detect_multi_changepoints`, but with ``penalty_adjust`` tuned.

    Computes the penalty path of every metric in parallel, selects the plateau
    multiplier via :func:`select_penalty_adjust`, and assembles the final
    change points from the already-computed path at the chosen grid point (no
    re-detection), unioned with the penalty-invariant missing-value boundaries.

    Returns ``(flatten_change_points, cp_to_metrics, metric_to_cps,
    resolved_penalty_adjust, diagnostics)``.
    """
    metrics: list[str] = X.columns.tolist()
    grid = tuple(float(a) for a in penalty_adjust_grid)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_univariate_penalty_path)(
            X[metric].to_numpy(), search_method, cost_model, penalty, grid, sigma_estimator
        )
        for metric in metrics
    )
    paths = [path for path, _ in results]
    missing_value_cps = [mv_cps for _, mv_cps in results]

    resolved, diagnostics = select_penalty_adjust(paths, series_length=X.shape[0], penalty_adjust_grid=grid)
    if not metrics:
        diagnostics["reason"] = "no_metrics"

    if resolved in grid:
        g_star = grid.index(resolved)
        multi_change_points = [
            sorted(set(path[g_star]) | set(mv_cps)) for path, mv_cps in zip(paths, missing_value_cps)
        ]
        flatten, cp_to_metrics, metric_to_cps = _aggregate_multi_changepoints(metrics, multi_change_points)
    else:
        # A custom grid may not contain the fallback multiplier; detect once at it.
        flatten, cp_to_metrics, metric_to_cps = detect_multi_changepoints(
            X, search_method, cost_model, penalty, resolved, n_jobs=n_jobs, sigma_estimator=sigma_estimator
        )
    return flatten, cp_to_metrics, metric_to_cps, resolved, diagnostics
